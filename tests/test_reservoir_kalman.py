from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from kalmanflow import (
    InflowUnits,
    InitializationStrategy,
    Observation,
    OnlineFixedLagRTS,
    OnlineReservoirInflow,
    ReservoirConfig,
    ReservoirStateSpaceModel,
    UnitSystem,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
    initial_filter_step,
    kalman_filter,
    kalman_step,
    smooth_filter_steps,
)

Q = np.diag([2.0, 0.5, 0.25])
R = np.diag([0.25, 0.5])
P0 = np.diag([4.0, 9.0, 16.0])


def _config(**overrides: object) -> ReservoirConfig:
    values: dict[str, object] = {
        "reservoir_id": "demo",
        "reservoir_name": "Demo Reservoir",
        "q": Q,
        "r": R,
        "p0": P0,
        "smoothing_lag": timedelta(minutes=30),
        "initialization_strategy": InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        "inflow_units": InflowUnits.CUBIC_FEET_PER_SECOND,
        "model_version": "model-v1",
        "configuration_version": "config-v1",
        "metadata": {"source": "test"},
    }
    values.update(overrides)
    return ReservoirConfig(**values)


def _forward_steps(count: int = 4):
    start = datetime(2024, 1, 1, tzinfo=UTC)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([0.2, 0.05])
    observation_matrix = np.array([[1.0, 0.0]])
    observation_covariance = np.array([[0.4]])
    steps = [
        initial_filter_step(
            timestamp=start,
            initial_mean=np.array([10.0, 0.5]),
            initial_covariance=np.diag([2.0, 3.0]),
            observation=np.array([10.0]),
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
        )
    ]
    for index in range(1, count):
        previous = steps[-1]
        steps.append(
            kalman_step(
                timestamp=start + timedelta(hours=index),
                previous_filtered_mean=previous.filtered_mean,
                previous_filtered_covariance=previous.filtered_covariance,
                transition_matrix=transition,
                process_covariance=process,
                observation=np.array([10.0 + 0.7 * index]),
                observation_matrix=observation_matrix,
                observation_covariance=observation_covariance,
                control_offset=np.array([-0.1, 0.0]),
            )
        )
    return steps


def test_reservoir_model_scales_time_process_noise_and_flow_units() -> None:
    model = ReservoirStateSpaceModel(Q)

    npt.assert_allclose(
        model.transition_matrix(900.0),
        [
            [1.0, 900.0 / 43560.0, -900.0 / 43560.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    alpha = 1.0 / 43560.0
    expected_process = np.array(
        [
            [
                2.0 * 1800.0 + alpha**2 * (0.5 + 0.25) * 1800.0**3 / 3.0,
                alpha * 0.5 * 1800.0**2 / 2.0,
                -alpha * 0.25 * 1800.0**2 / 2.0,
            ],
            [
                alpha * 0.5 * 1800.0**2 / 2.0,
                0.5 * 1800.0,
                0.0,
            ],
            [
                -alpha * 0.25 * 1800.0**2 / 2.0,
                0.0,
                0.25 * 1800.0,
            ],
        ]
    )
    npt.assert_allclose(model.process_covariance(1800.0), expected_process)
    npt.assert_allclose(model.discharge_volume(4.0, 900.0), 4.0 * 900.0 / 43560.0)
    npt.assert_allclose(
        model.initial_inflow(100.0, 101.0, 4.0, 900.0),
        1.0 / (900.0 / 43560.0) + 4.0,
    )
    assert model.initial_outflow(4.0) == pytest.approx(4.0)


@pytest.mark.parametrize("elapsed_seconds", [300.0, 600.0, 900.0, 1800.0, 3600.0])
def test_initial_inflow_rate_is_cadence_invariant(elapsed_seconds: float) -> None:
    model = ReservoirStateSpaceModel(np.zeros((3, 3)))
    expected_rate = 100.0
    outflow_rate = 4.0
    storage_change = model.unit_system.flow_to_volume(
        expected_rate - outflow_rate, elapsed_seconds
    )

    initial_rate = model.initial_inflow(
        100.0,
        100.0 + storage_change,
        outflow_rate,
        elapsed_seconds,
    )

    assert initial_rate == pytest.approx(expected_rate)


def test_continuous_process_covariance_composes_across_split_intervals() -> None:
    model = ReservoirStateSpaceModel(Q)
    first_interval = 600.0
    second_interval = 900.0
    second_transition = model.transition_matrix(second_interval)
    composed_covariance = second_transition @ model.process_covariance(
        first_interval
    ) @ second_transition.T + model.process_covariance(second_interval)

    npt.assert_allclose(
        composed_covariance,
        model.process_covariance(first_interval + second_interval),
    )


def test_unit_system_supports_si_and_rejects_invalid_conversion_factors() -> None:
    si = UnitSystem.si()
    assert si.volume_label == "m^3"
    assert si.flow_label == "m^3/s"
    assert si.flow_to_volume(2.0, 30.0) == 60.0
    assert si.volume_to_flow_rate(60.0, 30.0) == 2.0

    with pytest.raises(ValueError, match="positive and finite"):
        UnitSystem(flow_to_volume_per_second=0.0)


def test_reservoir_config_freezes_arrays_and_metadata() -> None:
    config = _config()

    npt.assert_allclose(config.q, Q)
    npt.assert_allclose(config.r, R)
    assert config.metadata["source"] == "test"

    with pytest.raises(ValueError):
        config.q[0, 0] = 0.0
    with pytest.raises(TypeError):
        config.metadata["new"] = "value"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"reservoir_id": ""}, "reservoir_id must not be empty"),
        ({"q": np.eye(2)}, "q must have shape"),
        ({"r": np.diag([0.0, 1.0])}, "r diagonal entries"),
        ({"p0": np.diag([1.0, -1.0, 1.0])}, "positive semidefinite"),
        ({"smoothing_lag": timedelta(0)}, "smoothing_lag must be positive"),
    ],
)
def test_reservoir_config_validates_domain_constraints(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_observation_validates_timezone() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    observation = Observation(
        timestamp=timestamp,
        storage=float("nan"),
        discharge=float("nan"),
    )
    assert np.isnan(observation.storage)

    with pytest.raises(ValueError, match="timezone-aware"):
        Observation(
            timestamp=datetime(2024, 1, 1),
            storage=100.0,
            discharge=4.0,
        )


def test_full_rts_smoothing_returns_immutable_smoothed_states() -> None:
    steps = _forward_steps()
    smoothed = smooth_filter_steps(steps)

    assert len(smoothed) == len(steps)
    assert [state.timestamp for state in smoothed] == [step.timestamp for step in steps]
    npt.assert_allclose(smoothed[-1].mean, steps[-1].filtered_mean)
    with pytest.raises(ValueError):
        smoothed[0].mean[0] = 0.0


def test_fixed_lag_rts_finalizes_only_elapsed_states() -> None:
    steps = _forward_steps()
    smoother = OnlineFixedLagRTS(timedelta(hours=2), max_window_steps=4)

    assert smoother.add_step(steps[0]) == ()
    assert smoother.add_step(steps[1]) == ()
    expected = smooth_filter_steps(steps[:3])[0]
    (actual,) = smoother.add_step(steps[2])

    assert actual.timestamp == expected.timestamp
    npt.assert_allclose(actual.mean, expected.mean)
    assert [step.timestamp for step in smoother.provisional()] == [
        steps[1].timestamp,
        steps[2].timestamp,
    ]


def test_get_reservoir_inflow_default_pipeline_returns_batch_frame() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    storage = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    outflow = pd.Series([4.0, 4.0, 4.0, 4.0], index=index)

    result = get_reservoir_inflow(
        storage,
        outflow,
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.25,
        smoothing_lag=timedelta(minutes=15),
    )

    assert result.index.equals(index)
    assert list(result.columns) == [
        "estimated_inflow",
        "revised_inflow",
        "estimated_inflow_flag",
        "revised_inflow_flag",
        "estimated_inflow_smoothing_flag",
        "revised_inflow_smoothing_flag",
    ]
    assert result["estimated_inflow"].notna().all()
    assert result["revised_inflow"].iloc[:-1].notna().all()
    assert result["revised_inflow"].iloc[-1:].isna().all()


def test_get_reservoir_inflow_reports_cfs_for_ten_minute_samples() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="10min", tz="UTC")
    units = UnitSystem.us_customary()
    expected_rate = 100.0
    outflow_rate = 4.0
    storage = pd.Series(
        [
            100.0,
            100.0 + units.flow_to_volume(expected_rate - outflow_rate, 10.0 * 60.0),
        ],
        index=index,
    )
    outflow = pd.Series([outflow_rate, outflow_rate], index=index)

    result = get_reservoir_inflow(
        storage,
        outflow,
        q_storage=1e-12,
        q_inflow=1e-12,
        q_outflow=1e-12,
        r_storage=1e-12,
        r_outflow=1e-12,
        smoothing_lag=timedelta(hours=1),
    )

    assert result.loc[index[0], "estimated_inflow"] == pytest.approx(outflow_rate)
    assert result.loc[index[1], "estimated_inflow"] == pytest.approx(expected_rate)


@pytest.mark.parametrize("column", ["storage", "outflow"])
def test_batch_rejects_infinity_instead_of_treating_it_as_missing(
    column: str,
) -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC")
    storage = pd.Series([100.0, 101.0, 102.0], index=index)
    outflow = pd.Series([4.0, 4.0, 4.0], index=index)
    (storage if column == "storage" else outflow).iloc[1] = np.inf

    input_name = "storage" if column == "storage" else "discharge"
    with pytest.raises(ValueError, match=f"{input_name}.*finite values or NaN"):
        get_reservoir_inflow(
            storage,
            outflow,
            q_storage=0.1,
            q_inflow=0.1,
            q_outflow=0.1,
            r_storage=0.25,
            r_outflow=0.5,
        )


def test_stream_rejects_infinity_before_mutating_state() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    stream = OnlineReservoirInflow(
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
    )

    with pytest.raises(ValueError, match="storage must be finite or NaN"):
        stream.process(timestamp=start, storage=np.inf, discharge=4.0)
    assert stream.initialized is False
    assert stream.process(
        timestamp=start, storage=100.0, discharge=4.0
    ).filtered_inflows == ()


def test_noisy_outflow_measurements_inform_smoothed_inflow() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    units = UnitSystem.us_customary()
    storage_values = [100.0]
    for _ in range(len(index) - 1):
        storage_values.append(
            storage_values[-1] + units.flow_to_volume(6.0, 15.0 * 60.0)
        )
    measured_outflow = np.array([4.0, 20.0, 4.0, 20.0, 4.0, 20.0, 4.0, 20.0])

    result = get_reservoir_inflow(
        pd.Series(storage_values, index=index),
        pd.Series(measured_outflow, index=index),
        q_storage=1e-8,
        q_inflow=1e-8,
        q_outflow=1e-10,
        r_storage=1e-6,
        r_outflow=100.0,
        smoothing_lag=timedelta(minutes=15),
    )

    assert np.isfinite(result["estimated_inflow"]).all()
    assert np.isfinite(result["revised_inflow"].iloc[:-1]).all()


def test_batch_kernel_matches_streaming_for_irregular_missing_data() -> None:
    index = pd.DatetimeIndex(
        [
            datetime(2024, 1, 1, tzinfo=UTC) + timedelta(minutes=minute)
            for minute in (0, 5, 20, 50, 70, 105)
        ]
    )
    storage = pd.Series([np.nan, 100.0, 101.0, np.nan, 103.0, 104.0], index=index)
    outflow = pd.Series([4.0, 4.0, 4.5, np.nan, 4.0, 4.0], index=index)
    kwargs = {
        "q_storage": 0.1,
        "q_inflow": 0.1,
        "q_outflow": 0.1,
        "r_storage": 0.25,
        "r_outflow": 0.5,
        "smoothing_lag": timedelta(minutes=30),
    }

    actual = get_reservoir_inflow(storage, outflow, **kwargs)
    stream = OnlineReservoirInflow(**kwargs)
    expected = {
        "estimated_inflow": np.full(len(index), np.nan),
        "revised_inflow": np.full(len(index), np.nan),
        "estimated_inflow_flag": np.full(len(index), None, dtype=object),
        "revised_inflow_flag": np.full(len(index), None, dtype=object),
        "estimated_inflow_smoothing_flag": np.full(
            len(index), "NON_SMOOTHED", dtype=object
        ),
        "revised_inflow_smoothing_flag": np.full(
            len(index), "NON_SMOOTHED", dtype=object
        ),
    }
    for timestamp, storage_value, outflow_value in zip(
        index, storage, outflow, strict=True
    ):
        update = stream.process(
            Observation(timestamp, storage_value, outflow_value)
        )
        for estimate in update.filtered_inflows:
            output_position = index.get_loc(estimate.timestamp)
            expected["estimated_inflow"][output_position] = estimate.value
            expected["estimated_inflow_flag"][output_position] = (
                estimate.prediction_flag.value
            )
        for estimate in update.revised_inflows:
            output_position = index.get_loc(estimate.timestamp)
            expected["revised_inflow"][output_position] = estimate.value
            expected["revised_inflow_flag"][output_position] = (
                estimate.prediction_flag.value
            )
            expected["revised_inflow_smoothing_flag"][output_position] = (
                estimate.smoothing_flag.value
            )

    for column, values in expected.items():
        if values.dtype == object:
            actual_values = actual[column].to_numpy(dtype=object)
            assert np.array_equal(pd.isna(actual_values), pd.isna(values))
            present = ~pd.isna(values)
            assert actual_values[present].tolist() == values[present].tolist()
        else:
            npt.assert_allclose(actual[column], values, equal_nan=True)


@pytest.mark.parametrize("cadence_minutes", [5, 15, 60])
def test_synthetic_inflow_truth_is_recovered_and_smoothing_reduces_error(
    cadence_minutes: int,
) -> None:
    rng = np.random.default_rng(2026 + cadence_minutes)
    count = 160
    elapsed_seconds = cadence_minutes * 60.0
    index = pd.date_range(
        "2024-01-01", periods=count, freq=f"{cadence_minutes}min", tz="UTC"
    )
    true_inflow = np.empty(count)
    true_outflow = np.empty(count)
    true_storage = np.empty(count)
    true_inflow[0] = 30.0
    true_outflow[0] = 10.0
    true_storage[0] = 1_000.0
    scale = np.sqrt(elapsed_seconds / 900.0)
    true_inflow[1:] = 30.0 + np.cumsum(rng.normal(0.0, scale, count - 1))
    true_outflow[1:] = 10.0 + np.cumsum(
        rng.normal(0.0, 0.25 * scale, count - 1)
    )
    units = UnitSystem.us_customary()
    for position in range(1, count):
        true_storage[position] = true_storage[position - 1] + units.flow_to_volume(
            true_inflow[position - 1] - true_outflow[position - 1],
            elapsed_seconds,
        )

    measured_storage = true_storage + rng.normal(0.0, 0.05, count)
    measured_outflow = true_outflow + rng.normal(0.0, 0.25, count)
    config = _config(
        q=np.diag([1e-10, 1.0 / 900.0, 0.25**2 / 900.0]),
        r=np.diag([0.05**2, 0.25**2]),
        p0=np.diag([0.1, 25.0, 4.0]),
        smoothing_lag=timedelta(hours=3),
    )

    result = get_reservoir_inflow_from_config(
        pd.Series(measured_storage, index=index),
        pd.Series(measured_outflow, index=index),
        config,
    )
    warmup = 10
    filtered = result["estimated_inflow"].to_numpy()
    revised = result["revised_inflow"].to_numpy()
    filtered_mask = np.arange(count) >= warmup
    revised_mask = filtered_mask & np.isfinite(revised)
    filtered_rmse = float(
        np.sqrt(np.mean((filtered[filtered_mask] - true_inflow[filtered_mask]) ** 2))
    )
    revised_rmse = float(
        np.sqrt(np.mean((revised[revised_mask] - true_inflow[revised_mask]) ** 2))
    )

    assert filtered_rmse < 3.0
    assert revised_rmse < filtered_rmse


@pytest.mark.parametrize("seed", range(5))
def test_randomized_batch_and_stream_outputs_are_equivalent(seed: int) -> None:
    rng = np.random.default_rng(seed)
    count = 40
    elapsed = rng.integers(1, 7_201, size=count - 1)
    offsets = np.concatenate(([0], np.cumsum(elapsed)))
    start = datetime(2024, 1, 1, tzinfo=UTC)
    index = pd.DatetimeIndex(
        [start + timedelta(seconds=int(offset)) for offset in offsets]
    )
    storage = 1_000.0 + np.cumsum(rng.normal(0.0, 0.5, count))
    outflow = 10.0 + rng.normal(0.0, 1.0, count)
    storage[2:][rng.random(count - 2) < 0.2] = np.nan
    outflow[2:][rng.random(count - 2) < 0.2] = np.nan
    kwargs = {
        "q_storage": float(10 ** rng.uniform(-4.0, -1.0)),
        "q_inflow": float(10 ** rng.uniform(-4.0, -1.0)),
        "q_outflow": float(10 ** rng.uniform(-4.0, -1.0)),
        "r_storage": float(10 ** rng.uniform(-3.0, 0.0)),
        "r_outflow": float(10 ** rng.uniform(-3.0, 0.0)),
        "smoothing_lag": timedelta(minutes=int(rng.integers(15, 181))),
    }
    batch = get_reservoir_inflow(
        pd.Series(storage, index=index), pd.Series(outflow, index=index), **kwargs
    )

    stream = OnlineReservoirInflow(**kwargs)
    streamed_filtered: dict[datetime, object] = {}
    streamed_revised: dict[datetime, object] = {}
    for timestamp, storage_value, outflow_value in zip(
        index, storage, outflow, strict=True
    ):
        update = stream.process(Observation(timestamp, storage_value, outflow_value))
        for estimate in update.filtered_inflows:
            assert estimate.timestamp not in streamed_filtered
            streamed_filtered[estimate.timestamp] = estimate
        for estimate in update.revised_inflows:
            assert estimate.timestamp not in streamed_revised
            streamed_revised[estimate.timestamp] = estimate

    expected_filtered = pd.Series(
        {timestamp: estimate.value for timestamp, estimate in streamed_filtered.items()}
    ).reindex(index)
    expected_revised = pd.Series(
        {timestamp: estimate.value for timestamp, estimate in streamed_revised.items()}
    ).reindex(index)
    npt.assert_allclose(batch["estimated_inflow"], expected_filtered, equal_nan=True)
    npt.assert_allclose(batch["revised_inflow"], expected_revised, equal_nan=True)
    assert batch.loc[expected_filtered.notna(), "estimated_inflow_flag"].tolist() == [
        estimate.prediction_flag.value for estimate in streamed_filtered.values()
    ]
    assert batch.loc[expected_revised.notna(), "revised_inflow_flag"].tolist() == [
        estimate.prediction_flag.value for estimate in streamed_revised.values()
    ]


@pytest.mark.parametrize("elapsed_seconds", [1e-6, 1.0, 31_536_000.0])
def test_process_covariance_stays_finite_symmetric_and_psd_at_time_extremes(
    elapsed_seconds: float,
) -> None:
    model = ReservoirStateSpaceModel(np.diag([1e-12, 1e-8, 1e-8]))
    covariance = model.process_covariance(elapsed_seconds)
    scale = max(1.0, float(np.max(np.abs(covariance))))

    assert np.isfinite(covariance).all()
    npt.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-12 * scale)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12 * scale


def test_filter_remains_finite_and_psd_across_extreme_time_and_state_scales() -> None:
    model = ReservoirStateSpaceModel(np.diag([1e-12, 1e-8, 1e-8]))
    elapsed = np.array([1e-6, 1.0, 86_400.0, 31_536_000.0])
    result = kalman_filter(
        observations=np.array(
            [
                [1e9, 1e5],
                [1e9 + 1e-6, 1e5],
                [1e9 + 1.0, 1e5 + 1.0],
                [1e9 + 1e4, 1e5 - 100.0],
                [np.nan, np.nan],
            ]
        ),
        initial_mean=np.array([1e9, 1e5, 1e5]),
        initial_covariance=np.diag([1e12, 1e8, 1e8]),
        transition_matrix=np.asarray(
            [model.transition_matrix(value) for value in elapsed]
        ),
        process_covariance=np.asarray(
            [model.process_covariance(value) for value in elapsed]
        ),
        observation_matrix=model.observation_matrix,
        observation_covariance=np.diag([1e-12, 1e-12]),
    )

    assert np.isfinite(result.filtered_means).all()
    assert np.isfinite(result.filtered_covariances).all()
    assert np.isfinite(result.predicted_covariances).all()
    assert np.isfinite(result.log_likelihood)
    for covariance in np.concatenate(
        (result.filtered_covariances, result.predicted_covariances)
    ):
        scale = max(1.0, float(np.max(np.abs(covariance))))
        npt.assert_allclose(covariance, covariance.T, rtol=0.0, atol=1e-12 * scale)
        assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-12 * scale
