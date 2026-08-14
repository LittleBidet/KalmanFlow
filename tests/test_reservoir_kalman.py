from datetime import UTC, datetime, timedelta

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    Observation,
    OnlineFixedLagRTS,
    OnlineReservoirInflow,
    ReservoirConfig,
    ReservoirStateSpaceModel,
    UnitSystem,
    get_reservoir_inflow,
    initial_filter_step,
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
        "tuning_metadata": {"tunable": ["Q", "R"]},
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
    npt.assert_allclose(
        model.inflow_to_flow_rate(1.0),
        1.0,
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


def test_reservoir_config_freezes_arrays_metadata_and_tunes_only_q_r() -> None:
    config = _config()
    tuned = config.with_tuned_noise(q=Q * 2.0, r=R * 3.0)

    npt.assert_allclose(tuned.q, Q * 2.0)
    npt.assert_allclose(tuned.r, R * 3.0)
    npt.assert_allclose(tuned.p0, config.p0)
    assert tuned.smoothing_lag == config.smoothing_lag
    assert tuned.tuning_metadata["tunable"] == ("Q", "R")

    with pytest.raises(ValueError):
        config.q[0, 0] = 0.0
    with pytest.raises(TypeError):
        config.tuning_metadata["new"] = "value"


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

    assert result.loc[index[0], "estimated_inflow"] == pytest.approx(expected_rate)


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
