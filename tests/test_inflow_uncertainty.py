from datetime import UTC, datetime, timedelta
from statistics import NormalDist

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from kalmanflow import (
    InflowUnits,
    InitializationStrategy,
    Observation,
    OnlineReservoirInflow,
    ReservoirConfig,
    ReservoirFlowEstimate,
    UnitSystem,
    add_inflow_uncertainty_intervals,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
    run_inflow_model,
)
from kalmanflow.core import _inflow_standard_deviation


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="uncertainty-test",
        reservoir_name="Uncertainty Test",
        q=np.array(
            [
                [0.4, 0.05, -0.03],
                [0.05, 0.2, 0.04],
                [-0.03, 0.04, 0.3],
            ]
        ),
        r=np.array([[0.25, 0.05], [0.05, 0.5]]),
        p0=np.array(
            [
                [4.0, 0.3, -0.2],
                [0.3, 9.0, 0.8],
                [-0.2, 0.8, 16.0],
            ]
        ),
        smoothing_lag=timedelta(minutes=20),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.us_customary(),
    )


def _observations() -> tuple[Observation, ...]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    offsets = (0, 7, 25, 60, 90, 135)
    storage = (100.0, 100.7, np.nan, 102.0, 102.8, 103.2)
    discharge = (4.0, 4.2, 4.1, np.nan, 4.4, 4.5)
    return tuple(
        Observation(
            start + timedelta(minutes=offset),
            storage=storage[index],
            discharge=discharge[index],
        )
        for index, offset in enumerate(offsets)
    )


def test_uncertainty_is_opt_in_and_default_outputs_are_unchanged() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    storage = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    discharge = pd.Series([4.0, 4.0, 4.0, 4.0], index=index)
    kwargs = {
        "q_storage": 0.1,
        "q_inflow": 0.1,
        "q_outflow": 0.1,
        "r_storage": 0.25,
        "r_outflow": 0.5,
        "smoothing_lag": timedelta(minutes=15),
    }

    result = get_reservoir_inflow(storage, discharge, **kwargs)
    assert list(result.columns) == [
        "estimated_inflow",
        "revised_inflow",
        "estimated_inflow_flag",
        "revised_inflow_flag",
        "estimated_inflow_smoothing_flag",
        "revised_inflow_smoothing_flag",
    ]
    uncertain_result = get_reservoir_inflow(
        storage,
        discharge,
        include_uncertainty=True,
        **kwargs,
    )
    pd.testing.assert_frame_equal(uncertain_result[result.columns], result)
    assert list(uncertain_result.columns)[-2:] == [
        "estimated_inflow_standard_deviation",
        "revised_inflow_standard_deviation",
    ]

    stream = OnlineReservoirInflow(**kwargs)
    updates = [
        stream.process(Observation(timestamp, s, d))
        for timestamp, s, d in zip(index, storage, discharge, strict=True)
    ]
    estimates = [
        estimate for update in updates for estimate in update.filtered_inflows
    ]
    revisions = [
        estimate for update in updates for estimate in update.revised_inflows
    ]
    assert estimates and revisions
    assert all(estimate.standard_deviation is None for estimate in estimates)
    assert all(estimate.standard_deviation is None for estimate in revisions)


def test_opt_in_uncertainty_uses_actual_covariance_and_matches_streaming() -> None:
    config = _config()
    observations = _observations()
    index = pd.DatetimeIndex([item.timestamp for item in observations])
    storage = pd.Series([item.storage for item in observations], index=index)
    discharge = pd.Series([item.discharge for item in observations], index=index)

    batch = get_reservoir_inflow_from_config(
        storage,
        discharge,
        config,
        include_uncertainty=True,
    )
    assert list(batch.columns)[-2:] == [
        "estimated_inflow_standard_deviation",
        "revised_inflow_standard_deviation",
    ]
    assert batch["estimated_inflow_standard_deviation"].iloc[2:].notna().all()
    assert batch["revised_inflow_standard_deviation"].iloc[-1:].isna().all()
    assert (batch["estimated_inflow_standard_deviation"].dropna() >= 0.0).all()

    stream = OnlineReservoirInflow.from_config(
        config,
        include_uncertainty=True,
    )
    streamed_filtered = {}
    streamed_revised = {}
    for observation in observations:
        update = stream.process(observation)
        streamed_filtered.update(
            {estimate.timestamp: estimate for estimate in update.filtered_inflows}
        )
        streamed_revised.update(
            {estimate.timestamp: estimate for estimate in update.revised_inflows}
        )

    for timestamp, estimate in streamed_filtered.items():
        assert estimate.standard_deviation is not None
        assert estimate.standard_deviation == pytest.approx(
            batch.loc[timestamp, "estimated_inflow_standard_deviation"]
        )
    for timestamp, estimate in streamed_revised.items():
        assert estimate.standard_deviation is not None
        assert estimate.standard_deviation == pytest.approx(
            batch.loc[timestamp, "revised_inflow_standard_deviation"]
        )
    latest_step = stream._pipeline.latest_filter_step
    assert latest_step is not None
    assert streamed_filtered[latest_step.timestamp].standard_deviation == pytest.approx(
        np.sqrt(latest_step.filtered_covariance[1, 1])
    )


def test_uncertainty_intervals_append_columns_without_mutating_input() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    storage = pd.Series([100.0, 101.0, 102.0, 103.0], index=index)
    discharge = pd.Series([4.0, 4.0, 4.0, 4.0], index=index)
    result = get_reservoir_inflow(
        storage,
        discharge,
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
        smoothing_lag=timedelta(minutes=15),
        include_uncertainty=True,
    )

    output = add_inflow_uncertainty_intervals(result, level=0.90)
    assert list(result.columns)[-2:] == [
        "estimated_inflow_standard_deviation",
        "revised_inflow_standard_deviation",
    ]
    assert list(output.columns)[-4:] == [
        "estimated_inflow_lower",
        "estimated_inflow_upper",
        "revised_inflow_lower",
        "revised_inflow_upper",
    ]
    multiplier = -NormalDist().inv_cdf(0.05)
    npt.assert_allclose(
        output["estimated_inflow_lower"],
        result["estimated_inflow"]
        - multiplier * result["estimated_inflow_standard_deviation"],
        equal_nan=True,
    )
    npt.assert_allclose(
        output["revised_inflow_upper"],
        result["revised_inflow"]
        + multiplier * result["revised_inflow_standard_deviation"],
        equal_nan=True,
    )

    with pytest.raises(ValueError, match="uncertainty columns"):
        add_inflow_uncertainty_intervals(
            result.drop(columns=["revised_inflow_standard_deviation"])
        )


@pytest.mark.parametrize("level", [0.0, 1.0, np.nan, np.inf, -np.inf, "bad"])
def test_uncertainty_interval_rejects_invalid_levels(level: object) -> None:
    estimate = ReservoirFlowEstimate(
        datetime(2024, 1, 1, tzinfo=UTC),
        value=10.0,
        standard_deviation=2.0,
    )
    with pytest.raises(ValueError, match="level must be finite"):
        estimate.uncertainty_interval(level)  # type: ignore[arg-type]


def test_uncertainty_interval_requires_opt_in_and_validates_standard_deviation(
) -> None:
    estimate = ReservoirFlowEstimate(datetime(2024, 1, 1, tzinfo=UTC), value=10.0)
    with pytest.raises(ValueError, match="uncertainty is unavailable"):
        estimate.uncertainty_interval()

    with pytest.raises(ValueError, match="standard_deviation must be non-negative"):
        ReservoirFlowEstimate(
            datetime(2024, 1, 1, tzinfo=UTC),
            value=10.0,
            standard_deviation=-1.0,
        )
    with pytest.raises(ValueError, match="standard_deviation must be finite"):
        ReservoirFlowEstimate(
            datetime(2024, 1, 1, tzinfo=UTC),
            value=10.0,
            standard_deviation=np.inf,
        )


def test_opt_in_uncertainty_preserves_startup_and_predict_only_missing_rows() -> None:
    index = pd.date_range("2024-01-01", periods=6, freq="15min", tz="UTC")
    storage = pd.Series([np.nan, 100.0, 101.0, np.nan, 102.0, 103.0], index=index)
    discharge = pd.Series([4.0, 4.0, 4.1, np.nan, 4.2, 4.3], index=index)
    result = get_reservoir_inflow(
        storage,
        discharge,
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
        smoothing_lag=timedelta(minutes=30),
        include_uncertainty=True,
    )

    assert pd.isna(result.loc[index[0], "estimated_inflow"])
    assert pd.isna(result.loc[index[0], "estimated_inflow_standard_deviation"])
    assert pd.isna(result.loc[index[0], "revised_inflow_standard_deviation"])
    assert result.loc[index[3], "estimated_inflow_flag"] == "PREDICTED"
    assert result.loc[index[3], "revised_inflow_flag"] == "PREDICTED"
    assert np.isfinite(
        result.loc[index[3], "estimated_inflow_standard_deviation"]
    )
    assert np.isfinite(
        result.loc[index[3], "revised_inflow_standard_deviation"]
    )
    assert pd.isna(result.loc[index[-1], "revised_inflow"])
    assert pd.isna(
        result.loc[index[-1], "revised_inflow_standard_deviation"]
    )


def test_inflow_covariance_roundoff_is_clamped_but_material_negative_is_rejected():
    covariance = np.array([[1.0e12, 0.0], [0.0, -1.0e-13]])
    assert _inflow_standard_deviation(covariance, name="test covariance") == 0.0

    with pytest.raises(ValueError, match="materially negative"):
        _inflow_standard_deviation(
            np.array([[1.0e12, 0.0], [0.0, -1.0e-4]]),
            name="test covariance",
        )
    with pytest.raises(ValueError, match="finite"):
        _inflow_standard_deviation(
            np.array([[1.0, 0.0], [0.0, np.nan]]),
            name="test covariance",
        )


def test_checkpoint_can_restore_with_a_different_uncertainty_output_setting() -> None:
    config = _config()
    observations = _observations()
    continuous = OnlineReservoirInflow.from_config(
        config,
        include_uncertainty=True,
    )
    expected = [continuous.process(observation) for observation in observations]
    enabled = OnlineReservoirInflow.from_config(config, include_uncertainty=True)
    disabled = OnlineReservoirInflow.from_config(config, include_uncertainty=False)
    for observation in observations[:4]:
        enabled.process(observation)
        disabled.process(observation)
    assert enabled.checkpoint() == disabled.checkpoint()

    restored = OnlineReservoirInflow.from_checkpoint(
        disabled.checkpoint(),
        config=config,
        include_uncertainty=True,
    )
    actual = [restored.process(observation) for observation in observations[4:]]
    for actual_update, expected_update in zip(actual, expected[4:], strict=True):
        for actual_values, expected_values in (
            (actual_update.filtered_inflows, expected_update.filtered_inflows),
            (actual_update.revised_inflows, expected_update.revised_inflows),
        ):
            assert len(actual_values) == len(expected_values)
            for actual_estimate, expected_estimate in zip(
                actual_values,
                expected_values,
                strict=True,
            ):
                assert actual_estimate.timestamp == expected_estimate.timestamp
                assert actual_estimate.value == pytest.approx(expected_estimate.value)
                assert (
                    actual_estimate.standard_deviation
                    == pytest.approx(expected_estimate.standard_deviation)
                )

    restored_without_uncertainty = OnlineReservoirInflow.from_checkpoint(
        disabled.checkpoint(),
        config=config,
        include_uncertainty=False,
    )
    for update, expected_update in zip(
        (
            restored_without_uncertainty.process(observation)
            for observation in observations[4:]
        ),
        expected[4:],
        strict=True,
    ):
        for actual_values, expected_values in (
            (update.filtered_inflows, expected_update.filtered_inflows),
            (update.revised_inflows, expected_update.revised_inflows),
        ):
            assert [estimate.value for estimate in actual_values] == pytest.approx(
                [estimate.value for estimate in expected_values]
            )
            assert all(
                estimate.standard_deviation is None for estimate in actual_values
            )


def test_pandas_wrapper_passes_through_uncertainty_option() -> None:
    index = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    result = run_inflow_model(
        pd.DataFrame(
            {"storage": [100.0, 101.0, 102.0], "outflow": [4.0, 4.0, 4.0]},
            index=index,
        ),
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
        smoothing_lag=timedelta(minutes=15),
        include_uncertainty=True,
    )
    assert "estimated_inflow_standard_deviation" in result
    assert "revised_inflow_standard_deviation" in result
