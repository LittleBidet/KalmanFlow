"""Focused coverage for robust dataframe-first forecast tuning."""

from __future__ import annotations

from math import lgamma

import numpy as np
import pandas as pd
import pytest

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    NoiseTuningData,
    ReservoirConfig,
    run_filter_with_noise,
    tune_inflow_model,
    tune_noise,
)
from kalmone.tuning import (
    _filter_candidate,
    _forecast_target_index,
    _prepare_arrays,
    _propagate_state,
    _score_multihorizon,
    _validation_origin_blocks,
    student_t_predictive_nll,
)
from kalmone.units import UnitSystem


def _frame(count: int = 120) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    elapsed = np.r_[0.0, (index[1:] - index[:-1]).total_seconds()]
    time = np.arange(count, dtype=float)
    outflow = 15.0 + 0.5 * np.sin(time / 30.0)
    inflow = 20.0 + 1.5 * np.sin(time / 80.0)
    storage = 1_500.0 + np.cumsum((inflow - outflow) * elapsed / 43_560.0)
    return pd.DataFrame({"storage": storage, "outflow": outflow}, index=index)


def _prepared(frame: pd.DataFrame):
    return _prepare_arrays(
        frame.index,
        frame["storage"].to_numpy(),
        frame["outflow"].to_numpy(),
        unit_system=UnitSystem.us_customary(),
        validation_fraction=1.0,
    )


def _score(frame: pd.DataFrame, *, horizons=(1.0,), blocks=1):
    prepared = _prepared(frame)
    return _score_multihorizon(
        prepared,
        np.ones(5),
        forecast_horizons=tuple(float(value) * 3_600.0 for value in horizons),
        horizon_weights=tuple(1.0 / len(horizons) for _ in horizons),
        student_t_degrees_of_freedom=5.0,
        validation_fraction=1.0,
        validation_blocks=blocks,
    )


def test_student_t_nll_matches_reference_formula() -> None:
    innovation = np.array([2.0, -1.0])
    covariance = np.array([[4.0, 1.0], [1.0, 3.0]])
    degrees_of_freedom = 5.0
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0.0
    mahalanobis = float(innovation @ np.linalg.solve(covariance, innovation))
    reference = (
        lgamma(degrees_of_freedom / 2.0)
        - lgamma((degrees_of_freedom + 2.0) / 2.0)
        + 0.5 * logdet
        + (2.0 / 2.0) * np.log(degrees_of_freedom * np.pi)
        + ((degrees_of_freedom + 2.0) / 2.0)
        * np.log1p(mahalanobis / degrees_of_freedom)
    )
    assert student_t_predictive_nll(
        innovation, covariance, degrees_of_freedom
    ) == pytest.approx(reference)


def test_student_t_loss_is_less_aggressive_than_gaussian_for_a_spike() -> None:
    innovation = np.array([20.0])
    covariance = np.array([[1.0]])
    student_t = student_t_predictive_nll(innovation, covariance, 5.0)
    gaussian = 0.5 * (np.log(2.0 * np.pi) + 20.0**2)
    assert student_t < gaussian


def test_regular_horizons_match_exact_15_minute_targets() -> None:
    index = _frame(120).index
    assert _forecast_target_index(index, 0, 3_600.0, 450.0) == 4
    assert _forecast_target_index(index, 0, 21_600.0, 450.0) == 24
    assert _forecast_target_index(index, 0, 86_400.0, 450.0) == 96


def test_irregular_horizon_uses_first_acceptable_future_observation() -> None:
    index = pd.to_datetime(
        [
            "2025-01-01 00:00:00",
            "2025-01-01 00:50:00",
            "2025-01-01 01:23:20",
        ],
        utc=True,
    )
    # The median cadence is 2,500 seconds; the target is 1:00 and 1:23:20 is
    # 1,400 seconds late, inside the documented half-cadence tolerance.
    assert _forecast_target_index(index, 0, 3_600.0, 1_250.0) is None
    assert _forecast_target_index(index, 0, 3_600.0, 1_500.0) == 2


def test_forecast_propagation_does_not_assimilate_intervening_observations() -> None:
    base = _frame(12)
    changed = base.copy(deep=True)
    changed.iloc[2, 0] += 500.0
    changed.iloc[2, 1] += 50.0
    parameters = np.ones(5)
    base_prepared = _prepared(base)
    changed_prepared = _prepare_arrays(
        changed.index,
        changed["storage"].to_numpy(),
        changed["outflow"].to_numpy(),
        unit_system=UnitSystem.us_customary(),
        validation_fraction=1.0,
        initial_covariance=base_prepared.initial_covariance,
    )
    base_filter = _filter_candidate(
        base_prepared, parameters, collect=True
    ).filter_result
    changed_filter = _filter_candidate(
        changed_prepared, parameters, collect=True
    ).filter_result
    assert base_filter is not None and changed_filter is not None
    base_state = _propagate_state(
        base_prepared,
        parameters,
        base_filter.filtered_means[1],
        base_filter.filtered_covariances[1],
        1,
        4,
    )
    changed_state = _propagate_state(
        changed_prepared,
        parameters,
        changed_filter.filtered_means[1],
        changed_filter.filtered_covariances[1],
        1,
        4,
    )
    np.testing.assert_allclose(base_state[0], changed_state[0])
    np.testing.assert_allclose(base_state[1], changed_state[1])


def test_partial_observations_count_only_usable_target_components() -> None:
    frame = _frame(24)
    frame.loc[frame.index[1:], "outflow"] = np.nan
    result = _score(frame)
    assert result.score == pytest.approx(result.block_scores[0])
    assert result.horizon_counts["1h"] == 18


def test_unavailable_horizon_renormalizes_weights() -> None:
    result = _score(_frame(20), horizons=(1.0, 6.0), blocks=1)
    assert result.horizon_counts["1h"] > 0
    assert result.horizon_counts["6h"] == 0
    assert result.horizon_scores["6h"] is None
    assert result.block_scores[0] == pytest.approx(result.horizon_scores["1h"])


def test_validation_blocks_are_distributed_and_contribute() -> None:
    prepared = _prepared(_frame(120))
    blocks = _validation_origin_blocks(prepared, (3_600.0,), 1.0, 4)
    assert len(blocks) == 4
    assert all(block for block in blocks)
    assert blocks[0][-1] < blocks[1][0] < blocks[2][0] < blocks[3][0]
    result = _score(_frame(120), blocks=4)
    assert len(result.block_scores) == 4
    assert all(score is not None for score in result.block_scores)


def test_no_usable_horizon_returns_infinity() -> None:
    result = _score(_frame(20), horizons=(24.0,))
    assert np.isinf(result.score)


def test_public_diagnostics_report_robust_forecast_configuration() -> None:
    result = tune_inflow_model(_frame(), "diagnostics", max_evaluations=4)
    assert (
        result.diagnostics["objective"]
        == "robust_multihorizon_student_t_predictive_negative_log_likelihood"
    )
    assert result.config.configuration_version == "dataframe-tuned-v2"
    assert result.diagnostics["forecast_horizons"] == (1.0, 6.0, 24.0)
    assert result.diagnostics["forecast_horizon_labels"] == ("1h", "6h", "24h")
    assert result.diagnostics["horizon_weights"] == {
        "1h": 0.5,
        "6h": 0.3,
        "24h": 0.2,
    }
    assert result.diagnostics["student_t_degrees_of_freedom"] == 5.0
    assert len(result.diagnostics["per_block_scores"]) == 4
    assert result.diagnostics["aggregate_robust_forecast_score"] == pytest.approx(
        result.score
    )
    for name in (
        "median_absolute_causal_inflow_change",
        "p95_absolute_causal_inflow_change",
        "causal_inflow_std",
        "raw_water_balance_inflow_std",
        "filtered_to_raw_std_ratio",
        "negative_causal_inflow_fraction",
    ):
        assert name in result.diagnostics["inflow_behavior"]


def test_spiky_sensor_data_gets_robust_causal_inflow_without_losing_storage_forecasts(
) -> None:
    frame = _frame(240)
    true_storage = frame["storage"].to_numpy(copy=True)
    frame.loc[frame.index[[80, 130, 180]], "storage"] += [100.0, -120.0, 90.0]
    robust = tune_inflow_model(frame, "spiky", max_evaluations=64)

    legacy_config = ReservoirConfig(
        reservoir_id="spiky",
        reservoir_name="spiky",
        q=np.diag([1.0, 1.0, 1.0]),
        r=np.diag([100.0, 100.0]),
        p0=np.diag([100.0, 100.0, 100.0]),
        smoothing_lag=pd.Timedelta(hours=1).to_pytimedelta(),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="test",
        configuration_version="test",
        unit_system=UnitSystem.us_customary(),
    )
    legacy_data = NoiseTuningData(
        tuple(frame.index.to_pydatetime()),
        frame["storage"].to_numpy(),
        frame["outflow"].to_numpy(),
    )
    legacy = tune_noise(legacy_config, legacy_data, maxiter=64)
    legacy_filter = run_filter_with_noise(
        legacy_config,
        legacy_data,
        q_storage=legacy.q_storage,
        q_inflow=legacy.q_inflow,
        q_outflow=legacy.q_outflow,
        r_storage=legacy.r_storage,
        r_outflow=legacy.r_outflow,
    )
    robust_changes = robust.model_output["estimated_inflow"].diff().abs().dropna()
    legacy_changes = pd.Series(legacy_filter.filtered_means[:, 1]).diff().abs().dropna()
    assert robust_changes.quantile(0.95) < legacy_changes.quantile(0.95)

    # Compare a six-hour causal storage forecast on a held-out period. The
    # robust candidate may trade a small amount of point error for stability,
    # but should not materially degrade this held-out forecast.
    robust_prepared = _prepared(frame)
    robust_filter = _filter_candidate(
        robust_prepared,
        np.asarray(tuple(robust.parameters.values())),
        collect=True,
    ).filter_result
    assert robust_filter is not None
    legacy_parameters = np.array(
        [
            legacy.q_storage,
            legacy.q_inflow,
            legacy.q_outflow,
            legacy.r_storage,
            legacy.r_outflow,
        ]
    )
    legacy_origin_errors = []
    robust_origin_errors = []
    legacy_prepared = _prepared(frame)
    for origin in range(100, 140):
        target = origin + 24
        robust_state = _propagate_state(
            robust_prepared,
            np.asarray(tuple(robust.parameters.values())),
            robust_filter.filtered_means[origin],
            robust_filter.filtered_covariances[origin],
            origin,
            target,
        )[0]
        legacy_state = _propagate_state(
            legacy_prepared,
            legacy_parameters,
            legacy_filter.filtered_means[origin],
            legacy_filter.filtered_covariances[origin],
            origin,
            target,
        )[0]
        robust_origin_errors.append(abs(robust_state[0] - true_storage[target]))
        legacy_origin_errors.append(abs(legacy_state[0] - true_storage[target]))
    assert np.median(robust_origin_errors) <= 2.0 * np.median(legacy_origin_errors)
