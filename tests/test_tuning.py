from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from kalmone import (
    InflowTuningSettings,
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    TuningError,
    TuningWindow,
    UnitSystem,
    elapsed_lag_autocorrelation,
    joint_predictive_nlpd,
    marginal_predictive_nlpd,
    prior_hourly_increment_to_q,
    storage_conditional_nlpd,
    tune_inflow_process_noise,
)
from kalmone.tuning import _paired_standard_error, _prepare


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="demo",
        reservoir_name="Demo",
        q=np.diag([0.0, 0.1, 0.2]),
        r=np.diag([1.0, 2.0]),
        p0=np.diag([10.0, 20.0, 30.0]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.us_customary(),
    )


def _series(count: int = 40) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")
    storage = pd.Series(100.0 + np.arange(count), index=index)
    discharge = pd.Series(4.0, index=index)
    return storage, discharge


def _windows(index: pd.DatetimeIndex) -> tuple[TuningWindow, ...]:
    return tuple(
        TuningWindow(f"window-{i}", index[4 + i * 8], index[8 + i * 8])
        for i in range(3)
    )


def test_predictive_density_helpers_match_scalar_calculation() -> None:
    expected = 0.5 * (np.log(2.0 * np.pi) + np.log(4.0) + 0.25)
    assert marginal_predictive_nlpd(1.0, 4.0) == pytest.approx(expected)
    assert joint_predictive_nlpd(np.array([1.0]), np.array([[4.0]])) == pytest.approx(
        expected
    )


def test_conditional_storage_density_uses_outflow_covariance() -> None:
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    conditional_variance = 4.0 - 1.0 / 2.0
    conditional_innovation = 1.0 - 1.0 * 2.0 / 2.0
    expected = marginal_predictive_nlpd(conditional_innovation, conditional_variance)
    assert storage_conditional_nlpd(1.0, 2.0, covariance) == pytest.approx(expected)


def test_candidate_parameterization_is_hourly_increment_sd() -> None:
    assert prior_hourly_increment_to_q(60.0) == pytest.approx(1.0)


def test_tuning_returns_proposed_config_without_mutating_base() -> None:
    index = pd.date_range("2025-01-01", periods=30, freq="h", tz="UTC")
    storage = pd.Series(100.0 + np.arange(len(index)), index=index)
    discharge = pd.Series(4.0, index=index)
    windows = tuple(
        TuningWindow(f"window-{i}", index[2 + i * 8], index[2 + (i + 1) * 8])
        for i in range(3)
    )
    base = _config()
    result = tune_inflow_process_noise(
        storage,
        discharge,
        base,
        [2.0, 5.0, 10.0],
        windows,
        settings=InflowTuningSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        proposed_configuration_version="config-v2",
    )
    assert result.selected_config.configuration_version == "config-v2"
    assert result.selected_config.q[1, 1] == pytest.approx(result.selected_q_inflow)
    assert base.q[1, 1] == pytest.approx(0.1)
    assert result.regime_diagnostics.shape[0] == 3
    assert result.horizon_diagnostics is not None


def test_tuning_requires_three_validation_windows() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="h", tz="UTC")
    values = pd.Series(np.arange(8.0), index=index)
    window = TuningWindow("only", index[2], index[-1])
    with pytest.raises(ValueError, match="three validation windows"):
        tune_inflow_process_noise(
            values,
            pd.Series(1.0, index=index),
            _config(),
            [2.0, 5.0, 10.0],
            [window],
            settings=InflowTuningSettings(warmup=timedelta(0)),
            proposed_configuration_version="config-v2",
        )


def test_elapsed_autocorrelation_uses_time_not_row_lag() -> None:
    timestamps = pd.DatetimeIndex(
        [
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 1, 3, tzinfo=UTC),
            datetime(2025, 1, 1, 4, tzinfo=UTC),
        ]
    )
    output = elapsed_lag_autocorrelation(
        timestamps, [1.0, 2.0, 4.0, 8.0], timedelta(hours=3)
    )
    assert output.loc[output["lag_seconds"] == 3600.0, "pair_count"].iloc[0] == 2


def test_initialization_excludes_exactly_two_rows() -> None:
    storage, discharge = _series()
    index = storage.index
    windows = (
        TuningWindow("first", index[1], index[4]),
        TuningWindow("second", index[8], index[12]),
        TuningWindow("third", index[16], index[20]),
    )
    result = tune_inflow_process_noise(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=InflowTuningSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        proposed_configuration_version="config-v2",
    )
    first_window = result.window_diagnostics.query("window == 'first'")
    assert first_window["storage_count"].iloc[0] == 2


def test_validation_diagnostics_ignore_out_of_window_future_changes() -> None:
    storage, discharge = _series()
    windows = _windows(storage.index)
    settings = InflowTuningSettings(
        warmup=timedelta(0), min_scored_storage_observations=2
    )
    base = tune_inflow_process_noise(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        proposed_configuration_version="config-v2",
    )
    changed_storage = storage.copy()
    changed_discharge = discharge.copy()
    changed_storage.iloc[32:] += 10_000.0
    changed_discharge.iloc[32:] += 100.0
    changed = tune_inflow_process_noise(
        changed_storage,
        changed_discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        proposed_configuration_version="config-v2",
    )
    columns = [
        "q_inflow",
        "weighted_storage_nlpd",
        "outflow_nis",
        "conditional_storage_nis",
    ]
    pd.testing.assert_frame_equal(
        base.candidate_summary[columns], changed.candidate_summary[columns]
    )
    assert np.isfinite(base.candidate_summary["outflow_nis"]).all()
    assert np.isfinite(base.candidate_summary["conditional_storage_nis"]).all()


def test_coverage_counts_missing_storage_in_eligible_window_rows() -> None:
    storage, discharge = _series()
    missing_index = storage.index[5]
    storage.loc[missing_index] = np.nan
    result = tune_inflow_process_noise(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        _windows(storage.index),
        settings=InflowTuningSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        proposed_configuration_version="config-v2",
    )
    first = result.window_diagnostics.query("window == 'window-0'")
    assert first["coverage"].iloc[0] == pytest.approx(3.0 / 4.0)


def test_multi_horizon_forecast_does_not_cross_window_boundary() -> None:
    storage, discharge = _series()
    index = storage.index
    windows = (
        TuningWindow("first", index[4], index[8]),
        TuningWindow("second", index[12], index[16]),
        TuningWindow("third", index[20], index[24]),
    )
    result = tune_inflow_process_noise(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=InflowTuningSettings(
            warmup=timedelta(0),
            min_scored_storage_observations=2,
            forecast_horizons=(timedelta(hours=3),),
        ),
        proposed_configuration_version="config-v2",
    )
    assert result.horizon_diagnostics["count"].iloc[0] == 3


def test_window_names_are_unique() -> None:
    storage, discharge = _series()
    index = storage.index
    windows = (
        TuningWindow("duplicate", index[4], index[8]),
        TuningWindow("duplicate", index[12], index[16]),
        TuningWindow("third", index[20], index[24]),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        tune_inflow_process_noise(
            storage,
            discharge,
            _config(),
            [2.0, 5.0, 10.0],
            windows,
            settings=InflowTuningSettings(warmup=timedelta(0)),
            proposed_configuration_version="config-v2",
        )


def test_irregular_cached_covariance_matches_direct_model() -> None:
    storage = pd.Series(
        [100.0, 101.0, 103.0, 104.0],
        index=pd.to_datetime(
            [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:17:00Z",
                "2025-01-01T02:03:00Z",
                "2025-01-01T04:11:00Z",
            ]
        ),
    )
    discharge = pd.Series(4.0, index=storage.index)
    prepared = _prepare(storage, discharge, _config())
    q_inflow = 0.7
    cached = (
        prepared.q_storage * prepared.storage_basis
        + q_inflow * prepared.inflow_basis
        + prepared.q_outflow * prepared.outflow_basis
    )
    direct_q = np.asarray(prepared.model.q_continuous).copy()
    direct_q[1, 1] = q_inflow
    direct_model = type(prepared.model)(direct_q, prepared.model.unit_system)
    for index, elapsed in enumerate(prepared.elapsed_seconds):
        np.testing.assert_allclose(
            cached[index],
            direct_model.process_covariance(elapsed),
        )


def test_off_diagonal_covariance_is_rejected() -> None:
    storage, discharge = _series()
    config = _config()
    r = np.asarray(config.r).copy()
    r[0, 1] = r[1, 0] = 0.1
    off_diagonal = ReservoirConfig(
        reservoir_id=config.reservoir_id,
        reservoir_name=config.reservoir_name,
        q=config.q,
        r=r,
        p0=config.p0,
        smoothing_lag=config.smoothing_lag,
        initialization_strategy=config.initialization_strategy,
        inflow_units=config.inflow_units,
        model_version=config.model_version,
        configuration_version=config.configuration_version,
        unit_system=config.unit_system,
    )
    with pytest.raises(ValueError, match="constant diagonal"):
        tune_inflow_process_noise(
            storage,
            discharge,
            off_diagonal,
            [2.0, 5.0, 10.0],
            _windows(storage.index),
            settings=InflowTuningSettings(warmup=timedelta(0)),
            proposed_configuration_version="config-v2",
        )


def test_conservative_selection_and_only_q_inflow_changes() -> None:
    storage, discharge = _series()
    base = _config()
    result = tune_inflow_process_noise(
        storage,
        discharge,
        base,
        [2.0, 5.0, 10.0],
        _windows(storage.index),
        settings=InflowTuningSettings(
            warmup=timedelta(0),
            min_scored_storage_observations=2,
            practical_equivalence_tolerance=1_000.0,
        ),
        proposed_configuration_version="config-v2",
    )
    assert result.selected_prior_hourly_increment_sd == pytest.approx(2.0)
    expected_q = np.asarray(base.q).copy()
    expected_q[1, 1] = result.selected_q_inflow
    np.testing.assert_allclose(result.selected_config.q, expected_q)
    np.testing.assert_allclose(result.selected_config.r, base.r)
    np.testing.assert_allclose(result.selected_config.p0, base.p0)


def test_ineligible_candidate_cannot_be_selected() -> None:
    storage, discharge = _series()
    storage.iloc[4:8] = np.nan
    with pytest.raises(TuningError, match="no candidate passed"):
        tune_inflow_process_noise(
            storage,
            discharge,
            _config(),
            [2.0, 5.0, 10.0],
            _windows(storage.index),
            settings=InflowTuningSettings(
                warmup=timedelta(0), min_scored_storage_observations=2
            ),
            proposed_configuration_version="config-v2",
        )


def test_tuning_does_not_call_rts_smoothing(monkeypatch: pytest.MonkeyPatch) -> None:
    import kalmone.rts as rts

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("RTS smoothing must not be called during tuning")

    monkeypatch.setattr(rts, "smooth_filter_steps", fail)
    storage, discharge = _series()
    tune_inflow_process_noise(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        _windows(storage.index),
        settings=InflowTuningSettings(warmup=timedelta(0)),
        proposed_configuration_version="config-v2",
    )


def test_final_evaluation_is_separate_from_candidate_selection() -> None:
    storage, discharge = _series()
    validation = tune_inflow_process_noise(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        _windows(storage.index),
        settings=InflowTuningSettings(warmup=timedelta(0)),
        proposed_configuration_version="config-v2",
    )
    from kalmone import evaluate_inflow_config

    test_window = TuningWindow("test", storage.index[32], storage.index[-1])
    evaluation = evaluate_inflow_config(
        storage,
        discharge,
        validation.selected_config,
        evaluation_window=test_window,
        settings=InflowTuningSettings(warmup=timedelta(0)),
    )
    assert evaluation.config.configuration_version == "config-v2"
    assert len(evaluation.candidate_summary) == 1
    assert "selected" not in evaluation.candidate_summary.columns


def test_analytic_paired_standard_error_has_finite_sample_correction() -> None:
    differences = np.array([0.0, 2.0, 4.0])
    settings = InflowTuningSettings(bootstrap_samples=1)
    # Sample variance is 4; standard error is sqrt(4 / 3).
    assert _paired_standard_error(
        differences, np.full(3, 1.0 / 3.0), settings
    ) == pytest.approx(np.sqrt(4.0 / 3.0))
