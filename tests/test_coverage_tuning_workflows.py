from __future__ import annotations

import builtins
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    BayesianEvaluationSettings,
    BayesianTuningError,
    BayesianTuningSettings,
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
    evaluate_configuration,
    tune_inflow_noise_bayesian,
)
from kalmanflow.bayesian_tuning import _diagnostics, _optimization, _workflows


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="workflow-coverage",
        reservoir_name="Workflow coverage",
        q=np.diag([0.002, 1.0, 0.02]),
        r=np.diag([0.25, 4.0]),
        p0=np.diag([10.0, 20.0, 30.0]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.us_customary(),
    )


def _inputs(count: int = 30) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")
    return (
        pd.Series(100.0 + np.arange(count), index=index),
        pd.Series(4.0, index=index),
    )


def _windows(index: pd.DatetimeIndex, count: int = 3) -> tuple[ValidationWindow, ...]:
    width = (len(index) - 2) // count
    return tuple(
        ValidationWindow(
            f"window-{i}",
            index[2 + i * width],
            (
                index[2 + (i + 1) * width]
                if i + 1 < count
                else index[-1] + pd.Timedelta(hours=1)
            ),
        )
        for i in range(count)
    )


def _tune_kwargs(
    storage: pd.Series,
    discharge: pd.Series,
    windows: tuple[ValidationWindow, ...],
    *,
    settings: BayesianEvaluationSettings | None = None,
    search: BayesianTuningSettings | None = None,
) -> dict[str, object]:
    return {
        "storage": storage,
        "discharge": discharge,
        "base_config": _config(),
        "inflow_increment_sd_seeds": [2.0, 5.0, 10.0],
        "validation_windows": windows,
        "settings": settings
        or BayesianEvaluationSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        "bayesian_settings": search
        or BayesianTuningSettings(
            total_trials=3, initial_trials=3, acquisition_pool_size=128
        ),
        "proposed_configuration_version": "config-v2",
    }


def test_workflow_calibration_and_validity_helpers_cover_threshold_edges() -> None:
    search = BayesianTuningSettings(max_abs_elapsed_lag_autocorrelation=0.25)
    no_thresholds = BayesianEvaluationSettings(
        nis_warning_range=None,
        innovation_bias_warning=None,
    )
    violation, calibrated = _workflows._calibration_violation(
        {"max_material_elapsed_lag_autocorrelation": 0.0},
        no_thresholds,
        search,
    )
    assert violation == 0.0
    assert calibrated is True

    values = {
        "joint_nis": 0.1,
        "storage_nis": 2.0,
        "outflow_nis": np.nan,
        "storage_bias": 1.0,
        "outflow_bias": np.nan,
        "conditional_storage_bias": -1.0,
        "max_material_elapsed_lag_autocorrelation": 0.1,
    }
    violation, calibrated = _workflows._calibration_violation(
        values,
        BayesianEvaluationSettings(
            nis_warning_range=(0.5, 1.5), innovation_bias_warning=0.25
        ),
        search,
    )
    assert violation > 0.0
    assert calibrated is False

    candidate = type(
        "Candidate",
        (),
        {
            "reasons": ["duplicate"],
            "window_rows": {
                "one": {"storage_count": "bad", "joint_nlpd": "bad"}
            },
        },
    )()
    valid, reasons = _workflows._hard_validity(
        candidate,
        (
            ValidationWindow(
                "one",
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-01-02", tz="UTC"),
            ),
        ),
        BayesianEvaluationSettings(min_scored_storage_observations=2),
    )
    assert valid is False
    assert reasons.count("duplicate") == 1

    warnings = _workflows._calibration_warnings(
        {}, BayesianEvaluationSettings()
    )
    assert any("unavailable" in warning for warning in warnings)
    assert _workflows._calibration_warnings(
        {
            "joint_nis": 1.0,
            "storage_nis": 1.0,
            "outflow_nis": 1.0,
            "storage_bias": 0.0,
            "outflow_bias": 0.0,
            "conditional_storage_bias": 0.0,
        },
        BayesianEvaluationSettings(),
    ) == ()
    bias_warnings = _workflows._calibration_warnings(
        {
            "joint_nis": 1.0,
            "storage_nis": 1.0,
            "outflow_nis": 1.0,
            "storage_bias": 1.0,
            "outflow_bias": 0.0,
            "conditional_storage_bias": 0.0,
        },
        BayesianEvaluationSettings(),
    )
    assert any("storage_bias=" in warning for warning in bias_warnings)


def test_tuner_validates_public_arguments() -> None:
    storage, discharge = _inputs()
    windows = _windows(storage.index)
    kwargs = _tune_kwargs(storage, discharge, windows)
    with pytest.raises(TypeError, match="base_config"):
        tune_inflow_noise_bayesian(**{**kwargs, "base_config": object()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="required"):
        tune_inflow_noise_bayesian(
            **{**kwargs, "proposed_configuration_version": None}
        )  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be new"):
        tune_inflow_noise_bayesian(
            **{**kwargs, "proposed_configuration_version": "config-v1"}
        )
    with pytest.raises(BayesianTuningError, match="at least three"):
        tune_inflow_noise_bayesian(
            **{**kwargs, "validation_windows": windows[:2]}
        )
    with pytest.raises(TypeError, match="pandas Series"):
        tune_inflow_noise_bayesian(**{**kwargs, "upstream_proxy": object()})  # type: ignore[arg-type]


def test_tuner_random_fallback_handles_duplicate_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge = _inputs()
    windows = _windows(storage.index)
    search = BayesianTuningSettings(
        total_trials=4,
        initial_trials=3,
        acquisition_pool_size=128,
        random_seed=9,
    )
    def duplicate_acquire(*args: object, **kwargs: object) -> tuple[np.ndarray, str]:
        observed = args[2]
        return np.asarray(observed[0], dtype=float), "expected-improvement"  # type: ignore[index]

    monkeypatch.setattr(_optimization, "_fit_and_acquire", duplicate_acquire)
    result = tune_inflow_noise_bayesian(
        **_tune_kwargs(storage, discharge, windows, search=search)
    )
    assert "random-fallback" in set(result.candidate_summary["acquisition_source"])


def test_tuner_uses_best_trial_when_competitive_set_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge = _inputs()
    windows = _windows(storage.index)
    monkeypatch.setattr(_diagnostics, "_paired_standard_error", lambda *args: np.nan)
    result = tune_inflow_noise_bayesian(
        **_tune_kwargs(storage, discharge, windows)
    )
    assert len(result.competitive_trial_ids) == 1
    assert bool(result.candidate_summary["selected"].sum())


def test_tuner_reports_calibrated_selection_with_five_windows() -> None:
    storage, discharge = _inputs(32)
    windows = _windows(storage.index, count=5)
    settings = BayesianEvaluationSettings(
        warmup=timedelta(0),
        min_scored_storage_observations=2,
        nis_warning_range=(0.0, 1e9),
        innovation_bias_warning=None,
    )
    search = BayesianTuningSettings(
        total_trials=3,
        initial_trials=3,
        acquisition_pool_size=128,
        max_abs_elapsed_lag_autocorrelation=1e9,
    )
    result = tune_inflow_noise_bayesian(
        **_tune_kwargs(storage, discharge, windows, settings=settings, search=search)
    )
    assert len(windows) == 5
    assert not any("fewer than five" in warning for warning in result.warnings)
    selected = result.candidate_summary.loc[
        result.candidate_summary["selected"]
    ].iloc[0]
    assert bool(selected["calibrated"])


def test_frozen_evaluation_rejects_invalid_configuration_type() -> None:
    storage, discharge = _inputs()
    with pytest.raises(TypeError, match="config must be"):
        evaluate_configuration(storage, discharge, object())  # type: ignore[arg-type]


def test_tuner_explains_missing_optional_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kalmanflow.bayesian_tuning as package

    module_name = "kalmanflow.bayesian_tuning._optimization"
    saved_module = sys.modules.pop(module_name, None)
    saved_attribute = getattr(package, "_optimization", None)
    if hasattr(package, "_optimization"):
        delattr(package, "_optimization")
    original_import = builtins.__import__

    def missing_scipy(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "scipy" or name.startswith("scipy."):
            raise ModuleNotFoundError("No module named 'scipy'", name="scipy")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_scipy)
    storage, discharge = _inputs()
    windows = _windows(storage.index)
    try:
        with pytest.raises(ImportError, match="optional dependencies"):
            tune_inflow_noise_bayesian(**_tune_kwargs(storage, discharge, windows))
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)
        if saved_module is not None:
            sys.modules[module_name] = saved_module
        if saved_attribute is not None:
            package._optimization = saved_attribute


def test_tuner_propagates_unrelated_optimizer_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kalmanflow.bayesian_tuning as package

    module_name = "kalmanflow.bayesian_tuning._optimization"
    saved_module = sys.modules.pop(module_name, None)
    saved_attribute = getattr(package, "_optimization", None)
    if hasattr(package, "_optimization"):
        delattr(package, "_optimization")
    original_import = builtins.__import__

    def unrelated_failure(
        name: str,
        globals: object | None = None,
        locals: object | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("scipy."):
            raise ModuleNotFoundError("synthetic unrelated failure", name="unrelated")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", unrelated_failure)
    storage, discharge = _inputs()
    windows = _windows(storage.index)
    try:
        with pytest.raises(ModuleNotFoundError, match="unrelated"):
            tune_inflow_noise_bayesian(**_tune_kwargs(storage, discharge, windows))
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)
        if saved_module is not None:
            sys.modules[module_name] = saved_module
        if saved_attribute is not None:
            package._optimization = saved_attribute
