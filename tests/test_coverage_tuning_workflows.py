from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    NoiseEvaluationSettings,
    NoiseTuningError,
    NoiseTuningSettings,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
    evaluate_configuration,
    tune_inflow_noise,
)
from kalmanflow.noise_tuning import _optimization, _workflows


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="workflow-coverage",
        reservoir_name="Workflow coverage",
        q=np.diag([0.002, 1.0, 0.02]),
        r=np.diag([0.25, 4.0]),
        p0=np.diag([10.0, 20.0, 30.0]),
        smoothing_lag=timedelta(hours=1),
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
    count: int = 30,
    *,
    window_count: int = 3,
    search: NoiseTuningSettings | None = None,
) -> dict[str, object]:
    storage, discharge = _inputs(count)
    return {
        "storage": storage,
        "discharge": discharge,
        "base_config": _config(),
        "inflow_increment_sd_seeds": [2.0, 5.0, 10.0],
        "validation_windows": _windows(storage.index, window_count),
        "settings": NoiseEvaluationSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        "tuning_settings": search or NoiseTuningSettings(max_evaluations_per_search=10),
        "proposed_configuration_version": "config-v2",
    }


def test_hard_validity_tolerates_malformed_window_rows() -> None:
    candidate = SimpleNamespace(
        reasons=["duplicate"],
        window_rows={"one": {"storage_count": "bad", "joint_nlpd": "bad"}},
    )
    valid, reasons = _workflows._hard_validity(
        candidate,
        (
            ValidationWindow(
                "one",
                pd.Timestamp("2025-01-01", tz="UTC"),
                pd.Timestamp("2025-01-02", tz="UTC"),
            ),
        ),
        NoiseEvaluationSettings(min_scored_storage_observations=2),
    )
    assert valid is False
    assert reasons.count("duplicate") == 1
    assert any("scored storage observations" in reason for reason in reasons)
    assert any("no finite joint NLPD" in reason for reason in reasons)


@pytest.mark.parametrize(
    ("override", "error", "message"),
    [
        ({"base_config": object()}, TypeError, "base_config"),
        ({"proposed_configuration_version": None}, ValueError, "required"),
        ({"proposed_configuration_version": "  "}, ValueError, "required"),
        ({"proposed_configuration_version": "config-v1"}, ValueError, "must be new"),
        ({"upstream_proxy": object()}, TypeError, "pandas Series"),
        ({"progress": 42}, TypeError, "progress must be callable"),
    ],
)
def test_tuner_validates_public_arguments(
    override: dict[str, object], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        tune_inflow_noise(**{**_tune_kwargs(), **override})  # type: ignore[arg-type]


def test_tuner_requires_three_validation_windows() -> None:
    kwargs = _tune_kwargs()
    windows = kwargs["validation_windows"]
    with pytest.raises(NoiseTuningError, match="at least three"):
        tune_inflow_noise(**{**kwargs, "validation_windows": windows[:2]})  # type: ignore[arg-type,index]


def test_tuner_warns_when_selected_search_did_not_converge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def stalled_search(
        objective: object, start: np.ndarray, max_evaluations: int
    ) -> _optimization._SearchOutcome:
        return _optimization._SearchOutcome(start, 1, False, "budget exhausted")

    monkeypatch.setattr(_optimization, "_local_search", stalled_search)
    result = tune_inflow_noise(**_tune_kwargs())  # type: ignore[arg-type]
    assert any("did not converge: budget exhausted" in w for w in result.warnings)


def test_tuner_warns_when_selection_sits_on_a_search_bound() -> None:
    search = NoiseTuningSettings(
        r_outflow_multiplier_bounds=(10.0, 20.0), max_evaluations_per_search=10
    )
    result = tune_inflow_noise(**_tune_kwargs(search=search))  # type: ignore[arg-type]
    selected = result.candidate_summary.query("selected").iloc[0]
    assert "r_outflow" in selected["at_bounds"]
    assert any("edge of the search bounds" in w for w in result.warnings)
    assert "r_outflow" not in result.identifiability.index


def test_identifiability_is_skipped_when_every_term_is_at_a_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def search_to_corner(
        objective: object, start: np.ndarray, max_evaluations: int
    ) -> _optimization._SearchOutcome:
        # The lower corner beats the base here, so it is selected.
        return _optimization._SearchOutcome(np.zeros(5), 1, True, "corner")

    monkeypatch.setattr(_optimization, "_local_search", search_to_corner)
    result = tune_inflow_noise(**_tune_kwargs())  # type: ignore[arg-type]
    assert result.candidate_summary.query("selected")["candidate"].item() == (
        "search-1"
    )
    assert result.identifiability.empty
    assert not any("identified" in warning for warning in result.warnings)


def test_five_windows_drop_the_weak_uncertainty_warning() -> None:
    result = tune_inflow_noise(**_tune_kwargs(32, window_count=5))  # type: ignore[arg-type]
    assert not any("fewer than five" in warning for warning in result.warnings)


def test_improvement_warnings_cover_unknown_unclear_and_clear_gains() -> None:
    settings = NoiseEvaluationSettings(practical_equivalence_tolerance=0.05)
    warn = _workflows._improvement_warnings
    assert "unknown" in warn(np.nan, np.nan, settings)[0]
    assert "not clearly better" in warn(0.04, 0.01, settings)[0]
    assert "not clearly better" in warn(0.5, 1.0, settings)[0]
    assert warn(0.5, 0.1, settings) == []


def test_improvement_is_unknown_when_base_cannot_be_scored() -> None:
    objective = SimpleNamespace(
        window_losses=lambda result: result, weights=np.array([0.5, 0.5])
    )
    base = SimpleNamespace(result=np.array([np.nan, 1.0]))
    selected = SimpleNamespace(result=np.array([1.0, 1.0]))
    improvement, error = _workflows._improvement_over_base(
        objective,  # type: ignore[arg-type]
        base,  # type: ignore[arg-type]
        selected,  # type: ignore[arg-type]
    )
    assert np.isnan(improvement)
    assert np.isnan(error)


def test_frozen_evaluation_rejects_invalid_configuration_type() -> None:
    storage, discharge = _inputs()
    with pytest.raises(TypeError, match="config must be"):
        evaluate_configuration(storage, discharge, object())  # type: ignore[arg-type]


@pytest.fixture
def unloaded_optimizer() -> Iterator[None]:
    """Force the next tuning call to import the optimizer module afresh."""

    import kalmanflow.noise_tuning as package

    module_name = "kalmanflow.noise_tuning._optimization"
    saved_module = sys.modules.pop(module_name)
    saved_attribute = package.__dict__.pop("_optimization")
    yield
    sys.modules[module_name] = saved_module
    package._optimization = saved_attribute  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("missing_name", "expected", "message"),
    [
        ("scipy", ImportError, "kalmanflow\\[tuning\\]"),
        ("unrelated", ModuleNotFoundError, "synthetic failure"),
    ],
)
@pytest.mark.usefixtures("unloaded_optimizer")
def test_optimizer_import_failures_are_explained_or_propagated(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
    expected: type[Exception],
    message: str,
) -> None:
    original_import = builtins.__import__

    def failing_import(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] == "scipy":
            raise ModuleNotFoundError("synthetic failure", name=missing_name)
        return original_import(name, *args, **kwargs)

    kwargs = _tune_kwargs()
    monkeypatch.setattr(builtins, "__import__", failing_import)
    with pytest.raises(expected, match=message):
        tune_inflow_noise(**kwargs)  # type: ignore[arg-type]
