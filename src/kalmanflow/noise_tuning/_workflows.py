from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from ..reservoir_config import ReservoirConfig
from . import _candidate, _diagnostics, _identifiability, _preparation, _proxy
from ._parameters import _ParameterSpace
from ._types import (
    _PARAMETER_NAMES,
    ConfigEvaluationResult,
    NoiseEvaluationSettings,
    NoiseTuningError,
    NoiseTuningProgress,
    NoiseTuningResult,
    NoiseTuningSettings,
    ValidationWindow,
    _DiagnosticPlan,
    _Pass,
    _Prepared,
)

if TYPE_CHECKING:
    from ._optimization import _SearchOutcome

Array = np.ndarray

_CALIBRATION_COLUMNS = (
    "joint_nis",
    "storage_nis",
    "outflow_nis",
    "storage_nis_variance",
    "outflow_nis_variance",
    "storage_coverage_50",
    "storage_coverage_90",
    "outflow_coverage_50",
    "outflow_coverage_90",
    "storage_bias",
    "outflow_bias",
    "max_material_elapsed_lag_autocorrelation",
)
_WINDOW_COLUMNS = (
    "window",
    "start",
    "end",
    "storage_nlpd",
    "joint_nlpd",
    "joint_nis",
    "storage_nis",
    "outflow_nis",
    "scored_fraction",
    "storage_count",
    "joint_count",
)
_PROXY_COLUMNS = (
    "proxy_aligned_count",
    "proxy_best_lag_seconds",
    "proxy_shape_correlation",
    "proxy_change_correlation",
    "proxy_check_passed",
)


def _hard_validity(
    candidate: Any,
    windows: Sequence[ValidationWindow],
    settings: NoiseEvaluationSettings,
) -> tuple[bool, list[str]]:
    """Apply the numerical checks shared by search and frozen evaluation."""

    reasons = list(candidate.reasons)
    for window in windows:
        row = candidate.window_rows.get(window.name, {})
        try:
            storage_count = int(row.get("storage_count", 0))
        except (TypeError, ValueError):
            storage_count = 0
        if storage_count < settings.min_scored_storage_observations:
            reasons.append(
                f"window {window.name!r} has {storage_count} scored storage "
                "observations; "
                f"at least {settings.min_scored_storage_observations} are required"
            )
        try:
            joint_loss = float(row.get("joint_nlpd", np.nan))
        except (TypeError, ValueError):
            joint_loss = np.nan
        if not np.isfinite(joint_loss):
            reasons.append(f"window {window.name!r} has no finite joint NLPD")
    unique_reasons = list(dict.fromkeys(reasons))
    return not unique_reasons, unique_reasons


class _ProgressReporter:
    """Send :class:`NoiseTuningProgress` updates to an optional callback."""

    def __init__(
        self,
        callback: Callable[[NoiseTuningProgress], None] | None,
        phase_count: int,
    ) -> None:
        self._callback = callback
        self._phase_count = phase_count
        self._started = perf_counter()
        self._phase = ""
        self._phase_number = 0
        self._limit = 0
        self._evaluations = 0
        self._best = np.inf

    def start_phase(self, phase: str, limit: int) -> None:
        self._phase, self._limit, self._evaluations = phase, limit, 0
        self._phase_number += 1
        self._emit()

    def evaluated(self, score: float) -> None:
        self._evaluations += 1
        self._best = min(self._best, score)
        self._emit()

    def _emit(self) -> None:
        if self._callback is None:
            return
        self._callback(
            NoiseTuningProgress(
                phase=self._phase,
                phase_number=self._phase_number,
                phase_count=self._phase_count,
                evaluations=self._evaluations,
                evaluation_limit=self._limit,
                best_objective=float(self._best),
                elapsed_seconds=perf_counter() - self._started,
            )
        )


class _Objective:
    """Weighted mean causal joint NLPD of a parameter set over the windows.

    This is the negative log-likelihood of the validation-window innovations
    per observed component, so minimizing it is maximum-likelihood
    estimation. Scores are cached, so finite-difference steps and the
    identifiability check never refilter the same parameters. A parameter set
    that fails the hard validity checks scores ``inf``.
    """

    def __init__(
        self,
        prepared: _Prepared,
        windows: Sequence[ValidationWindow],
        settings: NoiseEvaluationSettings,
        plan: _DiagnosticPlan,
        progress: _ProgressReporter,
    ) -> None:
        self.prepared = prepared
        self.windows = tuple(windows)
        self.settings = settings
        self.plan = plan
        self.weights = _preparation._window_weights(windows)
        self.progress = progress
        self._scores: dict[tuple[float, ...], float] = {}

    def evaluate(self, parameters: Mapping[str, float]) -> _Pass:
        """Run one causal filtering pass and keep its full diagnostics."""

        result = _candidate._evaluate_candidate(
            self.prepared,
            q_inflow=parameters["q_inflow"],
            q_storage_override=parameters["q_storage"],
            q_outflow_override=parameters["q_outflow"],
            r_override=np.diag([parameters["r_storage"], parameters["r_outflow"]]),
            windows=self.windows,
            settings=self.settings,
            plan=self.plan,
        )
        self.progress.evaluated(self.score(result))
        return result

    def window_losses(self, candidate: _Pass) -> Array:
        return np.asarray(
            [
                candidate.window_rows[window.name].get("joint_nlpd", np.nan)
                for window in self.windows
            ],
            dtype=float,
        )

    def scored_components(self, candidate: _Pass) -> int:
        return sum(
            int(candidate.window_rows[window.name]["joint_count"])
            for window in self.windows
        )

    def score(self, candidate: _Pass) -> float:
        valid, _ = _hard_validity(candidate, self.windows, self.settings)
        if not valid:
            return np.inf
        return float(np.dot(self.weights, self.window_losses(candidate)))

    def __call__(self, parameters: Mapping[str, float]) -> float:
        key = tuple(float(parameters[name]) for name in _PARAMETER_NAMES)
        if key not in self._scores:
            self._scores[key] = self.score(self.evaluate(parameters))
        return self._scores[key]


@dataclass(frozen=True)
class _Candidate:
    """A fully evaluated noise setting: the base configuration or an optimum."""

    name: str
    parameters: dict[str, float]
    result: _Pass
    search: _SearchOutcome | None = None  # None for the base configuration.


def _load_optimizer() -> Any:
    # Keep filtering and frozen evaluation usable without the optimizer stack.
    try:
        from . import _optimization
    except ModuleNotFoundError as error:
        if error.name != "scipy":
            raise
        raise ImportError(
            "Noise tuning requires optional dependencies. "
            "Install them with: pip install 'kalmanflow[tuning]'"
        ) from error
    return _optimization


def tune_inflow_noise(
    storage: pd.Series,
    discharge: pd.Series,
    base_config: ReservoirConfig,
    inflow_increment_sd_seeds: Sequence[float],
    validation_windows: Sequence[ValidationWindow],
    *,
    settings: NoiseEvaluationSettings | None = None,
    tuning_settings: NoiseTuningSettings | None = None,
    upstream_proxy: pd.Series | None = None,
    proposed_configuration_version: str | None = None,
    progress: Callable[[NoiseTuningProgress], None] | None = None,
) -> NoiseTuningResult:
    """Fit diagonal ``Q`` and ``R`` by maximizing the causal innovation likelihood.

    Each inflow increment seed (an hourly standard deviation) gives one
    starting point. The best ``local_searches`` starts are refined with
    bounded L-BFGS-B in log-parameter space, and the best valid optimum is
    proposed. Requires the optional ``kalmanflow[tuning]`` dependencies.
    Calibration, identifiability, and upstream-proxy checks are reported as
    warnings and never change the selection. Proposals require independent
    validation before operational use.

    ``progress``, when given, receives a :class:`NoiseTuningProgress` update
    at the start of each phase and after every filtering pass. Exceptions it
    raises propagate, so raising one cancels the run.
    """

    optimizer = _load_optimizer()
    options = settings or NoiseEvaluationSettings()
    search = tuning_settings or NoiseTuningSettings()
    _validate_tuning_arguments(
        base_config, proposed_configuration_version, upstream_proxy, progress
    )
    windows = _preparation._validate_windows(validation_windows)
    if len(windows) < 3:
        raise NoiseTuningError("at least three validation windows are required")
    prepared = _preparation._prepare(storage, discharge, base_config)
    plan = _preparation._diagnostic_plan(prepared, windows, options)
    space = _ParameterSpace.from_settings(base_config, search)
    starts = space.starting_points(inflow_increment_sd_seeds)
    # Screening, one phase per local search, then the optional identifiability check.
    phase_count = (
        1 + min(search.local_searches, len(starts)) + int(search.check_identifiability)
    )
    objective = _Objective(
        prepared, windows, options, plan, _ProgressReporter(progress, phase_count)
    )

    started = perf_counter()
    candidates = _search(optimizer, objective, space, starts, search)
    search_seconds = perf_counter() - started
    base, selected = candidates[0], _select(candidates, objective)

    started = perf_counter()
    identifiability, identifiability_warnings = (
        _check_identifiability(objective, space, selected)
        if search.check_identifiability
        else (pd.DataFrame(), [])
    )
    identifiability_seconds = perf_counter() - started

    summary = pd.DataFrame(
        [_candidate_row(candidate, objective, space) for candidate in candidates]
    )
    summary["selected"] = summary["candidate"] == selected.name
    proxy_frame = _proxy_frame(
        candidates, selected, prepared, upstream_proxy, search, plan
    )
    improvement, improvement_error = _improvement_over_base(objective, base, selected)
    warnings = [
        *_improvement_warnings(improvement, improvement_error, options),
        *_selection_warnings(selected, space, len(windows)),
        *(
            f"calibration: {warning}"
            for warning in _diagnostics._calibration_warnings(
                selected.result.physical, options
            )
        ),
        *identifiability_warnings,
        *_proxy_warnings(proxy_frame),
    ]
    selected_objective = objective.score(selected.result)
    base_objective = objective.score(base.result)
    return NoiseTuningResult(
        selected_config=_proposed_config(
            base_config,
            selected.parameters,
            str(proposed_configuration_version),
            selection={
                "selected_candidate": selected.name,
                "selected_objective": selected_objective,
                "base_objective": base_objective,
                "improvement_standard_error": improvement_error,
            },
        ),
        selected_parameters=selected.parameters,
        selected_objective=selected_objective,
        base_objective=base_objective,
        improvement_standard_error=improvement_error,
        candidate_summary=summary,
        window_diagnostics=_window_frame((base, selected), windows),
        identifiability=identifiability,
        proxy_diagnostics=proxy_frame,
        warnings=tuple(dict.fromkeys(warnings)),
        timing_seconds={
            "search_seconds": search_seconds,
            "identifiability_seconds": identifiability_seconds,
        },
    )


def _validate_tuning_arguments(
    base_config: object,
    proposed_configuration_version: object,
    upstream_proxy: object,
    progress: object,
) -> None:
    if not isinstance(base_config, ReservoirConfig):
        raise TypeError("base_config must be a ReservoirConfig")
    version = (
        ""
        if proposed_configuration_version is None
        else str(proposed_configuration_version)
    )
    if not version.strip():
        raise ValueError("proposed_configuration_version is required")
    if version == base_config.configuration_version:
        raise ValueError("proposed_configuration_version must be new")
    if upstream_proxy is not None and not isinstance(upstream_proxy, pd.Series):
        raise TypeError("upstream_proxy must be a pandas Series or None")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable or None")


def _search(
    optimizer: Any,
    objective: _Objective,
    space: _ParameterSpace,
    starts: Sequence[Array],
    search: NoiseTuningSettings,
) -> list[_Candidate]:
    """Screen every start, refine the best ones, and return base plus optima."""

    def unit_objective(point: Array) -> float:
        return objective(space.decode(point))

    # Screening scores every start, then the base configuration.
    objective.progress.start_phase("screening", len(starts) + 1)
    ranked_starts = sorted(starts, key=unit_objective)
    candidates = [_Candidate("base", dict(space.base), objective.evaluate(space.base))]
    for number, start in enumerate(ranked_starts[: search.local_searches], start=1):
        # The extra pass keeps the optimum's full diagnostics.
        objective.progress.start_phase(
            f"search-{number}", search.max_evaluations_per_search + 1
        )
        outcome = optimizer._local_search(
            unit_objective, start, search.max_evaluations_per_search
        )
        parameters = space.decode(outcome.point)
        candidates.append(
            _Candidate(
                f"search-{number}",
                parameters,
                objective.evaluate(parameters),
                outcome,
            )
        )
    return candidates


def _select(candidates: Sequence[_Candidate], objective: _Objective) -> _Candidate:
    """Return the valid local optimum with the lowest objective."""

    optima = [
        candidate
        for candidate in candidates
        if candidate.search is not None
        and np.isfinite(objective.score(candidate.result))
    ]
    if not optima:
        raise NoiseTuningError(
            "no local search reached noise settings that pass the hard "
            "validity checks"
        )
    return min(optima, key=lambda candidate: objective.score(candidate.result))


def _check_identifiability(
    objective: _Objective, space: _ParameterSpace, selected: _Candidate
) -> tuple[pd.DataFrame, list[str]]:
    at_bounds = space.names_at_bounds(selected.parameters)
    free = [name for name in _PARAMETER_NAMES if name not in at_bounds]
    if not free:
        return pd.DataFrame(), []
    # A central-difference Hessian over n terms needs 2n² points plus the center.
    objective.progress.start_phase("identifiability", 2 * len(free) ** 2 + 1)
    hessian = _identifiability._log_space_hessian(objective, selected.parameters, free)
    return _identifiability._identifiability_report(
        hessian, free, objective.scored_components(selected.result)
    )


def _candidate_row(
    candidate: _Candidate, objective: _Objective, space: _ParameterSpace
) -> dict[str, Any]:
    valid, reasons = _hard_validity(
        candidate.result, objective.windows, objective.settings
    )
    losses = objective.window_losses(candidate.result)
    calibration = _diagnostics._calibration_warnings(
        candidate.result.physical, objective.settings
    )
    search = candidate.search
    return {
        "candidate": candidate.name,
        **candidate.parameters,
        "inflow_increment_sd": sqrt(candidate.parameters["q_inflow"] * 3600.0),
        "objective": objective.score(candidate.result),
        "window_standard_error": (
            _diagnostics._weighted_standard_error(losses, objective.weights)
            if valid
            else np.nan
        ),
        "eligible": valid,
        "rejection_reasons": "; ".join(reasons),
        "calibrated": valid and not calibration,
        "at_bounds": (
            ", ".join(space.names_at_bounds(candidate.parameters)) if search else ""
        ),
        "evaluations": search.evaluations if search else 1,
        "converged": search.converged if search else True,
        "search_message": search.message if search else "",
        **{
            column: candidate.result.physical.get(column, np.nan)
            for column in _CALIBRATION_COLUMNS
        },
    }


def _improvement_over_base(
    objective: _Objective, base: _Candidate, selected: _Candidate
) -> tuple[float, float]:
    """Return the weighted per-window NLPD gain over base and its paired SE."""

    differences = objective.window_losses(base.result) - objective.window_losses(
        selected.result
    )
    if not np.isfinite(differences).all():
        return np.nan, np.nan
    improvement = float(np.dot(objective.weights, differences))
    error = _diagnostics._paired_standard_error(
        differences, objective.weights, objective.settings
    )
    return improvement, error


def _improvement_warnings(
    improvement: float, error: float, settings: NoiseEvaluationSettings
) -> list[str]:
    if not np.isfinite(improvement):
        return ["the base configuration could not be scored; improvement is unknown"]
    if improvement > max(error, settings.practical_equivalence_tolerance):
        return []
    return [
        "the selected configuration is not clearly better than the base "
        f"configuration (improvement {improvement:.4g} ± {error:.4g} nats per "
        "observed component)"
    ]


def _selection_warnings(
    selected: _Candidate, space: _ParameterSpace, window_count: int
) -> list[str]:
    warnings = []
    if not selected.search.converged:
        warnings.append(
            f"the selected local search did not converge: {selected.search.message}"
        )
    at_bounds = space.names_at_bounds(selected.parameters)
    if at_bounds:
        warnings.append(
            f"selected {', '.join(at_bounds)} at the edge of the search bounds; "
            "the likelihood optimum may lie outside them"
        )
    if window_count < 5:
        warnings.append(
            "fewer than five validation windows are available; uncertainty is weak"
        )
    return warnings


def _window_frame(
    candidates: Sequence[_Candidate], windows: Sequence[ValidationWindow]
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate": candidate.name,
                **{
                    column: candidate.result.window_rows[window.name].get(column)
                    for column in _WINDOW_COLUMNS
                },
            }
            for candidate in candidates
            for window in windows
        ]
    )


def _proxy_frame(
    candidates: Sequence[_Candidate],
    selected: _Candidate,
    prepared: _Prepared,
    upstream_proxy: pd.Series | None,
    search: NoiseTuningSettings,
    plan: _DiagnosticPlan,
) -> pd.DataFrame:
    if upstream_proxy is None:
        return pd.DataFrame()
    rows = []
    for candidate in candidates:
        metrics = _proxy._proxy_metrics(
            candidate.result.filter_result, prepared, upstream_proxy, search, plan
        )
        rows.append(
            {
                "candidate": candidate.name,
                **{column: metrics[column] for column in _PROXY_COLUMNS},
                "selected": candidate is selected,
            }
        )
    return pd.DataFrame(rows)


def _proxy_warnings(proxy_frame: pd.DataFrame) -> list[str]:
    if proxy_frame.empty:
        return []
    if proxy_frame.loc[proxy_frame["selected"], "proxy_check_passed"].iloc[0]:
        return []
    return ["the selected configuration fails the upstream-proxy shape/timing check"]


def _proposed_config(
    base: ReservoirConfig,
    selected: Mapping[str, float],
    proposed_configuration_version: str,
    *,
    selection: Mapping[str, Any],
) -> ReservoirConfig:
    q = np.asarray(base.q, dtype=float).copy()
    r = np.asarray(base.r, dtype=float).copy()
    q[0, 0], q[1, 1], q[2, 2] = (
        selected["q_storage"],
        selected["q_inflow"],
        selected["q_outflow"],
    )
    r[0, 0], r[1, 1] = selected["r_storage"], selected["r_outflow"]
    metadata = dict(base.metadata)
    metadata.update(
        {
            "calibration_method": "causal-innovation-maximum-likelihood",
            "tuned_parameters": list(_PARAMETER_NAMES),
            "selected_parameters": dict(selected),
            "selection": dict(selection),
        }
    )
    return ReservoirConfig(
        reservoir_id=base.reservoir_id,
        reservoir_name=base.reservoir_name,
        q=q,
        r=r,
        p0=np.asarray(base.p0).copy(),
        smoothing_lag=base.smoothing_lag,
        model_version=base.model_version,
        configuration_version=proposed_configuration_version,
        metadata=metadata,
        unit_system=base.unit_system,
    )


def evaluate_configuration(
    storage: pd.Series,
    discharge: pd.Series,
    config: ReservoirConfig,
    *,
    evaluation_window: ValidationWindow | None = None,
    settings: NoiseEvaluationSettings | None = None,
) -> ConfigEvaluationResult:
    """Evaluate one frozen configuration on an independent validation period.

    This is an assessment operation only: it does not search or modify the
    supplied configuration and is suitable for held-out post-search checks.
    """

    options = settings or NoiseEvaluationSettings()
    if not isinstance(config, ReservoirConfig):
        raise TypeError("config must be a ReservoirConfig")
    prepared = _preparation._prepare(storage, discharge, config)
    if evaluation_window is None:
        evaluation_window = ValidationWindow(
            "evaluation",
            prepared.index[0],
            prepared.index[-1] + pd.Timedelta(nanoseconds=1),
        )
    windows = _preparation._validate_windows((evaluation_window,))
    q_inflow = float(config.q[1, 1])
    candidate = _candidate._evaluate_candidate(
        prepared,
        q_inflow=q_inflow,
        windows=windows,
        settings=options,
        plan=_preparation._diagnostic_plan(prepared, windows, options),
    )
    eligible, validity_reasons = _hard_validity(candidate, windows, options)
    calibration_warnings = _diagnostics._calibration_warnings(
        candidate.physical, options
    )
    window_frame = pd.DataFrame(
        [
            {
                "window": window.name,
                "start": window.start,
                "end": window.end,
                **candidate.window_rows[window.name],
            }
            for window in windows
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "q_storage": float(config.q[0, 0]),
                "q_inflow": q_inflow,
                "q_outflow": float(config.q[2, 2]),
                "r_storage": float(config.r[0, 0]),
                "r_outflow": float(config.r[1, 1]),
                "eligible": eligible,
                "calibrated": eligible and not calibration_warnings,
                "rejection_reasons": "; ".join(validity_reasons),
                **candidate.physical,
            }
        ]
    )
    return ConfigEvaluationResult(
        config=config,
        candidate_summary=summary,
        window_diagnostics=window_frame,
        warnings=tuple(dict.fromkeys((*validity_reasons, *calibration_warnings))),
    )


__all__ = [
    "ConfigEvaluationResult",
    "NoiseEvaluationSettings",
    "NoiseTuningError",
    "NoiseTuningProgress",
    "NoiseTuningResult",
    "NoiseTuningSettings",
    "ValidationWindow",
    "evaluate_configuration",
    "tune_inflow_noise",
]
