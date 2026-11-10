from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from .._validation import bounded_integer, nonempty_string
from ..kalman import KalmanFilterResult
from ..models import ReservoirStateSpaceModel
from ..reservoir_config import ReservoirConfig

Array = np.ndarray

_PARAMETER_NAMES = (
    "q_storage",
    "q_inflow",
    "q_outflow",
    "r_storage",
    "r_outflow",
)


class NoiseTuningError(ValueError):
    """Raised when a causal noise evaluation cannot be produced."""


def _timestamp(value: datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


def _nonnegative(value: object, name: str) -> float:
    number = float(value)  # type: ignore[arg-type]
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")
    return number


def _optional_nonnegative(value: object | None, name: str) -> float | None:
    return None if value is None else _nonnegative(value, name)


def _positive_increasing_pair(value: object, name: str) -> tuple[float, float]:
    pair = tuple(float(item) for item in value)  # type: ignore[union-attr]
    if (
        len(pair) != 2
        or not np.isfinite(pair).all()
        or pair[0] <= 0.0
        or pair[0] >= pair[1]
    ):
        raise ValueError(f"{name} must be a finite increasing positive pair")
    return pair  # type: ignore[return-value]


@dataclass(frozen=True)
class ValidationWindow:
    """A non-overlapping half-open validation interval."""

    name: str
    start: datetime | pd.Timestamp
    end: datetime | pd.Timestamp
    weight: float | None = None

    def __post_init__(self) -> None:
        name = nonempty_string(self.name, name="window name")
        start = _timestamp(self.start, "window start")
        end = _timestamp(self.end, "window end")
        if end <= start:
            raise ValueError("window end must be after window start")
        if self.weight is not None:
            weight = float(self.weight)
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("window weight must be positive and finite")
            object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index >= self.start) & (index < self.end)


@dataclass(frozen=True)
class NoiseEvaluationSettings:
    """Scoring and calibration settings shared by tuning and frozen evaluation.

    The ``*_warning`` thresholds only produce calibration warnings; they never
    reject a configuration. Set any of them to ``None`` to disable that check.
    """

    warmup: timedelta = timedelta(hours=24)
    innovation_max_lag: timedelta = timedelta(hours=24)
    min_scored_storage_observations: int = 3
    practical_equivalence_tolerance: float = 0.0
    bootstrap_samples: int = 1000
    random_seed: int = 0
    max_jitter_fraction: float = 1e-9
    max_regularized_steps: int = 0
    nis_warning_range: tuple[float, float] | None = (0.7, 1.5)
    innovation_bias_warning: float | None = 0.25
    autocorrelation_warning: float | None = 0.25
    coverage_warning: float | None = 0.10

    def __post_init__(self) -> None:
        if not isinstance(self.warmup, timedelta) or self.warmup < timedelta(0):
            raise ValueError("warmup must be a nonnegative timedelta")
        if (
            not isinstance(self.innovation_max_lag, timedelta)
            or self.innovation_max_lag <= timedelta(0)
        ):
            raise ValueError("innovation_max_lag must be a positive timedelta")
        integers = {
            "min_scored_storage_observations": 1,
            "bootstrap_samples": 1,
            "random_seed": 0,
            "max_regularized_steps": 0,
        }
        for name, minimum in integers.items():
            value = bounded_integer(getattr(self, name), name=name, minimum=minimum)
            object.__setattr__(self, name, value)
        for name in ("practical_equivalence_tolerance", "max_jitter_fraction"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))
        for name in (
            "innovation_bias_warning",
            "autocorrelation_warning",
            "coverage_warning",
        ):
            value = _optional_nonnegative(getattr(self, name), name)
            object.__setattr__(self, name, value)
        if self.nis_warning_range is not None:
            low, high = (float(value) for value in self.nis_warning_range)
            if not (np.isfinite(low) and np.isfinite(high) and 0.0 <= low < high):
                raise ValueError("nis_warning_range must be a finite increasing pair")
            object.__setattr__(self, "nis_warning_range", (low, high))


@dataclass(frozen=True)
class NoiseTuningSettings:
    """Search bounds, search budget, and optional proxy-check thresholds.

    Each ``*_multiplier_bounds`` pair is applied to the matching diagonal
    term of the base configuration. Storage noise is capped at its reviewed
    base value by default: the innovation likelihood alone cannot tell a rapid
    inflow event from an over-dispersed storage channel. Callers with
    independent evidence may widen those bounds explicitly.

    With ``estimate_r_storage`` (the default), ``r_storage`` is instead
    searched between ``estimate / r_storage_estimate_factor`` and
    ``estimate * r_storage_estimate_factor``, where ``estimate`` is the
    storage reading noise measured from the data by
    :func:`estimate_storage_noise`. If the data give no positive estimate,
    ``r_storage_multiplier_bounds`` apply.
    """

    q_storage_multiplier_bounds: tuple[float, float] = (0.5, 1.0)
    q_inflow_multiplier_bounds: tuple[float, float] = (0.001, 10.0)
    q_outflow_multiplier_bounds: tuple[float, float] = (0.1, 10.0)
    r_storage_multiplier_bounds: tuple[float, float] = (0.5, 1.0)
    r_outflow_multiplier_bounds: tuple[float, float] = (0.01, 4.0)
    estimate_r_storage: bool = True
    r_storage_estimate_factor: float = 2.0
    local_searches: int = 2
    max_evaluations_per_search: int = 200
    check_identifiability: bool = True
    # The upstream proxy is a shape/timing diagnostic only. It never enters
    # the likelihood objective or the selection.
    proxy_min_aligned_points: int = 12
    proxy_max_lag: timedelta = timedelta(hours=12)
    proxy_diagnostic_frequency: str | timedelta = "1h"
    proxy_min_shape_correlation: float = 0.20
    proxy_min_change_correlation: float = 0.10

    def __post_init__(self) -> None:
        for name in self.multiplier_bounds_names():
            pair = _positive_increasing_pair(getattr(self, name), name)
            object.__setattr__(self, name, pair)
        integers = {
            "local_searches": 1,
            "max_evaluations_per_search": 1,
            "proxy_min_aligned_points": 3,
        }
        for name, minimum in integers.items():
            value = bounded_integer(getattr(self, name), name=name, minimum=minimum)
            object.__setattr__(self, name, value)
        for name in ("check_identifiability", "estimate_r_storage"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        factor = float(self.r_storage_estimate_factor)
        if not np.isfinite(factor) or factor <= 1.0:
            raise ValueError("r_storage_estimate_factor must be finite and above 1")
        object.__setattr__(self, "r_storage_estimate_factor", factor)
        if (
            not isinstance(self.proxy_max_lag, timedelta)
            or self.proxy_max_lag < timedelta(0)
        ):
            raise ValueError("proxy_max_lag must be a nonnegative timedelta")
        frequency = pd.Timedelta(self.proxy_diagnostic_frequency)
        if frequency <= pd.Timedelta(0):
            raise ValueError("proxy_diagnostic_frequency must be positive")
        object.__setattr__(self, "proxy_diagnostic_frequency", frequency)
        for name in ("proxy_min_shape_correlation", "proxy_min_change_correlation"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
            object.__setattr__(self, name, value)

    @staticmethod
    def multiplier_bounds_names() -> tuple[str, ...]:
        """Return the bound field names in ``_PARAMETER_NAMES`` order."""

        return tuple(f"{name}_multiplier_bounds" for name in _PARAMETER_NAMES)


@dataclass(frozen=True)
class ConfigEvaluationResult:
    """Compact causal diagnostics for one already-frozen configuration."""

    config: ReservoirConfig
    candidate_summary: pd.DataFrame
    window_diagnostics: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _copy_frames(self, ("candidate_summary", "window_diagnostics"))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))


@dataclass(frozen=True)
class NoiseTuningResult:
    """Outcome of a multi-start maximum-likelihood noise search.

    ``candidate_summary`` has one row for the base configuration and one per
    local search. ``identifiability`` holds, for each selected term not held
    at a search bound, the curvature of the objective, an approximate standard
    error of the log value, and the correlations implied by the inverse
    Hessian. It is empty when the check is disabled. ``storage_noise`` is the
    per-window :func:`estimate_storage_noise` table over the scored rows; it
    is empty when ``estimate_r_storage`` is off.
    """

    selected_config: ReservoirConfig
    selected_parameters: Mapping[str, float]
    selected_objective: float
    base_objective: float
    improvement_standard_error: float
    candidate_summary: pd.DataFrame
    window_diagnostics: pd.DataFrame
    identifiability: pd.DataFrame = field(default_factory=pd.DataFrame)
    storage_noise: pd.DataFrame = field(default_factory=pd.DataFrame)
    proxy_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: tuple[str, ...] = ()
    timing_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _copy_frames(
            self,
            (
                "candidate_summary",
                "window_diagnostics",
                "identifiability",
                "storage_noise",
                "proxy_diagnostics",
            ),
        )
        object.__setattr__(
            self,
            "selected_parameters",
            {str(key): float(value) for key, value in self.selected_parameters.items()},
        )
        for name in (
            "selected_objective",
            "base_objective",
            "improvement_standard_error",
        ):
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))
        object.__setattr__(
            self,
            "timing_seconds",
            {str(key): float(value) for key, value in self.timing_seconds.items()},
        )


@dataclass(frozen=True)
class NoiseTuningProgress:
    """One progress update from :func:`tune_inflow_noise`.

    An update is sent when a phase starts (``evaluations == 0``) and after
    every new causal filtering pass. Phases run in order: ``"screening"``,
    ``"search-1"`` to ``"search-k"``, then ``"identifiability"`` when that
    check runs. ``evaluation_limit`` bounds the phase's passes; a search can
    finish well below it, and SciPy may overrun it by one gradient step.
    ``phase_number`` counts from 1 to ``phase_count``. The identifiability
    phase is counted when the check is enabled but skipped when every term
    ends on a search bound, so the run can finish one phase early.
    ``best_objective`` is the lowest objective of any screening or search
    pass so far, the base configuration included. Identifiability passes are
    not counted, because their perturbed settings are never proposed.
    """

    phase: str
    phase_number: int
    phase_count: int
    evaluations: int
    evaluation_limit: int
    best_objective: float
    elapsed_seconds: float


def _copy_frames(instance: object, names: tuple[str, ...]) -> None:
    for name in names:
        frame = getattr(instance, name)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        object.__setattr__(instance, name, frame.copy(deep=True))


@dataclass(frozen=True)
class _Prepared:
    index: pd.DatetimeIndex
    timestamps: pd.DatetimeIndex
    observations: np.ndarray
    transitions: np.ndarray
    storage_basis: np.ndarray
    inflow_basis: np.ndarray
    outflow_basis: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    q_storage: float
    q_outflow: float
    r: np.ndarray
    model: ReservoirStateSpaceModel
    finite_observations: np.ndarray


@dataclass(frozen=True)
class _DiagnosticPlan:
    window_masks: tuple[np.ndarray, ...]
    score_mask: np.ndarray


@dataclass
class _Pass:
    filter_result: KalmanFilterResult | None
    window_rows: dict[str, dict[str, Any]]
    physical: dict[str, float]
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)
