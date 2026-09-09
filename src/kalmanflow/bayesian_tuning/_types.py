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


class BayesianTuningError(ValueError):
    """Raised when a Bayesian causal evaluation cannot be produced."""


def _timestamp(value: datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


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
class BayesianEvaluationSettings:
    """Settings shared by Bayesian search and frozen-config evaluation."""

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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.warmup, timedelta)
            or self.warmup.total_seconds() < 0.0
        ):
            raise ValueError("warmup must be a nonnegative timedelta")
        if (
            not isinstance(self.innovation_max_lag, timedelta)
            or self.innovation_max_lag.total_seconds() <= 0.0
        ):
            raise ValueError("innovation_max_lag must be a positive timedelta")
        minimum_count = bounded_integer(
            self.min_scored_storage_observations,
            name="min_scored_storage_observations",
            minimum=1,
        )
        practical = float(self.practical_equivalence_tolerance)
        if not np.isfinite(practical) or practical < 0.0:
            raise ValueError("practical_equivalence_tolerance must be nonnegative")
        bootstrap_samples = bounded_integer(
            self.bootstrap_samples, name="bootstrap_samples", minimum=1
        )
        random_seed = bounded_integer(
            self.random_seed, name="random_seed", minimum=0
        )
        max_regularized_steps = bounded_integer(
            self.max_regularized_steps, name="max_regularized_steps", minimum=0
        )
        jitter = float(self.max_jitter_fraction)
        if not np.isfinite(jitter) or jitter < 0.0:
            raise ValueError("max_jitter_fraction must be nonnegative")
        nis_range = self.nis_warning_range
        if nis_range is not None:
            low, high = (float(nis_range[0]), float(nis_range[1]))
            if (
                not np.isfinite(low)
                or not np.isfinite(high)
                or low < 0.0
                or low >= high
            ):
                raise ValueError("nis_warning_range must be a finite increasing pair")
            object.__setattr__(self, "nis_warning_range", (low, high))
        if self.innovation_bias_warning is not None:
            bias = float(self.innovation_bias_warning)
            if not np.isfinite(bias) or bias < 0.0:
                raise ValueError("innovation_bias_warning must be nonnegative")
            object.__setattr__(self, "innovation_bias_warning", bias)
        object.__setattr__(self, "warmup", self.warmup)
        object.__setattr__(self, "innovation_max_lag", self.innovation_max_lag)
        object.__setattr__(
            self,
            "min_scored_storage_observations",
            minimum_count,
        )
        object.__setattr__(self, "practical_equivalence_tolerance", practical)
        object.__setattr__(self, "bootstrap_samples", bootstrap_samples)
        object.__setattr__(self, "random_seed", random_seed)
        object.__setattr__(self, "max_jitter_fraction", jitter)
        object.__setattr__(
            self, "max_regularized_steps", max_regularized_steps
        )


@dataclass(frozen=True)
class ConfigEvaluationResult:
    """Compact causal diagnostics for one already-frozen configuration."""

    config: ReservoirConfig
    candidate_summary: pd.DataFrame
    window_diagnostics: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("candidate_summary", "window_diagnostics"):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            object.__setattr__(self, name, frame.copy(deep=True))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))


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


@dataclass(frozen=True)
class BayesianTuningSettings:
    """Controls the deterministic Bayesian search and calibration gate."""

    total_trials: int = 40
    initial_trials: int = 12
    random_seed: int = 0
    # Storage process/measurement noise is deliberately not allowed to grow
    # above its reviewed base value by default.  Innovation likelihood alone
    # cannot tell a rapid inflow event from an over-dispersed storage channel.
    # Callers with independent evidence may opt into wider bounds explicitly.
    q_storage_multiplier_bounds: tuple[float, float] = (0.5, 1.0)
    q_outflow_multiplier_bounds: tuple[float, float] = (0.1, 10.0)
    r_storage_multiplier_bounds: tuple[float, float] = (0.5, 1.0)
    r_outflow_multiplier_bounds: tuple[float, float] = (0.25, 4.0)
    one_standard_error_weight: float = 1.0
    max_abs_elapsed_lag_autocorrelation: float = 0.25
    acquisition_pool_size: int = 8192
    expected_improvement_xi: float = 0.01
    # Optional upstream gauge/proxy checks.  These are shape/timing diagnostics
    # only; proxy values are never used as total-inflow observations or in the
    # innovation objective.
    proxy_min_aligned_points: int = 12
    proxy_max_lag: timedelta = timedelta(hours=12)
    proxy_diagnostic_frequency: str | timedelta = "1h"
    proxy_min_shape_correlation: float = 0.20
    proxy_min_change_correlation: float = 0.10
    proxy_max_shape_rmse: float = 2.0
    proxy_require_gate: bool = True

    def __post_init__(self) -> None:
        total = bounded_integer(self.total_trials, name="total_trials", minimum=1)
        initial = bounded_integer(
            self.initial_trials, name="initial_trials", minimum=1
        )
        if initial > total:
            raise ValueError("initial_trials must be between one and total_trials")
        random_seed = bounded_integer(
            self.random_seed, name="random_seed", minimum=0
        )
        object.__setattr__(self, "total_trials", total)
        object.__setattr__(self, "initial_trials", initial)
        object.__setattr__(self, "random_seed", random_seed)
        for name in (
            "q_storage_multiplier_bounds",
            "q_outflow_multiplier_bounds",
            "r_storage_multiplier_bounds",
            "r_outflow_multiplier_bounds",
        ):
            bounds = tuple(float(value) for value in getattr(self, name))
            if (
                len(bounds) != 2
                or not np.isfinite(bounds).all()
                or bounds[0] <= 0.0
                or bounds[0] >= bounds[1]
            ):
                raise ValueError(f"{name} must be a finite increasing positive pair")
            object.__setattr__(self, name, bounds)
        weight = float(self.one_standard_error_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("one_standard_error_weight must be nonnegative")
        autocorrelation = float(self.max_abs_elapsed_lag_autocorrelation)
        if not np.isfinite(autocorrelation) or autocorrelation < 0.0:
            raise ValueError("max_abs_elapsed_lag_autocorrelation must be nonnegative")
        acquisition_pool_size = bounded_integer(
            self.acquisition_pool_size,
            name="acquisition_pool_size",
            minimum=128,
        )
        xi = float(self.expected_improvement_xi)
        if not np.isfinite(xi) or xi < 0.0:
            raise ValueError("expected_improvement_xi must be nonnegative")
        proxy_min_aligned_points = bounded_integer(
            self.proxy_min_aligned_points,
            name="proxy_min_aligned_points",
            minimum=3,
        )
        invalid_proxy_lag = not isinstance(self.proxy_max_lag, timedelta) or (
            self.proxy_max_lag < timedelta(0)
        )
        if invalid_proxy_lag:
            raise ValueError("proxy_max_lag must be a nonnegative timedelta")
        proxy_frequency = pd.Timedelta(self.proxy_diagnostic_frequency)
        if proxy_frequency <= pd.Timedelta(0):
            raise ValueError("proxy_diagnostic_frequency must be positive")
        for name in (
            "proxy_min_shape_correlation",
            "proxy_min_change_correlation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < -1.0 or value > 1.0:
                raise ValueError(f"{name} must be between -1 and 1")
            object.__setattr__(self, name, value)
        shape_rmse = float(self.proxy_max_shape_rmse)
        if not np.isfinite(shape_rmse) or shape_rmse < 0.0:
            raise ValueError("proxy_max_shape_rmse must be nonnegative and finite")
        if not isinstance(self.proxy_require_gate, bool):
            raise TypeError("proxy_require_gate must be a bool")
        object.__setattr__(self, "one_standard_error_weight", weight)
        object.__setattr__(self, "max_abs_elapsed_lag_autocorrelation", autocorrelation)
        object.__setattr__(
            self, "acquisition_pool_size", acquisition_pool_size
        )
        object.__setattr__(self, "expected_improvement_xi", xi)
        object.__setattr__(
            self, "proxy_min_aligned_points", proxy_min_aligned_points
        )
        object.__setattr__(self, "proxy_max_lag", self.proxy_max_lag)
        object.__setattr__(self, "proxy_diagnostic_frequency", proxy_frequency)
        object.__setattr__(self, "proxy_max_shape_rmse", shape_rmse)
        object.__setattr__(self, "proxy_require_gate", self.proxy_require_gate)


@dataclass(frozen=True)
class BayesianTuningResult:
    """Compact result of a Bayesian five-parameter innovation search.

    The candidate table is intentionally limited to search and selection
    fields. Detailed Kalman histories are not retained so reports stay small.
    """

    selected_config: ReservoirConfig
    selected_parameters: Mapping[str, float]
    candidate_summary: pd.DataFrame
    window_diagnostics: pd.DataFrame
    competitive_trial_ids: tuple[int, ...]
    selected_objective: float
    selection_threshold: float
    selection_reason: str
    proxy_diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: tuple[str, ...] = ()
    timing_seconds: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "candidate_summary",
            "window_diagnostics",
            "proxy_diagnostics",
        ):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            object.__setattr__(self, name, frame.copy(deep=True))
        object.__setattr__(
            self,
            "selected_parameters",
            {str(key): float(value) for key, value in self.selected_parameters.items()},
        )
        object.__setattr__(
            self,
            "competitive_trial_ids",
            tuple(int(v) for v in self.competitive_trial_ids),
        )
        object.__setattr__(self, "selected_objective", float(self.selected_objective))
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))
        object.__setattr__(
            self,
            "timing_seconds",
            {str(key): float(value) for key, value in self.timing_seconds.items()},
        )
