"""Bayesian innovation tuning for five diagonal ``Q``/``R`` terms.

It searches a five-dimensional log parameter space using a causal, pre-update
innovation evaluator. No true inflow or smoothed state is used by the
objective.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import pi, sqrt
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from .kalman import KalmanFilterResult, kalman_filter
from .models import ReservoirStateSpaceModel
from .reservoir_config import ReservoirConfig

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
        if not str(self.name).strip():
            raise ValueError("window name must not be empty")
        start = _timestamp(self.start, "window start")
        end = _timestamp(self.end, "window end")
        if end <= start:
            raise ValueError("window end must be after window start")
        if self.weight is not None:
            weight = float(self.weight)
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("window weight must be positive and finite")
            object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "name", str(self.name))
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
        for name in ("warmup", "innovation_max_lag"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value.total_seconds() < 0.0:
                raise ValueError(f"{name} must be a nonnegative timedelta")
        if int(self.min_scored_storage_observations) < 1:
            raise ValueError("min_scored_storage_observations must be positive")
        practical = float(self.practical_equivalence_tolerance)
        if not np.isfinite(practical) or practical < 0.0:
            raise ValueError("practical_equivalence_tolerance must be nonnegative")
        if int(self.bootstrap_samples) < 1:
            raise ValueError("bootstrap_samples must be positive")
        if int(self.max_regularized_steps) < 0:
            raise ValueError("max_regularized_steps must be nonnegative")
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
            int(self.min_scored_storage_observations),
        )
        object.__setattr__(self, "practical_equivalence_tolerance", practical)
        object.__setattr__(self, "bootstrap_samples", int(self.bootstrap_samples))
        object.__setattr__(self, "random_seed", int(self.random_seed))
        object.__setattr__(self, "max_jitter_fraction", jitter)
        object.__setattr__(
            self, "max_regularized_steps", int(self.max_regularized_steps)
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
    source_positions: np.ndarray
    timestamps: pd.DatetimeIndex
    observations: np.ndarray
    elapsed_seconds: np.ndarray
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
    timestamp_seconds: np.ndarray
    finite_observations: np.ndarray


@dataclass(frozen=True)
class _DiagnosticPlan:
    window_masks: tuple[np.ndarray, ...]
    score_mask: np.ndarray


@dataclass
class _Pass:
    q_inflow: float
    filter_result: KalmanFilterResult | None
    window_rows: dict[str, dict[str, Any]]
    physical: dict[str, float]
    regularization_count: int
    max_jitter: float
    reasons: list[str] = field(default_factory=list)
    diagnostics: dict[str, np.ndarray] = field(default_factory=dict)


def _hourly_increment_sd_to_q(hourly_increment_sd: float) -> float:
    value = float(hourly_increment_sd)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("hourly inflow increment SD must be positive and finite")
    return value * value / 3600.0


def _marginal_predictive_nlpd(
    innovation: float, variance: float, *, max_jitter_fraction: float = 0.0
) -> float:
    value, _, _ = _stable_logpdf(
        np.array([innovation]), np.array([[variance]]), max_jitter_fraction
    )
    if value is None:
        raise ValueError("predictive variance is not positive")
    return float(value)


def _storage_conditional_nlpd(
    storage_innovation: float,
    outflow_innovation: float,
    innovation_covariance: Array,
    *,
    max_jitter_fraction: float = 0.0,
) -> float:
    covariance = np.asarray(innovation_covariance, dtype=float)
    if covariance.shape != (2, 2):
        raise ValueError("innovation_covariance must have shape (2, 2)")
    covariance = (covariance + covariance.T) / 2.0
    soo = float(covariance[1, 1])
    if soo <= 0.0 and max_jitter_fraction > 0.0:
        _, _, jitter = _stable_logpdf(np.zeros(2), covariance, max_jitter_fraction)
        if np.isfinite(jitter) and jitter > 0.0:
            covariance = covariance + np.eye(2) * jitter
            soo = float(covariance[1, 1])
    if not np.isfinite(soo) or soo <= 0.0:
        raise ValueError("outflow predictive variance must be positive")
    conditional_innovation = (
        float(storage_innovation)
        - float(covariance[0, 1]) * float(outflow_innovation) / soo
    )
    conditional_variance = (
        float(covariance[0, 0])
        - float(covariance[0, 1]) * float(covariance[1, 0]) / soo
    )
    return _marginal_predictive_nlpd(
        conditional_innovation,
        conditional_variance,
        max_jitter_fraction=max_jitter_fraction,
    )


def _validate_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("storage and discharge indexes must be DatetimeIndex")
    if index.tz is None:
        raise ValueError("storage and discharge indexes must be timezone-aware")
    if (
        len(index) < 2
        or index.hasnans
        or not index.is_monotonic_increasing
        or index.has_duplicates
    ):
        raise ValueError(
            "timestamps must be timezone-aware, strictly increasing, and nonmissing"
        )
    return index


def _timestamp_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(index.tz_convert("UTC").tz_localize(None))
    unit, multiplier = np.datetime_data(values.dtype)
    seconds_per_unit = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "m": 60.0,
        "h": 3600.0,
        "D": 86400.0,
    }
    try:
        scale = seconds_per_unit[unit]
    except KeyError as error:
        raise ValueError(f"unsupported timestamp resolution: {unit}") from error
    return values.astype(np.int64) * (float(multiplier) * scale)


def _elapsed_lag_autocorrelation(
    timestamps: Sequence[datetime] | pd.DatetimeIndex,
    values: Sequence[float],
    max_lag: timedelta,
    *,
    lag_tolerance: timedelta | None = None,
) -> pd.DataFrame:
    """Calculate pairwise innovation autocorrelation in elapsed-time bins."""

    index = _validate_index(pd.DatetimeIndex(timestamps))
    series = np.asarray(values, dtype=float)
    if series.shape != (len(index),):
        raise ValueError("values must match timestamps")
    limit = float(max_lag.total_seconds())
    if limit <= 0.0:
        raise ValueError("max_lag must be positive")
    finite = np.isfinite(series)
    if finite.sum() < 2:
        return pd.DataFrame(columns=["lag_seconds", "autocorrelation", "pair_count"])
    seconds = _timestamp_seconds(index)
    cadence = float(np.min(np.diff(seconds))) if len(index) > 1 else limit
    tolerance = float(
        (lag_tolerance or timedelta(seconds=max(cadence * 0.25, 1.0))).total_seconds()
    )
    targets = np.arange(cadence, limit + cadence * 0.5, cadence)
    finite_values = series[finite]
    centre = float(np.mean(finite_values))
    variance = float(np.sum((finite_values - centre) ** 2))
    if variance <= 0.0:
        variance = np.nan
    tick_differences = np.diff(index.asi8)
    regular = len(seconds) > 2 and np.all(tick_differences == tick_differences[0])
    if regular and finite.all():
        centred = series - centre
        size = 1 << (2 * len(centred) - 1).bit_length()
        convolution = np.fft.irfft(
            np.fft.rfft(centred, size) * np.conjugate(np.fft.rfft(centred, size)),
            size,
        )
        rows: list[dict[str, float | int]] = []
        for target in targets:
            lower = max(1, int(np.ceil((target - tolerance) / cadence - 1e-12)))
            upper = min(
                len(centred) - 1,
                int(np.floor((target + tolerance) / cadence + 1e-12)),
            )
            steps = np.arange(lower, upper + 1, dtype=int)
            pairs = int(np.sum(len(centred) - steps)) if len(steps) else 0
            covariance = float(np.sum(convolution[steps])) if len(steps) else np.nan
            rows.append(
                {
                    "lag_seconds": float(target),
                    "autocorrelation": covariance / variance
                    if pairs and np.isfinite(variance)
                    else np.nan,
                    "pair_count": pairs,
                }
            )
        return pd.DataFrame(rows)
    finite_seconds = seconds[finite]
    centered = finite_values - centre
    rows = []
    for target in targets:
        lower = np.searchsorted(
            finite_seconds, finite_seconds + target - tolerance, side="left"
        )
        upper = np.searchsorted(
            finite_seconds, finite_seconds + target + tolerance, side="right"
        )
        counts = np.maximum(upper - lower, 0)
        total = int(counts.sum())
        pair_count = 0
        covariance = 0.0
        if total:
            if total <= 2_000_000:
                starts = np.cumsum(counts) - counts
                source = np.repeat(np.arange(len(finite_seconds)), counts)
                local = np.arange(total) - np.repeat(starts, counts)
                matching = np.repeat(lower, counts) + local
                valid = matching > source
                pair_count = int(np.sum(valid))
                covariance = float(
                    np.dot(centered[source[valid]], centered[matching[valid]])
                )
            else:
                for begin in range(0, len(finite_seconds), 4096):
                    end = min(begin + 4096, len(finite_seconds))
                    chunk_counts = counts[begin:end]
                    chunk_total = int(chunk_counts.sum())
                    if not chunk_total:
                        continue
                    starts = np.cumsum(chunk_counts) - chunk_counts
                    source = np.repeat(np.arange(begin, end), chunk_counts)
                    local = np.arange(chunk_total) - np.repeat(starts, chunk_counts)
                    matching = np.repeat(lower[begin:end], chunk_counts) + local
                    valid = matching > source
                    pair_count += int(np.sum(valid))
                    covariance += float(
                        np.dot(centered[source[valid]], centered[matching[valid]])
                    )
        rows.append(
            {
                "lag_seconds": float(target),
                "autocorrelation": covariance / variance
                if pair_count and np.isfinite(variance)
                else np.nan,
                "pair_count": pair_count,
            }
        )
    return pd.DataFrame(rows)


def _validate_windows(
    windows: Sequence[ValidationWindow],
) -> tuple[ValidationWindow, ...]:
    result = tuple(
        window if isinstance(window, ValidationWindow) else ValidationWindow(*window)
        for window in windows
    )
    if not result:
        raise ValueError("at least one validation window is required")
    if len({window.name for window in result}) != len(result):
        raise ValueError("validation window names must be unique")
    ordered = sorted(result, key=lambda value: value.start)
    if any(
        left.end > right.start
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("validation windows must not overlap")
    return result


def _window_weights(windows: Sequence[ValidationWindow]) -> np.ndarray:
    values = np.asarray(
        [1.0 if window.weight is None else window.weight for window in windows],
        dtype=float,
    )
    return values / values.sum()


def _is_diagonal(value: Array) -> bool:
    array = np.asarray(value, dtype=float)
    return array.ndim == 2 and np.allclose(
        array, np.diag(np.diag(array)), rtol=0.0, atol=0.0
    )


def _process_covariance_bases(
    elapsed: Array, conversion: float
) -> tuple[Array, Array, Array]:
    storage = np.zeros((len(elapsed), 3, 3), dtype=float)
    inflow = np.zeros_like(storage)
    outflow = np.zeros_like(storage)
    storage[:, 0, 0] = elapsed
    coupling = conversion * elapsed**2 / 2.0
    integrated = conversion**2 * elapsed**3 / 3.0
    inflow[:, 0, 0] = integrated
    inflow[:, 0, 1] = coupling
    inflow[:, 1, 0] = coupling
    inflow[:, 1, 1] = elapsed
    outflow[:, 0, 0] = integrated
    outflow[:, 0, 2] = -coupling
    outflow[:, 2, 0] = -coupling
    outflow[:, 2, 2] = elapsed
    return storage, inflow, outflow


def _prepare(
    storage: pd.Series, discharge: pd.Series, config: ReservoirConfig
) -> _Prepared:
    if not isinstance(storage, pd.Series) or not isinstance(discharge, pd.Series):
        raise TypeError("storage and discharge must be pandas Series")
    if not storage.index.equals(discharge.index):
        raise ValueError("storage and discharge indexes must match exactly")
    index = _validate_index(storage.index)
    storage_values = storage.to_numpy(dtype=float, copy=True)
    discharge_values = discharge.to_numpy(dtype=float, copy=True)
    q = np.asarray(config.q, dtype=float)
    r = np.asarray(config.r, dtype=float)
    p0 = np.asarray(config.p0, dtype=float)
    if not _is_diagonal(q) or not _is_diagonal(r) or not _is_diagonal(p0):
        raise ValueError(
            "Bayesian evaluation requires constant diagonal q, r, and p0 matrices"
        )
    finite_storage = np.flatnonzero(np.isfinite(storage_values))
    if len(finite_storage) < 2:
        raise BayesianTuningError(
            "at least two finite storage observations are required"
        )
    first, second = int(finite_storage[0]), int(finite_storage[1])
    if not np.isfinite(discharge_values[first]):
        raise ValueError("the first finite storage observation needs finite discharge")
    positions = np.concatenate(
        (
            np.array([first, second], dtype=int),
            np.arange(second + 1, len(index), dtype=int),
        )
    )
    timestamps = index[positions]
    observations = np.column_stack(
        (storage_values[positions], discharge_values[positions])
    )
    elapsed = np.asarray(
        [
            (timestamps[i] - timestamps[i - 1]).total_seconds()
            for i in range(1, len(timestamps))
        ],
        dtype=float,
    )
    model = ReservoirStateSpaceModel(q_continuous=q, unit_system=config.unit_system)
    conversion = float(config.unit_system.flow_to_volume_per_second)
    transitions = np.asarray(
        [model.transition_matrix(value) for value in elapsed], dtype=float
    )
    storage_basis, inflow_basis, outflow_basis = _process_covariance_bases(
        elapsed, conversion
    )
    initial_mean = np.array(
        [
            storage_values[first],
            model.initial_inflow(
                storage_values[first],
                storage_values[second],
                discharge_values[first],
                elapsed[0],
            ),
            model.initial_outflow(discharge_values[first]),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(initial_mean)):
        raise BayesianTuningError("initial state is not finite")
    return _Prepared(
        index,
        positions,
        timestamps,
        observations,
        elapsed,
        transitions,
        storage_basis,
        inflow_basis,
        outflow_basis,
        initial_mean,
        p0.copy(),
        float(q[0, 0]),
        float(q[2, 2]),
        r.copy(),
        model,
        _timestamp_seconds(index[positions]),
        np.isfinite(observations),
    )


def _diagnostic_plan(
    prepared: _Prepared,
    windows: Sequence[ValidationWindow],
    settings: BayesianEvaluationSettings,
) -> _DiagnosticPlan:
    timestamps = prepared.timestamps
    window_masks = tuple(window.mask(timestamps) for window in windows)
    score_mask = np.arange(len(timestamps), dtype=int) >= 2
    score_mask &= timestamps >= timestamps[1] + pd.Timedelta(
        seconds=settings.warmup.total_seconds()
    )
    return _DiagnosticPlan(window_masks=window_masks, score_mask=score_mask)


def _stable_logpdf(
    vector: Array, covariance: Array, max_jitter_fraction: float
) -> tuple[float | None, float, float]:
    vector = np.asarray(vector, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    if (
        matrix.shape != (len(vector), len(vector))
        or not np.isfinite(vector).all()
        or not np.isfinite(matrix).all()
    ):
        return None, np.nan, 0.0
    matrix = (matrix + matrix.T) / 2.0
    scale = max(float(np.max(np.abs(np.diag(matrix)))), 1.0)
    jitter = 0.0
    limit = float(max_jitter_fraction) * scale
    while True:
        candidate = matrix + np.eye(len(vector)) * jitter
        try:
            chol = np.linalg.cholesky(candidate)
            solved = np.linalg.solve(chol, vector)
            quad = float(solved @ solved)
            value = 0.5 * (
                len(vector) * np.log(2.0 * np.pi)
                + 2.0 * float(np.log(np.diag(chol)).sum())
                + quad
            )
            return value, quad, jitter
        except np.linalg.LinAlgError:
            if limit <= 0.0:
                return None, np.nan, jitter
            jitter = max(
                np.finfo(float).eps * scale,
                jitter * 10.0 if jitter else np.finfo(float).eps * scale * 10.0,
            )
            if jitter > limit:
                return None, np.nan, jitter


def _aggregate_arrays(
    diagnostics: Mapping[str, np.ndarray],
    window_mask: np.ndarray,
    window: ValidationWindow,
    score_mask: np.ndarray,
) -> dict[str, Any]:
    eligible_mask = window_mask & score_mask
    primary_mask = eligible_mask & np.isfinite(diagnostics["primary_nlpd"])
    joint_mask = eligible_mask & np.isfinite(diagnostics["joint_nlpd"])

    def mean_metric(key: str) -> float:
        values = diagnostics[key][primary_mask]
        return float(np.mean(values)) if len(values) else np.nan

    joint_components = diagnostics["joint_components"][joint_mask]
    joint_count = int(np.sum(joint_components))
    return {
        "window": window.name,
        "start": window.start,
        "end": window.end,
        "storage_nlpd": float(np.mean(diagnostics["primary_nlpd"][primary_mask]))
        if np.any(primary_mask)
        else np.nan,
        "storage_count": int(np.sum(primary_mask)),
        "joint_nlpd": float(np.sum(diagnostics["joint_nlpd"][joint_mask]) / joint_count)
        if joint_count
        else np.nan,
        "joint_count": joint_count,
        "joint_nis": float(np.sum(diagnostics["joint_nis"][joint_mask]) / joint_count)
        if joint_count
        else np.nan,
        "storage_nis": mean_metric("storage_nis"),
        "outflow_nis": mean_metric("outflow_nis"),
        "conditional_storage_nis": mean_metric("conditional_storage_nis"),
        "storage_bias": mean_metric("storage_z"),
        "outflow_bias": mean_metric("outflow_z"),
        "conditional_storage_bias": mean_metric("conditional_storage_z"),
        "coverage": float(np.sum(primary_mask) / max(1, np.sum(eligible_mask))),
        "regularization_count": int(np.sum(diagnostics["jitter"][primary_mask] > 0)),
    }


def _physical_metrics_arrays(
    result: KalmanFilterResult,
    diagnostics: Mapping[str, np.ndarray],
    timestamps: pd.DatetimeIndex,
    validation_mask: np.ndarray,
    settings: BayesianEvaluationSettings,
) -> dict[str, float]:
    indices = np.flatnonzero(validation_mask)
    validation_timestamps = timestamps[indices]
    inflow = result.filtered_means[indices, 1] if len(indices) else np.array([])
    finite_inflow = np.isfinite(inflow)
    differences = np.diff(inflow)
    intervals = (
        np.diff(_timestamp_seconds(validation_timestamps))
        if len(validation_timestamps) > 1
        else np.array([], dtype=float)
    )
    normalized = differences / intervals if len(differences) else np.array([])
    finite_norm = normalized[np.isfinite(normalized)]

    def finite_mean(key: str) -> float:
        values = diagnostics[key][indices]
        values = values[np.isfinite(values)]
        return float(np.mean(values)) if len(values) else np.nan

    def autocorrelation_metrics(key: str) -> tuple[float, float]:
        if len(validation_timestamps) < 2:
            return np.nan, np.nan
        frame = _elapsed_lag_autocorrelation(
            validation_timestamps,
            diagnostics[key][indices],
            settings.innovation_max_lag,
        )
        finite_frame = frame[np.isfinite(frame["autocorrelation"])]
        if finite_frame.empty:
            return np.nan, np.nan
        row = finite_frame.iloc[
            int(np.argmax(np.abs(finite_frame["autocorrelation"].to_numpy())))
        ]
        return float(abs(row["autocorrelation"])), float(row["lag_seconds"])

    max_storage_ac, max_storage_lag = autocorrelation_metrics("storage_z")
    max_outflow_ac, max_outflow_lag = autocorrelation_metrics("outflow_z")
    max_conditional_ac, max_conditional_lag = autocorrelation_metrics(
        "conditional_storage_z"
    )
    scored_storage = diagnostics["storage_innovation"][indices]
    scored_storage = scored_storage[np.isfinite(scored_storage)]
    joint_values = diagnostics["joint_nis"][indices]
    joint_mask = np.isfinite(joint_values)
    joint_components = diagnostics["joint_components"][indices][joint_mask]
    joint_nis = (
        float(np.sum(joint_values[joint_mask]) / np.sum(joint_components))
        if np.any(joint_mask) and np.sum(joint_components)
        else np.nan
    )
    autocorrelations = (max_storage_ac, max_outflow_ac, max_conditional_ac)
    return {
        "negative_inflow_frequency": float(np.mean(inflow[finite_inflow] < 0.0))
        if np.any(finite_inflow)
        else np.nan,
        "inflow_change_median_per_second": float(np.median(finite_norm))
        if len(finite_norm)
        else np.nan,
        "inflow_change_q95_per_second": float(np.quantile(np.abs(finite_norm), 0.95))
        if len(finite_norm)
        else np.nan,
        "storage_bias": finite_mean("storage_z"),
        "outflow_bias": finite_mean("outflow_z"),
        "conditional_storage_bias": finite_mean("conditional_storage_z"),
        "joint_nis": joint_nis,
        "storage_nis": finite_mean("storage_nis"),
        "outflow_nis": finite_mean("outflow_nis"),
        "conditional_storage_nis": finite_mean("conditional_storage_nis"),
        "causal_storage_prediction_rmse": float(np.sqrt(np.mean(scored_storage**2)))
        if len(scored_storage)
        else np.nan,
        "causal_storage_prediction_bias": float(np.mean(scored_storage))
        if len(scored_storage)
        else np.nan,
        "max_storage_elapsed_lag_autocorrelation": max_storage_ac,
        "max_storage_elapsed_lag_seconds": max_storage_lag,
        "max_outflow_elapsed_lag_autocorrelation": max_outflow_ac,
        "max_outflow_elapsed_lag_seconds": max_outflow_lag,
        "max_conditional_storage_elapsed_lag_autocorrelation": max_conditional_ac,
        "max_conditional_storage_elapsed_lag_seconds": max_conditional_lag,
        "max_material_elapsed_lag_autocorrelation": max(
            value for value in autocorrelations if np.isfinite(value)
        )
        if any(np.isfinite(value) for value in autocorrelations)
        else np.nan,
    }


def _paired_standard_error(
    differences: Array,
    weights: Array,
    settings: BayesianEvaluationSettings,
) -> float:
    mean = float(np.dot(weights, differences))
    if len(differences) >= 5 and settings.bootstrap_samples > 1:
        rng = np.random.default_rng(settings.random_seed)
        samples = rng.choice(
            len(differences),
            size=(settings.bootstrap_samples, len(differences)),
            replace=True,
        )
        values = np.sum(differences[samples] * weights[samples], axis=1) / np.sum(
            weights[samples], axis=1
        )
        return float(np.std(values, ddof=1))
    sum_squared_weights = float(np.sum(weights * weights))
    denominator = 1.0 - sum_squared_weights
    if denominator <= 0.0:
        return 0.0
    unbiased_variance = float(np.dot(weights, (differences - mean) ** 2) / denominator)
    return sqrt(max(0.0, unbiased_variance * sum_squared_weights))


def _evaluate_candidate(
    prepared: _Prepared,
    *,
    q_inflow: float,
    windows: Sequence[ValidationWindow],
    settings: BayesianEvaluationSettings,
    r_override: Array | None = None,
    q_storage_override: float | None = None,
    q_outflow_override: float | None = None,
    plan: _DiagnosticPlan | None = None,
) -> _Pass:
    q_storage = (
        prepared.q_storage if q_storage_override is None else float(q_storage_override)
    )
    q_outflow = (
        prepared.q_outflow if q_outflow_override is None else float(q_outflow_override)
    )
    q_discrete = (
        q_storage * prepared.storage_basis
        + q_inflow * prepared.inflow_basis
        + q_outflow * prepared.outflow_basis
    )
    r = prepared.r if r_override is None else np.asarray(r_override, dtype=float)
    reasons: list[str] = []
    try:
        result = kalman_filter(
            prepared.observations,
            initial_mean=prepared.initial_mean,
            initial_covariance=prepared.initial_covariance,
            transition_matrix=prepared.transitions,
            process_covariance=q_discrete,
            observation_matrix=prepared.model.observation_matrix,
            observation_covariance=r,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        reasons.append(f"filter failed: {error}")
        return _Pass(
            q_inflow,
            None,
            {w.name: {"storage_nlpd": np.nan, "storage_count": 0} for w in windows},
            {},
            0,
            0.0,
            reasons,
        )
    if not (
        np.isfinite(result.filtered_means).all()
        and np.isfinite(result.predicted_means).all()
        and np.isfinite(result.predicted_covariances).all()
    ):
        reasons.append("nonfinite filtered or predicted state")
    if plan is None:
        plan = _diagnostic_plan(prepared, windows, settings)
    count = len(prepared.timestamps)
    diagnostics = {
        key: np.full(count, np.nan, dtype=float)
        for key in (
            "storage_innovation",
            "primary_nlpd",
            "joint_nlpd",
            "joint_nis",
            "storage_nis",
            "outflow_nis",
            "conditional_storage_nis",
            "storage_z",
            "outflow_z",
            "conditional_storage_z",
            "jitter",
        )
    }
    diagnostics["joint_components"] = prepared.finite_observations.sum(axis=1).astype(
        float
    )
    diagnostics["score"] = plan.score_mask.astype(float)
    regularization_count = 0
    max_jitter = 0.0
    for row, timestamp in enumerate(prepared.timestamps):
        finite = prepared.finite_observations[row]
        innovation = result.innovations[row]
        covariance = result.innovation_covariances[row]
        if finite[0] and np.isfinite(innovation[0]):
            diagnostics["storage_innovation"][row] = innovation[0]
        if finite.any():
            observed = np.flatnonzero(finite)
            vector = innovation[observed]
            matrix = covariance[np.ix_(observed, observed)]
            joint, quad, jitter = _stable_logpdf(
                vector, matrix, settings.max_jitter_fraction
            )
            if joint is None:
                reasons.append(
                    "non-positive-definite scored covariance at "
                    f"{timestamp.isoformat()}"
                )
            else:
                diagnostics["joint_nlpd"][row] = joint
                diagnostics["joint_nis"][row] = quad
                diagnostics["jitter"][row] = jitter
                regularization_count += int(jitter > 0.0)
                max_jitter = max(max_jitter, jitter)
            for component, key in ((0, "storage"), (1, "outflow")):
                if finite[component]:
                    variance = covariance[component, component]
                    if np.isfinite(variance) and variance > 0.0:
                        z = innovation[component] / sqrt(variance)
                        diagnostics[f"{key}_z"][row] = z
                        diagnostics[f"{key}_nis"][row] = z * z
            if finite[0]:
                if finite[1]:
                    try:
                        conditional = _storage_conditional_nlpd(
                            innovation[0],
                            innovation[1],
                            covariance,
                            max_jitter_fraction=settings.max_jitter_fraction,
                        )
                        soo = covariance[1, 1]
                        conditional_variance = (
                            covariance[0, 0] - covariance[0, 1] * covariance[1, 0] / soo
                        )
                        conditional_innovation = (
                            innovation[0] - covariance[0, 1] * innovation[1] / soo
                        )
                        diagnostics["conditional_storage_z"][row] = (
                            conditional_innovation
                            / sqrt(max(conditional_variance, np.finfo(float).tiny))
                        )
                    except ValueError as error:
                        reasons.append(str(error))
                        conditional = np.nan
                    diagnostics["conditional_storage_nis"][row] = (
                        diagnostics["conditional_storage_z"][row] ** 2
                    )
                    diagnostics["primary_nlpd"][row] = conditional
                else:
                    try:
                        diagnostics["primary_nlpd"][row] = _marginal_predictive_nlpd(
                            innovation[0],
                            covariance[0, 0],
                            max_jitter_fraction=settings.max_jitter_fraction,
                        )
                    except ValueError as error:
                        reasons.append(str(error))
    window_rows: dict[str, dict[str, Any]] = {}
    for window, window_mask in zip(windows, plan.window_masks, strict=True):
        window_rows[window.name] = _aggregate_arrays(
            diagnostics, window_mask, window, plan.score_mask
        )
    validation_mask = plan.score_mask & np.logical_or.reduce(plan.window_masks)
    physical = _physical_metrics_arrays(
        result, diagnostics, prepared.timestamps, validation_mask, settings
    )
    if regularization_count > settings.max_regularized_steps:
        reasons.append("excessive covariance regularization")
    return _Pass(
        q_inflow,
        result,
        window_rows,
        physical,
        regularization_count,
        max_jitter,
        list(dict.fromkeys(reasons)),
        diagnostics,
    )


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
        total = int(self.total_trials)
        initial = int(self.initial_trials)
        if total < 1:
            raise ValueError("total_trials must be positive")
        if initial < 1 or initial > total:
            raise ValueError("initial_trials must be between one and total_trials")
        object.__setattr__(self, "total_trials", total)
        object.__setattr__(self, "initial_trials", initial)
        object.__setattr__(self, "random_seed", int(self.random_seed))
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
        if int(self.acquisition_pool_size) < 128:
            raise ValueError("acquisition_pool_size must be at least 128")
        xi = float(self.expected_improvement_xi)
        if not np.isfinite(xi) or xi < 0.0:
            raise ValueError("expected_improvement_xi must be nonnegative")
        if int(self.proxy_min_aligned_points) < 3:
            raise ValueError("proxy_min_aligned_points must be at least 3")
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
        object.__setattr__(self, "one_standard_error_weight", weight)
        object.__setattr__(self, "max_abs_elapsed_lag_autocorrelation", autocorrelation)
        object.__setattr__(
            self, "acquisition_pool_size", int(self.acquisition_pool_size)
        )
        object.__setattr__(self, "expected_improvement_xi", xi)
        object.__setattr__(
            self, "proxy_min_aligned_points", int(self.proxy_min_aligned_points)
        )
        object.__setattr__(self, "proxy_max_lag", self.proxy_max_lag)
        object.__setattr__(self, "proxy_diagnostic_frequency", proxy_frequency)
        object.__setattr__(self, "proxy_max_shape_rmse", shape_rmse)
        object.__setattr__(self, "proxy_require_gate", bool(self.proxy_require_gate))


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


def _normal_pdf(value: Array) -> Array:
    return np.exp(-0.5 * value * value) / sqrt(2.0 * pi)


def _weighted_standard_error(values: Array, weights: Array) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0
    values = values[finite]
    weights = np.asarray(weights, dtype=float)[finite]
    weights = weights / weights.sum()
    mean = float(np.dot(weights, values))
    denominator = 1.0 - float(np.sum(weights * weights))
    if denominator <= 0.0:
        return 0.0
    variance = float(np.dot(weights, (values - mean) ** 2) / denominator)
    return sqrt(max(0.0, variance * float(np.sum(weights * weights))))


def _base_parameters(base: ReservoirConfig) -> np.ndarray:
    q = np.asarray(base.q, dtype=float)
    r = np.asarray(base.r, dtype=float)
    return np.asarray([q[0, 0], q[1, 1], q[2, 2], r[0, 0], r[1, 1]], dtype=float)


def _parameter_bounds(
    base: ReservoirConfig,
    inflow_increment_sd_seeds: Sequence[float],
    options: BayesianTuningSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    seeds = sorted(
        {_hourly_increment_sd_to_q(value) for value in inflow_increment_sd_seeds}
    )
    if len(seeds) < 3:
        raise ValueError("at least three positive finite candidates are required")
    base_values = _base_parameters(base)
    if not np.isfinite(base_values).all() or np.any(base_values <= 0.0):
        raise ValueError(
            "all tuned diagonal q and r values must be positive and finite"
        )
    scales = base_values.copy()
    multiplier_bounds = (
        options.q_storage_multiplier_bounds,
        (1.0, 1.0),
        options.q_outflow_multiplier_bounds,
        options.r_storage_multiplier_bounds,
        options.r_outflow_multiplier_bounds,
    )
    lows = np.asarray(
        [
            scale * bound[0]
            for scale, bound in zip(scales, multiplier_bounds, strict=True)
        ]
    )
    highs = np.asarray(
        [
            scale * bound[1]
            for scale, bound in zip(scales, multiplier_bounds, strict=True)
        ]
    )
    lows[1] = min(seeds)
    highs[1] = max(seeds)
    # Seed values define the exact q_inflow interval.
    return scales, lows, highs, seeds


def _initial_design(
    seeds: Sequence[float],
    initial_trials: int,
    lows: Array,
    highs: Array,
    base_values: Array,
    rng: np.random.Generator,
) -> list[Array]:
    # A base value need not lie inside a caller-supplied multiplier interval
    # (for example, all multipliers may intentionally be above 1.0).  Seed
    # points are still useful in that case, but must be projected into the
    # declared log-space box before they are evaluated.
    base_normalized = np.clip(
        (np.log(base_values) - np.log(lows)) / (np.log(highs) - np.log(lows)),
        0.0,
        1.0,
    )
    normalized_seed = [
        np.asarray(
            [
                base_normalized[0],
                (np.log(value) - np.log(lows[1]))
                / (np.log(highs[1]) - np.log(lows[1])),
                base_normalized[2],
                base_normalized[3],
                base_normalized[4],
            ],
            dtype=float,
        ).clip(0.0, 1.0)
        for value in seeds
    ]
    design = normalized_seed[:initial_trials]
    if len(design) >= initial_trials:
        return design
    # Greedy maximin selection makes the additional initial evaluations
    # reproducible and spreads them across the five-dimensional box.
    pool = rng.random((max(256, 32 * initial_trials), 5))
    while len(design) < initial_trials:
        distances = np.min(
            np.linalg.norm(pool[:, None, :] - np.asarray(design)[None, :, :], axis=2),
            axis=1,
        )
        selected = int(np.argmax(distances))
        design.append(pool[selected].copy())
        pool = np.delete(pool, selected, axis=0)
    return design


def _decode(normalized: Array, lows: Array, highs: Array) -> dict[str, float]:
    values = np.exp(
        np.log(lows) + np.asarray(normalized) * (np.log(highs) - np.log(lows))
    )
    return {
        name: float(value) for name, value in zip(_PARAMETER_NAMES, values, strict=True)
    }


def _encode(parameters: Mapping[str, float], lows: Array, highs: Array) -> Array:
    values = np.asarray([parameters[name] for name in _PARAMETER_NAMES], dtype=float)
    result = (np.log(values) - np.log(lows)) / (np.log(highs) - np.log(lows))
    return np.clip(result, 0.0, 1.0)


def _calibration_violation(
    physical: Mapping[str, float],
    settings: BayesianEvaluationSettings,
    options: BayesianTuningSettings,
) -> tuple[float, bool]:
    violation = 0.0
    calibrated = True
    nis_range = settings.nis_warning_range
    if nis_range is not None:
        for name in ("joint_nis", "storage_nis", "outflow_nis"):
            value = float(physical.get(name, np.nan))
            if not np.isfinite(value):
                calibrated = False
                violation += 1.0
            elif value < nis_range[0]:
                calibrated = False
                violation += ((nis_range[0] - value) / max(nis_range[0], 1e-12)) ** 2
            elif value > nis_range[1]:
                calibrated = False
                violation += ((value - nis_range[1]) / nis_range[1]) ** 2
    if settings.innovation_bias_warning is not None:
        limit = float(settings.innovation_bias_warning)
        for name in ("storage_bias", "outflow_bias", "conditional_storage_bias"):
            value = abs(float(physical.get(name, np.nan)))
            if not np.isfinite(value):
                calibrated = False
                violation += 1.0
            elif value > limit:
                calibrated = False
                violation += ((value - limit) / max(limit, 1e-12)) ** 2
    autocorrelation = abs(
        float(physical.get("max_material_elapsed_lag_autocorrelation", np.nan))
    )
    if not np.isfinite(autocorrelation):
        calibrated = False
        violation += 1.0
    elif autocorrelation > options.max_abs_elapsed_lag_autocorrelation:
        calibrated = False
        limit = max(options.max_abs_elapsed_lag_autocorrelation, 1e-12)
        violation += ((autocorrelation - limit) / limit) ** 2
    return float(violation), calibrated


def _correlation(left: Array, right: Array) -> float:
    """Return a finite Pearson correlation, or NaN for a flat/short pair."""

    if len(left) < 3 or len(right) != len(left):
        return np.nan
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        return np.nan
    if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def _robust_scale(values: Array) -> float:
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if not np.isfinite(mad) or mad <= np.finfo(float).eps:
        mad = float(np.std(values))
    return max(mad, np.finfo(float).eps)


def _proxy_metrics(
    result: Any,
    prepared: Any,
    upstream_proxy: pd.Series | None,
    options: BayesianTuningSettings,
    diagnostic_plan: Any,
) -> dict[str, float | bool]:
    """Compare causal filtered inflow with an upstream proxy by shape only.

    The proxy is aligned by timestamp and a bounded offline lag search is used
    solely for diagnostics.  No proxy value enters the Kalman objective, and no
    amplitude equality is assumed.  A positive lag means the proxy series is
    shifted later relative to the causal estimate.
    """

    metrics: dict[str, float | bool] = {
        "proxy_available": False,
        "proxy_aligned_count": 0,
        "proxy_best_lag_seconds": np.nan,
        "proxy_shape_correlation": np.nan,
        "proxy_change_correlation": np.nan,
        "proxy_shape_rmse": np.nan,
        # Absence of an optional proxy must not invalidate Bayesian tuning.
        "proxy_gate_passed": True,
    }
    if upstream_proxy is None:
        return metrics
    if result is None:
        metrics["proxy_available"] = True
        metrics["proxy_gate_passed"] = False
        return metrics
    if not isinstance(upstream_proxy, pd.Series):
        raise TypeError("upstream_proxy must be a pandas Series or None")
    if not isinstance(upstream_proxy.index, pd.DatetimeIndex):
        raise TypeError("upstream_proxy must use a pandas DatetimeIndex")
    if upstream_proxy.index.tz is None:
        raise ValueError("upstream_proxy index must be timezone-aware")
    proxy = upstream_proxy.copy()
    proxy.index = proxy.index.tz_convert("UTC")
    proxy = proxy[~proxy.index.duplicated(keep="last")].sort_index()
    aligned_proxy = pd.to_numeric(proxy.reindex(prepared.timestamps), errors="coerce")
    estimate = np.asarray(result.filtered_means[:, 1], dtype=float)
    proxy_values = aligned_proxy.to_numpy(dtype=float, copy=False)
    if len(estimate) != len(proxy_values):
        raise ValueError("upstream_proxy cannot be aligned to filter timestamps")
    metrics["proxy_available"] = True
    validation_mask = diagnostic_plan.score_mask & np.logical_or.reduce(
        diagnostic_plan.window_masks
    )
    proxy_frame = pd.DataFrame(
        {"estimate": estimate, "proxy": proxy_values}, index=prepared.timestamps
    ).loc[validation_mask]
    # This is an offline shape/timing diagnostic. Hourly aggregation is enough
    # for event response while avoiding a high-frequency lag scan per trial.
    proxy_frame = proxy_frame.resample(options.proxy_diagnostic_frequency).mean()
    estimate = proxy_frame["estimate"].to_numpy(dtype=float, copy=False)
    proxy_values = proxy_frame["proxy"].to_numpy(dtype=float, copy=False)
    finite = np.isfinite(estimate) & np.isfinite(proxy_values)
    if int(finite.sum()) < options.proxy_min_aligned_points:
        metrics["proxy_aligned_count"] = int(finite.sum())
        metrics["proxy_gate_passed"] = False
        return metrics
    interval_seconds = options.proxy_diagnostic_frequency.total_seconds()
    max_steps = int(np.ceil(options.proxy_max_lag.total_seconds() / interval_seconds))
    best: tuple[float, int, float, float, float] | None = None
    for offset in range(-max_steps, max_steps + 1):
        if offset >= 0:
            end = len(estimate) - offset
            if end <= 0:
                continue
            estimate_slice = estimate[:end]
            proxy_slice = proxy_values[offset : offset + end]
        else:
            start = -offset
            end = len(estimate) + offset
            if end <= 0:
                continue
            estimate_slice = estimate[start : start + end]
            proxy_slice = proxy_values[:end]
        if len(estimate_slice) != len(proxy_slice):
            continue
        pair_finite = np.isfinite(estimate_slice) & np.isfinite(proxy_slice)
        if int(pair_finite.sum()) < options.proxy_min_aligned_points:
            continue
        estimate_pair = estimate_slice[pair_finite]
        proxy_pair = proxy_slice[pair_finite]
        shape_corr = _correlation(estimate_pair, proxy_pair)
        change_corr = _correlation(np.diff(estimate_pair), np.diff(proxy_pair))
        if not np.isfinite(shape_corr):
            continue
        if not np.isfinite(change_corr):
            change_corr = 0.0
        estimate_scale = _robust_scale(estimate_pair)
        proxy_scale = _robust_scale(proxy_pair)
        estimate_z = (estimate_pair - np.median(estimate_pair)) / estimate_scale
        proxy_z = (proxy_pair - np.median(proxy_pair)) / proxy_scale
        shape_rmse = float(np.sqrt(np.mean((estimate_z - proxy_z) ** 2)))
        score = 0.6 * shape_corr + 0.4 * change_corr
        candidate = (
            score,
            offset,
            shape_corr,
            change_corr,
            shape_rmse,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        metrics["proxy_aligned_count"] = int(finite.sum())
        metrics["proxy_gate_passed"] = False
        return metrics
    _score, offset, shape_corr, change_corr, shape_rmse = best
    metrics.update(
        {
            "proxy_aligned_count": int(finite.sum()),
            "proxy_best_lag_seconds": float(offset * interval_seconds),
            "proxy_shape_correlation": float(shape_corr),
            "proxy_change_correlation": float(change_corr),
            "proxy_shape_rmse": float(shape_rmse),
            "proxy_gate_passed": bool(
                finite.sum() >= options.proxy_min_aligned_points
                and shape_corr >= options.proxy_min_shape_correlation
                and change_corr >= options.proxy_min_change_correlation
                and shape_rmse <= options.proxy_max_shape_rmse
            ),
        }
    )
    return metrics


def _fit_and_acquire(
    x_values: list[Array],
    y_values: list[float],
    excluded_x_values: list[Array],
    rng: np.random.Generator,
    options: BayesianTuningSettings,
) -> tuple[Array, str]:
    if not x_values or not y_values:
        pool = rng.random((options.acquisition_pool_size, 5))
        if excluded_x_values:
            excluded = np.asarray(excluded_x_values, dtype=float)
            distances = np.min(
                np.linalg.norm(pool[:, None, :] - excluded[None, :, :], axis=2),
                axis=1,
            )
            return pool[int(np.argmax(distances))], "space-filling-fallback"
        return pool[0], "space-filling-fallback"
    global_count = max(1, options.acquisition_pool_size * 3 // 4)
    pool = rng.random((global_count, 5))
    existing = np.asarray(x_values, dtype=float)
    excluded = np.asarray(excluded_x_values, dtype=float)
    best_point = existing[int(np.argmin(np.asarray(y_values, dtype=float)))]
    local = np.clip(
        best_point
        + rng.normal(
            0.0,
            0.1,
            size=(options.acquisition_pool_size - global_count, 5),
        ),
        0.0,
        1.0,
    )
    pool = np.vstack((pool, local))
    distance = np.min(
        np.linalg.norm(pool[:, None, :] - excluded[None, :, :], axis=2), axis=1
    )
    # Avoid resampling both valid and invalid evaluated neighborhoods.
    pool = pool[distance > 0.02]
    if not len(pool):
        return rng.random(5), "random-fallback"
    try:
        model = GaussianProcessRegressor(
            kernel=Matern(length_scale=np.ones(5), nu=2.5)
            + WhiteKernel(noise_level=1e-6, noise_level_bounds="fixed"),
            normalize_y=True,
            random_state=options.random_seed,
            n_restarts_optimizer=0,
        )
        y = np.asarray(y_values, dtype=float)
        model.fit(existing, y)
        mean, std = model.predict(pool, return_std=True)
        best = float(np.min(y))
        improvement = best - mean - options.expected_improvement_xi
        z = np.divide(improvement, std, out=np.zeros_like(improvement), where=std > 0)
        expected = improvement * ndtr(z) + std * _normal_pdf(z)
        expected[std <= 0.0] = 0.0
        return pool[int(np.argmax(expected))], "expected-improvement"
    except ValueError, RuntimeError, np.linalg.LinAlgError:
        # A numerical GP failure should not prevent a bounded deterministic
        # search from completing.
        distances = np.min(
            np.linalg.norm(pool[:, None, :] - existing[None, :, :], axis=2), axis=1
        )
        return pool[int(np.argmax(distances))], "space-filling-fallback"


def _proposed_config(
    base: ReservoirConfig,
    selected: Mapping[str, float],
    proposed_configuration_version: str,
    selected_trial: int,
    competitive_trials: Sequence[int],
    selected_objective: float,
    selected_calibration_violation: float,
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
            "calibration_method": "causal-innovation-bayesian-optimization",
            "tuned_parameters": list(_PARAMETER_NAMES),
            "selected_parameters": dict(selected),
            "selection": {
                "selected_trial_id": selected_trial,
                "selected_objective": float(selected_objective),
                "selected_calibration_violation": float(selected_calibration_violation),
                "competitive_trial_ids": [int(t) for t in competitive_trials],
            },
        }
    )
    return ReservoirConfig(
        reservoir_id=base.reservoir_id,
        reservoir_name=base.reservoir_name,
        q=q,
        r=r,
        p0=np.asarray(base.p0).copy(),
        smoothing_lag=base.smoothing_lag,
        initialization_strategy=base.initialization_strategy,
        inflow_units=base.inflow_units,
        model_version=base.model_version,
        configuration_version=proposed_configuration_version,
        metadata=metadata,
        unit_system=base.unit_system,
    )


def tune_inflow_noise_bayesian(
    storage: pd.Series,
    discharge: pd.Series,
    base_config: ReservoirConfig,
    inflow_increment_sd_seeds: Sequence[float],
    validation_windows: Sequence[ValidationWindow],
    *,
    settings: BayesianEvaluationSettings | None = None,
    bayesian_settings: BayesianTuningSettings | None = None,
    upstream_proxy: pd.Series | None = None,
    proposed_configuration_version: str | None = None,
) -> BayesianTuningResult:
    """Tune diagonal ``Q`` and ``R`` using causal innovation NLPD."""

    options = settings or BayesianEvaluationSettings()
    search = bayesian_settings or BayesianTuningSettings()
    if not isinstance(base_config, ReservoirConfig):
        raise TypeError("base_config must be a ReservoirConfig")
    if (
        not proposed_configuration_version
        or not str(proposed_configuration_version).strip()
    ):
        raise ValueError("proposed_configuration_version is required")
    if str(proposed_configuration_version) == base_config.configuration_version:
        raise ValueError("proposed_configuration_version must be new")
    windows = _validate_windows(validation_windows)
    if len(windows) < 3:
        raise BayesianTuningError("at least three validation windows are required")
    prepared = _prepare(storage, discharge, base_config)
    if upstream_proxy is not None and not isinstance(upstream_proxy, pd.Series):
        raise TypeError("upstream_proxy must be a pandas Series or None")
    plan = _diagnostic_plan(prepared, windows, options)
    weights = _window_weights(windows)
    base_values, lows, highs, seeds = _parameter_bounds(
        base_config, inflow_increment_sd_seeds, search
    )
    effective_initial_trials = max(search.initial_trials, len(seeds))
    if effective_initial_trials > search.total_trials:
        raise ValueError(
            "total_trials must be at least the number of supplied inflow seeds"
        )
    base_point = _encode(
        dict(zip(_PARAMETER_NAMES, base_values, strict=True)), lows, highs
    )
    rng = np.random.default_rng(search.random_seed)
    design = _initial_design(
        seeds, effective_initial_trials, lows, highs, base_values, rng
    )
    # Design coordinates are stored in log space, which is the natural scale
    # for covariance parameters and the scale seen by the GP.
    observed_x_values: list[Array] = []
    gp_x_values: list[Array] = []
    objective_values: list[float] = []
    trial_parameters: dict[int, dict[str, float]] = {}
    passes: dict[int, Any] = {}
    proxy_trial_metrics: dict[int, dict[str, float | bool]] = {}
    records: list[dict[str, Any]] = []
    timings: dict[str, float] = {
        # _evaluate_candidate performs both the Kalman pass and all per-row
        # likelihood/diagnostic calculations, so this deliberately does not
        # call the result "filtering_seconds" or "scoring_seconds".
        "candidate_evaluation_seconds": 0.0,
        "acquisition_seconds": 0.0,
    }
    total_started = perf_counter()

    for trial_id in range(search.total_trials):
        acquisition_source = "initial-design"
        if trial_id < len(design):
            point = design[trial_id]
        else:
            acquisition_started = perf_counter()
            point, acquisition_source = _fit_and_acquire(
                gp_x_values,
                objective_values,
                observed_x_values,
                rng,
                search,
            )
            timings["acquisition_seconds"] += perf_counter() - acquisition_started
        # Avoid exact duplicates after finite-precision transforms.
        if observed_x_values:
            point = np.asarray(point, dtype=float)
            if (
                np.min(np.linalg.norm(np.asarray(observed_x_values) - point, axis=1))
                <= 1e-8
            ):
                point = rng.random(5)
                acquisition_source = "random-fallback"
        parameters = _decode(point, lows, highs)
        trial_parameters[trial_id] = parameters
        started = perf_counter()
        candidate = _evaluate_candidate(
            prepared,
            q_inflow=parameters["q_inflow"],
            q_storage_override=parameters["q_storage"],
            q_outflow_override=parameters["q_outflow"],
            r_override=np.diag([parameters["r_storage"], parameters["r_outflow"]]),
            windows=windows,
            settings=options,
            plan=plan,
        )
        timings["candidate_evaluation_seconds"] += perf_counter() - started
        passes[trial_id] = candidate
        violation, calibrated = _calibration_violation(
            candidate.physical, options, search
        )
        proxy_metrics = _proxy_metrics(
            candidate.filter_result, prepared, upstream_proxy, search, plan
        )
        proxy_trial_metrics[trial_id] = proxy_metrics
        if (
            search.proxy_require_gate
            and proxy_metrics["proxy_available"]
            and not proxy_metrics["proxy_gate_passed"]
        ):
            violation += 1.0
        losses = np.asarray(
            [candidate.window_rows[w.name]["joint_nlpd"] for w in windows], dtype=float
        )
        valid = (
            not candidate.reasons
            and np.isfinite(losses).all()
            and all(
                candidate.window_rows[w.name]["storage_count"]
                >= options.min_scored_storage_observations
                for w in windows
            )
        )
        mean_loss = float(np.dot(weights, losses)) if valid else np.inf
        standard_error = _weighted_standard_error(losses, weights) if valid else np.inf
        robust_objective = (
            mean_loss + search.one_standard_error_weight * standard_error
            if valid
            else np.inf
        )
        rejection = "; ".join(candidate.reasons)
        if not valid and not rejection:
            rejection = "failed hard validity gates"
        row = {
            "trial_id": trial_id,
            **parameters,
            "inflow_increment_sd": sqrt(parameters["q_inflow"] * 3600.0),
            "acquisition_source": acquisition_source,
            "objective": mean_loss,
            "window_standard_error": standard_error,
            "robust_objective": robust_objective,
            "calibration_violation": violation,
            "calibrated": calibrated,
            "eligible": valid,
            "rejection_reasons": rejection,
            "max_elapsed_lag_autocorrelation": candidate.physical.get(
                "max_material_elapsed_lag_autocorrelation", np.nan
            ),
            "joint_nis": candidate.physical.get("joint_nis", np.nan),
            "storage_nis": candidate.physical.get("storage_nis", np.nan),
            "outflow_nis": candidate.physical.get("outflow_nis", np.nan),
            # Keep only the calibration metrics needed to interpret a trial.
            # Per-row and physical plausibility details are intentionally not
            # copied into the report; they are available from a separate
            # evaluation when needed.
        }
        records.append(row)
        observed_x_values.append(np.asarray(point, dtype=float))
        if valid:
            gp_x_values.append(np.asarray(point, dtype=float))
            objective_values.append(robust_objective)
        # Window aggregates and physical metrics above are sufficient for
        # selection. Release the full filter history after each trial.
        candidate.filter_result = None
        candidate.diagnostics.clear()

    candidate_frame = pd.DataFrame(records)
    valid_frame = candidate_frame[candidate_frame["eligible"]].copy()
    if valid_frame.empty:
        raise BayesianTuningError("no candidate passed the hard validity gates")
    proxy_passed = pd.Series(
        {
            trial_id: bool(metrics["proxy_gate_passed"])
            for trial_id, metrics in proxy_trial_metrics.items()
        }
    )
    proxy_gate_restricted = bool(
        search.proxy_require_gate
        and upstream_proxy is not None
        and valid_frame.trial_id.map(proxy_passed).fillna(False).any()
    )
    selection_frame = (
        valid_frame[valid_frame.trial_id.map(proxy_passed).fillna(False)].copy()
        if proxy_gate_restricted
        else valid_frame
    )
    best_trial = int(
        selection_frame.sort_values(["robust_objective", "trial_id"]).iloc[0].trial_id
    )
    best_losses = np.asarray(
        [passes[best_trial].window_rows[w.name]["joint_nlpd"] for w in windows],
        dtype=float,
    )
    thresholds: dict[int, float] = {}
    excess: dict[int, float] = {}
    for trial_id in selection_frame.trial_id.astype(int):
        losses = np.asarray(
            [passes[trial_id].window_rows[w.name]["joint_nlpd"] for w in windows],
            dtype=float,
        )
        differences = losses - best_losses
        excess[trial_id] = float(np.dot(weights, differences))
        thresholds[trial_id] = max(
            _paired_standard_error(differences, weights, options),
            options.practical_equivalence_tolerance,
        )
    competitive = tuple(
        trial_id
        for trial_id in selection_frame.trial_id.astype(int)
        if excess[trial_id] <= thresholds[trial_id] + 1e-12
    )
    if not competitive:
        competitive = (best_trial,)
    selected_trial = min(
        competitive,
        key=lambda trial_id: (
            float(
                candidate_frame.loc[
                    candidate_frame.trial_id == trial_id, "calibration_violation"
                ].iloc[0]
            ),
            float(
                np.linalg.norm(
                    _encode(trial_parameters[trial_id], lows, highs) - base_point
                )
            ),
            float(
                candidate_frame.loc[
                    candidate_frame.trial_id == trial_id, "robust_objective"
                ].iloc[0]
            ),
            trial_id,
        ),
    )
    candidate_frame["paired_excess_loss"] = candidate_frame.trial_id.map(excess)
    candidate_frame["selection_threshold"] = candidate_frame.trial_id.map(thresholds)
    candidate_frame["competitive"] = candidate_frame.trial_id.isin(competitive)
    candidate_frame["selected"] = candidate_frame.trial_id == selected_trial
    candidate_frame["selected_reason"] = np.where(
        candidate_frame["selected"],
        "lowest calibration violation in competitive set",
        "",
    )
    candidate_frame = candidate_frame.sort_values(
        ["selected", "competitive", "robust_objective", "trial_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = trial_parameters[selected_trial]
    selected_objective = float(
        candidate_frame.loc[
            candidate_frame.trial_id == selected_trial, "robust_objective"
        ].iloc[0]
    )
    selected_calibration_violation = float(
        candidate_frame.loc[
            candidate_frame.trial_id == selected_trial, "calibration_violation"
        ].iloc[0]
    )
    selection_reason = (
        "lowest calibration violation among paired one-standard-error "
        "competitive trials"
    )
    selected_config = _proposed_config(
        base_config,
        selected,
        str(proposed_configuration_version),
        selected_trial=selected_trial,
        competitive_trials=competitive,
        selected_objective=selected_objective,
        selected_calibration_violation=selected_calibration_violation,
    )
    window_columns = (
        "window",
        "start",
        "end",
        "storage_nlpd",
        "joint_nlpd",
        "joint_nis",
        "storage_nis",
        "outflow_nis",
        "coverage",
        "storage_count",
        "joint_count",
    )
    window_frame = pd.DataFrame(
        [
            {
                "trial_id": selected_trial,
                **{
                    name: passes[selected_trial].window_rows[window.name].get(name)
                    for name in window_columns
                },
            }
            for window in windows
        ]
    )
    if upstream_proxy is None:
        proxy_frame = pd.DataFrame()
    else:
        proxy_columns = (
            "proxy_aligned_count",
            "proxy_best_lag_seconds",
            "proxy_shape_correlation",
            "proxy_change_correlation",
            "proxy_shape_rmse",
            "proxy_gate_passed",
        )
        proxy_frame = pd.DataFrame(
            [
                {
                    "trial_id": trial_id,
                    **{name: metrics[name] for name in proxy_columns},
                    "competitive": trial_id in competitive,
                    "selected": trial_id == selected_trial,
                }
                for trial_id, metrics in proxy_trial_metrics.items()
            ]
        )
    warnings: list[str] = []
    if (
        upstream_proxy is not None
        and search.proxy_require_gate
        and not proxy_gate_restricted
    ):
        warnings.append(
            "no valid Bayesian trial passed the upstream-proxy shape/timing gate; "
            "selection was made from innovation-valid trials and is diagnostic only"
        )
    elif proxy_gate_restricted:
        warnings.append(
            "upstream-proxy shape/timing gate restricted selection; proxy was not "
            "treated as a total-inflow target"
        )
    if any(
        abs(selected[name] - bound) <= 1e-12
        for name, bound in zip(_PARAMETER_NAMES, lows, strict=True)
    ) or any(
        abs(selected[name] - bound) <= 1e-12
        for name, bound in zip(_PARAMETER_NAMES, highs, strict=True)
    ):
        warnings.append(
            "selected parameter is at the edge of the Bayesian search bounds"
        )
    if len(windows) < 5:
        warnings.append(
            "fewer than five valid windows are available; uncertainty is weak"
        )
    if not bool(
        candidate_frame.loc[
            candidate_frame.trial_id == selected_trial, "calibrated"
        ].iloc[0]
    ):
        warnings.append(
            "selected trial does not satisfy all innovation calibration checks"
        )
    timings["total_seconds"] = perf_counter() - total_started
    return BayesianTuningResult(
        selected_config=selected_config,
        selected_parameters=selected,
        candidate_summary=candidate_frame,
        window_diagnostics=window_frame,
        competitive_trial_ids=tuple(int(t) for t in competitive),
        selected_objective=selected_objective,
        selection_threshold=float(thresholds.get(selected_trial, 0.0)),
        selection_reason=selection_reason,
        proxy_diagnostics=proxy_frame,
        warnings=tuple(dict.fromkeys(warnings)),
        timing_seconds=timings,
    )


def evaluate_configuration(
    storage: pd.Series,
    discharge: pd.Series,
    config: ReservoirConfig,
    *,
    evaluation_window: ValidationWindow | None = None,
    settings: BayesianEvaluationSettings | None = None,
) -> ConfigEvaluationResult:
    """Evaluate one frozen configuration on an independent validation period.

    This is an assessment operation only: it does not search or modify the
    supplied configuration and is suitable for held-out post-search checks.
    """

    options = settings or BayesianEvaluationSettings()
    if not isinstance(config, ReservoirConfig):
        raise TypeError("config must be a ReservoirConfig")
    prepared = _prepare(storage, discharge, config)
    if evaluation_window is None:
        evaluation_window = ValidationWindow(
            "evaluation",
            prepared.index[0],
            prepared.index[-1] + pd.Timedelta(nanoseconds=1),
        )
    windows = _validate_windows((evaluation_window,))
    q_inflow = float(config.q[1, 1])
    candidate = _evaluate_candidate(
        prepared,
        q_inflow=q_inflow,
        windows=windows,
        settings=options,
        plan=_diagnostic_plan(prepared, windows, options),
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
                "eligible": not candidate.reasons,
                "rejection_reasons": "; ".join(candidate.reasons),
                **candidate.physical,
            }
        ]
    )
    return ConfigEvaluationResult(
        config=config,
        candidate_summary=summary,
        window_diagnostics=window_frame,
        warnings=tuple(candidate.reasons),
    )


__all__ = [
    "BayesianEvaluationSettings",
    "BayesianTuningError",
    "BayesianTuningResult",
    "BayesianTuningSettings",
    "ConfigEvaluationResult",
    "ValidationWindow",
    "evaluate_configuration",
    "tune_inflow_noise_bayesian",
]
