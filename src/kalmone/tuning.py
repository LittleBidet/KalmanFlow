"""Conservative, causal calibration of reservoir inflow process noise.

The public tuner in this module deliberately has a small scope: it compares a
fixed grid of inflow random-walk process noises while keeping the supplied
model, measurement noise, and initial covariance unchanged.  Candidate scores
are calculated from the pre-update predictive distribution produced by one
continuous forward filter pass.  The selected configuration is a proposed
immutable value; this module never writes configuration files or changes an
operational stream.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from hashlib import blake2b
from math import log, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .kalman import KalmanFilterResult, kalman_filter, predict_state
from .models import ReservoirStateSpaceModel
from .reservoir_config import ReservoirConfig

Array = np.ndarray


class TuningError(ValueError):
    """Raised when a causal tuning result cannot be produced."""


@dataclass(frozen=True)
class TuningWindow:
    """A non-overlapping half-open validation interval.

    ``start`` is inclusive and ``end`` is exclusive.  A timezone-aware
    timestamp is required even when the input index uses a different timezone;
    comparisons are performed by pandas on absolute instants.
    """

    name: str
    start: datetime | pd.Timestamp
    end: datetime | pd.Timestamp
    regime: str = "unspecified"
    weight: float | None = None

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("window name must not be empty")
        start = _timestamp(self.start, "window start")
        end = _timestamp(self.end, "window end")
        if end <= start:
            raise ValueError("window end must be after window start")
        if not str(self.regime).strip():
            raise ValueError("window regime must not be empty")
        if self.weight is not None:
            weight = float(self.weight)
            if not np.isfinite(weight) or weight <= 0.0:
                raise ValueError("window weight must be positive and finite")
            object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "regime", str(self.regime))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    def mask(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Return this window's half-open mask over ``index``."""

        return (index >= self.start) & (index < self.end)


@dataclass(frozen=True)
class InflowTuningSettings:
    """Fixed choices controlling candidate scoring and diagnostics."""

    warmup: timedelta = timedelta(hours=24)
    innovation_max_lag: timedelta = timedelta(hours=24)
    forecast_horizons: tuple[timedelta, ...] = (
        timedelta(hours=1),
        timedelta(hours=3),
        timedelta(hours=6),
    )
    horizon_tolerance: timedelta | None = None
    min_scored_storage_observations: int = 3
    practical_equivalence_tolerance: float = 0.0
    bootstrap_samples: int = 1000
    random_seed: int = 0
    max_jitter_fraction: float = 1e-9
    max_regularized_steps: int = 0
    r_sensitivity_multipliers: tuple[float, ...] = ()
    nis_warning_range: tuple[float, float] | None = (0.7, 1.5)
    innovation_bias_warning: float | None = 0.25

    def __post_init__(self) -> None:
        for name in ("warmup", "innovation_max_lag"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value.total_seconds() < 0.0:
                raise ValueError(f"{name} must be a nonnegative timedelta")
        horizons = tuple(self.forecast_horizons)
        if any(
            not isinstance(value, timedelta) or value.total_seconds() <= 0.0
            for value in horizons
        ):
            raise ValueError("forecast_horizons must contain positive timedeltas")
        if tuple(sorted(horizons)) != horizons or len(set(horizons)) != len(horizons):
            raise ValueError("forecast_horizons must be sorted and unique")
        tolerance = self.horizon_tolerance
        if tolerance is not None and (
            not isinstance(tolerance, timedelta) or tolerance.total_seconds() < 0.0
        ):
            raise ValueError("horizon_tolerance must be a nonnegative timedelta")
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
        multipliers = tuple(float(value) for value in self.r_sensitivity_multipliers)
        if any(not np.isfinite(value) or value <= 0.0 for value in multipliers):
            raise ValueError("r_sensitivity_multipliers must be positive and finite")
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
        object.__setattr__(self, "forecast_horizons", horizons)
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
        object.__setattr__(self, "r_sensitivity_multipliers", multipliers)


@dataclass(frozen=True)
class InflowTuningResult:
    """Auditable proposed configuration and candidate diagnostics."""

    selected_config: ReservoirConfig
    selected_prior_hourly_increment_sd: float
    selected_q_inflow: float
    candidate_summary: pd.DataFrame
    window_diagnostics: pd.DataFrame
    regime_diagnostics: pd.DataFrame
    horizon_diagnostics: pd.DataFrame
    r_sensitivity: pd.DataFrame | None
    competitive_candidates: tuple[float, ...]
    selection_threshold: float
    selection_reason: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_summary",
            "window_diagnostics",
            "regime_diagnostics",
            "horizon_diagnostics",
        ):
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise TypeError(f"{name} must be a pandas DataFrame")
            object.__setattr__(self, name, frame.copy(deep=True))
        if self.r_sensitivity is not None:
            if not isinstance(self.r_sensitivity, pd.DataFrame):
                raise TypeError("r_sensitivity must be a pandas DataFrame")
            object.__setattr__(
                self, "r_sensitivity", self.r_sensitivity.copy(deep=True)
            )
        object.__setattr__(
            self,
            "competitive_candidates",
            tuple(float(v) for v in self.competitive_candidates),
        )
        object.__setattr__(self, "warnings", tuple(str(v) for v in self.warnings))


@dataclass(frozen=True)
class InflowConfigEvaluationResult:
    """Causal diagnostics for one already-frozen configuration."""

    config: ReservoirConfig
    candidate_summary: pd.DataFrame
    window_diagnostics: pd.DataFrame
    regime_diagnostics: pd.DataFrame
    horizon_diagnostics: pd.DataFrame
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "candidate_summary",
            "window_diagnostics",
            "regime_diagnostics",
            "horizon_diagnostics",
        ):
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


@dataclass
class _Pass:
    q_inflow: float
    prior_sd: float
    filter_result: KalmanFilterResult | None
    rows: list[dict[str, Any]]
    window_rows: dict[str, dict[str, Any]]
    physical: dict[str, float]
    regularization_count: int
    max_jitter: float
    reasons: list[str] = field(default_factory=list)


def prior_hourly_increment_to_q(prior_hourly_increment_sd: float) -> float:
    """Convert one-hour random-walk increment SD to continuous ``q_inflow``."""

    value = float(prior_hourly_increment_sd)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("prior_hourly_increment_sd must be positive and finite")
    return value * value / 3600.0


def joint_predictive_nlpd(
    innovation: Array, covariance: Array, *, max_jitter_fraction: float = 0.0
) -> float:
    """Return Gaussian negative log predictive density using a stable Cholesky."""

    value, _, _ = _stable_logpdf(innovation, covariance, max_jitter_fraction)
    if value is None:
        raise ValueError("innovation covariance is not positive definite")
    return float(value)


def marginal_predictive_nlpd(
    innovation: float, variance: float, *, max_jitter_fraction: float = 0.0
) -> float:
    """Return a scalar Gaussian predictive negative log density."""

    value, _, _ = _stable_logpdf(
        np.array([innovation]), np.array([[variance]]), max_jitter_fraction
    )
    if value is None:
        raise ValueError("predictive variance is not positive")
    return float(value)


def storage_conditional_nlpd(
    storage_innovation: float,
    outflow_innovation: float,
    innovation_covariance: Array,
    *,
    max_jitter_fraction: float = 0.0,
) -> float:
    """Score storage conditional on a simultaneous outflow observation."""

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
    return marginal_predictive_nlpd(
        conditional_innovation,
        conditional_variance,
        max_jitter_fraction=max_jitter_fraction,
    )


def normalized_innovation_squared(innovation: Array, covariance: Array) -> float:
    """Return NIS without forming a matrix inverse."""

    vector = np.asarray(innovation, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (len(vector), len(vector)):
        raise ValueError("innovation and covariance dimensions do not match")
    try:
        solution = np.linalg.solve((matrix + matrix.T) / 2.0, vector)
    except np.linalg.LinAlgError as error:
        raise ValueError("innovation covariance is singular") from error
    return float(vector @ solution)


def elapsed_lag_autocorrelation(
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
    seconds = np.asarray([timestamp.value / 1e9 for timestamp in index], dtype=float)
    cadence = float(np.median(np.diff(seconds))) if len(index) > 1 else limit
    tolerance = float(
        (lag_tolerance or timedelta(seconds=max(cadence * 0.25, 1.0))).total_seconds()
    )
    targets = np.arange(cadence, limit + cadence * 0.5, cadence)
    rows: list[dict[str, float | int]] = []
    finite_values = series[finite]
    centre = float(np.mean(finite_values))
    variance = float(np.sum((finite_values - centre) ** 2))
    if variance <= 0.0:
        variance = np.nan
    for target in targets:
        pair_count = 0
        covariance = 0.0
        for i, current in enumerate(seconds):
            if not finite[i]:
                continue
            lower = int(np.searchsorted(seconds, current + target - tolerance))
            upper = int(
                np.searchsorted(seconds, current + target + tolerance, side="right")
            )
            matching = np.flatnonzero(finite[lower:upper]) + lower
            matching = matching[matching > i]
            if len(matching):
                pair_count += len(matching)
                covariance += float(
                    np.sum((series[i] - centre) * (series[matching] - centre))
                )
        if pair_count and np.isfinite(variance):
            correlation = covariance / variance if variance > 0 else np.nan
        else:
            correlation = np.nan
        rows.append(
            {
                "lag_seconds": float(target),
                "autocorrelation": float(correlation),
                "pair_count": pair_count,
            }
        )
    return pd.DataFrame(rows)


def tune_inflow_process_noise(
    storage: pd.Series,
    discharge: pd.Series,
    base_config: ReservoirConfig,
    candidate_prior_hourly_increment_sd: Sequence[float],
    validation_windows: Sequence[TuningWindow],
    *,
    settings: InflowTuningSettings | None = None,
    proposed_configuration_version: str | None = None,
) -> InflowTuningResult:
    """Propose a conservative ``q_inflow`` from causal validation windows."""

    options = settings or InflowTuningSettings()
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
        raise TuningError("at least three validation windows are required")
    prepared = _prepare(storage, discharge, base_config)
    prior_values, q_values = _candidate_values(candidate_prior_hourly_increment_sd)
    weights = _window_weights(windows)
    passes = {
        q: _evaluate_candidate(
            prepared,
            prior_sd=prior,
            q_inflow=q,
            windows=windows,
            settings=options,
            window_weights=weights,
        )
        for prior, q in zip(prior_values, q_values, strict=True)
    }
    candidate_frame, window_frame, _, eligible_q, warnings = _summary_frames(
        passes, windows, weights, options
    )
    if len(eligible_q) < 1:
        raise TuningError("no candidate passed the hard validity gates")
    best_q = min(
        eligible_q,
        key=lambda value: (
            float(
                candidate_frame.loc[
                    candidate_frame.q_inflow == value, "weighted_storage_nlpd"
                ].iloc[0]
            ),
            value,
        ),
    )
    best_pass = passes[best_q]
    excess: dict[float, float] = {}
    standard_errors: dict[float, float] = {}
    thresholds: dict[float, float] = {}
    for q in eligible_q:
        differences = np.asarray(
            [
                passes[q].window_rows[w.name]["storage_nlpd"]
                - best_pass.window_rows[w.name]["storage_nlpd"]
                for w in windows
            ],
            dtype=float,
        )
        mean_difference = float(np.dot(weights, differences))
        se = _paired_standard_error(differences, weights, options)
        excess[q] = mean_difference
        standard_errors[q] = se
        thresholds[q] = max(se, options.practical_equivalence_tolerance)
    competitive_q = tuple(
        sorted(q for q in eligible_q if excess[q] <= thresholds[q] + 1e-12)
    )
    if not competitive_q:
        competitive_q = (best_q,)
    selected_q = min(competitive_q)
    selected_prior = float(prior_values[q_values.index(selected_q)])
    selected_pass = passes[selected_q]
    selection_threshold = float(thresholds[selected_q])
    candidate_frame["paired_standard_error"] = candidate_frame["q_inflow"].map(
        standard_errors
    )
    candidate_frame["paired_excess_loss"] = candidate_frame["q_inflow"].map(excess)
    candidate_frame["selection_threshold"] = candidate_frame["q_inflow"].map(thresholds)
    candidate_frame["competitive"] = candidate_frame["q_inflow"].isin(competitive_q)
    candidate_frame["selected"] = candidate_frame["q_inflow"] == selected_q
    candidate_frame["weighted_mean_storage_nlpd"] = candidate_frame[
        "weighted_storage_nlpd"
    ]
    candidate_frame["global_observation_weighted_storage_nlpd"] = candidate_frame[
        "global_storage_nlpd"
    ]
    candidate_frame["joint_nlpd_per_component"] = candidate_frame["joint_nlpd"]
    candidate_frame["max_elapsed_lag_autocorrelation"] = candidate_frame[
        "max_material_elapsed_lag_autocorrelation"
    ]
    candidate_frame["eligible_flag"] = candidate_frame["eligible"]
    candidate_frame["competitive_flag"] = candidate_frame["competitive"]
    candidate_frame["selected_flag"] = candidate_frame["selected"]
    candidate_frame["selected_reason"] = np.where(
        candidate_frame["selected"], "smallest competitive candidate", ""
    )
    candidate_frame = _sort_summary(candidate_frame, selected_q, competitive_q)
    selected_horizons = _horizon_frame(
        prepared, passes, competitive_q, windows, options, selected_q
    )
    r_sensitivity = _r_sensitivity(
        prepared, q_values, prior_values, windows, options, base_config, selected_q
    )
    selected_config = _proposed_config(
        base_config,
        selected_q,
        proposed_configuration_version=str(proposed_configuration_version),
        prior_values=prior_values,
        q_values=q_values,
        windows=windows,
        weights=weights,
        settings=options,
        candidate_frame=candidate_frame,
        competitive_q=competitive_q,
        selection_threshold=selection_threshold,
        best_q=best_q,
        data_fingerprint=_data_fingerprint(storage, discharge),
    )
    order = {q: i for i, q in enumerate(competitive_q)}
    candidate_priority = {
        q: (0, 0) if q == selected_q else (1, order.get(q, 0)) for q in q_values
    }
    window_frame = (
        window_frame.assign(
            _priority=window_frame["q_inflow"].map(
                lambda value: candidate_priority.get(value, (2, value))[0]
            ),
            _order=window_frame["q_inflow"].map(
                lambda value: candidate_priority.get(value, (2, value))[1]
            ),
        )
        .sort_values(["_priority", "_order", "q_inflow"], kind="stable")
        .drop(columns=["_priority", "_order"])
        .reset_index(drop=True)
    )
    window_frame["storage_targeted_nlpd"] = window_frame["storage_nlpd"]
    window_frame["joint_nlpd_per_component"] = window_frame["joint_nlpd"]
    regime_frame = _regime_frame(window_frame)
    if selected_q in {min(q_values), max(q_values)}:
        warnings.append("selected candidate is at the edge of the supplied grid")
    if len(windows) < 5:
        warnings.append(
            "fewer than five valid windows are available; uncertainty is weak"
        )
    for metric, threshold in (
        ("joint_nis", options.nis_warning_range),
        ("storage_nis", options.nis_warning_range),
    ):
        if threshold is not None:
            value = float(selected_pass.physical.get(metric, np.nan))
            if np.isfinite(value) and not (threshold[0] <= value <= threshold[1]):
                warnings.append(
                    f"selected {metric} is outside the configured warning range"
                )
    if options.innovation_bias_warning is not None:
        for metric in ("storage_bias", "outflow_bias", "conditional_storage_bias"):
            value = abs(float(selected_pass.physical.get(metric, np.nan)))
            if np.isfinite(value) and value > options.innovation_bias_warning:
                warnings.append(
                    f"selected {metric} exceeds the configured warning limit"
                )
    selection_reason = (
        "smallest q_inflow within paired one-standard-error practical equivalence"
    )
    return InflowTuningResult(
        selected_config=selected_config,
        selected_prior_hourly_increment_sd=selected_prior,
        selected_q_inflow=selected_q,
        candidate_summary=candidate_frame,
        window_diagnostics=window_frame,
        regime_diagnostics=regime_frame,
        horizon_diagnostics=selected_horizons,
        r_sensitivity=r_sensitivity,
        competitive_candidates=tuple(float(v) for v in competitive_q),
        selection_threshold=selection_threshold,
        selection_reason=selection_reason,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def evaluate_inflow_config(
    storage: pd.Series,
    discharge: pd.Series,
    config: ReservoirConfig,
    *,
    evaluation_window: TuningWindow | None = None,
    settings: InflowTuningSettings | None = None,
) -> InflowConfigEvaluationResult:
    """Evaluate a frozen configuration on an independent period.

    This function performs no candidate selection and cannot modify the
    supplied configuration.  A caller can therefore keep a final test period
    separate from :func:`tune_inflow_process_noise`.
    """

    options = settings or InflowTuningSettings()
    prepared = _prepare(storage, discharge, config)
    if evaluation_window is None:
        evaluation_window = TuningWindow(
            "evaluation",
            prepared.index[0],
            prepared.index[-1] + pd.Timedelta(nanoseconds=1),
        )
    windows = _validate_windows((evaluation_window,))
    prior = sqrt(float(config.q[1, 1]) * 3600.0)
    weights = (1.0,)
    candidate = _evaluate_candidate(
        prepared,
        prior_sd=prior,
        q_inflow=float(config.q[1, 1]),
        windows=windows,
        settings=options,
        window_weights=weights,
    )
    passes = {candidate.q_inflow: candidate}
    candidate_frame, window_frame, _, _, warnings = _summary_frames(
        passes, windows, weights, options
    )
    regime_frame = _regime_frame(window_frame)
    horizon_frame = _horizon_frame(
        prepared, passes, (candidate.q_inflow,), windows, options, candidate.q_inflow
    )
    return InflowConfigEvaluationResult(
        config,
        candidate_frame,
        window_frame,
        regime_frame,
        horizon_frame,
        tuple(warnings),
    )


def _timestamp(value: datetime | pd.Timestamp, name: str) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return result


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


def _validate_windows(windows: Sequence[TuningWindow]) -> tuple[TuningWindow, ...]:
    result = tuple(
        window if isinstance(window, TuningWindow) else TuningWindow(*window)
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


def _candidate_values(values: Sequence[float]) -> tuple[list[float], list[float]]:
    prior = [float(value) for value in values]
    if len(prior) < 3 or any(not np.isfinite(value) or value <= 0.0 for value in prior):
        raise ValueError("at least three positive finite candidates are required")
    q = [prior_hourly_increment_to_q(value) for value in prior]
    if len(set(q)) != len(q):
        raise ValueError("candidate q_inflow values must be unique")
    order = np.argsort(q)
    return [prior[int(i)] for i in order], [q[int(i)] for i in order]


def _window_weights(windows: Sequence[TuningWindow]) -> np.ndarray:
    values = np.asarray(
        [1.0 if window.weight is None else window.weight for window in windows],
        dtype=float,
    )
    return values / values.sum()


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
        raise ValueError("tuning requires constant diagonal q, r, and p0 matrices")
    finite_storage = np.flatnonzero(np.isfinite(storage_values))
    if len(finite_storage) < 2:
        raise TuningError("at least two finite storage observations are required")
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
        raise TuningError("initial state is not finite")
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
    )


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


def _evaluate_candidate(
    prepared: _Prepared,
    *,
    prior_sd: float,
    q_inflow: float,
    windows: Sequence[TuningWindow],
    settings: InflowTuningSettings,
    window_weights: Array,
    r_override: Array | None = None,
) -> _Pass:
    q_discrete = (
        prepared.q_storage * prepared.storage_basis
        + q_inflow * prepared.inflow_basis
        + prepared.q_outflow * prepared.outflow_basis
    )
    r = prepared.r if r_override is None else np.asarray(r_override, dtype=float)
    result: KalmanFilterResult | None = None
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
            prior_sd,
            None,
            [],
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
    rows: list[dict[str, Any]] = []
    regularization_count = 0
    max_jitter = 0.0
    warmup_end = prepared.timestamps[1] + pd.Timedelta(
        seconds=settings.warmup.total_seconds()
    )
    for row, timestamp in enumerate(prepared.timestamps):
        finite = np.isfinite(prepared.observations[row])
        record: dict[str, Any] = {
            "row": row,
            "timestamp": timestamp,
            "storage_innovation": np.nan,
            "primary_nlpd": np.nan,
            "joint_nlpd": np.nan,
            "joint_nis": np.nan,
            "joint_components": int(finite.sum()),
            "storage_nis": np.nan,
            "outflow_nis": np.nan,
            "conditional_storage_nis": np.nan,
            "storage_z": np.nan,
            "outflow_z": np.nan,
            "conditional_storage_z": np.nan,
            "jitter": 0.0,
        }
        innovation = result.innovations[row]
        covariance = result.innovation_covariances[row]
        if finite[0] and np.isfinite(innovation[0]):
            record["storage_innovation"] = innovation[0]
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
                record["joint_nlpd"] = joint
                record["joint_nis"] = quad
                record["jitter"] = jitter
                regularization_count += int(jitter > 0.0)
                max_jitter = max(max_jitter, jitter)
            for component, key in ((0, "storage"), (1, "outflow")):
                if finite[component]:
                    variance = covariance[component, component]
                    if np.isfinite(variance) and variance > 0.0:
                        z = innovation[component] / sqrt(variance)
                        record[f"{key}_z"] = z
                        record[f"{key}_nis"] = z * z
            if finite[0]:
                if finite[1]:
                    try:
                        conditional = storage_conditional_nlpd(
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
                        record["conditional_storage_z"] = conditional_innovation / sqrt(
                            max(conditional_variance, np.finfo(float).tiny)
                        )
                        record["conditional_storage_nis"] = (
                            record["conditional_storage_z"] ** 2
                        )
                    except ValueError as error:
                        reasons.append(str(error))
                        conditional = np.nan
                    record["primary_nlpd"] = conditional
                else:
                    try:
                        record["primary_nlpd"] = marginal_predictive_nlpd(
                            innovation[0],
                            covariance[0, 0],
                            max_jitter_fraction=settings.max_jitter_fraction,
                        )
                    except ValueError as error:
                        reasons.append(str(error))
        # ``score`` marks an eligible post-initialization timestamp.  The
        # primary storage score may still be missing when storage is missing;
        # joint and outflow diagnostics remain valid for that row.
        record["score"] = row >= 2 and timestamp >= warmup_end
        rows.append(record)
    window_rows: dict[str, dict[str, Any]] = {}
    for window in windows:
        eligible = [
            row
            for row in rows
            if window.start <= row["timestamp"] < window.end and row["score"]
        ]
        selected = [row for row in eligible if np.isfinite(row["primary_nlpd"])]
        primary = np.asarray([row["primary_nlpd"] for row in selected], dtype=float)
        joint = [row for row in eligible if np.isfinite(row["joint_nlpd"])]
        window_rows[window.name] = _aggregate_rows(
            primary, joint, selected, eligible, window
        )
    validation_rows = [
        row
        for row in rows
        if row["score"]
        and any(window.start <= row["timestamp"] < window.end for window in windows)
    ]
    physical = _physical_metrics(result, validation_rows, settings)
    if regularization_count > settings.max_regularized_steps:
        reasons.append("excessive covariance regularization")
    return _Pass(
        q_inflow,
        prior_sd,
        result,
        rows,
        window_rows,
        physical,
        regularization_count,
        max_jitter,
        list(dict.fromkeys(reasons)),
    )


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
                len(vector) * log(2.0 * np.pi)
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


def _aggregate_rows(
    primary: Array,
    joint: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    window: TuningWindow,
) -> dict[str, Any]:
    def mean_metric(key: str) -> float:
        values = np.asarray(
            [row[key] for row in selected if np.isfinite(row[key])], dtype=float
        )
        return float(np.mean(values)) if len(values) else np.nan

    joint_count = int(sum(row["joint_components"] for row in joint))
    return {
        "window": window.name,
        "regime": window.regime,
        "start": window.start,
        "end": window.end,
        "storage_nlpd": float(np.mean(primary)) if len(primary) else np.nan,
        "storage_count": int(len(primary)),
        "joint_nlpd": float(
            sum(row["joint_nlpd"] * row["joint_components"] for row in joint)
            / joint_count
        )
        if joint_count
        else np.nan,
        "joint_count": joint_count,
        "joint_nis": float(sum(row["joint_nis"] for row in joint) / joint_count)
        if joint_count
        else np.nan,
        "storage_nis": mean_metric("storage_nis"),
        "outflow_nis": mean_metric("outflow_nis"),
        "conditional_storage_nis": mean_metric("conditional_storage_nis"),
        "storage_bias": mean_metric("storage_z"),
        "outflow_bias": mean_metric("outflow_z"),
        "conditional_storage_bias": mean_metric("conditional_storage_z"),
        "coverage": float(len(primary) / max(1, len(eligible))),
        "regularization_count": int(sum(row["jitter"] > 0 for row in selected)),
    }


def _mean_finite(rows: list[dict[str, Any]], key: str) -> float:
    values = np.asarray(
        [row[key] for row in rows if np.isfinite(row[key])], dtype=float
    )
    return float(np.mean(values)) if len(values) else np.nan


def _physical_metrics(
    result: KalmanFilterResult,
    rows: list[dict[str, Any]],
    settings: InflowTuningSettings,
) -> dict[str, float]:
    indices = np.asarray([row["row"] for row in rows], dtype=int)
    timestamps = pd.DatetimeIndex([row["timestamp"] for row in rows])
    inflow = result.filtered_means[indices, 1] if len(indices) else np.array([])
    finite = np.isfinite(inflow)
    differences = np.diff(inflow)
    intervals = np.asarray(
        [
            (timestamps[index] - timestamps[index - 1]).total_seconds()
            for index in range(1, len(timestamps))
        ],
        dtype=float,
    )
    normalized = (
        differences / intervals if len(differences) else np.array([], dtype=float)
    )
    finite_norm = normalized[np.isfinite(normalized)]

    def autocorrelation_metrics(values: np.ndarray) -> tuple[float, float]:
        if len(timestamps) < 2 or settings.innovation_max_lag.total_seconds() <= 0.0:
            return np.nan, np.nan
        frame = elapsed_lag_autocorrelation(
            timestamps, values, settings.innovation_max_lag
        )
        finite_frame = frame[np.isfinite(frame["autocorrelation"])]
        if finite_frame.empty:
            return np.nan, np.nan
        row = finite_frame.iloc[
            int(np.argmax(np.abs(finite_frame["autocorrelation"].to_numpy())))
        ]
        return float(abs(row["autocorrelation"])), float(row["lag_seconds"])

    max_storage_ac, max_storage_lag = autocorrelation_metrics(
        np.asarray([row["storage_z"] for row in rows], dtype=float)
    )
    max_outflow_ac, max_outflow_lag = autocorrelation_metrics(
        np.asarray([row["outflow_z"] for row in rows], dtype=float)
    )
    max_conditional_ac, max_conditional_lag = autocorrelation_metrics(
        np.asarray([row["conditional_storage_z"] for row in rows], dtype=float)
    )
    scored_storage_innovations = np.asarray(
        [
            row["storage_innovation"]
            for row in rows
            if row["score"] and np.isfinite(row["storage_innovation"])
        ],
        dtype=float,
    )
    autocorrelations = (max_storage_ac, max_outflow_ac, max_conditional_ac)
    return {
        "negative_inflow_frequency": float(np.mean(inflow[finite] < 0.0))
        if finite.any()
        else np.nan,
        "inflow_change_median_per_second": float(np.median(finite_norm))
        if len(finite_norm)
        else np.nan,
        "inflow_change_q95_per_second": float(np.quantile(np.abs(finite_norm), 0.95))
        if len(finite_norm)
        else np.nan,
        "storage_bias": _mean_finite(rows, "storage_z"),
        "outflow_bias": _mean_finite(rows, "outflow_z"),
        "conditional_storage_bias": _mean_finite(rows, "conditional_storage_z"),
        "joint_nis": (
            float(
                sum(row["joint_nis"] for row in rows if np.isfinite(row["joint_nis"]))
                / sum(
                    row["joint_components"]
                    for row in rows
                    if np.isfinite(row["joint_nis"])
                )
            )
            if any(np.isfinite(row["joint_nis"]) for row in rows)
            else np.nan
        ),
        "storage_nis": _mean_finite(rows, "storage_nis"),
        "outflow_nis": _mean_finite(rows, "outflow_nis"),
        "conditional_storage_nis": _mean_finite(rows, "conditional_storage_nis"),
        "causal_storage_prediction_rmse": (
            float(np.sqrt(np.mean(scored_storage_innovations**2)))
            if len(scored_storage_innovations)
            else np.nan
        ),
        "causal_storage_prediction_bias": (
            float(np.mean(scored_storage_innovations))
            if len(scored_storage_innovations)
            else np.nan
        ),
        "max_storage_elapsed_lag_autocorrelation": max_storage_ac,
        "max_storage_elapsed_lag_seconds": max_storage_lag,
        "max_outflow_elapsed_lag_autocorrelation": max_outflow_ac,
        "max_outflow_elapsed_lag_seconds": max_outflow_lag,
        "max_conditional_storage_elapsed_lag_autocorrelation": max_conditional_ac,
        "max_conditional_storage_elapsed_lag_seconds": max_conditional_lag,
        "max_material_elapsed_lag_autocorrelation": (
            max(value for value in autocorrelations if np.isfinite(value))
            if any(np.isfinite(value) for value in autocorrelations)
            else np.nan
        ),
    }


def _summary_frames(
    passes: Mapping[float, _Pass],
    windows: Sequence[TuningWindow],
    weights: Array,
    settings: InflowTuningSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[float, ...], list[str]]:
    window_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    eligible: list[float] = []
    warnings: list[str] = []
    for q, candidate in passes.items():
        valid = not candidate.reasons
        for window in windows:
            record = dict(candidate.window_rows[window.name])
            record.update(
                {
                    "q_inflow": q,
                    "prior_hourly_increment_sd": candidate.prior_sd,
                    "eligible": valid
                    and record["storage_count"]
                    >= settings.min_scored_storage_observations,
                    "rejection_reasons": "; ".join(candidate.reasons),
                }
            )
            if not record["eligible"]:
                valid = False
                if record["storage_count"] < settings.min_scored_storage_observations:
                    record["rejection_reasons"] = (
                        record["rejection_reasons"] + "; "
                        if record["rejection_reasons"]
                        else ""
                    ) + "insufficient scored storage observations"
            window_records.append(record)
        if valid:
            eligible.append(q)
        means = [
            candidate.window_rows[window.name]["storage_nlpd"] for window in windows
        ]
        valid_means = np.asarray(means, dtype=float)
        weighted = (
            float(np.dot(weights, valid_means))
            if np.isfinite(valid_means).all()
            else np.nan
        )
        all_rows = [
            row for window in windows for row in [candidate.window_rows[window.name]]
        ]
        joint_count = sum(row["joint_count"] for row in all_rows)
        joint_score = sum(
            row["joint_nlpd"] * row["joint_count"]
            for row in all_rows
            if np.isfinite(row["joint_nlpd"])
        )
        storage_count = sum(row["storage_count"] for row in all_rows)
        storage_score = sum(
            row["storage_nlpd"] * row["storage_count"]
            for row in all_rows
            if np.isfinite(row["storage_nlpd"])
        )
        summary_records.append(
            {
                "prior_hourly_increment_sd": candidate.prior_sd,
                "q_inflow": q,
                "weighted_storage_nlpd": weighted,
                "global_storage_nlpd": (
                    float(storage_score / storage_count) if storage_count else np.nan
                ),
                "joint_nlpd": float(joint_score / joint_count)
                if joint_count
                else np.nan,
                "joint_nis": candidate.physical.get("joint_nis", np.nan),
                "storage_nis": candidate.physical.get("storage_nis", np.nan),
                "outflow_nis": candidate.physical.get("outflow_nis", np.nan),
                "conditional_storage_nis": candidate.physical.get(
                    "conditional_storage_nis", np.nan
                ),
                "storage_bias": candidate.physical.get("storage_bias", np.nan),
                "outflow_bias": candidate.physical.get("outflow_bias", np.nan),
                "conditional_storage_bias": candidate.physical.get(
                    "conditional_storage_bias", np.nan
                ),
                "negative_inflow_frequency": candidate.physical.get(
                    "negative_inflow_frequency", np.nan
                ),
                "inflow_change_q95_per_second": candidate.physical.get(
                    "inflow_change_q95_per_second", np.nan
                ),
                "causal_storage_prediction_rmse": candidate.physical.get(
                    "causal_storage_prediction_rmse", np.nan
                ),
                "causal_storage_prediction_bias": candidate.physical.get(
                    "causal_storage_prediction_bias", np.nan
                ),
                "max_material_elapsed_lag_autocorrelation": candidate.physical.get(
                    "max_material_elapsed_lag_autocorrelation", np.nan
                ),
                "regularization_count": candidate.regularization_count,
                "max_jitter": candidate.max_jitter,
                "eligible": bool(valid),
                "rejection_reasons": "; ".join(candidate.reasons),
                "windows_passed": int(
                    sum(
                        candidate.window_rows[w.name]["storage_count"]
                        >= settings.min_scored_storage_observations
                        for w in windows
                    )
                ),
                "scored_storage_count": int(
                    sum(candidate.window_rows[w.name]["storage_count"] for w in windows)
                ),
                "joint_observed_component_count": int(joint_count),
            }
        )
    return (
        pd.DataFrame(summary_records),
        pd.DataFrame(window_records),
        pd.DataFrame(),
        tuple(eligible),
        warnings,
    )


def _paired_standard_error(
    differences: Array, weights: Array, settings: InflowTuningSettings
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


def _sort_summary(
    frame: pd.DataFrame, selected_q: float, competitive_q: Sequence[float]
) -> pd.DataFrame:
    priority = np.where(frame["selected"], 0, np.where(frame["competitive"], 1, 2))
    result = (
        frame.assign(_priority=priority)
        .sort_values(["_priority", "q_inflow"], kind="stable")
        .drop(columns="_priority")
    )
    return result.reset_index(drop=True)


def _regime_frame(window_frame: pd.DataFrame) -> pd.DataFrame:
    if window_frame.empty:
        return pd.DataFrame()
    numeric = [
        "storage_nlpd",
        "joint_nlpd",
        "joint_nis",
        "storage_nis",
        "outflow_nis",
        "conditional_storage_nis",
        "storage_bias",
        "outflow_bias",
        "conditional_storage_bias",
    ]
    grouped = window_frame.groupby(
        ["q_inflow", "prior_hourly_increment_sd", "regime"],
        as_index=False,
        dropna=False,
    )
    result = grouped[numeric].mean()
    counts = (
        grouped["storage_count"]
        .sum()
        .rename(columns={"storage_count": "scored_storage_count"})
    )
    return result.merge(
        counts, on=["q_inflow", "prior_hourly_increment_sd", "regime"], how="left"
    )


def _horizon_frame(
    prepared: _Prepared,
    passes: Mapping[float, _Pass],
    q_values: Sequence[float],
    windows: Sequence[TuningWindow],
    settings: InflowTuningSettings,
    selected_q: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for q in q_values:
        candidate = passes.get(q)
        if candidate is None or candidate.filter_result is None:
            continue
        tolerance = settings.horizon_tolerance
        if tolerance is None:
            cadence = float(np.median(prepared.elapsed_seconds))
            tolerance_seconds = cadence / 2.0
        else:
            tolerance_seconds = tolerance.total_seconds()
        for horizon in settings.forecast_horizons:
            values: list[tuple[float, float, float, bool]] = []
            horizon_seconds = horizon.total_seconds()
            for window in windows:
                for origin in range(2, len(prepared.timestamps) - 1):
                    origin_timestamp = prepared.timestamps[origin]
                    if not (window.start <= origin_timestamp < window.end):
                        continue
                    if origin_timestamp < prepared.timestamps[1] + pd.Timedelta(
                        seconds=settings.warmup.total_seconds()
                    ):
                        continue
                    target = origin_timestamp + pd.Timedelta(seconds=horizon_seconds)
                    target_rows = [
                        row
                        for row in range(origin + 1, len(prepared.timestamps))
                        if window.start <= prepared.timestamps[row] < window.end
                        and prepared.timestamps[row] >= target
                        and (prepared.timestamps[row] - target).total_seconds()
                        <= tolerance_seconds
                        and candidate.rows[row]["score"]
                        and np.isfinite(prepared.observations[row, 0])
                    ]
                    if not target_rows:
                        continue
                    target_row = target_rows[0]
                    mean = candidate.filter_result.filtered_means[origin].copy()
                    covariance = candidate.filter_result.filtered_covariances[
                        origin
                    ].copy()
                    for step in range(origin, target_row):
                        mean, covariance = predict_state(
                            mean,
                            covariance,
                            prepared.transitions[step],
                            prepared.q_storage * prepared.storage_basis[step]
                            + q * prepared.inflow_basis[step]
                            + prepared.q_outflow * prepared.outflow_basis[step],
                        )
                    predicted = float(mean[0])
                    variance = float(covariance[0, 0] + prepared.r[0, 0])
                    observed = float(prepared.observations[target_row, 0])
                    try:
                        nll = marginal_predictive_nlpd(
                            observed - predicted,
                            variance,
                            max_jitter_fraction=settings.max_jitter_fraction,
                        )
                        z = (observed - predicted) / sqrt(variance)
                        values.append((nll, z, observed - predicted, abs(z) <= 1.96))
                    except ValueError:
                        continue
            if values:
                arr = np.asarray(values, dtype=float)
                records.append(
                    {
                        "q_inflow": q,
                        "prior_hourly_increment_sd": candidate.prior_sd,
                        "selected": q == selected_q,
                        "horizon_seconds": horizon_seconds,
                        "storage_nlpd": float(np.mean(arr[:, 0])),
                        "bias": float(np.mean(arr[:, 1])),
                        "rmse": float(np.sqrt(np.mean(arr[:, 2] ** 2))),
                        "coverage_95": float(np.mean(arr[:, 3])),
                        "nis": float(np.mean(arr[:, 1] ** 2)),
                        "count": len(values),
                    }
                )
            else:
                records.append(
                    {
                        "q_inflow": q,
                        "prior_hourly_increment_sd": candidate.prior_sd,
                        "selected": q == selected_q,
                        "horizon_seconds": horizon_seconds,
                        "storage_nlpd": np.nan,
                        "bias": np.nan,
                        "rmse": np.nan,
                        "coverage_95": np.nan,
                        "nis": np.nan,
                        "count": 0,
                    }
                )
    return pd.DataFrame(records)


def _r_sensitivity(
    prepared: _Prepared,
    q_values: Sequence[float],
    prior_values: Sequence[float],
    windows: Sequence[TuningWindow],
    settings: InflowTuningSettings,
    config: ReservoirConfig,
    selected_q: float,
) -> pd.DataFrame | None:
    if not settings.r_sensitivity_multipliers:
        return None
    rows: list[dict[str, Any]] = []
    weights = _window_weights(windows)
    for multiplier in settings.r_sensitivity_multipliers:
        scenario: list[_Pass] = []
        for prior, q in zip(prior_values, q_values, strict=True):
            scenario.append(
                _evaluate_candidate(
                    prepared,
                    prior_sd=prior,
                    q_inflow=q,
                    windows=windows,
                    settings=settings,
                    window_weights=weights,
                    r_override=np.asarray(config.r) * multiplier,
                )
            )
        eligible = [
            candidate
            for candidate in scenario
            if not candidate.reasons
            and all(
                np.isfinite(candidate.window_rows[w.name]["storage_nlpd"])
                for w in windows
            )
        ]
        if not eligible:
            rows.append(
                {
                    "r_multiplier": multiplier,
                    "best_q_inflow": np.nan,
                    "conservative_q_inflow": np.nan,
                    "base_selected_competitive": False,
                }
            )
            continue
        means = {
            candidate.q_inflow: float(
                np.dot(
                    weights,
                    [candidate.window_rows[w.name]["storage_nlpd"] for w in windows],
                )
            )
            for candidate in eligible
        }
        best = min(means, key=lambda q: (means[q], q))
        differences = {
            q: np.asarray(
                [
                    candidate.window_rows[w.name]["storage_nlpd"]
                    - next(
                        item for item in eligible if item.q_inflow == best
                    ).window_rows[w.name]["storage_nlpd"]
                    for w in windows
                ]
            )
            for q, candidate in (
                (candidate.q_inflow, candidate) for candidate in eligible
            )
        }
        thresholds = {
            q: max(
                _paired_standard_error(value, weights, settings),
                settings.practical_equivalence_tolerance,
            )
            for q, value in differences.items()
        }
        conservative = min(
            q
            for q, value in differences.items()
            if float(np.dot(weights, value)) <= thresholds[q] + 1e-12
        )
        rows.append(
            {
                "r_multiplier": multiplier,
                "best_q_inflow": best,
                "conservative_q_inflow": conservative,
                "base_selected_competitive": selected_q in differences
                and float(np.dot(weights, differences[selected_q]))
                <= thresholds[selected_q] + 1e-12,
                "best_prior_hourly_increment_sd": sqrt(best * 3600.0),
                "conservative_prior_hourly_increment_sd": sqrt(conservative * 3600.0),
                "score_movement_grid_positions": q_values.index(conservative)
                - q_values.index(selected_q),
            }
        )
    return pd.DataFrame(rows)


def _data_fingerprint(storage: pd.Series, discharge: pd.Series) -> str:
    digest = blake2b(digest_size=16)
    digest.update(storage.index.asi8.tobytes())
    digest.update(storage.to_numpy(dtype=float, copy=False).tobytes())
    digest.update(discharge.to_numpy(dtype=float, copy=False).tobytes())
    return digest.hexdigest()


def _proposed_config(
    base: ReservoirConfig,
    selected_q: float,
    *,
    proposed_configuration_version: str,
    prior_values: Sequence[float],
    q_values: Sequence[float],
    windows: Sequence[TuningWindow],
    weights: Array,
    settings: InflowTuningSettings,
    candidate_frame: pd.DataFrame,
    competitive_q: Sequence[float],
    selection_threshold: float,
    best_q: float,
    data_fingerprint: str,
) -> ReservoirConfig:
    q = np.asarray(base.q, dtype=float).copy()
    q[1, 1] = selected_q
    metadata = dict(base.metadata)
    metadata.update(
        {
            "calibration_method": (
                "causal-storage-conditional-nlpd-paired-block-conservative-selection"
            ),
            "tuned_parameters": ["q_inflow"],
            "fixed_parameters": ["q_storage", "q_outflow", "r", "p0"],
            "candidate_prior_hourly_increment_sd": list(prior_values),
            "candidate_q_inflow": list(q_values),
            "validation_windows": [
                {
                    "name": w.name,
                    "start": w.start.isoformat(),
                    "end": w.end.isoformat(),
                    "regime": w.regime,
                }
                for w in windows
            ],
            "window_weights": list(weights),
            "warmup_seconds": settings.warmup.total_seconds(),
            "forecast_horizons_seconds": [
                value.total_seconds() for value in settings.forecast_horizons
            ],
            "selection_settings": {
                "practical_equivalence_tolerance": (
                    settings.practical_equivalence_tolerance
                ),
                "bootstrap_samples": settings.bootstrap_samples,
                "random_seed": settings.random_seed,
            },
            "selected_prior_hourly_increment_sd": sqrt(selected_q * 3600.0),
            "selected_q_inflow": selected_q,
            "paired_selection_threshold": selection_threshold,
            "competitive_candidates": list(competitive_q),
            "best_mean_storage_nlpd": float(
                candidate_frame.loc[
                    candidate_frame["q_inflow"] == best_q,
                    "weighted_storage_nlpd",
                ].iloc[0]
            ),
            "selected_mean_storage_nlpd": float(
                candidate_frame.loc[
                    candidate_frame["q_inflow"] == selected_q, "weighted_storage_nlpd"
                ].iloc[0]
            ),
            "data_fingerprint": data_fingerprint,
        }
    )
    return ReservoirConfig(
        reservoir_id=base.reservoir_id,
        reservoir_name=base.reservoir_name,
        q=q,
        r=np.asarray(base.r).copy(),
        p0=np.asarray(base.p0).copy(),
        smoothing_lag=base.smoothing_lag,
        initialization_strategy=base.initialization_strategy,
        inflow_units=base.inflow_units,
        model_version=base.model_version,
        configuration_version=proposed_configuration_version,
        metadata=metadata,
        unit_system=base.unit_system,
    )


__all__ = [
    "InflowConfigEvaluationResult",
    "InflowTuningResult",
    "InflowTuningSettings",
    "TuningError",
    "TuningWindow",
    "elapsed_lag_autocorrelation",
    "evaluate_inflow_config",
    "joint_predictive_nlpd",
    "marginal_predictive_nlpd",
    "normalized_innovation_squared",
    "prior_hourly_increment_to_q",
    "storage_conditional_nlpd",
    "tune_inflow_process_noise",
]
