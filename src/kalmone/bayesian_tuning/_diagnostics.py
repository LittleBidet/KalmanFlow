from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd

from ..kalman import KalmanFilterResult
from ._preparation import _timestamp_seconds, _validate_index
from ._types import BayesianEvaluationSettings, ValidationWindow

Array = np.ndarray


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
    if not len(targets):
        return pd.DataFrame(
            columns=["lag_seconds", "autocorrelation", "pair_count"]
        )
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
        values = diagnostics[key][eligible_mask]
        values = values[np.isfinite(values)]
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
        "regularization_count": int(np.sum(diagnostics["jitter"][joint_mask] > 0)),
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
        if frame.empty:
            return np.nan, np.nan
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
