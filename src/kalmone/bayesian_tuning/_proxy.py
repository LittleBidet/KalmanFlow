from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ._types import BayesianTuningSettings

Array = np.ndarray


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
