"""Validation tools for reservoir inflow estimates.

The validation helpers in this module deliberately keep the upstream gauge in
its proper role: it is a partial-catchment timing and agreement proxy, not a
measurement of total reservoir inflow.  Comparisons are made on a regular
clock, with pairwise-complete observations and no interpolation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from kalmone.units import CFS_TO_ACRE_FEET_PER_SECOND

_ESTIMATE_COLUMNS = (
    "raw_inflow",
    "centered_rolling_inflow",
    "estimated_inflow",
    "delayed_revised_inflow",
)
_DISPLAY_NAMES = {
    "raw_inflow": "raw_inflow",
    "centered_rolling_inflow": "centered_rolling_inflow",
    "estimated_inflow": "estimated_inflow (causal)",
    "delayed_revised_inflow": "revised_inflow (delayed diagnostic)",
}
_METRIC_COLUMNS = (
    "paired_observations",
    "coverage",
    "pearson_correlation",
    "spearman_correlation",
    "kge",
    "kge_correlation",
    "kge_variability_ratio",
    "kge_mean_flow_ratio",
    "percent_bias",
    "nse",
    "normalized_rmse",
)


@dataclass(frozen=True)
class ValidationSettings:
    """Settings controlling regularization, overlap, and validation windows.

    ``training_*`` and ``evaluation_*`` bounds are optional timezone-aware
    timestamps.  When a training window is supplied, validation output uses
    that window to select a lag and freezes the selected lag before evaluating
    the held-out window.  The evaluation window should normally be supplied as
    well; if it is omitted and ``training_end`` is present, rows after the
    training end are used as the evaluation window.
    """

    evaluation_frequency: str | pd.Timedelta = "1h"
    cross_correlation_range: int = 48
    minimum_paired_observations: int = 3
    centered_rolling_window: str | pd.Timedelta = "6h"
    trailing_rolling_window: str | pd.Timedelta | None = None
    training_start: object | None = None
    training_end: object | None = None
    evaluation_start: object | None = None
    evaluation_end: object | None = None
    flow_to_volume_per_second: float = CFS_TO_ACRE_FEET_PER_SECOND

    def __post_init__(self) -> None:
        frequency = _positive_timedelta(
            self.evaluation_frequency, name="evaluation_frequency"
        )
        rolling_window_value = (
            self.trailing_rolling_window
            if self.trailing_rolling_window is not None
            else self.centered_rolling_window
        )
        rolling_window = _positive_timedelta(
            rolling_window_value, name="centered_rolling_window"
        )
        correlation_range = _nonnegative_integer(
            self.cross_correlation_range, name="cross_correlation_range"
        )
        minimum = _minimum_integer(
            self.minimum_paired_observations,
            name="minimum_paired_observations",
            minimum=2,
        )
        factor = float(self.flow_to_volume_per_second)
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("flow_to_volume_per_second must be positive and finite")

        bounds: dict[str, pd.Timestamp | None] = {}
        for name in (
            "training_start",
            "training_end",
            "evaluation_start",
            "evaluation_end",
        ):
            value = getattr(self, name)
            bounds[name] = None if value is None else _utc_timestamp(value, name)

        _validate_bounds(bounds, "training_start", "training_end")
        _validate_bounds(bounds, "evaluation_start", "evaluation_end")

        object.__setattr__(self, "evaluation_frequency", frequency)
        object.__setattr__(self, "centered_rolling_window", rolling_window)
        object.__setattr__(self, "trailing_rolling_window", rolling_window)
        object.__setattr__(self, "cross_correlation_range", correlation_range)
        object.__setattr__(self, "minimum_paired_observations", minimum)
        object.__setattr__(self, "flow_to_volume_per_second", factor)
        for name, value in bounds.items():
            object.__setattr__(self, name, value)

    @property
    def frequency(self) -> pd.Timedelta:
        """Alias for the regular evaluation frequency."""

        return self.evaluation_frequency

    @property
    def max_lag_hours(self) -> int:
        """Alias for the positive side of the lag search range."""

        return self.cross_correlation_range


@dataclass(frozen=True)
class ValidationOutputs:
    """Tables and optional figures produced by :func:`generate_validation_outputs`."""

    validation_frame: pd.DataFrame
    upstream_proxy_agreement: pd.DataFrame
    best_lag_summary: pd.DataFrame
    storage_closure: pd.DataFrame
    inflow_behavior: pd.DataFrame
    cross_correlation_plot: Any | None = None
    estimate_upstream_scatter_plot: Any | None = None
    interpretation: str = (
        "Pearson correlation and KGE measure agreement with a partial-catchment "
        "upstream proxy, not total-inflow accuracy. KGE bias and variability "
        "components may be poor even when timing is correct because the gauge "
        "represents only part of the contributing flow. Storage closure is "
        "stronger evidence of internal consistency. Inflow behavior reports "
        "the frequency and mean magnitude of negative estimates plus the mean "
        "absolute change between adjacent finite estimates. These diagnostics "
        "do not require knowing total inflow."
    )

    @property
    def inflow_behavior_metrics(self) -> pd.DataFrame:
        """Alias for the inflow behavior metrics table."""

        return self.inflow_behavior


def build_validation_frame(
    comparison: pd.DataFrame,
    *,
    settings: ValidationSettings | None = None,
    raw_inflow: pd.Series | Iterable[float] | None = None,
    centered_rolling_inflow: pd.Series | Iterable[float] | None = None,
    trailing_rolling_inflow: pd.Series | Iterable[float] | None = None,
    estimated_inflow: pd.Series | Iterable[float] | None = None,
    delayed_revised_inflow: pd.Series | Iterable[float] | None = None,
    upstream_flow: pd.Series | Iterable[float] | None = None,
) -> pd.DataFrame:
    """Create a regular, hourly comparison frame without interpolation.

    ``comparison`` must have ``storage`` and ``outflow`` columns.  The
    upstream series and four estimate series may be columns in the frame or
    supplied explicitly.  ``revised_inflow`` is accepted as an input alias but
    is returned as ``delayed_revised_inflow`` to make its delayed diagnostic
    status explicit. The rolling baseline is centered and returned as
    ``centered_rolling_inflow``. The old ``trailing_rolling_inflow`` input name
    remains accepted as an alias.

    Values are converted to numeric, then averaged into the requested regular
    frequency.  An all-missing bin remains missing.  No forward fill,
    backfill, or interpolation is performed.
    """

    if settings is None:
        settings = ValidationSettings()
    source = _prepare_comparison(comparison)
    if "storage" not in source or "outflow" not in source:
        raise ValueError("comparison must contain 'storage' and 'outflow' columns")

    supplied = {
        "raw_inflow": raw_inflow,
        "centered_rolling_inflow": (
            centered_rolling_inflow
            if centered_rolling_inflow is not None
            else trailing_rolling_inflow
        ),
        "estimated_inflow": estimated_inflow,
        "delayed_revised_inflow": delayed_revised_inflow,
        "upstream_flow": upstream_flow,
    }
    aliases = {
        "raw_inflow": ("raw_inflow",),
        "centered_rolling_inflow": (
            "centered_rolling_inflow",
            "trailing_rolling_inflow",
            "rolling_inflow",
        ),
        "estimated_inflow": ("estimated_inflow",),
        "delayed_revised_inflow": (
            "delayed_revised_inflow",
            "revised_inflow",
        ),
        "upstream_flow": ("upstream_flow",),
    }
    base = pd.DataFrame(index=source.index)
    for name, value in supplied.items():
        if value is not None:
            base[name] = _series_on_index(value, source.index, name)
            continue
        source_name = next((alias for alias in aliases[name] if alias in source), None)
        if source_name is not None:
            base[name] = pd.to_numeric(source[source_name], errors="coerce")

    if "raw_inflow" not in base:
        base["raw_inflow"] = _derive_raw_inflow(
            source["storage"],
            source["outflow"],
            source.index,
            flow_to_volume_per_second=settings.flow_to_volume_per_second,
        )
    if "upstream_flow" not in base:
        raise ValueError("comparison must contain or supply 'upstream_flow'")
    for name in ("estimated_inflow", "delayed_revised_inflow"):
        if name not in base:
            original = "revised_inflow" if name == "delayed_revised_inflow" else name
            raise ValueError(f"comparison must contain or supply '{original}'")

    numeric = pd.concat(
        [
            pd.to_numeric(source["storage"], errors="coerce").rename("storage"),
            pd.to_numeric(source["outflow"], errors="coerce").rename("outflow"),
            base,
        ],
        axis=1,
    )
    regular = numeric.resample(settings.evaluation_frequency).mean()
    if "centered_rolling_inflow" not in regular:
        regular["centered_rolling_inflow"] = regular["raw_inflow"].rolling(
            settings.centered_rolling_window, min_periods=1, center=True
        ).mean()

    columns = [
        "storage",
        "outflow",
        "raw_inflow",
        "centered_rolling_inflow",
        "estimated_inflow",
        "delayed_revised_inflow",
        "upstream_flow",
    ]
    result = regular.loc[:, columns]
    result.attrs["revised_inflow_note"] = (
        "delayed_revised_inflow is a fixed-lag diagnostic and is not causal"
    )
    result.attrs["evaluation_frequency"] = settings.evaluation_frequency
    return result


def upstream_proxy_metrics(
    estimate: pd.Series | Iterable[float],
    upstream: pd.Series | Iterable[float],
    *,
    minimum_paired_observations: int = 2,
) -> dict[str, float | int]:
    """Return pairwise-complete agreement metrics against an upstream proxy.

    Coverage is paired finite observations divided by the number of candidate
    timestamps in the aligned inputs.  Pearson and KGE describe agreement with
    the partial-catchment proxy; they do not establish total-inflow accuracy.
    Metrics that are undefined for constant series, near-zero means, or too
    little overlap are returned as ``NaN`` rather than raising.
    """

    minimum = _minimum_integer(
        minimum_paired_observations,
        name="minimum_paired_observations",
        minimum=2,
    )
    estimate_values, upstream_values, candidate_count = _paired_values(
        estimate, upstream
    )
    finite = np.isfinite(estimate_values) & np.isfinite(upstream_values)
    estimate_values = estimate_values[finite]
    upstream_values = upstream_values[finite]
    paired = int(estimate_values.size)
    coverage = _safe_divide(float(paired), float(candidate_count))
    nan_metrics = {
        name: float("nan")
        for name in _METRIC_COLUMNS
        if name not in {"paired_observations", "coverage"}
    }
    result: dict[str, float | int] = {
        "paired_observations": paired,
        "coverage": coverage,
        **nan_metrics,
    }
    if paired < minimum:
        return result

    estimate_mean = float(np.mean(estimate_values))
    upstream_mean = float(np.mean(upstream_values))
    estimate_std = float(np.std(estimate_values, ddof=1))
    upstream_std = float(np.std(upstream_values, ddof=1))
    correlation = _correlation(estimate_values, upstream_values)
    spearman = _correlation(
        _average_ranks(estimate_values), _average_ranks(upstream_values)
    )
    mean_flow_ratio = _safe_ratio(estimate_mean, upstream_mean)
    variability_ratio = _safe_ratio(estimate_std, upstream_std)
    kge = float("nan")
    if (
        np.isfinite(correlation)
        and np.isfinite(mean_flow_ratio)
        and np.isfinite(variability_ratio)
    ):
        kge = float(
            1.0
            - np.sqrt(
                (correlation - 1.0) ** 2
                + (variability_ratio - 1.0) ** 2
                + (mean_flow_ratio - 1.0) ** 2
            )
        )

    residual = estimate_values - upstream_values
    upstream_anomaly = upstream_values - upstream_mean
    result.update(
        {
            "pearson_correlation": correlation,
            "spearman_correlation": spearman,
            "kge": kge,
            "kge_correlation": correlation,
            "kge_variability_ratio": variability_ratio,
            "kge_mean_flow_ratio": mean_flow_ratio,
            "percent_bias": (
                _safe_ratio(
                    float(np.sum(residual)), float(np.sum(upstream_values))
                )
                * 100.0
            ),
            "nse": _safe_ratio(
                float(np.sum(residual**2)), float(np.sum(upstream_anomaly**2))
            ),
            "normalized_rmse": _safe_ratio(
                float(np.sqrt(np.mean(residual**2))), abs(upstream_mean)
            ),
        }
    )
    if np.isfinite(result["nse"]):
        result["nse"] = 1.0 - float(result["nse"])
    return result


def lagged_correlation(
    estimate: pd.Series | Iterable[float],
    upstream: pd.Series | Iterable[float],
    *,
    max_lag_hours: int = 48,
    minimum_paired_observations: int = 2,
    max_lag: int | None = None,
) -> dict[str, object]:
    """Measure Pearson correlation over a symmetric time-lag range.

    A positive lag means that the reservoir estimate responds after the
    upstream gauge: for lag ``+h``, estimate at ``t + h`` is paired with
    upstream at ``t``.  Series with datetime indexes are aligned by timestamp;
    array-like inputs use positional hourly shifts.  Missing values are never
    filled or interpolated.
    """

    if max_lag is not None:
        max_lag_hours = max_lag
    max_lag_hours = _nonnegative_integer(
        max_lag_hours, name="max_lag_hours"
    )
    minimum = _minimum_integer(
        minimum_paired_observations,
        name="minimum_paired_observations",
        minimum=2,
    )
    estimate_series, upstream_series, indexed = _lag_inputs(estimate, upstream)
    lags = range(-max_lag_hours, max_lag_hours + 1)
    correlations: dict[int, float] = {}
    paired_counts: dict[int, int] = {}
    for lag in lags:
        estimate_values, upstream_values = _values_at_lag(
            estimate_series,
            upstream_series,
            lag,
            indexed=indexed,
        )
        finite = np.isfinite(estimate_values) & np.isfinite(upstream_values)
        estimate_values = estimate_values[finite]
        upstream_values = upstream_values[finite]
        paired_counts[lag] = int(estimate_values.size)
        correlations[lag] = (
            _correlation(estimate_values, upstream_values)
            if estimate_values.size >= minimum
            else float("nan")
        )

    curve = pd.Series(correlations, dtype=float)
    curve.index.name = "lag_hours"
    zero = float(curve.loc[0])
    valid_lags = [lag for lag, value in correlations.items() if np.isfinite(value)]
    if not valid_lags:
        best_lag: int | float = float("nan")
        maximum = float("nan")
        paired_at_best = 0
    else:
        best_lag = min(
            valid_lags,
            key=lambda lag: (-correlations[lag], abs(lag), lag),
        )
        maximum = float(correlations[best_lag])
        paired_at_best = paired_counts[best_lag]

    improvement = (
        maximum - zero if np.isfinite(maximum) and np.isfinite(zero) else float("nan")
    )
    return {
        "zero_lag_pearson_correlation": zero,
        "max_correlation": maximum,
        "maximum_correlation": maximum,
        "best_lag_hours": best_lag,
        "paired_observations_at_best_lag": paired_at_best,
        "improvement_over_zero_lag": improvement,
        "correlation_by_lag": curve,
        "paired_observations_by_lag": pd.Series(paired_counts, dtype=int),
    }


def inflow_behavior_metrics(
    estimate: pd.Series | Iterable[float],
    *,
    evaluation_frequency: str | pd.Timedelta = "1h",
) -> dict[str, float | int]:
    """Return finite-count, negative-inflow, and adjacent-change metrics.

    ``negative_hour_frequency_percent`` is the percentage of finite estimates
    below zero. ``mean_negative_inflow_cfs`` is the mean magnitude of those
    negative estimates, in cfs. ``mean_absolute_hourly_change_cfs`` is the
    mean absolute change between adjacent finite estimates, in cfs. Missing
    and nonfinite values are excluded from the finite count and break the
    adjacency sequence; a finite estimate before a gap is never compared with
    a finite estimate after it. ``evaluation_frequency`` is validated for API
    compatibility; callers should use this helper on the regular hourly frame
    produced by :func:`build_validation_frame` when interpreting the hourly
    change metric.

    The frequency is ``NaN`` when no finite estimates are available. The mean
    negative inflow is ``NaN`` when there are no negative estimates, and the
    mean absolute change is ``NaN`` when there are fewer than two adjacent
    finite estimates.
    """

    _positive_timedelta(evaluation_frequency, name="evaluation_frequency")
    values = _inflow_values(estimate)
    finite = np.isfinite(values)
    finite_count = int(np.sum(finite))

    if finite_count:
        negative = finite & (values < 0.0)
        negative_count = int(np.sum(negative))
        negative_frequency = float(negative_count / finite_count * 100.0)
        mean_negative = (
            float(np.mean(np.abs(values[negative])))
            if negative_count
            else float("nan")
        )
    else:
        negative_frequency = float("nan")
        mean_negative = float("nan")

    adjacent = finite[:-1] & finite[1:]
    changes = np.abs(values[1:][adjacent] - values[:-1][adjacent])
    mean_absolute_change = (
        float(np.mean(changes)) if changes.size else float("nan")
    )

    return {
        "finite_observations": finite_count,
        "negative_hour_frequency_percent": negative_frequency,
        "mean_negative_inflow_cfs": mean_negative,
        "mean_absolute_hourly_change_cfs": mean_absolute_change,
    }


def storage_closure_metrics(
    estimate: pd.Series | Iterable[float],
    storage: pd.Series | Iterable[float],
    outflow: pd.Series | Iterable[float],
    *,
    flow_to_volume_per_second: float = CFS_TO_ACRE_FEET_PER_SECOND,
    evaluation_frequency: str | pd.Timedelta = "1h",
) -> dict[str, float | int]:
    """Return one-step storage-closure errors for one inflow estimate.

    For each adjacent regular interval, the predicted storage change is
    ``(estimate[t] - outflow[t]) * duration`` and is compared with
    ``storage[t] - storage[t - 1]``.  The returned residual is predicted
    change minus observed storage change.  Coverage is valid one-step pairs
    divided by all possible adjacent steps.  Missing values are handled
    pairwise and are not interpolated.
    """

    factor = float(flow_to_volume_per_second)
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError("flow_to_volume_per_second must be positive and finite")
    frequency = _positive_timedelta(evaluation_frequency, name="evaluation_frequency")
    estimate_values, storage_values, outflow_values, count, elapsed = _triple_values(
        estimate, storage, outflow, frequency
    )
    possible = max(count - 1, 0)
    result: dict[str, float | int] = {
        "paired_observations": 0,
        "coverage": _safe_divide(0.0, float(possible)),
        "rmse": float("nan"),
        "mae": float("nan"),
        "bias": float("nan"),
        "rmse_cfs": float("nan"),
        "mae_cfs": float("nan"),
        "bias_cfs": float("nan"),
    }
    if possible == 0:
        return result

    observed_change = storage_values[1:] - storage_values[:-1]
    predicted_change = (
        estimate_values[1:] - outflow_values[1:]
    ) * elapsed[1:] * factor
    valid = (
        np.isfinite(observed_change)
        & np.isfinite(predicted_change)
        & (elapsed[1:] > 0.0)
    )
    residual = predicted_change[valid] - observed_change[valid]
    elapsed_valid = elapsed[1:][valid]
    paired = int(residual.size)
    result.update(
        {
            "paired_observations": paired,
            "coverage": _safe_divide(float(paired), float(possible)),
        }
    )
    if not paired:
        return result
    residual_cfs = residual / (elapsed_valid * factor)
    result.update(
        {
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "bias": float(np.mean(residual)),
            "rmse_cfs": float(np.sqrt(np.mean(residual_cfs**2))),
            "mae_cfs": float(np.mean(np.abs(residual_cfs))),
            "bias_cfs": float(np.mean(residual_cfs)),
        }
    )
    return result


def generate_validation_outputs(
    comparison: pd.DataFrame,
    *,
    settings: ValidationSettings | None = None,
    make_plots: bool = False,
) -> ValidationOutputs:
    """Generate proxy, lag, and storage-closure validation outputs.

    If a training window is configured, the best lag for each estimate is
    selected only from that window.  The selected lag is then evaluated on the
    held-out evaluation window without re-selection.  ``make_plots=True``
    returns interactive Plotly figures.
    """

    if settings is None:
        settings = ValidationSettings()
    frame = build_validation_frame(comparison, settings=settings)
    training = _window(frame, settings, training=True)
    evaluation = _window(frame, settings, training=False)
    estimate_names = list(_ESTIMATE_COLUMNS)

    agreement_rows: list[dict[str, object]] = []
    lag_rows: list[dict[str, object]] = []
    closure_rows: list[dict[str, object]] = []
    behavior_rows: list[dict[str, object]] = []
    lag_results: dict[str, dict[str, object]] = {}
    train_lag_results: dict[str, dict[str, object]] = {}
    for name in estimate_names:
        display_name = _DISPLAY_NAMES[name]
        agreement = upstream_proxy_metrics(
            evaluation[name],
            evaluation["upstream_flow"],
            minimum_paired_observations=settings.minimum_paired_observations,
        )
        agreement_rows.append({"estimate": display_name, **agreement})

        evaluation_lag = lagged_correlation(
            evaluation[name],
            evaluation["upstream_flow"],
            max_lag_hours=settings.cross_correlation_range,
            minimum_paired_observations=settings.minimum_paired_observations,
        )
        lag_results[name] = evaluation_lag
        train_lag = None
        if not training.empty:
            train_lag = lagged_correlation(
                training[name],
                training["upstream_flow"],
                max_lag_hours=settings.cross_correlation_range,
                minimum_paired_observations=settings.minimum_paired_observations,
            )
            train_lag_results[name] = train_lag

        if train_lag is not None and np.isfinite(train_lag["best_lag_hours"]):
            selected_lag = int(train_lag["best_lag_hours"])
            selected_corr, selected_count = _lag_value(
                evaluation_lag, selected_lag
            )
            zero = float(evaluation_lag["zero_lag_pearson_correlation"])
            row = {
                "estimate": display_name,
                "zero_lag_pearson_correlation": zero,
                "max_correlation": selected_corr,
                "best_lag_hours": selected_lag,
                "paired_observations_at_best_lag": selected_count,
                "improvement_over_zero_lag": (
                    selected_corr - zero
                    if np.isfinite(selected_corr) and np.isfinite(zero)
                    else float("nan")
                ),
                "lag_selection": "training (frozen for evaluation)",
            }
        else:
            row = {
                "estimate": display_name,
                "zero_lag_pearson_correlation": evaluation_lag[
                    "zero_lag_pearson_correlation"
                ],
                "max_correlation": evaluation_lag["max_correlation"],
                "best_lag_hours": evaluation_lag["best_lag_hours"],
                "paired_observations_at_best_lag": evaluation_lag[
                    "paired_observations_at_best_lag"
                ],
                "improvement_over_zero_lag": evaluation_lag[
                    "improvement_over_zero_lag"
                ],
                "lag_selection": "evaluation",
            }
        lag_rows.append(row)

        closure = storage_closure_metrics(
            evaluation[name],
            evaluation["storage"],
            evaluation["outflow"],
            flow_to_volume_per_second=settings.flow_to_volume_per_second,
            evaluation_frequency=settings.evaluation_frequency,
        )
        closure_rows.append({"estimate": display_name, **closure})
        behavior = inflow_behavior_metrics(
            evaluation[name],
            evaluation_frequency=settings.evaluation_frequency,
        )
        behavior_rows.append({"estimate": display_name, **behavior})

    agreement_table = pd.DataFrame(agreement_rows).set_index("estimate")
    lag_table = pd.DataFrame(lag_rows).set_index("estimate")
    closure_table = pd.DataFrame(closure_rows).set_index("estimate")
    behavior_table = pd.DataFrame(behavior_rows).set_index("estimate")

    cross_plot = None
    scatter_plot = None
    if make_plots:
        cross_plot = plot_cross_correlation(lag_results)
        scatter_plot = plot_estimate_upstream_scatter(evaluation)

    return ValidationOutputs(
        validation_frame=frame,
        upstream_proxy_agreement=agreement_table,
        best_lag_summary=lag_table,
        storage_closure=closure_table,
        inflow_behavior=behavior_table,
        cross_correlation_plot=cross_plot,
        estimate_upstream_scatter_plot=scatter_plot,
    )


def plot_cross_correlation(
    lag_results: Mapping[str, Mapping[str, object]],
) -> Any:
    """Return an interactive Plotly cross-correlation-versus-lag chart."""

    go = _plotly_graph_objects()
    figure = go.Figure()
    for name, result in lag_results.items():
        curve = result["correlation_by_lag"]
        assert isinstance(curve, pd.Series)
        figure.add_trace(
            go.Scatter(
                x=curve.index.to_numpy(),
                y=curve.to_numpy(),
                mode="lines+markers",
                name=_DISPLAY_NAMES.get(name, name),
                connectgaps=False,
                hovertemplate=(
                    "Lag: %{x} h<br>Pearson: %{y:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.add_vline(x=0, line_width=1, line_dash="dot", line_color="gray")
    figure.update_layout(
        title="Cross-correlation versus lag",
        template="plotly_white",
        xaxis_title="Lag (hours; positive means estimate responds later)",
        yaxis_title="Pearson correlation",
        hovermode="x unified",
        height=520,
        legend={"orientation": "v", "x": 1.02, "xanchor": "left"},
    )
    return figure


def plot_estimate_upstream_scatter(
    validation_frame: pd.DataFrame,
) -> Any:
    """Return an interactive Plotly estimate-versus-upstream scatter chart."""

    go = _plotly_graph_objects()
    figure = go.Figure()
    upstream = pd.to_numeric(validation_frame["upstream_flow"], errors="coerce")
    for name in _ESTIMATE_COLUMNS:
        estimate = pd.to_numeric(validation_frame[name], errors="coerce")
        valid = np.isfinite(estimate.to_numpy()) & np.isfinite(upstream.to_numpy())
        if valid.any():
            figure.add_trace(
                go.Scattergl(
                    x=upstream.to_numpy()[valid],
                    y=estimate.to_numpy()[valid],
                    mode="markers",
                    name=_DISPLAY_NAMES[name],
                    marker={"size": 6, "opacity": 0.55},
                    hovertemplate=(
                        "Upstream: %{x:,.2f}<br>Estimate: %{y:,.2f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    figure.update_layout(
        title="Inflow estimates versus upstream proxy",
        template="plotly_white",
        xaxis_title="Upstream flow proxy",
        yaxis_title="Inflow estimate",
        hovermode="closest",
        height=600,
        legend={"orientation": "v", "x": 1.02, "xanchor": "left"},
    )
    return figure


def _prepare_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(comparison, pd.DataFrame):
        raise TypeError("comparison must be a pandas DataFrame")
    if not isinstance(comparison.index, pd.DatetimeIndex):
        raise TypeError("comparison must have a pandas DateTimeIndex")
    if comparison.index.tz is None:
        raise ValueError("comparison index must be timezone-aware")
    if not comparison.index.is_monotonic_increasing:
        raise ValueError("comparison index must be increasing")
    if comparison.index.has_duplicates:
        raise ValueError("comparison index must not contain duplicates")
    result = comparison.copy()
    result.index = result.index.tz_convert("UTC")
    return result


def _series_on_index(
    value: pd.Series | Iterable[float], index: pd.DatetimeIndex, name: str
) -> pd.Series:
    if isinstance(value, pd.Series):
        series = value.copy()
        if isinstance(series.index, pd.DatetimeIndex):
            if series.index.tz is None:
                raise ValueError(f"{name} index must be timezone-aware")
            series.index = series.index.tz_convert("UTC")
            return pd.to_numeric(series.reindex(index), errors="coerce")
        if len(series) != len(index):
            raise ValueError(f"{name} must have the same length as comparison")
        series = pd.Series(series.to_numpy(), index=index, name=name)
        return pd.to_numeric(series, errors="coerce")
    values = np.asarray(list(value), dtype=float)
    if values.ndim != 1 or len(values) != len(index):
        raise ValueError(f"{name} must have the same length as comparison")
    return pd.Series(values, index=index, name=name)


def _derive_raw_inflow(
    storage: pd.Series,
    outflow: pd.Series,
    index: pd.DatetimeIndex,
    *,
    flow_to_volume_per_second: float,
) -> pd.Series:
    storage_values = pd.to_numeric(storage, errors="coerce")
    outflow_values = pd.to_numeric(outflow, errors="coerce")
    elapsed = index.to_series().diff().dt.total_seconds()
    raw = (
        storage_values.diff() / (elapsed * flow_to_volume_per_second)
        + outflow_values.shift(1)
    )
    return raw.rename("raw_inflow")


def _paired_values(
    estimate: pd.Series | Iterable[float], upstream: pd.Series | Iterable[float]
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(estimate, pd.Series) and isinstance(upstream, pd.Series):
        left = _numeric_series(estimate, "estimate")
        right = _numeric_series(upstream, "upstream")
        joined = pd.concat([left.rename("estimate"), right.rename("upstream")], axis=1)
        return (
            joined["estimate"].to_numpy(dtype=float),
            joined["upstream"].to_numpy(dtype=float),
            len(joined),
        )
    left = np.asarray(list(estimate), dtype=float)
    right = np.asarray(list(upstream), dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
        raise ValueError(
            "estimate and upstream must be equal-length one-dimensional data"
        )
    return left, right, len(left)


def _lag_inputs(
    estimate: pd.Series | Iterable[float], upstream: pd.Series | Iterable[float]
) -> tuple[pd.Series | np.ndarray, pd.Series | np.ndarray, bool]:
    if isinstance(estimate, pd.Series) and isinstance(upstream, pd.Series):
        left = _numeric_series(estimate, "estimate")
        right = _numeric_series(upstream, "upstream")
        if isinstance(left.index, pd.DatetimeIndex) and isinstance(
            right.index, pd.DatetimeIndex
        ):
            if left.index.tz is None or right.index.tz is None:
                raise ValueError("lagged-correlation indexes must be timezone-aware")
            left.index = left.index.tz_convert("UTC")
            right.index = right.index.tz_convert("UTC")
            return left, right, True
    left = np.asarray(list(estimate), dtype=float)
    right = np.asarray(list(upstream), dtype=float)
    if left.ndim != 1 or right.ndim != 1 or len(left) != len(right):
        raise ValueError(
            "estimate and upstream must be equal-length one-dimensional data"
        )
    return left, right, False


def _values_at_lag(
    estimate: pd.Series | np.ndarray,
    upstream: pd.Series | np.ndarray,
    lag: int,
    *,
    indexed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if indexed:
        assert isinstance(estimate, pd.Series)
        assert isinstance(upstream, pd.Series)
        shifted_index = upstream.index + pd.Timedelta(hours=lag)
        left = estimate.reindex(shifted_index).to_numpy(dtype=float)
        right = upstream.to_numpy(dtype=float)
        return left, right
    assert isinstance(estimate, np.ndarray)
    assert isinstance(upstream, np.ndarray)
    if lag > 0:
        return estimate[lag:], upstream[:-lag]
    if lag < 0:
        return estimate[:lag], upstream[-lag:]
    return estimate, upstream


def _triple_values(
    estimate: pd.Series | Iterable[float],
    storage: pd.Series | Iterable[float],
    outflow: pd.Series | Iterable[float],
    frequency: pd.Timedelta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]:
    if all(isinstance(value, pd.Series) for value in (estimate, storage, outflow)):
        left = _numeric_series(estimate, "estimate")
        middle = _numeric_series(storage, "storage")
        right = _numeric_series(outflow, "outflow")
        joined = pd.concat(
            [
                left.rename("estimate"),
                middle.rename("storage"),
                right.rename("outflow"),
            ],
            axis=1,
        )
        if isinstance(joined.index, pd.DatetimeIndex):
            elapsed = joined.index.to_series().diff().dt.total_seconds().to_numpy(
                dtype=float
            )
        else:
            elapsed = np.full(len(joined), frequency.total_seconds(), dtype=float)
        return (
            joined["estimate"].to_numpy(dtype=float),
            joined["storage"].to_numpy(dtype=float),
            joined["outflow"].to_numpy(dtype=float),
            len(joined),
            elapsed,
        )
    values = [
        np.asarray(list(value), dtype=float)
        for value in (estimate, storage, outflow)
    ]
    if any(value.ndim != 1 for value in values) or len(
        {len(value) for value in values}
    ) != 1:
        raise ValueError(
            "estimate, storage, and outflow must be equal-length one-dimensional data"
        )
    count = len(values[0])
    elapsed = np.full(count, frequency.total_seconds(), dtype=float)
    return values[0], values[1], values[2], count, elapsed


def _inflow_values(estimate: pd.Series | Iterable[float]) -> np.ndarray:
    if isinstance(estimate, pd.Series):
        series = _numeric_series(estimate, "estimate")
        return series.to_numpy(dtype=float)

    values = np.asarray(list(estimate), dtype=float)
    if values.ndim != 1:
        raise ValueError("estimate must be a one-dimensional data series")
    return values


def _numeric_series(series: pd.Series, name: str) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce").copy()
    if isinstance(result.index, pd.DatetimeIndex) and result.index.tz is not None:
        result.index = result.index.tz_convert("UTC")
    result.name = name
    return result


def _lag_value(result: Mapping[str, object], lag: int) -> tuple[float, int]:
    curve = result["correlation_by_lag"]
    counts = result["paired_observations_by_lag"]
    assert isinstance(curve, pd.Series)
    assert isinstance(counts, pd.Series)
    if lag not in curve.index:
        return float("nan"), 0
    return float(curve.loc[lag]), int(counts.loc[lag])


def _window(
    frame: pd.DataFrame,
    settings: ValidationSettings,
    *,
    training: bool,
) -> pd.DataFrame:
    if training:
        start, end = settings.training_start, settings.training_end
    else:
        start, end = settings.evaluation_start, settings.evaluation_end
        if start is None and end is None and settings.training_end is not None:
            start = settings.training_end + settings.evaluation_frequency
    if start is None and end is None:
        return frame
    mask = np.ones(len(frame), dtype=bool)
    if start is not None:
        mask &= frame.index >= start
    if end is not None:
        mask &= frame.index <= end
    return frame.loc[mask]


def _window_bounds(
    settings: ValidationSettings,
    start_name: str,
    end_name: str,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    return getattr(settings, start_name), getattr(settings, end_name)


def _validate_bounds(
    bounds: Mapping[str, pd.Timestamp | None], start_name: str, end_name: str
) -> None:
    start, end = bounds[start_name], bounds[end_name]
    if start is not None and end is not None and start > end:
        raise ValueError(f"{start_name} must be earlier than or equal to {end_name}")


def _positive_timedelta(value: object, *, name: str) -> pd.Timedelta:
    try:
        result = pd.Timedelta(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive duration") from error
    if pd.isna(result) or result <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive duration")
    return result


def _utc_timestamp(value: object, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _nonnegative_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if result != value:
        raise TypeError(f"{name} must be an integer")
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _minimum_integer(value: object, *, name: str, minimum: int) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if not np.isfinite(denominator) or denominator <= np.finfo(float).eps:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")
    scale = max(1.0, abs(numerator), abs(denominator))
    if abs(denominator) <= np.finfo(float).eps * scale:
        return float("nan")
    return numerator / denominator


def _safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _plotly_graph_objects() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting validation outputs requires the optional plotly dependency"
        ) from error
    return go


__all__ = [
    "ValidationOutputs",
    "ValidationSettings",
    "build_validation_frame",
    "generate_validation_outputs",
    "inflow_behavior_metrics",
    "lagged_correlation",
    "plot_cross_correlation",
    "plot_estimate_upstream_scatter",
    "storage_closure_metrics",
    "upstream_proxy_metrics",
]
