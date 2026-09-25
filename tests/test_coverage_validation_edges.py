"""Exercise validation-module contracts that need deliberate edge inputs."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Notebooks import validation  # noqa: E402


def _index(count: int = 8) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")


def _complete_comparison(count: int = 8) -> pd.DataFrame:
    index = _index(count)
    values = np.arange(1.0, count + 1.0)
    return pd.DataFrame(
        {
            "storage": 100.0 + values,
            "outflow": np.ones(count),
            "raw_inflow": values,
            "centered_rolling_inflow": values + 0.5,
            "estimated_inflow": values + 1.0,
            "revised_inflow": values + 1.5,
            "upstream_flow": values,
        },
        index=index,
    )


def test_validation_settings_aliases_and_invalid_options() -> None:
    settings = validation.ValidationSettings(
        trailing_rolling_window="2h",
        training_start="2025-01-01T00:00:00-08:00",
        training_end="2025-01-02T00:00:00-08:00",
        evaluation_start="2025-01-03T00:00:00-08:00",
        evaluation_end="2025-01-04T00:00:00-08:00",
    )
    assert settings.frequency == pd.Timedelta("1h")
    assert settings.max_lag_hours == 48
    assert settings.trailing_rolling_window == pd.Timedelta("2h")
    assert str(settings.training_start.tz) == "UTC"

    with pytest.raises(ValueError, match="positive and finite"):
        validation.ValidationSettings(flow_to_volume_per_second=0)
    with pytest.raises(ValueError, match="earlier than or equal"):
        validation.ValidationSettings(
            training_start="2025-01-02T00:00:00Z",
            training_end="2025-01-01T00:00:00Z",
        )


def test_build_validation_frame_input_aliases_and_missing_columns() -> None:
    comparison = _complete_comparison()
    # Explicit values exercise the Series-on-index path.  The old trailing
    # keyword remains accepted when the centered value is omitted.
    index = comparison.index
    frame = validation.build_validation_frame(
        comparison.drop(
            columns=["raw_inflow", "centered_rolling_inflow", "revised_inflow"]
        ),
        raw_inflow=pd.Series(np.arange(8.0), index=index),
        trailing_rolling_inflow=np.arange(8.0) + 0.25,
        estimated_inflow=pd.Series(np.arange(8.0), index=index),
        delayed_revised_inflow=np.arange(8.0) + 0.5,
        upstream_flow=pd.Series(np.arange(8.0), index=index),
    )
    assert frame.attrs["evaluation_frequency"] == pd.Timedelta("1h")
    assert list(frame.columns) == [
        "storage",
        "outflow",
        "raw_inflow",
        "centered_rolling_inflow",
        "estimated_inflow",
        "delayed_revised_inflow",
        "upstream_flow",
    ]

    with pytest.raises(ValueError, match="storage.*outflow"):
        validation.build_validation_frame(
            comparison.drop(columns=["outflow"]),
            upstream_flow=np.ones(8),
            estimated_inflow=np.ones(8),
            delayed_revised_inflow=np.ones(8),
        )
    with pytest.raises(ValueError, match="upstream_flow"):
        validation.build_validation_frame(
            comparison.drop(columns=["raw_inflow", "upstream_flow"]),
            estimated_inflow=np.ones(8),
            delayed_revised_inflow=np.ones(8),
        )
    with pytest.raises(ValueError, match="estimated_inflow"):
        validation.build_validation_frame(
            comparison.drop(
                columns=["raw_inflow", "estimated_inflow", "revised_inflow"]
            ),
            upstream_flow=np.ones(8),
        )


def test_proxy_and_lag_metrics_cover_low_overlap_and_no_valid_lag() -> None:
    low_overlap = validation.upstream_proxy_metrics(
        [1.0, np.nan], [1.0, 2.0], minimum_paired_observations=3
    )
    assert low_overlap["paired_observations"] == 1
    assert np.isnan(low_overlap["pearson_correlation"])

    no_lag = validation.lagged_correlation(
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        max_lag=1,
        minimum_paired_observations=2,
    )
    assert np.isnan(no_lag["best_lag_hours"])
    assert no_lag["paired_observations_at_best_lag"] == 0


def test_storage_closure_invalid_empty_and_unpaired_inputs() -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        validation.storage_closure_metrics(
            [1.0], [1.0], [1.0], flow_to_volume_per_second=0
        )
    empty = validation.storage_closure_metrics([], [], [])
    assert empty["paired_observations"] == 0
    assert np.isnan(empty["coverage"])
    unpaired = validation.storage_closure_metrics(
        [np.nan, np.nan], [1.0, 1.0], [1.0, 1.0]
    )
    assert unpaired["paired_observations"] == 0
    assert np.isnan(unpaired["rmse"])


def test_generate_outputs_supports_default_settings_plots_and_empty_training() -> None:
    comparison = _complete_comparison(12)
    plotted = validation.generate_validation_outputs(comparison, make_plots=True)
    assert plotted.cross_correlation_plot is not None
    assert plotted.estimate_upstream_scatter_plot is not None
    assert plotted.inflow_behavior_metrics is plotted.inflow_behavior

    index = comparison.index
    empty_training = validation.ValidationSettings(
        training_start=index[-1] + pd.Timedelta(days=1),
        training_end=index[-1] + pd.Timedelta(days=2),
        evaluation_start=index[0],
        evaluation_end=index[-1],
        cross_correlation_range=1,
        minimum_paired_observations=2,
    )
    outputs = validation.generate_validation_outputs(
        comparison, settings=empty_training
    )
    assert set(outputs.best_lag_summary["lag_selection"]) == {"evaluation"}


def test_plot_helpers_handle_known_and_missing_points() -> None:
    comparison = _complete_comparison(4)
    comparison["upstream_flow"] = [1.0, np.nan, np.nan, np.nan]
    comparison["raw_inflow"] = [1.0, np.nan, np.nan, np.nan]
    comparison["centered_rolling_inflow"] = [np.nan] * 4
    comparison["estimated_inflow"] = [np.nan] * 4
    comparison["revised_inflow"] = [np.nan] * 4
    frame = validation.build_validation_frame(comparison)
    scatter = validation.plot_estimate_upstream_scatter(frame)
    assert len(scatter.data) == 1

    lag = validation.lagged_correlation(
        [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], max_lag_hours=1
    )
    figure = validation.plot_cross_correlation({"unknown": lag})
    assert len(figure.data) == 1


def test_validation_private_input_normalization_and_windows() -> None:
    index = _index(3)
    valid_frame = pd.DataFrame(
        {"storage": [1, 2, 3], "outflow": [1, 1, 1]}, index=index
    )
    with pytest.raises(TypeError, match="DataFrame"):
        validation._prepare_comparison([1, 2, 3])
    with pytest.raises(TypeError, match="DateTimeIndex"):
        validation._prepare_comparison(
            pd.DataFrame(valid_frame.to_numpy(), index=[0, 1, 2])
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        validation._prepare_comparison(
            pd.DataFrame(
                valid_frame.to_numpy(),
                columns=valid_frame.columns,
                index=pd.date_range("2025-01-01", periods=3, freq="h"),
            )
        )
    with pytest.raises(ValueError, match="increasing"):
        validation._prepare_comparison(valid_frame.iloc[::-1])
    with pytest.raises(ValueError, match="duplicates"):
        validation._prepare_comparison(
            valid_frame.set_axis([index[0], index[0], index[2]])
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        validation._series_on_index(
            pd.Series(
                [1, 2, 3], index=pd.date_range("2025-01-01", periods=3, freq="h")
            ),
            index,
            "series",
        )
    with pytest.raises(ValueError, match="same length"):
        validation._series_on_index(pd.Series([1, 2]), index, "series")
    with pytest.raises(ValueError, match="same length"):
        validation._series_on_index([1, 2], index, "series")
    with pytest.raises(ValueError, match="same length"):
        validation._series_on_index([[1, 2], [3, 4], [5, 6]], index, "series")
    assert validation._series_on_index(
        pd.Series([1, 2, 3]), index, "series"
    ).index.equals(index)
    assert validation._series_on_index([1, 2, 3], index, "series").iloc[-1] == 3

    with pytest.raises(ValueError, match="equal-length"):
        validation._paired_values([1, 2], [1])
    with pytest.raises(ValueError, match="equal-length"):
        validation._lag_inputs([1, 2], [1])
    with pytest.raises(ValueError, match="timezone-aware"):
        validation._lag_inputs(
            pd.Series([1, 2], index=pd.date_range("2025-01-01", periods=2, freq="h")),
            pd.Series([1, 2], index=pd.date_range("2025-01-01", periods=2, freq="h")),
        )
    # A datetime estimate paired with a positional upstream series takes the
    # short-circuit branch where only the first index is datetime-like.
    mixed = validation.lagged_correlation(
        pd.Series([1, 2, 3], index=index),
        pd.Series([1, 2, 3]),
        max_lag_hours=1,
    )
    assert mixed["best_lag_hours"] == 0

    arrays = validation.lagged_correlation([1, 2, 3, 4], [1, 2, 3, 4], max_lag_hours=2)
    assert arrays["best_lag_hours"] == 0
    missing_lag = validation._lag_value(arrays, 999)
    assert np.isnan(missing_lag[0]) and missing_lag[1] == 0


def test_validation_private_storage_and_window_helpers() -> None:
    frequency = pd.Timedelta("1h")
    estimate = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    storage = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    outflow = pd.Series([0.0, 0.0], index=pd.RangeIndex(2))
    values = validation._triple_values(estimate, storage, outflow, frequency)
    assert np.allclose(values[4], [3600.0, 3600.0], equal_nan=True)
    with pytest.raises(ValueError, match="equal-length"):
        validation._triple_values([1, 2], [1], [1, 2], frequency)
    with pytest.raises(ValueError, match="one-dimensional"):
        validation._inflow_values([[1.0, 2.0]])

    aware = pd.Series(
        [1, 2], index=pd.date_range("2025-01-01", periods=2, freq="h", tz="US/Pacific")
    )
    assert str(validation._numeric_series(aware, "x").index.tz) == "UTC"

    frame = pd.DataFrame(index=_index(3))
    base = validation.ValidationSettings()
    assert validation._window(frame, base, training=True).equals(frame)
    training_end = validation.ValidationSettings(training_end=_index(3)[0])
    evaluation = validation._window(frame, training_end, training=False)
    assert evaluation.empty
    only_start = validation.ValidationSettings(evaluation_start=_index(3)[1])
    assert len(validation._window(frame, only_start, training=False)) == 2
    only_end = validation.ValidationSettings(evaluation_end=_index(3)[1])
    assert len(validation._window(frame, only_end, training=False)) == 2
    assert validation._window_bounds(base, "training_start", "training_end") == (
        None,
        None,
    )
    with pytest.raises(ValueError, match="earlier than or equal"):
        validation._validate_bounds(
            {
                "start": pd.Timestamp("2025-01-02", tz="UTC"),
                "end": pd.Timestamp("2025-01-01", tz="UTC"),
            },
            "start",
            "end",
        )


def test_validation_scalar_helpers_and_optional_plotly_error(monkeypatch) -> None:
    with pytest.raises(ValueError, match="positive duration"):
        validation._positive_timedelta("not-a-duration", name="duration")
    with pytest.raises(ValueError, match="positive duration"):
        validation._positive_timedelta(0, name="duration")
    with pytest.raises(ValueError, match="timezone-aware"):
        validation._utc_timestamp("2025-01-01", "when")
    with pytest.raises(TypeError, match="integer"):
        validation._nonnegative_integer(True, name="value")
    with pytest.raises(TypeError, match="integer"):
        validation._nonnegative_integer("bad", name="value")
    with pytest.raises(TypeError, match="integer"):
        validation._nonnegative_integer(1.5, name="value")
    with pytest.raises(ValueError, match="nonnegative"):
        validation._nonnegative_integer(-1, name="value")
    with pytest.raises(ValueError, match="at least 2"):
        validation._minimum_integer(1, name="value", minimum=2)
    assert np.isnan(validation._correlation(np.array([1.0]), np.array([1.0])))
    assert np.isnan(validation._safe_ratio(1.0, np.nan))
    assert np.isnan(validation._safe_divide(1.0, 0.0))

    original_import = builtins.__import__

    def fail_plotly(name, *args, **kwargs):
        if name == "plotly.graph_objects":
            raise ImportError("plotly unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_plotly)
    with pytest.raises(ImportError, match="optional plotly"):
        validation._plotly_graph_objects()
