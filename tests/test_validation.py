import sys
from datetime import UTC
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, str(Path(__file__).parents[1] / "Notebooks")
)

from validation import (  # noqa: E402
    ValidationSettings,
    build_validation_frame,
    generate_validation_outputs,
    lagged_correlation,
    storage_closure_metrics,
    upstream_proxy_metrics,
)


def _hourly_index(count: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=count, freq="h", tz=UTC)


def test_upstream_proxy_metrics_perfect_agreement() -> None:
    index = _hourly_index(5)
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)

    result = upstream_proxy_metrics(values, values)

    assert result["paired_observations"] == 5
    assert result["coverage"] == pytest.approx(1.0)
    assert result["pearson_correlation"] == pytest.approx(1.0)
    assert result["spearman_correlation"] == pytest.approx(1.0)
    assert result["kge"] == pytest.approx(1.0)
    assert result["kge_correlation"] == pytest.approx(1.0)
    assert result["kge_variability_ratio"] == pytest.approx(1.0)
    assert result["kge_mean_flow_ratio"] == pytest.approx(1.0)
    assert result["percent_bias"] == pytest.approx(0.0)
    assert result["nse"] == pytest.approx(1.0)
    assert result["normalized_rmse"] == pytest.approx(0.0)


def test_lagged_correlation_uses_positive_lag_for_delayed_estimate() -> None:
    index = _hourly_index(9)
    upstream = pd.Series([2.0, 5.0, 1.0, 7.0, 3.0, 8.0, 4.0, 6.0, 9.0], index=index)
    estimate = upstream.shift(1)

    result = lagged_correlation(
        estimate,
        upstream,
        max_lag_hours=2,
        minimum_paired_observations=5,
    )

    assert result["best_lag_hours"] == 1
    assert result["max_correlation"] == pytest.approx(1.0)
    assert result["paired_observations_at_best_lag"] == 8
    assert result["improvement_over_zero_lag"] > 0.0


def test_metrics_are_safe_for_missing_and_constant_series() -> None:
    result = upstream_proxy_metrics(
        [1.0, np.nan, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
    )
    assert result["paired_observations"] == 3
    assert result["coverage"] == pytest.approx(0.75)

    constant = upstream_proxy_metrics([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
    assert np.isnan(constant["pearson_correlation"])
    assert np.isnan(constant["spearman_correlation"])
    assert np.isnan(constant["kge"])

    near_zero = upstream_proxy_metrics([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert np.isnan(near_zero["percent_bias"])
    assert np.isnan(near_zero["normalized_rmse"])


def test_validation_frame_is_hourly_and_does_not_fill_revised_tail() -> None:
    index = pd.to_datetime(
        [
            "2024-01-01 00:00Z",
            "2024-01-01 00:30Z",
            "2024-01-01 01:00Z",
            "2024-01-01 02:30Z",
        ]
    )
    comparison = pd.DataFrame(
        {
            "storage": [100.0, 101.0, 102.0, 103.0],
            "outflow": [1.0, 1.0, 1.0, 1.0],
            "raw_inflow": [2.0, 4.0, 6.0, 8.0],
            "estimated_inflow": [2.0, 4.0, 6.0, 8.0],
            "revised_inflow": [2.0, 4.0, np.nan, np.nan],
            "upstream_flow": [2.0, 4.0, 6.0, 8.0],
        },
        index=index,
    )

    frame = build_validation_frame(
        comparison,
        settings=ValidationSettings(centered_rolling_window="2h"),
    )

    assert frame.index.freq == pd.Timedelta("1h")
    assert list(frame.columns) == [
        "storage",
        "outflow",
        "raw_inflow",
        "centered_rolling_inflow",
        "estimated_inflow",
        "delayed_revised_inflow",
        "upstream_flow",
    ]
    assert frame.loc[index[0], "raw_inflow"] == pytest.approx(3.0)
    assert pd.isna(frame.loc[index[2], "delayed_revised_inflow"])
    assert frame.attrs["revised_inflow_note"].startswith("delayed_revised_inflow")


def test_validation_frame_uses_a_centered_rolling_mean() -> None:
    index = _hourly_index(5)
    comparison = pd.DataFrame(
        {
            "storage": np.arange(5, dtype=float),
            "outflow": np.ones(5),
            "raw_inflow": [0.0, 0.0, 100.0, 0.0, 0.0],
            "estimated_inflow": np.ones(5),
            "revised_inflow": np.ones(5),
            "upstream_flow": np.ones(5),
        },
        index=index,
    )

    frame = build_validation_frame(
        comparison,
        settings=ValidationSettings(centered_rolling_window="3h"),
    )

    assert frame.loc[index[1], "centered_rolling_inflow"] == pytest.approx(100.0 / 3.0)
    assert frame.loc[index[3], "centered_rolling_inflow"] == pytest.approx(100.0 / 3.0)


def test_storage_closure_is_perfect_for_a_known_water_balance() -> None:
    result = storage_closure_metrics(
        [1.0, 1.0 + 1.0 / 3600.0, 1.0 + 2.0 / 3600.0],
        [0.0, 1.0, 3.0],
        [1.0, 1.0, 1.0],
        flow_to_volume_per_second=1.0,
    )

    assert result["paired_observations"] == 2
    assert result["coverage"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["bias"] == pytest.approx(0.0)


def test_training_lag_is_frozen_for_evaluation() -> None:
    index = _hourly_index(12)
    upstream = pd.Series(
        [2.0, 5.0, 1.0, 7.0, 3.0, 8.0, 4.0, 6.0, 9.0, 2.0, 5.0, 1.0],
        index=index,
    )
    comparison = pd.DataFrame(
        {
            "storage": np.arange(12, dtype=float),
            "outflow": np.ones(12),
            "raw_inflow": upstream.to_numpy(),
            "centered_rolling_inflow": upstream.to_numpy(),
            "estimated_inflow": upstream.shift(1).to_numpy(),
            "revised_inflow": upstream.shift(1).to_numpy(),
            "upstream_flow": upstream.to_numpy(),
        },
        index=index,
    )
    settings = ValidationSettings(
        cross_correlation_range=2,
        minimum_paired_observations=3,
        training_start=index[0],
        training_end=index[5],
        evaluation_start=index[6],
        evaluation_end=index[-1],
    )

    outputs = generate_validation_outputs(comparison, settings=settings)

    assert (
        outputs.best_lag_summary.loc[
            "estimated_inflow (causal)", "lag_selection"
        ]
        == "training (frozen for evaluation)"
    )
    assert outputs.best_lag_summary.loc[
        "estimated_inflow (causal)", "best_lag_hours"
    ] == 1
