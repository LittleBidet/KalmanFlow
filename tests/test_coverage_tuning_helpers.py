from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    BayesianEvaluationSettings,
    BayesianTuningSettings,
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
)
from kalmanflow.bayesian_tuning import (
    _candidate,
    _diagnostics,
    _optimization,
    _preparation,
    _proxy,
    _types,
)


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="coverage",
        reservoir_name="Coverage",
        q=np.diag([0.002, 1.0, 0.02]),
        r=np.diag([0.25, 4.0]),
        p0=np.diag([10.0, 20.0, 30.0]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.us_customary(),
    )


def _data(count: int = 12) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")
    return (
        pd.Series(100.0 + np.arange(count), index=index),
        pd.Series(4.0, index=index),
    )


def _windows(index: pd.DatetimeIndex) -> tuple[ValidationWindow, ...]:
    return (
        ValidationWindow("first", index[0], index[4]),
        ValidationWindow("second", index[4], index[8]),
        ValidationWindow("third", index[8], index[-1] + pd.Timedelta(hours=1)),
    )


def test_types_reject_invalid_timestamps_and_window_values() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="timezone-aware"):
        ValidationWindow("naive", index[0].tz_localize(None), index[1])
    with pytest.raises(ValueError, match="after"):
        ValidationWindow("backwards", index[1], index[0])
    with pytest.raises(ValueError, match="positive and finite"):
        ValidationWindow("zero-weight", index[0], index[1], weight=0.0)
    with pytest.raises(ValueError, match="positive and finite"):
        ValidationWindow("nan-weight", index[0], index[1], weight=np.nan)
    window = ValidationWindow("weighted", index[0], index[1], weight=2)
    np.testing.assert_array_equal(window.mask(index), [True, False])


def test_evaluation_settings_validate_all_numeric_thresholds() -> None:
    invalid = (
        ("warmup", "nonnegative"),
        ("practical_equivalence_tolerance", "nonnegative"),
        ("max_jitter_fraction", "nonnegative"),
        ("nis_warning_range", "finite increasing"),
        ("innovation_bias_warning", "nonnegative"),
    )
    values = {
        "warmup": "1h",
        "practical_equivalence_tolerance": -1.0,
        "max_jitter_fraction": -1.0,
        "nis_warning_range": (2.0, 1.0),
        "innovation_bias_warning": -1.0,
    }
    for field, message in invalid:
        with pytest.raises((TypeError, ValueError), match=message):
            BayesianEvaluationSettings(**{field: values[field]})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("q_storage_multiplier_bounds", (0.0, 1.0)),
        ("q_outflow_multiplier_bounds", (2.0, 1.0)),
        ("r_storage_multiplier_bounds", (np.nan, 1.0)),
        ("r_outflow_multiplier_bounds", (1.0, 1.0)),
        ("one_standard_error_weight", -1.0),
        ("max_abs_elapsed_lag_autocorrelation", -1.0),
        ("expected_improvement_xi", -1.0),
        ("proxy_max_lag", timedelta(days=-1)),
        ("proxy_diagnostic_frequency", timedelta(0)),
        ("proxy_min_shape_correlation", 2.0),
        ("proxy_min_change_correlation", -2.0),
        ("proxy_max_shape_rmse", -1.0),
    ],
)
def test_tuning_settings_reject_invalid_bounds(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        BayesianTuningSettings(**{field: value})


def test_result_dataclasses_require_dataframes() -> None:
    storage, discharge = _data()
    index = storage.index
    config = _config()
    with pytest.raises(TypeError, match="candidate_summary"):
        _types.ConfigEvaluationResult(config, [], pd.DataFrame())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="proxy_diagnostics"):
        _types.BayesianTuningResult(
            selected_config=config,
            selected_parameters={},
            candidate_summary=pd.DataFrame(),
            window_diagnostics=pd.DataFrame(),
            competitive_trial_ids=(),
            selected_objective=0.0,
            selection_threshold=0.0,
            selection_reason="test",
            proxy_diagnostics=[],  # type: ignore[arg-type]
        )
    assert index.equals(discharge.index)


@pytest.mark.parametrize(
    "index",
    [
        pd.Index([1, 2]),
        pd.DatetimeIndex(["2025-01-01"]),
        pd.DatetimeIndex(["2025-01-01", "2025-01-01"]),
        pd.DatetimeIndex(["2025-01-01 01:00Z", "2025-01-01 00:00Z"]),
    ],
)
def test_preparation_rejects_invalid_indexes(index: pd.Index) -> None:
    with pytest.raises(ValueError):
        _preparation._validate_index(index)  # type: ignore[arg-type]


def test_preparation_rejects_unsupported_timestamp_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    monkeypatch.setattr(
        _preparation.np,
        "datetime_data",
        lambda value: ("fortnight", 1),
    )
    with pytest.raises(ValueError, match="unsupported timestamp resolution"):
        _preparation._timestamp_seconds(index)


def test_preparation_rejects_bad_series_values_and_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge = _data()
    config = _config()
    with pytest.raises(TypeError, match="pandas Series"):
        _preparation._prepare(storage.to_numpy(), discharge, config)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match exactly"):
        _preparation._prepare(storage, discharge.iloc[:-1], config)
    bad_storage = storage.copy()
    bad_storage.iloc[2] = np.inf
    with pytest.raises(ValueError, match="storage must contain"):
        _preparation._prepare(bad_storage, discharge, config)
    bad_discharge = discharge.copy()
    bad_discharge.iloc[2] = -np.inf
    with pytest.raises(ValueError, match="discharge must contain"):
        _preparation._prepare(storage, bad_discharge, config)
    sparse_storage = storage.copy()
    sparse_storage.iloc[1:] = np.nan
    with pytest.raises(ValueError, match="at least two finite"):
        _preparation._prepare(sparse_storage, discharge, config)
    first_discharge_missing = discharge.copy()
    first_discharge_missing.iloc[0] = np.nan
    with pytest.raises(ValueError, match="first finite storage"):
        _preparation._prepare(storage, first_discharge_missing, config)

    monkeypatch.setattr(
        _preparation.ReservoirStateSpaceModel,
        "initial_outflow",
        lambda self, value: np.nan,
    )
    with pytest.raises(ValueError, match="initial state"):
        _preparation._prepare(storage, discharge, config)


def test_preparation_requires_at_least_one_window() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _preparation._validate_windows(())


def test_diagnostic_scalar_guards_and_jitter_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        _diagnostics._hourly_increment_sd_to_q(0.0)
    with pytest.raises(ValueError, match="predictive variance"):
        _diagnostics._marginal_predictive_nlpd(1.0, 0.0)
    with pytest.raises(ValueError, match="shape"):
        _diagnostics._storage_conditional_nlpd(1.0, 1.0, np.eye(3))
    with pytest.raises(ValueError, match="outflow predictive"):
        _diagnostics._storage_conditional_nlpd(1.0, 1.0, np.zeros((2, 2)))
    value = _diagnostics._storage_conditional_nlpd(
        1.0,
        1.0,
        np.zeros((2, 2)),
        max_jitter_fraction=1.0,
    )
    assert np.isfinite(value)
    assert (
        _diagnostics._stable_logpdf(
            np.array([1.0]), np.array([[np.nan]]), 1.0
        )[0]
        is None
    )
    assert (
        _diagnostics._stable_logpdf(
            np.array([1.0]), np.array([[-1.0]]), 0.0
        )[0]
        is None
    )
    assert (
        _diagnostics._stable_logpdf(
            np.array([1.0]), np.array([[-1.0]]), 0.5
        )[0]
        is None
    )
    value, quad, jitter = _diagnostics._stable_logpdf(
        np.array([1.0]), np.array([[-1.0]]), 3.0
    )
    assert 1.0 < jitter <= 3.0
    variance = jitter - 1.0
    assert quad == pytest.approx(1.0 / variance)
    assert value == pytest.approx(
        0.5 * (np.log(2.0 * np.pi * variance) + 1.0 / variance)
    )

    monkeypatch.setattr(
        _diagnostics,
        "_stable_logpdf",
        lambda *args, **kwargs: (None, np.nan, 0.0),
    )
    with pytest.raises(ValueError, match="outflow predictive"):
        _diagnostics._storage_conditional_nlpd(
            1.0, 1.0, np.zeros((2, 2)), max_jitter_fraction=1.0
        )


def test_elapsed_autocorrelation_edge_cases_and_chunked_pair_scan() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="match timestamps"):
        _diagnostics._elapsed_lag_autocorrelation(index, [1.0], timedelta(hours=1))
    with pytest.raises(ValueError, match="max_lag"):
        _diagnostics._elapsed_lag_autocorrelation(index, [1.0, 2.0, 3.0], timedelta(0))
    assert _diagnostics._elapsed_lag_autocorrelation(
        index, [np.nan, 2.0, np.nan], timedelta(hours=1)
    ).empty
    assert _diagnostics._elapsed_lag_autocorrelation(
        index, [1.0, 2.0, 3.0], timedelta(minutes=30)
    ).empty
    assert np.isnan(_diagnostics._normalized_pair_correlation(1, np.inf, 1, 3))

    increments = np.tile([1, 2], 1100)[:2200]
    ticks = np.cumsum(np.r_[0, increments])
    tail = np.arange(1, 2801, dtype=np.int64) * 2_000_000 + 10_000_000
    ticks = np.r_[ticks, tail]
    irregular = pd.DatetimeIndex(
        pd.Timestamp("2025-01-01", tz="UTC")
        + pd.to_timedelta(ticks, unit="us")
    )
    result = _diagnostics._elapsed_lag_autocorrelation(
        irregular,
        np.sin(np.arange(len(irregular), dtype=float) / 20.0),
        timedelta(microseconds=1),
    )
    assert len(result) == 1
    # All distinct pairs in the dense cluster match; isolated tail points do not.
    assert result["pair_count"].iloc[0] == 2201 * 2200 // 2


def test_elapsed_autocorrelation_chunked_scan_skips_empty_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    increments = np.tile([1, 2], 2500)[:5000]
    ticks = np.cumsum(np.r_[0, increments])
    index = pd.DatetimeIndex(
        pd.Timestamp("2025-01-01", tz="UTC")
        + pd.to_timedelta(ticks, unit="us")
    )
    original_searchsorted = _diagnostics.np.searchsorted

    def synthetic_searchsorted(
        values: np.ndarray,
        query: np.ndarray,
        *,
        side: str,
    ) -> np.ndarray:
        output = original_searchsorted(values, query, side=side)
        if side == "left":
            return np.zeros_like(output)
        output = np.zeros_like(output)
        output[:4096] = 1000
        return output

    monkeypatch.setattr(_diagnostics.np, "searchsorted", synthetic_searchsorted)
    values = np.sin(np.arange(len(index), dtype=float))
    result = _diagnostics._elapsed_lag_autocorrelation(
        index, values, timedelta(microseconds=1)
    )
    # The injected bounds select every forward pair among the first 1000 values.
    # Compute the expected correlation directly, independently of chunking.
    source, target = np.triu_indices(1000, k=1)
    centered = values - values.mean()
    left, right = centered[source], centered[target]
    expected = np.dot(left, right) / np.sqrt(np.dot(left, left) * np.dot(right, right))
    assert result["pair_count"].iloc[0] == 1000 * 999 // 2
    assert result["autocorrelation"].iloc[0] == pytest.approx(expected)


def test_physical_metrics_reports_unavailable_sparse_autocorrelation() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    diagnostics = {
        key: np.ones(3, dtype=float)
        for key in (
            "storage_innovation",
            "storage_z",
            "outflow_z",
            "conditional_storage_z",
            "joint_nis",
            "joint_components",
            "storage_nis",
            "outflow_nis",
            "conditional_storage_nis",
        )
    }
    result = SimpleNamespace(
        filtered_means=np.column_stack(
            (np.ones(3), np.arange(3, dtype=float), np.ones(3))
        )
    )
    metrics = _diagnostics._physical_metrics_arrays(
        result,
        diagnostics,
        timestamps,
        np.ones(3, dtype=bool),
        BayesianEvaluationSettings(warmup=timedelta(0)),
    )
    assert np.isnan(metrics["max_storage_elapsed_lag_autocorrelation"])


def test_paired_standard_error_bootstrap_and_degenerate_weights() -> None:
    settings = BayesianEvaluationSettings(bootstrap_samples=3, random_seed=4)
    output = _diagnostics._paired_standard_error(
        np.arange(5, dtype=float), np.full(5, 0.2), settings
    )
    assert np.isfinite(output)
    assert _diagnostics._paired_standard_error(
        np.array([1.0]), np.array([1.0]), settings
    ) == 0.0


def test_optimization_helpers_cover_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _optimization._weighted_standard_error(
        np.array([1.0, np.nan]), np.array([1.0, 1.0])
    ) == 0.0
    assert _optimization._weighted_standard_error(
        np.array([1.0, 2.0]), np.array([1.0, 0.0])
    ) == 0.0
    options = BayesianTuningSettings(acquisition_pool_size=128)
    with pytest.raises(ValueError, match="at least three"):
        _optimization._parameter_bounds(_config(), [1.0, 2.0], options)

    rng = np.random.default_rng(2)
    point, source = _optimization._fit_and_acquire([], [], [], rng, options)
    assert point.shape == (5,)
    assert source == "space-filling-fallback"
    point, source = _optimization._fit_and_acquire(
        [], [], [np.zeros(5)], np.random.default_rng(2), options
    )
    assert point.shape == (5,)
    assert source == "space-filling-fallback"

    class ZeroRng:
        def random(self, shape: object) -> np.ndarray:
            return np.zeros(shape, dtype=float)

        def normal(self, *args: object, size: tuple[int, int]) -> np.ndarray:
            return np.zeros(size, dtype=float)

    point, source = _optimization._fit_and_acquire(
        [np.zeros(5)], [1.0], [np.zeros(5)], ZeroRng(), options  # type: ignore[arg-type]
    )
    assert point.shape == (5,)
    assert source == "random-fallback"

    class FailingModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def fit(self, *args: object, **kwargs: object) -> None:
            raise ValueError("synthetic GP failure")

    monkeypatch.setattr(_optimization, "GaussianProcessRegressor", FailingModel)
    point, source = _optimization._fit_and_acquire(
        [np.zeros(5), np.ones(5)],
        [1.0, 2.0],
        [np.zeros(5), np.ones(5)],
        np.random.default_rng(3),
        options,
    )
    assert point.shape == (5,)
    assert source == "space-filling-fallback"


def test_candidate_handles_nonfinite_filter_and_scoring_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge = _data(5)
    storage.iloc[2] = np.nan
    discharge.iloc[2] = np.nan
    discharge.iloc[3] = np.nan
    prepared = _preparation._prepare(storage, discharge, _config())
    windows = (
        ValidationWindow("first", prepared.index[0], prepared.index[2]),
        ValidationWindow("second", prepared.index[2], prepared.index[4]),
        ValidationWindow(
            "third",
            prepared.index[4],
            prepared.index[-1] + pd.Timedelta(hours=1),
        ),
    )
    count = len(prepared.timestamps)
    fake = SimpleNamespace(
        filtered_means=np.column_stack(
            (np.arange(count, dtype=float), np.full(count, np.nan), np.ones(count))
        ),
        predicted_means=np.zeros((count, 3)),
        predicted_covariances=np.tile(np.eye(3), (count, 1, 1)),
        innovations=np.zeros((count, 2)),
        innovation_covariances=np.zeros((count, 2, 2)),
    )
    monkeypatch.setattr(_candidate, "kalman_filter", lambda *args, **kwargs: fake)
    candidate = _candidate._evaluate_candidate(
        prepared,
        q_inflow=1.0,
        windows=windows,
        settings=BayesianEvaluationSettings(
            warmup=timedelta(0), max_jitter_fraction=0.0
        ),
    )
    assert any("non-positive-definite" in reason for reason in candidate.reasons)
    assert any("predictive variance" in reason for reason in candidate.reasons)

    positive = SimpleNamespace(
        filtered_means=np.column_stack(
            (
                np.arange(count, dtype=float),
                np.arange(count, dtype=float),
                np.ones(count),
            )
        ),
        predicted_means=np.zeros((count, 3)),
        predicted_covariances=np.tile(np.eye(3), (count, 1, 1)),
        innovations=np.zeros((count, 2)),
        innovation_covariances=np.tile(np.eye(2), (count, 1, 1)),
    )
    monkeypatch.setattr(_candidate, "kalman_filter", lambda *args, **kwargs: positive)
    monkeypatch.setattr(
        _candidate._diagnostics,
        "_stable_logpdf",
        lambda *args, **kwargs: (1.0, 1.0, 1.0),
    )
    regularized = _candidate._evaluate_candidate(
        prepared,
        q_inflow=1.0,
        windows=windows,
        settings=BayesianEvaluationSettings(
            warmup=timedelta(0), max_regularized_steps=0
        ),
    )
    assert "excessive covariance regularization" in regularized.reasons


def test_proxy_scalar_guards_and_unavailable_paths() -> None:
    assert np.isnan(_proxy._correlation(np.ones(2), np.ones(2)))
    assert np.isnan(_proxy._correlation(np.ones(3), np.ones(4)))
    assert np.isnan(_proxy._correlation(np.array([1.0, np.nan, 3.0]), np.arange(3.0)))
    assert np.isnan(_proxy._correlation(np.ones(3), np.arange(3.0)))
    assert _proxy._robust_scale(np.ones(4)) > 0.0

    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    prepared = SimpleNamespace(timestamps=index)
    plan = _types._DiagnosticPlan((np.ones(5, dtype=bool),), np.ones(5, dtype=bool))
    result = SimpleNamespace(filtered_means=np.zeros((5, 3)))
    metrics = _proxy._proxy_metrics(
        result, prepared, None, BayesianTuningSettings(), plan
    )
    assert metrics["proxy_gate_passed"] is True
    metrics = _proxy._proxy_metrics(
        result,
        prepared,
        pd.Series([1.0] * 5, index=index),
        BayesianTuningSettings(proxy_min_aligned_points=3),
        plan,
    )
    assert metrics["proxy_gate_passed"] is False

    with pytest.raises(TypeError, match="pandas Series"):
        _proxy._proxy_metrics(result, prepared, [1.0], BayesianTuningSettings(), plan)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _proxy._proxy_metrics(
            result,
            prepared,
            pd.Series([1.0] * 5, index=range(5)),
            BayesianTuningSettings(),
            plan,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        _proxy._proxy_metrics(
            result,
            prepared,
            pd.Series([1.0] * 5, index=index.tz_localize(None)),
            BayesianTuningSettings(),
            plan,
        )
    metrics = _proxy._proxy_metrics(
        None,
        prepared,
        pd.Series([1.0] * 5, index=index),
        BayesianTuningSettings(),
        plan,
    )
    assert metrics["proxy_gate_passed"] is False


def test_proxy_lag_search_covers_sparse_and_flat_pairs() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    plan = _types._DiagnosticPlan((np.ones(5, dtype=bool),), np.ones(5, dtype=bool))
    prepared = SimpleNamespace(timestamps=index)
    estimate = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    proxy = np.array([np.nan, 4.0, 5.0, 6.0, np.nan])
    result = SimpleNamespace(
        filtered_means=np.column_stack((np.zeros(5), estimate, np.zeros(5)))
    )
    options = BayesianTuningSettings(
        proxy_min_aligned_points=3,
        proxy_max_lag=timedelta(hours=1),
        proxy_diagnostic_frequency="1h",
    )
    metrics = _proxy._proxy_metrics(
        result, prepared, pd.Series(proxy, index=index), options, plan
    )
    assert metrics["proxy_aligned_count"] == 3
    assert metrics["proxy_shape_correlation"] == pytest.approx(1.0)
    assert metrics["proxy_change_correlation"] == 0.0
    assert metrics["proxy_best_lag_seconds"] == 0.0
    assert metrics["proxy_gate_passed"] is False

    flat = np.ones(5)
    metrics = _proxy._proxy_metrics(
        SimpleNamespace(filtered_means=np.column_stack((flat, flat, flat))),
        prepared,
        pd.Series(flat, index=index),
        BayesianTuningSettings(
            proxy_min_aligned_points=3,
            proxy_max_lag=timedelta(0),
        ),
        plan,
    )
    assert metrics["proxy_gate_passed"] is False

    sparse_proxy = pd.Series(
        [1.0, 2.0, np.nan, np.nan, np.nan], index=index
    )
    sparse_result = SimpleNamespace(
        filtered_means=np.column_stack(
            (np.zeros(5), [1.0, 2.0, np.nan, np.nan, np.nan], np.zeros(5))
        )
    )
    sparse_metrics = _proxy._proxy_metrics(
        sparse_result,
        prepared,
        sparse_proxy,
        BayesianTuningSettings(proxy_min_aligned_points=3),
        plan,
    )
    assert sparse_metrics["proxy_aligned_count"] == 2
    assert sparse_metrics["proxy_gate_passed"] is False


def test_proxy_alignment_length_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    prepared = SimpleNamespace(timestamps=index)
    plan = _types._DiagnosticPlan((np.ones(3, dtype=bool),), np.ones(3, dtype=bool))
    result = SimpleNamespace(filtered_means=np.zeros((3, 3)))
    original = pd.Series.to_numpy

    def short_to_numpy(self: pd.Series, *args: object, **kwargs: object) -> np.ndarray:
        values = original(self, *args, **kwargs)
        if self.name == "proxy":
            return values[:-1]
        return values

    monkeypatch.setattr(pd.Series, "to_numpy", short_to_numpy)
    with pytest.raises(ValueError, match="aligned"):
        _proxy._proxy_metrics(
            result,
            prepared,
            pd.Series([1.0, 2.0, 3.0], index=index, name="proxy"),
            BayesianTuningSettings(),
            plan,
        )


def test_proxy_lag_search_skips_malformed_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    prepared = SimpleNamespace(timestamps=index)
    plan = _types._DiagnosticPlan((np.ones(3, dtype=bool),), np.ones(3, dtype=bool))
    result = SimpleNamespace(
        filtered_means=np.column_stack((np.zeros(3), [1.0, 2.0, 3.0], np.zeros(3)))
    )
    original = pd.Series.to_numpy

    class ShortSlice(np.ndarray):
        def __new__(cls, values: np.ndarray) -> ShortSlice:
            return np.asarray(values).view(cls)

        def __getitem__(self, key: object) -> np.ndarray:
            values = super().__getitem__(key)
            if isinstance(key, slice) and key.start == 0 and key.stop == 3:
                return np.asarray(values)[:-1]
            return values

    def malformed_to_numpy(
        self: pd.Series, *args: object, **kwargs: object
    ) -> np.ndarray:
        values = original(self, *args, **kwargs)
        if self.name == "proxy":
            return ShortSlice(values)
        return values

    monkeypatch.setattr(pd.Series, "to_numpy", malformed_to_numpy)
    metrics = _proxy._proxy_metrics(
        result,
        prepared,
        pd.Series([1.0, 2.0, 3.0], index=index, name="proxy"),
        BayesianTuningSettings(proxy_min_aligned_points=3, proxy_max_lag=timedelta(0)),
        plan,
    )
    assert metrics["proxy_gate_passed"] is False
    assert np.isnan(metrics["proxy_shape_correlation"])
