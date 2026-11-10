from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    NoiseEvaluationSettings,
    NoiseTuningSettings,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
)
from kalmanflow.noise_tuning import (
    _candidate,
    _diagnostics,
    _identifiability,
    _optimization,
    _parameters,
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup", "1h", "nonnegative"),
        ("innovation_max_lag", "1h", "positive"),
        ("practical_equivalence_tolerance", -1.0, "nonnegative"),
        ("max_jitter_fraction", np.inf, "nonnegative"),
        ("nis_warning_range", (2.0, 1.0), "finite increasing"),
        ("innovation_bias_warning", -1.0, "nonnegative"),
        ("autocorrelation_warning", np.nan, "nonnegative"),
        ("coverage_warning", -0.1, "nonnegative"),
    ],
)
def test_evaluation_settings_validate_all_numeric_thresholds(
    field: str, value: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        NoiseEvaluationSettings(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("q_storage_multiplier_bounds", (0.0, 1.0)),
        ("q_inflow_multiplier_bounds", (1.0,)),
        ("q_outflow_multiplier_bounds", (2.0, 1.0)),
        ("r_storage_multiplier_bounds", (np.nan, 1.0)),
        ("r_outflow_multiplier_bounds", (1.0, 1.0)),
        ("local_searches", 0),
        ("max_evaluations_per_search", 0),
        ("proxy_max_lag", timedelta(days=-1)),
        ("proxy_max_lag", "1h"),
        ("proxy_diagnostic_frequency", timedelta(0)),
        ("proxy_min_shape_correlation", 2.0),
        ("proxy_min_change_correlation", -2.0),
    ],
)
def test_tuning_settings_reject_invalid_bounds(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        NoiseTuningSettings(**{field: value})


def test_result_dataclasses_require_dataframes() -> None:
    config = _config()
    with pytest.raises(TypeError, match="candidate_summary"):
        _types.ConfigEvaluationResult(config, [], pd.DataFrame())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="identifiability"):
        _types.NoiseTuningResult(
            selected_config=config,
            selected_parameters={},
            selected_objective=0.0,
            base_objective=0.0,
            improvement_standard_error=0.0,
            candidate_summary=pd.DataFrame(),
            window_diagnostics=pd.DataFrame(),
            identifiability=[],  # type: ignore[arg-type]
        )


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
        NoiseEvaluationSettings(warmup=timedelta(0)),
    )
    assert np.isnan(metrics["max_storage_elapsed_lag_autocorrelation"])


def test_paired_standard_error_bootstrap_and_degenerate_weights() -> None:
    settings = NoiseEvaluationSettings(bootstrap_samples=3, random_seed=4)
    output = _diagnostics._paired_standard_error(
        np.arange(5, dtype=float), np.full(5, 0.2), settings
    )
    assert np.isfinite(output)
    assert _diagnostics._paired_standard_error(
        np.array([1.0]), np.array([1.0]), settings
    ) == 0.0


def test_weighted_standard_error_handles_degenerate_inputs() -> None:
    assert _diagnostics._weighted_standard_error(
        np.array([1.0, np.nan]), np.array([1.0, 1.0])
    ) == 0.0
    assert _diagnostics._weighted_standard_error(
        np.array([1.0, 2.0]), np.array([1.0, 0.0])
    ) == 0.0
    assert _diagnostics._weighted_standard_error(
        np.array([0.0, 2.0]), np.array([1.0, 1.0])
    ) == pytest.approx(1.0)


def test_local_search_reports_convergence_and_invalid_values() -> None:
    target = np.array([0.25, 0.75])

    def quadratic(point: np.ndarray) -> float:
        return float(np.sum((point - target) ** 2))

    outcome = _optimization._local_search(quadratic, np.full(2, 0.5), 100)
    assert outcome.converged
    np.testing.assert_allclose(outcome.point, target, atol=1e-5)
    assert outcome.evaluations > 1

    def invalid_near_start(point: np.ndarray) -> float:
        return np.inf if point[0] > 0.5 else quadratic(point)

    stopped = _optimization._local_search(invalid_near_start, np.full(2, 0.5), 100)
    assert not stopped.converged
    assert "numerically invalid" in stopped.message


def test_parameter_space_round_trips_and_reports_bounds() -> None:
    space = _parameters._ParameterSpace.from_settings(
        _config(), NoiseTuningSettings(q_storage_multiplier_bounds=(2.0, 3.0))
    )
    parameters = space.decode(np.full(5, 0.5))
    np.testing.assert_allclose(space.encode(parameters), 0.5)
    assert space.names_at_bounds(parameters) == ()
    assert space.names_at_bounds(space.decode(np.array([0, 0.5, 1, 0.5, 0.5]))) == (
        "q_storage",
        "q_outflow",
    )
    # The base q_storage (1x) lies below the 2x-3x box and starts on its edge.
    starts = space.starting_points([2.0, 2.0, 5.0])
    assert len(starts) == 2
    assert all(start[0] == 0.0 for start in starts)


def test_identifiability_report_flags_confounded_and_flat_terms() -> None:
    names = ["q_storage", "q_inflow"]
    ridge = np.array([[1.0, 0.99], [0.99, 1.0]])
    frame, warnings = _identifiability._identifiability_report(ridge, names, 10_000)
    assert frame.loc["q_storage", "q_inflow"] == pytest.approx(-0.99)
    assert frame.loc["q_storage", "log_standard_error"] < 1.0
    assert any("strongly confounded" in warning for warning in warnings)

    flat = np.array([[1.0, 0.0], [0.0, 0.0]])
    frame, warnings = _identifiability._identifiability_report(flat, names, 100)
    assert np.isinf(frame.loc["q_inflow", "log_standard_error"])
    assert frame[names].isna().all().all()
    assert warnings[0].startswith("q_inflow are poorly identified")
    assert "not strictly convex" in warnings[1]


def test_log_space_hessian_matches_a_known_quadratic() -> None:
    def objective(parameters: dict[str, float]) -> float:
        a, b = np.log(parameters["q_storage"]), np.log(parameters["q_inflow"])
        return 2.0 * a * a + a * b + 3.0 * b * b

    parameters = dict.fromkeys(_types._PARAMETER_NAMES, 1.0)
    hessian = _identifiability._log_space_hessian(
        objective, parameters, ["q_storage", "q_inflow"]
    )
    np.testing.assert_allclose(hessian, [[4.0, 1.0], [1.0, 6.0]], atol=1e-9)


def test_calibration_warnings_cover_each_check_and_disabling() -> None:
    calibrated = {
        "joint_nis": 1.0,
        "storage_nis": 1.0,
        "outflow_nis": 1.0,
        "storage_bias": 0.0,
        "outflow_bias": 0.0,
        "conditional_storage_bias": 0.0,
        "max_material_elapsed_lag_autocorrelation": 0.1,
        "storage_coverage_50": 0.5,
        "storage_coverage_90": 0.9,
        "outflow_coverage_50": 0.5,
        "outflow_coverage_90": 0.9,
    }
    settings = NoiseEvaluationSettings()
    assert _diagnostics._calibration_warnings(calibrated, settings) == ()
    miscalibrated = {
        **calibrated,
        "storage_nis": 3.0,
        "outflow_bias": -1.0,
        "max_material_elapsed_lag_autocorrelation": 0.5,
        "storage_coverage_50": 0.8,
        "outflow_coverage_90": np.nan,
    }
    warnings = _diagnostics._calibration_warnings(miscalibrated, settings)
    assert {warning.split("=")[0].split(" ")[0] for warning in warnings} == {
        "storage_nis",
        "outflow_bias",
        "max_material_elapsed_lag_autocorrelation",
        "storage_coverage_50",
        "outflow_coverage_90",
    }
    disabled = NoiseEvaluationSettings(
        nis_warning_range=None,
        innovation_bias_warning=None,
        autocorrelation_warning=None,
        coverage_warning=None,
    )
    assert _diagnostics._calibration_warnings(miscalibrated, disabled) == ()


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
        settings=NoiseEvaluationSettings(
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
        settings=NoiseEvaluationSettings(
            warmup=timedelta(0), max_regularized_steps=0
        ),
    )
    assert "excessive covariance regularization" in regularized.reasons


def test_failed_window_rows_share_the_completed_row_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _preparation._prepare(*_data(), _config())
    windows = _windows(prepared.index)
    settings = NoiseEvaluationSettings(warmup=timedelta(0))
    completed = _candidate._evaluate_candidate(
        prepared, q_inflow=1.0, windows=windows, settings=settings
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr(_candidate, "kalman_filter", fail)
    failed = _candidate._evaluate_candidate(
        prepared, q_inflow=1.0, windows=windows, settings=settings
    )
    assert failed.filter_result is None
    assert "filter failed: boom" in failed.reasons
    for window in windows:
        assert set(failed.window_rows[window.name]) == set(
            completed.window_rows[window.name]
        )
        assert failed.window_rows[window.name]["scored_fraction"] == 0.0


def test_proxy_scalar_guards_and_unavailable_paths() -> None:
    assert np.isnan(_proxy._correlation(np.ones(2), np.ones(2)))
    assert np.isnan(_proxy._correlation(np.ones(3), np.ones(4)))
    assert np.isnan(_proxy._correlation(np.array([1.0, np.nan, 3.0]), np.arange(3.0)))
    assert np.isnan(_proxy._correlation(np.ones(3), np.arange(3.0)))

    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    prepared = SimpleNamespace(timestamps=index)
    plan = _types._DiagnosticPlan((np.ones(5, dtype=bool),), np.ones(5, dtype=bool))
    result = SimpleNamespace(filtered_means=np.zeros((5, 3)))
    metrics = _proxy._proxy_metrics(
        result, prepared, None, NoiseTuningSettings(), plan
    )
    assert metrics["proxy_check_passed"] is True
    metrics = _proxy._proxy_metrics(
        result,
        prepared,
        pd.Series([1.0] * 5, index=index),
        NoiseTuningSettings(proxy_min_aligned_points=3),
        plan,
    )
    assert metrics["proxy_check_passed"] is False

    with pytest.raises(TypeError, match="pandas Series"):
        _proxy._proxy_metrics(result, prepared, [1.0], NoiseTuningSettings(), plan)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="DatetimeIndex"):
        _proxy._proxy_metrics(
            result,
            prepared,
            pd.Series([1.0] * 5, index=range(5)),
            NoiseTuningSettings(),
            plan,
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        _proxy._proxy_metrics(
            result,
            prepared,
            pd.Series([1.0] * 5, index=index.tz_localize(None)),
            NoiseTuningSettings(),
            plan,
        )
    metrics = _proxy._proxy_metrics(
        None,
        prepared,
        pd.Series([1.0] * 5, index=index),
        NoiseTuningSettings(),
        plan,
    )
    assert metrics["proxy_check_passed"] is False


def test_proxy_lag_search_covers_sparse_and_flat_pairs() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    plan = _types._DiagnosticPlan((np.ones(5, dtype=bool),), np.ones(5, dtype=bool))
    prepared = SimpleNamespace(timestamps=index)
    estimate = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
    proxy = np.array([np.nan, 4.0, 5.0, 6.0, np.nan])
    result = SimpleNamespace(
        filtered_means=np.column_stack((np.zeros(5), estimate, np.zeros(5)))
    )
    options = NoiseTuningSettings(
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
    assert metrics["proxy_check_passed"] is False

    flat = np.ones(5)
    metrics = _proxy._proxy_metrics(
        SimpleNamespace(filtered_means=np.column_stack((flat, flat, flat))),
        prepared,
        pd.Series(flat, index=index),
        NoiseTuningSettings(
            proxy_min_aligned_points=3,
            proxy_max_lag=timedelta(0),
        ),
        plan,
    )
    assert metrics["proxy_check_passed"] is False

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
        NoiseTuningSettings(proxy_min_aligned_points=3),
        plan,
    )
    assert sparse_metrics["proxy_aligned_count"] == 2
    assert sparse_metrics["proxy_check_passed"] is False


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
            NoiseTuningSettings(),
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
        NoiseTuningSettings(proxy_min_aligned_points=3, proxy_max_lag=timedelta(0)),
        plan,
    )
    assert metrics["proxy_check_passed"] is False
    assert np.isnan(metrics["proxy_shape_correlation"])
