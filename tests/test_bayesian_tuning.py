from dataclasses import replace
from datetime import timedelta
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    BayesianEvaluationSettings,
    BayesianTuningError,
    BayesianTuningSettings,
    ConfigEvaluationResult,
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
    evaluate_configuration,
    tune_inflow_noise_bayesian,
)
from kalmanflow.bayesian_tuning import (
    _candidate,
    _types,
)
from kalmanflow.bayesian_tuning._diagnostics import (
    _aggregate_arrays,
    _elapsed_lag_autocorrelation,
    _marginal_predictive_nlpd,
    _paired_standard_error,
    _storage_conditional_nlpd,
)
from kalmanflow.bayesian_tuning._preparation import _prepare, _validate_windows
from kalmanflow.bayesian_tuning._proxy import _proxy_metrics
from kalmanflow.models import ReservoirStateSpaceModel


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="demo",
        reservoir_name="Demo",
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


def _inputs(count: int = 30):
    index = pd.date_range("2025-01-01", periods=count, freq="h", tz="UTC")
    storage = pd.Series(100.0 + np.arange(count), index=index)
    discharge = pd.Series(4.0, index=index)
    windows = tuple(
        ValidationWindow(
            f"window-{i}",
            index[2 + i * 8],
            index[2 + (i + 1) * 8],
        )
        for i in range(3)
    )
    return storage, discharge, windows


def _settings() -> tuple[BayesianEvaluationSettings, BayesianTuningSettings]:
    return (
        BayesianEvaluationSettings(
            warmup=timedelta(0),
            min_scored_storage_observations=2,
        ),
        BayesianTuningSettings(
            total_trials=6,
            initial_trials=3,
            acquisition_pool_size=128,
            random_seed=7,
        ),
    )


def _synthetic_config_and_observations() -> tuple[
    ReservoirConfig, pd.Series, pd.Series, tuple[ValidationWindow, ...]
]:
    """Build a deterministic noisy record from a known state-space model."""

    index = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    true_q = np.diag([0.005, 0.04, 0.02])
    true_r = np.diag([0.25, 0.16])
    # Deliberately misspecify the starting configuration. The known generating
    # values below remain inside the Bayesian search bounds, but are not all
    # represented by the initial seed coordinates.
    config = ReservoirConfig(
        reservoir_id="synthetic",
        reservoir_name="Synthetic",
        q=np.diag([0.002, 0.16, 0.01]),
        r=np.diag([0.5, 0.04]),
        p0=np.diag([1.0, 0.25, 0.25]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.SYSTEM_FLOW_RATE,
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.si(),
    )
    model = ReservoirStateSpaceModel(true_q, UnitSystem.si())
    rng = np.random.default_rng(1234)
    state = np.array([10_000.0, 10.0, 3.0])
    storage_values: list[float] = []
    discharge_values: list[float] = []
    process = model.process_covariance(3600.0)
    for _ in index:
        storage_values.append(state[0] + rng.normal(0.0, sqrt(true_r[0, 0])))
        discharge_values.append(state[2] + rng.normal(0.0, sqrt(true_r[1, 1])))
        state = model.transition_matrix(3600.0) @ state
        state += rng.multivariate_normal(np.zeros(3), process)
    windows = tuple(
        ValidationWindow(
            f"window-{i}",
            index[start],
            index[end],
        )
        for i, (start, end) in enumerate(((20, 60), (80, 120), (140, 180), (200, 230)))
    )
    return (
        config,
        pd.Series(storage_values, index=index),
        pd.Series(discharge_values, index=index),
        windows,
    )


def test_predictive_density_helpers_match_scalar_calculation() -> None:
    expected = 0.5 * (np.log(2.0 * np.pi) + np.log(4.0) + 0.25)
    assert _marginal_predictive_nlpd(1.0, 4.0) == pytest.approx(expected)


def test_conditional_storage_density_uses_outflow_covariance() -> None:
    covariance = np.array([[4.0, 1.0], [1.0, 2.0]])
    conditional_variance = 4.0 - 1.0 / 2.0
    conditional_innovation = 1.0 - 1.0 * 2.0 / 2.0
    expected = _marginal_predictive_nlpd(conditional_innovation, conditional_variance)
    assert _storage_conditional_nlpd(1.0, 2.0, covariance) == pytest.approx(expected)


def test_elapsed_autocorrelation_uses_elapsed_time_for_irregular_cadence() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T03:00:00Z",
            "2025-01-01T04:00:00Z",
        ]
    )
    output = _elapsed_lag_autocorrelation(
        timestamps, [1.0, 2.0, 4.0, 8.0], timedelta(hours=3)
    )
    assert output.loc[output["lag_seconds"] == 3600.0, "pair_count"].iloc[0] == 2


def test_elapsed_autocorrelation_regular_cadence_has_expected_pair_counts() -> None:
    timestamps = pd.date_range("2025-01-01", periods=16, freq="h", tz="UTC")
    values = np.sin(np.arange(len(timestamps), dtype=float))
    output = _elapsed_lag_autocorrelation(timestamps, values, timedelta(hours=4))
    assert output["pair_count"].tolist() == [15, 14, 13, 12]
    assert output["autocorrelation"].notna().all()


def test_elapsed_autocorrelation_wide_bins_use_pair_normalized_energy() -> None:
    timestamps = pd.date_range("2025-01-01", periods=100, freq="s", tz="UTC")
    values = np.sin(np.arange(len(timestamps), dtype=float) / 20.0)
    output = _elapsed_lag_autocorrelation(
        timestamps, values, timedelta(seconds=5)
    )

    centered = values - values.mean()
    steps = (1, 2, 3)
    covariance = sum(
        np.dot(centered[:-step], centered[step:]) for step in steps
    )
    source_energy = sum(
        np.dot(centered[:-step], centered[:-step]) for step in steps
    )
    target_energy = sum(
        np.dot(centered[step:], centered[step:]) for step in steps
    )
    expected = covariance / np.sqrt(source_energy * target_energy)
    row = output.loc[output["lag_seconds"] == 2.0].iloc[0]
    assert row["pair_count"] == sum(len(values) - step for step in steps)
    assert row["autocorrelation"] == pytest.approx(expected)
    assert abs(row["autocorrelation"]) <= 1.0


def test_elapsed_autocorrelation_reports_sparse_bins_as_unavailable() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    output = _elapsed_lag_autocorrelation(
        timestamps, [1.0, 2.0, 4.0], timedelta(hours=1)
    )
    assert output["pair_count"].tolist() == [2]
    assert np.isnan(output["autocorrelation"].iloc[0])


def test_elapsed_autocorrelation_irregular_missing_matches_pair_oracle() -> None:
    timestamps = pd.DatetimeIndex(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T00:00:01Z",
            "2025-01-01T00:00:03Z",
            "2025-01-01T00:00:04Z",
            "2025-01-01T00:00:07Z",
            "2025-01-01T00:00:09Z",
            "2025-01-01T00:00:10Z",
            "2025-01-01T00:00:13Z",
        ]
    )
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0])
    tolerance = timedelta(seconds=1.5)
    output = _elapsed_lag_autocorrelation(
        timestamps,
        values,
        timedelta(seconds=4),
        lag_tolerance=tolerance,
    )

    centre = float(np.nanmean(values))
    target = 2.0
    pairs = [
        (values[left] - centre, values[right] - centre)
        for left in range(len(values))
        for right in range(left + 1, len(values))
        if np.isfinite(values[left])
        and np.isfinite(values[right])
        and abs(
            (timestamps[right] - timestamps[left]).total_seconds() - target
        )
        <= tolerance.total_seconds()
    ]
    pair_values = np.asarray(pairs)
    expected = np.dot(pair_values[:, 0], pair_values[:, 1]) / np.sqrt(
        np.dot(pair_values[:, 0], pair_values[:, 0])
        * np.dot(pair_values[:, 1], pair_values[:, 1])
    )
    row = output.loc[output["lag_seconds"] == target].iloc[0]
    assert row["pair_count"] == len(pairs)
    assert row["autocorrelation"] == pytest.approx(expected)


def _proxy_metrics_for_delay(
    delay: int,
    *,
    max_lag: timedelta,
    diagnostic_frequency: str | timedelta = "1h",
    min_aligned_points: int = 12,
) -> dict[str, float | bool]:
    index = pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC")
    proxy = np.random.default_rng(7).normal(size=len(index))
    if delay > 0:
        estimate = np.r_[np.full(delay, np.nan), proxy[:-delay]]
    elif delay < 0:
        estimate = np.r_[proxy[-delay:], np.full(-delay, np.nan)]
    else:
        estimate = proxy.copy()
    mask = np.ones(len(index), dtype=bool)
    return _proxy_metrics(
        SimpleNamespace(
            filtered_means=np.column_stack(
                [np.zeros(len(index)), estimate, np.zeros(len(index))]
            )
        ),
        SimpleNamespace(timestamps=index),
        pd.Series(proxy, index=index),
        BayesianTuningSettings(
            proxy_max_lag=max_lag,
            proxy_diagnostic_frequency=diagnostic_frequency,
            proxy_min_aligned_points=min_aligned_points,
        ),
        _types._DiagnosticPlan((mask,), mask),
    )


@pytest.mark.parametrize(
    ("delay", "expected_lag"), [(2, 7200.0), (-2, -7200.0)]
)
def test_proxy_lag_reports_shift_of_proxy_with_documented_sign(
    delay: int, expected_lag: float
) -> None:
    metrics = _proxy_metrics_for_delay(delay, max_lag=timedelta(hours=4))
    assert metrics["proxy_best_lag_seconds"] == expected_lag


def test_proxy_lag_never_exceeds_subcadence_or_below_bound_maximum() -> None:
    subcadence = _proxy_metrics_for_delay(
        1, max_lag=timedelta(minutes=30), min_aligned_points=12
    )
    assert subcadence["proxy_best_lag_seconds"] == 0.0

    # A one-microsecond deficit must not be swallowed by floating-point epsilon.
    index = pd.date_range("2026-01-01", periods=100 * 24, freq="h", tz="UTC")
    proxy = np.random.default_rng(4).normal(size=len(index))
    delay = 20 * 24
    estimate = np.r_[np.full(delay, np.nan), proxy[:-delay]]
    mask = np.ones(len(index), dtype=bool)
    metrics = _proxy_metrics(
        SimpleNamespace(
            filtered_means=np.column_stack(
                [np.zeros(len(index)), estimate, np.zeros(len(index))]
            )
        ),
        SimpleNamespace(timestamps=index),
        pd.Series(proxy, index=index),
        BayesianTuningSettings(
            proxy_max_lag=timedelta(days=20) - timedelta(microseconds=1),
            proxy_diagnostic_frequency=timedelta(days=20),
            proxy_min_aligned_points=3,
        ),
        _types._DiagnosticPlan((mask,), mask),
    )
    assert metrics["proxy_best_lag_seconds"] == 0.0


def test_bayesian_tuner_changes_all_five_diagonal_noise_terms() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    base = _config()
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        base,
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    assert set(result.selected_parameters) == {
        "q_storage",
        "q_inflow",
        "q_outflow",
        "r_storage",
        "r_outflow",
    }
    np.testing.assert_allclose(
        np.diag(result.selected_config.q),
        [
            result.selected_parameters["q_storage"],
            result.selected_parameters["q_inflow"],
            result.selected_parameters["q_outflow"],
        ],
    )
    np.testing.assert_allclose(
        np.diag(result.selected_config.r),
        [
            result.selected_parameters["r_storage"],
            result.selected_parameters["r_outflow"],
        ],
    )
    assert np.diag(base.q).tolist() == [0.002, 1.0, 0.02]
    assert np.diag(base.r).tolist() == [0.25, 4.0]
    assert len(result.candidate_summary) == search.total_trials


def test_bayesian_seed_candidates_define_q_inflow_bounds() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    expected = {2.0**2 / 3600.0, 5.0**2 / 3600.0, 10.0**2 / 3600.0}
    initial = result.candidate_summary.query("trial_id < 3")
    np.testing.assert_allclose(sorted(initial["q_inflow"]), sorted(expected))
    assert result.candidate_summary["q_inflow"].min() == pytest.approx(min(expected))
    assert result.candidate_summary["q_inflow"].max() == pytest.approx(max(expected))


def test_bayesian_initial_design_stays_inside_custom_multiplier_bounds() -> None:
    storage, discharge, windows = _inputs()
    settings = BayesianEvaluationSettings(
        warmup=timedelta(0),
        min_scored_storage_observations=2,
    )
    search = BayesianTuningSettings(
        total_trials=6,
        initial_trials=3,
        acquisition_pool_size=128,
        random_seed=7,
        q_storage_multiplier_bounds=(2.0, 3.0),
        q_outflow_multiplier_bounds=(2.0, 4.0),
        r_storage_multiplier_bounds=(2.0, 3.0),
        r_outflow_multiplier_bounds=(2.0, 4.0),
    )
    base = _config()
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        base,
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    base_values = np.array(
        [base.q[0, 0], base.q[1, 1], base.q[2, 2], base.r[0, 0], base.r[1, 1]]
    )
    lower = base_values * np.array([2.0, 1.0, 2.0, 2.0, 2.0])
    upper = base_values * np.array([3.0, 1.0, 4.0, 3.0, 4.0])
    lower[1] = min(value**2 / 3600.0 for value in (2.0, 5.0, 10.0))
    upper[1] = max(value**2 / 3600.0 for value in (2.0, 5.0, 10.0))
    values = result.candidate_summary.sort_values("trial_id")[
        ["q_storage", "q_inflow", "q_outflow", "r_storage", "r_outflow"]
    ].to_numpy()
    assert np.all(values >= lower - 1e-12)
    assert np.all(values <= upper + 1e-12)
    # The base point is projected to the nearest declared edge for every
    # non-inflow parameter rather than being evaluated outside the box.
    np.testing.assert_allclose(values[0, [0, 2, 3, 4]], lower[[0, 2, 3, 4]])


def test_bayesian_timing_names_match_candidate_work() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    assert {
        "candidate_evaluation_seconds",
        "acquisition_seconds",
        "total_seconds",
    } <= set(result.timing_seconds)
    assert "filtering_seconds" not in result.timing_seconds
    assert "scoring_seconds" not in result.timing_seconds
    assert all(value >= 0.0 for value in result.timing_seconds.values())


def test_bayesian_first_pass_releases_filter_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    real_evaluate = _candidate._evaluate_candidate
    first_passes = []

    def spy_evaluate(*args, **kwargs):
        candidate = real_evaluate(*args, **kwargs)
        first_passes.append(candidate)
        return candidate

    monkeypatch.setattr(_candidate, "_evaluate_candidate", spy_evaluate)
    tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    assert len(first_passes) == search.total_trials
    assert all(candidate.filter_result is None for candidate in first_passes)
    assert all(not candidate.diagnostics for candidate in first_passes)


def test_failed_trial_is_rejected_and_later_valid_trial_can_be_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge, windows = _inputs()
    settings = BayesianEvaluationSettings(
        warmup=timedelta(0), min_scored_storage_observations=2
    )
    search = BayesianTuningSettings(
        total_trials=3, initial_trials=3, acquisition_pool_size=128
    )
    real_filter = _candidate.kalman_filter
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("injected numerical failure")
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(_candidate, "kalman_filter", fail_first)
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )

    failed = result.candidate_summary.loc[
        result.candidate_summary["trial_id"] == 0
    ].iloc[0]
    assert not failed["eligible"]
    assert "filter failed" in failed["rejection_reasons"]
    assert "joint NLPD" in failed["rejection_reasons"]
    assert result.candidate_summary["eligible"].sum() == 2


def test_all_failed_trials_raise_bayesian_tuning_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge, windows = _inputs()

    def fail_all(*args, **kwargs):
        raise np.linalg.LinAlgError("injected numerical failure")

    monkeypatch.setattr(_candidate, "kalman_filter", fail_all)
    with pytest.raises(BayesianTuningError, match="no candidate passed"):
        tune_inflow_noise_bayesian(
            storage,
            discharge,
            _config(),
            [2.0, 5.0, 10.0],
            windows,
            settings=BayesianEvaluationSettings(
                warmup=timedelta(0), min_scored_storage_observations=2
            ),
            bayesian_settings=BayesianTuningSettings(
                total_trials=3, initial_trials=3, acquisition_pool_size=128
            ),
            proposed_configuration_version="config-v2",
        )


def test_bayesian_search_improves_known_synthetic_model_objective() -> None:
    base, storage, discharge, windows = _synthetic_config_and_observations()
    settings = BayesianEvaluationSettings(
        warmup=timedelta(0),
        min_scored_storage_observations=10,
    )
    search = BayesianTuningSettings(
        total_trials=16,
        initial_trials=6,
        acquisition_pool_size=256,
        random_seed=19,
        q_storage_multiplier_bounds=(0.5, 3.0),
        q_outflow_multiplier_bounds=(0.5, 3.0),
        r_storage_multiplier_bounds=(0.25, 2.0),
        r_outflow_multiplier_bounds=(0.25, 5.0),
    )
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        base,
        [6.0, 12.0, 24.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    initial = result.candidate_summary.query("acquisition_source == 'initial-design'")
    acquired = result.candidate_summary.query("acquisition_source != 'initial-design'")
    assert len(initial) == search.initial_trials
    assert np.isfinite(initial["robust_objective"]).all()
    assert np.isfinite(acquired["robust_objective"]).all()
    assert acquired["robust_objective"].min() < initial["robust_objective"].min()
    # The selected inflow diffusion should remain close to the known synthetic
    # value (q=0.04, corresponding to a 12-unit hourly increment SD).
    selected_prior = sqrt(result.selected_parameters["q_inflow"] * 3600.0)
    assert selected_prior == pytest.approx(12.0, abs=6.0)


def test_bayesian_search_is_reproducible() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    kwargs = dict(
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    first = tune_inflow_noise_bayesian(
        storage, discharge, _config(), [2.0, 5.0, 10.0], windows, **kwargs
    )
    second = tune_inflow_noise_bayesian(
        storage, discharge, _config(), [2.0, 5.0, 10.0], windows, **kwargs
    )
    assert first.selected_parameters == second.selected_parameters
    pd.testing.assert_frame_equal(
        first.candidate_summary.drop(columns=["timing_seconds"], errors="ignore"),
        second.candidate_summary.drop(columns=["timing_seconds"], errors="ignore"),
    )


def test_bayesian_settings_validate_trial_budget() -> None:
    with pytest.raises(ValueError, match="between one and total_trials"):
        BayesianTuningSettings(total_trials=2, initial_trials=3)


@pytest.mark.parametrize(
    ("settings_type", "field", "value"),
    [
        (BayesianEvaluationSettings, "min_scored_storage_observations", 3.5),
        (BayesianEvaluationSettings, "bootstrap_samples", True),
        (BayesianEvaluationSettings, "random_seed", 1.5),
        (BayesianEvaluationSettings, "max_regularized_steps", False),
        (BayesianTuningSettings, "total_trials", 40.5),
        (BayesianTuningSettings, "initial_trials", True),
        (BayesianTuningSettings, "random_seed", 1.5),
        (BayesianTuningSettings, "acquisition_pool_size", 128.5),
        (BayesianTuningSettings, "proxy_min_aligned_points", np.bool_(False)),
    ],
)
def test_integer_settings_reject_lossy_coercion(
    settings_type: type, field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        settings_type(**{field: value})


def test_proxy_gate_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="proxy_require_gate must be a bool"):
        BayesianTuningSettings(proxy_require_gate="false")


def test_validation_window_name_must_be_a_string() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    with pytest.raises(TypeError, match="window name must be a string"):
        ValidationWindow(123, index[0], index[1])


def test_bayesian_settings_expose_only_canonical_names() -> None:
    with pytest.raises(TypeError):
        BayesianTuningSettings(n_trials=4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BayesianTuningSettings(initial_design_size=4)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        BayesianTuningSettings(q_storage_bounds=(0.5, 1.0))  # type: ignore[call-arg]


def test_all_supplied_seeds_are_evaluated_when_initial_default_is_smaller() -> None:
    storage, discharge, windows = _inputs()
    settings, _ = _settings()
    seeds = [2.0, 3.0, 4.0, 5.0]
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        seeds,
        windows,
        settings=settings,
        bayesian_settings=BayesianTuningSettings(
            total_trials=4,
            initial_trials=3,
            acquisition_pool_size=128,
            random_seed=7,
        ),
        proposed_configuration_version="config-v2",
    )
    np.testing.assert_allclose(
        sorted(result.candidate_summary["inflow_increment_sd"]), seeds
    )
    assert set(result.candidate_summary["acquisition_source"]) == {"initial-design"}
    assert "inflow_increment_sd_seeds" not in result.selected_config.metadata
    assert result.selected_config.metadata["selection"]["competitive_trial_ids"]


def test_trial_budget_must_cover_all_supplied_seeds() -> None:
    storage, discharge, windows = _inputs()
    settings, _ = _settings()
    with pytest.raises(ValueError, match="at least the number of supplied"):
        tune_inflow_noise_bayesian(
            storage,
            discharge,
            _config(),
            [2.0, 3.0, 4.0, 5.0],
            windows,
            settings=settings,
            bayesian_settings=BayesianTuningSettings(
                total_trials=3,
                initial_trials=3,
                acquisition_pool_size=128,
            ),
            proposed_configuration_version="config-v2",
        )


def test_bayesian_tuner_rejects_zero_base_term_for_multiplier_bounds() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    base = _config()
    zero_q = ReservoirConfig(
        reservoir_id=base.reservoir_id,
        reservoir_name=base.reservoir_name,
        q=np.diag([0.0, 1.0, 0.02]),
        r=base.r,
        p0=base.p0,
        smoothing_lag=base.smoothing_lag,
        initialization_strategy=base.initialization_strategy,
        inflow_units=base.inflow_units,
        model_version=base.model_version,
        configuration_version=base.configuration_version,
        unit_system=base.unit_system,
    )
    with pytest.raises(ValueError, match="positive and finite"):
        tune_inflow_noise_bayesian(
            storage,
            discharge,
            zero_q,
            [2.0, 5.0, 10.0],
            windows,
            settings=settings,
            bayesian_settings=search,
            proposed_configuration_version="config-v2",
        )


def test_joint_nlpd_is_aggregated_per_observed_component() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    diagnostics = {
        key: np.zeros(2, dtype=float)
        for key in (
            "primary_nlpd",
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
    diagnostics["joint_nlpd"] = np.array([2.0, 4.0])
    diagnostics["joint_components"] = np.array([2.0, 1.0])
    row = _aggregate_arrays(
        diagnostics,
        np.ones(2, dtype=bool),
        ValidationWindow("window", index[0], index[-1] + pd.Timedelta(hours=1)),
        np.ones(2, dtype=bool),
    )
    assert row["joint_nlpd"] == pytest.approx((2.0 + 4.0) / 3.0)


def test_window_metrics_use_their_own_finite_observations() -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    diagnostics = {
        "primary_nlpd": np.array([1.0, 1.0, np.nan]),
        "joint_nlpd": np.ones(3),
        "joint_nis": np.ones(3),
        "joint_components": np.array([2.0, 1.0, 1.0]),
        "storage_nis": np.array([1.0, 2.0, np.nan]),
        "outflow_nis": np.array([3.0, np.nan, 5.0]),
        "conditional_storage_nis": np.array([4.0, np.nan, np.nan]),
        "storage_z": np.array([1.0, 2.0, np.nan]),
        "outflow_z": np.array([3.0, np.nan, 5.0]),
        "conditional_storage_z": np.array([4.0, np.nan, np.nan]),
        "jitter": np.zeros(3),
    }
    row = _aggregate_arrays(
        diagnostics,
        np.ones(3, dtype=bool),
        ValidationWindow("window", index[0], index[-1] + pd.Timedelta(hours=1)),
        np.ones(3, dtype=bool),
    )

    assert row["storage_nis"] == pytest.approx(1.5)
    assert row["outflow_nis"] == pytest.approx(4.0)
    assert row["conditional_storage_nis"] == pytest.approx(4.0)


def test_short_positive_autocorrelation_lag_returns_empty_diagnostic() -> None:
    storage, discharge, windows = _inputs()
    result = evaluate_configuration(
        storage,
        discharge,
        _config(),
        evaluation_window=windows[0],
        settings=BayesianEvaluationSettings(
            warmup=timedelta(0),
            innovation_max_lag=timedelta(minutes=30),
        ),
    )

    assert np.isnan(
        result.candidate_summary.loc[
            0, "max_material_elapsed_lag_autocorrelation"
        ]
    )


def test_zero_autocorrelation_lag_is_rejected_at_settings_boundary() -> None:
    with pytest.raises(ValueError, match="innovation_max_lag must be a positive"):
        BayesianEvaluationSettings(innovation_max_lag=timedelta(0))


def test_result_exposes_competitive_trial_ids() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    expected = tuple(
        result.candidate_summary.loc[
            result.candidate_summary["competitive"], "trial_id"
        ].astype(int)
    )
    assert set(result.competitive_trial_ids) == set(expected)


def test_optional_upstream_proxy_is_reported_without_entering_objective() -> None:
    storage, discharge, windows = _inputs(60)
    settings, _ = _settings()
    proxy = pd.Series(
        np.sin(np.linspace(0.0, 4.0 * np.pi, len(storage))) + 10.0,
        index=storage.index,
    )
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=BayesianTuningSettings(
            total_trials=6,
            initial_trials=3,
            acquisition_pool_size=128,
            proxy_min_aligned_points=6,
        ),
        upstream_proxy=proxy,
        proposed_configuration_version="config-v2",
    )
    assert len(result.proxy_diagnostics) == 6
    assert {
        "proxy_best_lag_seconds",
        "proxy_shape_correlation",
        "proxy_gate_passed",
        "competitive",
        "selected",
    }.issubset(result.proxy_diagnostics.columns)
    assert "proxy_available" not in result.candidate_summary
    assert "upstream_proxy_available" not in result.selected_config.metadata


def test_bayesian_report_surface_is_compact() -> None:
    storage, discharge, windows = _inputs()
    settings, search = _settings()
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        proposed_configuration_version="config-v2",
    )
    assert not hasattr(result, "detailed_regime_report")
    assert not hasattr(result, "forecast_horizon_report")
    assert {
        "trial_id",
        "acquisition_source",
        "objective",
        "robust_objective",
        "calibration_violation",
        "eligible",
        "competitive",
        "selected",
        *_types._PARAMETER_NAMES,
    }.issubset(result.candidate_summary.columns)
    assert {
        "storage_nlpd",
        "joint_nlpd",
        "joint_nis",
        "storage_nis",
        "outflow_nis",
    }.issubset(result.window_diagnostics.columns)


def test_validation_excludes_exactly_two_initialization_rows() -> None:
    storage, discharge, _ = _inputs()
    index = storage.index
    windows = (
        ValidationWindow("first", index[1], index[4]),
        ValidationWindow("second", index[8], index[12]),
        ValidationWindow("third", index[16], index[20]),
    )
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=BayesianEvaluationSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        bayesian_settings=BayesianTuningSettings(
            total_trials=3, initial_trials=3, acquisition_pool_size=128
        ),
        proposed_configuration_version="config-v2",
    )
    first_window = result.window_diagnostics.query("window == 'first'")
    assert first_window["storage_count"].iloc[0] == 2


def test_validation_diagnostics_ignore_future_outside_window_changes() -> None:
    storage, discharge, windows = _inputs(60)
    settings = BayesianEvaluationSettings(
        warmup=timedelta(0), min_scored_storage_observations=2
    )
    kwargs = dict(
        base_config=_config(),
        inflow_increment_sd_seeds=[2.0, 5.0, 10.0],
        validation_windows=windows,
        settings=settings,
        bayesian_settings=BayesianTuningSettings(
            total_trials=6, initial_trials=3, acquisition_pool_size=128, random_seed=7
        ),
        proposed_configuration_version="config-v2",
    )
    original = tune_inflow_noise_bayesian(storage, discharge, **kwargs)
    changed_storage = storage.copy()
    changed_discharge = discharge.copy()
    changed_storage.iloc[40:] += 10_000.0
    changed_discharge.iloc[40:] += 100.0
    changed = tune_inflow_noise_bayesian(changed_storage, changed_discharge, **kwargs)
    columns = ["q_inflow", "objective", "outflow_nis", "joint_nis"]
    pd.testing.assert_frame_equal(
        original.candidate_summary[columns], changed.candidate_summary[columns]
    )


def test_missing_storage_is_excluded_from_window_coverage() -> None:
    storage, discharge, windows = _inputs()
    storage.iloc[5] = np.nan
    result = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=BayesianEvaluationSettings(
            warmup=timedelta(0), min_scored_storage_observations=2
        ),
        bayesian_settings=BayesianTuningSettings(
            total_trials=3, initial_trials=3, acquisition_pool_size=128
        ),
        proposed_configuration_version="config-v2",
    )
    first = result.window_diagnostics.query("window == 'window-0'")
    assert first["coverage"].iloc[0] == pytest.approx(7.0 / 8.0)


def test_validation_windows_require_unique_names_and_no_overlap() -> None:
    index = _inputs()[0].index
    duplicate_names = (
        ValidationWindow("duplicate", index[2], index[8]),
        ValidationWindow("duplicate", index[10], index[16]),
    )
    with pytest.raises(ValueError, match="names must be unique"):
        _validate_windows(duplicate_names)
    overlapping = (
        ValidationWindow("first", index[2], index[12]),
        ValidationWindow("second", index[10], index[16]),
    )
    with pytest.raises(ValueError, match="must not overlap"):
        _validate_windows(overlapping)


def test_proxy_gate_ignores_values_outside_validation_windows() -> None:
    storage, discharge, windows = _inputs(60)
    settings, _ = _settings()
    proxy = pd.Series(np.linspace(1.0, 10.0, len(storage)), index=storage.index)
    changed_proxy = proxy.copy()
    changed_proxy.loc[storage.index[30:]] += 100_000.0
    search = BayesianTuningSettings(
        total_trials=6,
        initial_trials=3,
        acquisition_pool_size=128,
        proxy_min_aligned_points=6,
        proxy_require_gate=False,
    )
    first = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        upstream_proxy=proxy,
        proposed_configuration_version="config-v2",
    )
    second = tune_inflow_noise_bayesian(
        storage,
        discharge,
        _config(),
        [2.0, 5.0, 10.0],
        windows,
        settings=settings,
        bayesian_settings=search,
        upstream_proxy=changed_proxy,
        proposed_configuration_version="config-v2",
    )
    columns = [
        "trial_id",
        "proxy_best_lag_seconds",
        "proxy_shape_correlation",
        "proxy_change_correlation",
        "proxy_shape_rmse",
        "proxy_gate_passed",
    ]
    pd.testing.assert_frame_equal(
        first.proxy_diagnostics[columns], second.proxy_diagnostics[columns]
    )


def test_irregular_process_covariance_matches_model_conversion() -> None:
    storage = pd.Series(
        [100.0, 101.0, 103.0, 104.0],
        index=pd.to_datetime(
            [
                "2025-01-01T00:00:00Z",
                "2025-01-01T00:17:00Z",
                "2025-01-01T02:03:00Z",
                "2025-01-01T04:11:00Z",
            ]
        ),
    )
    prepared = _prepare(storage, pd.Series(4.0, index=storage.index), _config())
    q_inflow = 0.7
    cached = (
        prepared.q_storage * prepared.storage_basis
        + q_inflow * prepared.inflow_basis
        + prepared.q_outflow * prepared.outflow_basis
    )
    direct_q = np.asarray(prepared.model.q_continuous).copy()
    direct_q[1, 1] = q_inflow
    direct_model = type(prepared.model)(direct_q, prepared.model.unit_system)
    elapsed_seconds = np.asarray(
        [
            (prepared.timestamps[row] - prepared.timestamps[row - 1]).total_seconds()
            for row in range(1, len(prepared.timestamps))
        ]
    )
    for row, elapsed in enumerate(elapsed_seconds):
        np.testing.assert_allclose(
            cached[row], direct_model.process_covariance(elapsed)
        )


@pytest.mark.parametrize("matrix_name", ["q", "r", "p0"])
def test_off_diagonal_covariances_are_rejected(matrix_name: str) -> None:
    storage, discharge, windows = _inputs()
    config = _config()
    matrix = np.asarray(getattr(config, matrix_name)).copy()
    if matrix_name == "r":
        matrix[0, 1] = matrix[1, 0] = 0.1
    else:
        matrix[0, 1] = matrix[1, 0] = 0.001
    off_diagonal = replace(config, **{matrix_name: matrix})
    with pytest.raises(ValueError, match="constant diagonal"):
        tune_inflow_noise_bayesian(
            storage,
            discharge,
            off_diagonal,
            [2.0, 5.0, 10.0],
            windows,
            settings=BayesianEvaluationSettings(warmup=timedelta(0)),
            bayesian_settings=BayesianTuningSettings(
                total_trials=3, initial_trials=3, acquisition_pool_size=128
            ),
            proposed_configuration_version="config-v2",
        )


def test_paired_standard_error_applies_finite_sample_correction() -> None:
    differences = np.array([0.0, 2.0, 4.0])
    settings = BayesianEvaluationSettings(bootstrap_samples=1)
    assert _paired_standard_error(
        differences, np.full(3, 1.0 / 3.0), settings
    ) == pytest.approx(np.sqrt(4.0 / 3.0))


def test_frozen_configuration_evaluation_is_compact_and_non_mutating() -> None:
    storage, discharge, windows = _inputs()
    config = _config()
    before_q = config.q.copy()
    evaluation = evaluate_configuration(
        storage,
        discharge,
        config,
        evaluation_window=windows[0],
        settings=BayesianEvaluationSettings(warmup=timedelta(0)),
    )
    assert isinstance(evaluation, ConfigEvaluationResult)
    np.testing.assert_allclose(config.q, before_q)
    assert len(evaluation.candidate_summary) == 1
    assert len(evaluation.window_diagnostics) == 1
    assert "eligible" in evaluation.candidate_summary
    assert not hasattr(evaluation, "detailed_regime_report")


def test_frozen_evaluation_requires_scored_data_before_calibration() -> None:
    index = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    storage = pd.Series(1000.0 + np.arange(len(index)), index=index)
    discharge = pd.Series(10.0, index=index)
    evaluation = evaluate_configuration(
        storage,
        discharge,
        _config(),
        settings=BayesianEvaluationSettings(
            # Disable calibration thresholds to isolate the hard eligibility gate.
            nis_warning_range=None,
            innovation_bias_warning=None,
        ),
    )

    summary = evaluation.candidate_summary.iloc[0]
    assert not summary["eligible"]
    assert not summary["calibrated"]
    assert summary["rejection_reasons"]
    assert any(
        "scored storage observations" in warning
        for warning in evaluation.warnings
    )


def test_frozen_evaluation_rejects_submicrosecond_observation_index() -> None:
    index = pd.DatetimeIndex(
        [
            "2025-01-01T00:00:00.000000123Z",
            "2025-01-01T01:00:00.000000123Z",
        ]
    )
    storage = pd.Series([100.0, 101.0], index=index)
    discharge = pd.Series([4.0, 4.0], index=index)
    with pytest.raises(ValueError, match="finer than microsecond"):
        evaluate_configuration(storage, discharge, _config())
