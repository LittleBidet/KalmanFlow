from dataclasses import replace
from datetime import timedelta
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    ConfigEvaluationResult,
    NoiseEvaluationSettings,
    NoiseTuningError,
    NoiseTuningProgress,
    NoiseTuningResult,
    NoiseTuningSettings,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
    evaluate_configuration,
    tune_inflow_noise,
)
from kalmanflow.models import ReservoirStateSpaceModel
from kalmanflow.noise_tuning import (
    _candidate,
    _proxy,
    _types,
)
from kalmanflow.noise_tuning._diagnostics import (
    _aggregate_arrays,
    _elapsed_lag_autocorrelation,
    _marginal_predictive_nlpd,
    _paired_standard_error,
    _storage_conditional_nlpd,
)
from kalmanflow.noise_tuning._preparation import _prepare, _validate_windows
from kalmanflow.noise_tuning._proxy import _proxy_metrics


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="demo",
        reservoir_name="Demo",
        q=np.diag([0.002, 1.0, 0.02]),
        r=np.diag([0.25, 4.0]),
        p0=np.diag([10.0, 20.0, 30.0]),
        smoothing_lag=timedelta(hours=1),
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


def _settings() -> tuple[NoiseEvaluationSettings, NoiseTuningSettings]:
    return (
        NoiseEvaluationSettings(
            warmup=timedelta(0),
            min_scored_storage_observations=2,
        ),
        NoiseTuningSettings(max_evaluations_per_search=30),
    )


def _tune(
    storage: pd.Series,
    discharge: pd.Series,
    windows: tuple[ValidationWindow, ...],
    *,
    config: ReservoirConfig | None = None,
    seeds: tuple[float, ...] = (2.0, 5.0, 10.0),
    settings: NoiseEvaluationSettings | None = None,
    tuning_settings: NoiseTuningSettings | None = None,
    upstream_proxy: pd.Series | None = None,
    progress: object = None,
) -> NoiseTuningResult:
    default_settings, default_tuning = _settings()
    return tune_inflow_noise(
        storage,
        discharge,
        config or _config(),
        seeds,
        windows,
        settings=settings or default_settings,
        tuning_settings=tuning_settings or default_tuning,
        upstream_proxy=upstream_proxy,
        proposed_configuration_version="config-v2",
        progress=progress,  # type: ignore[arg-type]
    )


def _synthetic_config_and_observations() -> tuple[
    ReservoirConfig, pd.Series, pd.Series, tuple[ValidationWindow, ...]
]:
    """Build a deterministic noisy record from a known state-space model."""

    index = pd.date_range("2025-01-01", periods=240, freq="h", tz="UTC")
    true_q = np.diag([0.005, 0.04, 0.02])
    true_r = np.diag([0.25, 0.16])
    # Deliberately misspecify the starting configuration. The known generating
    # values below remain inside the search bounds used by the tests.
    config = ReservoirConfig(
        reservoir_id="synthetic",
        reservoir_name="Synthetic",
        q=np.diag([0.002, 0.16, 0.01]),
        r=np.diag([0.5, 0.04]),
        p0=np.diag([1.0, 0.25, 0.25]),
        smoothing_lag=timedelta(hours=1),
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
        NoiseTuningSettings(
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
        NoiseTuningSettings(
            proxy_max_lag=timedelta(days=20) - timedelta(microseconds=1),
            proxy_diagnostic_frequency=timedelta(days=20),
            proxy_min_aligned_points=3,
        ),
        _types._DiagnosticPlan((mask,), mask),
    )
    assert metrics["proxy_best_lag_seconds"] == 0.0


def test_tuner_changes_all_five_diagonal_noise_terms() -> None:
    storage, discharge, windows = _inputs()
    base = _config()
    result = _tune(storage, discharge, windows, config=base)

    assert set(result.selected_parameters) == set(_types._PARAMETER_NAMES)
    np.testing.assert_allclose(
        np.diag(result.selected_config.q),
        [
            result.selected_parameters[name]
            for name in ("q_storage", "q_inflow", "q_outflow")
        ],
    )
    np.testing.assert_allclose(
        np.diag(result.selected_config.r),
        [result.selected_parameters[name] for name in ("r_storage", "r_outflow")],
    )
    assert np.diag(base.q).tolist() == [0.002, 1.0, 0.02]
    assert np.diag(base.r).tolist() == [0.25, 4.0]
    assert result.candidate_summary["candidate"].tolist() == [
        "base",
        "search-1",
        "search-2",
    ]
    metadata = result.selected_config.metadata
    assert metadata["calibration_method"] == "causal-innovation-maximum-likelihood"
    assert metadata["selection"]["selected_candidate"] in {"search-1", "search-2"}


def test_maximum_likelihood_recovers_identifiable_synthetic_noise_terms() -> None:
    base, storage, discharge, windows = _synthetic_config_and_observations()
    result = _tune(
        storage,
        discharge,
        windows,
        config=base,
        seeds=(6.0, 12.0, 24.0),
        settings=NoiseEvaluationSettings(
            warmup=timedelta(0), min_scored_storage_observations=10
        ),
        tuning_settings=NoiseTuningSettings(
            q_storage_multiplier_bounds=(0.1, 10.0),
            q_inflow_multiplier_bounds=(0.01, 10.0),
            q_outflow_multiplier_bounds=(0.1, 10.0),
            r_storage_multiplier_bounds=(0.05, 10.0),
            r_outflow_multiplier_bounds=(0.1, 50.0),
        ),
    )

    # Inflow and outflow diffusion dominate this record, so they are the
    # identifiable terms; the generating values are q=0.04 and q=0.02.
    assert result.selected_parameters["q_inflow"] == pytest.approx(0.04, rel=0.1)
    assert result.selected_parameters["q_outflow"] == pytest.approx(0.02, rel=0.1)
    assert result.selected_objective < result.base_objective
    identifiability = result.identifiability
    assert identifiability.loc["q_inflow", "log_standard_error"] < 1.0
    assert identifiability.loc["q_storage", "log_standard_error"] > 1.0
    assert any("poorly identified" in warning for warning in result.warnings)


def test_identifiability_does_not_depend_on_window_weights() -> None:
    from kalmanflow.noise_tuning import _parameters, _preparation, _workflows

    storage, discharge, _ = _inputs(40)
    index = storage.index
    settings, tuning = _settings()
    config = _config()
    prepared = _prepare(storage, discharge, config)
    space = _parameters._ParameterSpace.from_settings(config, tuning)
    errors = []
    # Two short windows and one long one, weighted equally and then not.
    for weights in ((1.0, 1.0, 1.0), (1.0, 1.0, 4.0)):
        windows = _validate_windows(
            tuple(
                ValidationWindow(f"w{i}", index[a], index[b], weight=weight)
                for i, ((a, b), weight) in enumerate(
                    zip(((2, 8), (8, 14), (14, 38)), weights, strict=True)
                )
            )
        )
        plan = _preparation._diagnostic_plan(prepared, windows, settings)
        objective = _workflows._Objective(
            prepared, windows, settings, plan, _workflows._ProgressReporter(None, 1)
        )
        selected = _workflows._Candidate(
            "search-1", dict(space.base), objective.evaluate(space.base)
        )
        frame, _ = _workflows._check_identifiability(objective, space, selected)
        errors.append(frame["log_standard_error"].to_numpy())
    assert len(errors[0]) == 3
    np.testing.assert_allclose(errors[0], errors[1], rtol=1e-9)


def test_inflow_seeds_must_lie_inside_q_inflow_bounds() -> None:
    storage, discharge, windows = _inputs()
    with pytest.raises(ValueError, match="inside q_inflow_multiplier_bounds"):
        _tune(storage, discharge, windows, seeds=(2.0, 500.0))
    with pytest.raises(ValueError, match="at least one inflow increment seed"):
        _tune(storage, discharge, windows, seeds=())


def test_search_stays_inside_custom_multiplier_bounds() -> None:
    storage, discharge, windows = _inputs()
    base = _config()
    bounds = {
        "q_storage_multiplier_bounds": (2.0, 3.0),
        "q_outflow_multiplier_bounds": (2.0, 4.0),
        "r_storage_multiplier_bounds": (2.0, 3.0),
        "r_outflow_multiplier_bounds": (2.0, 4.0),
    }
    result = _tune(
        storage,
        discharge,
        windows,
        config=base,
        tuning_settings=NoiseTuningSettings(max_evaluations_per_search=30, **bounds),
    )
    searched = result.candidate_summary.query("candidate != 'base'")
    base_values = {
        "q_storage": base.q[0, 0],
        "q_outflow": base.q[2, 2],
        "r_storage": base.r[0, 0],
        "r_outflow": base.r[1, 1],
    }
    for name, value in base_values.items():
        low, high = bounds[f"{name}_multiplier_bounds"]
        assert (searched[name] >= value * low * (1 - 1e-9)).all()
        assert (searched[name] <= value * high * (1 + 1e-9)).all()


def test_timing_names_match_search_work() -> None:
    storage, discharge, windows = _inputs()
    result = _tune(
        storage,
        discharge,
        windows,
        tuning_settings=NoiseTuningSettings(max_evaluations_per_search=10),
    )
    assert set(result.timing_seconds) == {"search_seconds", "identifiability_seconds"}
    assert all(value >= 0.0 for value in result.timing_seconds.values())


def test_skipping_identifiability_leaves_an_empty_report() -> None:
    storage, discharge, windows = _inputs()
    result = _tune(
        storage,
        discharge,
        windows,
        tuning_settings=NoiseTuningSettings(
            max_evaluations_per_search=10, check_identifiability=False
        ),
    )
    assert result.identifiability.empty


def test_failed_evaluation_ranks_last_and_search_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge, windows = _inputs()
    real_filter = _candidate.kalman_filter
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise np.linalg.LinAlgError("injected numerical failure")
        return real_filter(*args, **kwargs)

    monkeypatch.setattr(_candidate, "kalman_filter", fail_first)
    result = _tune(
        storage,
        discharge,
        windows,
        tuning_settings=NoiseTuningSettings(max_evaluations_per_search=10),
    )
    assert result.candidate_summary.query("selected")["eligible"].all()
    assert np.isfinite(result.selected_objective)


def test_all_failed_evaluations_raise_noise_tuning_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge, windows = _inputs()

    def fail_all(*args, **kwargs):
        raise np.linalg.LinAlgError("injected numerical failure")

    monkeypatch.setattr(_candidate, "kalman_filter", fail_all)
    with pytest.raises(NoiseTuningError, match="nor any local search reached"):
        _tune(storage, discharge, windows)


def test_base_is_kept_when_every_search_stops_at_worse_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kalmanflow.noise_tuning import _optimization

    storage, discharge, windows = _inputs()

    def stop_at_upper_corner(objective, start, max_evaluations):
        # Every term on its upper bound scores far worse than the base here.
        return _optimization._SearchOutcome(
            point=np.ones(len(start)),
            evaluations=1,
            converged=False,
            message="evaluation budget reached",
        )

    monkeypatch.setattr(_optimization, "_local_search", stop_at_upper_corner)
    result = _tune(storage, discharge, windows)
    summary = result.candidate_summary.set_index("candidate")
    searched = summary.loc[["search-1", "search-2"], "objective"]
    assert (searched > summary.loc["base", "objective"]).all()
    assert summary.loc["base", "selected"]
    assert result.selected_objective == result.base_objective
    assert result.selected_parameters == {
        "q_storage": 0.002,
        "q_inflow": 1.0,
        "q_outflow": 0.02,
        "r_storage": 0.25,
        "r_outflow": 4.0,
    }
    assert result.selected_config.metadata["selection"]["selected_candidate"] == (
        "base"
    )
    assert any("keeps the base noise terms" in w for w in result.warnings)
    assert not any("did not converge" in w for w in result.warnings)
    assert not any("not clearly better" in w for w in result.warnings)


def test_search_is_reproducible() -> None:
    storage, discharge, windows = _inputs()
    first = _tune(storage, discharge, windows)
    second = _tune(storage, discharge, windows)
    assert first.selected_parameters == second.selected_parameters
    pd.testing.assert_frame_equal(first.candidate_summary, second.candidate_summary)


@pytest.mark.parametrize(
    ("settings_type", "field", "value"),
    [
        (NoiseEvaluationSettings, "min_scored_storage_observations", 3.5),
        (NoiseEvaluationSettings, "bootstrap_samples", True),
        (NoiseEvaluationSettings, "random_seed", 1.5),
        (NoiseEvaluationSettings, "max_regularized_steps", False),
        (NoiseTuningSettings, "local_searches", 2.5),
        (NoiseTuningSettings, "max_evaluations_per_search", True),
        (NoiseTuningSettings, "proxy_min_aligned_points", np.bool_(False)),
    ],
)
def test_integer_settings_reject_lossy_coercion(
    settings_type: type, field: str, value: object
) -> None:
    with pytest.raises(TypeError, match=f"{field} must be an integer"):
        settings_type(**{field: value})


def test_identifiability_flag_requires_a_boolean() -> None:
    with pytest.raises(TypeError, match="check_identifiability must be a bool"):
        NoiseTuningSettings(check_identifiability="false")


def test_validation_window_name_must_be_a_string() -> None:
    index = pd.date_range("2024-01-01", periods=2, freq="1h", tz="UTC")
    with pytest.raises(TypeError, match="window name must be a string"):
        ValidationWindow(123, index[0], index[1])


def test_tuning_settings_expose_only_canonical_names() -> None:
    for removed in ("total_trials", "acquisition_pool_size", "proxy_max_shape_rmse"):
        with pytest.raises(TypeError):
            NoiseTuningSettings(**{removed: 4})


def test_local_searches_are_limited_by_the_number_of_seeds() -> None:
    storage, discharge, windows = _inputs()
    result = _tune(
        storage,
        discharge,
        windows,
        tuning_settings=NoiseTuningSettings(
            local_searches=5, max_evaluations_per_search=10
        ),
    )
    assert len(result.candidate_summary) == 1 + 3
    assert "inflow_increment_sd_seeds" not in result.selected_config.metadata


def test_tuner_rejects_zero_base_term_for_multiplier_bounds() -> None:
    storage, discharge, windows = _inputs()
    zero_q = replace(_config(), q=np.diag([0.0, 1.0, 0.02]))
    with pytest.raises(ValueError, match="positive and finite"):
        _tune(storage, discharge, windows, config=zero_q)


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
        settings=NoiseEvaluationSettings(
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
        NoiseEvaluationSettings(innovation_max_lag=timedelta(0))


def test_optional_upstream_proxy_is_reported_without_entering_objective() -> None:
    storage, discharge, windows = _inputs(60)
    proxy = pd.Series(
        np.sin(np.linspace(0.0, 4.0 * np.pi, len(storage))) + 10.0,
        index=storage.index,
    )
    tuning = NoiseTuningSettings(
        max_evaluations_per_search=20, proxy_min_aligned_points=6
    )
    with_proxy = _tune(
        storage, discharge, windows, tuning_settings=tuning, upstream_proxy=proxy
    )
    without_proxy = _tune(storage, discharge, windows, tuning_settings=tuning)

    assert with_proxy.proxy_diagnostics["candidate"].tolist() == [
        "base",
        "search-1",
        "search-2",
    ]
    assert {
        "proxy_best_lag_seconds",
        "proxy_shape_correlation",
        "proxy_change_correlation",
        "proxy_check_passed",
        "selected",
    }.issubset(with_proxy.proxy_diagnostics.columns)
    assert "proxy_shape_rmse" not in with_proxy.proxy_diagnostics
    assert with_proxy.proxy_diagnostics["selected"].sum() == 1
    assert with_proxy.selected_parameters == without_proxy.selected_parameters
    assert without_proxy.proxy_diagnostics.empty
    assert "upstream_proxy_available" not in with_proxy.selected_config.metadata


@pytest.mark.parametrize("passes", [True, False])
def test_failed_proxy_check_warns_but_never_changes_selection(
    monkeypatch: pytest.MonkeyPatch, passes: bool
) -> None:
    storage, discharge, windows = _inputs(60)
    proxy = pd.Series(1.0, index=storage.index)
    baseline = _tune(storage, discharge, windows)

    def controlled_proxy_metrics(*args: object, **kwargs: object) -> dict:
        return {
            "proxy_available": True,
            "proxy_aligned_count": len(storage),
            "proxy_best_lag_seconds": 0.0,
            "proxy_shape_correlation": 1.0 if passes else -1.0,
            "proxy_change_correlation": 1.0 if passes else -1.0,
            "proxy_check_passed": passes,
        }

    monkeypatch.setattr(_proxy, "_proxy_metrics", controlled_proxy_metrics)
    result = _tune(storage, discharge, windows, upstream_proxy=proxy)

    assert result.selected_parameters == baseline.selected_parameters
    warned = any("upstream-proxy" in warning for warning in result.warnings)
    assert warned is not passes


def test_proxy_correlations_ignore_amplitude() -> None:
    index = pd.date_range("2026-01-01", periods=48, freq="h", tz="UTC")
    storm = np.exp(-(((np.arange(48) - 24.0) / 3.0) ** 2))
    estimate = 5.0 + 300.0 * storm
    mask = np.ones(len(index), dtype=bool)
    metrics = _proxy_metrics(
        SimpleNamespace(
            filtered_means=np.column_stack(
                [np.zeros(len(index)), estimate, np.zeros(len(index))]
            )
        ),
        SimpleNamespace(timestamps=index),
        # A scaled hydrograph with very different baseflow and peak sizes has
        # the same shape and must pass the check.
        pd.Series(0.2 + 3.0 * storm, index=index),
        NoiseTuningSettings(proxy_max_lag=timedelta(0)),
        _types._DiagnosticPlan((mask,), mask),
    )
    assert metrics["proxy_shape_correlation"] == pytest.approx(1.0)
    assert metrics["proxy_check_passed"]


def test_report_surface_is_compact() -> None:
    storage, discharge, windows = _inputs()
    result = _tune(storage, discharge, windows)
    assert {
        "candidate",
        "objective",
        "window_standard_error",
        "eligible",
        "calibrated",
        "at_bounds",
        "evaluations",
        "converged",
        "selected",
        "storage_coverage_50",
        "outflow_nis_variance",
        *_types._PARAMETER_NAMES,
    }.issubset(result.candidate_summary.columns)
    assert {
        "candidate",
        "storage_nlpd",
        "joint_nlpd",
        "joint_nis",
        "storage_nis",
        "outflow_nis",
        "scored_fraction",
    }.issubset(result.window_diagnostics.columns)
    assert set(result.window_diagnostics["candidate"]) == {
        "base",
        result.selected_config.metadata["selection"]["selected_candidate"],
    }


def test_validation_excludes_exactly_two_initialization_rows() -> None:
    storage, discharge, _ = _inputs()
    index = storage.index
    windows = (
        ValidationWindow("first", index[1], index[4]),
        ValidationWindow("second", index[8], index[12]),
        ValidationWindow("third", index[16], index[20]),
    )
    result = _tune(storage, discharge, windows)
    first_window = result.window_diagnostics.query("window == 'first'")
    assert first_window["storage_count"].tolist() == [2, 2]


def test_validation_diagnostics_ignore_future_outside_window_changes() -> None:
    storage, discharge, windows = _inputs(60)
    original = _tune(storage, discharge, windows)
    changed_storage = storage.copy()
    changed_discharge = discharge.copy()
    changed_storage.iloc[40:] += 10_000.0
    changed_discharge.iloc[40:] += 100.0
    changed = _tune(changed_storage, changed_discharge, windows)
    columns = ["q_inflow", "objective", "outflow_nis", "joint_nis"]
    pd.testing.assert_frame_equal(
        original.candidate_summary[columns], changed.candidate_summary[columns]
    )


def test_missing_storage_is_excluded_from_window_scoring() -> None:
    storage, discharge, windows = _inputs()
    storage.iloc[5] = np.nan
    result = _tune(storage, discharge, windows)
    first = result.window_diagnostics.query("window == 'window-0'")
    np.testing.assert_allclose(first["scored_fraction"], 7.0 / 8.0)


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


def test_proxy_check_ignores_values_outside_validation_windows() -> None:
    storage, discharge, windows = _inputs(60)
    proxy = pd.Series(np.linspace(1.0, 10.0, len(storage)), index=storage.index)
    changed_proxy = proxy.copy()
    changed_proxy.loc[storage.index[30:]] += 100_000.0
    tuning = NoiseTuningSettings(
        max_evaluations_per_search=20, proxy_min_aligned_points=6
    )
    first = _tune(
        storage, discharge, windows, tuning_settings=tuning, upstream_proxy=proxy
    )
    second = _tune(
        storage,
        discharge,
        windows,
        tuning_settings=tuning,
        upstream_proxy=changed_proxy,
    )
    pd.testing.assert_frame_equal(first.proxy_diagnostics, second.proxy_diagnostics)


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
        _tune(storage, discharge, windows, config=off_diagonal)


def test_paired_standard_error_applies_finite_sample_correction() -> None:
    differences = np.array([0.0, 2.0, 4.0])
    settings = NoiseEvaluationSettings(bootstrap_samples=1)
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
        settings=NoiseEvaluationSettings(warmup=timedelta(0)),
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
        settings=NoiseEvaluationSettings(
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


def test_frozen_evaluation_reports_calibration_warnings_for_finite_data() -> None:
    storage, discharge, _ = _inputs(30)
    evaluation = evaluate_configuration(
        storage,
        discharge,
        _config(),
        settings=NoiseEvaluationSettings(
            warmup=timedelta(0),
            min_scored_storage_observations=2,
            nis_warning_range=(1e9, 1e10),
            innovation_bias_warning=None,
        ),
    )

    summary = evaluation.candidate_summary.iloc[0]
    assert bool(summary["eligible"])
    assert not bool(summary["calibrated"])
    for metric in ("joint_nis", "storage_nis", "outflow_nis"):
        assert np.isfinite(summary[metric])
        assert any(
            warning.startswith(f"{metric}=")
            and "outside the calibration range" in warning
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


def test_progress_reports_each_filter_pass_by_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage, discharge, windows = _inputs()
    real_evaluate = _candidate._evaluate_candidate
    passes = 0

    def counting_evaluate(*args, **kwargs):
        nonlocal passes
        passes += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(_candidate, "_evaluate_candidate", counting_evaluate)
    updates: list[NoiseTuningProgress] = []
    _tune(storage, discharge, windows, progress=updates.append)

    phases = list(dict.fromkeys(update.phase for update in updates))
    assert phases == ["screening", "search-1", "search-2", "identifiability"]
    numbers = list(dict.fromkeys(update.phase_number for update in updates))
    assert numbers == [1, 2, 3, 4]
    assert {update.phase_count for update in updates} == {4}
    for phase in phases:
        counts = [u.evaluations for u in updates if u.phase == phase]
        assert counts == list(range(len(counts)))
    screening = [u for u in updates if u.phase == "screening"]
    assert screening[0].evaluation_limit == 3 + 1
    assert screening[-1].evaluations == 3 + 1
    # Every update after a phase start is one new filtering pass; cached
    # objective calls never report.
    assert sum(u.evaluations > 0 for u in updates) == passes
    best = [u.best_objective for u in updates]
    assert best == sorted(best, reverse=True)
    elapsed = [u.elapsed_seconds for u in updates]
    assert elapsed == sorted(elapsed)


def test_progress_has_no_identifiability_phase_when_the_check_is_off() -> None:
    storage, discharge, windows = _inputs()
    updates: list[NoiseTuningProgress] = []
    _tune(
        storage,
        discharge,
        windows,
        tuning_settings=NoiseTuningSettings(
            max_evaluations_per_search=10, check_identifiability=False
        ),
        progress=updates.append,
    )
    assert "identifiability" not in {update.phase for update in updates}
    assert {update.phase_count for update in updates} == {3}


def test_progress_callback_exceptions_cancel_tuning() -> None:
    storage, discharge, windows = _inputs()

    def cancel(update: NoiseTuningProgress) -> None:
        if update.phase == "search-1":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _tune(storage, discharge, windows, progress=cancel)
