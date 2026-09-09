from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from ..reservoir_config import ReservoirConfig
from . import _candidate, _diagnostics, _preparation, _proxy, _types

BayesianEvaluationSettings = _types.BayesianEvaluationSettings
BayesianTuningError = _types.BayesianTuningError
BayesianTuningResult = _types.BayesianTuningResult
BayesianTuningSettings = _types.BayesianTuningSettings
ConfigEvaluationResult = _types.ConfigEvaluationResult
ValidationWindow = _types.ValidationWindow
Array = np.ndarray


def _calibration_violation(
    physical: Mapping[str, float],
    settings: BayesianEvaluationSettings,
    options: BayesianTuningSettings,
) -> tuple[float, bool]:
    violation = 0.0
    calibrated = True
    nis_range = settings.nis_warning_range
    if nis_range is not None:
        for name in ("joint_nis", "storage_nis", "outflow_nis"):
            value = float(physical.get(name, np.nan))
            if not np.isfinite(value):
                calibrated = False
                violation += 1.0
            elif value < nis_range[0]:
                calibrated = False
                violation += ((nis_range[0] - value) / max(nis_range[0], 1e-12)) ** 2
            elif value > nis_range[1]:
                calibrated = False
                violation += ((value - nis_range[1]) / nis_range[1]) ** 2
    if settings.innovation_bias_warning is not None:
        limit = float(settings.innovation_bias_warning)
        for name in ("storage_bias", "outflow_bias", "conditional_storage_bias"):
            value = abs(float(physical.get(name, np.nan)))
            if not np.isfinite(value):
                calibrated = False
                violation += 1.0
            elif value > limit:
                calibrated = False
                violation += ((value - limit) / max(limit, 1e-12)) ** 2
    autocorrelation = abs(
        float(physical.get("max_material_elapsed_lag_autocorrelation", np.nan))
    )
    if not np.isfinite(autocorrelation):
        calibrated = False
        violation += 1.0
    elif autocorrelation > options.max_abs_elapsed_lag_autocorrelation:
        calibrated = False
        limit = max(options.max_abs_elapsed_lag_autocorrelation, 1e-12)
        violation += ((autocorrelation - limit) / limit) ** 2
    return float(violation), calibrated


def _hard_validity(
    candidate: Any,
    windows: Sequence[ValidationWindow],
    settings: BayesianEvaluationSettings,
) -> tuple[bool, list[str]]:
    """Apply the numerical gates shared by search and frozen evaluation."""

    reasons = list(candidate.reasons)
    for window in windows:
        row = candidate.window_rows.get(window.name, {})
        try:
            storage_count = int(row.get("storage_count", 0))
        except (TypeError, ValueError):
            storage_count = 0
        if storage_count < settings.min_scored_storage_observations:
            reasons.append(
                f"window {window.name!r} has {storage_count} scored storage "
                "observations; "
                f"at least {settings.min_scored_storage_observations} are required"
            )
        try:
            joint_loss = float(row.get("joint_nlpd", np.nan))
        except (TypeError, ValueError):
            joint_loss = np.nan
        if not np.isfinite(joint_loss):
            reasons.append(f"window {window.name!r} has no finite joint NLPD")
    unique_reasons = list(dict.fromkeys(reasons))
    return not unique_reasons, unique_reasons


def _calibration_warnings(
    physical: Mapping[str, float], settings: BayesianEvaluationSettings
) -> tuple[str, ...]:
    """Describe configured calibration threshold failures for frozen reports."""

    warnings: list[str] = []
    if settings.nis_warning_range is not None:
        low, high = settings.nis_warning_range
        for name in ("joint_nis", "storage_nis", "outflow_nis"):
            value = float(physical.get(name, np.nan))
            if not np.isfinite(value):
                warnings.append(f"{name} is unavailable for calibration")
            elif value < low or value > high:
                warnings.append(
                    f"{name}={value:.6g} is outside the configured "
                    f"warning range [{low:g}, {high:g}]"
                )
    if settings.innovation_bias_warning is not None:
        limit = float(settings.innovation_bias_warning)
        for name in ("storage_bias", "outflow_bias", "conditional_storage_bias"):
            value = float(physical.get(name, np.nan))
            if not np.isfinite(value):
                warnings.append(f"{name} is unavailable for calibration")
            elif abs(value) > limit:
                warnings.append(
                    f"{name}={value:.6g} exceeds the configured absolute "
                    f"warning limit {limit:g}"
                )
    return tuple(warnings)


def _proposed_config(
    base: ReservoirConfig,
    selected: Mapping[str, float],
    proposed_configuration_version: str,
    selected_trial: int,
    competitive_trials: Sequence[int],
    selected_objective: float,
    selected_calibration_violation: float,
) -> ReservoirConfig:
    q = np.asarray(base.q, dtype=float).copy()
    r = np.asarray(base.r, dtype=float).copy()
    q[0, 0], q[1, 1], q[2, 2] = (
        selected["q_storage"],
        selected["q_inflow"],
        selected["q_outflow"],
    )
    r[0, 0], r[1, 1] = selected["r_storage"], selected["r_outflow"]
    metadata = dict(base.metadata)
    metadata.update(
        {
            "calibration_method": "causal-innovation-bayesian-optimization",
            "tuned_parameters": list(_types._PARAMETER_NAMES),
            "selected_parameters": dict(selected),
            "selection": {
                "selected_trial_id": selected_trial,
                "selected_objective": float(selected_objective),
                "selected_calibration_violation": float(selected_calibration_violation),
                "competitive_trial_ids": [int(t) for t in competitive_trials],
            },
        }
    )
    return ReservoirConfig(
        reservoir_id=base.reservoir_id,
        reservoir_name=base.reservoir_name,
        q=q,
        r=r,
        p0=np.asarray(base.p0).copy(),
        smoothing_lag=base.smoothing_lag,
        initialization_strategy=base.initialization_strategy,
        inflow_units=base.inflow_units,
        model_version=base.model_version,
        configuration_version=proposed_configuration_version,
        metadata=metadata,
        unit_system=base.unit_system,
    )


def tune_inflow_noise_bayesian(
    storage: pd.Series,
    discharge: pd.Series,
    base_config: ReservoirConfig,
    inflow_increment_sd_seeds: Sequence[float],
    validation_windows: Sequence[ValidationWindow],
    *,
    settings: BayesianEvaluationSettings | None = None,
    bayesian_settings: BayesianTuningSettings | None = None,
    upstream_proxy: pd.Series | None = None,
    proposed_configuration_version: str | None = None,
) -> BayesianTuningResult:
    """Experimentally tune diagonal ``Q`` and ``R`` using causal innovation NLPD.

    Requires the optional ``kalmanflow[tuning]`` dependencies. The search API,
    selection rules, and result schema may change between releases. Proposals
    require independent validation before operational use.
    """

    # Keep filtering and frozen evaluation usable without the optimizer stack.
    try:
        from . import _optimization
    except ModuleNotFoundError as error:
        if error.name not in {"scipy", "sklearn"}:
            raise
        raise ImportError(
            "Experimental Bayesian tuning requires optional dependencies. "
            "Install them with: pip install 'kalmanflow[tuning]'"
        ) from error

    options = settings or BayesianEvaluationSettings()
    search = bayesian_settings or BayesianTuningSettings()
    if not isinstance(base_config, ReservoirConfig):
        raise TypeError("base_config must be a ReservoirConfig")
    if (
        not proposed_configuration_version
        or not str(proposed_configuration_version).strip()
    ):
        raise ValueError("proposed_configuration_version is required")
    if str(proposed_configuration_version) == base_config.configuration_version:
        raise ValueError("proposed_configuration_version must be new")
    windows = _preparation._validate_windows(validation_windows)
    if len(windows) < 3:
        raise BayesianTuningError("at least three validation windows are required")
    prepared = _preparation._prepare(storage, discharge, base_config)
    if upstream_proxy is not None and not isinstance(upstream_proxy, pd.Series):
        raise TypeError("upstream_proxy must be a pandas Series or None")
    plan = _preparation._diagnostic_plan(prepared, windows, options)
    weights = _preparation._window_weights(windows)
    base_values, lows, highs, seeds = _optimization._parameter_bounds(
        base_config, inflow_increment_sd_seeds, search
    )
    effective_initial_trials = max(search.initial_trials, len(seeds))
    if effective_initial_trials > search.total_trials:
        raise ValueError(
            "total_trials must be at least the number of supplied inflow seeds"
        )
    base_point = _optimization._encode(
        dict(zip(_types._PARAMETER_NAMES, base_values, strict=True)), lows, highs
    )
    rng = np.random.default_rng(search.random_seed)
    design = _optimization._initial_design(
        seeds, effective_initial_trials, lows, highs, base_values, rng
    )
    # Design coordinates are stored in log space, which is the natural scale
    # for covariance parameters and the scale seen by the GP.
    observed_x_values: list[Array] = []
    gp_x_values: list[Array] = []
    objective_values: list[float] = []
    trial_parameters: dict[int, dict[str, float]] = {}
    passes: dict[int, Any] = {}
    proxy_trial_metrics: dict[int, dict[str, float | bool]] = {}
    records: list[dict[str, Any]] = []
    timings: dict[str, float] = {
        # _candidate._evaluate_candidate performs both the Kalman pass and all per-row
        # likelihood/diagnostic calculations, so this deliberately does not
        # call the result "filtering_seconds" or "scoring_seconds".
        "candidate_evaluation_seconds": 0.0,
        "acquisition_seconds": 0.0,
    }
    total_started = perf_counter()

    for trial_id in range(search.total_trials):
        acquisition_source = "initial-design"
        if trial_id < len(design):
            point = design[trial_id]
        else:
            acquisition_started = perf_counter()
            point, acquisition_source = _optimization._fit_and_acquire(
                gp_x_values,
                objective_values,
                observed_x_values,
                rng,
                search,
            )
            timings["acquisition_seconds"] += perf_counter() - acquisition_started
        # Avoid exact duplicates after finite-precision transforms.
        if observed_x_values:
            point = np.asarray(point, dtype=float)
            if (
                np.min(np.linalg.norm(np.asarray(observed_x_values) - point, axis=1))
                <= 1e-8
            ):
                point = rng.random(5)
                acquisition_source = "random-fallback"
        parameters = _optimization._decode(point, lows, highs)
        trial_parameters[trial_id] = parameters
        started = perf_counter()
        candidate = _candidate._evaluate_candidate(
            prepared,
            q_inflow=parameters["q_inflow"],
            q_storage_override=parameters["q_storage"],
            q_outflow_override=parameters["q_outflow"],
            r_override=np.diag([parameters["r_storage"], parameters["r_outflow"]]),
            windows=windows,
            settings=options,
            plan=plan,
        )
        timings["candidate_evaluation_seconds"] += perf_counter() - started
        passes[trial_id] = candidate
        violation, calibrated = _calibration_violation(
            candidate.physical, options, search
        )
        proxy_metrics = _proxy._proxy_metrics(
            candidate.filter_result, prepared, upstream_proxy, search, plan
        )
        proxy_trial_metrics[trial_id] = proxy_metrics
        if (
            search.proxy_require_gate
            and proxy_metrics["proxy_available"]
            and not proxy_metrics["proxy_gate_passed"]
        ):
            violation += 1.0
        losses = np.asarray(
            [candidate.window_rows[w.name].get("joint_nlpd", np.nan) for w in windows],
            dtype=float,
        )
        valid, validity_reasons = _hard_validity(candidate, windows, options)
        mean_loss = float(np.dot(weights, losses)) if valid else np.inf
        standard_error = (
            _optimization._weighted_standard_error(losses, weights) if valid else np.inf
        )
        robust_objective = (
            mean_loss + search.one_standard_error_weight * standard_error
            if valid
            else np.inf
        )
        rejection = "; ".join(validity_reasons)
        row = {
            "trial_id": trial_id,
            **parameters,
            "inflow_increment_sd": sqrt(parameters["q_inflow"] * 3600.0),
            "acquisition_source": acquisition_source,
            "objective": mean_loss,
            "window_standard_error": standard_error,
            "robust_objective": robust_objective,
            "calibration_violation": violation,
            "calibrated": valid and calibrated,
            "eligible": valid,
            "rejection_reasons": rejection,
            "max_elapsed_lag_autocorrelation": candidate.physical.get(
                "max_material_elapsed_lag_autocorrelation", np.nan
            ),
            "joint_nis": candidate.physical.get("joint_nis", np.nan),
            "storage_nis": candidate.physical.get("storage_nis", np.nan),
            "outflow_nis": candidate.physical.get("outflow_nis", np.nan),
            # Keep only the calibration metrics needed to interpret a trial.
            # Per-row and physical plausibility details are intentionally not
            # copied into the report; they are available from a separate
            # evaluation when needed.
        }
        records.append(row)
        observed_x_values.append(np.asarray(point, dtype=float))
        if valid:
            gp_x_values.append(np.asarray(point, dtype=float))
            objective_values.append(robust_objective)
        # Window aggregates and physical metrics above are sufficient for
        # selection. Release the full filter history after each trial.
        candidate.filter_result = None
        candidate.diagnostics.clear()

    candidate_frame = pd.DataFrame(records)
    valid_frame = candidate_frame[candidate_frame["eligible"]].copy()
    if valid_frame.empty:
        raise BayesianTuningError("no candidate passed the hard validity gates")
    proxy_passed = pd.Series(
        {
            trial_id: bool(metrics["proxy_gate_passed"])
            for trial_id, metrics in proxy_trial_metrics.items()
        }
    )
    proxy_gate_restricted = bool(
        search.proxy_require_gate
        and upstream_proxy is not None
        and valid_frame.trial_id.map(proxy_passed).fillna(False).any()
    )
    selection_frame = (
        valid_frame[valid_frame.trial_id.map(proxy_passed).fillna(False)].copy()
        if proxy_gate_restricted
        else valid_frame
    )
    best_trial = int(
        selection_frame.sort_values(["robust_objective", "trial_id"]).iloc[0].trial_id
    )
    best_losses = np.asarray(
        [passes[best_trial].window_rows[w.name]["joint_nlpd"] for w in windows],
        dtype=float,
    )
    thresholds: dict[int, float] = {}
    excess: dict[int, float] = {}
    for trial_id in selection_frame.trial_id.astype(int):
        losses = np.asarray(
            [passes[trial_id].window_rows[w.name]["joint_nlpd"] for w in windows],
            dtype=float,
        )
        differences = losses - best_losses
        excess[trial_id] = float(np.dot(weights, differences))
        thresholds[trial_id] = max(
            _diagnostics._paired_standard_error(differences, weights, options),
            options.practical_equivalence_tolerance,
        )
    competitive = tuple(
        trial_id
        for trial_id in selection_frame.trial_id.astype(int)
        if excess[trial_id] <= thresholds[trial_id] + 1e-12
    )
    if not competitive:
        competitive = (best_trial,)
    selected_trial = min(
        competitive,
        key=lambda trial_id: (
            float(
                candidate_frame.loc[
                    candidate_frame.trial_id == trial_id, "calibration_violation"
                ].iloc[0]
            ),
            float(
                np.linalg.norm(
                    _optimization._encode(trial_parameters[trial_id], lows, highs)
                    - base_point
                )
            ),
            float(
                candidate_frame.loc[
                    candidate_frame.trial_id == trial_id, "robust_objective"
                ].iloc[0]
            ),
            trial_id,
        ),
    )
    candidate_frame["paired_excess_loss"] = candidate_frame.trial_id.map(excess)
    candidate_frame["selection_threshold"] = candidate_frame.trial_id.map(thresholds)
    candidate_frame["competitive"] = candidate_frame.trial_id.isin(competitive)
    candidate_frame["selected"] = candidate_frame.trial_id == selected_trial
    candidate_frame["selected_reason"] = np.where(
        candidate_frame["selected"],
        "lowest calibration violation in competitive set",
        "",
    )
    candidate_frame = candidate_frame.sort_values(
        ["selected", "competitive", "robust_objective", "trial_id"],
        ascending=[False, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    selected = trial_parameters[selected_trial]
    selected_objective = float(
        candidate_frame.loc[
            candidate_frame.trial_id == selected_trial, "robust_objective"
        ].iloc[0]
    )
    selected_calibration_violation = float(
        candidate_frame.loc[
            candidate_frame.trial_id == selected_trial, "calibration_violation"
        ].iloc[0]
    )
    selection_reason = (
        "lowest calibration violation among paired one-standard-error "
        "competitive trials"
    )
    selected_config = _proposed_config(
        base_config,
        selected,
        str(proposed_configuration_version),
        selected_trial=selected_trial,
        competitive_trials=competitive,
        selected_objective=selected_objective,
        selected_calibration_violation=selected_calibration_violation,
    )
    window_columns = (
        "window",
        "start",
        "end",
        "storage_nlpd",
        "joint_nlpd",
        "joint_nis",
        "storage_nis",
        "outflow_nis",
        "coverage",
        "storage_count",
        "joint_count",
    )
    window_frame = pd.DataFrame(
        [
            {
                "trial_id": selected_trial,
                **{
                    name: passes[selected_trial].window_rows[window.name].get(name)
                    for name in window_columns
                },
            }
            for window in windows
        ]
    )
    if upstream_proxy is None:
        proxy_frame = pd.DataFrame()
    else:
        proxy_columns = (
            "proxy_aligned_count",
            "proxy_best_lag_seconds",
            "proxy_shape_correlation",
            "proxy_change_correlation",
            "proxy_shape_rmse",
            "proxy_gate_passed",
        )
        proxy_frame = pd.DataFrame(
            [
                {
                    "trial_id": trial_id,
                    **{name: metrics[name] for name in proxy_columns},
                    "competitive": trial_id in competitive,
                    "selected": trial_id == selected_trial,
                }
                for trial_id, metrics in proxy_trial_metrics.items()
            ]
        )
    warnings: list[str] = []
    if (
        upstream_proxy is not None
        and search.proxy_require_gate
        and not proxy_gate_restricted
    ):
        warnings.append(
            "no valid Bayesian trial passed the upstream-proxy shape/timing gate; "
            "selection was made from innovation-valid trials and is diagnostic only"
        )
    elif proxy_gate_restricted:
        warnings.append(
            "upstream-proxy shape/timing gate restricted selection; proxy was not "
            "treated as a total-inflow target"
        )
    if any(
        abs(selected[name] - bound) <= 1e-12
        for name, bound in zip(_types._PARAMETER_NAMES, lows, strict=True)
    ) or any(
        abs(selected[name] - bound) <= 1e-12
        for name, bound in zip(_types._PARAMETER_NAMES, highs, strict=True)
    ):
        warnings.append(
            "selected parameter is at the edge of the Bayesian search bounds"
        )
    if len(windows) < 5:
        warnings.append(
            "fewer than five valid windows are available; uncertainty is weak"
        )
    if not bool(
        candidate_frame.loc[
            candidate_frame.trial_id == selected_trial, "calibrated"
        ].iloc[0]
    ):
        warnings.append(
            "selected trial does not satisfy all innovation calibration checks"
        )
    timings["total_seconds"] = perf_counter() - total_started
    return BayesianTuningResult(
        selected_config=selected_config,
        selected_parameters=selected,
        candidate_summary=candidate_frame,
        window_diagnostics=window_frame,
        competitive_trial_ids=tuple(int(t) for t in competitive),
        selected_objective=selected_objective,
        selection_threshold=float(thresholds.get(selected_trial, 0.0)),
        selection_reason=selection_reason,
        proxy_diagnostics=proxy_frame,
        warnings=tuple(dict.fromkeys(warnings)),
        timing_seconds=timings,
    )


def evaluate_configuration(
    storage: pd.Series,
    discharge: pd.Series,
    config: ReservoirConfig,
    *,
    evaluation_window: ValidationWindow | None = None,
    settings: BayesianEvaluationSettings | None = None,
) -> ConfigEvaluationResult:
    """Evaluate one frozen configuration on an independent validation period.

    This is an assessment operation only: it does not search or modify the
    supplied configuration and is suitable for held-out post-search checks.
    """

    options = settings or BayesianEvaluationSettings()
    if not isinstance(config, ReservoirConfig):
        raise TypeError("config must be a ReservoirConfig")
    prepared = _preparation._prepare(storage, discharge, config)
    if evaluation_window is None:
        evaluation_window = ValidationWindow(
            "evaluation",
            prepared.index[0],
            prepared.index[-1] + pd.Timedelta(nanoseconds=1),
        )
    windows = _preparation._validate_windows((evaluation_window,))
    q_inflow = float(config.q[1, 1])
    candidate = _candidate._evaluate_candidate(
        prepared,
        q_inflow=q_inflow,
        windows=windows,
        settings=options,
        plan=_preparation._diagnostic_plan(prepared, windows, options),
    )
    eligible, validity_reasons = _hard_validity(candidate, windows, options)
    calibration_warnings = _calibration_warnings(candidate.physical, options)
    window_frame = pd.DataFrame(
        [
            {
                "window": window.name,
                "start": window.start,
                "end": window.end,
                **candidate.window_rows[window.name],
            }
            for window in windows
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "q_storage": float(config.q[0, 0]),
                "q_inflow": q_inflow,
                "q_outflow": float(config.q[2, 2]),
                "r_storage": float(config.r[0, 0]),
                "r_outflow": float(config.r[1, 1]),
                "eligible": eligible,
                "calibrated": eligible and not calibration_warnings,
                "rejection_reasons": "; ".join(validity_reasons),
                **candidate.physical,
            }
        ]
    )
    return ConfigEvaluationResult(
        config=config,
        candidate_summary=summary,
        window_diagnostics=window_frame,
        warnings=tuple(dict.fromkeys((*validity_reasons, *calibration_warnings))),
    )


__all__ = [
    "BayesianEvaluationSettings",
    "BayesianTuningError",
    "BayesianTuningResult",
    "BayesianTuningSettings",
    "ConfigEvaluationResult",
    "ValidationWindow",
    "evaluate_configuration",
    "tune_inflow_noise_bayesian",
]
