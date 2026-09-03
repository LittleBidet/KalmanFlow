from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np

from ..kalman import kalman_filter
from . import _diagnostics
from ._preparation import _diagnostic_plan
from ._types import (
    BayesianEvaluationSettings,
    ValidationWindow,
    _DiagnosticPlan,
    _Pass,
    _Prepared,
)

Array = np.ndarray


def _evaluate_candidate(
    prepared: _Prepared,
    *,
    q_inflow: float,
    windows: Sequence[ValidationWindow],
    settings: BayesianEvaluationSettings,
    r_override: Array | None = None,
    q_storage_override: float | None = None,
    q_outflow_override: float | None = None,
    plan: _DiagnosticPlan | None = None,
) -> _Pass:
    q_storage = (
        prepared.q_storage if q_storage_override is None else float(q_storage_override)
    )
    q_outflow = (
        prepared.q_outflow if q_outflow_override is None else float(q_outflow_override)
    )
    q_discrete = (
        q_storage * prepared.storage_basis
        + q_inflow * prepared.inflow_basis
        + q_outflow * prepared.outflow_basis
    )
    r = prepared.r if r_override is None else np.asarray(r_override, dtype=float)
    reasons: list[str] = []
    try:
        result = kalman_filter(
            prepared.observations,
            initial_mean=prepared.initial_mean,
            initial_covariance=prepared.initial_covariance,
            transition_matrix=prepared.transitions,
            process_covariance=q_discrete,
            observation_matrix=prepared.model.observation_matrix,
            observation_covariance=r,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        reasons.append(f"filter failed: {error}")
        return _Pass(
            filter_result=None,
            window_rows={
                w.name: {"storage_nlpd": np.nan, "storage_count": 0}
                for w in windows
            },
            physical={},
            reasons=reasons,
        )
    if not (
        np.isfinite(result.filtered_means).all()
        and np.isfinite(result.predicted_means).all()
        and np.isfinite(result.predicted_covariances).all()
    ):
        reasons.append("nonfinite filtered or predicted state")
    if plan is None:
        plan = _diagnostic_plan(prepared, windows, settings)
    count = len(prepared.timestamps)
    diagnostics = {
        key: np.full(count, np.nan, dtype=float)
        for key in (
            "storage_innovation",
            "primary_nlpd",
            "joint_nlpd",
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
    diagnostics["joint_components"] = prepared.finite_observations.sum(axis=1).astype(
        float
    )
    regularization_count = 0
    for row, timestamp in enumerate(prepared.timestamps):
        finite = prepared.finite_observations[row]
        innovation = result.innovations[row]
        covariance = result.innovation_covariances[row]
        if finite[0] and np.isfinite(innovation[0]):
            diagnostics["storage_innovation"][row] = innovation[0]
        if finite.any():
            observed = np.flatnonzero(finite)
            vector = innovation[observed]
            matrix = covariance[np.ix_(observed, observed)]
            joint, quad, jitter = _diagnostics._stable_logpdf(
                vector, matrix, settings.max_jitter_fraction
            )
            if joint is None:
                reasons.append(
                    "non-positive-definite scored covariance at "
                    f"{timestamp.isoformat()}"
                )
            else:
                diagnostics["joint_nlpd"][row] = joint
                diagnostics["joint_nis"][row] = quad
                diagnostics["jitter"][row] = jitter
                regularization_count += int(jitter > 0.0)
            for component, key in ((0, "storage"), (1, "outflow")):
                if finite[component]:
                    variance = covariance[component, component]
                    if np.isfinite(variance) and variance > 0.0:
                        z = innovation[component] / sqrt(variance)
                        diagnostics[f"{key}_z"][row] = z
                        diagnostics[f"{key}_nis"][row] = z * z
            if finite[0]:
                if finite[1]:
                    try:
                        conditional = _diagnostics._storage_conditional_nlpd(
                            innovation[0],
                            innovation[1],
                            covariance,
                            max_jitter_fraction=settings.max_jitter_fraction,
                        )
                        soo = covariance[1, 1]
                        conditional_variance = (
                            covariance[0, 0] - covariance[0, 1] * covariance[1, 0] / soo
                        )
                        conditional_innovation = (
                            innovation[0] - covariance[0, 1] * innovation[1] / soo
                        )
                        diagnostics["conditional_storage_z"][row] = (
                            conditional_innovation
                            / sqrt(max(conditional_variance, np.finfo(float).tiny))
                        )
                    except ValueError as error:
                        reasons.append(str(error))
                        conditional = np.nan
                    diagnostics["conditional_storage_nis"][row] = (
                        diagnostics["conditional_storage_z"][row] ** 2
                    )
                    diagnostics["primary_nlpd"][row] = conditional
                else:
                    try:
                        diagnostics["primary_nlpd"][row] = (
                            _diagnostics._marginal_predictive_nlpd(
                                innovation[0],
                                covariance[0, 0],
                                max_jitter_fraction=settings.max_jitter_fraction,
                            )
                        )
                    except ValueError as error:
                        reasons.append(str(error))
    window_rows: dict[str, dict[str, Any]] = {}
    for window, window_mask in zip(windows, plan.window_masks, strict=True):
        window_rows[window.name] = _diagnostics._aggregate_arrays(
            diagnostics, window_mask, window, plan.score_mask
        )
    validation_mask = plan.score_mask & np.logical_or.reduce(plan.window_masks)
    physical = _diagnostics._physical_metrics_arrays(
        result, diagnostics, prepared.timestamps, validation_mask, settings
    )
    if regularization_count > settings.max_regularized_steps:
        reasons.append("excessive covariance regularization")
    return _Pass(
        filter_result=result,
        window_rows=window_rows,
        physical=physical,
        reasons=list(dict.fromkeys(reasons)),
        diagnostics=diagnostics,
    )
