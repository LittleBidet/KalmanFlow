from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from typing import Any

import numpy as np

from ..kalman import kalman_filter
from . import _diagnostics
from ._preparation import _diagnostic_plan
from ._types import (
    NoiseEvaluationSettings,
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
    settings: NoiseEvaluationSettings,
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
            window_rows={w.name: _failed_window_row(w) for w in windows},
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
    finite = prepared.finite_observations
    innovations = result.innovations
    covariances = result.innovation_covariances
    clean = _clean_rows(finite, innovations, covariances)
    _score_clean_rows(diagnostics, clean, finite, innovations, covariances)
    # Rows that need regularization, or cannot be scored, keep the scalar
    # path, so their jitter and failure reasons are unchanged.
    regularization_count = 0
    for row in np.flatnonzero(finite.any(axis=1) & ~clean):
        regularization_count += _score_row(
            diagnostics,
            int(row),
            prepared.timestamps[row],
            finite[row],
            innovations[row],
            covariances[row],
            settings,
            reasons,
        )
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


def _score_row(
    diagnostics: dict[str, Array],
    row: int,
    timestamp: Any,
    finite: Array,
    innovation: Array,
    covariance: Array,
    settings: NoiseEvaluationSettings,
    reasons: list[str],
) -> int:
    """Score one observed row with jitter and failure handling.

    Returns 1 when the joint covariance needed regularization, else 0.
    """

    regularized = 0
    if finite[0] and np.isfinite(innovation[0]):
        diagnostics["storage_innovation"][row] = innovation[0]
    observed = np.flatnonzero(finite)
    joint, quad, jitter = _diagnostics._stable_logpdf(
        innovation[observed],
        covariance[np.ix_(observed, observed)],
        settings.max_jitter_fraction,
    )
    if joint is None:
        reasons.append(
            f"non-positive-definite scored covariance at {timestamp.isoformat()}"
        )
    else:
        diagnostics["joint_nlpd"][row] = joint
        diagnostics["joint_nis"][row] = quad
        diagnostics["jitter"][row] = jitter
        regularized = int(jitter > 0.0)
    for component, key in ((0, "storage"), (1, "outflow")):
        if finite[component]:
            variance = covariance[component, component]
            if np.isfinite(variance) and variance > 0.0:
                z = innovation[component] / sqrt(variance)
                diagnostics[f"{key}_z"][row] = z
                diagnostics[f"{key}_nis"][row] = z * z
    if not finite[0]:
        return regularized
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
            diagnostics["conditional_storage_z"][row] = conditional_innovation / sqrt(
                max(conditional_variance, np.finfo(float).tiny)
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
            diagnostics["primary_nlpd"][row] = _diagnostics._marginal_predictive_nlpd(
                innovation[0],
                covariance[0, 0],
                max_jitter_fraction=settings.max_jitter_fraction,
            )
        except ValueError as error:
            reasons.append(str(error))
    return regularized


# Rows this close to singular take the scalar path, so closed-form and
# LAPACK Cholesky factors never disagree about whether a row is valid.
_PIVOT_MARGIN = 1e-10
_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _clean_rows(finite: Array, innovations: Array, covariances: Array) -> Array:
    """Mark rows whose scores need no regularization.

    For those rows the 1x1 and 2x2 densities have closed forms, so
    :func:`_score_clean_rows` computes them for all rows at once. Every other
    row with an observation is scored by :func:`_score_row`.
    """

    storage, outflow = finite[:, 0], finite[:, 1]
    a, c = covariances[:, 0, 0], covariances[:, 1, 1]
    off = covariances[:, 0, 1], covariances[:, 1, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        storage_ok = np.isfinite(innovations[:, 0]) & np.isfinite(a) & (a > 0.0)
        outflow_ok = np.isfinite(innovations[:, 1]) & np.isfinite(c) & (c > 0.0)
        b = 0.5 * (off[0] + off[1])
        l10 = b / np.sqrt(a)
        both_ok = (
            storage_ok
            & outflow_ok
            & np.isfinite(off[0])
            & np.isfinite(off[1])
            & (c - l10 * l10 > _PIVOT_MARGIN * c)
            & (a - b * b / c > _PIVOT_MARGIN * a)
        )
    return np.where(
        storage & outflow,
        both_ok,
        np.where(storage, storage_ok, outflow & outflow_ok),
    )


def _score_clean_rows(
    diagnostics: dict[str, Array],
    clean: Array,
    finite: Array,
    innovations: Array,
    covariances: Array,
) -> None:
    """Write the row diagnostics of :func:`_score_row` for ``clean`` rows.

    The formulas are the Cholesky ones :func:`_diagnostics._stable_logpdf`
    uses, so values agree with the scalar path to rounding.
    """

    storage_only = clean & finite[:, 0] & ~finite[:, 1]
    outflow_only = clean & ~finite[:, 0] & finite[:, 1]
    both = clean & finite[:, 0] & finite[:, 1]

    for rows, component in ((storage_only, 0), (outflow_only, 1)):
        root = np.sqrt(covariances[rows, component, component])
        solved = innovations[rows, component] / root
        quad = solved * solved
        diagnostics["joint_nlpd"][rows] = 0.5 * (
            _LOG_TWO_PI + 2.0 * np.log(root) + quad
        )
        diagnostics["joint_nis"][rows] = quad
    # A storage-only row's primary density is its joint density.
    diagnostics["primary_nlpd"][storage_only] = diagnostics["joint_nlpd"][storage_only]

    v0, v1 = innovations[both, 0], innovations[both, 1]
    a, c = covariances[both, 0, 0], covariances[both, 1, 1]
    raw01, raw10 = covariances[both, 0, 1], covariances[both, 1, 0]
    b = 0.5 * (raw01 + raw10)
    l00 = np.sqrt(a)
    l10 = b / l00
    l11 = np.sqrt(c - l10 * l10)
    y0 = v0 / l00
    y1 = (v1 - l10 * y0) / l11
    quad = y0 * y0 + y1 * y1
    diagnostics["joint_nlpd"][both] = 0.5 * (
        2.0 * _LOG_TWO_PI + 2.0 * (np.log(l00) + np.log(l11)) + quad
    )
    diagnostics["joint_nis"][both] = quad
    # Storage conditioned on the same row's outflow is the primary density.
    root = np.sqrt(a - b * b / c)
    solved = (v0 - b * v1 / c) / root
    diagnostics["primary_nlpd"][both] = 0.5 * (
        _LOG_TWO_PI + 2.0 * np.log(root) + solved * solved
    )
    conditional_z = (v0 - raw01 * v1 / c) / np.sqrt(
        np.maximum(a - raw01 * raw10 / c, np.finfo(float).tiny)
    )
    diagnostics["conditional_storage_z"][both] = conditional_z
    diagnostics["conditional_storage_nis"][both] = conditional_z * conditional_z

    for component, key in ((0, "storage"), (1, "outflow")):
        rows = clean & finite[:, component]
        z = innovations[rows, component] / np.sqrt(
            covariances[rows, component, component]
        )
        diagnostics[f"{key}_z"][rows] = z
        diagnostics[f"{key}_nis"][rows] = z * z
    storage_rows = clean & finite[:, 0]
    diagnostics["storage_innovation"][storage_rows] = innovations[storage_rows, 0]
    diagnostics["jitter"][clean] = 0.0


def _failed_window_row(window: ValidationWindow) -> dict[str, Any]:
    """Return the same window-row schema as a completed candidate."""

    return {
        "window": window.name,
        "start": window.start,
        "end": window.end,
        "storage_nlpd": np.nan,
        "storage_count": 0,
        "joint_nlpd": np.nan,
        "joint_count": 0,
        "joint_nis": np.nan,
        "storage_nis": np.nan,
        "outflow_nis": np.nan,
        "conditional_storage_nis": np.nan,
        "storage_bias": np.nan,
        "outflow_bias": np.nan,
        "conditional_storage_bias": np.nan,
        "scored_fraction": 0.0,
        "regularization_count": 0,
    }
