"""Local identifiability of the tuned noise terms at the selected optimum.

A flat or ridge-shaped objective means the data cannot separate some noise
terms, for example storage process noise from inflow process noise. The
curvature of the total negative log-likelihood in log-parameter space, and
the correlations implied by its inverse, make such trade-offs visible. Terms
held at a search bound are excluded: the optimum is not stationary in their
direction, so curvature there says nothing about identifiability.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

Array = np.ndarray
Objective = Callable[[Mapping[str, float]], float]

_LOG_STEP = 0.05
_STRONG_CORRELATION = 0.95
# A log standard error of one means the value is uncertain by a factor of e.
_MAX_LOG_STANDARD_ERROR = 1.0


def _log_space_hessian(
    objective: Objective, parameters: Mapping[str, float], names: Sequence[str]
) -> Array:
    """Return the central-difference Hessian with respect to the log of ``names``."""

    center = np.log([parameters[name] for name in names])
    size = len(names)
    step = _LOG_STEP

    def value(offset: Array) -> float:
        moved = dict(parameters)
        moved.update(zip(names, np.exp(center + offset), strict=True))
        return objective(moved)

    def unit(axis: int) -> Array:
        vector = np.zeros(size)
        vector[axis] = step
        return vector

    middle = value(np.zeros(size))
    hessian = np.empty((size, size))
    for i in range(size):
        hessian[i, i] = (value(unit(i)) - 2.0 * middle + value(-unit(i))) / step**2
    for i, j in combinations(range(size), 2):
        cross = (
            value(unit(i) + unit(j))
            - value(unit(i) - unit(j))
            - value(-unit(i) + unit(j))
            + value(-unit(i) - unit(j))
        ) / (4.0 * step**2)
        hessian[i, j] = hessian[j, i] = cross
    return hessian


def _identifiability_report(
    hessian: Array, names: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Summarize how well the data pin down each log noise term.

    ``hessian`` is taken of the total negative log-likelihood over every
    scored component, not of the window-weighted objective, so it is the
    observed Fisher information of the log parameters whatever the window
    sizes and weights. Its inverse gives approximate standard errors and
    correlations. They assume a correct Gaussian model; heavy-tailed
    innovations make them optimistic.
    """

    names = list(names)
    information = np.asarray(hessian, dtype=float)
    frame = pd.DataFrame(np.nan, index=names, columns=["curvature", *names])
    frame.index.name = "parameter"
    frame["curvature"] = np.diag(hessian)
    warnings = []
    positive_definite = np.isfinite(information).all() and (
        float(np.min(np.linalg.eigvalsh(information))) > 0.0
    )
    if positive_definite:
        covariance = np.linalg.inv(information)
        log_standard_error = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(log_standard_error, log_standard_error)
        frame[names] = correlation
        warnings += [
            f"{names[i]} and {names[j]} are strongly confounded "
            f"(correlation {correlation[i, j]:+.2f})"
            for i, j in combinations(range(len(names)), 2)
            if abs(correlation[i, j]) > _STRONG_CORRELATION
        ]
    else:
        # Without a positive-definite information matrix only each term's own
        # curvature is usable, and correlations are unavailable. Infinite
        # curvature comes from a probe that failed the validity checks, so it
        # says nothing about how well the term is identified.
        diagonal = np.diag(information)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_standard_error = np.where(
                ~np.isfinite(diagonal),
                np.nan,
                np.where(diagonal > 0.0, 1.0 / np.sqrt(diagonal), np.inf),
            )
        warnings.append(
            "the objective is not strictly convex at the selected optimum, so "
            "parameter correlations are unavailable"
        )
        unknown = [
            name
            for name, curvature in zip(names, diagonal, strict=True)
            if not np.isfinite(curvature)
        ]
        if unknown:
            warnings.insert(
                0,
                f"{', '.join(unknown)} could not be checked: a step of "
                f"{_LOG_STEP:g} in log space reached noise settings that fail "
                "the validity checks",
            )
    frame.insert(1, "log_standard_error", log_standard_error)
    poorly_identified = [
        name
        for name, error in zip(names, log_standard_error, strict=True)
        if not np.isnan(error) and not error <= _MAX_LOG_STANDARD_ERROR
    ]
    if poorly_identified:
        warnings.insert(
            0,
            f"{', '.join(poorly_identified)} are poorly identified by these data "
            "(each is uncertain by more than a factor of e)",
        )
    return frame, warnings
