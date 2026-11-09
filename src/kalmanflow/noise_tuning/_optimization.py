"""Bounded local likelihood search. Requires SciPy (``kalmanflow[tuning]``)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

Array = np.ndarray


@dataclass(frozen=True)
class _SearchOutcome:
    point: Array
    evaluations: int
    converged: bool
    message: str


def _local_search(
    objective: Callable[[Array], float], start: Array, max_evaluations: int
) -> _SearchOutcome:
    """Minimize ``objective`` over the unit cube from ``start`` with L-BFGS-B.

    The objective is deterministic and smooth in log-parameter space, so a
    quasi-Newton method with finite-difference gradients converges in far
    fewer evaluations than a global surrogate search. L-BFGS-B cannot step
    through non-finite values; a search that meets one is reported as not
    converged instead of trusting its stopping message.
    """

    evaluations = 0
    met_invalid_value = False

    def counted(point: Array) -> float:
        nonlocal evaluations, met_invalid_value
        evaluations += 1
        value = objective(point)
        met_invalid_value |= not np.isfinite(value)
        return value

    # Differences of infinite values are expected here and handled below.
    with np.errstate(invalid="ignore"):
        result = minimize(
            counted,
            np.asarray(start, dtype=float),
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * len(start),
            options={"maxfun": max_evaluations},
        )
    if met_invalid_value:
        return _SearchOutcome(
            point=np.asarray(result.x),
            evaluations=evaluations,
            converged=False,
            message="stopped at numerically invalid noise settings",
        )
    return _SearchOutcome(
        point=np.asarray(result.x),
        evaluations=evaluations,
        converged=bool(result.success),
        message=str(result.message),
    )
