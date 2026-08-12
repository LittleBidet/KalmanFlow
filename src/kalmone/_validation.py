"""Small validation helpers shared by package internals."""

from __future__ import annotations

from operator import index

import numpy as np
from numpy.typing import ArrayLike


def readonly_array(value: ArrayLike) -> np.ndarray:
    """Return a private, read-only floating-point array."""

    array = np.asarray(value, dtype=float).copy()
    array.setflags(write=False)
    return array


def covariance_array(
    value: ArrayLike,
    *,
    name: str,
    shape: tuple[int, int],
    positive_diagonal: bool = False,
) -> np.ndarray:
    """Validate a covariance matrix and return a read-only copy."""

    array = np.asarray(value, dtype=float).copy()
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}")
    if not np.all(np.isfinite(array)) or not np.allclose(array, array.T):
        raise ValueError(f"{name} must be finite and symmetric")
    if np.min(np.linalg.eigvalsh(array)) < -1e-12:
        raise ValueError(f"{name} must be positive semidefinite")
    if positive_diagonal and np.any(np.diag(array) <= 0.0):
        raise ValueError(f"{name} diagonal entries must be strictly positive")
    array.setflags(write=False)
    return array


def bounded_integer(value: object, *, name: str, minimum: int) -> int:
    """Return an integer at or above ``minimum`` without lossy coercion."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        integer = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer
