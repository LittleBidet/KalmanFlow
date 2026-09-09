"""Small validation helpers shared by package internals."""

from __future__ import annotations

from math import isinf
from numbers import Real
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
    if not np.all(np.isfinite(array)) or not np.allclose(
        array, array.T, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(f"{name} must be finite and symmetric")
    eigenvalues = np.linalg.eigvalsh(array)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(eigenvalues))))
    if np.min(eigenvalues) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if positive_diagonal and np.any(np.diag(array) <= 0.0):
        raise ValueError(f"{name} diagonal entries must be strictly positive")
    array.setflags(write=False)
    return array


def bounded_integer(value: object, *, name: str, minimum: int) -> int:
    """Return an integer at or above ``minimum`` without lossy coercion."""

    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer")
    try:
        integer = index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return integer


def nonempty_string(value: object, *, name: str) -> str:
    """Return a nonempty string without coercing unrelated objects."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def measurement_value(value: object, *, name: str) -> float:
    """Return a finite measurement or NaN, rejecting infinities and coercion."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if isinf(result):
        raise ValueError(f"{name} must be finite or NaN")
    return result
