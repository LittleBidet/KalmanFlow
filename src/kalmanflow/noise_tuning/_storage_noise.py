"""Storage reading noise measured directly from the observations.

White reading error ``e`` with variance ``r`` turns consecutive storage
changes ``ΔS_t = ΔS_true,t + e_t - e_(t-1)`` into negatively correlated
pairs: ``Cov(ΔS_t, ΔS_(t-1)) = Cov(true changes) - r``. Over one short step
the true storage change is smooth, so ``r ≈ -Cov(ΔS_t, ΔS_(t-1))``.

The estimate assumes reading errors are independent between readings.
Persistent inflow changes bias it low, and held or quantized readings make
it approximate. A negative result is reported as zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._preparation import _timestamp_seconds, _validate_index, _validate_windows
from ._types import ValidationWindow

Array = np.ndarray

_MIN_PAIRS = 100
_DISAGREEMENT_RATIO = 4.0


def estimate_storage_noise(
    storage: pd.Series,
    validation_windows: Sequence[ValidationWindow] | None = None,
) -> pd.DataFrame:
    """Estimate storage reading noise (``r_storage``) from the data, per window.

    Returns one row per window and a pooled ``"all"`` row, with the variance
    estimate ``r_storage`` (squared storage units), ``reading_sd``, the number
    of evenly spaced change ``pairs`` used, and ``unchanged_share``, the share
    of those changes that are exactly zero (a sign of held or quantized
    readings). Estimates need at least 100 pairs; fewer give NaN. Windows
    whose estimates differ by much more than a factor of 4 suggest a change
    in the gauge or its processing, and are better tuned separately.
    """

    if not isinstance(storage, pd.Series):
        raise TypeError("storage must be a pandas Series")
    index = _validate_index(storage.index)
    windows = (
        () if validation_windows is None else _validate_windows(validation_windows)
    )
    return _storage_noise_frame(
        index,
        storage.to_numpy(dtype=float),
        windows,
        np.ones(len(index), dtype=bool),
    )


def _storage_noise_frame(
    index: pd.DatetimeIndex,
    storage: Array,
    windows: Sequence[ValidationWindow],
    allowed: Array,
) -> pd.DataFrame:
    """Build the per-window and pooled table over the ``allowed`` rows."""

    seconds = _timestamp_seconds(index)
    rows = []
    union = np.zeros(len(index), dtype=bool) if windows else allowed.copy()
    for window in windows:
        mask = allowed & window.mask(index)
        union |= mask
        rows.append(_row(window.name, window.start, window.end, seconds, storage, mask))
    pooled = index[union]
    start, end = (pooled[0], pooled[-1]) if len(pooled) else (pd.NaT, pd.NaT)
    rows.append(_row("all", start, end, seconds, storage, union))
    return pd.DataFrame(rows)


def _row(
    name: str,
    start: object,
    end: object,
    seconds: Array,
    storage: Array,
    mask: Array,
) -> dict[str, object]:
    estimate, pairs, unchanged = _lag1_storage_noise(seconds, storage, mask)
    return {
        "window": name,
        "start": start,
        "end": end,
        "r_storage": estimate,
        "reading_sd": float(np.sqrt(estimate)),
        "pairs": pairs,
        "unchanged_share": unchanged,
    }


def _lag1_storage_noise(
    seconds: Array, storage: Array, mask: Array
) -> tuple[float, int, float]:
    """Return ``(r_storage, pairs, unchanged_share)`` from consecutive triples.

    A triple is three consecutive finite readings inside ``mask`` with equal
    elapsed times, giving one pair of adjacent changes.
    """

    usable = np.isfinite(storage) & mask
    elapsed = np.diff(seconds)
    triple = (
        usable[2:] & usable[1:-1] & usable[:-2] & (elapsed[1:] == elapsed[:-1])
    )
    pairs = int(triple.sum())
    if pairs < _MIN_PAIRS:
        return np.nan, pairs, np.nan
    later = (storage[2:] - storage[1:-1])[triple]
    earlier = (storage[1:-1] - storage[:-2])[triple]
    mean = float(np.mean(np.concatenate((later, earlier))))
    covariance = float(np.mean((later - mean) * (earlier - mean)))
    return max(-covariance, 0.0), pairs, float(np.mean(later == 0.0))


def _pooled_estimate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return float(frame.loc[frame["window"] == "all", "r_storage"].iloc[0])


def _storage_noise_warnings(frame: pd.DataFrame) -> list[str]:
    """Explain an unusable estimate or windows that disagree about it."""

    if frame.empty:
        return []
    warnings = []
    pooled = _pooled_estimate(frame)
    if not (np.isfinite(pooled) and pooled > 0.0):
        warnings.append(
            "r_storage could not be estimated from the data (too few evenly "
            "spaced storage readings, or no sign of reading noise); "
            "r_storage_multiplier_bounds were used instead"
        )
    windows = frame[(frame["window"] != "all") & np.isfinite(frame["r_storage"])]
    if len(windows) < 2:
        return warnings
    low = windows.loc[windows["r_storage"].idxmin()]
    high = windows.loc[windows["r_storage"].idxmax()]
    if high.r_storage > 0.0 and (
        low.r_storage == 0.0 or high.r_storage / low.r_storage > _DISAGREEMENT_RATIO
    ):
        warnings.append(
            "storage reading noise differs between validation windows "
            f"({low.window}: r_storage {low.r_storage:.3g}; {high.window}: "
            f"{high.r_storage:.3g}), so one r_storage cannot fit them all. "
            "Compare periods with kalmanflow.estimate_storage_noise(storage, "
            "validation_windows) and consider tuning them separately"
        )
    return warnings
