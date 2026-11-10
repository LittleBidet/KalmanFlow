"""Storage reading noise measured directly from the observations.

White reading error ``e`` with variance ``r`` turns consecutive storage
changes ``ΔS_t = ΔS_true,t + e_t - e_(t-1)`` into negatively correlated
pairs: ``Cov(ΔS_t, ΔS_(t-1)) = Cov(true changes) - r``. Over one short step
the true storage change is smooth, so ``r ≈ -Cov(ΔS_t, ΔS_(t-1))``.

Storm-driven storage changes are persistent, so they add positive
covariance and can hide the noise entirely. The estimate is therefore taken
in one-week blocks, and the median across blocks is used: storm weeks are
usually a minority, so they cannot drag the result to zero. The estimate
assumes reading errors are independent between readings; held or quantized
readings make it approximate. A negative median is reported as zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ._preparation import _timestamp_seconds, _validate_index, _validate_windows
from ._types import ValidationWindow

Array = np.ndarray

_MIN_PAIRS = 100
_BLOCK_SECONDS = 7 * 86_400
_MIN_BLOCK_PAIRS = 20
_DISAGREEMENT_RATIO = 4.0
_POOLED = "all"


def estimate_storage_noise(
    storage: pd.Series,
    validation_windows: Sequence[ValidationWindow] | None = None,
) -> pd.DataFrame:
    """Estimate storage reading noise (``r_storage``) from the data, per window.

    Returns one row per window and a pooled ``"all"`` row, so no window may
    be named ``"all"``. Each row has the variance estimate ``r_storage``
    (squared storage units), ``reading_sd``, the number of evenly spaced
    change ``pairs`` and week-long ``blocks`` used, and ``unchanged_share``,
    the share of changes that are exactly zero (a sign of held or quantized
    readings). Each estimate is the median of weekly
    estimates and needs at least 100 pairs; fewer give NaN. Windows whose
    estimates differ by much more than a factor of 4 suggest a change in the
    gauge or its processing, and are better tuned separately.
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

    if any(window.name == _POOLED for window in windows):
        raise ValueError(
            f"validation window name {_POOLED!r} is reserved for the pooled "
            "storage-noise row"
        )
    seconds = _timestamp_seconds(index)
    rows = []
    union = np.zeros(len(index), dtype=bool) if windows else allowed.copy()
    for window in windows:
        mask = allowed & window.mask(index)
        union |= mask
        rows.append(_row(window.name, window.start, window.end, seconds, storage, mask))
    pooled = index[union]
    start, end = (pooled[0], pooled[-1]) if len(pooled) else (pd.NaT, pd.NaT)
    rows.append(_row(_POOLED, start, end, seconds, storage, union))
    return pd.DataFrame(rows)


def _row(
    name: str,
    start: object,
    end: object,
    seconds: Array,
    storage: Array,
    mask: Array,
) -> dict[str, object]:
    estimate, pairs, blocks, unchanged = _block_median_storage_noise(
        seconds, storage, mask
    )
    return {
        "window": name,
        "start": start,
        "end": end,
        "r_storage": estimate,
        "reading_sd": float(np.sqrt(estimate)),
        "pairs": pairs,
        "blocks": blocks,
        "unchanged_share": unchanged,
    }


def _block_median_storage_noise(
    seconds: Array, storage: Array, mask: Array
) -> tuple[float, int, int, float]:
    """Return ``(r_storage, pairs, blocks, unchanged_share)``.

    A pair is two adjacent changes from three consecutive finite readings
    inside ``mask`` with equal elapsed times. Pairs are grouped into weeks
    counted from the first reading, so every window uses the same calendar
    blocks; weeks with fewer than 20 pairs are skipped.
    """

    usable = np.isfinite(storage) & mask
    elapsed = np.diff(seconds)
    triple = (
        usable[2:] & usable[1:-1] & usable[:-2] & (elapsed[1:] == elapsed[:-1])
    )
    pairs = int(triple.sum())
    if pairs < _MIN_PAIRS:
        return np.nan, pairs, 0, np.nan
    later = (storage[2:] - storage[1:-1])[triple]
    earlier = (storage[1:-1] - storage[:-2])[triple]
    week = ((seconds[2:] - seconds[0]) // _BLOCK_SECONDS)[triple]
    estimates = [
        _negative_lag1_covariance(later[in_week], earlier[in_week])
        for in_week in (week == key for key in np.unique(week))
        if in_week.sum() >= _MIN_BLOCK_PAIRS
    ]
    if not estimates:
        return np.nan, pairs, 0, np.nan
    estimate = max(float(np.median(estimates)), 0.0)
    return estimate, pairs, len(estimates), float(np.mean(later == 0.0))


def _negative_lag1_covariance(later: Array, earlier: Array) -> float:
    mean = float(np.mean(np.concatenate((later, earlier))))
    return -float(np.mean((later - mean) * (earlier - mean)))


def _pooled_estimate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    return float(frame.loc[frame["window"] == _POOLED, "r_storage"].iloc[0])


def _storage_noise_warnings(frame: pd.DataFrame) -> list[str]:
    """Explain an unusable estimate, unscored windows, or windows that disagree."""

    if frame.empty:
        return []
    warnings = []
    pooled = _pooled_estimate(frame)
    fallback = "r_storage_multiplier_bounds were used instead"
    if not np.isfinite(pooled):
        warnings.append(
            "r_storage could not be estimated: fewer than 100 evenly spaced "
            f"storage change pairs (or no week with 20); {fallback}"
        )
    elif pooled == 0.0:
        warnings.append(
            "r_storage could not be estimated: storage changes showed no "
            "reading-noise signal (they were not negatively correlated), which "
            f"often happens when storm-driven changes dominate; {fallback}"
        )
    windows = frame[(frame["window"] != _POOLED) & np.isfinite(frame["r_storage"])]
    zero = windows[windows["r_storage"] == 0.0]
    if len(zero):
        warnings.append(
            f"{', '.join(zero['window'])} gave no reading-noise estimate "
            "(possibly storm-dominated, or a nearly noiseless gauge), so "
            "they were not compared with the other windows"
        )
    positive = windows[windows["r_storage"] > 0.0]
    if len(positive) < 2:
        return warnings
    low = positive.loc[positive["r_storage"].idxmin()]
    high = positive.loc[positive["r_storage"].idxmax()]
    if high.r_storage / low.r_storage > _DISAGREEMENT_RATIO:
        warnings.append(
            "storage reading noise differs between validation windows "
            f"({low.window}: r_storage {low.r_storage:.3g}; {high.window}: "
            f"{high.r_storage:.3g}), so one r_storage cannot fit them all. "
            "Compare periods with kalmanflow.estimate_storage_noise(storage, "
            "validation_windows) and consider tuning them separately"
        )
    return warnings
