"""Timezone-safe helpers for ordered elapsed-time calculations."""

from __future__ import annotations

from datetime import UTC, datetime
from math import isfinite


def to_utc(timestamp: datetime) -> datetime:
    """Return a timezone-aware timestamp converted to UTC."""
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return timestamp.astimezone(UTC)


def validate_timestamp_precision(timestamp: datetime) -> None:
    """Reject timestamp values finer than the checkpoint's public precision.

    Python ``datetime`` values are microsecond-precise.  Pandas timestamps can
    carry an additional nanosecond remainder, which the compact checkpoint
    format cannot represent without changing the public timestamp type.  Keep
    that precision loss from occurring by rejecting such values at every
    observation and state boundary.
    """

    nanosecond_remainder = int(getattr(timestamp, "nanosecond", 0))
    if nanosecond_remainder:
        raise ValueError(
            "timestamps finer than microsecond precision are not supported"
        )


def elapsed_seconds(
    later: datetime, earlier: datetime, *, allow_zero: bool = False
) -> float:
    """Return elapsed seconds between two timestamps.

    Both timestamps must include time-zone information. By default ``later``
    must be after ``earlier``; set ``allow_zero`` to permit equal timestamps.
    """
    elapsed = (to_utc(later) - to_utc(earlier)).total_seconds()
    if not isfinite(elapsed) or (elapsed < 0.0 if allow_zero else elapsed <= 0.0):
        raise ValueError(
            "elapsed time must be nonnegative and finite"
            if allow_zero
            else "elapsed time must be positive and finite"
        )
    return elapsed
