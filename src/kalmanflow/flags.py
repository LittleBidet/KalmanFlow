"""Flags describing the provenance of public reservoir outputs."""

from __future__ import annotations

from enum import StrEnum


class OutputFlag(StrEnum):
    """Output provenance values used by the public reservoir APIs.

    ``NORMAL`` and ``PREDICTED`` describe whether the estimate was made with
    complete observations or with at least one missing observation.  A
    missing observation means ``PREDICTED`` for both single- and
    double-missing steps.

    ``SMOOTHED`` and ``NON_SMOOTHED`` describe whether the estimate came from
    the fixed-lag smoother or the causal filter.
    """

    NORMAL = "NORMAL"
    PREDICTED = "PREDICTED"

    SMOOTHED = "SMOOTHED"
    NON_SMOOTHED = "NON_SMOOTHED"


__all__ = ["OutputFlag"]
