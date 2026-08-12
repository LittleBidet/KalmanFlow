"""Online observation input contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Observation:
    """One online input sample for the reservoir inflow pipeline.

    Callers must supply pre-cleaned timestamps and measured values. Use NaN for
    missing storage or discharge; the pipeline treats each finite value as a
    noisy observation and applies the documented partial-observation rules at
    runtime.
    """

    timestamp: datetime
    storage: float
    discharge: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
