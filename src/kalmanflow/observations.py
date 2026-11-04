"""Online observation input contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ._validation import measurement_value
from .time_utils import aware_timestamp


@dataclass(frozen=True)
class Observation:
    """One online input sample for the reservoir inflow pipeline.

    Callers must supply pre-cleaned timestamps and real-valued measurements.
    Use NaN for missing storage or discharge; infinities are rejected. The
    pipeline treats each finite value as a noisy observation and applies the
    documented partial-observation rules at runtime.
    """

    timestamp: datetime
    storage: float
    discharge: float

    def __post_init__(self) -> None:
        aware_timestamp(self.timestamp)
        object.__setattr__(
            self, "storage", measurement_value(self.storage, name="storage")
        )
        object.__setattr__(
            self,
            "discharge",
            measurement_value(self.discharge, name="discharge"),
        )
