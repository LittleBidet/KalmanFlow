"""Configurable volume and flow-rate conversion for the water-balance model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

# US customary: 1 cfs for 1 second = 1/43560 acre-ft.
CFS_TO_ACRE_FEET_PER_SECOND = 1.0 / 43560.0


@dataclass(frozen=True)
class UnitSystem:
    """Maps inflow and outflow rates into model volume units.

    ``flow_to_volume_per_second`` converts a physical flow rate (for example
    cfs or m^3/s) into volume per second in the same volume unit as storage.
    """

    volume_label: str = "acre-ft"
    flow_label: str = "cfs"
    flow_to_volume_per_second: float = CFS_TO_ACRE_FEET_PER_SECOND

    def __post_init__(self) -> None:
        if not str(self.volume_label).strip():
            raise ValueError("volume_label must not be empty")
        if not str(self.flow_label).strip():
            raise ValueError("flow_label must not be empty")
        factor = float(self.flow_to_volume_per_second)
        if not isfinite(factor) or factor <= 0.0:
            raise ValueError("flow_to_volume_per_second must be positive and finite")
        object.__setattr__(self, "flow_to_volume_per_second", factor)

    @classmethod
    def us_customary(cls) -> UnitSystem:
        """Acre-feet storage with cubic-feet-per-second flow states."""

        return cls(
            volume_label="acre-ft",
            flow_label="cfs",
            flow_to_volume_per_second=CFS_TO_ACRE_FEET_PER_SECOND,
        )

    @classmethod
    def si(cls) -> UnitSystem:
        """Cubic-metre storage with cubic-metres-per-second flow states."""

        return cls(
            volume_label="m^3",
            flow_label="m^3/s",
            flow_to_volume_per_second=1.0,
        )

    def flow_to_volume(self, flow_rate: float, elapsed_seconds: float) -> float:
        """Convert a flow rate over a time interval into a volume."""

        rate = float(flow_rate)
        elapsed = float(elapsed_seconds)
        if not isfinite(rate):
            raise ValueError("flow_rate must be finite")
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("elapsed_seconds must be positive and finite")
        return rate * elapsed * self.flow_to_volume_per_second

    def validate_flow_rate(self, flow_rate: float) -> float:
        """Return a finite flow rate or raise an error."""

        rate = float(flow_rate)
        if not isfinite(rate):
            raise ValueError("flow_rate must be finite")
        return rate

    def volume_to_flow_rate(self, volume: float, elapsed_seconds: float) -> float:
        """Convert a volume accumulated over an interval into a flow rate."""

        volume_value = float(volume)
        elapsed = float(elapsed_seconds)
        if not isfinite(volume_value):
            raise ValueError("volume must be finite")
        if not isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("elapsed_seconds must be positive and finite")
        return volume_value / elapsed / self.flow_to_volume_per_second

    def volume_rate_to_flow(
        self, volume_per_interval: float, interval_seconds: float
    ) -> float:
        """Backward-compatible alias for :meth:`volume_to_flow_rate`."""

        volume = float(volume_per_interval)
        interval = float(interval_seconds)
        if not isfinite(volume):
            raise ValueError("volume_per_interval must be finite")
        if not isfinite(interval) or interval <= 0.0:
            raise ValueError("interval_seconds must be positive and finite")
        return self.volume_to_flow_rate(volume, interval)
