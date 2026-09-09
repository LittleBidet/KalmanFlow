"""Joint storage, inflow, and outflow state-space model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ._validation import covariance_array, readonly_array
from .units import UnitSystem

Array = np.ndarray


RESERVOIR_OBSERVATION_MATRIX = readonly_array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


@runtime_checkable
class StateSpaceModel(Protocol):
    """Operations the online pipeline needs from a reservoir model."""

    @property
    def observation_matrix(self) -> Array:
        """Describe which state values are measured by the observations."""

        ...

    def transition_matrix(self, elapsed_seconds: float) -> Array:
        """Return how the state changes over one time interval."""

        ...

    def process_covariance(self, elapsed_seconds: float) -> Array:
        """Return the uncertainty added over one time interval."""

        ...

    def initial_outflow(self, first_discharge_rate: float) -> float:
        """Return the first measured outflow used to start the model."""

        ...

@dataclass(frozen=True)
class ReservoirStateSpaceModel:
    """Estimate storage, inflow, and actual outflow together.

    The three tracked values are ``[storage, inflow rate, outflow rate]``.
    Storage changes according to the difference between inflow and outflow.
    The model uses the actual time between observations and the configured
    volume and flow-rate units.
    """

    q_continuous: Array
    unit_system: UnitSystem = UnitSystem.us_customary()

    def __post_init__(self) -> None:
        q = covariance_array(
            self.q_continuous,
            name="q_continuous",
            shape=(3, 3),
        )

        units = self.unit_system
        if not isinstance(units, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem instance")

        object.__setattr__(self, "q_continuous", q)
        object.__setattr__(self, "unit_system", units)

    @property
    def observation_matrix(self) -> Array:
        """Describe the storage and outflow measurements used by the model."""

        return RESERVOIR_OBSERVATION_MATRIX

    def _elapsed_seconds(self, elapsed_seconds: float) -> float:
        """Validate and normalize an elapsed-time value."""

        elapsed = float(elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed <= 0.0:
            raise ValueError("elapsed_seconds must be positive and finite")
        return elapsed

    def transition_matrix(self, elapsed_seconds: float) -> Array:
        """Build the state-change matrix for one elapsed-time interval."""

        elapsed = self._elapsed_seconds(elapsed_seconds)
        volume_per_rate_second = self.unit_system.flow_to_volume_per_second
        return np.array(
            [
                [
                    1.0,
                    volume_per_rate_second * elapsed,
                    -volume_per_rate_second * elapsed,
                ],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

    def process_covariance(self, elapsed_seconds: float) -> Array:
        """Return the uncertainty added by the model during an interval."""

        elapsed = self._elapsed_seconds(elapsed_seconds)
        volume_per_rate_second = self.unit_system.flow_to_volume_per_second
        coupling = np.array(
            [
                [0.0, volume_per_rate_second, -volume_per_rate_second],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        first_order = coupling @ self.q_continuous + self.q_continuous @ coupling.T
        second_order = coupling @ self.q_continuous @ coupling.T
        return (
            self.q_continuous * elapsed
            + first_order * elapsed**2 / 2.0
            + second_order * elapsed**3 / 3.0
        )

    def discharge_volume(self, discharge_rate: float, elapsed_seconds: float) -> float:
        """Convert a discharge flow rate over an interval into model volume units."""

        return self.unit_system.flow_to_volume(discharge_rate, elapsed_seconds)

    def initial_inflow(
        self,
        first_storage: float,
        second_storage: float,
        first_discharge_rate: float,
        elapsed_seconds: float,
    ) -> float:
        """Calculate interval-average inflow from storage change and outflow.

        This diagnostic helper uses both interval endpoints. The online and
        batch filters do not use it for their first timestamped causal value.
        """

        water_balance_volume = (
            float(second_storage)
            - float(first_storage)
            + self.discharge_volume(first_discharge_rate, elapsed_seconds)
        )
        return self.unit_system.volume_to_flow_rate(
            water_balance_volume, elapsed_seconds
        )

    def initial_outflow(self, first_discharge_rate: float) -> float:
        """Use the first finite outflow reading to seed the true outflow state."""

        return self.unit_system.validate_flow_rate(first_discharge_rate)
