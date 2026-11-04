"""Immutable, validated per-reservoir configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import timedelta
from types import MappingProxyType
from typing import Any

import numpy as np

from ._validation import covariance_array, nonempty_string
from .models import Array
from .units import UnitSystem


def _freeze_metadata(value: Any) -> Any:
    """Make nested configuration metadata immutable without changing its values."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value).copy()
        array.setflags(write=False)
        return array
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_metadata(item) for item in value)
    return value


@dataclass(frozen=True)
class ReservoirConfig:
    """Validated configuration for a physical-rate reservoir model.

    ``q`` is the continuous-time diffusion covariance for the state
    ``[storage, inflow_rate, true_outflow_rate]``. ``p0`` uses the same
    physical state units. ``unit_system`` specifies storage and flow units.
    """

    reservoir_id: str
    reservoir_name: str
    q: Array
    r: Array
    p0: Array
    smoothing_lag: timedelta
    model_version: str
    configuration_version: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    unit_system: UnitSystem = field(default_factory=UnitSystem.us_customary)

    def __post_init__(self) -> None:
        for name in (
            "reservoir_id",
            "reservoir_name",
            "model_version",
            "configuration_version",
        ):
            nonempty_string(getattr(self, name), name=name)

        if not isinstance(self.smoothing_lag, timedelta):
            raise TypeError("smoothing_lag must be a timedelta")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        q = covariance_array(self.q, name="q", shape=(3, 3))
        r = covariance_array(
            self.r,
            name="r",
            shape=(2, 2),
            positive_diagonal=True,
        )
        p0 = covariance_array(self.p0, name="p0", shape=(3, 3))

        lag_seconds = self.smoothing_lag.total_seconds()
        if not np.isfinite(lag_seconds) or lag_seconds <= 0.0:
            raise ValueError("smoothing_lag must be positive and finite")
        if not isinstance(self.unit_system, UnitSystem):
            raise TypeError("unit_system must be a UnitSystem instance")

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "p0", p0)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))
        object.__setattr__(self, "unit_system", self.unit_system)

    def to_units(self, unit_system: UnitSystem) -> ReservoirConfig:
        """Return a copy with covariances converted to the target unit system.

        Converts all entries, including cross-covariances, in q, r and p0.
        Observations must be converted separately. Only built-in US/SI systems
        (or identical custom systems) support automatic conversion.
        """
        volume_factor, flow_factor = self.unit_system.conversion_factors_to(unit_system)
        state = np.array([volume_factor, flow_factor, flow_factor])
        observation = np.array([volume_factor, flow_factor])
        return replace(
            self,
            q=self.q * np.outer(state, state),
            r=self.r * np.outer(observation, observation),
            p0=self.p0 * np.outer(state, state),
            unit_system=unit_system,
        )
