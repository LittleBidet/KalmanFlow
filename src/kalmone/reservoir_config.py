"""Immutable, validated per-reservoir configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from ._validation import covariance_array
from .models import Array
from .units import UnitSystem


class InitializationStrategy(StrEnum):
    """Ways to choose the observations that start a reservoir stream."""

    FIRST_TWO_VALID_STORAGE = "first_two_valid_storage"


class InflowUnits(StrEnum):
    """Units available for the model's inflow and outflow rates."""

    CUBIC_FEET_PER_SECOND = "cfs"
    SYSTEM_FLOW_RATE = "system-flow-rate"


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
    physical state units, and ``inflow_units`` describes the flow-rate unit
    represented by both flow-rate state elements.
    """

    reservoir_id: str
    reservoir_name: str
    q: Array
    r: Array
    p0: Array
    smoothing_lag: timedelta
    initialization_strategy: InitializationStrategy
    inflow_units: InflowUnits
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
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must not be empty")

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
        inflow_units = InflowUnits(self.inflow_units)
        if (
            inflow_units is InflowUnits.CUBIC_FEET_PER_SECOND
            and self.unit_system.flow_label != "cfs"
        ):
            raise ValueError(
                "cfs inflow_units requires a UnitSystem with flow_label='cfs'"
            )

        object.__setattr__(self, "q", q)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "p0", p0)
        object.__setattr__(
            self,
            "initialization_strategy",
            InitializationStrategy(self.initialization_strategy),
        )
        object.__setattr__(self, "inflow_units", inflow_units)
        object.__setattr__(
            self, "metadata", _freeze_metadata(self.metadata)
        )
        object.__setattr__(self, "unit_system", self.unit_system)
