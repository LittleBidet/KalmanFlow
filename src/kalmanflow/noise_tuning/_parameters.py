"""The five tuned noise terms and their log-scaled search box."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..reservoir_config import ReservoirConfig
from ._diagnostics import _hourly_increment_sd_to_q
from ._types import _PARAMETER_NAMES, NoiseTuningSettings

Array = np.ndarray


def _base_parameters(config: ReservoirConfig) -> dict[str, float]:
    q = np.asarray(config.q, dtype=float)
    r = np.asarray(config.r, dtype=float)
    values = (q[0, 0], q[1, 1], q[2, 2], r[0, 0], r[1, 1])
    return {
        name: float(value) for name, value in zip(_PARAMETER_NAMES, values, strict=True)
    }


@dataclass(frozen=True)
class _ParameterSpace:
    """Map named noise terms to and from the unit cube in log space.

    Covariances are positive and scale-free, so the search works on their
    logarithms. Coordinate 0 is the lower bound and 1 the upper bound.
    """

    base: Mapping[str, float]
    lows: Array
    highs: Array

    @classmethod
    def from_settings(
        cls, base_config: ReservoirConfig, options: NoiseTuningSettings
    ) -> _ParameterSpace:
        base = _base_parameters(base_config)
        base_values = np.asarray(list(base.values()))
        if not np.isfinite(base_values).all() or np.any(base_values <= 0.0):
            raise ValueError(
                "all tuned diagonal q and r values must be positive and finite"
            )
        multipliers = np.asarray(
            [getattr(options, name) for name in options.multiplier_bounds_names()]
        )
        return cls(
            base=base,
            lows=base_values * multipliers[:, 0],
            highs=base_values * multipliers[:, 1],
        )

    def encode(self, parameters: Mapping[str, float]) -> Array:
        values = np.asarray([parameters[name] for name in _PARAMETER_NAMES])
        return (np.log(values) - np.log(self.lows)) / self._log_widths

    def decode(self, point: Array) -> dict[str, float]:
        values = np.exp(np.log(self.lows) + np.asarray(point) * self._log_widths)
        return {
            name: float(value)
            for name, value in zip(_PARAMETER_NAMES, values, strict=True)
        }

    def starting_points(
        self, inflow_increment_sd_seeds: Sequence[float]
    ) -> list[Array]:
        """Return one start per seed: base noise terms with the seed ``q_inflow``.

        Base terms outside the box are moved to the nearest bound; a seed
        outside the ``q_inflow`` bounds is an error rather than a silent clip.
        """

        q_values = sorted(
            {_hourly_increment_sd_to_q(seed) for seed in inflow_increment_sd_seeds}
        )
        if not q_values:
            raise ValueError("at least one inflow increment seed is required")
        low, high = self.lows[1], self.highs[1]
        outside = [q for q in q_values if not low <= q <= high]
        if outside:
            raise ValueError(
                "inflow increment seeds must lie inside q_inflow_multiplier_bounds "
                f"(q_inflow between {low:.6g} and {high:.6g})"
            )
        base_point = np.clip(self.encode(self.base), 0.0, 1.0)
        starts = []
        for q_inflow in q_values:
            point = base_point.copy()
            point[1] = self.encode({**self.base, "q_inflow": q_inflow})[1]
            starts.append(point)
        return starts

    def names_at_bounds(self, parameters: Mapping[str, float]) -> tuple[str, ...]:
        point = self.encode(parameters)
        return tuple(
            name
            for name, value in zip(_PARAMETER_NAMES, point, strict=True)
            if value <= 1e-6 or value >= 1.0 - 1e-6
        )

    @property
    def _log_widths(self) -> Array:
        return np.log(self.highs) - np.log(self.lows)
