"""Online filtering and fixed-lag smoothing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
from typing import Protocol, TypeVar

from ._validation import bounded_integer
from .time_utils import to_utc

FilterStep = TypeVar("FilterStep")
SmoothedState = TypeVar("SmoothedState")


@dataclass(frozen=True)
class InitializationObservation:
    """A storage/outflow observation used to initialize a stream."""

    timestamp: datetime
    storage: float
    discharge: float


class PipelineBackend(Protocol[FilterStep]):
    """Model-specific operations required by the online pipeline.

    ``initialize`` constructs the first two forward filter steps. ``advance`` constructs
    one subsequent predict/update step.
    """

    def initialize(
        self,
        first: InitializationObservation,
        second: InitializationObservation,
    ) -> tuple[FilterStep, FilterStep]:
        """Return forward steps for the two observations that start a stream."""

    def advance(
        self,
        previous: FilterStep,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> FilterStep:
        """Return one forward step after ``previous``."""


class PipelineSmoother(Protocol[FilterStep, SmoothedState]):
    """Smoother that releases estimates after a chosen delay."""

    @property
    def pending_count(self) -> int:
        """Number of active, not finalized filter steps."""

    def add_step(self, step: FilterStep) -> tuple[SmoothedState, ...]:
        """Add a forward step and return newly finalized states."""

    def provisional(self) -> tuple[SmoothedState, ...]:
        """Return smoothed states currently in the active lag window."""


class ObservationLike(Protocol):
    """Structural input for ``process_observation``."""

    timestamp: datetime
    storage: float
    discharge: float


@dataclass(frozen=True)
class PipelineUpdate[FilterStep, SmoothedState]:
    """Outputs produced by one pipeline call.

    ``filtered_states`` contains every new forward-filter state created by the
    call. ``filtered_state`` remains the latest one for compatibility with the
    original streaming API. Smoothed states are only included once the fixed
    lag has finalized them.
    """

    filtered_state: FilterStep | None
    smoothed_states: tuple[SmoothedState, ...]
    filtered_states: tuple[FilterStep, ...] = ()


class OnlineInflowPipeline:
    """Coordinate incoming observations, filtering, and delayed smoothing.

    The pipeline handles the stream contract and lifecycle:

    * timestamps must be timezone-aware and strictly increasing;
    * initialization waits for two finite storage observations;
    * the first initialization observation must have finite discharge;
    * missing storage or discharge is passed as a partial observation after
      initialization; and
    * the smoother's active window is bounded by ``max_window_steps``.

    The backend contains the model equations and the smoother contains the fixed-lag RTS
    calculations. Each process call returns the latest forward-filter state,
    every new forward-filter state created by that call, and any states that
    have been smoothed.

    Input streams are assumed to be pre-cleaned by the caller. Missing storage or
    discharge is represented as NaN and handled at runtime: non-finite storage
    is skipped during initialization, and after initialization each finite
    observation component is used independently while missing components are
    ignored by the Kalman update.
    """

    def __init__(
        self,
        backend: PipelineBackend[FilterStep],
        smoother: PipelineSmoother[FilterStep, SmoothedState],
        *,
        max_window_steps: int = 100_000,
    ) -> None:
        """Create a pipeline with a model backend and a smoothing component."""

        self.backend = backend
        self._smoother = smoother
        self.max_window_steps = bounded_integer(
            max_window_steps,
            name="max_window_steps",
            minimum=2,
        )
        self._initial_observation: InitializationObservation | None = None
        self._last_input_timestamp: datetime | None = None
        self._last_filter_step: FilterStep | None = None

    @property
    def initialized(self) -> bool:
        """Whether the first two valid storage observations have been used."""

        return self._last_filter_step is not None

    @property
    def pending_count(self) -> int:
        """Number of active states waiting for the smoothing lag."""

        return self._smoother.pending_count

    @property
    def latest_filter_step(self) -> FilterStep | None:
        """The latest forward-filter state, if initialization has completed."""

        return self._last_filter_step

    def process_observation(
        self, observation: ObservationLike
    ) -> PipelineUpdate[FilterStep, SmoothedState]:
        """Process an object exposing ``timestamp``, ``storage``, and ``discharge``."""

        return self.process(
            timestamp=observation.timestamp,
            storage=observation.storage,
            discharge=observation.discharge,
        )

    def process(
        self,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> PipelineUpdate[FilterStep, SmoothedState]:
        """Process one observation and return available outputs."""

        self._validate_timestamp(timestamp)

        if not self.initialized:
            update = self._process_initialization(
                timestamp=timestamp,
                storage=storage,
                discharge=discharge,
            )
            self._last_input_timestamp = timestamp
            return update

        assert self._last_filter_step is not None
        step = self.backend.advance(
            self._last_filter_step,
            timestamp=timestamp,
            storage=storage,
            discharge=discharge,
        )
        finalized = self._smoother.add_step(step)
        self._last_filter_step = step
        self._last_input_timestamp = timestamp
        return PipelineUpdate(
            filtered_state=step,
            smoothed_states=finalized,
            filtered_states=(step,),
        )

    def provisional_states(self) -> tuple[SmoothedState, ...]:
        """Return active-window estimates without finalizing them."""

        return self._smoother.provisional()

    def _process_initialization(
        self,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> PipelineUpdate[FilterStep, SmoothedState]:
        """Collect the starting observations or build the first filter steps."""

        if not isfinite(storage):
            return PipelineUpdate(
                filtered_state=None,
                smoothed_states=(),
                filtered_states=(),
            )

        if self._initial_observation is None:
            if not isfinite(discharge):
                raise ValueError(
                    "the first valid storage observation needs a finite discharge value"
                )
            self._initial_observation = InitializationObservation(
                timestamp=timestamp,
                storage=storage,
                discharge=discharge,
            )
            return PipelineUpdate(
                filtered_state=None,
                smoothed_states=(),
                filtered_states=(),
            )

        first = self._initial_observation
        second = InitializationObservation(
            timestamp=timestamp,
            storage=storage,
            discharge=discharge,
        )
        self._check_window_capacity(required_steps=2)
        steps = self.backend.initialize(first, second)
        if len(steps) != 2:
            raise ValueError("backend.initialize must return exactly two filter steps")

        finalized: list[SmoothedState] = []
        finalized.extend(self._smoother.add_step(steps[0]))
        finalized.extend(self._smoother.add_step(steps[1]))
        self._last_filter_step = steps[1]
        self._initial_observation = None
        return PipelineUpdate(
            filtered_state=steps[1],
            smoothed_states=tuple(finalized),
            filtered_states=steps,
        )

    def _check_window_capacity(self, *, required_steps: int = 1) -> None:
        """Raise an error if adding steps would exceed the active window."""

        if self.pending_count + required_steps > self.max_window_steps:
            raise OverflowError(
                "Smoothing window reached max_window_steps before a state finalized"
            )

    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Check that a new timestamp is aware and later than the previous one."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        if self._last_input_timestamp is not None and to_utc(timestamp) <= to_utc(
            self._last_input_timestamp
        ):
            raise ValueError("observation timestamps must be strictly increasing")


__all__ = [
    "InitializationObservation",
    "ObservationLike",
    "OnlineInflowPipeline",
    "PipelineBackend",
    "PipelineSmoother",
    "PipelineUpdate",
]
