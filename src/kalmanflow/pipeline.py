"""Online filtering and fixed-lag smoothing pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from enum import IntEnum
from math import isfinite
from typing import Protocol, TypeVar, cast

from ._validation import bounded_integer, measurement_value
from .observations import Observation
from .time_utils import aware_timestamp, to_utc

FilterStep = TypeVar("FilterStep")
SmoothedState = TypeVar("SmoothedState")

_FAILED_PIPELINE_MESSAGE = (
    "pipeline failed after its smoothing window changed; "
    "restore_state must replace it before further use"
)


@dataclass(frozen=True, slots=True)
class _WaitingForFirst:
    """No usable first observation has been accepted yet."""

    last_input_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class _WaitingForSecond:
    """The first observation is held until a second finite storage arrives."""

    first: Observation
    last_input_timestamp: datetime


@dataclass(frozen=True, slots=True)
class _Running[FilterStep]:
    """Initialized: every observation advances the forward filter.

    ``anchor`` is the most recently finalized filter step, or ``None`` until
    the smoother has released one.
    """

    last_step: FilterStep
    anchor: FilterStep | None
    last_input_timestamp: datetime


@dataclass(frozen=True, slots=True)
class _Failed:
    """A call failed after the smoother accepted a step; state is unreliable."""

    last_input_timestamp: None = None


type _Phase = _WaitingForFirst | _WaitingForSecond | _Running | _Failed


class PipelineInitializationPhase(IntEnum):
    """The restart-relevant phase of an online inflow pipeline."""

    WAITING_FOR_FIRST = 0
    WAITING_FOR_SECOND = 1
    REPLAY_FROM_INITIALIZATION = 2
    REPLAY_FROM_ANCHOR = 3


@dataclass(frozen=True)
class PipelineState[FilterStep]:
    """Structured, backend-neutral state needed to resume a pipeline.

    ``filter_anchor`` is the last finalized filter step, when one exists.
    ``unfinished_observations`` are the active smoothing-window observations
    that must be replayed after that anchor. The reservoir-specific layer is
    responsible for compactly encoding this structure.
    """

    phase: PipelineInitializationPhase
    last_input_timestamp: datetime | None
    initial_observation: Observation | None
    filter_anchor: FilterStep | None
    unfinished_observations: tuple[Observation, ...]


class PipelineBackend(Protocol[FilterStep]):
    """Model-specific operations required by the online pipeline.

    ``initialize`` constructs the first two forward filter steps. ``advance``
    constructs one subsequent predict/update step.
    """

    def initialize(
        self,
        first: Observation,
        second: Observation,
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
    def max_window_steps(self) -> int:
        """Maximum number of active forward-filter steps."""

    @property
    def pending_count(self) -> int:
        """Number of active, not finalized filter steps."""

    def add_step(self, step: FilterStep) -> tuple[SmoothedState, ...]:
        """Add a forward step and return newly finalized states.

        An implementation that raises must leave its window unchanged.
        """

    def provisional(self) -> tuple[SmoothedState, ...]:
        """Return smoothed states currently in the active lag window."""

    def active_steps(self) -> tuple[FilterStep, ...]:
        """Return active forward-filter steps in timestamp order."""

    def clear(self) -> None:
        """Discard every active forward-filter step."""

    @property
    def last_finalized_steps(self) -> tuple[FilterStep, ...]:
        """Forward-filter steps released by the most recent ``add_step``."""


class ObservationLike(Protocol):
    """Structural input for the public reservoir stream."""

    timestamp: datetime
    storage: float
    discharge: float


@dataclass(frozen=True)
class PipelineUpdate[FilterStep, SmoothedState]:
    """Outputs produced by one pipeline call.

    ``filtered_states`` contains every new forward-filter state created by the
    call. Smoothed states are only included once the fixed
    lag has finalized them.
    """

    filtered_states: tuple[FilterStep, ...]
    smoothed_states: tuple[SmoothedState, ...]


class OnlineInflowPipeline:
    """Coordinate incoming observations, filtering, and delayed smoothing.

    The pipeline handles timestamp ordering, two-observation initialization,
    partial post-initialization updates, and a bounded smoothing window.

    A call that raises before the smoother accepts a step leaves the pipeline
    unchanged. A call that raises afterwards cannot be undone, so the pipeline
    becomes :attr:`failed` and rejects further use until
    :meth:`restore_state` replaces its state.
    """

    def __init__(
        self,
        backend: PipelineBackend[FilterStep],
        smoother: PipelineSmoother[FilterStep, SmoothedState],
        *,
        max_window_steps: int | None = None,
    ) -> None:
        """Create a pipeline with a model backend and a smoothing component."""

        methods = ("add_step", "provisional", "active_steps", "clear")
        attributes = ("max_window_steps", "pending_count", "last_finalized_steps")
        missing = [
            name for name in methods if not callable(getattr(smoother, name, None))
        ]
        missing.extend(name for name in attributes if not hasattr(smoother, name))
        if missing:
            raise TypeError(
                "smoother does not implement PipelineSmoother: missing "
                + ", ".join(missing)
            )
        self.backend = backend
        self._smoother = smoother
        smoother_limit = bounded_integer(
            smoother.max_window_steps,
            name="smoother.max_window_steps",
            minimum=2,
        )
        if max_window_steps is not None:
            requested_limit = bounded_integer(
                max_window_steps,
                name="max_window_steps",
                minimum=2,
            )
            if requested_limit != smoother_limit:
                raise ValueError(
                    "max_window_steps must match smoother.max_window_steps"
                )
        self._phase: _Phase = _WaitingForFirst(last_input_timestamp=None)
        # Raw observations for the smoother's active steps, kept in lockstep.
        self._unfinished_observations: list[Observation] = []

    @property
    def initialized(self) -> bool:
        """Whether the first two valid storage observations have been used."""

        return isinstance(self._phase, _Running)

    @property
    def failed(self) -> bool:
        """Whether a call failed after the smoothing window had changed."""

        return isinstance(self._phase, _Failed)

    @property
    def pending_count(self) -> int:
        """Number of active states waiting for the smoothing lag."""

        return self._smoother.pending_count

    @property
    def max_window_steps(self) -> int:
        """Maximum active-window size enforced by the smoother."""

        return int(self._smoother.max_window_steps)

    @property
    def latest_filter_step(self) -> FilterStep | None:
        """The latest forward-filter state, if initialization has completed."""

        phase = self._phase
        return phase.last_step if isinstance(phase, _Running) else None

    @property
    def last_input_timestamp(self) -> datetime | None:
        """The latest accepted timestamp, including skipped start-up data."""

        return self._phase.last_input_timestamp

    def process(
        self,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> PipelineUpdate[FilterStep, SmoothedState]:
        """Process one observation and return available outputs."""

        phase = self._phase
        if isinstance(phase, _Failed):
            raise RuntimeError(_FAILED_PIPELINE_MESSAGE)
        self._validate_timestamp(timestamp)
        storage = measurement_value(storage, name="storage")
        discharge = measurement_value(discharge, name="discharge")
        if not isinstance(phase, _Running):
            return self._process_initialization(
                phase,
                timestamp=timestamp,
                storage=storage,
                discharge=discharge,
            )

        step = self.backend.advance(
            phase.last_step,
            timestamp=timestamp,
            storage=storage,
            discharge=discharge,
        )
        finalized, anchor = self._add_steps(
            ((step, Observation(timestamp, storage, discharge)),)
        )
        self._phase = _Running(
            last_step=step,
            anchor=phase.anchor if anchor is None else anchor,
            last_input_timestamp=timestamp,
        )
        return PipelineUpdate(
            smoothed_states=finalized,
            filtered_states=(step,),
        )

    def provisional_states(self) -> tuple[SmoothedState, ...]:
        """Return active-window estimates without finalizing them."""

        return self._smoother.provisional()

    def export_state(self) -> PipelineState[FilterStep]:
        """Return the minimal structured state needed to resume this pipeline.

        Active filter steps are intentionally not exported: their transient
        predicted values and transition matrices are rebuilt from raw active
        observations during :meth:`restore_state`.
        """

        phase = self._phase
        if isinstance(phase, _Failed):
            raise RuntimeError(_FAILED_PIPELINE_MESSAGE)
        active_steps = self._active_steps()
        if len(active_steps) != len(self._unfinished_observations):
            raise RuntimeError(
                "pipeline replay state is inconsistent with the smoother"
            )
        for step, observation in zip(
            active_steps, self._unfinished_observations, strict=True
        ):
            if to_utc(step.timestamp) != to_utc(observation.timestamp):
                raise RuntimeError(
                    "pipeline replay observations are inconsistent with the smoother"
                )

        if not isinstance(phase, _Running):
            if active_steps:
                raise RuntimeError("uninitialized pipeline has unexpected filter state")
            waiting_for_first = isinstance(phase, _WaitingForFirst)
            return PipelineState(
                phase=(
                    PipelineInitializationPhase.WAITING_FOR_FIRST
                    if waiting_for_first
                    else PipelineInitializationPhase.WAITING_FOR_SECOND
                ),
                last_input_timestamp=phase.last_input_timestamp,
                initial_observation=None if waiting_for_first else phase.first,
                filter_anchor=None,
                unfinished_observations=(),
            )

        if not active_steps:
            raise RuntimeError("initialized pipeline has no active smoothing state")
        return PipelineState(
            phase=(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION
                if phase.anchor is None
                else PipelineInitializationPhase.REPLAY_FROM_ANCHOR
            ),
            last_input_timestamp=phase.last_input_timestamp,
            initial_observation=None,
            filter_anchor=phase.anchor,
            unfinished_observations=tuple(self._unfinished_observations),
        )

    def restore_state(self, state: PipelineState[FilterStep]) -> None:
        """Restore state and rebuild the active RTS window by replay.

        Replay outputs are deliberately discarded so restoration does not
        re-emit public results that were already returned before checkpointing.
        A snapshot that fails validation leaves the pipeline unchanged; a
        restore that fails after it has begun replacing state leaves the
        pipeline :attr:`failed`. Restoring is the only way out of that state.
        """

        if not isinstance(state, PipelineState):
            raise TypeError("state must be a PipelineState instance")
        phase = self._validate_state(state)
        try:
            self._restore_validated_state(state, phase)
        except BaseException:
            self._phase = _Failed()
            raise

    def _restore_validated_state(
        self,
        state: PipelineState[FilterStep],
        phase: PipelineInitializationPhase,
    ) -> None:
        """Replace live state with an already validated snapshot."""

        self._clear_state()
        # _validate_state guarantees the fields each phase reads below.
        if phase is PipelineInitializationPhase.WAITING_FOR_FIRST:
            self._phase = _WaitingForFirst(state.last_input_timestamp)
            return
        if phase is PipelineInitializationPhase.WAITING_FOR_SECOND:
            self._phase = _WaitingForSecond(
                first=cast(Observation, state.initial_observation),
                last_input_timestamp=cast(datetime, state.last_input_timestamp),
            )
            return

        if phase is PipelineInitializationPhase.REPLAY_FROM_ANCHOR:
            anchor = cast(FilterStep, state.filter_anchor)
            self._phase = _Running(
                last_step=anchor,
                anchor=anchor,
                last_input_timestamp=anchor.timestamp,
            )
        for observation in state.unfinished_observations:
            self.process(
                timestamp=observation.timestamp,
                storage=observation.storage,
                discharge=observation.discharge,
            )

        restored_timestamp = self.last_input_timestamp
        if restored_timestamp is None or state.last_input_timestamp is None:
            raise RuntimeError("restored pipeline is missing its last input timestamp")
        if to_utc(restored_timestamp) != to_utc(state.last_input_timestamp):
            raise RuntimeError(
                "restored pipeline timestamp does not match its snapshot"
            )

    def _clear_state(self) -> None:
        """Reset state while preserving the backend and smoother settings."""

        self._smoother.clear()
        self._unfinished_observations.clear()
        self._phase = _WaitingForFirst(last_input_timestamp=None)

    def _validate_state(
        self, state: PipelineState[FilterStep]
    ) -> PipelineInitializationPhase:
        """Reject structurally impossible snapshots before changing live state."""

        try:
            phase = PipelineInitializationPhase(state.phase)
        except (TypeError, ValueError) as error:
            raise ValueError("pipeline snapshot has an unsupported phase") from error

        if state.last_input_timestamp is not None:
            aware_timestamp(state.last_input_timestamp)
        observations = list(state.unfinished_observations)
        if state.initial_observation is not None:
            observations.append(state.initial_observation)
        if not all(isinstance(item, Observation) for item in observations):
            raise TypeError(
                "pipeline snapshot observations must be Observation instances"
            )

        if phase is PipelineInitializationPhase.WAITING_FOR_FIRST:
            if (
                state.initial_observation is not None
                or state.filter_anchor is not None
                or state.unfinished_observations
            ):
                raise ValueError("waiting-for-first snapshot contains filter state")
            return phase

        if phase is PipelineInitializationPhase.WAITING_FOR_SECOND:
            first = state.initial_observation
            if (
                first is None
                or state.filter_anchor is not None
                or state.unfinished_observations
            ):
                raise ValueError("waiting-for-second snapshot is incomplete")
            if state.last_input_timestamp is None:
                raise ValueError("waiting-for-second snapshot has no last timestamp")
            if not (isfinite(first.storage) and isfinite(first.discharge)):
                raise ValueError(
                    "initial observation needs finite storage and discharge"
                )
            if to_utc(first.timestamp) > to_utc(state.last_input_timestamp):
                raise ValueError(
                    "initial observation is after the last input timestamp"
                )
            return phase

        if state.initial_observation is not None:
            raise ValueError("initialized snapshot contains an initial observation")
        if state.last_input_timestamp is None or not state.unfinished_observations:
            raise ValueError("initialized snapshot is missing active observations")
        if phase is PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION:
            if state.filter_anchor is not None:
                raise ValueError("initialization replay snapshot has a filter anchor")
            if len(state.unfinished_observations) < 2:
                raise ValueError("initialization replay needs two observations")
            # Replay must rebuild the same two initialization steps; a row it
            # would skip as start-up data would shift the whole window.
            first, second = state.unfinished_observations[:2]
            if not (isfinite(first.storage) and isfinite(first.discharge)):
                raise ValueError(
                    "initialization replay must start with finite storage and "
                    "discharge"
                )
            if not isfinite(second.storage):
                raise ValueError(
                    "initialization replay needs finite storage in its second "
                    "observation"
                )
            previous_timestamp: datetime | None = None
        else:
            anchor = state.filter_anchor
            if anchor is None:
                raise ValueError("anchor replay snapshot is missing a filter anchor")
            previous_timestamp = anchor.timestamp

        for observation in state.unfinished_observations:
            if previous_timestamp is not None and to_utc(
                observation.timestamp
            ) <= to_utc(previous_timestamp):
                raise ValueError("replay observations must be strictly increasing")
            previous_timestamp = observation.timestamp
        if to_utc(previous_timestamp) != to_utc(state.last_input_timestamp):
            raise ValueError("last input timestamp does not match replay observations")
        return phase

    def _process_initialization(
        self,
        phase: _WaitingForFirst | _WaitingForSecond,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> PipelineUpdate[FilterStep, SmoothedState]:
        """Collect the starting observations or build the first filter steps."""

        if not isfinite(storage):
            self._phase = replace(phase, last_input_timestamp=timestamp)
            return PipelineUpdate(
                smoothed_states=(),
                filtered_states=(),
            )

        if isinstance(phase, _WaitingForFirst):
            if not isfinite(discharge):
                raise ValueError(
                    "the first valid storage observation needs a finite discharge value"
                )
            self._phase = _WaitingForSecond(
                first=Observation(
                    timestamp=timestamp,
                    storage=storage,
                    discharge=discharge,
                ),
                last_input_timestamp=timestamp,
            )
            return PipelineUpdate(
                smoothed_states=(),
                filtered_states=(),
            )

        first = phase.first
        second = Observation(
            timestamp=timestamp,
            storage=storage,
            discharge=discharge,
        )
        steps = self.backend.initialize(first, second)
        if len(steps) != 2:
            raise ValueError("backend.initialize must return exactly two filter steps")

        finalized, anchor = self._add_steps(((steps[0], first), (steps[1], second)))
        self._phase = _Running(
            last_step=steps[1],
            anchor=anchor,
            last_input_timestamp=timestamp,
        )
        return PipelineUpdate(
            smoothed_states=finalized,
            filtered_states=steps,
        )

    def _add_steps(
        self,
        steps: Iterable[tuple[FilterStep, Observation]],
    ) -> tuple[tuple[SmoothedState, ...], FilterStep | None]:
        """Add filter steps, keeping only the raw data needed for replay.

        Returns the newly finalized states and the last step they finalized,
        if any. Once the smoother has accepted a step, a later failure cannot
        be undone, so the pipeline moves to its failed phase.
        """

        finalized: list[SmoothedState] = []
        anchor: FilterStep | None = None
        accepted = False
        try:
            for step, observation in steps:
                released = self._smoother.add_step(step)
                accepted = True
                released_steps = tuple(self._smoother.last_finalized_steps)
                if len(released_steps) != len(released):
                    raise RuntimeError(
                        "smoother finalized state is inconsistent with its outputs"
                    )
                self._unfinished_observations.append(observation)
                if released_steps:
                    anchor = released_steps[-1]
                    del self._unfinished_observations[: len(released_steps)]
                finalized.extend(released)
        except BaseException:
            if accepted:
                self._phase = _Failed()
            raise
        return tuple(finalized), anchor

    def _active_steps(self) -> tuple[FilterStep, ...]:
        """Read active forward-filter steps in timestamp order."""

        return tuple(self._smoother.active_steps())

    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Check that a new timestamp is aware and later than the previous one."""

        aware_timestamp(timestamp)
        previous = self.last_input_timestamp
        if previous is not None and to_utc(timestamp) <= to_utc(previous):
            raise ValueError("observation timestamps must be strictly increasing")


__all__ = [
    "ObservationLike",
    "OnlineInflowPipeline",
    "PipelineInitializationPhase",
    "PipelineBackend",
    "PipelineSmoother",
    "PipelineState",
    "PipelineUpdate",
]
