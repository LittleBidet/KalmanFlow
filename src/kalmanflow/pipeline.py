"""Online filtering and fixed-lag smoothing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from math import isfinite
from typing import Protocol, TypeVar

from ._validation import bounded_integer, measurement_value
from .time_utils import to_utc, validate_timestamp_precision

FilterStep = TypeVar("FilterStep")
SmoothedState = TypeVar("SmoothedState")


@dataclass(frozen=True)
class InitializationObservation:
    """A storage/outflow observation used to initialize a stream."""

    timestamp: datetime
    storage: float
    discharge: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")
        to_utc(self.timestamp)
        validate_timestamp_precision(self.timestamp)
        object.__setattr__(
            self, "storage", measurement_value(self.storage, name="storage")
        )
        object.__setattr__(
            self,
            "discharge",
            measurement_value(self.discharge, name="discharge"),
        )


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
    initial_observation: InitializationObservation | None
    filter_anchor: FilterStep | None
    unfinished_observations: tuple[InitializationObservation, ...]


class PipelineBackend(Protocol[FilterStep]):
    """Model-specific operations required by the online pipeline.

    ``initialize`` constructs the first two forward filter steps. ``advance``
    constructs one subsequent predict/update step.
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
    def max_window_steps(self) -> int:
        """Maximum number of active forward-filter steps."""

    @property
    def pending_count(self) -> int:
        """Number of active, not finalized filter steps."""

    def add_step(self, step: FilterStep) -> tuple[SmoothedState, ...]:
        """Add a forward step and return newly finalized states."""

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
    call. ``filtered_state`` remains the latest one for compatibility with the
    original streaming API. Smoothed states are only included once the fixed
    lag has finalized them.
    """

    filtered_state: FilterStep | None
    smoothed_states: tuple[SmoothedState, ...]
    filtered_states: tuple[FilterStep, ...] = ()


class OnlineInflowPipeline:
    """Coordinate incoming observations, filtering, and delayed smoothing.

    The pipeline handles timestamp ordering, two-observation initialization,
    partial post-initialization updates, and a bounded smoothing window.
    """

    def __init__(
        self,
        backend: PipelineBackend[FilterStep],
        smoother: PipelineSmoother[FilterStep, SmoothedState],
        *,
        max_window_steps: int | None = None,
    ) -> None:
        """Create a pipeline with a model backend and a smoothing component."""

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
        self._initial_observation: InitializationObservation | None = None
        self._last_input_timestamp: datetime | None = None
        self._last_filter_step: FilterStep | None = None
        self._filter_anchor: FilterStep | None = None
        self._unfinished_observations: list[InitializationObservation] = []
        self._fallback_active_steps: list[FilterStep] = []

    @property
    def initialized(self) -> bool:
        """Whether the first two valid storage observations have been used."""

        return self._last_filter_step is not None

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

        return self._last_filter_step

    @property
    def last_input_timestamp(self) -> datetime | None:
        """The latest accepted timestamp, including skipped start-up data."""

        return self._last_input_timestamp

    def process(
        self,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> PipelineUpdate[FilterStep, SmoothedState]:
        """Process one observation and return available outputs."""

        self._validate_timestamp(timestamp)
        storage = measurement_value(storage, name="storage")
        discharge = measurement_value(discharge, name="discharge")
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
        finalized = self._add_step(
            step,
            observation=InitializationObservation(timestamp, storage, discharge),
        )
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

    def export_state(self) -> PipelineState[FilterStep]:
        """Return the minimal structured state needed to resume this pipeline.

        Active filter steps are intentionally not exported: their transient
        predicted values and transition matrices are rebuilt from raw active
        observations during :meth:`restore_state`.
        """

        self._require_replayable_smoother()
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

        if not self.initialized:
            if active_steps or self._filter_anchor is not None:
                raise RuntimeError("uninitialized pipeline has unexpected filter state")
            phase = (
                PipelineInitializationPhase.WAITING_FOR_FIRST
                if self._initial_observation is None
                else PipelineInitializationPhase.WAITING_FOR_SECOND
            )
            return PipelineState(
                phase=phase,
                last_input_timestamp=self._last_input_timestamp,
                initial_observation=self._initial_observation,
                filter_anchor=None,
                unfinished_observations=(),
            )

        if not active_steps:
            raise RuntimeError("initialized pipeline has no active smoothing state")
        phase = (
            PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION
            if self._filter_anchor is None
            else PipelineInitializationPhase.REPLAY_FROM_ANCHOR
        )
        return PipelineState(
            phase=phase,
            last_input_timestamp=self._last_input_timestamp,
            initial_observation=None,
            filter_anchor=self._filter_anchor,
            unfinished_observations=tuple(self._unfinished_observations),
        )

    def restore_state(self, state: PipelineState[FilterStep]) -> None:
        """Restore state and rebuild the active RTS window by replay.

        Replay outputs are deliberately discarded so restoration does not
        re-emit public results that were already returned before checkpointing.
        """

        if not isinstance(state, PipelineState):
            raise TypeError("state must be a PipelineState instance")
        phase = self._validate_state(state)
        self._clear_state()

        if phase is PipelineInitializationPhase.WAITING_FOR_FIRST:
            self._last_input_timestamp = state.last_input_timestamp
            return
        if phase is PipelineInitializationPhase.WAITING_FOR_SECOND:
            self._initial_observation = state.initial_observation
            self._last_input_timestamp = state.last_input_timestamp
            return

        if phase is PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION:
            for observation in state.unfinished_observations:
                self.process(
                    timestamp=observation.timestamp,
                    storage=observation.storage,
                    discharge=observation.discharge,
                )
        else:
            assert state.filter_anchor is not None
            self._filter_anchor = state.filter_anchor
            self._last_filter_step = state.filter_anchor
            self._last_input_timestamp = state.filter_anchor.timestamp
            for observation in state.unfinished_observations:
                self.process(
                    timestamp=observation.timestamp,
                    storage=observation.storage,
                    discharge=observation.discharge,
                )

        if self._last_input_timestamp is None or state.last_input_timestamp is None:
            raise RuntimeError("restored pipeline is missing its last input timestamp")
        if to_utc(self._last_input_timestamp) != to_utc(state.last_input_timestamp):
            raise RuntimeError(
                "restored pipeline timestamp does not match its snapshot"
            )

    def _clear_state(self) -> None:
        """Reset state while preserving the backend and smoother settings."""

        clear = getattr(self._smoother, "clear", None)
        if not callable(clear):
            raise RuntimeError("pipeline smoother does not support state restoration")
        clear()
        self._initial_observation = None
        self._last_input_timestamp = None
        self._last_filter_step = None
        self._filter_anchor = None
        self._unfinished_observations.clear()
        self._fallback_active_steps.clear()

    def _validate_state(
        self, state: PipelineState[FilterStep]
    ) -> PipelineInitializationPhase:
        """Reject structurally impossible snapshots before changing live state."""

        try:
            phase = PipelineInitializationPhase(state.phase)
        except (TypeError, ValueError) as error:
            raise ValueError("pipeline snapshot has an unsupported phase") from error

        if state.last_input_timestamp is not None:
            to_utc(state.last_input_timestamp)

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
        steps = self.backend.initialize(first, second)
        if len(steps) != 2:
            raise ValueError("backend.initialize must return exactly two filter steps")

        finalized: list[SmoothedState] = []
        finalized.extend(self._add_step(steps[0], observation=first))
        finalized.extend(self._add_step(steps[1], observation=second))
        self._last_filter_step = steps[1]
        self._initial_observation = None
        return PipelineUpdate(
            filtered_state=steps[1],
            smoothed_states=tuple(finalized),
            filtered_states=steps,
        )

    def _add_step(
        self,
        step: FilterStep,
        *,
        observation: InitializationObservation,
    ) -> tuple[SmoothedState, ...]:
        """Add one filter step and keep only raw data needed for replay."""

        active_steps: tuple[FilterStep, ...] = ()
        use_smoother_finalized_steps = self._has_finalized_steps_api()
        if not use_smoother_finalized_steps:
            active_steps = self._active_steps()
            if len(active_steps) != len(self._unfinished_observations):
                raise RuntimeError(
                    "pipeline replay state is inconsistent with the smoother"
                )
        finalized = self._smoother.add_step(step)
        finalized_count = len(finalized)
        if use_smoother_finalized_steps:
            finalized_steps = self._last_finalized_steps()
            if len(finalized_steps) != finalized_count:
                raise RuntimeError(
                    "smoother finalized state is inconsistent with its outputs"
                )
        else:
            all_steps = (*active_steps, step)
            if finalized_count > len(all_steps):
                raise RuntimeError("smoother finalized more steps than it received")
            finalized_steps = all_steps[:finalized_count]

        self._unfinished_observations.append(observation)
        if not self._has_active_steps_api():
            self._fallback_active_steps.append(step)
        if finalized_count:
            self._filter_anchor = finalized_steps[-1]
            del self._unfinished_observations[:finalized_count]
            if not self._has_active_steps_api():
                del self._fallback_active_steps[:finalized_count]
        return finalized

    def _active_steps(self) -> tuple[FilterStep, ...]:
        """Read active steps, with a compatibility fallback for basic smoothers."""

        active_steps = getattr(self._smoother, "active_steps", None)
        if callable(active_steps):
            return tuple(active_steps())
        return tuple(self._fallback_active_steps)

    def _has_active_steps_api(self) -> bool:
        """Whether the smoother can expose active steps itself."""

        return callable(getattr(self._smoother, "active_steps", None))

    def _has_finalized_steps_api(self) -> bool:
        """Whether the smoother reports the forward steps it just finalized."""

        return hasattr(self._smoother, "last_finalized_steps")

    def _last_finalized_steps(self) -> tuple[FilterStep, ...]:
        """Read the forward steps released by the latest smoother update."""

        return tuple(self._smoother.last_finalized_steps)

    def _require_replayable_smoother(self) -> None:
        """Require the reset hook used by checkpoint restoration."""

        if not callable(getattr(self._smoother, "clear", None)):
            raise RuntimeError(
                "pipeline smoother does not support checkpoint state "
                "export and restoration"
            )

    def _validate_timestamp(self, timestamp: datetime) -> None:
        """Check that a new timestamp is aware and later than the previous one."""

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        validate_timestamp_precision(timestamp)
        if self._last_input_timestamp is not None and to_utc(timestamp) <= to_utc(
            self._last_input_timestamp
        ):
            raise ValueError("observation timestamps must be strictly increasing")


__all__ = [
    "InitializationObservation",
    "ObservationLike",
    "OnlineInflowPipeline",
    "PipelineInitializationPhase",
    "PipelineBackend",
    "PipelineSmoother",
    "PipelineState",
    "PipelineUpdate",
]
