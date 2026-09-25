"""Snapshot compatibility and validation coverage for pipeline and RTS code."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kalmanflow import OnlineFixedLagRTS, OnlineInflowPipeline
from kalmanflow import rts as rts_module
from kalmanflow.kalman import FilterStep
from kalmanflow.pipeline import (
    InitializationObservation,
    PipelineInitializationPhase,
    PipelineState,
)

START = datetime(2024, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class _Step:
    timestamp: datetime


class _Backend:
    def initialize(self, first, second):
        return _Step(first.timestamp), _Step(second.timestamp)

    def advance(self, previous, *, timestamp, storage, discharge):
        del previous, storage, discharge
        return _Step(timestamp)


class _ModernSmoother:
    max_window_steps = 4

    def __init__(self, *, releases: tuple[object, ...] = ()) -> None:
        self._active: list[_Step] = []
        self._releases = releases
        self.last_finalized_steps: tuple[_Step, ...] = ()

    @property
    def pending_count(self) -> int:
        return len(self._active)

    def add_step(self, step: _Step) -> tuple[object, ...]:
        self._active.append(step)
        self.last_finalized_steps = self._active[: len(self._releases)]
        if self._releases:
            self._active = self._active[len(self._releases) :]
        return self._releases

    def provisional(self) -> tuple[object, ...]:
        return tuple(self._active)

    def active_steps(self) -> tuple[_Step, ...]:
        return tuple(self._active)

    def clear(self) -> None:
        self._active.clear()
        self.last_finalized_steps = ()


class _LegacySmoother:
    """Smoother implementing the original active-step compatibility API."""

    max_window_steps = 4

    def __init__(self, releases: tuple[object, ...] = ()) -> None:
        self._active: list[_Step] = []
        self._releases = releases

    @property
    def pending_count(self) -> int:
        return len(self._active)

    def add_step(self, step: _Step) -> tuple[object, ...]:
        self._active.append(step)
        releases = self._releases
        if releases:
            self._active = self._active[len(releases) :]
        return releases

    def provisional(self) -> tuple[object, ...]:
        return tuple(self._active)

    def clear(self) -> None:
        self._active.clear()


def _pipeline(smoother: object | None = None) -> OnlineInflowPipeline:
    return OnlineInflowPipeline(
        _Backend(),
        smoother if smoother is not None else _ModernSmoother(),
    )


def _observation(minutes: int, *, storage: float = 100.0) -> InitializationObservation:
    return InitializationObservation(
        START + timedelta(minutes=minutes), storage=storage, discharge=4.0
    )


def _state(
    phase: PipelineInitializationPhase,
    *,
    last: datetime | None = START,
    initial: InitializationObservation | None = None,
    anchor: object | None = None,
    observations: tuple[InitializationObservation, ...] = (),
) -> PipelineState:
    return PipelineState(
        phase=phase,
        last_input_timestamp=last,
        initial_observation=initial,
        filter_anchor=anchor,
        unfinished_observations=observations,
    )


def _valid_filter_step(timestamp: datetime = START) -> FilterStep:
    return FilterStep(
        timestamp=timestamp,
        filtered_mean=np.array([1.0]),
        filtered_covariance=np.array([[1.0]]),
        predicted_mean=np.array([1.0]),
        predicted_covariance=np.array([[1.0]]),
        transition_matrix=np.array([[1.0]]),
    )


def test_initialization_observation_and_pipeline_properties() -> None:
    with pytest.raises(TypeError, match="timestamp must be a datetime"):
        InitializationObservation("bad", 1.0, 2.0)

    pipeline = _pipeline()
    assert pipeline.max_window_steps == 4
    assert pipeline.provisional_states() == ()
    assert (
        pipeline.export_state().phase is PipelineInitializationPhase.WAITING_FOR_FIRST
    )
    pipeline.process(timestamp=START, storage=100.0, discharge=4.0)
    assert (
        pipeline.export_state().phase is PipelineInitializationPhase.WAITING_FOR_SECOND
    )


def test_pipeline_export_rejects_inconsistent_live_state() -> None:
    pipeline = _pipeline()
    pipeline._unfinished_observations.append(_observation(0))
    with pytest.raises(RuntimeError, match="inconsistent with the smoother"):
        pipeline.export_state()

    pipeline = _pipeline()
    pipeline._unfinished_observations.append(_observation(0))
    pipeline._smoother._active.append(_Step(START + timedelta(minutes=1)))
    with pytest.raises(RuntimeError, match="replay observations are inconsistent"):
        pipeline.export_state()

    pipeline = _pipeline()
    pipeline._filter_anchor = _Step(START)
    with pytest.raises(RuntimeError, match="unexpected filter state"):
        pipeline.export_state()

    pipeline = _pipeline()
    pipeline._last_filter_step = _Step(START)
    with pytest.raises(RuntimeError, match="no active smoothing state"):
        pipeline.export_state()


def test_pipeline_restore_and_snapshot_validation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _pipeline()
    with pytest.raises(TypeError, match="PipelineState"):
        pipeline.restore_state(object())

    valid_first = _observation(0)
    valid_second = _observation(15)
    anchor = _Step(START)
    cases = [
        (_state(99), "unsupported phase"),
        (
            _state(PipelineInitializationPhase.WAITING_FOR_FIRST, initial=valid_first),
            "waiting-for-first",
        ),
        (
            _state(PipelineInitializationPhase.WAITING_FOR_SECOND),
            "waiting-for-second snapshot is incomplete",
        ),
        (
            _state(
                PipelineInitializationPhase.WAITING_FOR_SECOND,
                initial=valid_first,
                last=None,
            ),
            "no last timestamp",
        ),
        (
            _state(
                PipelineInitializationPhase.WAITING_FOR_SECOND,
                initial=valid_second,
                last=START,
            ),
            "after the last",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
                initial=valid_first,
                observations=(valid_first, valid_second),
            ),
            "contains an initial",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
                observations=(valid_first,),
            ),
            "needs two observations",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
                anchor=anchor,
                observations=(valid_first, valid_second),
            ),
            "has a filter anchor",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_ANCHOR,
                observations=(valid_first, valid_second),
            ),
            "missing a filter anchor",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
                last=valid_first.timestamp,
                observations=(valid_first, valid_first),
            ),
            "strictly increasing",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
                last=START + timedelta(hours=1),
                observations=(valid_first, valid_second),
            ),
            "does not match",
        ),
        (
            _state(
                PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
                last=START,
                observations=(),
            ),
            "missing active observations",
        ),
    ]
    for state, message in cases:
        with pytest.raises(ValueError, match=message):
            pipeline._validate_state(state)

    replay_state = _state(
        PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION,
        last=valid_second.timestamp,
        observations=(valid_first, valid_second),
    )
    monkeypatch.setattr(pipeline, "process", lambda **kwargs: None)
    with pytest.raises(RuntimeError, match="missing its last input timestamp"):
        pipeline.restore_state(replay_state)

    pipeline = _pipeline()
    monkeypatch.setattr(
        pipeline,
        "process",
        lambda **kwargs: setattr(
            pipeline, "_last_input_timestamp", START + timedelta(hours=1)
        ),
    )
    with pytest.raises(RuntimeError, match="does not match its snapshot"):
        pipeline.restore_state(replay_state)


def test_pipeline_requires_smoother_clear_and_releases_legacy_state() -> None:
    pipeline = _pipeline()
    pipeline._smoother.clear = None
    with pytest.raises(RuntimeError, match="does not support checkpoint state"):
        pipeline.export_state()
    with pytest.raises(RuntimeError, match="does not support state restoration"):
        pipeline._clear_state()

    legacy = _LegacySmoother()
    pipeline = _pipeline(legacy)
    first, second = _observation(0), _observation(15)
    pipeline._add_step(_Step(first.timestamp), observation=first)
    pipeline._add_step(_Step(second.timestamp), observation=second)
    assert len(pipeline._fallback_active_steps) == 2
    assert len(pipeline._unfinished_observations) == 2

    released = _LegacySmoother((object(),))
    pipeline = _pipeline(released)
    pipeline._add_step(_Step(START), observation=first)
    assert pipeline._fallback_active_steps == []


def test_pipeline_add_step_detects_legacy_and_modern_inconsistencies() -> None:
    legacy = _LegacySmoother()
    pipeline = _pipeline(legacy)
    pipeline._fallback_active_steps.append(_Step(START))
    with pytest.raises(RuntimeError, match="inconsistent with the smoother"):
        pipeline._add_step(
            _Step(START + timedelta(minutes=15)), observation=_observation(15)
        )

    legacy = _LegacySmoother((object(), object()))
    pipeline = _pipeline(legacy)
    with pytest.raises(RuntimeError, match="more steps than it received"):
        pipeline._add_step(_Step(START), observation=_observation(0))

    modern = _ModernSmoother(releases=(object(),))
    modern.last_finalized_steps = ()
    modern.add_step = lambda step: (object(),)
    pipeline = _pipeline(modern)
    with pytest.raises(RuntimeError, match="inconsistent with its outputs"):
        pipeline._add_step(_Step(START), observation=_observation(0))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", object(), "timestamp must be a datetime"),
        ("timestamp", datetime(2024, 1, 1), "timezone-aware"),
        ("prediction_flag", "SMOOTHED", "prediction_flag"),
        ("smoothing_flag", "PREDICTED", "smoothing_flag"),
        ("mean", np.ones((1, 1)), "shapes are inconsistent"),
    ],
)
def test_smoothed_step_rejects_invalid_fields(
    field: str, value: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "timestamp": START,
        "mean": np.array([1.0]),
        "covariance": np.array([[1.0]]),
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        rts_module.SmoothedStep(**kwargs)


def test_rts_array_shape_validation_and_lag_property() -> None:
    with pytest.raises(ValueError, match="at least one filter step"):
        rts_module._rts_smooth_arrays(
            np.empty((0, 1)),
            np.empty((0, 1, 1)),
            np.empty((0, 1)),
            np.empty((0, 1, 1)),
            np.empty((0, 1, 1)),
        )

    base = np.zeros((1, 1))
    with pytest.raises(ValueError, match="filtered_covariances"):
        rts_module._rts_smooth_arrays(
            base, np.zeros((1, 2, 2)), base, np.zeros((1, 1, 1)), np.empty((0, 1, 1))
        )
    with pytest.raises(ValueError, match="predicted_means"):
        rts_module._rts_smooth_arrays(
            base,
            np.zeros((1, 1, 1)),
            np.zeros((1, 2)),
            np.zeros((1, 1, 1)),
            np.empty((0, 1, 1)),
        )
    with pytest.raises(ValueError, match="predicted_covariances"):
        rts_module._rts_smooth_arrays(
            base, np.zeros((1, 1, 1)), base, np.zeros((1, 2, 2)), np.empty((0, 1, 1))
        )
    with pytest.raises(ValueError, match="transition_matrices"):
        rts_module._rts_smooth_arrays(
            base, np.zeros((1, 1, 1)), base, np.zeros((1, 1, 1)), np.ones((1, 1, 1))
        )

    smoother = OnlineFixedLagRTS(timedelta(minutes=5), max_window_steps=3)
    assert smoother.lag == timedelta(minutes=5)
