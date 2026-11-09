"""Regression tests for explicit stream and pipeline lifecycle states."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import kalmanflow.core as core
from kalmanflow import (
    Observation,
    OnlineFixedLagRTS,
    OnlineInflowPipeline,
    OnlineReservoirInflow,
    ReservoirConfig,
    UnitSystem,
)
from kalmanflow.pipeline import PipelineInitializationPhase

START = datetime(2024, 1, 1, tzinfo=UTC)


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="reservoir-a",
        reservoir_name="Reservoir A",
        q=np.diag([0.1, 0.2, 0.3]),
        r=np.diag([0.25, 0.5]),
        p0=np.diag([4.0, 9.0, 16.0]),
        smoothing_lag=timedelta(minutes=30),
        model_version="model-v1",
        configuration_version="config-v1",
        unit_system=UnitSystem.us_customary(),
    )


def _observations(count: int = 6) -> tuple[Observation, ...]:
    return tuple(
        Observation(
            timestamp=START + timedelta(minutes=15 * index),
            storage=100.0 + index,
            discharge=4.0 + 0.1 * index,
        )
        for index in range(count)
    )


def _revised_timestamps(updates) -> list[datetime]:
    return [
        estimate.timestamp
        for update in updates
        for estimate in update.revised_inflows
    ]


def _reservoir_pipeline(
    lag: timedelta, *, max_window_steps: int = 100_000
) -> OnlineInflowPipeline:
    backend = core._build_default_backend(
        q_storage=0.1, q_inflow=0.2, q_outflow=0.3, r_storage=0.25, r_outflow=0.5
    )
    return OnlineInflowPipeline(
        backend, OnlineFixedLagRTS(lag, max_window_steps=max_window_steps)
    )


def _revised_values(updates) -> list[float]:
    return [
        estimate.value
        for update in updates
        for estimate in update.revised_inflows
    ]


class _Step:
    def __init__(self, timestamp: datetime) -> None:
        self.timestamp = timestamp


class _Backend:
    def initialize(self, first, second):
        return _Step(first.timestamp), _Step(second.timestamp)

    def advance(self, previous, *, timestamp, storage, discharge):
        del previous, storage, discharge
        return _Step(timestamp)


class _RejectingSmoother:
    """Honours the add_step contract: a rejected step leaves no trace."""

    max_window_steps = 4

    def __init__(self, *, reject_call: int) -> None:
        self._active: list[_Step] = []
        self._calls = 0
        self._reject_call = reject_call
        self.last_finalized_steps: tuple[_Step, ...] = ()

    @property
    def pending_count(self) -> int:
        return len(self._active)

    def add_step(self, step: _Step) -> tuple[object, ...]:
        self._calls += 1
        if self._calls == self._reject_call:
            raise ValueError("step rejected")
        self._active.append(step)
        return ()

    def provisional(self) -> tuple[object, ...]:
        return ()

    def active_steps(self) -> tuple[_Step, ...]:
        return tuple(self._active)

    def clear(self) -> None:
        self._active.clear()


def test_process_failure_after_absorbing_an_observation_fails_the_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost update must not leave a stream that looks healthy.

    Before the fix, the pipeline kept the observation, the revision released
    by that call was discarded, and a retry failed as out of order.
    """

    config = _config()
    observations = _observations()
    reference = OnlineReservoirInflow.from_config(config, include_uncertainty=True)
    expected = [reference.process(item) for item in observations]

    stream = OnlineReservoirInflow.from_config(config, include_uncertainty=True)
    for item in observations[:3]:
        stream.process(item)
    checkpoint = stream.checkpoint()

    with monkeypatch.context() as patch:
        patch.setattr(
            core,
            "_inflow_standard_deviation",
            lambda covariance, *, name: (_ for _ in ()).throw(
                ValueError("covariance check failed")
            ),
        )
        with pytest.raises(ValueError, match="covariance check failed"):
            stream.process(observations[3])

    assert stream.failed
    assert not stream._pipeline.failed
    for call in (
        lambda: stream.process(observations[3]),
        lambda: stream.process(observations[4]),
        lambda: stream.process_many(()),
        lambda: stream.process_many(observations[4:]),
        stream.checkpoint,
    ):
        with pytest.raises(RuntimeError, match="stream failed"):
            call()

    # The documented recovery path replays the lost observation.
    restored = OnlineReservoirInflow.from_checkpoint(
        checkpoint, config=config, include_uncertainty=True
    )
    assert not restored.failed
    recovered = [restored.process(item) for item in observations[3:]]
    assert _revised_timestamps(recovered) == _revised_timestamps(expected[3:])
    assert _revised_values(recovered) == pytest.approx(_revised_values(expected[3:]))


def test_rejected_process_leaves_the_stream_usable() -> None:
    stream = OnlineReservoirInflow.from_config(_config())
    first, second, third = _observations(3)
    stream.process(first)
    stream.process(second)

    with pytest.raises(ValueError, match="strictly increasing"):
        stream.process(first)
    with pytest.raises(TypeError, match="must provide timestamp"):
        stream.process(object())

    assert not stream.failed
    stream.process(third)
    assert stream.checkpoint()


def test_smoother_failure_after_accepting_a_step_fails_pipeline_and_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = OnlineReservoirInflow.from_config(_config())
    first, second, third = _observations(3)
    stream.process(first)
    stream.process(second)

    smoother = stream._pipeline._smoother
    original_add_step = smoother.add_step

    def accept_with_inconsistent_release(step):
        # Accepts the step, then reports a release it does not account for.
        return (*original_add_step(step), object())

    monkeypatch.setattr(smoother, "add_step", accept_with_inconsistent_release)
    with pytest.raises(RuntimeError, match="inconsistent with its outputs"):
        stream.process(third)

    assert stream._pipeline.failed
    assert stream.failed
    with pytest.raises(RuntimeError, match="stream failed"):
        stream.checkpoint()


def test_process_many_rollback_keeps_stream_usable() -> None:
    stream = OnlineReservoirInflow.from_config(_config())
    observations = _observations(4)
    stream.process(observations[0])

    with pytest.raises(ValueError, match="strictly increasing"):
        stream.process_many((observations[1], observations[1]))

    assert not stream.failed
    stream.process_many(observations[1:])
    assert stream.checkpoint()


def test_pipeline_rejected_step_before_acceptance_changes_nothing() -> None:
    pipeline = _reservoir_pipeline(timedelta(days=1), max_window_steps=2)
    pipeline.process(timestamp=START, storage=100.0, discharge=4.0)
    pipeline.process(
        timestamp=START + timedelta(minutes=15), storage=101.0, discharge=4.0
    )
    before = pipeline.export_state()

    with pytest.raises(OverflowError, match="max_window_steps"):
        pipeline.process(
            timestamp=START + timedelta(minutes=30), storage=102.0, discharge=4.0
        )

    assert not pipeline.failed
    assert pipeline.export_state() == before


def test_pipeline_initialization_failure_after_first_step_is_terminal() -> None:
    pipeline = OnlineInflowPipeline(_Backend(), _RejectingSmoother(reject_call=2))
    pipeline.process(timestamp=START, storage=100.0, discharge=4.0)
    snapshot = pipeline.export_state()

    with pytest.raises(ValueError, match="step rejected"):
        pipeline.process(
            timestamp=START + timedelta(minutes=15), storage=101.0, discharge=4.0
        )

    assert pipeline.failed
    assert not pipeline.initialized
    assert pipeline.latest_filter_step is None
    assert pipeline.last_input_timestamp is None
    with pytest.raises(RuntimeError, match="pipeline failed"):
        pipeline.process(
            timestamp=START + timedelta(minutes=30), storage=102.0, discharge=4.0
        )
    with pytest.raises(RuntimeError, match="pipeline failed"):
        pipeline.export_state()

    pipeline.restore_state(snapshot)
    assert not pipeline.failed
    assert pipeline.export_state() == snapshot


def test_pipeline_rejected_first_step_leaves_initialization_pending() -> None:
    pipeline = OnlineInflowPipeline(_Backend(), _RejectingSmoother(reject_call=1))
    pipeline.process(timestamp=START, storage=100.0, discharge=4.0)
    snapshot = pipeline.export_state()

    with pytest.raises(ValueError, match="step rejected"):
        pipeline.process(
            timestamp=START + timedelta(minutes=15), storage=101.0, discharge=4.0
        )

    assert not pipeline.failed
    assert pipeline.export_state() == snapshot


def test_missing_storage_while_waiting_for_second_advances_the_cursor() -> None:
    pipeline = _reservoir_pipeline(timedelta(minutes=30))
    first = Observation(START, 100.0, 4.0)
    pipeline.process(timestamp=first.timestamp, storage=100.0, discharge=4.0)
    skipped = START + timedelta(minutes=15)
    pipeline.process(timestamp=skipped, storage=float("nan"), discharge=4.0)

    state = pipeline.export_state()
    assert state.phase is PipelineInitializationPhase.WAITING_FOR_SECOND
    assert state.initial_observation == first
    assert state.last_input_timestamp == skipped
    with pytest.raises(ValueError, match="strictly increasing"):
        pipeline.process(timestamp=skipped, storage=101.0, discharge=4.0)

    update = pipeline.process(
        timestamp=START + timedelta(minutes=30), storage=101.0, discharge=4.0
    )
    assert pipeline.initialized
    assert pipeline.latest_filter_step is update.filtered_states[-1]
