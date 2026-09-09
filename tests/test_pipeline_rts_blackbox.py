"""Black-box tests for online pipeline and fixed-lag RTS state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import numpy.testing as npt
import pytest

from kalmanflow import (
    OnlineFixedLagRTS,
    OnlineInflowPipeline,
    ReservoirBackend,
    ReservoirStateSpaceModel,
    initial_filter_step,
    kalman_step,
    smooth_filter_steps,
)
from kalmanflow.kalman import FilterStep
from kalmanflow.pipeline import InitializationObservation

Q = np.diag([0.5, 0.1, 0.2])
R = np.diag([0.25, 0.5])
P0 = np.diag([4.0, 9.0, 16.0])


def _reservoir_pipeline(
    *,
    smoothing_lag: timedelta = timedelta(minutes=30),
    max_window_steps: int = 100,
) -> OnlineInflowPipeline:
    model = ReservoirStateSpaceModel(
        q_continuous=Q,
    )
    backend = ReservoirBackend(
        model=model,
        initial_covariance=P0,
        observation_covariance=R,
    )
    smoother = OnlineFixedLagRTS(smoothing_lag, max_window_steps=max_window_steps)
    return OnlineInflowPipeline(
        backend=backend,
        smoother=smoother,
        max_window_steps=max_window_steps,
    )


def _forward_steps(count: int = 4) -> tuple[FilterStep, ...]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([0.2, 0.05])
    observation_matrix = np.array([[1.0, 0.0]])
    observation_covariance = np.array([[0.4]])
    steps = [
        initial_filter_step(
            timestamp=start,
            initial_mean=np.array([10.0, 0.5]),
            initial_covariance=np.diag([2.0, 3.0]),
            observation=np.array([10.0]),
            observation_matrix=observation_matrix,
            observation_covariance=observation_covariance,
        )
    ]
    for index in range(1, count):
        previous = steps[-1]
        steps.append(
            kalman_step(
                timestamp=start + timedelta(hours=index),
                previous_filtered_mean=previous.filtered_mean,
                previous_filtered_covariance=previous.filtered_covariance,
                transition_matrix=transition,
                process_covariance=process,
                observation=np.array([10.0 + 0.7 * index]),
                observation_matrix=observation_matrix,
                observation_covariance=observation_covariance,
                control_offset=np.array([-0.1, 0.0]),
            )
        )
    return tuple(steps)


class TestReservoirBackendValidation:
    """EP checks for direct low-level backend construction."""

    def test_covariances_are_defensively_copied(self) -> None:
        initial_covariance = P0.copy()
        backend = ReservoirBackend(
            model=ReservoirStateSpaceModel(q_continuous=Q),
            initial_covariance=initial_covariance,
            observation_covariance=R,
        )

        initial_covariance[0, 0] = 999.0
        assert backend.initial_covariance[0, 0] == P0[0, 0]
        with pytest.raises(ValueError):
            backend.initial_covariance[0, 0] = 1.0

    def test_covariance_shape_is_checked_at_construction(self) -> None:
        with pytest.raises(ValueError, match="initial_covariance must have shape"):
            ReservoirBackend(
                model=ReservoirStateSpaceModel(q_continuous=Q),
                initial_covariance=np.eye(2),
                observation_covariance=R,
            )


class TestPipelineStateTransitions:
    """State Transition Testing for OnlineInflowPipeline."""

    def test_missing_storage_before_initialization_returns_no_state(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline()

        update = pipeline.process(
            timestamp=start,
            storage=float("nan"),
            discharge=4.0,
        )

        assert pipeline.initialized is False
        assert update.filtered_state is None
        assert update.smoothed_states == ()

    def test_first_valid_storage_waits_for_second_sample(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline()

        first = pipeline.process(timestamp=start, storage=100.0, discharge=4.0)

        assert pipeline.initialized is False
        assert first.filtered_state is None
        assert first.smoothed_states == ()

    def test_first_valid_storage_requires_finite_discharge(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline()

        with pytest.raises(ValueError, match="finite discharge"):
            pipeline.process(timestamp=start, storage=100.0, discharge=float("nan"))

    def test_second_valid_storage_initializes_and_returns_filter_state(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline(smoothing_lag=timedelta(hours=2))

        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)
        second = pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=4.0,
        )

        assert pipeline.initialized is True
        assert second.filtered_state is not None
        assert second.filtered_state.timestamp == start + timedelta(minutes=15)

    def test_missing_discharge_on_second_initialization_is_allowed(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline()

        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)
        second = pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=float("nan"),
        )

        assert pipeline.initialized is True
        assert second.filtered_state is not None

    def test_post_initialization_missing_storage_is_predict_only(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline(smoothing_lag=timedelta(hours=2))

        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)
        pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=4.0,
        )
        previous = pipeline.latest_filter_step
        assert previous is not None

        update = pipeline.process(
            timestamp=start + timedelta(minutes=30),
            storage=float("nan"),
            discharge=5.0,
        )

        assert update.filtered_state is not None
        assert update.filtered_state.timestamp == start + timedelta(minutes=30)
        assert np.isfinite(update.filtered_state.filtered_mean[0])

    def test_missing_discharge_is_a_partial_observation(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline(smoothing_lag=timedelta(hours=2))

        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)
        pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=4.0,
        )
        update = pipeline.process(
            timestamp=start + timedelta(minutes=30),
            storage=102.0,
            discharge=float("nan"),
        )

        assert update.filtered_state is not None
        assert update.filtered_state.timestamp == start + timedelta(minutes=30)
        assert np.isfinite(update.filtered_state.filtered_mean).all()

class TestPipelineTimestampBoundaries:
    """BVA: timestamp ordering and timezone boundaries."""

    def test_naive_timestamp_rejected(self) -> None:
        pipeline = _reservoir_pipeline()
        with pytest.raises(ValueError, match="timezone-aware"):
            pipeline.process(
                timestamp=datetime(2024, 1, 1),
                storage=100.0,
                discharge=4.0,
            )

    def test_equal_timestamp_rejected(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline()
        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)

        with pytest.raises(ValueError, match="strictly increasing"):
            pipeline.process(
                timestamp=start,
                storage=101.0,
                discharge=4.0,
            )

    def test_decreasing_timestamp_rejected(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline()
        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)

        with pytest.raises(ValueError, match="strictly increasing"):
            pipeline.process(
                timestamp=start - timedelta(minutes=15),
                storage=101.0,
                discharge=4.0,
            )

    def test_elapsed_time_crossing_dst_uses_absolute_seconds(self) -> None:
        zone = ZoneInfo("America/Los_Angeles")
        first_timestamp = datetime(2024, 3, 10, 1, 30, tzinfo=zone)
        second_timestamp = datetime(2024, 3, 10, 3, 30, tzinfo=zone)
        model = ReservoirStateSpaceModel(
            q_continuous=Q,
        )
        backend = ReservoirBackend(
            model=model,
            initial_covariance=P0,
            observation_covariance=R,
        )

        first_step, second_step = backend.initialize(
            InitializationObservation(first_timestamp, 100.0, 0.0),
            InitializationObservation(second_timestamp, 101.0, 0.0),
        )

        assert first_step.predicted_mean[1] == 0.0
        assert second_step.transition_matrix[0, 1] == pytest.approx(3600.0 / 43560.0)


class TestPipelineWindowBoundaries:
    """BVA: max_window_steps boundaries."""

    def test_max_window_steps_below_minimum_rejected(self) -> None:
        model = ReservoirStateSpaceModel(
            q_continuous=Q,
        )
        backend = ReservoirBackend(
            model=model,
            initial_covariance=P0,
            observation_covariance=R,
        )
        with pytest.raises(ValueError, match="max_window_steps must be at least 2"):
            OnlineInflowPipeline(
                backend=backend,
                smoother=OnlineFixedLagRTS(timedelta(minutes=30), max_window_steps=2),
                max_window_steps=1,
            )

    @pytest.mark.parametrize("invalid_size", [2.5, True])
    def test_max_window_steps_requires_an_integer(self, invalid_size: object) -> None:
        model = ReservoirStateSpaceModel(q_continuous=Q)
        backend = ReservoirBackend(
            model=model,
            initial_covariance=P0,
            observation_covariance=R,
        )
        with pytest.raises(TypeError, match="must be an integer"):
            OnlineInflowPipeline(
                backend=backend,
                smoother=OnlineFixedLagRTS(
                    timedelta(minutes=30),
                    max_window_steps=2,
                ),
                max_window_steps=invalid_size,
            )

    def test_pipeline_and_smoother_window_limits_must_match(self) -> None:
        model = ReservoirStateSpaceModel(q_continuous=Q)
        backend = ReservoirBackend(
            model=model,
            initial_covariance=P0,
            observation_covariance=R,
        )
        with pytest.raises(ValueError, match="must match smoother"):
            OnlineInflowPipeline(
                backend=backend,
                smoother=OnlineFixedLagRTS(
                    timedelta(minutes=30),
                    max_window_steps=3,
                ),
                max_window_steps=2,
            )

    def test_window_overflow_raises_without_advancing(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline(
            smoothing_lag=timedelta(hours=10),
            max_window_steps=2,
        )

        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)
        pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=4.0,
        )

        with pytest.raises(OverflowError, match="max_window_steps"):
            pipeline.process(
                timestamp=start + timedelta(minutes=30),
                storage=102.0,
                discharge=4.0,
            )

    def test_full_window_accepts_step_that_finalizes_oldest_state(self) -> None:
        start = datetime(2024, 1, 1, tzinfo=UTC)
        pipeline = _reservoir_pipeline(
            smoothing_lag=timedelta(minutes=30),
            max_window_steps=2,
        )

        pipeline.process(timestamp=start, storage=100.0, discharge=4.0)
        pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=4.0,
        )
        update = pipeline.process(
            timestamp=start + timedelta(minutes=30),
            storage=102.0,
            discharge=4.0,
        )

        assert [state.timestamp for state in update.smoothed_states] == [start]
        assert pipeline.pending_count == 2


class TestFixedLagRTSBoundaries:
    """BVA and state transitions for OnlineFixedLagRTS."""

    def test_zero_lag_rejected(self) -> None:
        with pytest.raises(ValueError, match="lag must be positive"):
            OnlineFixedLagRTS(timedelta(0))

    def test_max_window_steps_below_minimum_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_window_steps must be at least 2"):
            OnlineFixedLagRTS(timedelta(minutes=1), max_window_steps=1)

    def test_exact_lag_boundary_finalizes_state(self) -> None:
        steps = _forward_steps(count=3)
        lag = steps[2].timestamp - steps[0].timestamp
        smoother = OnlineFixedLagRTS(lag, max_window_steps=4)

        assert smoother.add_step(steps[0]) == ()
        assert smoother.add_step(steps[1]) == ()
        finalized = smoother.add_step(steps[2])

        assert len(finalized) == 1
        assert finalized[0].timestamp == steps[0].timestamp

    def test_one_microsecond_below_lag_does_not_finalize(self) -> None:
        steps = _forward_steps(count=2)
        lag = steps[1].timestamp - steps[0].timestamp + timedelta(microseconds=1)
        smoother = OnlineFixedLagRTS(lag, max_window_steps=4)

        smoother.add_step(steps[0])
        finalized = smoother.add_step(steps[1])

        assert finalized == ()

    def test_non_increasing_filter_step_timestamps_rejected(self) -> None:
        steps = _forward_steps(count=2)
        smoother = OnlineFixedLagRTS(timedelta(hours=1), max_window_steps=4)
        smoother.add_step(steps[0])

        with pytest.raises(ValueError, match="strictly increasing"):
            smoother.add_step(steps[0])

    def test_window_overflow_before_finalization(self) -> None:
        steps = _forward_steps(count=3)
        smoother = OnlineFixedLagRTS(timedelta(hours=10), max_window_steps=2)
        smoother.add_step(steps[0])
        smoother.add_step(steps[1])

        with pytest.raises(OverflowError, match="max_window_steps"):
            smoother.add_step(steps[2])

    def test_provisional_returns_active_window_without_finalizing(self) -> None:
        steps = _forward_steps(count=3)
        smoother = OnlineFixedLagRTS(timedelta(hours=10), max_window_steps=4)
        smoother.add_step(steps[0])
        smoother.add_step(steps[1])

        provisional = smoother.provisional()
        assert len(provisional) == 2
        assert smoother.pending_count == 2

    def test_empty_smooth_filter_steps_returns_empty_tuple(self) -> None:
        assert smooth_filter_steps(()) == ()

    def test_single_step_smoothing_is_identity(self) -> None:
        (step,) = _forward_steps(count=1)
        (smoothed,) = smooth_filter_steps((step,))
        assert smoothed.timestamp == step.timestamp
        npt.assert_allclose(smoothed.mean, step.filtered_mean)


@dataclass(frozen=True)
class _BadBackend:
    def initialize(self, first, second):
        return (object(),)

    def advance(self, previous, *, timestamp, storage, discharge):
        return object()


def test_backend_initialize_must_return_two_steps() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    pipeline = OnlineInflowPipeline(
        _BadBackend(),
        OnlineFixedLagRTS(timedelta(minutes=30), max_window_steps=4),
        max_window_steps=4,
    )
    pipeline.process(timestamp=start, storage=100.0, discharge=4.0)

    with pytest.raises(ValueError, match="exactly two filter steps"):
        pipeline.process(
            timestamp=start + timedelta(minutes=15),
            storage=101.0,
            discharge=4.0,
        )
