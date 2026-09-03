from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    OnlineReservoirInflow,
    OutputFlag,
    ReservoirConfig,
    UnitSystem,
)
from kalmone.pipeline import OnlineInflowPipeline


@dataclass(frozen=True)
class FakeFilterStep:
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class FakeSmoothedState:
    timestamp: datetime
    mean: np.ndarray


class FakeBackend:
    def initialize(self, first, second):
        return (
            FakeFilterStep(first.timestamp, first.storage),
            FakeFilterStep(second.timestamp, second.storage),
        )

    def advance(self, previous, *, timestamp, storage, discharge):
        return FakeFilterStep(timestamp, storage)


class FakeFixedLagSmoother:
    def __init__(self, lag: timedelta):
        self.lag = lag
        self.steps: list[FakeFilterStep] = []

    @property
    def max_window_steps(self):
        return 100_000

    @property
    def pending_count(self):
        return len(self.steps)

    def add_step(self, step):
        self.steps.append(step)
        eligible_count = 0
        for active_step in self.steps:
            if step.timestamp - active_step.timestamp >= self.lag:
                eligible_count += 1
            else:
                break

        finalized = tuple(
            FakeSmoothedState(
                timestamp=active_step.timestamp,
                mean=np.array([active_step.value, active_step.value + 10.0]),
            )
            for active_step in self.steps[:eligible_count]
        )
        del self.steps[:eligible_count]
        return finalized

    def provisional(self):
        return tuple(
            FakeSmoothedState(
                timestamp=active_step.timestamp,
                mean=np.array([active_step.value, active_step.value + 10.0]),
            )
            for active_step in self.steps
        )


def test_pipeline_returns_immediate_filter_and_delayed_smoother_outputs():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    pipeline = OnlineInflowPipeline(
        FakeBackend(),
        FakeFixedLagSmoother(timedelta(minutes=10)),
    )

    first = pipeline.process(
        timestamp=start,
        storage=100.0,
        discharge=4.0,
    )
    assert first.filtered_state is None
    assert first.smoothed_states == ()

    second = pipeline.process(
        timestamp=start + timedelta(minutes=5),
        storage=101.0,
        discharge=4.0,
    )
    assert second.filtered_state == FakeFilterStep(start + timedelta(minutes=5), 101.0)
    assert second.filtered_states == (
        FakeFilterStep(start, 100.0),
        FakeFilterStep(start + timedelta(minutes=5), 101.0),
    )
    assert second.smoothed_states == ()

    third = pipeline.process(
        timestamp=start + timedelta(minutes=10),
        storage=102.0,
        discharge=4.0,
    )
    assert third.filtered_state == FakeFilterStep(start + timedelta(minutes=10), 102.0)
    assert third.filtered_states == (
        FakeFilterStep(start + timedelta(minutes=10), 102.0),
    )
    assert [state.timestamp for state in third.smoothed_states] == [start]
    assert third.smoothed_states[0].mean.tolist() == [100.0, 110.0]


def test_reservoir_stream_emits_causal_inflow_before_lagged_revision():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    stream = OnlineReservoirInflow(
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
        smoothing_lag=timedelta(minutes=10),
    )

    first = stream.process(timestamp=start, storage=100.0, discharge=4.0)
    assert first.filtered_inflows == ()
    assert first.revised_inflows == ()
    assert tuple(first.__dataclass_fields__) == ("filtered_inflows", "revised_inflows")

    second = stream.process(
        timestamp=start + timedelta(minutes=5),
        storage=101.0,
        discharge=4.0,
    )
    assert [estimate.timestamp for estimate in second.filtered_inflows] == [
        start,
        start + timedelta(minutes=5),
    ]
    assert [estimate.prediction_flag for estimate in second.filtered_inflows] == [
        OutputFlag.NORMAL,
        OutputFlag.NORMAL,
    ]
    assert all(
        estimate.smoothing_flag is OutputFlag.NON_SMOOTHED
        for estimate in second.filtered_inflows
    )
    assert second.revised_inflows == ()

    third = stream.process(
        timestamp=start + timedelta(minutes=10),
        storage=102.0,
        discharge=4.0,
    )
    assert [estimate.timestamp for estimate in third.filtered_inflows] == [
        start + timedelta(minutes=10)
    ]
    assert [estimate.timestamp for estimate in third.revised_inflows] == [start]
    assert third.revised_inflows[0].prediction_flag is OutputFlag.NORMAL
    assert third.revised_inflows[0].smoothing_flag is OutputFlag.SMOOTHED


def test_reservoir_stream_marks_single_and_double_missing_as_predicted():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    stream = OnlineReservoirInflow(
        q_storage=0.1,
        q_inflow=0.1,
        q_outflow=0.1,
        r_storage=0.25,
        r_outflow=0.5,
        smoothing_lag=timedelta(minutes=10),
    )

    stream.process(timestamp=start, storage=100.0, discharge=4.0)
    stream.process(
        timestamp=start + timedelta(minutes=5),
        storage=101.0,
        discharge=4.0,
    )
    single_missing = stream.process(
        timestamp=start + timedelta(minutes=10),
        storage=float("nan"),
        discharge=4.0,
    )
    double_missing = stream.process(
        timestamp=start + timedelta(minutes=15),
        storage=float("nan"),
        discharge=float("nan"),
    )

    assert single_missing.filtered_inflows[0].prediction_flag is OutputFlag.PREDICTED
    assert double_missing.filtered_inflows[0].prediction_flag is OutputFlag.PREDICTED


def test_reservoir_stream_revises_with_smoothed_inflow_not_latent_outflow():
    start = datetime(2024, 1, 1, tzinfo=UTC)
    config = ReservoirConfig(
        reservoir_id="si-demo",
        reservoir_name="SI Demo",
        q=np.diag([1e-12, 1e-12, 1e-12]),
        r=np.diag([1e-12, 1e-12]),
        p0=np.diag([2.0, 3.0, 4.0]),
        smoothing_lag=timedelta(seconds=30),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.SYSTEM_FLOW_RATE,
        model_version="test-model",
        configuration_version="test-config",
        unit_system=UnitSystem.si(),
    )
    stream = OnlineReservoirInflow.from_config(config)

    stream.process(timestamp=start, storage=100.0, discharge=2.0)
    update = stream.process(
        timestamp=start + timedelta(seconds=30),
        storage=160.0,
        discharge=2.0,
    )

    assert update.filtered_inflows[0].value == 4.0
    assert update.revised_inflows[0].timestamp == start
    assert update.revised_inflows[0].value == 4.0
