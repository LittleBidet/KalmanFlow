"""Black-box coverage for resumable reservoir streaming."""

from __future__ import annotations

import inspect
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    Observation,
    OnlineReservoirInflow,
    ReservoirConfig,
    UnitSystem,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
)


def _config(**overrides: object) -> ReservoirConfig:
    values: dict[str, object] = {
        "reservoir_id": "reservoir-a",
        "reservoir_name": "Reservoir A",
        "q": np.diag([0.1, 0.2, 0.3]),
        "r": np.diag([0.25, 0.5]),
        "p0": np.diag([4.0, 9.0, 16.0]),
        "smoothing_lag": timedelta(minutes=30),
        "initialization_strategy": InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        "inflow_units": InflowUnits.CUBIC_FEET_PER_SECOND,
        "model_version": "model-v1",
        "configuration_version": "config-v1",
        "unit_system": UnitSystem.us_customary(),
    }
    values.update(overrides)
    return ReservoirConfig(**values)


def _observations(
    *,
    start: datetime = datetime(2024, 1, 1, tzinfo=UTC),
    count: int = 8,
) -> tuple[Observation, ...]:
    return tuple(
        Observation(
            timestamp=start + timedelta(minutes=15 * index),
            storage=100.0 + index,
            discharge=4.0 + 0.1 * index,
        )
        for index in range(count)
    )


def _assert_updates_equal(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_update, expected_update in zip(actual, expected, strict=True):
        for actual_values, expected_values in (
            (actual_update.filtered_inflows, expected_update.filtered_inflows),
            (actual_update.revised_inflows, expected_update.revised_inflows),
        ):
            assert len(actual_values) == len(expected_values)
            for actual_estimate, expected_estimate in zip(
                actual_values, expected_values, strict=True
            ):
                assert actual_estimate.timestamp == expected_estimate.timestamp
                assert (
                    actual_estimate.prediction_flag is expected_estimate.prediction_flag
                )
                assert (
                    actual_estimate.smoothing_flag is expected_estimate.smoothing_flag
                )
                assert actual_estimate.value == pytest.approx(expected_estimate.value)


def test_restore_after_every_observation_matches_uninterrupted_execution() -> None:
    config = _config()
    observations = _observations()
    uninterrupted = OnlineReservoirInflow.from_config(config)
    expected = tuple(uninterrupted.process(observation) for observation in observations)

    stream = OnlineReservoirInflow.from_config(config)
    actual = []
    for observation in observations:
        actual.append(stream.process(observation))
        stream = OnlineReservoirInflow.from_checkpoint(
            stream.checkpoint(),
            config=config,
        )

    _assert_updates_equal(actual, expected)


def test_interleaved_reservoir_streams_stay_isolated() -> None:
    first_config = _config(reservoir_id="reservoir-a")
    second_config = _config(reservoir_id="reservoir-b")
    observations = _observations()
    first_reference = OnlineReservoirInflow.from_config(first_config)
    second_reference = OnlineReservoirInflow.from_config(second_config)
    expected_first = [
        first_reference.process(observation) for observation in observations
    ]
    expected_second = [
        second_reference.process(observation) for observation in observations
    ]

    streams = {
        first_config.reservoir_id: OnlineReservoirInflow.from_config(first_config),
        second_config.reservoir_id: OnlineReservoirInflow.from_config(second_config),
    }
    actual_first = []
    actual_second = []
    for first, second in zip(observations, observations, strict=True):
        first_stream = streams[first_config.reservoir_id]
        actual_first.append(first_stream.process(first))
        streams[first_config.reservoir_id] = OnlineReservoirInflow.from_checkpoint(
            first_stream.checkpoint(), config=first_config
        )

        second_stream = streams[second_config.reservoir_id]
        actual_second.append(second_stream.process(second))
        streams[second_config.reservoir_id] = OnlineReservoirInflow.from_checkpoint(
            second_stream.checkpoint(), config=second_config
        )

    _assert_updates_equal(actual_first, expected_first)
    _assert_updates_equal(actual_second, expected_second)


def test_checkpoint_preserves_reservoir_id_and_rejects_mismatch() -> None:
    config = _config(reservoir_id="lake-alpha")
    stream = OnlineReservoirInflow.from_config(config)
    stream.process(_observations()[0])
    checkpoint = stream.checkpoint()

    restored = OnlineReservoirInflow.from_checkpoint(checkpoint, config=config)
    assert restored.reservoir_id == "lake-alpha"
    with pytest.raises(RuntimeError, match="successful process result"):
        restored.checkpoint()
    restored.process(_observations()[1])
    assert restored.checkpoint()

    with pytest.raises(ValueError, match="reservoir_id"):
        OnlineReservoirInflow.from_checkpoint(
            checkpoint,
            config=_config(reservoir_id="lake-beta"),
        )


@pytest.mark.parametrize("checkpoint_index", [0, 1, 2, 4])
def test_restore_at_initialization_and_window_positions(checkpoint_index: int) -> None:
    config = _config()
    observations = _observations(count=7)
    continuous = OnlineReservoirInflow.from_config(config)
    expected = [continuous.process(observation) for observation in observations]

    stream = OnlineReservoirInflow.from_config(config)
    actual = []
    for index, observation in enumerate(observations):
        actual.append(stream.process(observation))
        if index == checkpoint_index:
            stream = OnlineReservoirInflow.from_checkpoint(
                stream.checkpoint(), config=config
            )
    _assert_updates_equal(actual, expected)


def test_restore_before_initialization_retains_the_timestamp_cursor() -> None:
    config = _config()
    observations = _observations()
    stream = OnlineReservoirInflow.from_config(config)
    missing = Observation(
        timestamp=observations[0].timestamp,
        storage=float("nan"),
        discharge=4.0,
    )
    stream.process(missing)
    stream = OnlineReservoirInflow.from_checkpoint(stream.checkpoint(), config=config)

    with pytest.raises(ValueError, match="strictly increasing"):
        stream.process(missing)
    with pytest.raises(RuntimeError, match="successful process result"):
        stream.checkpoint()

    reference = OnlineReservoirInflow.from_config(config)
    reference.process(missing)
    expected = reference.process(observations[1])
    actual = stream.process(observations[1])
    _assert_updates_equal((actual,), (expected,))
    assert stream.checkpoint()


def test_missing_observations_replay_without_duplicate_outputs() -> None:
    config = _config()
    observations = list(_observations(count=7))
    observations[2] = Observation(
        timestamp=observations[2].timestamp,
        storage=float("nan"),
        discharge=observations[2].discharge,
    )
    observations[3] = Observation(
        timestamp=observations[3].timestamp,
        storage=observations[3].storage,
        discharge=float("nan"),
    )
    observations[4] = Observation(
        timestamp=observations[4].timestamp,
        storage=float("nan"),
        discharge=float("nan"),
    )

    uninterrupted = OnlineReservoirInflow.from_config(config)
    expected = [uninterrupted.process(observation) for observation in observations]
    stream = OnlineReservoirInflow.from_config(config)
    actual = []
    for index, observation in enumerate(observations):
        actual.append(stream.process(observation))
        if index == 3:
            stream = OnlineReservoirInflow.from_checkpoint(
                stream.checkpoint(), config=config
            )

    _assert_updates_equal(actual, expected)
    assert [
        estimate.timestamp
        for update in actual[4:]
        for estimate in update.revised_inflows
    ].count(observations[0].timestamp) == 0


def test_checkpoint_lifecycle_rejects_reentrant_and_failed_processing(
    monkeypatch,
) -> None:
    config = _config()
    stream = OnlineReservoirInflow.from_config(config)
    first, second, third = _observations(count=3)
    original_process = stream._pipeline.process

    def process_while_checkpointing(**kwargs):
        with pytest.raises(RuntimeError, match="while processing"):
            stream.checkpoint()
        return original_process(**kwargs)

    monkeypatch.setattr(stream._pipeline, "process", process_while_checkpointing)
    stream.process(first)
    assert stream.checkpoint()

    with pytest.raises(ValueError, match="strictly increasing"):
        stream.process(first)
    with pytest.raises(RuntimeError, match="successful process result"):
        stream.checkpoint()

    stream.process(second)
    assert stream.checkpoint()

    stream.process(third)
    assert stream.checkpoint()


def test_process_many_matches_repeated_processing_and_rolls_back_on_failure() -> None:
    config = _config()
    observations = _observations(count=5)
    repeated = OnlineReservoirInflow.from_config(config)
    expected = tuple(repeated.process(observation) for observation in observations)

    stream = OnlineReservoirInflow.from_config(config)
    actual = stream.process_many(observations)
    _assert_updates_equal(actual, expected)

    rollback_stream = OnlineReservoirInflow.from_config(config)
    rollback_stream.process(observations[0])
    with pytest.raises(ValueError, match="strictly increasing"):
        rollback_stream.process_many((observations[1], observations[1]))
    with pytest.raises(RuntimeError, match="successful process result"):
        rollback_stream.checkpoint()

    recovered = rollback_stream.process(observations[1])
    control = OnlineReservoirInflow.from_config(config)
    control.process(observations[0])
    _assert_updates_equal((recovered,), (control.process(observations[1]),))
    assert rollback_stream.checkpoint()


@pytest.mark.parametrize("max_window_steps", [3, 20])
def test_checkpoint_format_version_and_size_are_compact(
    max_window_steps: int,
) -> None:
    config = _config(smoothing_lag=timedelta(days=365))
    stream = OnlineReservoirInflow.from_config(
        config,
        max_window_steps=max_window_steps,
    )
    stream.process_many(_observations(count=max_window_steps))
    checkpoint = stream.checkpoint()

    assert len(checkpoint) <= 128 + 24 * max_window_steps
    unsupported = bytes([255]) + checkpoint[1:]
    with pytest.raises(ValueError, match="unsupported.*format version"):
        OnlineReservoirInflow.from_checkpoint(unsupported, config=config)


def test_batch_apis_remain_checkpoint_free() -> None:
    assert "checkpoint" not in inspect.signature(get_reservoir_inflow).parameters
    assert (
        "checkpoint"
        not in inspect.signature(get_reservoir_inflow_from_config).parameters
    )
