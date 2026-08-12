"""Compact trusted checkpoint encoding for reservoir streaming state."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite

import numpy as np

from .kalman import FilterStep
from .pipeline import (
    InitializationObservation,
    PipelineInitializationPhase,
    PipelineState,
)
from .time_utils import to_utc

_FORMAT_VERSION = 1
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_MICROSECONDS_PER_SECOND = 1_000_000
_SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class DecodedReservoirCheckpoint:
    """The trusted, decoded contents of one reservoir checkpoint."""

    reservoir_id: str
    pipeline_state: PipelineState[FilterStep]


class _Reader:
    """Bounds-checked reader for the private binary checkpoint layout."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._position = 0

    @property
    def remaining(self) -> int:
        """Number of unread bytes."""

        return len(self._data) - self._position

    def read(self, size: int) -> bytes:
        """Read exactly ``size`` bytes or reject a truncated checkpoint."""

        if size < 0 or self.remaining < size:
            raise ValueError("truncated reservoir checkpoint")
        start = self._position
        self._position += size
        return self._data[start : start + size]

    def read_byte(self) -> int:
        """Read one unsigned byte."""

        return self.read(1)[0]


def encode_reservoir_checkpoint(
    reservoir_id: str,
    state: PipelineState[FilterStep],
) -> bytes:
    """Encode one trusted reservoir pipeline state without pickling it."""

    phase = state.phase

    encoded_reservoir_id = reservoir_id.encode("utf-8")
    result = bytearray([_FORMAT_VERSION])
    result.extend(_encode_varint(len(encoded_reservoir_id)))
    result.extend(encoded_reservoir_id)
    result.append(int(phase))
    if state.last_input_timestamp is None:
        result.append(0)
    else:
        result.append(1)
        result.extend(_pack_timestamp(state.last_input_timestamp))

    if phase is PipelineInitializationPhase.WAITING_FOR_SECOND:
        first = state.initial_observation
        assert first is not None
        result.extend(_pack_timestamp(first.timestamp))
        result.extend(_pack_float(first.storage))
        result.extend(_pack_float(first.discharge))
    elif phase is PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION:
        observations = state.unfinished_observations
        result.extend(_encode_varint(len(observations)))
        assert state.last_input_timestamp is not None
        result.extend(
            _encode_observations_from_last(
                observations,
                state.last_input_timestamp,
            )
        )
    elif phase is PipelineInitializationPhase.REPLAY_FROM_ANCHOR:
        anchor = state.filter_anchor
        observations = state.unfinished_observations
        assert anchor is not None
        result.extend(_encode_anchor(anchor))
        result.extend(_encode_varint(len(observations)))
        result.extend(_encode_observations(observations, anchor.timestamp))

    return bytes(result)


def decode_reservoir_checkpoint(
    checkpoint: bytes | bytearray | memoryview,
) -> DecodedReservoirCheckpoint:
    """Decode one trusted checkpoint and reject incompatible wire layouts."""

    reader = _Reader(bytes(checkpoint))
    version = reader.read_byte()
    if version != _FORMAT_VERSION:
        raise ValueError(f"unsupported reservoir checkpoint format version: {version}")

    reservoir_id_size = _decode_varint(reader)
    reservoir_id = reader.read(reservoir_id_size).decode("utf-8")
    phase = PipelineInitializationPhase(reader.read_byte())
    has_last_timestamp = reader.read_byte()
    last_input_timestamp = _unpack_timestamp(reader) if has_last_timestamp else None

    initial_observation: InitializationObservation | None = None
    filter_anchor: FilterStep | None = None
    unfinished_observations: tuple[InitializationObservation, ...] = ()
    if phase is PipelineInitializationPhase.WAITING_FOR_SECOND:
        initial_observation = InitializationObservation(
            timestamp=_unpack_timestamp(reader),
            storage=_unpack_float(reader),
            discharge=_unpack_float(reader),
        )
    elif phase is PipelineInitializationPhase.REPLAY_FROM_INITIALIZATION:
        assert last_input_timestamp is not None
        unfinished_observations = _decode_observations_from_last(
            reader,
            last_input_timestamp,
        )
    elif phase is PipelineInitializationPhase.REPLAY_FROM_ANCHOR:
        filter_anchor = _decode_anchor(reader)
        unfinished_observations = _decode_observations(reader, filter_anchor.timestamp)

    if reader.remaining:
        raise ValueError("reservoir checkpoint has trailing bytes")

    state = PipelineState(
        phase=phase,
        last_input_timestamp=last_input_timestamp,
        initial_observation=initial_observation,
        filter_anchor=filter_anchor,
        unfinished_observations=unfinished_observations,
    )
    return DecodedReservoirCheckpoint(
        reservoir_id=reservoir_id,
        pipeline_state=state,
    )


def _encode_anchor(anchor: FilterStep) -> bytes:
    """Encode only the filtered anchor state, not its transient diagnostics."""

    mean = np.asarray(anchor.filtered_mean, dtype=float)
    covariance = np.asarray(anchor.filtered_covariance, dtype=float)
    result = bytearray(_pack_timestamp(anchor.timestamp))
    for value in mean:
        result.extend(_pack_float(value))
    for row, column in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        result.extend(_pack_float(covariance[row, column]))
    return bytes(result)


def _decode_anchor(reader: _Reader) -> FilterStep:
    """Recreate a lightweight anchor suitable for backend replay."""

    timestamp = _unpack_timestamp(reader)
    mean = np.asarray([_unpack_float(reader) for _ in range(3)], dtype=float)
    packed_covariance = [_unpack_float(reader) for _ in range(6)]
    covariance = np.asarray(
        [
            [packed_covariance[0], packed_covariance[1], packed_covariance[2]],
            [packed_covariance[1], packed_covariance[3], packed_covariance[4]],
            [packed_covariance[2], packed_covariance[4], packed_covariance[5]],
        ],
        dtype=float,
    )
    return FilterStep(
        timestamp=timestamp,
        filtered_mean=mean,
        filtered_covariance=covariance,
        predicted_mean=mean,
        predicted_covariance=covariance,
        transition_matrix=np.eye(3),
    )


def _encode_observations(
    observations: tuple[InitializationObservation, ...],
    origin: datetime,
) -> bytes:
    """Encode raw active-window observations using UTC microsecond deltas."""

    result = bytearray()
    previous_timestamp = origin
    for observation in observations:
        current_timestamp = observation.timestamp
        delta = _timestamp_to_microseconds(
            current_timestamp
        ) - _timestamp_to_microseconds(previous_timestamp)
        _encode_observation(result, observation, delta)
        previous_timestamp = current_timestamp
    return bytes(result)


def _decode_observations(
    reader: _Reader,
    origin: datetime,
) -> tuple[InitializationObservation, ...]:
    """Decode raw active-window observations and their compact missing masks."""

    observations: list[InitializationObservation] = []
    previous_timestamp = origin
    for _ in range(_decode_observation_count(reader)):
        delta = _decode_varint(reader)
        timestamp = _timestamp_from_microseconds(
            _timestamp_to_microseconds(previous_timestamp) + delta
        )
        observations.append(_decode_observation(reader, timestamp))
        previous_timestamp = timestamp
    return tuple(observations)


def _encode_observations_from_last(
    observations: tuple[InitializationObservation, ...],
    last_timestamp: datetime,
) -> bytes:
    """Encode start-up observations backwards from the required last timestamp.

    Before an anchor exists, the explicit last-input timestamp provides the
    only compact absolute base needed for unsigned deltas. The records are
    reversed again during restoration before they reach the pipeline.
    """

    result = bytearray()
    previous_timestamp = last_timestamp
    for observation in reversed(observations):
        current_timestamp = observation.timestamp
        delta = _timestamp_to_microseconds(
            previous_timestamp
        ) - _timestamp_to_microseconds(current_timestamp)
        _encode_observation(result, observation, delta)
        previous_timestamp = current_timestamp
    return bytes(result)


def _decode_observations_from_last(
    reader: _Reader,
    last_timestamp: datetime,
) -> tuple[InitializationObservation, ...]:
    """Decode reverse-ordered start-up observations into chronological order."""

    reverse_observations: list[InitializationObservation] = []
    previous_timestamp = last_timestamp
    for _ in range(_decode_observation_count(reader)):
        delta = _decode_varint(reader)
        timestamp = _timestamp_from_microseconds(
            _timestamp_to_microseconds(previous_timestamp) - delta
        )
        reverse_observations.append(_decode_observation(reader, timestamp))
        previous_timestamp = timestamp
    return tuple(reversed(reverse_observations))


def _encode_observation(
    result: bytearray,
    observation: InitializationObservation,
    delta: int,
) -> None:
    """Append one compact observation record."""

    result.extend(_encode_varint(delta))
    storage = _finite_or_none(observation.storage)
    discharge = _finite_or_none(observation.discharge)
    missing_mask = (1 if storage is None else 0) | (2 if discharge is None else 0)
    result.append(missing_mask)
    if storage is not None:
        result.extend(_pack_float(storage))
    if discharge is not None:
        result.extend(_pack_float(discharge))


def _decode_observation_count(reader: _Reader) -> int:
    """Read a count bounded by the remaining record bytes."""

    count = _decode_varint(reader)
    if count > reader.remaining // 2:
        raise ValueError("reservoir checkpoint has an impossible observation count")
    return count


def _decode_observation(
    reader: _Reader,
    timestamp: datetime,
) -> InitializationObservation:
    """Read one compact observation record."""

    missing_mask = reader.read_byte()
    storage = float("nan") if missing_mask & 0b01 else _unpack_float(reader)
    discharge = float("nan") if missing_mask & 0b10 else _unpack_float(reader)
    return InitializationObservation(timestamp, storage, discharge)


def _finite_or_none(value: float) -> float | None:
    """Represent missing observations compactly."""

    return value if isfinite(value) else None


def _pack_timestamp(timestamp: datetime) -> bytes:
    """Pack a UTC timestamp as a signed microsecond count."""

    return struct.pack(">q", _timestamp_to_microseconds(timestamp))


def _unpack_timestamp(reader: _Reader) -> datetime:
    """Read one signed UTC microsecond count."""

    return _timestamp_from_microseconds(struct.unpack(">q", reader.read(8))[0])


def _timestamp_to_microseconds(timestamp: datetime) -> int:
    """Convert an aware datetime to an exact UTC microsecond count."""

    elapsed = to_utc(timestamp) - _EPOCH
    return (
        elapsed.days * _SECONDS_PER_DAY + elapsed.seconds
    ) * _MICROSECONDS_PER_SECOND + elapsed.microseconds


def _timestamp_from_microseconds(value: int) -> datetime:
    """Convert an exact UTC microsecond count to an aware datetime."""

    try:
        return _EPOCH + timedelta(microseconds=value)
    except OverflowError as error:
        raise ValueError("checkpoint timestamp is out of range") from error


def _pack_float(value: float) -> bytes:
    """Pack one IEEE-754 double."""

    return struct.pack(">d", value)


def _unpack_float(reader: _Reader) -> float:
    """Read one IEEE-754 double."""

    return struct.unpack(">d", reader.read(8))[0]


def _encode_varint(value: int) -> bytes:
    """Encode one non-negative integer as an unsigned variable-length integer."""

    if value < 0:
        raise ValueError("checkpoint variable-length integers must be non-negative")
    result = bytearray()
    while value >= 0x80:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _decode_varint(reader: _Reader) -> int:
    """Decode one bounded unsigned variable-length integer."""

    value = 0
    for index in range(10):
        byte = reader.read_byte()
        value |= (byte & 0x7F) << (7 * index)
        if not byte & 0x80:
            return value
    raise ValueError("checkpoint variable-length integer is too large")
