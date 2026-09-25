"""Boundary and malformed-input regression tests for shared utilities."""

from datetime import UTC, datetime, timedelta, tzinfo
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kalmanflow import ReservoirBackend, ReservoirStateSpaceModel, UnitSystem
from kalmanflow import _reservoir_checkpoint as codec
from kalmanflow._validation import measurement_value
from kalmanflow.pipeline import PipelineInitializationPhase, PipelineState
from kalmanflow.reservoir_config import _freeze_metadata
from kalmanflow.time_utils import elapsed_seconds, to_utc


@pytest.mark.parametrize("value", [float("inf"), -float("inf")])
def test_measurement_rejects_infinity(value):
    with pytest.raises(ValueError, match="finite or NaN"):
        measurement_value(value, name="storage")


class UndefinedOffset(tzinfo):
    def utcoffset(self, dt):
        return None


@pytest.mark.parametrize("zone", [None, UndefinedOffset()])
def test_utc_conversion_rejects_missing_offset(zone):
    with pytest.raises(ValueError, match="timezone-aware"):
        to_utc(datetime(2024, 1, 1, tzinfo=zone))


@pytest.mark.parametrize("allow_zero", [False, True])
def test_elapsed_time_rejects_backwards_time(allow_zero):
    now = datetime(2024, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="elapsed time must be"):
        elapsed_seconds(now - timedelta(seconds=1), now, allow_zero=allow_zero)


def test_unit_conversion_rejects_nonfinite_inputs():
    units = UnitSystem.si()
    with pytest.raises(ValueError, match="flow_rate must be finite"):
        units.validate_flow_rate(float("nan"))
    with pytest.raises(ValueError, match="volume must be finite"):
        units.volume_to_flow_rate(float("inf"), 1)


def test_backend_rejects_vector_observation_matrix():
    model = SimpleNamespace(observation_matrix=np.ones(3))
    with pytest.raises(ValueError, match="two-dimensional"):
        ReservoirBackend(model, np.eye(3), np.eye(2))


def test_model_rejects_invalid_units():
    with pytest.raises(TypeError, match="UnitSystem"):
        ReservoirStateSpaceModel(np.eye(3), unit_system="cfs")


def test_metadata_sets_become_immutable():
    original = {"labels": {"upstream", "downstream"}}
    frozen = _freeze_metadata(original)
    original["labels"].add("changed")
    assert frozen["labels"] == frozenset({"upstream", "downstream"})


def test_version_falls_back_without_installed_distribution(monkeypatch):
    import kalmanflow

    def missing_distribution(name):
        assert name == "kalmanflow"
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", missing_distribution)
    # Execute in an isolated package namespace to preserve imported class identities.
    source = Path(kalmanflow.__file__)
    namespace = {"__name__": "kalmanflow", "__package__": "kalmanflow"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    assert namespace["__version__"] == "0+unknown"


def test_version_uses_installed_distribution_metadata(monkeypatch):
    import kalmanflow

    def installed_version(name):
        assert name == "kalmanflow"
        return "1.2.3"

    monkeypatch.setattr(metadata, "version", installed_version)
    source = Path(kalmanflow.__file__)
    namespace = {"__name__": "kalmanflow", "__package__": "kalmanflow"}
    exec(compile(source.read_text(), str(source), "exec"), namespace)
    assert namespace["__version__"] == "1.2.3"


def test_empty_checkpoint_roundtrip_and_fingerprint_validation():
    state = PipelineState(
        phase=PipelineInitializationPhase.WAITING_FOR_FIRST,
        last_input_timestamp=None,
        initial_observation=None,
        filter_anchor=None,
        unfinished_observations=(),
    )
    with pytest.raises(ValueError, match="32 bytes"):
        codec.encode_reservoir_checkpoint("reservoir", b"short", state)
    encoded = codec.encode_reservoir_checkpoint("reservoir", bytes(32), state)
    decoded = codec.decode_reservoir_checkpoint(encoded)
    assert decoded.reservoir_id == "reservoir"
    assert decoded.configuration_fingerprint == bytes(32)
    assert decoded.pipeline_state == state


def test_checkpoint_rejects_impossible_count():
    with pytest.raises(ValueError, match="impossible observation count"):
        codec._decode_observation_count(codec._Reader(b"\x02\x00\x00"))


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1])
def test_checkpoint_rejects_timestamp_outside_datetime_range(value):
    with pytest.raises(ValueError, match="timestamp is out of range"):
        codec._timestamp_from_microseconds(value)


def test_checkpoint_rejects_invalid_varints():
    with pytest.raises(ValueError, match="non-negative"):
        codec._encode_varint(-1)
    with pytest.raises(ValueError, match="too large"):
        codec._decode_varint(codec._Reader(b"\x80" * 10))


@pytest.mark.parametrize("value", [True, "1.0", None, complex(1, 2)])
def test_measurement_rejects_coercion(value):
    with pytest.raises(TypeError, match="storage must be a real number"):
        measurement_value(value, name="storage")
