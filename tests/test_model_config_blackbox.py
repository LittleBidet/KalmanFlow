"""Black-box tests for reservoir models, units, configuration, and observations."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import numpy as np
import numpy.testing as npt
import pytest

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    Observation,
    ReservoirConfig,
    ReservoirStateSpaceModel,
    UnitSystem,
)

Q = np.diag([2.0, 0.5, 0.25])
R = np.diag([0.25, 0.5])
P0 = np.diag([4.0, 9.0, 16.0])


def _config(**overrides: object) -> ReservoirConfig:
    values: dict[str, object] = {
        "reservoir_id": "demo",
        "reservoir_name": "Demo Reservoir",
        "q": Q,
        "r": R,
        "p0": P0,
        "smoothing_lag": timedelta(minutes=30),
        "initialization_strategy": InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        "inflow_units": InflowUnits.CUBIC_FEET_PER_SECOND,
        "model_version": "model-v1",
        "configuration_version": "config-v1",
        "tuning_metadata": {"tunable": ["Q", "R"]},
    }
    values.update(overrides)
    return ReservoirConfig(**values)


def _model(**overrides: object) -> ReservoirStateSpaceModel:
    values: dict[str, object] = {
        "q_continuous": Q,
    }
    values.update(overrides)
    return ReservoirStateSpaceModel(**values)


class TestReservoirModelBoundaries:
    """BVA/EP for ReservoirStateSpaceModel."""

    def test_continuous_process_noise_scales_with_elapsed_time(self) -> None:
        model = _model()
        short = model.process_covariance(900.0)
        long = model.process_covariance(1800.0)
        assert long[1, 1] == pytest.approx(2.0 * short[1, 1])
        assert long[0, 1] > 2.0 * short[0, 1]

    @pytest.mark.parametrize(
        ("elapsed", "message"),
        [
            (0.0, "elapsed_seconds must be positive and finite"),
            (-1.0, "elapsed_seconds must be positive and finite"),
            (float("inf"), "elapsed_seconds must be positive and finite"),
        ],
    )
    def test_nonpositive_or_nonfinite_elapsed_rejected(
        self, elapsed: float, message: str
    ) -> None:
        model = _model()
        with pytest.raises(ValueError, match=message):
            model.transition_matrix(elapsed)

    def test_q_shape_boundary_rejects_wrong_dimensions(self) -> None:
        with pytest.raises(ValueError, match="q_continuous must have shape"):
            _model(q_continuous=np.eye(2))

    def test_q_must_be_symmetric(self) -> None:
        with pytest.raises(ValueError, match="finite and symmetric"):
            _model(
                q_continuous=np.array(
                    [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]
                )
            )

    def test_q_psd_tolerance_boundary(self) -> None:
        barely_psd = np.diag([1.0, 1.0, -1e-13])
        model = _model(q_continuous=barely_psd)
        assert model.q_continuous.shape == (3, 3)

        not_psd = np.diag([1.0, 1.0, -1e-11])
        with pytest.raises(ValueError, match="positive semidefinite"):
            _model(q_continuous=not_psd)

    def test_from_config_builds_model(self) -> None:
        model = ReservoirStateSpaceModel.from_config(_config())
        npt.assert_allclose(model.q_continuous, Q)

    def test_negative_flow_rate_is_allowed_for_conversion(self) -> None:
        model = _model()
        assert model.discharge_volume(-4.0, 900.0) < 0.0


class TestUnitSystemPartitions:
    """EP/BVA for UnitSystem."""

    def test_us_customary_and_si_presets(self) -> None:
        us = UnitSystem.us_customary()
        si = UnitSystem.si()
        assert us.volume_label == "acre-ft"
        assert si.volume_label == "m^3"
        assert si.flow_to_volume(2.0, 30.0) == 60.0

    @pytest.mark.parametrize(
        ("factor", "message"),
        [
            (0.0, "positive and finite"),
            (-1.0, "positive and finite"),
            (float("inf"), "positive and finite"),
        ],
    )
    def test_invalid_conversion_factor_rejected(
        self, factor: float, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            UnitSystem(flow_to_volume_per_second=factor)

    @pytest.mark.parametrize("label", ["", "   "])
    def test_empty_labels_rejected(self, label: str) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            UnitSystem(volume_label=label)

    def test_nonfinite_flow_rate_rejected(self) -> None:
        units = UnitSystem.si()
        with pytest.raises(ValueError, match="flow_rate must be finite"):
            units.flow_to_volume(float("nan"), 30.0)

    def test_nonpositive_elapsed_rejected(self) -> None:
        units = UnitSystem.si()
        with pytest.raises(ValueError, match="elapsed_seconds must be positive"):
            units.flow_to_volume(2.0, 0.0)

    def test_nonpositive_interval_rejected(self) -> None:
        units = UnitSystem.si()
        with pytest.raises(ValueError, match="elapsed_seconds"):
            units.volume_to_flow_rate(1.0, 0.0)

    def test_physical_rate_model_supports_si_units(self) -> None:
        model = _model(unit_system=UnitSystem.si())
        npt.assert_allclose(
            model.transition_matrix(30.0),
            [[1.0, 30.0, -30.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )
        assert model.initial_inflow(100.0, 160.0, 2.0, 30.0) == pytest.approx(4.0)
        assert model.initial_outflow(2.0) == pytest.approx(2.0)


class TestReservoirConfigPartitions:
    """EP/BVA for ReservoirConfig validation."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"reservoir_id": ""}, "reservoir_id must not be empty"),
            ({"reservoir_name": "   "}, "reservoir_name must not be empty"),
            ({"q": np.eye(2)}, "q must have shape"),
            ({"r": np.diag([0.0, 1.0])}, "r diagonal entries"),
            ({"p0": np.diag([1.0, -1.0, 1.0])}, "positive semidefinite"),
            ({"smoothing_lag": timedelta(0)}, "smoothing_lag must be positive"),
        ],
    )
    def test_invalid_configuration_partitions(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            _config(**overrides)

    def test_enum_values_are_coerced(self) -> None:
        config = _config(
            initialization_strategy="first_two_valid_storage",
            inflow_units="cfs",
        )
        assert (
            config.initialization_strategy
            is InitializationStrategy.FIRST_TWO_VALID_STORAGE
        )
        assert config.inflow_units is InflowUnits.CUBIC_FEET_PER_SECOND

    def test_invalid_enum_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            _config(inflow_units="quadratic")

    def test_invalid_unit_system_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="unit_system must be a UnitSystem"):
            _config(unit_system="not-a-unit-system")

    def test_cfs_inflow_units_require_cfs_unit_system(self) -> None:
        with pytest.raises(ValueError, match="cfs inflow_units"):
            _config(unit_system=UnitSystem.si())

    def test_system_flow_rate_units_support_si_unit_system(self) -> None:
        config = _config(
            unit_system=UnitSystem.si(),
            inflow_units=InflowUnits.SYSTEM_FLOW_RATE,
        )
        assert config.inflow_units is InflowUnits.SYSTEM_FLOW_RATE

    def test_metadata_is_frozen(self) -> None:
        source_array = np.array([1.0])
        config = _config(
            tuning_metadata={"nested": {"array": source_array}},
        )
        assert isinstance(config.tuning_metadata, MappingProxyType)
        with pytest.raises(TypeError):
            config.tuning_metadata["new"] = "value"
        source_array[0] = 2.0
        assert config.tuning_metadata["nested"]["array"][0] == 1.0
        with pytest.raises(ValueError):
            config.tuning_metadata["nested"]["array"][0] = 2.0

    def test_covariance_arrays_are_read_only(self) -> None:
        config = _config()
        with pytest.raises(ValueError):
            config.q[0, 0] = 0.0

    def test_with_tuned_noise_does_not_mutate_original(self) -> None:
        config = _config()
        tuned = config.with_tuned_noise(q=Q * 2.0, r=R * 3.0)
        npt.assert_allclose(config.q, Q)
        npt.assert_allclose(tuned.q, Q * 2.0)


class TestObservationPartitions:
    """EP for Observation input contract."""

    def test_nan_storage_and_discharge_are_accepted(self) -> None:
        observation = Observation(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            storage=float("nan"),
            discharge=float("nan"),
        )
        assert np.isnan(observation.storage)
        assert np.isnan(observation.discharge)

    def test_naive_timestamp_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            Observation(
                timestamp=datetime(2024, 1, 1),
                storage=100.0,
                discharge=4.0,
            )
