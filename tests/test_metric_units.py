"""Metric API parity and physical equivalence across unit systems."""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    InflowUnits,
    InitializationStrategy,
    Observation,
    OnlineReservoirInflow,
    ReservoirConfig,
    UnitSystem,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
)


def config(units=None):
    return ReservoirConfig(
        reservoir_id="test",
        reservoir_name="Test",
        q=np.array([[2.0, 0.01, 0.02], [0.01, 0.3, 0.04], [0.02, 0.04, 0.5]]),
        r=np.array([[4.0, 0.2], [0.2, 2.0]]),
        p0=np.array([[100.0, 2.0, 3.0], [2.0, 1000.0, 4.0], [3.0, 4.0, 1000.0]]),
        smoothing_lag=timedelta(minutes=15),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        model_version="v1",
        configuration_version="v1",
        unit_system=units or UnitSystem.us_customary(),
    )


def observations():
    index = pd.date_range("2026-01-01", periods=8, freq="15min", tz="UTC")
    return (
        pd.Series(
            [10000.0, 10001.0, 10003.0, np.nan, 10006.0, 10008.0, 10010.0, 10011.0],
            index,
        ),
        pd.Series([25.0, 26.0, 24.0, 25.0, np.nan, 26.0, 25.0, 25.0], index),
    )


def test_conversion_preserves_physical_estimates_and_uncertainty():
    original = config()
    metric = original.to_units(UnitSystem.si())
    volume, flow = original.unit_system.conversion_factors_to(UnitSystem.si())
    storage, discharge = observations()
    us = get_reservoir_inflow_from_config(
        storage, discharge, original, include_uncertainty=True
    )
    si = get_reservoir_inflow_from_config(
        storage * volume, discharge * flow, metric, include_uncertainty=True
    )
    for column in (
        "estimated_inflow",
        "revised_inflow",
        "estimated_inflow_standard_deviation",
        "revised_inflow_standard_deviation",
    ):
        np.testing.assert_allclose(
            si[column], us[column] * flow, rtol=1e-7, atol=1e-9, equal_nan=True
        )
    restored = metric.to_units(UnitSystem.us_customary())
    for name in ("q", "r", "p0"):
        np.testing.assert_allclose(getattr(restored, name), getattr(original, name))
    assert original.unit_system == UnitSystem.us_customary()
    assert metric.inflow_units == InflowUnits.SYSTEM_FLOW_RATE


def test_simple_metric_batch_and_stream_match_config():
    units = UnitSystem.si()
    storage, discharge = observations()
    settings = dict(
        q_storage=2.0,
        q_inflow=0.3,
        q_outflow=0.5,
        r_storage=4.0,
        r_outflow=2.0,
        smoothing_lag=timedelta(minutes=15),
    )
    metric = config(units)
    from dataclasses import replace

    metric = replace(
        metric,
        q=np.diag([2.0, 0.3, 0.5]),
        r=np.diag([4.0, 2.0]),
        p0=np.diag([100.0, 1000.0, 1000.0]),
    )
    expected = get_reservoir_inflow_from_config(storage, discharge, metric)
    actual = get_reservoir_inflow(storage, discharge, **settings, unit_system=units)
    pd.testing.assert_frame_equal(actual, expected)
    stream = OnlineReservoirInflow(**settings, unit_system=units)
    reference = OnlineReservoirInflow.from_config(metric)
    for timestamp, value in storage.items():
        observation = Observation(
            timestamp.to_pydatetime(), value, discharge[timestamp]
        )
        assert stream.process(observation) == reference.process(observation)


def test_unit_conversion_validation():
    us, si = UnitSystem.us_customary(), UnitSystem.si()
    assert si.conversion_factors_to(si) == (1.0, 1.0)
    np.testing.assert_allclose(
        us.conversion_factors_to(si), (1233.48183754752, 0.028316846592)
    )
    with pytest.raises(TypeError, match="UnitSystem"):
        us.conversion_factors_to("si")
    with pytest.raises(ValueError, match="built-in"):
        us.conversion_factors_to(UnitSystem("litres", "litres/s", 1.0))
    assert config(si).inflow_units == InflowUnits.SYSTEM_FLOW_RATE
