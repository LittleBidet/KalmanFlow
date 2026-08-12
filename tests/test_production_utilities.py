from datetime import UTC, datetime, timedelta

import numpy as np

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    NoiseTuningData,
    ReservoirConfig,
    UnitSystem,
    run_filter_with_noise,
)


def _config() -> ReservoirConfig:
    return ReservoirConfig(
        reservoir_id="test",
        reservoir_name="Test Reservoir",
        q=np.diag([1.0, 0.1, 0.2]),
        r=np.diag([1.0, 2.0]),
        p0=np.diag([100.0, 1000.0, 1000.0]),
        smoothing_lag=timedelta(hours=1),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="test-model",
        configuration_version="test-config",
        unit_system=UnitSystem.us_customary(),
    )


def test_noise_tuning_data_runs_a_candidate_without_scipy() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    data = NoiseTuningData(
        timestamps=tuple(start + timedelta(minutes=15 * i) for i in range(4)),
        storage=np.array([100.0, 100.5, np.nan, 101.5]),
        discharge=np.array([4.0, 4.5, 5.0, 4.0]),
    )

    result = run_filter_with_noise(
        _config(),
        data,
        q_storage=1.0,
        q_inflow=0.1,
        q_outflow=0.2,
        r_storage=1.0,
        r_outflow=2.0,
    )

    assert result.filtered_means.shape == (4, 3)


def test_noise_tuning_handles_irregular_cadence() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    data = NoiseTuningData(
        timestamps=(
            start,
            start + timedelta(minutes=5),
            start + timedelta(minutes=15),
            start + timedelta(minutes=75),
        ),
        storage=np.array([100.0, 100.1, 100.3, 101.0]),
        discharge=np.array([4.0, 4.5, 5.0, 4.0]),
    )

    result = run_filter_with_noise(
        _config(),
        data,
        q_storage=1.0,
        q_inflow=0.1,
        q_outflow=0.2,
        r_storage=1.0,
        r_outflow=2.0,
    )

    assert result.filtered_means.shape == (4, 3)
    assert np.isfinite(result.filtered_means).all()
