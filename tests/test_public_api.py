"""Smoke coverage for the package's documented top-level API."""

from __future__ import annotations

import kalmanflow

EXPECTED_EXPORTS = {
    "CFS_TO_ACRE_FEET_PER_SECOND",
    "FilterStep",
    "InflowUnits",
    "InitializationStrategy",
    "KalmanFilterResult",
    "Observation",
    "OnlineFixedLagRTS",
    "OnlineInflowPipeline",
    "PipelineUpdate",
    "ReservoirConfig",
    "ReservoirBackend",
    "ReservoirStateSpaceModel",
    "SmoothedStep",
    "StateSpaceModel",
    "UnitSystem",
    "OnlineReservoirInflow",
    "OutputFlag",
    "ReservoirFlowEstimate",
    "ReservoirFlowUpdate",
    "add_inflow_uncertainty_intervals",
    "get_reservoir_inflow",
    "get_reservoir_inflow_from_config",
    "initial_filter_step",
    "kalman_filter",
    "kalman_step",
    "predict_state",
    "run_inflow_model",
    "smooth_filter_steps",
    "BayesianEvaluationSettings",
    "BayesianTuningError",
    "ConfigEvaluationResult",
    "ValidationWindow",
    "evaluate_configuration",
    "BayesianTuningResult",
    "BayesianTuningSettings",
    "tune_inflow_noise_bayesian",
}


def test_top_level_public_exports_are_bound_and_unique() -> None:
    exported_names = kalmanflow.__all__

    assert len(exported_names) == len(set(exported_names))
    assert set(exported_names) == EXPECTED_EXPORTS
    assert all(getattr(kalmanflow, name) is not None for name in exported_names)
