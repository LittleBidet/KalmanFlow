"""Public tools for estimating reservoir inflow from storage and outflow data."""

from importlib.metadata import PackageNotFoundError, version

from .core import (
    OnlineReservoirInflow,
    ReservoirFlowEstimate,
    ReservoirFlowUpdate,
    get_reservoir_inflow,
    get_reservoir_inflow_from_config,
)
from .flags import OutputFlag
from .kalman import (
    FilterStep,
    KalmanFilterResult,
    initial_filter_step,
    kalman_filter,
    kalman_step,
    predict_state,
)
from .models import (
    ReservoirStateSpaceModel,
    StateSpaceModel,
)
from .observations import Observation
from .pandas_api import run_inflow_model
from .pipeline import (
    OnlineInflowPipeline,
    PipelineUpdate,
)
from .reservoir_backend import ReservoirBackend
from .reservoir_config import (
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
)
from .rts import OnlineFixedLagRTS, SmoothedStep, smooth_filter_steps
from .tuning import (
    InflowConfigEvaluationResult,
    InflowTuningResult,
    InflowTuningSettings,
    TuningError,
    TuningWindow,
    elapsed_lag_autocorrelation,
    evaluate_inflow_config,
    joint_predictive_nlpd,
    marginal_predictive_nlpd,
    normalized_innovation_squared,
    prior_hourly_increment_to_q,
    storage_conditional_nlpd,
    tune_inflow_process_noise,
)
from .units import CFS_TO_ACRE_FEET_PER_SECOND, UnitSystem

try:
    __version__ = version("kalmone")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
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
    "get_reservoir_inflow",
    "get_reservoir_inflow_from_config",
    "initial_filter_step",
    "kalman_filter",
    "kalman_step",
    "predict_state",
    "run_inflow_model",
    "smooth_filter_steps",
    "InflowConfigEvaluationResult",
    "InflowTuningResult",
    "InflowTuningSettings",
    "TuningError",
    "TuningWindow",
    "elapsed_lag_autocorrelation",
    "evaluate_inflow_config",
    "joint_predictive_nlpd",
    "marginal_predictive_nlpd",
    "normalized_innovation_squared",
    "prior_hourly_increment_to_q",
    "storage_conditional_nlpd",
    "tune_inflow_process_noise",
]
