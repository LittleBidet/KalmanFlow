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
    BatchTuningResult,
    InflowModelTuningResult,
    NoiseTuningData,
    NoiseTuningResult,
    ReservoirTuningBatch,
    TuningError,
    TuningFailure,
    TuningResult,
    load_reservoir_configs,
    run_filter_with_noise,
    save_reservoir_configs,
    tune_inflow_model,
    tune_noise,
    tune_reservoirs,
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
    "BatchTuningResult",
    "InflowModelTuningResult",
    "NoiseTuningData",
    "NoiseTuningResult",
    "ReservoirTuningBatch",
    "TuningError",
    "TuningFailure",
    "TuningResult",
    "OnlineReservoirInflow",
    "OutputFlag",
    "ReservoirFlowEstimate",
    "ReservoirFlowUpdate",
    "get_reservoir_inflow",
    "get_reservoir_inflow_from_config",
    "initial_filter_step",
    "kalman_filter",
    "kalman_step",
    "load_reservoir_configs",
    "predict_state",
    "run_filter_with_noise",
    "run_inflow_model",
    "save_reservoir_configs",
    "smooth_filter_steps",
    "tune_inflow_model",
    "tune_noise",
    "tune_reservoirs",
]
