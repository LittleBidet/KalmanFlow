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
    NoiseTuningData,
    NoiseTuningResult,
    run_filter_with_noise,
    tune_noise,
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
    "NoiseTuningData",
    "NoiseTuningResult",
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
    "run_filter_with_noise",
    "smooth_filter_steps",
    "tune_noise",
]
