"""Optional tools for choosing model and measurement noise settings.

SciPy is imported only when :func:`tune_noise` is called.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import numpy as np

from .kalman import KalmanFilterResult, kalman_filter
from .models import (
    RESERVOIR_OBSERVATION_MATRIX,
    ReservoirStateSpaceModel,
)
from .reservoir_config import ReservoirConfig
from .time_utils import elapsed_seconds, to_utc

ObjectiveName = Literal["loglik", "rmse"]


@dataclass(frozen=True)
class NoiseTuningData:
    """Pre-cleaned storage and measured outflow arrays for one tuning interval.

    The first two storage values and first discharge value must be finite.
    Later missing storage or discharge values are allowed and are treated as
    missing observation components.
    """

    timestamps: tuple[datetime, ...]
    storage: np.ndarray
    discharge: np.ndarray

    def __post_init__(self) -> None:
        timestamps = tuple(self.timestamps)
        if len(timestamps) < 2:
            raise ValueError("tuning data needs at least two timestamps")
        for index, timestamp in enumerate(timestamps):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("tuning timestamps must be timezone-aware")
            if index and to_utc(timestamp) <= to_utc(timestamps[index - 1]):
                raise ValueError("tuning timestamps must be strictly increasing")

        storage = np.asarray(self.storage, dtype=float).copy()
        discharge = np.asarray(self.discharge, dtype=float).copy()
        expected_shape = (len(timestamps),)
        if storage.shape != expected_shape:
            raise ValueError(f"storage must have shape {expected_shape}")
        if discharge.shape != expected_shape:
            raise ValueError(f"discharge must have shape {expected_shape}")

        storage.setflags(write=False)
        discharge.setflags(write=False)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "discharge", discharge)


@dataclass(frozen=True)
class NoiseTuningResult:
    """Result of optimizing diagonal continuous-time Q and diagonal R."""

    objective: ObjectiveName
    q_storage: float
    q_inflow: float
    q_outflow: float
    r_storage: float
    r_outflow: float
    objective_value: float
    success: bool
    iterations: int
    method: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return the tuning result as ordinary serializable values."""

        return {
            "objective": self.objective,
            "q_storage": self.q_storage,
            "q_inflow": self.q_inflow,
            "q_outflow": self.q_outflow,
            "q_convention": "continuous_time_diffusion",
            "r_storage": self.r_storage,
            "r_outflow": self.r_outflow,
            "objective_value": self.objective_value,
            "success": self.success,
            "iterations": self.iterations,
            "method": self.method,
            "message": self.message,
        }


def run_filter_with_noise(
    config: ReservoirConfig,
    data: NoiseTuningData,
    *,
    q_storage: float,
    q_inflow: float,
    q_outflow: float,
    r_storage: float,
    r_outflow: float,
) -> KalmanFilterResult:
    """Run one candidate set of noise values through the filter."""

    parameters = np.asarray(
        [q_storage, q_inflow, q_outflow, r_storage, r_outflow],
        dtype=float,
    )
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("candidate noise parameters must be positive and finite")

    model = ReservoirStateSpaceModel(
        q_continuous=np.diag([q_storage, q_inflow, q_outflow]),
        unit_system=config.unit_system,
    )
    (
        observations,
        initial_mean,
        transitions,
        interval_seconds,
    ) = _prepare_data(model, data)
    process_covariance = np.stack(
        [model.process_covariance(elapsed) for elapsed in interval_seconds]
    )

    return kalman_filter(
        observations=observations,
        initial_mean=initial_mean,
        initial_covariance=config.p0,
        transition_matrix=transitions,
        process_covariance=process_covariance,
        observation_matrix=RESERVOIR_OBSERVATION_MATRIX,
        observation_covariance=np.diag([r_storage, r_outflow]),
    )


def tune_noise(
    config: ReservoirConfig,
    data: NoiseTuningData,
    *,
    objective: ObjectiveName = "loglik",
    initial: tuple[float, float, float, float, float] | None = None,
    method: str = "Nelder-Mead",
    maxiter: int = 500,
) -> NoiseTuningResult:
    """Optimize process and measurement noise in log space.

    This function requires SciPy at call time. It never mutates ``config``.
    """

    if objective not in {"loglik", "rmse"}:
        raise ValueError("objective must be 'loglik' or 'rmse'")
    try:
        from scipy.optimize import minimize
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "tune_noise requires SciPy; install it in the optional tuning environment"
        ) from error

    if initial is None:
        initial = (
            float(config.q[0, 0]),
            float(config.q[1, 1]),
            float(config.q[2, 2]),
            float(config.r[0, 0]),
            float(config.r[1, 1]),
        )
    initial_values = np.asarray(initial, dtype=float)
    if (
        initial_values.shape != (5,)
        or not np.all(np.isfinite(initial_values))
        or np.any(initial_values <= 0.0)
    ):
        raise ValueError("initial must contain five positive finite values")

    _validated_inputs(data)

    def evaluate(log_parameters: np.ndarray) -> float:
        with np.errstate(over="ignore", invalid="ignore"):
            parameters = np.exp(log_parameters)
        if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
            return float("inf")
        try:
            result = run_filter_with_noise(
                config,
                data,
                q_storage=float(parameters[0]),
                q_inflow=float(parameters[1]),
                q_outflow=float(parameters[2]),
                r_storage=float(parameters[3]),
                r_outflow=float(parameters[4]),
            )
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            return float("inf")
        if objective == "loglik":
            return -float(result.log_likelihood)
        innovations = result.innovations[1:]
        innovation_variances = np.diagonal(
            result.innovation_covariances[1:],
            axis1=1,
            axis2=2,
        )
        standard_deviations = np.sqrt(innovation_variances)
        standardized = innovations / standard_deviations
        finite_innovations = standardized[np.isfinite(standardized)]
        if finite_innovations.size == 0:
            return float("inf")
        return float(np.sqrt(np.mean(finite_innovations**2)))

    result = minimize(
        evaluate,
        np.log(initial_values),
        method=method,
        options={"maxiter": int(maxiter)},
    )
    values = np.exp(np.asarray(result.x, dtype=float))
    return NoiseTuningResult(
        objective=objective,
        q_storage=float(values[0]),
        q_inflow=float(values[1]),
        q_outflow=float(values[2]),
        r_storage=float(values[3]),
        r_outflow=float(values[4]),
        objective_value=float(result.fun),
        success=bool(result.success),
        iterations=int(getattr(result, "nit", 0)),
        method=method,
        message=str(result.message),
    )


def _prepare_data(
    model: ReservoirStateSpaceModel, data: NoiseTuningData
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Prepare observations and time-dependent model values for filtering."""

    timestamps = data.timestamps
    observations, discharge = _validated_inputs(data)
    observations = np.column_stack([observations, discharge])

    initial_elapsed = elapsed_seconds(timestamps[1], timestamps[0])
    initial_inflow = model.initial_inflow(
        observations[0, 0],
        observations[1, 0],
        discharge[0],
        initial_elapsed,
    )
    interval_seconds = np.asarray(
        [
            elapsed_seconds(timestamps[index + 1], timestamps[index])
            for index in range(len(timestamps) - 1)
        ],
        dtype=float,
    )
    transitions = np.stack(
        [model.transition_matrix(elapsed) for elapsed in interval_seconds]
    )
    return (
        observations,
        np.array(
            [observations[0, 0], initial_inflow, model.initial_outflow(discharge[0])],
            dtype=float,
        ),
        transitions,
        interval_seconds,
    )


def _validated_inputs(data: NoiseTuningData) -> tuple[np.ndarray, np.ndarray]:
    """Check the values needed to start tuning and return input arrays."""

    observations = np.asarray(data.storage, dtype=float)
    discharge = np.asarray(data.discharge, dtype=float)
    if not np.all(np.isfinite(observations[:2])):
        raise ValueError("tuning data needs two finite initial storage values")
    if not np.isfinite(discharge[0]):
        raise ValueError("the first tuning discharge value must be finite")
    return observations, discharge


__all__ = [
    "NoiseTuningData",
    "NoiseTuningResult",
    "run_filter_with_noise",
    "tune_noise",
]
