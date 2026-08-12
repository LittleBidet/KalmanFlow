"""Public adapters for reservoir-inflow estimation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas

from .flags import OutputFlag
from .models import ReservoirStateSpaceModel
from .pipeline import ObservationLike, OnlineInflowPipeline
from .reservoir_backend import ReservoirBackend
from .reservoir_config import ReservoirConfig
from .rts import OnlineFixedLagRTS


@dataclass(frozen=True)
class ReservoirFlowEstimate:
    """One timestamped public flow estimate."""

    timestamp: datetime
    value: float
    prediction_flag: OutputFlag = OutputFlag.NORMAL
    smoothing_flag: OutputFlag = OutputFlag.NON_SMOOTHED

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        prediction_flag = OutputFlag(self.prediction_flag)
        if prediction_flag not in {OutputFlag.NORMAL, OutputFlag.PREDICTED}:
            raise ValueError(
                "prediction_flag must be NORMAL or PREDICTED"
            )
        smoothing_flag = OutputFlag(self.smoothing_flag)
        if smoothing_flag not in {
            OutputFlag.SMOOTHED,
            OutputFlag.NON_SMOOTHED,
        }:
            raise ValueError(
                "smoothing_flag must be SMOOTHED or NON_SMOOTHED"
            )
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "prediction_flag", prediction_flag)
        object.__setattr__(self, "smoothing_flag", smoothing_flag)

    @property
    def flag(self) -> OutputFlag:
        """Backward-friendly short name for the prediction flag."""

        return self.prediction_flag

    @property
    def output_flag(self) -> OutputFlag:
        """Alias for the prediction flag used by output-oriented callers."""

        return self.prediction_flag

    @property
    def output_flags(self) -> tuple[OutputFlag, OutputFlag]:
        """Return prediction and smoothing flags in that order."""

        return self.prediction_flag, self.smoothing_flag

    @property
    def flags(self) -> tuple[OutputFlag, OutputFlag]:
        """Alias for :attr:`output_flags`."""

        return self.output_flags


@dataclass(frozen=True)
class ReservoirFlowUpdate:
    """Streaming outputs."""

    filtered_inflows: tuple[ReservoirFlowEstimate, ...]
    estimated_outflows: tuple[ReservoirFlowEstimate, ...]


class OnlineReservoirInflow:
    """Reservoir-specific streaming with staggered flow outputs.

    Filtered inflow estimates are returned as soon as forward filter steps are
    available. Estimated outflow estimates are returned only after the fixed-lag
    smoother finalizes their timestamps. Storage and provisional smoothed states
    remain internal.
    """

    def __init__(
        self,
        *,
        q_storage: float,
        q_inflow: float,
        q_outflow: float,
        r_storage: float,
        r_outflow: float,
        smoothing_lag: timedelta = timedelta(hours=12),
        max_window_steps: int = 100_000,
    ) -> None:
        """Create an estimator with the supplied noise settings."""

        self._pipeline = _build_default_pipeline(
            q_storage=q_storage,
            q_inflow=q_inflow,
            q_outflow=q_outflow,
            r_storage=r_storage,
            r_outflow=r_outflow,
            smoothing_lag=smoothing_lag,
            max_window_steps=max_window_steps,
        )

    @classmethod
    def from_config(
        cls,
        config: ReservoirConfig,
        *,
        max_window_steps: int = 100_000,
    ) -> OnlineReservoirInflow:
        """Create an estimator from a validated reservoir configuration."""

        instance = cls.__new__(cls)
        instance._pipeline = _build_configured_pipeline(
            config,
            max_window_steps=max_window_steps,
        )
        return instance

    @property
    def initialized(self) -> bool:
        """Whether the first two valid storage observations were processed."""

        return self._pipeline.initialized

    @property
    def pending_count(self) -> int:
        """Number of states still waiting for the smoothing lag."""

        return self._pipeline.pending_count

    def process_observation(self, observation: ObservationLike) -> ReservoirFlowUpdate:
        """Process an object exposing timestamp, storage, and discharge."""

        return self.process(
            timestamp=observation.timestamp,
            storage=observation.storage,
            discharge=observation.discharge,
        )

    def process(
        self,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> ReservoirFlowUpdate:
        """Process one observation and return currently available flow outputs."""

        update = self._pipeline.process(
            timestamp=timestamp,
            storage=storage,
            discharge=discharge,
        )
        return ReservoirFlowUpdate(
            filtered_inflows=tuple(
                ReservoirFlowEstimate(
                    timestamp=step.timestamp,
                    value=float(step.filtered_mean[1]),
                    prediction_flag=step.prediction_flag,
                    smoothing_flag=OutputFlag.NON_SMOOTHED,
                )
                for step in update.filtered_states
            ),
            estimated_outflows=tuple(
                ReservoirFlowEstimate(
                    timestamp=state.timestamp,
                    value=float(state.mean[2]),
                    prediction_flag=state.prediction_flag,
                    smoothing_flag=OutputFlag.SMOOTHED,
                )
                for state in update.smoothed_states
            ),
        )


def get_reservoir_inflow(
    reservoir_storage: pandas.Series,
    reservoir_outflow: pandas.Series,
    q_storage: float,
    q_inflow: float,
    q_outflow: float,
    r_storage: float,
    r_outflow: float,
    smoothing_lag: timedelta = timedelta(hours=12),
    *,
    max_window_steps: int = 100_000,
) -> pandas.DataFrame:
    """
    Adapt pandas storage/outflow series to an inflow pipeline.

    State is ``[storage, inflow_rate, true_outflow_rate]``. Storage and
    measured outflow are noisy observations.

    Units
    -----
    - ``reservoir_storage``: acre-ft
    - ``reservoir_outflow``: measured cfs
    - returned ``estimated_inflow``: cfs, causal filtered estimates
    - returned ``estimated_outflow``: cfs, finalized fixed-lag estimates

    :param reservoir_storage: Pre-cleaned storage series in acre-ft. Index must
        be a timezone-aware, strictly increasing ``DateTimeIndex``. NaN marks
        missing storage; any finite discharge still contributes a partial update.
    :param reservoir_outflow: Pre-cleaned outflow/discharge series in cfs on
        the same index and span as ``reservoir_storage``. NaN marks missing
        discharge; missing discharge is omitted from that Kalman update.
    :param q_storage: Continuous-time process-noise spectral density for storage.
    :param q_inflow: Continuous-time process-noise spectral density for inflow rate.
    :param q_outflow: Continuous-time process-noise spectral density for true
        outflow rate.
    :param r_storage: Positive storage measurement-noise variance.
    :param r_outflow: Positive outflow measurement-noise variance.
    :param smoothing_lag: Time lag before delayed outflow estimates are finalized.
    :param max_window_steps: Maximum number of active states for the fixed-lag
        smoother.
    Input series must be pre-cleaned by the caller: parsed, aligned, sorted, and
    deduplicated. Missing storage or discharge samples may remain as NaN; the
    pipeline applies the documented missing-value rules but does not otherwise
    clean the series.
    :return: DataFrame indexed like the inputs with estimate and provenance
        columns. ``estimated_inflow_flag`` and ``estimated_outflow_flag`` are
        ``NORMAL`` or ``PREDICTED``; the latter is used for both single- and
        double-missing observations. The corresponding ``*_smoothing_flag``
        columns are ``NON_SMOOTHED`` for causal inflow and ``SMOOTHED`` for
        finalized outflow. Trailing outflow values remain NaN and have a
        ``NON_SMOOTHED`` smoothing flag until their lag has elapsed.
    """
    if not reservoir_storage.index.equals(reservoir_outflow.index):
        raise ValueError("storage and outflow indexes must match exactly")

    stream = OnlineReservoirInflow(
        q_storage=q_storage,
        q_inflow=q_inflow,
        q_outflow=q_outflow,
        r_storage=r_storage,
        r_outflow=r_outflow,
        smoothing_lag=smoothing_lag,
        max_window_steps=max_window_steps,
    )
    return _run_batch(reservoir_storage, reservoir_outflow, stream)


def get_reservoir_inflow_from_config(
    reservoir_storage: pandas.Series,
    reservoir_outflow: pandas.Series,
    config: ReservoirConfig,
    *,
    max_window_steps: int = 100_000,
) -> pandas.DataFrame:
    """Estimate reservoir flows using one validated configuration.

    The input and output flow rates use ``config.unit_system.flow_label``;
    storage uses ``config.unit_system.volume_label``. Inputs follow the same
    pre-cleaned index and missing-value contract as :func:`get_reservoir_inflow`.
    """

    if not reservoir_storage.index.equals(reservoir_outflow.index):
        raise ValueError("storage and outflow indexes must match exactly")

    stream = OnlineReservoirInflow.from_config(
        config,
        max_window_steps=max_window_steps,
    )
    return _run_batch(reservoir_storage, reservoir_outflow, stream)


def _build_default_pipeline(
    *,
    q_storage: float,
    q_inflow: float,
    q_outflow: float,
    r_storage: float,
    r_outflow: float,
    smoothing_lag: timedelta,
    max_window_steps: int,
) -> OnlineInflowPipeline:
    """Build the standard reservoir model and streaming pipeline."""

    measurement_variances = np.asarray([r_storage, r_outflow], dtype=float)
    if not np.all(np.isfinite(measurement_variances)) or np.any(
        measurement_variances <= 0.0
    ):
        raise ValueError("measurement variances must be positive and finite")
    model = ReservoirStateSpaceModel(
        q_continuous=np.diag([q_storage, q_inflow, q_outflow]),
    )
    backend = ReservoirBackend(
        model=model,
        initial_covariance=np.diag([100.0, 1000.0, 1000.0]),
        observation_covariance=np.diag(measurement_variances),
    )
    return _build_pipeline(
        backend,
        smoothing_lag=smoothing_lag,
        max_window_steps=max_window_steps,
    )


def _build_configured_pipeline(
    config: ReservoirConfig,
    *,
    max_window_steps: int,
) -> OnlineInflowPipeline:
    """Build a streaming pipeline without discarding validated config fields."""

    return _build_pipeline(
        ReservoirBackend.from_config(config),
        smoothing_lag=config.smoothing_lag,
        max_window_steps=max_window_steps,
    )


def _build_pipeline(
    backend: ReservoirBackend,
    *,
    smoothing_lag: timedelta,
    max_window_steps: int,
) -> OnlineInflowPipeline:
    """Connect a reservoir backend to the standard fixed-lag smoother."""

    return OnlineInflowPipeline(
        backend=backend,
        smoother=OnlineFixedLagRTS(
            smoothing_lag,
            max_window_steps=max_window_steps,
        ),
        max_window_steps=max_window_steps,
    )


def _run_batch(
    reservoir_storage: pandas.Series,
    reservoir_outflow: pandas.Series,
    stream: OnlineReservoirInflow,
) -> pandas.DataFrame:
    """Process aligned series through a configured stream."""

    filtered_inflows: list[ReservoirFlowEstimate] = []
    estimated_outflows: list[ReservoirFlowEstimate] = []
    for timestamp, storage, discharge in zip(
        reservoir_storage.index,
        reservoir_storage.to_numpy(),
        reservoir_outflow.to_numpy(),
        strict=True,
    ):
        update = stream.process(
            timestamp=timestamp,
            storage=storage,
            discharge=discharge,
        )
        filtered_inflows.extend(update.filtered_inflows)
        estimated_outflows.extend(update.estimated_outflows)

    return _estimates_to_frame(
        reservoir_storage.index,
        filtered_inflows=filtered_inflows,
        estimated_outflows=estimated_outflows,
    )


def _estimates_to_frame(
    index: pandas.DatetimeIndex,
    *,
    filtered_inflows: list[ReservoirFlowEstimate],
    estimated_outflows: list[ReservoirFlowEstimate],
) -> pandas.DataFrame:
    """Place timestamped flow estimates into a result DataFrame."""

    return pandas.DataFrame(
        {
            "estimated_inflow": _estimates_to_array(index, filtered_inflows),
            "estimated_outflow": _estimates_to_array(index, estimated_outflows),
            "estimated_inflow_flag": _estimate_flags_to_array(
                index,
                filtered_inflows,
                field="prediction_flag",
            ),
            "estimated_outflow_flag": _estimate_flags_to_array(
                index,
                estimated_outflows,
                field="prediction_flag",
            ),
            "estimated_inflow_smoothing_flag": _estimate_flags_to_array(
                index,
                filtered_inflows,
                field="smoothing_flag",
                default=OutputFlag.NON_SMOOTHED,
            ),
            "estimated_outflow_smoothing_flag": _estimate_flags_to_array(
                index,
                estimated_outflows,
                field="smoothing_flag",
                default=OutputFlag.NON_SMOOTHED,
            ),
        },
        index=index,
    )


def _estimates_to_array(
    index: pandas.DatetimeIndex,
    estimates: list[ReservoirFlowEstimate],
) -> np.ndarray:
    """Map timestamped estimates to a dense array with one indexed lookup."""

    values = np.full(len(index), np.nan, dtype=float)
    if not estimates:
        return values
    try:
        timestamps = pandas.DatetimeIndex(
            [pandas.Timestamp(estimate.timestamp) for estimate in estimates]
        )
        estimate_values = np.asarray(
            [float(estimate.value) for estimate in estimates],
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise TypeError(
            "reservoir flow estimates must expose timestamp and numeric value"
        ) from error
    positions = index.get_indexer(timestamps)
    if np.any(positions < 0):
        raise ValueError("pipeline returned a state outside the input index")
    values[positions] = estimate_values
    return values


def _estimate_flags_to_array(
    index: pandas.DatetimeIndex,
    estimates: list[ReservoirFlowEstimate],
    *,
    field: str,
    default: OutputFlag | None = None,
) -> np.ndarray:
    """Map one estimate flag to a dense, indexed object array."""

    values = np.full(len(index), None, dtype=object)
    if default is not None:
        values[:] = default.value
    if not estimates:
        return values
    timestamps = pandas.DatetimeIndex(
        [pandas.Timestamp(estimate.timestamp) for estimate in estimates]
    )
    positions = index.get_indexer(timestamps)
    if np.any(positions < 0):
        raise ValueError("pipeline returned a state outside the input index")
    values[positions] = [getattr(estimate, field).value for estimate in estimates]
    return values
