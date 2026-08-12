"""Public adapters for reservoir-inflow estimation."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final

import numpy as np
import pandas

from ._reservoir_checkpoint import (
    decode_reservoir_checkpoint,
    encode_reservoir_checkpoint,
)
from .flags import OutputFlag
from .models import ReservoirStateSpaceModel
from .pipeline import ObservationLike, OnlineInflowPipeline
from .reservoir_backend import ReservoirBackend
from .reservoir_config import ReservoirConfig
from .rts import OnlineFixedLagRTS
from .time_utils import to_utc

_UNSET: Final = object()


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
            raise ValueError("prediction_flag must be NORMAL or PREDICTED")
        smoothing_flag = OutputFlag(self.smoothing_flag)
        if smoothing_flag not in {
            OutputFlag.SMOOTHED,
            OutputFlag.NON_SMOOTHED,
        }:
            raise ValueError("smoothing_flag must be SMOOTHED or NON_SMOOTHED")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "prediction_flag", prediction_flag)
        object.__setattr__(self, "smoothing_flag", smoothing_flag)

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
        reservoir_id: str | None = None,
    ) -> None:
        """Create an estimator with the supplied noise settings.

        ``reservoir_id`` is optional. It is required only when creating a
        checkpoint; callers using resumable streams should prefer
        :meth:`from_config`.
        """

        self._pipeline = _build_default_pipeline(
            q_storage=q_storage,
            q_inflow=q_inflow,
            q_outflow=q_outflow,
            r_storage=r_storage,
            r_outflow=r_outflow,
            smoothing_lag=smoothing_lag,
            max_window_steps=max_window_steps,
        )
        self._reservoir_id = reservoir_id
        self._processing = False
        self._checkpoint_ready = False

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
        instance._reservoir_id = config.reservoir_id
        instance._processing = False
        instance._checkpoint_ready = False
        return instance

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: bytes | bytearray | memoryview,
        *,
        config: ReservoirConfig,
        max_window_steps: int = 100_000,
    ) -> OnlineReservoirInflow:
        """Restore one reservoir stream from a checkpoint.

        Checkpoints deliberately contain no configuration fingerprint. The
        supplied configuration must therefore be compatible with the stream;
        only its reservoir identifier is checked here.
        """

        decoded = decode_reservoir_checkpoint(checkpoint)
        if decoded.reservoir_id != config.reservoir_id:
            raise ValueError(
                "checkpoint reservoir_id does not match config.reservoir_id"
            )
        instance = cls.from_config(config, max_window_steps=max_window_steps)
        instance._pipeline.restore_state(decoded.pipeline_state)
        return instance

    @property
    def initialized(self) -> bool:
        """Whether the first two valid storage observations were processed."""

        return self._pipeline.initialized

    @property
    def pending_count(self) -> int:
        """Number of states still waiting for the smoothing lag."""

        return self._pipeline.pending_count

    @property
    def reservoir_id(self) -> str | None:
        """Identifier bound to this stream, when it is checkpoint-capable."""

        return self._reservoir_id

    def process(
        self,
        observation: ObservationLike | None = None,
        *,
        timestamp: datetime | object = _UNSET,
        storage: float | object = _UNSET,
        discharge: float | object = _UNSET,
    ) -> ReservoirFlowUpdate:
        """Process one observation and return currently available flow outputs.

        An observation object may be supplied positionally, or the legacy
        keyword ``timestamp``, ``storage``, and ``discharge`` arguments may be
        used. Checkpoints become available only after this update is fully
        constructed.
        """

        try:
            values = _process_values_from_arguments(
                observation,
                timestamp=timestamp,
                storage=storage,
                discharge=discharge,
            )
        except Exception:
            self._mark_preprocessing_failure()
            raise
        return self._process_one_value(values)

    def process_many(
        self, observations: Iterable[ObservationLike]
    ) -> tuple[ReservoirFlowUpdate, ...]:
        """Atomically process an ordered group of observations.

        The returned updates are equivalent to calling :meth:`process` for
        each input in order. If any input fails, the stream is restored to its
        entry state and no partial group result is returned.
        """

        try:
            values = tuple(
                _process_values_from_observation(item) for item in observations
            )
        except Exception:
            self._mark_preprocessing_failure()
            raise
        if not values:
            return ()
        return self._process_many_values(values)

    def checkpoint(self) -> bytes:
        """Return a compact checkpoint after a fully completed process call."""

        if self._processing:
            raise RuntimeError("cannot create a checkpoint while processing")
        if not self._checkpoint_ready:
            raise RuntimeError(
                "cannot create a checkpoint before a successful process result"
            )
        if self._reservoir_id is None:
            raise RuntimeError(
                "checkpointing requires a reservoir_id; "
                "use OnlineReservoirInflow.from_config"
            )
        return encode_reservoir_checkpoint(
            self._reservoir_id,
            self._pipeline.export_state(),
        )

    def _process_many_values(
        self,
        values: tuple[tuple[datetime, float, float], ...],
    ) -> tuple[ReservoirFlowUpdate, ...]:
        """Process pre-extracted inputs transactionally and construct outputs."""

        if self._processing:
            raise RuntimeError("process calls may not be re-entered")
        self._processing = True
        self._checkpoint_ready = False
        entry_state = None
        try:
            entry_state = self._pipeline.export_state()
            self._validate_complete_order(values)
            updates = tuple(
                self._to_public_update(
                    self._pipeline.process(
                        timestamp=timestamp,
                        storage=storage,
                        discharge=discharge,
                    )
                )
                for timestamp, storage, discharge in values
            )
        except Exception:
            if entry_state is not None:
                try:
                    self._pipeline.restore_state(entry_state)
                except Exception as rollback_error:
                    raise RuntimeError(
                        "processing failed and the stream could not be restored"
                    ) from rollback_error
            raise
        else:
            self._checkpoint_ready = True
            return updates
        finally:
            self._processing = False

    def _process_one_value(
        self,
        value: tuple[datetime, float, float],
    ) -> ReservoirFlowUpdate:
        """Process one observation without copying the active replay window."""

        if self._processing:
            raise RuntimeError("process calls may not be re-entered")
        self._processing = True
        self._checkpoint_ready = False
        try:
            timestamp, storage, discharge = value
            update = self._to_public_update(
                self._pipeline.process(
                    timestamp=timestamp,
                    storage=storage,
                    discharge=discharge,
                )
            )
            self._checkpoint_ready = True
            return update
        finally:
            self._processing = False

    def _validate_complete_order(
        self,
        values: tuple[tuple[datetime, float, float], ...],
    ) -> None:
        """Validate every timestamp before a multi-observation mutation begins."""

        previous = self._pipeline.last_input_timestamp
        for timestamp, _, _ in values:
            if not isinstance(timestamp, datetime):
                raise TypeError("timestamp must be a datetime instance")
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("timestamp must be timezone-aware")
            if previous is not None and to_utc(timestamp) <= to_utc(previous):
                raise ValueError("observation timestamps must be strictly increasing")
            previous = timestamp

    def _mark_preprocessing_failure(self) -> None:
        """Make a failed public process call ineligible for checkpointing."""

        if not self._processing:
            self._checkpoint_ready = False

    @staticmethod
    def _to_public_update(update) -> ReservoirFlowUpdate:
        """Create public flow outputs before the stream becomes checkpointable."""

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


def _process_values_from_arguments(
    observation: ObservationLike | None,
    *,
    timestamp: datetime | object,
    storage: float | object,
    discharge: float | object,
) -> tuple[datetime, float, float]:
    """Normalize the object and legacy-keyword forms of ``process``."""

    if observation is not None:
        if any(value is not _UNSET for value in (timestamp, storage, discharge)):
            raise TypeError(
                "pass either an observation object or timestamp, storage, and discharge"
            )
        return _process_values_from_observation(observation)
    if any(value is _UNSET for value in (timestamp, storage, discharge)):
        raise TypeError("process requires timestamp, storage, and discharge")
    return timestamp, storage, discharge  # type: ignore[return-value]


def _process_values_from_observation(
    observation: ObservationLike,
) -> tuple[datetime, float, float]:
    """Extract structural observation fields without changing their values."""

    return observation.timestamp, observation.storage, observation.discharge


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

    model = ReservoirStateSpaceModel(
        q_continuous=np.diag([q_storage, q_inflow, q_outflow]),
    )
    backend = ReservoirBackend(
        model=model,
        initial_covariance=np.diag([100.0, 1000.0, 1000.0]),
        observation_covariance=np.diag([r_storage, r_outflow]),
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
        ReservoirBackend(
            model=ReservoirStateSpaceModel(
                q_continuous=config.q,
                unit_system=config.unit_system,
            ),
            initial_covariance=config.p0,
            observation_covariance=config.r,
        ),
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
    timestamps = pandas.DatetimeIndex(
        [pandas.Timestamp(estimate.timestamp) for estimate in estimates]
    )
    estimate_values = np.asarray(
        [estimate.value for estimate in estimates],
        dtype=float,
    )
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
