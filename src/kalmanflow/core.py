"""Public adapters for reservoir-inflow estimation."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas

from ._reservoir_checkpoint import (
    decode_reservoir_checkpoint,
    encode_reservoir_checkpoint,
)
from ._validation import bounded_integer, measurement_value
from .flags import OutputFlag
from .kalman import kalman_filter
from .models import ReservoirStateSpaceModel
from .pipeline import ObservationLike, OnlineInflowPipeline
from .reservoir_backend import ReservoirBackend
from .reservoir_config import ReservoirConfig
from .rts import OnlineFixedLagRTS, _rts_smooth_arrays
from .time_utils import to_utc, validate_timestamp_precision

_UNSET: Final = object()


def _validate_include_uncertainty(value: object) -> bool:
    """Validate the opt-in uncertainty output switch."""

    if not isinstance(value, bool):
        raise TypeError("include_uncertainty must be a bool")
    return value


def _validated_interval_level(level: object) -> float:
    """Return a finite central interval level strictly inside (0, 1)."""

    try:
        value = float(level)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "level must be finite and strictly between 0 and 1"
        ) from error
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("level must be finite and strictly between 0 and 1")
    return value


def _normal_interval_multiplier(level: object) -> float:
    """Return the two-sided normal multiplier for a central interval."""

    validated_level = _validated_interval_level(level)
    # Compute from the lower tail so a high, valid level does not round its
    # upper-tail probability to exactly one before NormalDist sees it.
    multiplier = -NormalDist().inv_cdf((1.0 - validated_level) / 2.0)
    if not math.isfinite(multiplier):
        raise ValueError("level must produce a finite normal interval")
    return multiplier


def _inflow_standard_deviation(
    covariance: object,
    *,
    name: str,
) -> float:
    """Extract and validate the inflow-rate standard deviation from covariance."""

    array = np.asarray(covariance, dtype=float)
    if array.ndim != 2 or array.shape[0] <= 1 or array.shape[1] <= 1:
        raise ValueError(f"{name} must contain an inflow covariance entry")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    variance = float(array[1, 1])
    tolerance = 1e-12 * max(1.0, abs(variance))
    if variance < -tolerance:
        raise ValueError(f"{name} has a materially negative inflow variance")
    return math.sqrt(max(variance, 0.0))


def _inflow_standard_deviations(
    covariances: object,
    *,
    name: str,
) -> np.ndarray:
    """Extract validated inflow standard deviations from covariance matrices."""

    array = np.asarray(covariances, dtype=float)
    if array.ndim != 3 or array.shape[1] <= 1 or array.shape[2] <= 1:
        raise ValueError(f"{name} must contain inflow covariance entries")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    variances = np.asarray(array[:, 1, 1], dtype=float)
    tolerance = 1e-12 * np.maximum(1.0, np.abs(variances))
    if np.any(variances < -tolerance):
        raise ValueError(f"{name} has a materially negative inflow variance")
    return np.sqrt(np.maximum(variances, 0.0))


@dataclass(frozen=True)
class ReservoirFlowEstimate:
    """One timestamped public flow estimate."""

    timestamp: datetime
    value: float
    prediction_flag: OutputFlag = OutputFlag.NORMAL
    smoothing_flag: OutputFlag = OutputFlag.NON_SMOOTHED
    standard_deviation: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime):
            raise TypeError("timestamp must be a datetime instance")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        validate_timestamp_precision(self.timestamp)
        prediction_flag = OutputFlag(self.prediction_flag)
        if prediction_flag not in {OutputFlag.NORMAL, OutputFlag.PREDICTED}:
            raise ValueError("prediction_flag must be NORMAL or PREDICTED")
        smoothing_flag = OutputFlag(self.smoothing_flag)
        if smoothing_flag not in {
            OutputFlag.SMOOTHED,
            OutputFlag.NON_SMOOTHED,
        }:
            raise ValueError("smoothing_flag must be SMOOTHED or NON_SMOOTHED")
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        standard_deviation = self.standard_deviation
        if standard_deviation is not None:
            standard_deviation = float(standard_deviation)
            if not np.isfinite(standard_deviation):
                raise ValueError("standard_deviation must be finite or None")
            if standard_deviation < 0.0:
                raise ValueError("standard_deviation must be non-negative")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "prediction_flag", prediction_flag)
        object.__setattr__(self, "smoothing_flag", smoothing_flag)
        object.__setattr__(self, "standard_deviation", standard_deviation)

    def uncertainty_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return a central normal interval for this estimate.

        Raises
        ------
        ValueError
            If ``level`` is not finite and strictly between zero and one, or
            if uncertainty was not requested for this estimate.
        """

        multiplier = _normal_interval_multiplier(level)
        if self.standard_deviation is None:
            raise ValueError(
                "uncertainty is unavailable; enable include_uncertainty"
            )
        spread = multiplier * self.standard_deviation
        return self.value - spread, self.value + spread

@dataclass(frozen=True)
class ReservoirFlowUpdate:
    """Causal inflows and their delayed, absolute revisions."""

    filtered_inflows: tuple[ReservoirFlowEstimate, ...]
    revised_inflows: tuple[ReservoirFlowEstimate, ...]


class OnlineReservoirInflow:
    """Reservoir-specific streaming with causal inflows and delayed revisions.

    Filtered inflow estimates are returned as soon as forward filter steps are
    available. Revised inflow estimates are returned only after the fixed-lag
    smoother finalizes their timestamps. A revision is the absolute smoothed
    value that replaces the causal inflow at the same timestamp. Storage,
    outflow, and provisional smoothed states remain internal.
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
        include_uncertainty: bool = False,
    ) -> None:
        """Create an estimator with the supplied noise settings.

        ``reservoir_id`` is optional. It is required only when creating a
        checkpoint; callers using resumable streams should prefer
        :meth:`from_config`.

        Set ``include_uncertainty=True`` to attach model-based inflow standard
        deviations to streaming estimates. The option does not affect
        checkpoint replay state.
        """

        backend = _build_default_backend(
            q_storage=q_storage,
            q_inflow=q_inflow,
            q_outflow=q_outflow,
            r_storage=r_storage,
            r_outflow=r_outflow,
        )
        self._pipeline = _build_pipeline(
            backend,
            smoothing_lag=smoothing_lag,
            max_window_steps=max_window_steps,
        )
        self._configuration_fingerprint = _configuration_fingerprint(
            backend,
            smoothing_lag,
        )
        self._reservoir_id = reservoir_id
        self._include_uncertainty = _validate_include_uncertainty(
            include_uncertainty
        )
        self._processing = False
        self._checkpoint_ready = False

    @classmethod
    def from_config(
        cls,
        config: ReservoirConfig,
        *,
        max_window_steps: int = 100_000,
        include_uncertainty: bool = False,
    ) -> OnlineReservoirInflow:
        """Create an estimator from a validated reservoir configuration."""

        instance = cls.__new__(cls)
        backend = _build_configured_backend(config)
        instance._pipeline = _build_pipeline(
            backend,
            smoothing_lag=config.smoothing_lag,
            max_window_steps=max_window_steps,
        )
        instance._configuration_fingerprint = _configuration_fingerprint(
            backend,
            config.smoothing_lag,
        )
        instance._reservoir_id = config.reservoir_id
        instance._include_uncertainty = _validate_include_uncertainty(
            include_uncertainty
        )
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
        include_uncertainty: bool = False,
    ) -> OnlineReservoirInflow:
        """Restore one reservoir stream from a checkpoint.

        The supplied configuration must match both the reservoir identifier
        and the model settings recorded by the checkpoint fingerprint.
        """

        decoded = decode_reservoir_checkpoint(checkpoint)
        if decoded.reservoir_id != config.reservoir_id:
            raise ValueError(
                "checkpoint reservoir_id does not match config.reservoir_id"
            )
        instance = cls.from_config(
            config,
            max_window_steps=max_window_steps,
            include_uncertainty=include_uncertainty,
        )
        if (
            decoded.configuration_fingerprint
            != instance._configuration_fingerprint
        ):
            raise ValueError("checkpoint configuration does not match config")
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
            self._configuration_fingerprint,
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

        previous_utc = (
            to_utc(self._pipeline.last_input_timestamp)
            if self._pipeline.last_input_timestamp is not None
            else None
        )
        for timestamp, _, _ in values:
            if not isinstance(timestamp, datetime):
                raise TypeError("timestamp must be a datetime instance")
            timestamp_utc = to_utc(timestamp)
            validate_timestamp_precision(timestamp)
            if previous_utc is not None and timestamp_utc <= previous_utc:
                raise ValueError("observation timestamps must be strictly increasing")
            previous_utc = timestamp_utc

    def _mark_preprocessing_failure(self) -> None:
        """Make a failed public process call ineligible for checkpointing."""

        if not self._processing:
            self._checkpoint_ready = False

    def _to_public_update(self, update) -> ReservoirFlowUpdate:
        """Create public flow outputs before the stream becomes checkpointable."""

        return ReservoirFlowUpdate(
            filtered_inflows=tuple(
                ReservoirFlowEstimate(
                    timestamp=step.timestamp,
                    value=float(step.filtered_mean[1]),
                    prediction_flag=step.prediction_flag,
                    smoothing_flag=OutputFlag.NON_SMOOTHED,
                    standard_deviation=(
                        _inflow_standard_deviation(
                            step.filtered_covariance,
                            name="filtered covariance",
                        )
                        if self._include_uncertainty
                        else None
                    ),
                )
                for step in update.filtered_states
            ),
            revised_inflows=tuple(
                ReservoirFlowEstimate(
                    timestamp=state.timestamp,
                    value=float(state.mean[1]),
                    prediction_flag=state.prediction_flag,
                    smoothing_flag=OutputFlag.SMOOTHED,
                    standard_deviation=(
                        _inflow_standard_deviation(
                            state.covariance,
                            name="smoothed covariance",
                        )
                        if self._include_uncertainty
                        else None
                    ),
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
    return _validated_process_values(timestamp, storage, discharge)


def _process_values_from_observation(
    observation: ObservationLike,
) -> tuple[datetime, float, float]:
    """Extract structural observation fields without changing their values."""

    try:
        timestamp = observation.timestamp
        storage = observation.storage
        discharge = observation.discharge
    except AttributeError as error:
        raise TypeError(
            "observation must provide timestamp, storage, and discharge"
        ) from error
    return _validated_process_values(timestamp, storage, discharge)


def _validated_process_values(
    timestamp: object,
    storage: object,
    discharge: object,
) -> tuple[datetime, float, float]:
    """Validate one structural observation before it can mutate a stream."""

    if not isinstance(timestamp, datetime):
        raise TypeError("timestamp must be a datetime instance")
    to_utc(timestamp)
    validate_timestamp_precision(timestamp)
    return (
        timestamp,
        measurement_value(storage, name="storage"),
        measurement_value(discharge, name="discharge"),
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
    include_uncertainty: bool = False,
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
    - returned ``revised_inflow``: cfs, finalized fixed-lag-smoothed
      replacements for the causal estimates at the same timestamps

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
    :param smoothing_lag: Time lag before delayed inflow revisions are finalized.
    :param max_window_steps: Maximum number of active states for the fixed-lag
        smoother.
    :param include_uncertainty: Append filtered and revised inflow standard
        deviations to the result when true. Disabled by default.
    Input series must be pre-cleaned by the caller: parsed, aligned, sorted, and
    deduplicated. Missing storage or discharge samples may remain as NaN; the
    pipeline applies the documented missing-value rules but does not otherwise
    clean the series.
    :return: DataFrame indexed like the inputs with estimate and provenance
        columns. ``estimated_inflow_flag`` and ``revised_inflow_flag`` are
        ``NORMAL`` or ``PREDICTED``; the latter is used for both single- and
        double-missing observations. The estimated-inflow smoothing flag is
        always ``NON_SMOOTHED``. Finalized revised inflows are ``SMOOTHED``;
        trailing revisions remain NaN and ``NON_SMOOTHED`` until their lag has
        elapsed. Outflow is used by the model but is not a public result.
    """
    if not reservoir_storage.index.equals(reservoir_outflow.index):
        raise ValueError("storage and outflow indexes must match exactly")
    include_uncertainty = _validate_include_uncertainty(include_uncertainty)

    backend = _build_default_backend(
        q_storage=q_storage,
        q_inflow=q_inflow,
        q_outflow=q_outflow,
        r_storage=r_storage,
        r_outflow=r_outflow,
    )
    return _run_batch(
        reservoir_storage,
        reservoir_outflow,
        backend=backend,
        smoothing_lag=smoothing_lag,
        max_window_steps=max_window_steps,
        include_uncertainty=include_uncertainty,
    )


def get_reservoir_inflow_from_config(
    reservoir_storage: pandas.Series,
    reservoir_outflow: pandas.Series,
    config: ReservoirConfig,
    *,
    max_window_steps: int = 100_000,
    include_uncertainty: bool = False,
) -> pandas.DataFrame:
    """Estimate reservoir flows using one validated configuration.

    The input and output flow rates use ``config.unit_system.flow_label``;
    storage uses ``config.unit_system.volume_label``. Inputs follow the same
    pre-cleaned index and missing-value contract as :func:`get_reservoir_inflow`.
    Set ``include_uncertainty=True`` to append model-based inflow standard
    deviations.
    """

    if not reservoir_storage.index.equals(reservoir_outflow.index):
        raise ValueError("storage and outflow indexes must match exactly")
    include_uncertainty = _validate_include_uncertainty(include_uncertainty)

    return _run_batch(
        reservoir_storage,
        reservoir_outflow,
        backend=_build_configured_backend(config),
        smoothing_lag=config.smoothing_lag,
        max_window_steps=max_window_steps,
        include_uncertainty=include_uncertainty,
    )


def add_inflow_uncertainty_intervals(
    result: pandas.DataFrame,
    level: float = 0.95,
) -> pandas.DataFrame:
    """Append central normal interval columns to an uncertain batch result.

    ``result`` must contain the standard-deviation columns produced by
    :func:`get_reservoir_inflow` or :func:`get_reservoir_inflow_from_config`
    with ``include_uncertainty=True``. The input frame is not modified.
    Missing estimates and their matching standard deviations remain missing.
    """

    if not isinstance(result, pandas.DataFrame):
        raise TypeError("result must be a pandas DataFrame")
    multiplier = _normal_interval_multiplier(level)
    required_columns = (
        "estimated_inflow",
        "revised_inflow",
        "estimated_inflow_standard_deviation",
        "revised_inflow_standard_deviation",
    )
    missing_columns = [
        column for column in required_columns if column not in result.columns
    ]
    if missing_columns:
        joined = ", ".join(missing_columns)
        raise ValueError(
            "result must include uncertainty columns; run with "
            f"include_uncertainty=True (missing: {joined})"
        )

    output = result.copy()
    for prefix in ("estimated_inflow", "revised_inflow"):
        values = output[prefix].to_numpy(dtype=float)
        standard_deviations = output[
            f"{prefix}_standard_deviation"
        ].to_numpy(dtype=float)
        _validate_batch_interval_inputs(
            values,
            standard_deviations,
            prefix=prefix,
        )
        spread = multiplier * standard_deviations
        output[f"{prefix}_lower"] = values - spread
        output[f"{prefix}_upper"] = values + spread
    return output


def _validate_batch_interval_inputs(
    values: np.ndarray,
    standard_deviations: np.ndarray,
    *,
    prefix: str,
) -> None:
    """Validate paired estimate and uncertainty arrays for interval output."""

    if values.shape != standard_deviations.shape:
        raise ValueError(f"{prefix} and uncertainty columns must have matching lengths")
    invalid_values = ~np.isfinite(values) & ~np.isnan(values)
    if np.any(invalid_values):
        raise ValueError(f"{prefix} must contain only finite values or NaN")
    invalid_deviations = ~np.isfinite(standard_deviations) & ~np.isnan(
        standard_deviations
    )
    if np.any(invalid_deviations):
        raise ValueError(
            f"{prefix}_standard_deviation must contain only finite values or NaN"
        )
    finite_values = np.isfinite(values)
    finite_deviations = np.isfinite(standard_deviations)
    if np.any(finite_values != finite_deviations):
        raise ValueError(
            f"{prefix} and {prefix}_standard_deviation must be missing together"
        )
    if np.any(standard_deviations[finite_deviations] < 0.0):
        raise ValueError(f"{prefix}_standard_deviation must be non-negative")


def _build_default_backend(
    *,
    q_storage: float,
    q_inflow: float,
    q_outflow: float,
    r_storage: float,
    r_outflow: float,
) -> ReservoirBackend:
    """Build the default reservoir backend for either public adapter."""

    return ReservoirBackend(
        model=ReservoirStateSpaceModel(
            q_continuous=np.diag([q_storage, q_inflow, q_outflow]),
        ),
        initial_covariance=np.diag([100.0, 1000.0, 1000.0]),
        observation_covariance=np.diag([r_storage, r_outflow]),
    )


def _build_configured_backend(config: ReservoirConfig) -> ReservoirBackend:
    """Build the validated reservoir backend for either public adapter."""

    return ReservoirBackend(
        model=ReservoirStateSpaceModel(
            q_continuous=config.q,
            unit_system=config.unit_system,
        ),
        initial_covariance=config.p0,
        observation_covariance=config.r,
    )


def _configuration_fingerprint(
    backend: ReservoirBackend,
    smoothing_lag: timedelta,
) -> bytes:
    """Hash the model settings that determine checkpoint replay semantics."""

    model = backend.model
    if not isinstance(model, ReservoirStateSpaceModel):
        raise TypeError("reservoir checkpointing requires ReservoirStateSpaceModel")
    # Bump this semantic version whenever identical numeric configuration
    # values would replay differently (for example, an initialization change).
    digest = hashlib.sha256(b"kalmanflow-reservoir-state-space-v2")
    for array in (
        model.q_continuous,
        backend.initial_covariance,
        backend.observation_covariance,
        model.observation_matrix,
    ):
        values = np.asarray(array, dtype=">f8")
        digest.update(np.asarray(values.shape, dtype=">u4").tobytes())
        digest.update(values.tobytes(order="C"))
    digest.update(
        np.asarray(
            [
                smoothing_lag.total_seconds(),
                model.unit_system.flow_to_volume_per_second,
            ],
            dtype=">f8",
        ).tobytes()
    )
    for label in (
        model.unit_system.volume_label,
        model.unit_system.flow_label,
    ):
        encoded = label.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.digest()


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
    )


def _run_batch(
    reservoir_storage: pandas.Series,
    reservoir_outflow: pandas.Series,
    *,
    backend: ReservoirBackend,
    smoothing_lag: timedelta,
    max_window_steps: int,
    include_uncertainty: bool,
) -> pandas.DataFrame:
    """Run the reservoir batch kernel without streaming output objects."""

    return _process_reservoir_batch_raw(
        reservoir_storage.index,
        reservoir_storage.to_numpy(dtype=float),
        reservoir_outflow.to_numpy(dtype=float),
        backend=backend,
        smoothing_lag=smoothing_lag,
        max_window_steps=max_window_steps,
        include_uncertainty=include_uncertainty,
    )


def _process_reservoir_batch_raw(
    index: pandas.DatetimeIndex,
    storage: np.ndarray,
    discharge: np.ndarray,
    *,
    backend: ReservoirBackend,
    smoothing_lag: timedelta,
    max_window_steps: int,
    include_uncertainty: bool,
) -> pandas.DataFrame:
    """Estimate one complete reservoir series into dense NumPy result columns.

    This deliberately bypasses :class:`OnlineReservoirInflow`: that public
    adapter constructs immutable updates for every source observation, whereas
    batch callers need only dense, index-aligned result columns.
    """

    if smoothing_lag.total_seconds() <= 0.0:
        raise ValueError("lag must be positive")
    max_window_steps = bounded_integer(
        max_window_steps,
        name="max_window_steps",
        minimum=2,
    )
    include_uncertainty = _validate_include_uncertainty(include_uncertainty)
    if len(index) != len(storage) or len(index) != len(discharge):
        raise ValueError("batch inputs must have matching lengths")
    if np.isinf(storage).any():
        raise ValueError("storage must contain only finite values or NaN")
    if np.isinf(discharge).any():
        raise ValueError("discharge must contain only finite values or NaN")

    _validate_batch_timestamps(index)
    result = _empty_batch_columns(
        len(index), include_uncertainty=include_uncertainty
    )
    first, second = _initialization_positions(storage, discharge)
    if second is None:
        return pandas.DataFrame(result, index=index)

    positions = np.concatenate(
        (np.array([first, second]), np.arange(second + 1, len(index)))
    )
    timestamps = index[positions]
    observations = np.column_stack((storage[positions], discharge[positions]))
    elapsed = np.asarray(
        [
            (timestamps[position] - timestamps[position - 1]).total_seconds()
            for position in range(1, len(timestamps))
        ],
        dtype=float,
    )
    transitions = np.asarray(
        [backend.model.transition_matrix(seconds) for seconds in elapsed], dtype=float
    )
    process_covariances = np.asarray(
        [backend.model.process_covariance(seconds) for seconds in elapsed], dtype=float
    )
    initial_mean = np.array(
        [
            storage[first],
            backend.model.initial_outflow(discharge[first]),
            backend.model.initial_outflow(discharge[first]),
        ]
    )
    filtered = kalman_filter(
        observations,
        initial_mean=initial_mean,
        initial_covariance=backend.initial_covariance,
        transition_matrix=transitions,
        process_covariance=process_covariances,
        observation_matrix=backend.model.observation_matrix,
        observation_covariance=backend.observation_covariance,
    )
    prediction_flags = np.where(
        np.isfinite(observations).all(axis=1), "NORMAL", "PREDICTED"
    )
    result["estimated_inflow"][positions] = filtered.filtered_means[:, 1]
    result["estimated_inflow_flag"][positions] = prediction_flags
    if include_uncertainty:
        result["estimated_inflow_standard_deviation"][positions] = (
            _inflow_standard_deviations(
                filtered.filtered_covariances,
                name="filtered covariance",
            )
        )
    _write_fixed_lag_inflow_revisions(
        result,
        positions=positions,
        timestamps=timestamps,
        filtered=filtered,
        prediction_flags=prediction_flags,
        smoothing_lag=smoothing_lag,
        max_window_steps=max_window_steps,
        include_uncertainty=include_uncertainty,
    )
    return pandas.DataFrame(result, index=index)


def _empty_batch_columns(
    length: int,
    *,
    include_uncertainty: bool = False,
) -> dict[str, np.ndarray]:
    """Create the public result layout with dense, preallocated columns."""

    columns = {
        "estimated_inflow": np.full(length, np.nan),
        "revised_inflow": np.full(length, np.nan),
        "estimated_inflow_flag": np.full(length, None, dtype=object),
        "revised_inflow_flag": np.full(length, None, dtype=object),
        "estimated_inflow_smoothing_flag": np.full(
            length, OutputFlag.NON_SMOOTHED.value, dtype=object
        ),
        "revised_inflow_smoothing_flag": np.full(
            length, OutputFlag.NON_SMOOTHED.value, dtype=object
        ),
    }
    if include_uncertainty:
        columns.update(
            {
                "estimated_inflow_standard_deviation": np.full(length, np.nan),
                "revised_inflow_standard_deviation": np.full(length, np.nan),
            }
        )
    return columns


def _initialization_positions(
    storage: np.ndarray,
    discharge: np.ndarray,
) -> tuple[int | None, int | None]:
    """Return the two input rows that start the reservoir filter."""

    valid_storage = np.flatnonzero(np.isfinite(storage))
    if not len(valid_storage):
        return None, None
    first = int(valid_storage[0])
    if not np.isfinite(discharge[first]):
        raise ValueError(
            "the first valid storage observation needs a finite discharge value"
        )
    if len(valid_storage) == 1:
        return first, None
    return first, int(valid_storage[1])


def _validate_batch_timestamps(index: pandas.DatetimeIndex) -> None:
    """Apply the streaming timestamp contract before batch computation."""

    previous_utc: datetime | None = None
    for timestamp in index:
        timestamp_utc = to_utc(timestamp)
        validate_timestamp_precision(timestamp)
        if previous_utc is not None and timestamp_utc <= previous_utc:
            raise ValueError("observation timestamps must be strictly increasing")
        previous_utc = timestamp_utc


def _write_fixed_lag_inflow_revisions(
    result: dict[str, np.ndarray],
    *,
    positions: np.ndarray,
    timestamps: pandas.DatetimeIndex,
    filtered,
    prediction_flags: np.ndarray,
    smoothing_lag: timedelta,
    max_window_steps: int,
    include_uncertainty: bool,
) -> None:
    """Write absolute fixed-lag RTS inflow replacements for finalized states."""

    active: deque[int] = deque()
    for current in range(len(timestamps)):
        eligible = 0
        for candidate in active:
            if timestamps[current] - timestamps[candidate] >= smoothing_lag:
                eligible += 1
            else:
                break
        if not eligible:
            if len(active) == max_window_steps:
                raise OverflowError(
                    "active RTS window reached max_window_steps before "
                    "a state finalized"
                )
            active.append(current)
            continue

        window = np.fromiter(active, dtype=int)
        window = np.append(window, current)
        smoothed_means, smoothed_covariances, _ = _rts_smooth_arrays(
            filtered.filtered_means[window],
            filtered.filtered_covariances[window],
            filtered.predicted_means[window],
            filtered.predicted_covariances[window],
            filtered.transition_matrices[window[1:] - 1],
        )
        finalized = window[:eligible]
        output_positions = positions[finalized]
        result["revised_inflow"][output_positions] = smoothed_means[:eligible, 1]
        if include_uncertainty:
            result["revised_inflow_standard_deviation"][output_positions] = (
                _inflow_standard_deviations(
                    smoothed_covariances[:eligible],
                    name="smoothed covariance",
                )
            )
        result["revised_inflow_flag"][output_positions] = prediction_flags[finalized]
        result["revised_inflow_smoothing_flag"][output_positions] = (
            OutputFlag.SMOOTHED.value
        )
        for _ in range(eligible):
            active.popleft()
        active.append(current)
