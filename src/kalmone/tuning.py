"""Fast, dataframe-first tuning for reservoir inflow models.

This module deliberately owns its preparation, filtering, scoring, and search
steps.  It accepts only storage and outflow observations; the water-balance
inflow calculated during preparation is a robust starting-value aid, never a
target or a measured observation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import blake2b
from math import ceil
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd

from .kalman import KalmanFilterResult
from .reservoir_config import (
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
)
from .units import UnitSystem

_PARAMETER_NAMES = (
    "q_storage",
    "q_inflow",
    "q_outflow",
    "r_storage",
    "r_outflow",
)
_OBSERVATION_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
_MODEL_VERSION = "reservoir-inflow-v1"
_CONFIGURATION_VERSION = "dataframe-tuned-v1"
_PERSISTENCE_VERSION = "reservoir-configurations-v1"


class TuningError(ValueError):
    """Raised when one reservoir cannot be prepared or tuned."""


def _freeze(value: Any) -> Any:
    """Recursively make public metadata read-only."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        copied = np.asarray(value).copy()
        copied.setflags(write=False)
        return copied
    return value


@dataclass(frozen=True)
class TuningFailure:
    """A reservoir-specific failure retained by :func:`tune_reservoirs`."""

    reservoir_id: str
    error_type: str
    message: str


@dataclass(frozen=True)
class InflowModelTuningResult:
    """The selected noise settings and diagnostics for one reservoir."""

    parameters: Mapping[str, float]
    config: ReservoirConfig
    score: float
    evaluations: int
    diagnostics: Mapping[str, Any]
    model_output: pd.DataFrame

    def __post_init__(self) -> None:
        values = {name: float(self.parameters[name]) for name in _PARAMETER_NAMES}
        if not np.all(np.isfinite(tuple(values.values()))) or any(
            value <= 0.0 for value in values.values()
        ):
            raise ValueError("tuned parameters must be positive and finite")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.evaluations < 1:
            raise ValueError("evaluations must be positive")
        object.__setattr__(self, "parameters", MappingProxyType(values))
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))
        # Results must not expose a view of caller-owned input data.
        object.__setattr__(self, "model_output", self.model_output.copy(deep=True))


@dataclass(frozen=True)
class ReservoirTuningBatch:
    """Independent tuning outcomes for a group of reservoirs."""

    results: Mapping[str, InflowModelTuningResult]
    failed: Mapping[str, TuningFailure]
    random_state: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "failed", MappingProxyType(dict(self.failed)))
        object.__setattr__(self, "random_state", int(self.random_state))

    @property
    def successful(self) -> tuple[str, ...]:
        """IDs that completed successfully, in input order."""

        return tuple(self.results)

    def save(self, path: str | Path) -> Path:
        """Save just the reviewed reusable configurations to JSON."""

        return save_reservoir_configs(
            {
                reservoir_id: result.config
                for reservoir_id, result in self.results.items()
            },
            path,
        )


# A short alias keeps the single-reservoir result name easy to discover.
TuningResult = InflowModelTuningResult
BatchTuningResult = ReservoirTuningBatch


@dataclass(frozen=True)
class NoiseTuningData:
    """Legacy array container retained for the temporary ``tune_noise`` wrapper.

    New applications should pass a dataframe directly to
    :func:`tune_inflow_model`.
    """

    timestamps: tuple[datetime, ...]
    storage: np.ndarray
    discharge: np.ndarray

    def __post_init__(self) -> None:
        index = _validate_datetime_index(pd.DatetimeIndex(self.timestamps))
        if len(index) < 2:
            raise ValueError("tuning data needs at least two timestamps")
        storage = np.asarray(self.storage, dtype=float).copy()
        discharge = np.asarray(self.discharge, dtype=float).copy()
        expected_shape = (len(index),)
        if storage.shape != expected_shape:
            raise ValueError(f"storage must have shape {expected_shape}")
        if discharge.shape != expected_shape:
            raise ValueError(f"discharge must have shape {expected_shape}")
        storage.setflags(write=False)
        discharge.setflags(write=False)
        object.__setattr__(self, "timestamps", tuple(index.to_pydatetime()))
        object.__setattr__(self, "storage", storage)
        object.__setattr__(self, "discharge", discharge)


@dataclass(frozen=True)
class NoiseTuningResult:
    """Compatibility representation returned by :func:`tune_noise`."""

    objective: Literal["loglik"]
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


@dataclass(frozen=True)
class _PreparedData:
    """All reservoir-specific arrays reused by every candidate evaluation."""

    index: pd.DatetimeIndex
    storage: np.ndarray
    outflow: np.ndarray
    elapsed_seconds: np.ndarray
    transitions: np.ndarray
    storage_basis: np.ndarray
    inflow_basis: np.ndarray
    outflow_basis: np.ndarray
    initial_index: int
    initial_state: np.ndarray
    initial_covariance: np.ndarray
    score_mask: np.ndarray
    cadence_seconds: float
    raw_inflow: np.ndarray
    storage_count: int
    outflow_count: int


@dataclass(frozen=True)
class _FilterPass:
    score: float
    usable_observations: int
    log_likelihood: float
    filter_result: KalmanFilterResult | None = None


def tune_inflow_model(
    observations: pd.DataFrame,
    reservoir_id: str,
    reservoir_name: str | None = None,
    *,
    max_evaluations: int = 64,
    max_tuning_rows: int = 5_000,
    validation_fraction: float = 0.25,
    random_state: int = 42,
    storage_column: str = "storage",
    outflow_column: str = "outflow",
    unit_system: UnitSystem | None = None,
    initial_parameters: Mapping[str, float] | Sequence[float] | None = None,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
) -> InflowModelTuningResult:
    """Tune one reservoir from storage and outflow observations only.

    The score is the mean one-step predictive negative log-likelihood over a
    final contiguous validation fraction.  Candidate search is bounded in log
    space and never changes ``observations``.
    """

    reservoir_id = _nonempty_name(reservoir_id, "reservoir_id")
    name = (
        reservoir_id
        if reservoir_name is None
        else _nonempty_name(reservoir_name, "reservoir_name")
    )
    evaluation_limit = _positive_integer(max_evaluations, "max_evaluations")
    tuning_rows = _positive_integer(max_tuning_rows, "max_tuning_rows", minimum=4)
    fraction = _validation_fraction(validation_fraction)
    seed = _integer_seed(random_state)
    units = UnitSystem.us_customary() if unit_system is None else unit_system
    if not isinstance(units, UnitSystem):
        raise TypeError("unit_system must be a UnitSystem instance")

    index, storage, outflow = _extract_dataframe_inputs(
        observations,
        storage_column=storage_column,
        outflow_column=outflow_column,
    )
    # Dataframes are supplied pre-cleaned by the caller.  The tuner therefore
    # uses their index as-is rather than sorting, parsing, or repairing it.
    # The production output path remains the boundary that validates it.
    prepared = _prepare_arrays(
        index,
        storage,
        outflow,
        unit_system=units,
        validation_fraction=fraction,
    )
    initial = _initial_parameter_values(prepared)
    initial = _apply_initial_parameters(initial, initial_parameters)
    lower, upper = _parameter_bounds(initial, parameter_bounds)
    best_parameters, score, evaluations, search_diagnostics = _search_parameters(
        prepared,
        initial=initial,
        lower=lower,
        upper=upper,
        max_evaluations=evaluation_limit,
        max_tuning_rows=tuning_rows,
        validation_fraction=fraction,
        random_state=seed,
    )
    if not np.isfinite(score):
        raise TuningError("no finite candidate score was produced")

    values = dict(zip(_PARAMETER_NAMES, best_parameters, strict=True))
    diagnostics = {
        "tuner": "dataframe-first",
        "objective": "mean_one_step_predictive_negative_log_likelihood",
        "search_space": "bounded_logarithmic",
        "validation_fraction": fraction,
        "random_seed": seed,
        "tuning_timestamp": datetime.now(UTC).isoformat(),
        "observation_count": len(prepared.index),
        "storage_observation_count": prepared.storage_count,
        "outflow_observation_count": prepared.outflow_count,
        "missing_storage_count": len(prepared.index) - prepared.storage_count,
        "missing_outflow_count": len(prepared.index) - prepared.outflow_count,
        "median_interval_seconds": prepared.cadence_seconds,
        "data_start": prepared.index[0].isoformat(),
        "data_end": prepared.index[-1].isoformat(),
        "data_interval": {
            "start": prepared.index[0].isoformat(),
            "end": prepared.index[-1].isoformat(),
            "median_seconds": prepared.cadence_seconds,
        },
        "initial_parameters": dict(zip(_PARAMETER_NAMES, initial, strict=True)),
        "parameter_bounds": {
            name: (float(low), float(high))
            for name, low, high in zip(_PARAMETER_NAMES, lower, upper, strict=True)
        },
        "evaluations": evaluations,
        "final_score": score,
        **search_diagnostics,
    }
    config = _build_config(
        reservoir_id=reservoir_id,
        reservoir_name=name,
        parameters=values,
        prepared=prepared,
        unit_system=units,
        diagnostics=diagnostics,
    )
    output = _winner_output(index, storage, outflow, config)
    return InflowModelTuningResult(
        parameters=values,
        config=config,
        score=score,
        evaluations=evaluations,
        diagnostics=diagnostics,
        model_output=output,
    )


def tune_reservoirs(
    reservoir_data: Mapping[str, pd.DataFrame],
    *,
    reservoir_names: Mapping[str, str] | None = None,
    max_evaluations: int = 64,
    max_tuning_rows: int = 5_000,
    validation_fraction: float = 0.25,
    workers: int = 1,
    random_state: int = 42,
    storage_column: str = "storage",
    outflow_column: str = "outflow",
    unit_system: UnitSystem | None = None,
    initial_parameters: Mapping[str, float] | Sequence[float] | None = None,
    parameter_bounds: Mapping[str, tuple[float, float]] | None = None,
    fail_fast: bool = False,
) -> ReservoirTuningBatch:
    """Tune independent configurations for a mapping of reservoir dataframes.

    Parallelism is deliberately only between reservoirs.  A stable hash derives
    each reservoir seed, so sequential and parallel runs select the same
    candidates for a given input mapping and main seed.
    """

    if not isinstance(reservoir_data, Mapping):
        raise TypeError("reservoir_data must be a mapping of IDs to dataframes")
    worker_count = _positive_integer(workers, "workers")
    main_seed = _integer_seed(random_state)
    items = tuple(
        (str(reservoir_id), frame) for reservoir_id, frame in reservoir_data.items()
    )
    results: dict[str, InflowModelTuningResult] = {}
    failures: dict[str, TuningFailure] = {}

    def tune_one(reservoir_id: str, frame: pd.DataFrame) -> InflowModelTuningResult:
        name = (
            reservoir_names.get(reservoir_id) if reservoir_names is not None else None
        )
        return tune_inflow_model(
            frame,
            reservoir_id=reservoir_id,
            reservoir_name=name,
            max_evaluations=max_evaluations,
            max_tuning_rows=max_tuning_rows,
            validation_fraction=validation_fraction,
            random_state=_derived_seed(main_seed, reservoir_id),
            storage_column=storage_column,
            outflow_column=outflow_column,
            unit_system=unit_system,
            initial_parameters=initial_parameters,
            parameter_bounds=parameter_bounds,
        )

    if worker_count == 1:
        for reservoir_id, frame in items:
            try:
                results[reservoir_id] = tune_one(reservoir_id, frame)
            except Exception as error:
                if fail_fast:
                    raise
                failures[reservoir_id] = _failure(reservoir_id, error)
    else:
        futures: dict[Future[InflowModelTuningResult], str] = {}
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for reservoir_id, frame in items:
                futures[executor.submit(tune_one, reservoir_id, frame)] = reservoir_id
            for future in as_completed(futures):
                reservoir_id = futures[future]
                try:
                    results[reservoir_id] = future.result()
                except Exception as error:
                    if fail_fast:
                        for pending in futures:
                            pending.cancel()
                        raise
                    failures[reservoir_id] = _failure(reservoir_id, error)

    # Futures finish out of order; restore the deterministic caller order.
    ordered_results = {
        reservoir_id: results[reservoir_id]
        for reservoir_id, _ in items
        if reservoir_id in results
    }
    ordered_failures = {
        reservoir_id: failures[reservoir_id]
        for reservoir_id, _ in items
        if reservoir_id in failures
    }
    return ReservoirTuningBatch(
        results=ordered_results,
        failed=ordered_failures,
        random_state=main_seed,
    )


def save_reservoir_configs(
    configs: Mapping[str, ReservoirConfig], path: str | Path
) -> Path:
    """Persist reusable per-reservoir configurations without observations."""

    target = Path(path)
    payload = {
        "format_version": _PERSISTENCE_VERSION,
        "saved_at": datetime.now(UTC).isoformat(),
        "configs": {
            str(reservoir_id): _serialize_config(config)
            for reservoir_id, config in configs.items()
        },
    }
    with target.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return target


def load_reservoir_configs(path: str | Path) -> dict[str, ReservoirConfig]:
    """Load configurations created by :func:`save_reservoir_configs`."""

    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format_version") != _PERSISTENCE_VERSION
    ):
        raise ValueError("not a supported reservoir configuration file")
    values = payload.get("configs")
    if not isinstance(values, Mapping):
        raise ValueError("configuration file has no configs mapping")
    configs = {
        str(reservoir_id): _deserialize_config(value)
        for reservoir_id, value in values.items()
    }
    if any(
        reservoir_id != config.reservoir_id for reservoir_id, config in configs.items()
    ):
        raise ValueError("configuration file keys do not match reservoir IDs")
    return configs


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
    """Run one legacy candidate through the new standalone filter kernel."""

    parameters = np.asarray(
        [q_storage, q_inflow, q_outflow, r_storage, r_outflow], dtype=float
    )
    if not np.all(np.isfinite(parameters)) or np.any(parameters <= 0.0):
        raise ValueError("candidate noise parameters must be positive and finite")
    if not np.all(np.isfinite(data.storage[:2])):
        raise ValueError("tuning data needs two finite initial storage values")
    if not np.isfinite(data.discharge[0]):
        raise ValueError("the first tuning discharge value must be finite")
    prepared = _prepare_arrays(
        pd.DatetimeIndex(data.timestamps),
        data.storage,
        data.discharge,
        unit_system=config.unit_system,
        validation_fraction=1.0,
        initial_covariance=config.p0,
        initial_positions=(0, 1),
        burn_in_rows=1,
    )
    outcome = _filter_candidate(prepared, parameters, collect=True)
    assert outcome.filter_result is not None
    return outcome.filter_result


def tune_noise(
    config: ReservoirConfig,
    data: NoiseTuningData,
    *,
    objective: Literal["loglik"] = "loglik",
    initial: tuple[float, float, float, float, float] | None = None,
    method: str = "dataframe-first",
    maxiter: int = 64,
) -> NoiseTuningResult:
    """Compatibility wrapper for the retired array-oriented tuner.

    The former SciPy and RMSE paths are intentionally gone.  This wrapper uses
    the dataframe-first predictive-likelihood tuner without requiring SciPy.
    """

    if objective != "loglik":
        raise ValueError("objective must be 'loglik'")
    initial_values: Mapping[str, float] | None = None
    if initial is not None:
        candidate = np.asarray(initial, dtype=float)
        if (
            candidate.shape != (5,)
            or not np.all(np.isfinite(candidate))
            or np.any(candidate <= 0.0)
        ):
            raise ValueError("initial must contain five positive finite values")
        initial_values = dict(zip(_PARAMETER_NAMES, candidate, strict=True))
    frame = pd.DataFrame(
        {"storage": np.asarray(data.storage), "outflow": np.asarray(data.discharge)},
        index=pd.DatetimeIndex(data.timestamps),
    )
    result = tune_inflow_model(
        frame,
        reservoir_id=config.reservoir_id,
        reservoir_name=config.reservoir_name,
        max_evaluations=_positive_integer(maxiter, "maxiter"),
        max_tuning_rows=max(4, len(frame)),
        validation_fraction=1.0,
        random_state=0,
        unit_system=config.unit_system,
        initial_parameters=initial_values,
    )
    return NoiseTuningResult(
        objective="loglik",
        q_storage=result.parameters["q_storage"],
        q_inflow=result.parameters["q_inflow"],
        q_outflow=result.parameters["q_outflow"],
        r_storage=result.parameters["r_storage"],
        r_outflow=result.parameters["r_outflow"],
        objective_value=result.score,
        success=True,
        iterations=result.evaluations,
        method=method,
        message="completed by dataframe-first bounded search",
    )


def _extract_dataframe_inputs(
    observations: pd.DataFrame,
    *,
    storage_column: str,
    outflow_column: str,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations must be a pandas DataFrame")
    missing = [
        column
        for column in (storage_column, outflow_column)
        if column not in observations.columns
    ]
    if missing:
        raise ValueError(
            f"observations is missing required columns: {', '.join(missing)}"
        )
    # Tuning follows the package's pre-cleaned input contract.  Copy only the
    # model columns; do not parse, sort, coerce, or otherwise clean them.
    return (
        observations.index,
        observations[storage_column].to_numpy(dtype=float, na_value=np.nan, copy=True),
        observations[outflow_column].to_numpy(dtype=float, na_value=np.nan, copy=True),
    )


def _validate_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("observations must use a DatetimeIndex")
    if index.tz is None:
        raise ValueError("observations must use a timezone-aware DatetimeIndex")
    if len(index) < 2:
        raise ValueError("observations need at least two timestamps")
    if index.hasnans:
        raise ValueError("observation timestamps must not be missing")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("observation timestamps must be strictly increasing")
    return index.copy()


def _prepare_arrays(
    index: pd.DatetimeIndex,
    storage: np.ndarray,
    outflow: np.ndarray,
    *,
    unit_system: UnitSystem,
    validation_fraction: float,
    initial_covariance: np.ndarray | None = None,
    initial_positions: tuple[int, int] | None = None,
    burn_in_rows: int | None = None,
) -> _PreparedData:
    """Prepare immutable candidate-invariant arrays once per data resolution."""

    storage_values = np.asarray(storage, dtype=float).copy()
    outflow_values = np.asarray(outflow, dtype=float).copy()

    storage_positions = np.flatnonzero(np.isfinite(storage_values))
    outflow_positions = np.flatnonzero(np.isfinite(outflow_values))
    if len(storage_positions) < 2:
        raise TuningError("insufficient usable storage data: need two observations")
    if len(outflow_positions) < 1:
        raise TuningError("insufficient usable outflow data: need one observation")
    if initial_positions is None:
        first, second = (int(storage_positions[0]), int(storage_positions[1]))
    else:
        first, second = initial_positions

    timestamp_values = index.asi8.astype(np.int64)
    elapsed = np.diff(timestamp_values).astype(float) / 1_000_000_000.0
    cadence = float(np.median(elapsed))
    initial_elapsed = float(
        (timestamp_values[second] - timestamp_values[first]) / 1_000_000_000.0
    )
    initial_outflow = (
        float(outflow_values[first])
        if np.isfinite(outflow_values[first])
        else float(np.nanmedian(outflow_values))
    )
    conversion = unit_system.flow_to_volume_per_second
    initial_inflow = initial_outflow + (
        (storage_values[second] - storage_values[first])
        / (initial_elapsed * conversion)
    )
    raw_inflow = _water_balance_inflow(
        storage_values, outflow_values, elapsed, conversion
    )
    if initial_covariance is None:
        p0 = _initial_covariance(storage_values, outflow_values, raw_inflow, cadence)
    else:
        p0 = np.asarray(initial_covariance, dtype=float).copy()

    transitions = _transition_bases(elapsed, conversion)
    storage_basis, inflow_basis, outflow_basis = _process_covariance_bases(
        elapsed, conversion
    )
    if burn_in_rows is None:
        burn_in_rows = max(2, int(ceil(0.05 * len(index))))
    score_start = min(len(index) - 1, first + burn_in_rows)
    possible_rows = np.arange(len(index)) >= score_start
    available_rows = np.flatnonzero(possible_rows)
    selected_count = max(1, int(ceil(len(available_rows) * validation_fraction)))
    score_mask = np.zeros(len(index), dtype=bool)
    score_mask[available_rows[-selected_count:]] = True

    return _PreparedData(
        index=index,
        storage=storage_values,
        outflow=outflow_values,
        elapsed_seconds=elapsed,
        transitions=transitions,
        storage_basis=storage_basis,
        inflow_basis=inflow_basis,
        outflow_basis=outflow_basis,
        initial_index=first,
        initial_state=np.array(
            [storage_values[first], initial_inflow, initial_outflow], dtype=float
        ),
        initial_covariance=p0,
        score_mask=score_mask,
        cadence_seconds=cadence,
        raw_inflow=raw_inflow,
        storage_count=int(np.isfinite(storage_values).sum()),
        outflow_count=int(np.isfinite(outflow_values).sum()),
    )


def _transition_bases(elapsed: np.ndarray, conversion: float) -> np.ndarray:
    transition = np.zeros((len(elapsed), 3, 3), dtype=float)
    transition[:, 0, 0] = 1.0
    transition[:, 1, 1] = 1.0
    transition[:, 2, 2] = 1.0
    transition[:, 0, 1] = conversion * elapsed
    transition[:, 0, 2] = -conversion * elapsed
    return transition


def _process_covariance_bases(
    elapsed: np.ndarray, conversion: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Q bases for the three diagonal continuous-time densities."""

    storage = np.zeros((len(elapsed), 3, 3), dtype=float)
    inflow = np.zeros_like(storage)
    outflow = np.zeros_like(storage)
    storage[:, 0, 0] = elapsed

    coupling = conversion * elapsed**2 / 2.0
    integrated = conversion**2 * elapsed**3 / 3.0
    inflow[:, 0, 0] = integrated
    inflow[:, 0, 1] = coupling
    inflow[:, 1, 0] = coupling
    inflow[:, 1, 1] = elapsed
    outflow[:, 0, 0] = integrated
    outflow[:, 0, 2] = -coupling
    outflow[:, 2, 0] = -coupling
    outflow[:, 2, 2] = elapsed
    return storage, inflow, outflow


def _water_balance_inflow(
    storage: np.ndarray,
    outflow: np.ndarray,
    elapsed: np.ndarray,
    conversion: float,
) -> np.ndarray:
    change = np.diff(storage)
    result = np.full(len(elapsed), np.nan, dtype=float)
    valid = (
        np.isfinite(change)
        & np.isfinite(outflow[1:])
        & np.isfinite(elapsed)
        & (elapsed > 0.0)
    )
    result[valid] = outflow[1:][valid] + change[valid] / (elapsed[valid] * conversion)
    return result


def _initial_covariance(
    storage: np.ndarray,
    outflow: np.ndarray,
    raw_inflow: np.ndarray,
    cadence: float,
) -> np.ndarray:
    storage_variance = _robust_variance(storage)
    outflow_variance = _robust_variance(outflow)
    inflow_variance = _robust_variance(raw_inflow)
    return np.diag(
        [
            max(storage_variance * 4.0, 1e-12),
            max(inflow_variance * 4.0, cadence * 1e-12),
            max(outflow_variance * 4.0, 1e-12),
        ]
    )


def _initial_parameter_values(prepared: _PreparedData) -> np.ndarray:
    storage_difference_variance = _robust_variance(np.diff(prepared.storage))
    outflow_difference_variance = _robust_variance(np.diff(prepared.outflow))
    raw_inflow_difference_variance = _robust_variance(np.diff(prepared.raw_inflow))
    r_storage = max(storage_difference_variance / 2.0, 1e-12)
    r_outflow = max(outflow_difference_variance / 2.0, 1e-12)
    values = np.array(
        [
            storage_difference_variance / prepared.cadence_seconds,
            raw_inflow_difference_variance / prepared.cadence_seconds,
            outflow_difference_variance / prepared.cadence_seconds,
            r_storage,
            r_outflow,
        ],
        dtype=float,
    )
    return np.maximum(values, 1e-12)


def _robust_variance(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)[np.isfinite(values)]
    if len(finite) == 0:
        return 1e-12
    centre = float(np.median(finite))
    scale = 1.4826 * float(np.median(np.abs(finite - centre)))
    reference = max(float(np.median(np.abs(finite))), 1.0)
    return max(scale**2, (reference * 1e-6) ** 2, 1e-12)


def _apply_initial_parameters(
    automatic: np.ndarray,
    supplied: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray:
    if supplied is None:
        return automatic
    result = automatic.copy()
    if isinstance(supplied, Mapping):
        unknown = set(supplied).difference(_PARAMETER_NAMES)
        if unknown:
            raise ValueError(
                f"unknown initial parameters: {', '.join(sorted(unknown))}"
            )
        for position, name in enumerate(_PARAMETER_NAMES):
            if name in supplied:
                result[position] = float(supplied[name])
    else:
        result = np.asarray(supplied, dtype=float)
    if result.shape != (5,) or not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("initial_parameters must contain positive finite values")
    return result.astype(float, copy=True)


def _parameter_bounds(
    initial: np.ndarray,
    supplied: Mapping[str, tuple[float, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.maximum(initial * 1e-5, 1e-18)
    upper = np.maximum(initial * 1e5, lower * 10.0)
    if supplied is not None:
        unknown = set(supplied).difference(_PARAMETER_NAMES)
        if unknown:
            raise ValueError(f"unknown parameter bounds: {', '.join(sorted(unknown))}")
        for position, name in enumerate(_PARAMETER_NAMES):
            if name in supplied:
                try:
                    low, high = supplied[name]
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"bounds for {name} must be a (low, high) pair"
                    ) from error
                lower[position] = float(low)
                upper[position] = float(high)
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower <= 0.0)
        or np.any(lower >= upper)
    ):
        raise ValueError("parameter bounds must be finite positive (low, high) pairs")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial_parameters must fall within parameter_bounds")
    return lower, upper


def _search_parameters(
    prepared: _PreparedData,
    *,
    initial: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    max_evaluations: int,
    max_tuning_rows: int,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, float, int, dict[str, Any]]:
    """Run a bounded, staged log-space search under a strict evaluation cap."""

    rng = np.random.default_rng(random_state)
    log_lower, log_upper, log_initial = np.log(lower), np.log(upper), np.log(initial)
    required_rows = _required_prefix_rows(prepared)
    if required_rows > max_tuning_rows:
        raise TuningError(
            "max_tuning_rows is too small to include the required initial observations"
        )
    small_rows = min(
        len(prepared.index),
        max(required_rows, 4, min(max_tuning_rows, max_tuning_rows // 4)),
    )
    medium_rows = min(
        len(prepared.index),
        max(required_rows, small_rows, min(max_tuning_rows, max_tuning_rows * 3 // 4)),
    )
    reduced_rows = min(len(prepared.index), max_tuning_rows)
    # Progressive stages are contiguous prefixes.  This lets every stage
    # reuse the transitions and Q bases prepared once for the reservoir.
    small = _prepared_prefix(
        prepared, small_rows, validation_fraction=validation_fraction
    )
    medium = _prepared_prefix(
        prepared, medium_rows, validation_fraction=validation_fraction
    )
    reduced = _prepared_prefix(
        prepared, reduced_rows, validation_fraction=validation_fraction
    )

    evaluations = 0
    stage_counts: dict[str, int] = {}

    def evaluate(
        log_values: np.ndarray, data: _PreparedData
    ) -> tuple[np.ndarray, float]:
        nonlocal evaluations
        if evaluations >= max_evaluations:
            raise RuntimeError("evaluation budget exhausted")
        evaluations += 1
        values = np.exp(np.clip(log_values, log_lower, log_upper))
        score = _filter_candidate(data, values, collect=False).score
        return values, score

    if max_evaluations == 1:
        final_values, final_score = evaluate(log_initial, prepared)
        return (
            final_values,
            final_score,
            evaluations,
            {
                "stage_evaluations": {"full": evaluations},
                "stage_rows": {"full": len(prepared.index)},
            },
        )

    broad_budget = max(1, int(ceil(max_evaluations * 0.50)))
    broad: list[tuple[np.ndarray, float]] = []
    for candidate_index in range(broad_budget):
        if candidate_index == 0:
            candidate = log_initial
        else:
            candidate = np.clip(
                log_initial + rng.normal(0.0, 2.0, size=5), log_lower, log_upper
            )
        broad.append(evaluate(candidate, small))
    stage_counts["broad"] = len(broad)

    broad.sort(key=lambda item: item[1])
    remaining = max_evaluations - evaluations
    mid_budget = min(len(broad), int(max_evaluations * 0.20), remaining)
    mid: list[tuple[np.ndarray, float]] = []
    for values, _ in broad[:mid_budget]:
        mid.append(evaluate(np.log(values), medium))
    stage_counts["survivors"] = len(mid)

    pool = mid if mid else broad[:1]
    pool.sort(key=lambda item: item[1])
    remaining = max_evaluations - evaluations
    refine_budget = min(int(max_evaluations * 0.15), max(0, remaining - 1))
    refined: list[tuple[np.ndarray, float]] = []
    if refine_budget:
        leader = np.log(pool[0][0])
        for _ in range(refine_budget):
            candidate = np.clip(
                leader + rng.normal(0.0, 0.45, size=5), log_lower, log_upper
            )
            refined.append(evaluate(candidate, reduced))
    stage_counts["refined"] = len(refined)

    finalists = pool + refined
    finalists.sort(key=lambda item: item[1])
    remaining = max_evaluations - evaluations
    final_count = min(len(finalists), remaining)
    full: list[tuple[np.ndarray, float]] = []
    for values, _ in finalists[:final_count]:
        full.append(evaluate(np.log(values), prepared))
    stage_counts["full"] = len(full)
    candidates = full if full else finalists
    candidates.sort(key=lambda item: item[1])
    best_values, best_score = candidates[0]
    return (
        best_values,
        best_score,
        evaluations,
        {
            "stage_evaluations": stage_counts,
            "stage_rows": {
                "broad": len(small.index),
                "survivors": len(medium.index),
                "refined": len(reduced.index),
                "full": len(prepared.index),
            },
        },
    )


def _prepared_prefix(
    prepared: _PreparedData,
    maximum_rows: int,
    *,
    validation_fraction: float,
) -> _PreparedData:
    """Return a bounded prefix sharing the reservoir's prepared arrays."""

    if maximum_rows >= len(prepared.index):
        return prepared
    required_rows = _required_prefix_rows(prepared)
    if maximum_rows < required_rows:
        raise TuningError(
            "max_tuning_rows is too small to include the required initial observations"
        )
    count = maximum_rows
    burn_in_rows = max(2, int(ceil(0.05 * count)))
    score_start = min(count - 1, prepared.initial_index + burn_in_rows)
    available_rows = np.arange(score_start, count)
    selected_count = max(1, int(ceil(len(available_rows) * validation_fraction)))
    score_mask = np.zeros(count, dtype=bool)
    score_mask[available_rows[-selected_count:]] = True
    return _PreparedData(
        index=prepared.index[:count],
        storage=prepared.storage[:count],
        outflow=prepared.outflow[:count],
        elapsed_seconds=prepared.elapsed_seconds[: count - 1],
        transitions=prepared.transitions[: count - 1],
        storage_basis=prepared.storage_basis[: count - 1],
        inflow_basis=prepared.inflow_basis[: count - 1],
        outflow_basis=prepared.outflow_basis[: count - 1],
        initial_index=prepared.initial_index,
        initial_state=prepared.initial_state,
        initial_covariance=prepared.initial_covariance,
        score_mask=score_mask,
        cadence_seconds=prepared.cadence_seconds,
        raw_inflow=prepared.raw_inflow[: count - 1],
        storage_count=int(np.isfinite(prepared.storage[:count]).sum()),
        outflow_count=int(np.isfinite(prepared.outflow[:count]).sum()),
    )


def _required_prefix_rows(prepared: _PreparedData) -> int:
    """Return the earliest prefix that can initialize the filter."""

    return max(
        int(np.flatnonzero(np.isfinite(prepared.storage))[1]) + 1,
        int(np.flatnonzero(np.isfinite(prepared.outflow))[0]) + 1,
    )


def _filter_candidate(
    prepared: _PreparedData,
    parameters: np.ndarray,
    *,
    collect: bool,
) -> _FilterPass:
    """Run a candidate with only scalar score state unless full output is needed."""

    values = np.asarray(parameters, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        return _FilterPass(float("inf"), 0, float("-inf"))
    r = np.diag(values[3:])
    state = prepared.initial_state.copy()
    covariance = prepared.initial_covariance.copy()
    total_score_nll = 0.0
    total_likelihood_nll = 0.0
    usable = 0
    n = len(prepared.index)

    filtered_means = filtered_covariances = None
    predicted_means = predicted_covariances = None
    innovations = innovation_covariances = update_mask = None
    if collect:
        filtered_means = np.full((n, 3), np.nan, dtype=float)
        filtered_covariances = np.full((n, 3, 3), np.nan, dtype=float)
        predicted_means = np.full((n, 3), np.nan, dtype=float)
        predicted_covariances = np.full((n, 3, 3), np.nan, dtype=float)
        innovations = np.full((n, 2), np.nan, dtype=float)
        innovation_covariances = np.full((n, 2, 2), np.nan, dtype=float)
        update_mask = np.zeros(n, dtype=bool)

    try:
        for row in range(prepared.initial_index, n):
            if row == prepared.initial_index:
                predicted_state = state
                predicted_covariance = covariance
            else:
                interval = row - 1
                process = (
                    values[0] * prepared.storage_basis[interval]
                    + values[1] * prepared.inflow_basis[interval]
                    + values[2] * prepared.outflow_basis[interval]
                )
                predicted_state = prepared.transitions[interval] @ state
                predicted_covariance = 0.5 * (
                    prepared.transitions[interval]
                    @ covariance
                    @ prepared.transitions[interval].T
                    + process
                    + (
                        prepared.transitions[interval]
                        @ covariance
                        @ prepared.transitions[interval].T
                        + process
                    ).T
                )
            if not np.all(np.isfinite(predicted_state)) or not np.all(
                np.isfinite(predicted_covariance)
            ):
                return _FilterPass(float("inf"), usable, float("-inf"))

            observation = np.array([prepared.storage[row], prepared.outflow[row]])
            observed = np.isfinite(observation)
            state = predicted_state.copy()
            covariance = predicted_covariance.copy()
            if collect:
                assert predicted_means is not None and predicted_covariances is not None
                predicted_means[row] = predicted_state
                predicted_covariances[row] = predicted_covariance
            if np.any(observed):
                h = _OBSERVATION_MATRIX[observed]
                observed_r = r[np.ix_(observed, observed)]
                innovation = observation[observed] - h @ predicted_state
                innovation_covariance = 0.5 * (
                    h @ predicted_covariance @ h.T
                    + observed_r
                    + (h @ predicted_covariance @ h.T + observed_r).T
                )
                sign, logdet = np.linalg.slogdet(innovation_covariance)
                if sign <= 0.0 or not np.isfinite(logdet):
                    return _FilterPass(float("inf"), usable, float("-inf"))
                solved_innovation = np.linalg.solve(innovation_covariance, innovation)
                kalman_gain = np.linalg.solve(
                    innovation_covariance, (predicted_covariance @ h.T).T
                ).T
                nll = 0.5 * (
                    len(innovation) * np.log(2.0 * np.pi)
                    + logdet
                    + float(innovation @ solved_innovation)
                )
                if not np.isfinite(nll):
                    return _FilterPass(float("inf"), usable, float("-inf"))
                total_likelihood_nll += nll
                if prepared.score_mask[row]:
                    total_score_nll += nll
                    usable += len(innovation)
                identity = np.eye(3)
                update = identity - kalman_gain @ h
                covariance = (
                    update @ predicted_covariance @ update.T
                    + kalman_gain @ observed_r @ kalman_gain.T
                )
                covariance = 0.5 * (covariance + covariance.T)
                state = predicted_state + kalman_gain @ innovation
                if collect:
                    assert (
                        innovations is not None and innovation_covariances is not None
                    )
                    assert update_mask is not None
                    innovations[row, observed] = innovation
                    innovation_covariances[row][np.ix_(observed, observed)] = (
                        innovation_covariance
                    )
                    update_mask[row] = True
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(covariance)):
                return _FilterPass(float("inf"), usable, float("-inf"))
            if collect:
                assert filtered_means is not None and filtered_covariances is not None
                filtered_means[row] = state
                filtered_covariances[row] = covariance
    except FloatingPointError, np.linalg.LinAlgError, ValueError:
        return _FilterPass(float("inf"), usable, float("-inf"))

    score = float(total_score_nll / usable) if usable else float("inf")
    filter_result = None
    if collect:
        assert filtered_means is not None and filtered_covariances is not None
        assert predicted_means is not None and predicted_covariances is not None
        assert innovations is not None and innovation_covariances is not None
        assert update_mask is not None
        filter_result = KalmanFilterResult(
            filtered_means=filtered_means,
            filtered_covariances=filtered_covariances,
            predicted_means=predicted_means,
            predicted_covariances=predicted_covariances,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            update_mask=update_mask,
            transition_matrices=prepared.transitions.copy(),
            log_likelihood=-float(total_likelihood_nll),
        )
    return _FilterPass(score, usable, -float(total_likelihood_nll), filter_result)


def _build_config(
    *,
    reservoir_id: str,
    reservoir_name: str,
    parameters: Mapping[str, float],
    prepared: _PreparedData,
    unit_system: UnitSystem,
    diagnostics: Mapping[str, Any],
) -> ReservoirConfig:
    lag_seconds = max(1.0, prepared.cadence_seconds * 6.0)
    return ReservoirConfig(
        reservoir_id=reservoir_id,
        reservoir_name=reservoir_name,
        q=np.diag(
            [
                parameters["q_storage"],
                parameters["q_inflow"],
                parameters["q_outflow"],
            ]
        ),
        r=np.diag([parameters["r_storage"], parameters["r_outflow"]]),
        p0=prepared.initial_covariance,
        smoothing_lag=timedelta(seconds=lag_seconds),
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=(
            InflowUnits.CUBIC_FEET_PER_SECOND
            if unit_system.flow_label == "cfs"
            else InflowUnits.SYSTEM_FLOW_RATE
        ),
        model_version=_MODEL_VERSION,
        configuration_version=_CONFIGURATION_VERSION,
        tuning_metadata=diagnostics,
        unit_system=unit_system,
    )


def _winner_output(
    index: pd.DatetimeIndex,
    storage: np.ndarray,
    outflow: np.ndarray,
    config: ReservoirConfig,
) -> pd.DataFrame:
    """Generate full output exactly once, after the winning candidate is known."""

    from .core import get_reservoir_inflow_from_config

    return get_reservoir_inflow_from_config(
        pd.Series(storage, index=index, copy=True),
        pd.Series(outflow, index=index, copy=True),
        config,
    )


def _serialize_config(config: ReservoirConfig) -> dict[str, Any]:
    return {
        "reservoir_id": config.reservoir_id,
        "reservoir_name": config.reservoir_name,
        "parameters": {
            "q_storage": float(config.q[0, 0]),
            "q_inflow": float(config.q[1, 1]),
            "q_outflow": float(config.q[2, 2]),
            "r_storage": float(config.r[0, 0]),
            "r_outflow": float(config.r[1, 1]),
        },
        "q": config.q.tolist(),
        "r": config.r.tolist(),
        "initial_covariance": config.p0.tolist(),
        "smoothing_lag_seconds": config.smoothing_lag.total_seconds(),
        "initialization_strategy": config.initialization_strategy.value,
        "inflow_units": config.inflow_units.value,
        "unit_system": {
            "volume_label": config.unit_system.volume_label,
            "flow_label": config.unit_system.flow_label,
            "flow_to_volume_per_second": config.unit_system.flow_to_volume_per_second,
        },
        "model_version": config.model_version,
        "configuration_version": config.configuration_version,
        "tuning_metadata": _json_value(config.tuning_metadata),
        "tuning_timestamp": config.tuning_metadata.get("tuning_timestamp"),
    }


def _deserialize_config(value: Any) -> ReservoirConfig:
    if not isinstance(value, Mapping):
        raise ValueError("each saved configuration must be an object")
    try:
        unit_data = value["unit_system"]
        if not isinstance(unit_data, Mapping):
            raise TypeError
        units = UnitSystem(
            volume_label=str(unit_data["volume_label"]),
            flow_label=str(unit_data["flow_label"]),
            flow_to_volume_per_second=float(unit_data["flow_to_volume_per_second"]),
        )
        metadata = value.get("tuning_metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError
        return ReservoirConfig(
            reservoir_id=str(value["reservoir_id"]),
            reservoir_name=str(value["reservoir_name"]),
            q=np.asarray(value["q"], dtype=float),
            r=np.asarray(value["r"], dtype=float),
            p0=np.asarray(value["initial_covariance"], dtype=float),
            smoothing_lag=timedelta(seconds=float(value["smoothing_lag_seconds"])),
            initialization_strategy=InitializationStrategy(
                value["initialization_strategy"]
            ),
            inflow_units=InflowUnits(value["inflow_units"]),
            model_version=str(value["model_version"]),
            configuration_version=str(value["configuration_version"]),
            tuning_metadata=metadata,
            unit_system=units,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("saved configuration is invalid") from error


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _failure(reservoir_id: str, error: Exception) -> TuningFailure:
    message = str(error) or error.__class__.__name__
    return TuningFailure(
        reservoir_id=reservoir_id,
        error_type=error.__class__.__name__,
        message=message,
    )


def _nonempty_name(value: object, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def _positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be an integer") from error
    if result != value or result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _integer_seed(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError("random_state must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise TypeError("random_state must be an integer") from error
    if result != value:
        raise TypeError("random_state must be an integer")
    return result


def _derived_seed(main_seed: int, reservoir_id: str) -> int:
    material = f"{main_seed}:{reservoir_id}".encode()
    return int.from_bytes(blake2b(material, digest_size=8).digest(), "little")


def _validation_fraction(value: object) -> float:
    try:
        fraction = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("validation_fraction must be a number") from error
    if not np.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("validation_fraction must be greater than 0 and at most 1")
    return fraction


__all__ = [
    "BatchTuningResult",
    "InflowModelTuningResult",
    "NoiseTuningData",
    "NoiseTuningResult",
    "ReservoirTuningBatch",
    "TuningError",
    "TuningFailure",
    "TuningResult",
    "load_reservoir_configs",
    "run_filter_with_noise",
    "save_reservoir_configs",
    "tune_inflow_model",
    "tune_noise",
    "tune_reservoirs",
]
