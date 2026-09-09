from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..models import ReservoirStateSpaceModel
from ..reservoir_config import ReservoirConfig
from ..time_utils import validate_timestamp_precision
from ._types import (
    BayesianEvaluationSettings,
    BayesianTuningError,
    ValidationWindow,
    _DiagnosticPlan,
    _Prepared,
)

Array = np.ndarray


def _validate_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("storage and discharge indexes must be DatetimeIndex")
    if index.tz is None:
        raise ValueError("storage and discharge indexes must be timezone-aware")
    for timestamp in index:
        validate_timestamp_precision(timestamp)
    if (
        len(index) < 2
        or index.hasnans
        or not index.is_monotonic_increasing
        or index.has_duplicates
    ):
        raise ValueError(
            "timestamps must be timezone-aware, strictly increasing, and nonmissing"
        )
    return index


def _timestamp_seconds(index: pd.DatetimeIndex) -> np.ndarray:
    values = np.asarray(index.tz_convert("UTC").tz_localize(None))
    unit, multiplier = np.datetime_data(values.dtype)
    seconds_per_unit = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "m": 60.0,
        "h": 3600.0,
        "D": 86400.0,
    }
    try:
        scale = seconds_per_unit[unit]
    except KeyError as error:
        raise ValueError(f"unsupported timestamp resolution: {unit}") from error
    return values.astype(np.int64) * (float(multiplier) * scale)


def _validate_windows(
    windows: Sequence[ValidationWindow],
) -> tuple[ValidationWindow, ...]:
    result = tuple(
        window if isinstance(window, ValidationWindow) else ValidationWindow(*window)
        for window in windows
    )
    if not result:
        raise ValueError("at least one validation window is required")
    if len({window.name for window in result}) != len(result):
        raise ValueError("validation window names must be unique")
    ordered = sorted(result, key=lambda value: value.start)
    if any(
        left.end > right.start
        for left, right in zip(ordered, ordered[1:], strict=False)
    ):
        raise ValueError("validation windows must not overlap")
    return result


def _window_weights(windows: Sequence[ValidationWindow]) -> np.ndarray:
    values = np.asarray(
        [1.0 if window.weight is None else window.weight for window in windows],
        dtype=float,
    )
    return values / values.sum()


def _is_diagonal(value: Array) -> bool:
    array = np.asarray(value, dtype=float)
    return array.ndim == 2 and np.allclose(
        array, np.diag(np.diag(array)), rtol=0.0, atol=0.0
    )


def _process_covariance_bases(
    elapsed: Array, conversion: float
) -> tuple[Array, Array, Array]:
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


def _prepare(
    storage: pd.Series, discharge: pd.Series, config: ReservoirConfig
) -> _Prepared:
    if not isinstance(storage, pd.Series) or not isinstance(discharge, pd.Series):
        raise TypeError("storage and discharge must be pandas Series")
    if not storage.index.equals(discharge.index):
        raise ValueError("storage and discharge indexes must match exactly")
    index = _validate_index(storage.index)
    storage_values = storage.to_numpy(dtype=float, copy=True)
    discharge_values = discharge.to_numpy(dtype=float, copy=True)
    if np.isinf(storage_values).any():
        raise ValueError("storage must contain only finite values or NaN")
    if np.isinf(discharge_values).any():
        raise ValueError("discharge must contain only finite values or NaN")
    q = np.asarray(config.q, dtype=float)
    r = np.asarray(config.r, dtype=float)
    p0 = np.asarray(config.p0, dtype=float)
    if not _is_diagonal(q) or not _is_diagonal(r) or not _is_diagonal(p0):
        raise ValueError(
            "Bayesian evaluation requires constant diagonal q, r, and p0 matrices"
        )
    finite_storage = np.flatnonzero(np.isfinite(storage_values))
    if len(finite_storage) < 2:
        raise BayesianTuningError(
            "at least two finite storage observations are required"
        )
    first, second = int(finite_storage[0]), int(finite_storage[1])
    if not np.isfinite(discharge_values[first]):
        raise ValueError("the first finite storage observation needs finite discharge")
    positions = np.concatenate(
        (
            np.array([first, second], dtype=int),
            np.arange(second + 1, len(index), dtype=int),
        )
    )
    timestamps = index[positions]
    observations = np.column_stack(
        (storage_values[positions], discharge_values[positions])
    )
    elapsed = np.asarray(
        [
            (timestamps[i] - timestamps[i - 1]).total_seconds()
            for i in range(1, len(timestamps))
        ],
        dtype=float,
    )
    model = ReservoirStateSpaceModel(q_continuous=q, unit_system=config.unit_system)
    conversion = float(config.unit_system.flow_to_volume_per_second)
    transitions = np.asarray(
        [model.transition_matrix(value) for value in elapsed], dtype=float
    )
    storage_basis, inflow_basis, outflow_basis = _process_covariance_bases(
        elapsed, conversion
    )
    initial_mean = np.array(
        [
            storage_values[first],
            model.initial_outflow(discharge_values[first]),
            model.initial_outflow(discharge_values[first]),
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(initial_mean)):
        raise BayesianTuningError("initial state is not finite")
    return _Prepared(
        index=index,
        timestamps=timestamps,
        observations=observations,
        transitions=transitions,
        storage_basis=storage_basis,
        inflow_basis=inflow_basis,
        outflow_basis=outflow_basis,
        initial_mean=initial_mean,
        initial_covariance=p0.copy(),
        q_storage=float(q[0, 0]),
        q_outflow=float(q[2, 2]),
        r=r.copy(),
        model=model,
        finite_observations=np.isfinite(observations),
    )


def _diagnostic_plan(
    prepared: _Prepared,
    windows: Sequence[ValidationWindow],
    settings: BayesianEvaluationSettings,
) -> _DiagnosticPlan:
    timestamps = prepared.timestamps
    window_masks = tuple(window.mask(timestamps) for window in windows)
    score_mask = np.arange(len(timestamps), dtype=int) >= 2
    score_mask &= timestamps >= timestamps[1] + pd.Timedelta(
        seconds=settings.warmup.total_seconds()
    )
    return _DiagnosticPlan(window_masks=window_masks, score_mask=score_mask)
