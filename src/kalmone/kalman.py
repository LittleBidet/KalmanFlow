"""Filtering helpers for models with noisy, time-ordered observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from ._validation import readonly_array
from .flags import OutputFlag

Array = np.ndarray


@dataclass(frozen=True)
class FilterStep:
    """The filter's estimate and supporting values at one timestamp."""

    timestamp: datetime
    filtered_mean: Array
    filtered_covariance: Array
    predicted_mean: Array
    predicted_covariance: Array
    transition_matrix: Array
    prediction_flag: OutputFlag = OutputFlag.NORMAL

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")

        prediction_flag = OutputFlag(self.prediction_flag)
        if prediction_flag not in {OutputFlag.NORMAL, OutputFlag.PREDICTED}:
            raise ValueError(
                "FilterStep prediction_flag must be NORMAL or PREDICTED"
            )

        filtered_mean = readonly_array(self.filtered_mean)
        predicted_mean = readonly_array(self.predicted_mean)
        if filtered_mean.ndim != 1 or predicted_mean.shape != filtered_mean.shape:
            raise ValueError(
                "filtered_mean and predicted_mean must have matching 1-D shapes"
            )
        state_size = filtered_mean.shape[0]
        filtered_covariance = readonly_array(self.filtered_covariance)
        predicted_covariance = readonly_array(self.predicted_covariance)
        transition_matrix = readonly_array(self.transition_matrix)
        expected_matrix_shape = (state_size, state_size)
        for name, value in (
            ("filtered_covariance", filtered_covariance),
            ("predicted_covariance", predicted_covariance),
            ("transition_matrix", transition_matrix),
        ):
            if value.shape != expected_matrix_shape:
                raise ValueError(
                    f"{name} must have shape {expected_matrix_shape}; got {value.shape}"
                )

        object.__setattr__(self, "filtered_mean", filtered_mean)
        object.__setattr__(self, "filtered_covariance", filtered_covariance)
        object.__setattr__(self, "predicted_mean", predicted_mean)
        object.__setattr__(self, "predicted_covariance", predicted_covariance)
        object.__setattr__(self, "transition_matrix", transition_matrix)
        object.__setattr__(self, "prediction_flag", prediction_flag)


@dataclass(frozen=True)
class KalmanFilterResult:
    """All estimates and diagnostics produced by a filtering pass."""

    filtered_means: np.ndarray
    filtered_covariances: np.ndarray
    predicted_means: np.ndarray
    predicted_covariances: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    update_mask: np.ndarray
    transition_matrices: np.ndarray
    log_likelihood: float


def kalman_filter(
    observations: np.ndarray,
    initial_mean: np.ndarray,
    initial_covariance: np.ndarray,
    transition_matrix: np.ndarray,
    process_covariance: np.ndarray,
    observation_matrix: np.ndarray,
    observation_covariance: np.ndarray,
    *,
    control_offsets: np.ndarray | None = None,
    control_matrix: np.ndarray | None = None,
    controls: np.ndarray | None = None,
) -> KalmanFilterResult:
    """Process a sequence of observations from start to finish.

    Missing observation values are ignored for that update. Matrices that
    describe changes over time may be shared by every step or supplied once
    per step.
    """

    y = _as_observation_array(observations)
    n_times, n_obs = y.shape
    if n_times == 0:
        raise ValueError("observations must contain at least one time step")

    x0 = np.asarray(initial_mean, dtype=float)
    p0 = np.asarray(initial_covariance, dtype=float)
    if x0.ndim != 1:
        raise ValueError("initial_mean must be one-dimensional")
    n_state = x0.shape[0]
    _require_shape(p0, (n_state, n_state), "initial_covariance")

    n_transitions = max(n_times - 1, 0)
    transitions = _transition_array(
        transition_matrix,
        n_transitions,
        (n_state, n_state),
        "transition_matrix",
    )
    process_covariances = _transition_array(
        process_covariance,
        n_transitions,
        (n_state, n_state),
        "process_covariance",
    )
    observation_matrices = _observation_array(
        observation_matrix,
        n_times,
        (n_obs, n_state),
        "observation_matrix",
    )
    observation_covariances = _observation_array(
        observation_covariance,
        n_times,
        (n_obs, n_obs),
        "observation_covariance",
    )
    offsets = _control_offset_array(control_offsets, n_transitions, n_state)
    control_matrices, control_values = _matrix_control_arrays(
        control_matrix,
        controls,
        n_transitions,
        n_state,
    )
    filtered_means = np.empty((n_times, n_state), dtype=float)
    filtered_covariances = np.empty((n_times, n_state, n_state), dtype=float)
    predicted_means = np.empty((n_times, n_state), dtype=float)
    predicted_covariances = np.empty((n_times, n_state, n_state), dtype=float)
    innovations = np.full((n_times, n_obs), np.nan, dtype=float)
    innovation_covariances = np.full((n_times, n_obs, n_obs), np.nan, dtype=float)
    update_mask = np.zeros(n_times, dtype=bool)
    transition_matrices = np.empty((n_transitions, n_state, n_state), dtype=float)

    predicted_means[0] = x0
    predicted_covariances[0] = _symmetrize(p0)
    log_likelihood = 0.0

    for k in range(n_times):
        if k > 0:
            step = k - 1
            f = _transition_at(transitions, step)
            q = _transition_at(process_covariances, step)
            transition_matrices[step] = f
            offset = _control_offset_at(offsets, step, n_state)
            offset = offset + _matrix_control_at(
                control_matrices, control_values, step, n_state
            )
            predicted_means[k], predicted_covariances[k] = predict_state(
                filtered_means[k - 1],
                filtered_covariances[k - 1],
                f,
                q,
                control_offset=offset,
            )

        h = _observation_at(observation_matrices, k)
        r = _observation_at(observation_covariances, k)

        (
            filtered_means[k],
            filtered_covariances[k],
            innovations[k],
            innovation_covariances[k],
            update_mask[k],
            likelihood_increment,
        ) = _update_prediction(predicted_means[k], predicted_covariances[k], y[k], h, r)
        log_likelihood += likelihood_increment

    return KalmanFilterResult(
        filtered_means=filtered_means,
        filtered_covariances=filtered_covariances,
        predicted_means=predicted_means,
        predicted_covariances=predicted_covariances,
        innovations=innovations,
        innovation_covariances=innovation_covariances,
        update_mask=update_mask,
        transition_matrices=transition_matrices,
        log_likelihood=float(log_likelihood),
    )


def predict_state(
    filtered_mean: Array,
    filtered_covariance: Array,
    transition_matrix: Array,
    process_covariance: Array,
    *,
    control_offset: Array | None = None,
) -> tuple[Array, Array]:
    """Project one filtered state forward by one time interval."""

    mean = np.asarray(filtered_mean, dtype=float)
    covariance = np.asarray(filtered_covariance, dtype=float)
    transition = np.asarray(transition_matrix, dtype=float)
    process = np.asarray(process_covariance, dtype=float)
    state_size = mean.shape[0]
    _require_shape(covariance, (state_size, state_size), "filtered_covariance")
    _require_shape(transition, (state_size, state_size), "transition_matrix")
    _require_shape(process, (state_size, state_size), "process_covariance")
    offset = (
        np.zeros(state_size, dtype=float)
        if control_offset is None
        else np.asarray(control_offset, dtype=float)
    )
    _require_shape(offset, (state_size,), "control_offset")
    predicted_mean = transition @ mean + offset
    predicted_covariance = _symmetrize(transition @ covariance @ transition.T + process)
    return predicted_mean, predicted_covariance


def _update_prediction(
    predicted_mean: Array,
    predicted_covariance: Array,
    observation: Array,
    observation_matrix: Array,
    observation_covariance: Array,
) -> tuple[Array, Array, Array, Array, bool, float]:
    """Use the available parts of one observation to update a prediction."""

    mean = np.asarray(predicted_mean, dtype=float)
    covariance = np.asarray(predicted_covariance, dtype=float)
    y = np.asarray(observation, dtype=float)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.ndim != 1:
        raise ValueError("observation must be one-dimensional")
    h = np.asarray(observation_matrix, dtype=float)
    r = np.asarray(observation_covariance, dtype=float)
    n_obs = y.shape[0]
    n_state = mean.shape[0]
    _require_shape(covariance, (n_state, n_state), "predicted_covariance")
    _require_shape(h, (n_obs, n_state), "observation_matrix")
    _require_shape(r, (n_obs, n_obs), "observation_covariance")

    innovations = np.full(n_obs, np.nan, dtype=float)
    innovation_covariances = np.full((n_obs, n_obs), np.nan, dtype=float)
    obs_mask = np.isfinite(y)
    if not np.any(obs_mask):
        return (
            mean.copy(),
            covariance.copy(),
            innovations,
            innovation_covariances,
            False,
            0.0,
        )

    y_obs = y[obs_mask]
    h_obs = h[obs_mask]
    r_obs = r[np.ix_(obs_mask, obs_mask)]
    innovation = y_obs - h_obs @ mean
    innovation_covariance = _symmetrize(h_obs @ covariance @ h_obs.T + r_obs)
    ph_t = covariance @ h_obs.T
    kalman_gain = _solve_right(innovation_covariance, ph_t)

    filtered_mean = mean + kalman_gain @ innovation
    identity = np.eye(n_state)
    update_matrix = identity - kalman_gain @ h_obs
    filtered_covariance = _symmetrize(
        update_matrix @ covariance @ update_matrix.T
        + kalman_gain @ r_obs @ kalman_gain.T
    )
    innovations[obs_mask] = innovation
    innovation_covariances[np.ix_(obs_mask, obs_mask)] = innovation_covariance
    log_likelihood = _logpdf_zero_mean(innovation, innovation_covariance)
    return (
        filtered_mean,
        filtered_covariance,
        innovations,
        innovation_covariances,
        True,
        log_likelihood,
    )


def initial_filter_step(
    *,
    timestamp: datetime,
    initial_mean: Array,
    initial_covariance: Array,
    observation: Array,
    observation_matrix: Array,
    observation_covariance: Array,
) -> FilterStep:
    """Create the first filter record from an initial state and observation."""

    predicted_mean = np.asarray(initial_mean, dtype=float).copy()
    predicted_covariance = _symmetrize(np.asarray(initial_covariance, dtype=float))
    filtered_mean, filtered_covariance, *_ = _update_prediction(
        predicted_mean,
        predicted_covariance,
        observation,
        observation_matrix,
        observation_covariance,
    )
    return FilterStep(
        timestamp=timestamp,
        filtered_mean=filtered_mean,
        filtered_covariance=filtered_covariance,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        transition_matrix=np.eye(predicted_mean.shape[0]),
        prediction_flag=_prediction_flag(observation),
    )


def kalman_step(
    *,
    timestamp: datetime,
    previous_filtered_mean: Array,
    previous_filtered_covariance: Array,
    transition_matrix: Array,
    process_covariance: Array,
    observation: Array,
    observation_matrix: Array,
    observation_covariance: Array,
    control_offset: Array | None = None,
) -> FilterStep:
    """Predict and update one timestamp, returning an immutable filter record."""

    predicted_mean, predicted_covariance = predict_state(
        previous_filtered_mean,
        previous_filtered_covariance,
        transition_matrix,
        process_covariance,
        control_offset=control_offset,
    )
    filtered_mean, filtered_covariance, *_ = _update_prediction(
        predicted_mean,
        predicted_covariance,
        observation,
        observation_matrix,
        observation_covariance,
    )
    return FilterStep(
        timestamp=timestamp,
        filtered_mean=filtered_mean,
        filtered_covariance=filtered_covariance,
        predicted_mean=predicted_mean,
        predicted_covariance=predicted_covariance,
        transition_matrix=transition_matrix,
        prediction_flag=_prediction_flag(observation),
    )


# Helper functions


def _prediction_flag(observation: Array) -> OutputFlag:
    """Return the prediction flag for one possibly partial observation."""

    values = np.asarray(observation, dtype=float)
    return (
        OutputFlag.NORMAL
        if np.all(np.isfinite(values))
        else OutputFlag.PREDICTED
    )


def _as_observation_array(observations: np.ndarray) -> np.ndarray:
    """Normalize one- or two-dimensional observations to two dimensions."""

    y = np.asarray(observations, dtype=float)
    if y.ndim == 1:
        return y[:, None]
    if y.ndim == 2:
        return y
    raise ValueError("observations must be one- or two-dimensional")


def _transition_array(
    value: np.ndarray,
    n_transitions: int,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """Normalize and validate a shared or per-transition matrix."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        _require_shape(array, expected_shape, name)
        return array
    per_transition_shape = (n_transitions, *expected_shape)
    if array.shape == per_transition_shape:
        return array
    raise ValueError(
        f"{name} must have shape {expected_shape}, "
        f"or {per_transition_shape}"
    )


def _transition_at(value: np.ndarray, step: int) -> np.ndarray:
    """Return the shared or selected per-transition matrix."""

    return value if value.ndim == 2 else value[step]


def _observation_array(
    value: np.ndarray,
    n_times: int,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """Normalize and validate a shared or per-observation matrix."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 2:
        _require_shape(array, expected_shape, name)
        return array
    per_observation_shape = (n_times, *expected_shape)
    if array.shape == per_observation_shape:
        return array
    raise ValueError(
        f"{name} must have shape {expected_shape} or "
        f"{per_observation_shape}"
    )


def _observation_at(value: np.ndarray, time_index: int) -> np.ndarray:
    """Return the shared or selected per-observation matrix."""

    return value if value.ndim == 2 else value[time_index]


def _control_offset_array(
    control_offsets: np.ndarray | None,
    n_transitions: int,
    n_state: int,
) -> np.ndarray | None:
    """Normalize and validate shared or per-transition control offsets."""

    if control_offsets is None:
        return None

    offsets = np.asarray(control_offsets, dtype=float)
    if offsets.ndim == 1:
        _require_shape(offsets, (n_state,), "control_offsets")
        return offsets
    per_transition_shape = (n_transitions, n_state)
    if offsets.shape == per_transition_shape:
        return offsets
    raise ValueError(
        "control_offsets must have shape "
        f"({n_state},) or {per_transition_shape}"
    )


def _control_offset_at(
    control_offsets: np.ndarray | None,
    step: int,
    n_state: int,
) -> np.ndarray:
    """Return the shared or selected control offset."""

    if control_offsets is None:
        return np.zeros(n_state, dtype=float)
    return control_offsets if control_offsets.ndim == 1 else control_offsets[step]


def _matrix_control_arrays(
    control_matrix: np.ndarray | None,
    controls: np.ndarray | None,
    n_transitions: int,
    n_state: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Normalize and validate matrix-control inputs."""

    if control_matrix is None and controls is None:
        return None, None
    if control_matrix is None or controls is None:
        raise ValueError("control_matrix and controls must be supplied together")

    u = np.asarray(controls, dtype=float)
    b = np.asarray(control_matrix, dtype=float)
    if b.ndim == 2 and b.shape[0] == n_state:
        n_control = b.shape[1]
    elif b.ndim == 3 and b.shape[:2] == (n_transitions, n_state):
        n_control = b.shape[2]
    else:
        raise ValueError(
            "control_matrix must have shape (n_state, n_control) or "
            "(n_transitions, n_state, n_control)"
        )

    if u.ndim == 1:
        if u.shape[0] == n_transitions and n_control == 1:
            u = u.reshape(n_transitions, 1)
        elif u.shape != (n_control,):
            raise ValueError(
                "controls must have shape (n_control,) or "
                "(n_transitions, n_control)"
            )
    elif u.shape != (n_transitions, n_control):
        raise ValueError(
            "controls must have shape (n_control,) or "
            "(n_transitions, n_control)"
        )

    return b, u


def _matrix_control_at(
    control_matrix: np.ndarray | None,
    controls: np.ndarray | None,
    step: int,
    n_state: int,
) -> np.ndarray:
    """Return the state change caused by a control input."""

    if control_matrix is None or controls is None:
        return np.zeros(n_state, dtype=float)
    matrix = control_matrix if control_matrix.ndim == 2 else control_matrix[step]
    control = controls if controls.ndim == 1 else controls[step]
    return matrix @ control


def _require_shape(
    array: np.ndarray, expected_shape: tuple[int, ...], name: str
) -> None:
    """Raise a helpful error when an array has the wrong shape."""

    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}; got {array.shape}")


def _solve_right(system_matrix: np.ndarray, right_factor: np.ndarray) -> np.ndarray:
    """Solve a right-side matrix equation without forming an inverse."""
    try:
        return np.linalg.solve(system_matrix.T, right_factor.T).T
    except np.linalg.LinAlgError:
        return right_factor @ np.linalg.pinv(system_matrix)


def _logpdf_zero_mean(value: np.ndarray, covariance: np.ndarray) -> float:
    """Return the log likelihood of a zero-centered multivariate value."""

    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        jitter = np.finfo(float).eps * max(1.0, np.trace(covariance))
        adjusted = covariance + jitter * np.eye(covariance.shape[0])
        sign, logdet = np.linalg.slogdet(adjusted)
        covariance = adjusted
    if sign <= 0:
        raise np.linalg.LinAlgError("innovation covariance is not positive definite")

    try:
        mahalanobis = float(value.T @ np.linalg.solve(covariance, value))
    except np.linalg.LinAlgError:
        mahalanobis = float(value.T @ np.linalg.pinv(covariance) @ value)

    dimension = value.shape[0]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + mahalanobis)


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Remove tiny numerical differences between a matrix and its transpose."""

    return 0.5 * (matrix + matrix.T)
