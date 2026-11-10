"""Shared RTS recursion and bounded elapsed-time online fixed-lag smoothing."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from ._validation import bounded_integer, readonly_array
from .flags import OutputFlag
from .kalman import Array, FilterStep, _solve_right, _symmetrize
from .time_utils import aware_timestamp, elapsed_seconds, to_utc


@dataclass(frozen=True)
class SmoothedStep:
    """A state estimate and its uncertainty at one timestamp."""

    timestamp: datetime
    mean: Array
    covariance: Array
    prediction_flag: OutputFlag = OutputFlag.NORMAL
    smoothing_flag: OutputFlag = OutputFlag.SMOOTHED

    def __post_init__(self) -> None:
        aware_timestamp(self.timestamp)
        prediction_flag = OutputFlag(self.prediction_flag)
        if prediction_flag not in {OutputFlag.NORMAL, OutputFlag.PREDICTED}:
            raise ValueError("SmoothedStep prediction_flag must be NORMAL or PREDICTED")
        smoothing_flag = OutputFlag(self.smoothing_flag)
        if smoothing_flag is not OutputFlag.SMOOTHED:
            raise ValueError("SmoothedStep smoothing_flag must be SMOOTHED")
        mean = readonly_array(self.mean)
        covariance = readonly_array(self.covariance)
        if mean.ndim != 1 or covariance.shape != (mean.shape[0], mean.shape[0]):
            raise ValueError("mean and covariance shapes are inconsistent")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "prediction_flag", prediction_flag)
        object.__setattr__(self, "smoothing_flag", smoothing_flag)


def _rts_gains(
    filtered_covariances: Array,
    predicted_covariances: Array,
    transition_matrices: Array,
) -> Array:
    """Compute forward-only RTS gains into one C-contiguous float array."""

    n_times, n_state, _ = filtered_covariances.shape
    gains = np.empty((max(n_times - 1, 0), n_state, n_state), dtype=float)
    for k in range(n_times - 2, -1, -1):
        gains[k] = _solve_right(
            predicted_covariances[k + 1],
            filtered_covariances[k] @ transition_matrices[k].T,
        )
    return gains


def _rts_smooth_arrays(
    filtered_means: Array,
    filtered_covariances: Array,
    predicted_means: Array,
    predicted_covariances: Array,
    transition_matrices: Array,
    *,
    gains: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Run the existing RTS equations over the supplied active window."""

    n_times, n_state = filtered_means.shape
    if n_times == 0:
        raise ValueError("RTS smoothing requires at least one filter step")
    if filtered_covariances.shape != (n_times, n_state, n_state):
        raise ValueError("filtered_covariances has an incompatible shape")
    if predicted_means.shape != (n_times, n_state):
        raise ValueError("predicted_means has an incompatible shape")
    if predicted_covariances.shape != (n_times, n_state, n_state):
        raise ValueError("predicted_covariances has an incompatible shape")
    if transition_matrices.shape != (max(n_times - 1, 0), n_state, n_state):
        raise ValueError("transition_matrices has an incompatible shape")

    gains = (
        _rts_gains(filtered_covariances, predicted_covariances, transition_matrices)
        if gains is None
        else gains
    )
    smoothed_means = _rts_smooth_means(filtered_means, predicted_means, gains)
    smoothed_covariances = np.empty_like(filtered_covariances)
    smoothed_covariances[-1] = filtered_covariances[-1]
    for k in range(n_times - 2, -1, -1):
        gain = gains[k]
        smoothed_covariances[k] = _symmetrize(
            filtered_covariances[k]
            + gain
            @ (smoothed_covariances[k + 1] - predicted_covariances[k + 1])
            @ gain.T
        )

    return smoothed_means, smoothed_covariances, gains


def _rts_smooth_means(
    filtered_means: Array, predicted_means: Array, gains: Array
) -> Array:
    """Run the RTS mean recursion over a nonempty window.

    This is the only copy of the mean recursion: :func:`_rts_smooth_arrays`
    calls it and adds the covariance recursion, and the batch path calls it
    alone when uncertainty is not requested.
    """

    n_times, n_state = filtered_means.shape
    if gains.shape != (max(n_times - 1, 0), n_state, n_state):
        raise ValueError("gains has an incompatible shape")

    smoothed_means = np.empty_like(filtered_means)
    smoothed_means[-1] = filtered_means[-1]
    for k in range(len(filtered_means) - 2, -1, -1):
        smoothed_means[k] = filtered_means[k] + gains[k] @ (
            smoothed_means[k + 1] - predicted_means[k + 1]
        )
    return smoothed_means


def smooth_filter_steps(steps: Iterable[FilterStep]) -> tuple[SmoothedStep, ...]:
    """Smooth all supplied filter steps and return one result per step.

    An empty input produces an empty tuple. The input steps are not changed.
    """

    return _smooth_filter_steps(steps)


def _smooth_filter_steps(
    steps: Iterable[FilterStep], *, gains: Array | None = None
) -> tuple[SmoothedStep, ...]:
    """Build generic smoothed records, optionally reusing active-window gains."""

    active = tuple(steps)
    if not active:
        return ()
    filtered_means = np.stack([step.filtered_mean for step in active])
    filtered_covariances = np.stack([step.filtered_covariance for step in active])
    predicted_means = np.stack([step.predicted_mean for step in active])
    predicted_covariances = np.stack([step.predicted_covariance for step in active])
    n_state = filtered_means.shape[1]
    transitions = (
        np.stack([step.transition_matrix for step in active[1:]])
        if len(active) > 1
        else np.empty((0, n_state, n_state), dtype=float)
    )
    means, covariances, _ = _rts_smooth_arrays(
        filtered_means,
        filtered_covariances,
        predicted_means,
        predicted_covariances,
        transitions,
        gains=gains,
    )
    return tuple(
        SmoothedStep(
            step.timestamp,
            means[index],
            covariances[index],
            prediction_flag=step.prediction_flag,
        )
        for index, step in enumerate(active)
    )


def _release_count[Step](
    active: Iterable[Step], lag_passed: Callable[[Step], bool]
) -> int:
    """Count the oldest active steps whose smoothing lag has passed.

    This is the release rule shared by :class:`OnlineFixedLagRTS` and the
    batch adapter; each supplies its own elapsed-time comparison. Active
    steps are in time order, so the scan stops at the first one still
    inside the lag.
    """

    count = 0
    for candidate in active:
        if not lag_passed(candidate):
            break
        count += 1
    return count


def _require_release_or_room(
    released: int, active_count: int, max_window_steps: int
) -> None:
    """Reject a step that would grow a full window without releasing any."""

    if released == 0 and active_count == max_window_steps:
        raise OverflowError(
            "active RTS window reached max_window_steps before a state finalized"
        )


class OnlineFixedLagRTS:
    """Keep recent filter steps until their smoothed results are ready.

    Results are released after the configured amount of time has passed. The
    number of steps kept in memory is capped by ``max_window_steps``.
    """

    def __init__(self, lag: timedelta, *, max_window_steps: int = 100_000) -> None:
        """Create a smoother with a time-based lag and a memory limit."""

        if lag.total_seconds() <= 0.0:
            raise ValueError("lag must be positive")
        window_size = bounded_integer(
            max_window_steps,
            name="max_window_steps",
            minimum=2,
        )
        self._lag = lag
        self._steps: deque[FilterStep] = deque(maxlen=window_size)
        self._gains: deque[Array] = deque(maxlen=window_size - 1)
        self._last_finalized_steps: tuple[FilterStep, ...] = ()

    @property
    def lag(self) -> timedelta:
        """The amount of later data required before a result is finalized."""

        return self._lag

    @property
    def pending_count(self) -> int:
        """The number of filter steps still waiting to be finalized."""

        return len(self._steps)

    @property
    def max_window_steps(self) -> int:
        """The maximum number of filter steps kept in memory."""

        maxlen = self._steps.maxlen
        return int(maxlen)

    @property
    def last_finalized_steps(self) -> tuple[FilterStep, ...]:
        """Forward-filter steps released by the most recent :meth:`add_step`."""

        return self._last_finalized_steps

    def add_step(self, step: FilterStep) -> tuple[SmoothedStep, ...]:
        """Add a filter step and return any newly finalized estimates.

        Estimates that are still within the lag remain available through
        :meth:`provisional`. Steps must arrive in strictly increasing time
        order.
        """

        if self._steps and to_utc(step.timestamp) <= to_utc(self._steps[-1].timestamp):
            raise ValueError("filter-step timestamps must be strictly increasing")
        lag_seconds = self._lag.total_seconds()
        # The new step has age zero and cannot finalize under a positive lag.
        eligible_count = _release_count(
            self._steps,
            lambda active_step: elapsed_seconds(
                step.timestamp, active_step.timestamp, allow_zero=True
            )
            >= lag_seconds,
        )
        _require_release_or_room(
            eligible_count, len(self._steps), self.max_window_steps
        )

        # Compute before mutating either deque. A releasing full window needs
        # its complete gain sequence until smoothing has succeeded.
        gain = None
        if self._steps:
            # np.stack preserves input layout; canonicalize before caching.
            gain = _solve_right(
                step.predicted_covariance,
                self._steps[-1].filtered_covariance @ step.transition_matrix.T,
            ).copy(order="C")
        if eligible_count == 0:
            self._steps.append(step)
            if gain is not None:
                self._gains.append(gain)
            self._last_finalized_steps = ()
            return ()

        active = (*self._steps, step)
        gains = (*self._gains, gain)
        smoothed = _smooth_filter_steps(active, gains=np.stack(gains))
        finalized = smoothed[:eligible_count]
        self._last_finalized_steps = active[:eligible_count]
        self._steps.clear()
        self._steps.extend(active[eligible_count:])
        self._gains.clear()
        self._gains.extend(gains[eligible_count:])
        return finalized

    def provisional(self) -> tuple[SmoothedStep, ...]:
        """Return lag-window estimates without finalizing or retaining copies."""

        if not self._steps:
            return ()
        n_state = self._steps[0].filtered_mean.size
        gains = (
            np.stack(self._gains)
            if self._gains
            else np.empty((0, n_state, n_state), dtype=float)
        )
        return _smooth_filter_steps(self._steps, gains=gains)

    def active_steps(self) -> tuple[FilterStep, ...]:
        """Return active forward-filter steps in timestamp order."""

        return tuple(self._steps)

    def clear(self) -> None:
        """Discard all active forward-filter steps."""

        self._steps.clear()
        self._gains.clear()
        self._last_finalized_steps = ()
