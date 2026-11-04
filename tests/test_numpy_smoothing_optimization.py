"""Bitwise parity and ownership checks for cached fixed-lag RTS gains."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from kalmanflow import (
    Observation,
    OnlineFixedLagRTS,
    OnlineReservoirInflow,
    UnitSystem,
    get_reservoir_inflow,
    rts,
    smooth_filter_steps,
)
from kalmanflow.core import _build_default_backend
from kalmanflow.kalman import FilterStep, _solve_right, kalman_filter
from kalmanflow.pipeline import InitializationObservation
from kalmanflow.rts import _rts_gains, _rts_smooth_arrays, _rts_smooth_means


def _arrays(dimension=5, count=9):
    rng = np.random.default_rng(141)
    means = rng.normal(size=(count, dimension))
    predicted_means = rng.normal(size=(count, dimension))
    factor = rng.normal(size=(count, dimension, dimension))
    covariances = factor @ factor.transpose(0, 2, 1) + np.eye(dimension)
    factor = rng.normal(size=(count, dimension, dimension))
    predicted_covariances = factor @ factor.transpose(0, 2, 1) + np.eye(dimension)
    transitions = rng.normal(size=(count - 1, dimension, dimension))
    return means, covariances, predicted_means, predicted_covariances, transitions


def _filter_arrays(dimension, count, singular):
    """Use consistent forward quantities for the cross-layout error bound."""

    rng = np.random.default_rng(141)
    # A bounded correlated model keeps this purely relative error check away
    # from cancellation in near-zero covariance entries. Unrelated random
    # covariance sequences retain exact internal-path comparisons below.
    coupling = np.ones((dimension, dimension))
    initial_covariance = 0.9 * np.eye(dimension) + 0.1 * coupling
    transitions = np.broadcast_to(
        0.7 * np.eye(dimension) + 0.02 * coupling,
        (count - 1, dimension, dimension),
    ).copy()
    process = np.broadcast_to(
        0.05 * np.eye(dimension) + 0.01 * coupling,
        (count - 1, dimension, dimension),
    ).copy()
    if singular:
        # Alternate rank-deficient and full-rank predictions without making
        # the forward covariance sequence inconsistent.
        transitions[::2, -1, :] = 0.0
        process[::2, -1, :] = 0.0
        process[::2, :, -1] = 0.0
    result = kalman_filter(
        5.0
        + 0.1 * np.arange(count)[:, None]
        + 0.02 * np.arange(dimension)[None, :]
        + rng.normal(0.0, 0.05, size=(count, dimension)),
        initial_mean=np.full(dimension, 5.0),
        initial_covariance=initial_covariance,
        transition_matrix=transitions,
        process_covariance=process,
        observation_matrix=np.eye(dimension),
        observation_covariance=0.2 * np.eye(dimension) + 0.05 * coupling,
    )
    return (
        result.filtered_means,
        result.filtered_covariances,
        result.predicted_means,
        result.predicted_covariances,
        result.transition_matrices,
    )


def _bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("dimension", [1, 3, 5, 8])
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("singular", [False, True])
def test_cached_gains_match_independent_recursion_and_do_not_mutate(
    dimension, count, singular
):
    # Arbitrary unrelated covariances can amplify rounding through an unstable
    # backward recursion. Keep those in the exact internal stress test below;
    # bound the old/new BLAS-order difference using valid forward-filter data.
    arrays = _filter_arrays(dimension, count, singular)
    originals = [array.copy() for array in arrays]
    means, covariances, predictions, predicted_covariances, transitions = arrays
    # Independent original equations: this remains an oracle after refactoring.
    expected_means, expected_covariances = means.copy(), covariances.copy()
    expected_gains = np.empty_like(transitions)
    for k in range(len(means) - 2, -1, -1):
        gain = _solve_right(
            predicted_covariances[k + 1], covariances[k] @ transitions[k].T
        )
        expected_gains[k] = gain
        expected_means[k] = means[k] + gain @ (
            expected_means[k + 1] - predictions[k + 1]
        )
        covariance = (
            covariances[k]
            + gain
            @ (expected_covariances[k + 1] - predicted_covariances[k + 1])
            @ gain.T
        )
        expected_covariances[k] = 0.5 * (covariance + covariance.T)
    gains = _rts_gains(covariances, predicted_covariances, transitions)
    _bits(gains, expected_gains)
    gain_snapshot = gains.copy()
    direct = _rts_smooth_arrays(*arrays)
    cached = _rts_smooth_arrays(*arrays, gains=gains)
    for actual in (direct, cached):
        for result, expected in zip(
            actual, (expected_means, expected_covariances, expected_gains), strict=True
        ):
            # The independent solve results are F-ordered (C for pseudoinverse),
            # while the implementation stores all gains in C order. BLAS may
            # therefore accumulate in a different order.
            np.testing.assert_allclose(result, expected, rtol=1e-13, atol=0)
    for uncached, supplied in zip(direct, cached, strict=True):
        _bits(uncached, supplied)
    assert cached[2] is gains
    for array, original in zip(arrays, originals, strict=True):
        _bits(array, original)
    _bits(gains, gain_snapshot)


@pytest.mark.parametrize("shape", [(2, 5, 5), (8, 4, 4), (8, 5), (8, 5, 5, 1)])
def test_supplied_gain_shape_is_checked(shape):
    arrays = _arrays()
    gains = np.zeros(shape)
    with pytest.raises(ValueError, match="gains has an incompatible shape"):
        _rts_smooth_arrays(*arrays, gains=gains)
    with pytest.raises(ValueError, match="gains has an incompatible shape"):
        _rts_smooth_means(arrays[0], arrays[2], gains)


@pytest.mark.parametrize("count", [0, 1, 6])
@pytest.mark.parametrize("dimension", [1, 3, 5])
def test_gain_cache_is_one_c_contiguous_float_array(count, dimension):
    covariance = np.broadcast_to(np.eye(dimension), (count, dimension, dimension))
    predicted = 2 * covariance
    transitions = np.broadcast_to(
        np.eye(dimension), (max(count - 1, 0), dimension, dimension)
    )
    gains = _rts_gains(covariance, predicted, transitions)
    assert gains.shape == (max(count - 1, 0), dimension, dimension)
    assert gains.dtype == np.dtype(float)
    assert gains.flags.c_contiguous
    assert gains.flags.owndata
    assert gains.base is None


def test_cached_singular_gain_uses_existing_pseudoinverse():
    means = np.ones((3, 2))
    covariances = np.zeros((3, 2, 2))
    transitions = np.zeros((2, 2, 2))
    arrays = means, covariances, means, covariances, transitions
    gains = _rts_gains(covariances, covariances, transitions)
    for cached, direct in zip(
        _rts_smooth_arrays(*arrays, gains=gains),
        _rts_smooth_arrays(*arrays),
        strict=True,
    ):
        _bits(cached, direct)


def test_stream_cache_matches_direct_windows_and_survives_release_and_overflow(
    monkeypatch,
):
    solve = Mock(wraps=_solve_right)
    monkeypatch.setattr(rts, "_solve_right", solve)
    arrays = _arrays(count=7)
    means, covariances, predictions, predicted_covariances, transitions = arrays
    start = datetime(2024, 1, 1, tzinfo=UTC)
    # Includes a gap releasing multiple states and a full-window release.
    steps = tuple(
        FilterStep(
            timestamp=start + timedelta(minutes=minutes),
            filtered_mean=means[k],
            filtered_covariance=covariances[k],
            predicted_mean=predictions[k],
            predicted_covariance=predicted_covariances[k],
            transition_matrix=np.eye(5) if k == 0 else transitions[k - 1],
        )
        for k, minutes in enumerate((0, 10, 20, 30, 65, 75, 85))
    )
    smoother = OnlineFixedLagRTS(timedelta(minutes=30), max_window_steps=3)
    assert smoother.provisional() == ()
    active = ()
    for step in steps:
        window = (*active, step)
        expected = smooth_filter_steps(window)
        eligible = sum(step.timestamp - old.timestamp >= smoother.lag for old in active)
        before = solve.call_count
        released = smoother.add_step(step)
        assert solve.call_count == before + bool(active)
        for actual, direct in zip(released, expected[:eligible], strict=True):
            _bits(actual.mean, direct.mean)
            _bits(actual.covariance, direct.covariance)
        assert smoother.last_finalized_steps == window[:eligible]
        active = window[eligible:]
        assert smoother.active_steps() == active
        assert smoother.pending_count == len(active)
        assert len(smoother._gains) == len(active) - 1
        assert smoother._gains.maxlen == smoother.max_window_steps - 1
        before = solve.call_count
        provisional = smoother.provisional()
        assert solve.call_count == before
        for actual, direct in zip(
            provisional, smooth_filter_steps(active), strict=True
        ):
            _bits(actual.mean, direct.mean)
            _bits(actual.covariance, direct.covariance)
    # Rejected steps leave both deques and the last release untouched.
    snapshot = smoother.active_steps(), tuple(smoother._gains)
    with pytest.raises(
        OverflowError,
        match=("active RTS window reached max_window_steps before a state finalized"),
    ):
        smoother.add_step(
            FilterStep(
                timestamp=steps[-1].timestamp + timedelta(minutes=1),
                filtered_mean=means[0],
                filtered_covariance=covariances[0],
                predicted_mean=predictions[0],
                predicted_covariance=predicted_covariances[0],
                transition_matrix=np.eye(5),
            )
        )
    assert smoother.active_steps() == snapshot[0]
    assert tuple(smoother._gains) == snapshot[1]
    smoother.clear()
    assert smoother.active_steps() == ()
    assert tuple(smoother._gains) == ()
    assert smoother.last_finalized_steps == ()
    assert smoother.provisional() == ()
    smoother.add_step(steps[0])
    assert len(smoother.provisional()) == 1


@pytest.mark.parametrize("dimension", [1, 3, 5])
@pytest.mark.parametrize("count", [1, 2, 9])
def test_means_only_is_bitwise_equal_to_full_recursion(dimension, count):
    arrays = _arrays(dimension, count)
    means, covariances, predictions, predicted_covariances, transitions = arrays
    gains = _rts_gains(covariances, predicted_covariances, transitions)
    originals = [array.copy() for array in arrays]
    expected, _, _ = _rts_smooth_arrays(*arrays, gains=gains)
    _bits(_rts_smooth_means(means, predictions, gains), expected)
    for array, original in zip(arrays, originals, strict=True):
        _bits(array, original)


@pytest.mark.parametrize("dimension", [1, 3, 5, 8])
@pytest.mark.parametrize("count", [1, 9])
@pytest.mark.parametrize("singular", [False, True])
def test_random_covariance_stress_paths_remain_bitwise_equal(
    dimension, count, singular
):
    arrays = _arrays(dimension, count)
    if singular:
        arrays[3][::2, -1, :] = 0.0
        arrays[3][::2, :, -1] = 0.0
    originals = [array.copy() for array in arrays]
    gains = _rts_gains(arrays[1], arrays[3], arrays[4])
    snapshot = gains.copy()
    direct = _rts_smooth_arrays(*arrays)
    cached = _rts_smooth_arrays(*arrays, gains=gains)
    for computed, supplied in zip(direct, cached, strict=True):
        _bits(computed, supplied)
    _bits(_rts_smooth_means(arrays[0], arrays[2], gains), direct[0])
    for array, original in zip(arrays, originals, strict=True):
        _bits(array, original)
    _bits(gains, snapshot)


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("si", [False, True])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_batch_stream_and_generic_revisions_agree_bitwise(irregular, si, uncertainty):
    rng = np.random.default_rng(616)
    count = 40
    intervals_us = (
        rng.integers(600_000_000, 1_200_000_000, count - 1)
        if irregular
        else np.full(count - 1, 900_000_000)
    )
    index = pd.DatetimeIndex(
        pd.Timestamp("2024-01-01", tz="UTC")
        + pd.to_timedelta(np.r_[0, np.cumsum(intervals_us)], unit="us")
    )
    storage = pd.Series(100.0 + np.arange(count) + rng.normal(0, 0.1, count), index)
    discharge = pd.Series(4.0 + rng.normal(0, 0.1, count), index)
    storage.iloc[5::7] = np.nan
    discharge.iloc[3::5] = np.nan
    settings = dict(
        q_storage=0.002,
        q_inflow=1.0,
        q_outflow=0.02,
        r_storage=0.25,
        r_outflow=4.0,
        unit_system=UnitSystem.si() if si else UnitSystem.us_customary(),
    )
    lag = timedelta(hours=1)
    batch = get_reservoir_inflow(
        storage,
        discharge,
        smoothing_lag=lag,
        include_uncertainty=uncertainty,
        **settings,
    )
    backend = _build_default_backend(**settings)
    observations = tuple(
        Observation(timestamp, value, outflow)
        for timestamp, value, outflow in zip(index, storage, discharge, strict=True)
    )
    steps = list(
        backend.initialize(
            *(
                InitializationObservation(o.timestamp, o.storage, o.discharge)
                for o in observations[:2]
            )
        )
    )
    for o in observations[2:]:
        steps.append(
            backend.advance(
                steps[-1],
                timestamp=o.timestamp,
                storage=o.storage,
                discharge=o.discharge,
            )
        )
    stream = OnlineReservoirInflow(
        smoothing_lag=lag, include_uncertainty=uncertainty, **settings
    )
    filtered, revised, generic = {}, {}, {}
    start = 0
    for current, observation in enumerate(observations):
        update = stream.process(observation)
        filtered.update((e.timestamp, e) for e in update.filtered_inflows)
        revised.update((e.timestamp, e) for e in update.revised_inflows)
        eligible = sum(index[current] - index[k] >= lag for k in range(start, current))
        if eligible:
            smoothed = smooth_filter_steps(steps[start : current + 1])
            generic.update((e.timestamp, e) for e in smoothed[:eligible])
            start += eligible
    assert revised.keys() == generic.keys()
    for timestamp, estimate in filtered.items():
        _bits(
            np.asarray(estimate.value),
            np.asarray(batch.loc[timestamp, "estimated_inflow"]),
        )
        assert (
            estimate.prediction_flag.value
            == batch.loc[timestamp, "estimated_inflow_flag"]
        )
        assert (
            estimate.smoothing_flag.value
            == batch.loc[timestamp, "estimated_inflow_smoothing_flag"]
        )
        if uncertainty:
            _bits(
                np.asarray(estimate.standard_deviation),
                np.asarray(batch.loc[timestamp, "estimated_inflow_standard_deviation"]),
            )
    for timestamp, estimate in revised.items():
        _bits(
            np.asarray(estimate.value),
            np.asarray(batch.loc[timestamp, "revised_inflow"]),
        )
        _bits(np.asarray(estimate.value), np.asarray(generic[timestamp].mean[1]))
        assert (
            estimate.prediction_flag.value
            == batch.loc[timestamp, "revised_inflow_flag"]
        )
        assert (
            estimate.smoothing_flag.value
            == batch.loc[timestamp, "revised_inflow_smoothing_flag"]
        )
        if uncertainty:
            _bits(
                np.asarray(estimate.standard_deviation),
                np.asarray(batch.loc[timestamp, "revised_inflow_standard_deviation"]),
            )
            _bits(
                np.asarray(estimate.standard_deviation),
                np.asarray(np.sqrt(generic[timestamp].covariance[1, 1])),
            )
