from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import numpy as np
import numpy.testing as npt
import pytest

from kalmone import FilterStep, kalman_filter, kalman_step, predict_state


def test_kalman_filter_matches_scalar_closed_form_updates() -> None:
    result = kalman_filter(
        observations=np.array([2.0, 2.5]),
        initial_mean=np.array([0.0]),
        initial_covariance=np.array([[4.0]]),
        transition_matrix=np.array([[1.0]]),
        process_covariance=np.array([[1.0]]),
        observation_matrix=np.array([[1.0]]),
        observation_covariance=np.array([[1.0]]),
    )

    npt.assert_allclose(result.predicted_means, [[0.0], [1.6]])
    npt.assert_allclose(result.filtered_means, [[1.6], [2.17857143]])
    npt.assert_allclose(result.filtered_covariances[0], [[0.8]])
    assert result.update_mask.tolist() == [True, True]


def test_predict_and_single_step_support_control_offsets() -> None:
    transition = np.array([[1.0, 1.0], [0.0, 1.0]])
    process = np.diag([0.2, 0.05])
    mean, covariance = predict_state(
        np.array([1.0, 2.0]),
        np.eye(2),
        transition,
        process,
        control_offset=np.array([0.5, -0.25]),
    )

    npt.assert_allclose(mean, [3.5, 1.75])
    npt.assert_allclose(covariance, transition @ transition.T + process)

    step = kalman_step(
        timestamp=datetime(2024, 1, 1, 0, 15, tzinfo=UTC),
        previous_filtered_mean=np.array([1.0, 2.0]),
        previous_filtered_covariance=np.eye(2),
        transition_matrix=transition,
        process_covariance=process,
        observation=np.array([3.0]),
        observation_matrix=np.array([[1.0, 0.0]]),
        observation_covariance=np.array([[0.5]]),
        control_offset=np.array([0.5, -0.25]),
    )
    npt.assert_allclose(step.predicted_mean, mean)
    assert step.timestamp.tzinfo is UTC
    assert step.filtered_mean.shape == (2,)


def test_missing_components_are_predict_only_without_discarding_observations() -> None:
    result = kalman_filter(
        observations=np.array([[1.0, 2.0], [np.nan, 3.0], [np.nan, np.nan]]),
        initial_mean=np.zeros(2),
        initial_covariance=np.eye(2),
        transition_matrix=np.eye(2),
        process_covariance=np.eye(2) * 0.1,
        observation_matrix=np.eye(2),
        observation_covariance=np.eye(2) * 0.5,
    )

    assert result.update_mask.tolist() == [True, True, False]
    assert np.isnan(result.innovations[1, 0])
    assert np.isfinite(result.innovations[1, 1])
    npt.assert_allclose(result.filtered_means[1, 0], result.predicted_means[1, 0])
    npt.assert_allclose(result.filtered_means[2], result.predicted_means[2])


def test_filter_step_is_frozen_and_deeply_immutable() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    step = FilterStep(
        timestamp=timestamp,
        filtered_mean=np.array([1.0, 2.0]),
        filtered_covariance=np.eye(2),
        predicted_mean=np.array([1.0, 2.0]),
        predicted_covariance=np.eye(2),
        transition_matrix=np.eye(2),
    )

    with pytest.raises(ValueError):
        step.filtered_mean[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        step.timestamp = datetime(2024, 1, 2, tzinfo=UTC)


def test_filter_step_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FilterStep(
            timestamp=datetime(2024, 1, 1),
            filtered_mean=np.array([1.0]),
            filtered_covariance=np.ones((1, 1)),
            predicted_mean=np.array([1.0]),
            predicted_covariance=np.ones((1, 1)),
            transition_matrix=np.ones((1, 1)),
        )
