"""Black-box tests for Kalman filter primitives.

Coverage techniques:
- Boundary Value Analysis (BVA)
- Equivalence Partitioning (EP)
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import numpy as np
import numpy.testing as npt
import pytest

from kalmone import FilterStep, kalman_filter, kalman_step, predict_state


def _scalar_filter(**overrides: object):
    defaults = {
        "observations": np.array([2.0, 2.5]),
        "initial_mean": np.array([0.0]),
        "initial_covariance": np.array([[4.0]]),
        "transition_matrix": np.array([[1.0]]),
        "process_covariance": np.array([[1.0]]),
        "observation_matrix": np.array([[1.0]]),
        "observation_covariance": np.array([[1.0]]),
    }
    defaults.update(overrides)
    return kalman_filter(**defaults)


class TestObservationPartitions:
    """EP: scalar vs vector observation layouts."""

    def test_scalar_observations_are_accepted(self) -> None:
        result = _scalar_filter(observations=np.array([1.0, 2.0, 3.0]))
        assert result.filtered_means.shape == (3, 1)
        assert result.update_mask.tolist() == [True, True, True]

    def test_two_dimensional_observations_are_accepted(self) -> None:
        result = _scalar_filter(
            observations=np.array([[1.0], [2.0], [3.0]]),
            initial_mean=np.array([0.0, 0.0]),
            initial_covariance=np.eye(2),
            transition_matrix=np.eye(2),
            process_covariance=np.eye(2) * 0.1,
            observation_matrix=np.array([[1.0, 0.0]]),
            observation_covariance=np.array([[0.5]]),
        )
        assert result.filtered_means.shape == (3, 2)

    @pytest.mark.parametrize(
        "observations",
        [
            np.empty((0,)),
            np.empty((0, 1)),
        ],
    )
    def test_empty_observations_rejected(self, observations: np.ndarray) -> None:
        with pytest.raises(ValueError, match="at least one time step"):
            _scalar_filter(observations=observations)

    def test_three_dimensional_observations_rejected(self) -> None:
        with pytest.raises(ValueError, match="one- or two-dimensional"):
            _scalar_filter(observations=np.ones((2, 1, 1)))


class TestShapeBoundaries:
    """BVA/EP: state and matrix shape partitions."""

    def test_initial_mean_must_be_one_dimensional(self) -> None:
        with pytest.raises(ValueError, match="initial_mean must be one-dimensional"):
            _scalar_filter(initial_mean=np.array([[0.0]]))

    def test_initial_covariance_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="initial_covariance must have shape"):
            _scalar_filter(initial_covariance=np.eye(2))

    def test_predict_state_rejects_mismatched_control_offset(self) -> None:
        with pytest.raises(ValueError, match="control_offset must have shape"):
            predict_state(
                np.array([1.0, 2.0]),
                np.eye(2),
                np.eye(2),
                np.eye(2),
                control_offset=np.array([1.0]),
            )


class TestTimeVaryingMatrixPartitions:
    """EP: constant vs time-varying matrix lengths."""

    def test_transition_matrix_accepts_n_transitions_length(self) -> None:
        result = _scalar_filter(
            observations=np.array([1.0, 2.0, 3.0]),
            transition_matrix=np.stack([np.array([[1.0]]), np.array([[1.0]])]),
        )
        assert result.transition_matrices.shape == (2, 1, 1)

    def test_transition_matrix_accepts_n_times_length(self) -> None:
        result = _scalar_filter(
            observations=np.array([1.0, 2.0, 3.0]),
            transition_matrix=np.stack(
                [np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]])]
            ),
        )
        assert result.transition_matrices.shape == (2, 1, 1)

    def test_transition_matrix_rejects_invalid_length(self) -> None:
        with pytest.raises(ValueError, match="transition_matrix must have shape"):
            _scalar_filter(
                observations=np.array([1.0, 2.0, 3.0]),
                transition_matrix=np.stack([np.array([[1.0]])]),
            )

    def test_observation_matrix_accepts_per_time_matrices(self) -> None:
        result = _scalar_filter(
            observations=np.array([1.0, 2.0]),
            observation_matrix=np.stack([np.array([[1.0]]), np.array([[1.0]])]),
        )
        assert result.filtered_means.shape == (2, 1)

    def test_observation_matrix_rejects_wrong_time_length(self) -> None:
        with pytest.raises(ValueError, match="observation_matrix must have shape"):
            _scalar_filter(
                observations=np.array([1.0, 2.0, 3.0]),
                observation_matrix=np.stack([np.array([[1.0]]), np.array([[1.0]])]),
            )


class TestControlPartitions:
    """EP: control argument pairing and shape partitions."""

    def test_control_matrix_without_controls_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be supplied together"):
            _scalar_filter(
                observations=np.array([1.0, 2.0]),
                control_matrix=np.array([[1.0]]),
            )

    def test_controls_without_control_matrix_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be supplied together"):
            _scalar_filter(
                observations=np.array([1.0, 2.0]),
                controls=np.array([0.5, 0.5]),
            )

    def test_per_step_controls_and_matrix_are_accepted(self) -> None:
        result = _scalar_filter(
            observations=np.array([1.0, 2.0, 3.0]),
            control_matrix=np.stack([np.array([[1.0]]), np.array([[1.0]])]),
            controls=np.array([0.5, 1.0]),
        )
        assert result.filtered_means.shape == (3, 1)

    def test_per_step_control_offsets_are_accepted(self) -> None:
        result = _scalar_filter(
            observations=np.array([1.0, 2.0, 3.0]),
            control_offsets=np.array([[0.5], [1.0]]),
        )
        assert result.filtered_means.shape == (3, 1)

    def test_invalid_control_offsets_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="control_offsets must have shape"):
            _scalar_filter(
                observations=np.array([1.0, 2.0, 3.0]),
                control_offsets=np.array([[0.5]]),
            )


class TestMissingObservationPartitions:
    """EP: all-missing, partial, and finite observation components."""

    def test_all_missing_observation_is_predict_only(self) -> None:
        result = _scalar_filter(observations=np.array([np.nan]))
        assert result.update_mask.tolist() == [False]
        npt.assert_allclose(result.filtered_means[0], result.predicted_means[0])
        assert result.log_likelihood == 0.0

    def test_partial_missing_updates_only_finite_components(self) -> None:
        result = kalman_filter(
            observations=np.array([[1.0, np.nan], [np.nan, 3.0]]),
            initial_mean=np.zeros(2),
            initial_covariance=np.eye(2),
            transition_matrix=np.eye(2),
            process_covariance=np.eye(2) * 0.1,
            observation_matrix=np.eye(2),
            observation_covariance=np.eye(2) * 0.5,
        )
        assert result.update_mask.tolist() == [True, True]
        assert np.isnan(result.innovations[1, 0])
        assert np.isfinite(result.innovations[1, 1])


class TestSingularInnovationFallback:
    """BVA: near-singular innovation covariance still produces finite output."""

    def test_singular_innovation_covariance_still_updates(self) -> None:
        result = _scalar_filter(
            observations=np.array([1.0]),
            initial_mean=np.array([0.0]),
            initial_covariance=np.array([[0.0]]),
            transition_matrix=np.array([[1.0]]),
            process_covariance=np.array([[0.0]]),
            observation_matrix=np.array([[1.0]]),
            observation_covariance=np.array([[0.0]]),
        )
        assert np.isfinite(result.filtered_means[0, 0])
        assert result.update_mask.tolist() == [True]


class TestFilterStepContract:
    """EP/BVA: FilterStep validation and immutability."""

    def test_filter_step_rejects_mismatched_covariance_shape(self) -> None:
        with pytest.raises(ValueError, match="filtered_covariance must have shape"):
            FilterStep(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                filtered_mean=np.array([1.0, 2.0]),
                filtered_covariance=np.eye(3),
                predicted_mean=np.array([1.0, 2.0]),
                predicted_covariance=np.eye(2),
                transition_matrix=np.eye(2),
            )

    def test_kalman_step_rejects_non_one_dimensional_observation(self) -> None:
        with pytest.raises(ValueError, match="observation must be one-dimensional"):
            kalman_step(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                previous_filtered_mean=np.array([1.0]),
                previous_filtered_covariance=np.array([[1.0]]),
                transition_matrix=np.array([[1.0]]),
                process_covariance=np.array([[0.1]]),
                observation=np.array([[1.0]]),
                observation_matrix=np.array([[1.0]]),
                observation_covariance=np.array([[0.5]]),
            )

    def test_filter_step_arrays_are_read_only(self) -> None:
        step = FilterStep(
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            filtered_mean=np.array([1.0]),
            filtered_covariance=np.array([[1.0]]),
            predicted_mean=np.array([1.0]),
            predicted_covariance=np.array([[1.0]]),
            transition_matrix=np.array([[1.0]]),
        )
        with pytest.raises(ValueError):
            step.filtered_mean[0] = 0.0
        with pytest.raises(FrozenInstanceError):
            step.timestamp = datetime(2024, 1, 2, tzinfo=UTC)
