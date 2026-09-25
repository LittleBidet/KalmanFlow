"""Focused validation and numerical fallback tests for Kalman primitives."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from kalmanflow import FilterStep, kalman_filter, predict_state
from kalmanflow import kalman as kalman_module


def _scalar_filter(**overrides: object):
    values: dict[str, object] = {
        "observations": np.array([1.0, 2.0]),
        "initial_mean": np.array([0.0]),
        "initial_covariance": np.array([[1.0]]),
        "transition_matrix": np.array([[1.0]]),
        "process_covariance": np.array([[0.1]]),
        "observation_matrix": np.array([[1.0]]),
        "observation_covariance": np.array([[0.5]]),
    }
    values.update(overrides)
    return kalman_filter(**values)


def _step_kwargs() -> dict[str, object]:
    return {
        "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        "filtered_mean": np.array([1.0]),
        "filtered_covariance": np.array([[1.0]]),
        "predicted_mean": np.array([1.0]),
        "predicted_covariance": np.array([[1.0]]),
        "transition_matrix": np.array([[1.0]]),
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestamp", object(), "timestamp must be a datetime"),
        ("prediction_flag", "SMOOTHED", "prediction_flag"),
        ("filtered_mean", np.array([[1.0]]), "matching 1-D shapes"),
        ("filtered_mean", np.array([]), "at least one state"),
        ("filtered_covariance", np.ones((2, 2)), "filtered_covariance must have shape"),
    ],
)
def test_filter_step_rejects_remaining_boundary_values(
    field: str, value: object, message: str
) -> None:
    kwargs = _step_kwargs()
    kwargs[field] = value
    if field == "filtered_mean" and isinstance(value, np.ndarray) and not value.size:
        kwargs["predicted_mean"] = np.array([])
    with pytest.raises((TypeError, ValueError), match=message):
        FilterStep(**kwargs)


def test_kalman_filter_rejects_empty_observation_components_and_state() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        _scalar_filter(observations=np.empty((2, 0)))
    with pytest.raises(
        ValueError, match="initial_mean must contain at least one state"
    ):
        _scalar_filter(initial_mean=np.array([]))


def test_predict_state_rejects_nonvector_and_empty_means() -> None:
    with pytest.raises(ValueError, match="filtered_mean must be one-dimensional"):
        predict_state(np.array([[0.0]]), np.eye(1), np.eye(1), np.eye(1))
    with pytest.raises(
        ValueError, match="filtered_mean must contain at least one state"
    ):
        predict_state(
            np.array([]), np.empty((0, 0)), np.empty((0, 0)), np.empty((0, 0))
        )


def test_update_helper_normalizes_scalar_and_rejects_nonvector_observation() -> None:
    updated = kalman_module._update_prediction(
        np.array([0.0]),
        np.array([[1.0]]),
        np.array(1.0),
        np.array([[1.0]]),
        np.array([[0.5]]),
    )
    assert updated[4] is True
    assert updated[0][0] == pytest.approx(2.0 / 3.0)
    assert updated[1][0, 0] == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="observation must be one-dimensional"):
        kalman_module._update_prediction(
            np.array([0.0]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[1.0]]),
            np.array([[0.5]]),
        )


def test_matrix_control_validation_reports_invalid_matrix_and_control_shapes() -> None:
    with pytest.raises(ValueError, match="control_matrix must have shape"):
        _scalar_filter(
            observations=np.array([1.0, 2.0]),
            control_matrix=np.ones((2, 1, 1)),
            controls=np.array([0.5]),
        )
    with pytest.raises(ValueError, match="controls must have shape"):
        _scalar_filter(
            observations=np.array([1.0, 2.0, 3.0]),
            control_matrix=np.array([[1.0]]),
            controls=np.array([0.5, 0.6, 0.7]),
        )
    with pytest.raises(ValueError, match="controls must have shape"):
        _scalar_filter(
            observations=np.array([1.0, 2.0, 3.0]),
            control_matrix=np.array([[1.0]]),
            controls=np.ones((1, 1)),
        )

    result = _scalar_filter(
        observations=np.array([np.nan, np.nan, np.nan]),
        initial_mean=np.array([0.0]),
        initial_covariance=np.array([[0.0]]),
        transition_matrix=np.array([[1.0]]),
        process_covariance=np.array([[0.0]]),
        control_matrix=np.array([[1.0]]),
        controls=np.array([[0.5], [0.75]]),
    )
    np.testing.assert_allclose(result.predicted_means[:, 0], [0.0, 0.5, 1.25])


def test_covariance_stack_handles_empty_nonsymmetric_and_indefinite_stacks() -> None:
    _scalar_filter(
        observations=np.array([1.0]),
        process_covariance=np.empty((0, 1, 1)),
    )
    with pytest.raises(ValueError, match=r"covariance\[0\] must be symmetric"):
        _scalar_filter(
            observations=np.array([1.0, 2.0]),
            initial_mean=np.zeros(2),
            initial_covariance=np.eye(2),
            transition_matrix=np.eye(2),
            process_covariance=np.array([[[1.0, 1.0], [0.0, 1.0]]]),
            observation_matrix=np.array([[1.0, 0.0]]),
        )
    with pytest.raises(
        ValueError, match=r"covariance\[0\] must be positive semidefinite"
    ):
        _scalar_filter(
            observations=np.array([1.0, 2.0]),
            initial_mean=np.zeros(2),
            initial_covariance=np.eye(2),
            transition_matrix=np.eye(2),
            process_covariance=np.array([[[1.0, 0.0], [0.0, -1.0]]]),
            observation_matrix=np.array([[1.0, 0.0]]),
        )


def test_logpdf_handles_nonpositive_and_singular_linear_systems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        kalman_module._logpdf_zero_mean(np.array([1.0]), np.array([[-1.0]]))

    original_solve = kalman_module.np.linalg.solve

    def fail_solve(*args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise np.linalg.LinAlgError("forced singular solve")

    monkeypatch.setattr(kalman_module.np.linalg, "solve", fail_solve)
    # The pseudo-inverse branch remains finite when the direct solve fails.
    value = kalman_module._logpdf_zero_mean(np.array([1.0]), np.array([[1.0]]))
    assert value == pytest.approx(-0.5 * (np.log(2.0 * np.pi) + 1.0))
    monkeypatch.setattr(kalman_module.np.linalg, "solve", original_solve)
