from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from kalmanflow import (
    FilterStep,
    get_reservoir_inflow,
    initial_filter_step,
    kalman_filter,
)


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




@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("initial_mean", np.array([[0.0]]), "predicted_mean must be one-dimensional"),
        (
            "initial_mean",
            np.array([]),
            "predicted_mean must contain at least one state",
        ),
        (
            "observation",
            np.array([[1.0]]),
            "observation must be one-dimensional",
        ),
        (
            "observation",
            np.array([]),
            "observation must contain at least one component",
        ),
    ],
)
def test_initial_filter_step_validates_update_inputs(
    field: str, value: np.ndarray, message: str
) -> None:
    kwargs: dict[str, object] = {
        "timestamp": datetime(2024, 1, 1, tzinfo=UTC),
        "initial_mean": np.array([0.0]),
        "initial_covariance": np.array([[1.0]]),
        "observation": np.array([1.0]),
        "observation_matrix": np.array([[1.0]]),
        "observation_covariance": np.array([[0.5]]),
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        initial_filter_step(**kwargs)


def test_initial_filter_step_normalizes_scalar_observation() -> None:
    step = initial_filter_step(
        timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        initial_mean=np.array([0.0]),
        initial_covariance=np.array([[1.0]]),
        observation=np.array(1.0),
        observation_matrix=np.array([[1.0]]),
        observation_covariance=np.array([[0.5]]),
    )

    assert step.filtered_mean.shape == (1,)


def test_filter_step_is_frozen_and_deeply_immutable() -> None:
    timestamp = datetime(2024, 1, 1, tzinfo=UTC)
    arrays = {
        "filtered_mean": np.array([1.0, 2.0]),
        "filtered_covariance": np.array([[2.0, 0.1], [0.1, 3.0]]),
        "predicted_mean": np.array([1.5, 1.75]),
        "predicted_covariance": np.array([[3.0, 0.2], [0.2, 4.0]]),
        "transition_matrix": np.array([[1.0, 0.1], [0.0, 1.0]]),
    }
    snapshots = {name: value.copy() for name, value in arrays.items()}
    step = FilterStep(
        timestamp=timestamp,
        **arrays,
    )

    for name, original in arrays.items():
        value = getattr(step, name)
        assert value.flags.writeable is False
        assert not np.shares_memory(value, original)
        npt.assert_array_equal(value, snapshots[name])
        original[...] += 10.0
        npt.assert_array_equal(value, snapshots[name])
        with pytest.raises(ValueError):
            value[...] = 0.0

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


def test_batch_rejects_submicrosecond_timestamps() -> None:
    index = pd.DatetimeIndex(
        [
            "2024-01-01T00:00:00.000000123Z",
            "2024-01-01T00:15:00.000000123Z",
        ]
    )
    storage = pd.Series([100.0, 101.0], index=index)
    discharge = pd.Series([4.0, 4.1], index=index)
    with pytest.raises(ValueError, match="finer than microsecond"):
        get_reservoir_inflow(storage, discharge, 1.0, 1.0, 1.0, 1.0, 1.0)
