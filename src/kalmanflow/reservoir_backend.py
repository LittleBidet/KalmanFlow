"""Three-state reservoir backend with joint storage and outflow observations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from ._validation import covariance_array
from .kalman import initial_filter_step, kalman_step
from .models import StateSpaceModel
from .pipeline import InitializationObservation
from .time_utils import elapsed_seconds


@dataclass(frozen=True)
class ReservoirBackend:
    """Connects a reservoir model to the generic streaming pipeline."""

    model: StateSpaceModel
    initial_covariance: np.ndarray
    observation_covariance: np.ndarray

    def __post_init__(self) -> None:
        observation_matrix = np.asarray(self.model.observation_matrix, dtype=float)
        if observation_matrix.ndim != 2:
            raise ValueError("model observation_matrix must be two-dimensional")
        observation_size, state_size = observation_matrix.shape
        initial_covariance = covariance_array(
            self.initial_covariance,
            name="initial_covariance",
            shape=(state_size, state_size),
        )
        observation_covariance = covariance_array(
            self.observation_covariance,
            name="observation_covariance",
            shape=(observation_size, observation_size),
            positive_diagonal=True,
        )
        object.__setattr__(self, "initial_covariance", initial_covariance)
        object.__setattr__(self, "observation_covariance", observation_covariance)

    def initialize(
        self,
        first: InitializationObservation,
        second: InitializationObservation,
    ) -> tuple[Any, Any]:
        """Create the first two filter steps from the starting observations."""

        interval_seconds = elapsed_seconds(second.timestamp, first.timestamp)
        # A storage derivative cannot be known at ``first.timestamp`` without
        # looking ahead. Use the measured outflow as a steady-state inflow
        # prior so this first filtered state depends only on information that
        # was available at its own timestamp.
        initial_outflow = self.model.initial_outflow(first.discharge)
        first_step = initial_filter_step(
            timestamp=first.timestamp,
            initial_mean=np.array(
                [
                    first.storage,
                    initial_outflow,
                    initial_outflow,
                ]
            ),
            initial_covariance=self.initial_covariance,
            observation=np.array([first.storage, first.discharge]),
            observation_matrix=self.model.observation_matrix,
            observation_covariance=self.observation_covariance,
        )
        second_step = kalman_step(
            timestamp=second.timestamp,
            previous_filtered_mean=first_step.filtered_mean,
            previous_filtered_covariance=first_step.filtered_covariance,
            transition_matrix=self.model.transition_matrix(interval_seconds),
            process_covariance=self.model.process_covariance(interval_seconds),
            observation=np.array([second.storage, second.discharge]),
            observation_matrix=self.model.observation_matrix,
            observation_covariance=self.observation_covariance,
        )
        return first_step, second_step

    def advance(
        self,
        previous: Any,
        *,
        timestamp: datetime,
        storage: float,
        discharge: float,
    ) -> Any:
        """Create the next filter step from one new observation."""

        interval_seconds = elapsed_seconds(timestamp, previous.timestamp)
        return kalman_step(
            timestamp=timestamp,
            previous_filtered_mean=previous.filtered_mean,
            previous_filtered_covariance=previous.filtered_covariance,
            transition_matrix=self.model.transition_matrix(interval_seconds),
            process_covariance=self.model.process_covariance(interval_seconds),
            observation=np.array([storage, discharge]),
            observation_matrix=self.model.observation_matrix,
            observation_covariance=self.observation_covariance,
        )


__all__ = ["ReservoirBackend"]
