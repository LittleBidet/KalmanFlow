"""Experimental Bayesian innovation tuning for reservoir noise covariances.

The public API is defined here; implementation details live in the focused
subpackage modules below. Search requires ``kalmanflow[tuning]``; frozen
configuration evaluation requires only the base dependencies. The tuning API,
selection rules, and result schema may change between releases.
"""

from ._types import (
    BayesianEvaluationSettings,
    BayesianTuningError,
    BayesianTuningResult,
    BayesianTuningSettings,
    ConfigEvaluationResult,
    ValidationWindow,
)
from ._workflows import evaluate_configuration, tune_inflow_noise_bayesian

__all__ = [
    "BayesianEvaluationSettings",
    "BayesianTuningError",
    "BayesianTuningResult",
    "BayesianTuningSettings",
    "ConfigEvaluationResult",
    "ValidationWindow",
    "evaluate_configuration",
    "tune_inflow_noise_bayesian",
]
