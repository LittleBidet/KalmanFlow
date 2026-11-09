"""Experimental maximum-likelihood tuning of reservoir noise covariances.

The public API is defined here; implementation details live in the focused
subpackage modules. Search requires ``kalmanflow[tuning]``; frozen
configuration evaluation requires only the base dependencies. The tuning API,
selection rules, and result schema may change between releases.
"""

from ._types import (
    ConfigEvaluationResult,
    NoiseEvaluationSettings,
    NoiseTuningError,
    NoiseTuningProgress,
    NoiseTuningResult,
    NoiseTuningSettings,
    ValidationWindow,
)
from ._workflows import evaluate_configuration, tune_inflow_noise

__all__ = [
    "ConfigEvaluationResult",
    "NoiseEvaluationSettings",
    "NoiseTuningError",
    "NoiseTuningProgress",
    "NoiseTuningResult",
    "NoiseTuningSettings",
    "ValidationWindow",
    "evaluate_configuration",
    "tune_inflow_noise",
]
