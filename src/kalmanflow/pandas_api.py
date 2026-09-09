"""Convenience adapter for pandas."""

from __future__ import annotations

from datetime import timedelta

import pandas

from .core import get_reservoir_inflow


def run_inflow_model(
    observations: pandas.DataFrame,
    *,
    q_storage: float,
    q_inflow: float,
    q_outflow: float,
    r_storage: float,
    r_outflow: float,
    smoothing_lag: timedelta = timedelta(hours=12),
    max_window_steps: int = 100_000,
    include_uncertainty: bool = False,
) -> pandas.DataFrame:
    """Estimate causal and revised inflows from storage and outflow inputs.

    ``observations`` must have ``storage`` and ``outflow`` columns on a
    timezone-aware, strictly increasing ``DateTimeIndex``. Both columns are
    model inputs: outflow observations inform the latent water-balance state
    but are never returned. The result contains only causal ``estimated_inflow``
    values and absolute fixed-lag ``revised_inflow`` replacements. Set
    ``include_uncertainty=True`` to append model-based standard deviations.
    """
    return get_reservoir_inflow(
        observations["storage"],
        observations["outflow"],
        q_storage=q_storage,
        q_inflow=q_inflow,
        q_outflow=q_outflow,
        r_storage=r_storage,
        r_outflow=r_outflow,
        smoothing_lag=smoothing_lag,
        max_window_steps=max_window_steps,
        include_uncertainty=include_uncertainty,
    )
