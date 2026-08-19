"""Black-box coverage for dataframe-first reservoir tuning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalmone import (
    OnlineReservoirInflow,
    get_reservoir_inflow_from_config,
    load_reservoir_configs,
    tune_inflow_model,
    tune_reservoirs,
)


def _frame(*, count: int = 48, irregular: bool = False) -> pd.DataFrame:
    if irregular:
        seconds = np.cumsum(np.resize(np.array([300, 900, 1_800, 600]), count))
        index = pd.to_datetime("2025-01-01", utc=True) + pd.to_timedelta(
            seconds, unit="s"
        )
    else:
        index = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    outflow = 15.0 + 1.5 * np.sin(np.linspace(0.0, 4.0, count))
    inflow = 19.0 + 3.0 * np.cos(np.linspace(0.0, 7.0, count))
    elapsed = np.r_[0.0, np.diff(index.asi8) / 1_000_000_000.0]
    storage = 1_500.0 + np.cumsum((inflow - outflow) * elapsed / 43_560.0)
    return pd.DataFrame({"storage": storage, "outflow": outflow}, index=index)


def test_single_tuner_uses_storage_and_outflow_only_without_mutating_input() -> None:
    observations = _frame()
    original = observations.copy(deep=True)

    result = tune_inflow_model(
        observations, reservoir_id="lexington", max_evaluations=12
    )

    assert observations.equals(original)
    assert set(result.parameters) == {
        "q_storage",
        "q_inflow",
        "q_outflow",
        "r_storage",
        "r_outflow",
    }
    assert all(
        np.isfinite(value) and value > 0.0 for value in result.parameters.values()
    )
    assert np.isfinite(result.score)
    assert result.evaluations <= 12
    with pytest.raises(ValueError):
        result.config.q[0, 0] = 0.0
    assert result.model_output.index.equals(observations.index)


def test_dataframe_contract_and_irregular_missing_observations() -> None:
    observations = _frame(irregular=True)
    observations.loc[observations.index[8], "storage"] = np.nan
    observations.loc[observations.index[11], "outflow"] = np.nan

    result = tune_inflow_model(observations, reservoir_id="calero", max_evaluations=8)

    assert result.diagnostics["missing_storage_count"] == 1
    assert result.diagnostics["missing_outflow_count"] == 1
    with pytest.raises(ValueError, match="required columns"):
        tune_inflow_model(observations.drop(columns="outflow"), reservoir_id="calero")


def test_search_is_deterministic_and_honors_evaluation_budget() -> None:
    observations = _frame()
    first = tune_inflow_model(
        observations, reservoir_id="anderson", max_evaluations=11, random_state=9
    )
    second = tune_inflow_model(
        observations, reservoir_id="anderson", max_evaluations=11, random_state=9
    )

    assert first.parameters == second.parameters
    assert first.score == second.score
    assert first.evaluations == second.evaluations <= 11


def test_progressive_search_reuses_bounded_prepared_prefixes() -> None:
    observations = _frame(count=20)
    observations.loc[observations.index[1], "storage"] = np.nan

    result = tune_inflow_model(
        observations,
        reservoir_id="anderson",
        max_evaluations=8,
        max_tuning_rows=5,
    )

    stage_rows = result.diagnostics["stage_rows"]
    assert all(stage_rows[stage] <= 5 for stage in ("broad", "survivors", "refined"))
    assert stage_rows["full"] == len(observations)


def test_batch_isolates_failures_and_parallel_matches_sequential() -> None:
    first = _frame()
    second = _frame()
    second["outflow"] = second["outflow"] * 2.0
    reservoir_data = {
        "lexington": first,
        "anderson": second,
        "bad": first.drop(columns="outflow"),
    }

    sequential = tune_reservoirs(
        reservoir_data, max_evaluations=8, max_tuning_rows=24, random_state=7
    )
    parallel = tune_reservoirs(
        reservoir_data,
        max_evaluations=8,
        max_tuning_rows=24,
        random_state=7,
        workers=2,
    )

    assert sequential.successful == ("lexington", "anderson")
    assert "bad" in sequential.failed
    assert (
        sequential.results["lexington"].parameters
        == parallel.results["lexington"].parameters
    )
    assert (
        sequential.results["anderson"].parameters
        == parallel.results["anderson"].parameters
    )
    assert (
        sequential.results["lexington"].config.p0
        is not sequential.results["anderson"].config.p0
    )


def test_configuration_round_trip_and_production_entry_points(tmp_path: Path) -> None:
    observations = _frame()
    batch = tune_reservoirs({"lexington": observations}, max_evaluations=8)
    path = batch.save(tmp_path / "reservoir-configurations.json")

    configs = load_reservoir_configs(path)
    payload = json.loads(path.read_text())
    config = configs["lexington"]
    stream = OnlineReservoirInflow.from_config(config)
    output = get_reservoir_inflow_from_config(
        observations["storage"], observations["outflow"], config
    )

    assert stream.reservoir_id == "lexington"
    assert output.index.equals(observations.index)
    assert np.allclose(config.q, batch.results["lexington"].config.q)
    assert (
        payload["configs"]["lexington"]["tuning_timestamp"]
        == batch.results["lexington"].diagnostics["tuning_timestamp"]
    )
    assert payload["saved_at"]
    assert "observations" not in payload["configs"]["lexington"]
