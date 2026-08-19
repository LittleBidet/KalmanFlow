"""Black-box coverage for dataframe-first reservoir tuning."""

from __future__ import annotations

import json
from datetime import timedelta
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
from kalmone.tuning import _prepare_arrays
from kalmone.units import UnitSystem


def _frame(*, count: int = 48, irregular: bool = False) -> pd.DataFrame:
    if irregular:
        elapsed = np.r_[0.0, np.resize(np.array([300, 900, 1_800, 600]), count - 1)]
        seconds = np.cumsum(elapsed)
        index = pd.to_datetime("2025-01-01", utc=True) + pd.to_timedelta(
            seconds, unit="s"
        )
    else:
        index = pd.date_range("2025-01-01", periods=count, freq="15min", tz="UTC")
    outflow = 15.0 + 1.5 * np.sin(np.linspace(0.0, 4.0, count))
    inflow = 19.0 + 3.0 * np.cos(np.linspace(0.0, 7.0, count))
    elapsed = np.r_[0.0, (index[1:] - index[:-1]).total_seconds()]
    storage = 1_500.0 + np.cumsum((inflow - outflow) * elapsed / 43_560.0)
    return pd.DataFrame({"storage": storage, "outflow": outflow}, index=index)


def _prepared(observations: pd.DataFrame):
    return _prepare_arrays(
        observations.index,
        observations["storage"].to_numpy(),
        observations["outflow"].to_numpy(),
        unit_system=UnitSystem.us_customary(),
        validation_fraction=0.25,
    )


def test_regular_15_minute_intervals_use_elapsed_seconds() -> None:
    observations = _frame(count=4)
    prepared = _prepared(observations)

    np.testing.assert_allclose(prepared.elapsed_seconds, [900.0, 900.0, 900.0])
    expected_initial_inflow = observations.outflow.iloc[0] + (
        observations.storage.iloc[1] - observations.storage.iloc[0]
    ) / (900.0 * UnitSystem.us_customary().flow_to_volume_per_second)
    assert prepared.initial_state[1] == pytest.approx(expected_initial_inflow)


def test_irregular_intervals_use_each_elapsed_seconds() -> None:
    observations = _frame(count=4, irregular=True)
    prepared = _prepared(observations)

    np.testing.assert_allclose(prepared.elapsed_seconds, [300.0, 900.0, 1_800.0])
    expected_initial_inflow = observations.outflow.iloc[0] + (
        observations.storage.iloc[1] - observations.storage.iloc[0]
    ) / (300.0 * UnitSystem.us_customary().flow_to_volume_per_second)
    assert prepared.initial_state[1] == pytest.approx(expected_initial_inflow)


def test_initialization_interval_uses_timedelta_between_selected_observations() -> None:
    observations = _frame(count=4, irregular=True)
    observations.loc[observations.index[1], "storage"] = np.nan
    prepared = _prepared(observations)

    expected_initial_inflow = observations.outflow.iloc[0] + (
        observations.storage.iloc[2] - observations.storage.iloc[0]
    ) / (1_200.0 * UnitSystem.us_customary().flow_to_volume_per_second)
    assert prepared.initial_state[1] == pytest.approx(expected_initial_inflow)


def test_resolution_equivalent_indexes_produce_equivalent_tuning_results() -> None:
    observations = _frame(count=24)
    equivalent = observations.copy()
    equivalent.index = equivalent.index.as_unit("us")

    nanosecond_result = tune_inflow_model(
        observations, reservoir_id="lexington", max_evaluations=12
    )
    microsecond_result = tune_inflow_model(
        equivalent, reservoir_id="lexington", max_evaluations=12
    )

    assert nanosecond_result.parameters == microsecond_result.parameters
    assert nanosecond_result.score == microsecond_result.score
    np.testing.assert_allclose(
        nanosecond_result.config.q, microsecond_result.config.q
    )
    np.testing.assert_allclose(
        nanosecond_result.config.p0, microsecond_result.config.p0
    )
    assert nanosecond_result.config.smoothing_lag == timedelta(seconds=5_400)


def test_cadence_and_smoothing_lag_are_six_15_minute_intervals() -> None:
    result = tune_inflow_model(_frame(), reservoir_id="lexington", max_evaluations=8)

    assert result.diagnostics["median_interval_seconds"] == pytest.approx(900.0)
    assert result.diagnostics["data_interval"]["median_seconds"] == pytest.approx(
        900.0
    )
    assert result.config.smoothing_lag.total_seconds() == pytest.approx(5_400.0)


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
