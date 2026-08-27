import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from applications import run_bayesian_tuner


def test_split_validation_windows_reject_short_or_naive_indexes() -> None:
    with pytest.raises(ValueError, match="at least three observations"):
        run_bayesian_tuner.split_validation_windows(
            pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        run_bayesian_tuner.split_validation_windows(
            pd.date_range("2025-01-01", periods=3, freq="h")
        )


def test_runner_rejects_invalid_dates_before_loading_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="DATA_START must be timezone-aware"):
        run_bayesian_tuner.run_bayesian_tuner(
            project_root=tmp_path,
            data_start="2025-01-01",
            data_end="2025-01-02T00:00:00Z",
        )
    with pytest.raises(ValueError, match="DATA_END must be after"):
        run_bayesian_tuner.run_bayesian_tuner(
            project_root=tmp_path,
            data_start="2025-01-02T00:00:00Z",
            data_end="2025-01-01T00:00:00Z",
        )


def test_runner_rejects_unsupported_reservoir(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported reservoir"):
        run_bayesian_tuner.run_bayesian_tuner(
            project_root=tmp_path,
            reservoir="Unknown",
            data_start="2025-01-01T00:00:00Z",
            data_end="2025-01-02T00:00:00Z",
        )


def test_runner_preserves_current_tuner_inputs_and_exports_trial_ids(
    monkeypatch, capsys, tmp_path
) -> None:
    index = pd.date_range("2025-01-01", periods=144, freq="h", tz="UTC")
    observations = pd.DataFrame(
        {
            "storage": np.linspace(100.0, 200.0, len(index)),
            "outflow": np.full(len(index), 4.0),
        },
        index=index,
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        run_bayesian_tuner,
        "sources_for_reservoir",
        lambda project_root, reservoir: (project_root, reservoir),
    )

    def fake_prepare(sources, *, start, end, asof_tolerance):
        calls.update(
            sources=sources,
            start=start,
            end=end,
            asof_tolerance=asof_tolerance,
        )
        return SimpleNamespace(observations=observations)

    monkeypatch.setattr(run_bayesian_tuner, "prepare_reservoir_data", fake_prepare)
    config = run_bayesian_tuner.build_base_config("Lexington")

    def fake_tune(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            selected_config=config,
            selected_parameters={
                "q_storage": 0.002,
                "q_inflow": 25.0 / 3600.0,
                "q_outflow": 0.02,
                "r_storage": 0.25,
                "r_outflow": 4.0,
            },
            selected_objective=0.02,
            selection_threshold=0.02,
            selection_reason="calibrated competitive trial",
            competitive_trial_ids=(2,),
            warnings=(),
            timing_seconds={"total_seconds": 0.1},
            candidate_summary=pd.DataFrame({"trial_id": [2], "selected": [True]}),
            window_diagnostics=pd.DataFrame({"window": ["validation-1"]}),
            proxy_diagnostics=pd.DataFrame(),
        )

    monkeypatch.setattr(run_bayesian_tuner, "tune_inflow_noise_bayesian", fake_tune)
    output_path = tmp_path / "bayesian-report.json"
    result = run_bayesian_tuner.run_bayesian_tuner(
        project_root=".",
        reservoir="Lexington",
        data_start="2025-01-01T00:00:00-08:00",
        data_end="2025-01-07T00:00:00-08:00",
        output_path=output_path,
    )

    assert result.competitive_trial_ids == (2,)
    assert calls["start"] == pd.Timestamp("2025-01-01T08:00:00Z")
    assert calls["end"] == pd.Timestamp("2025-01-07T08:00:00Z")
    assert calls["inflow_increment_sd_seeds"] == (
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
    )
    assert len(calls["validation_windows"]) == 3
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["selection"]["competitive_trial_ids"] == [2]
    assert exported["selection"]["selected_objective"] == 0.02
    assert "detailed_regime_report" not in exported
    assert "forecast_horizon_report" not in exported
    assert "competitive_trial_ids" in exported["selection"]
    assert str(output_path.resolve()) in capsys.readouterr().out
