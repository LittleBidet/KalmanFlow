import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import run_tuner


def test_split_validation_windows_are_contiguous_and_balanced() -> None:
    index = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")

    windows = run_tuner.split_validation_windows(index)
    masks = [window.mask(index) for window in windows]

    assert [int(mask.sum()) for mask in masks] == [4, 3, 3]
    assert np.sum(np.asarray(masks), axis=0).tolist() == [1] * len(index)
    assert windows[0].start == index[0]
    assert windows[1].start == index[4]
    assert windows[2].start == index[7]
    assert windows[-1].end > index[-1]


def test_split_validation_windows_rejects_short_or_naive_indexes() -> None:
    with pytest.raises(ValueError, match="at least three observations"):
        run_tuner.split_validation_windows(
            pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        run_tuner.split_validation_windows(
            pd.date_range("2025-01-01", periods=3, freq="h")
        )


def test_run_tuner_rejects_invalid_dates_before_loading_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="DATA_START must be timezone-aware"):
        run_tuner.run_tuner(
            project_root=tmp_path,
            data_start="2025-01-01",
            data_end="2025-01-02T00:00:00Z",
        )

    with pytest.raises(ValueError, match="DATA_END must be after"):
        run_tuner.run_tuner(
            project_root=tmp_path,
            data_start="2025-01-02T00:00:00Z",
            data_end="2025-01-01T00:00:00Z",
        )


def test_run_tuner_rejects_unsupported_reservoir(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported reservoir"):
        run_tuner.run_tuner(
            project_root=tmp_path,
            reservoir="Unknown",
            data_start="2025-01-01T00:00:00Z",
            data_end="2025-01-02T00:00:00Z",
        )


def test_run_tuner_prepares_segment_and_prints_tuning_tables(
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
        run_tuner,
        "sources_for_reservoir",
        lambda project_root, reservoir: (project_root, reservoir),
    )

    def fake_prepare(sources, *, start, end, asof_tolerance):
        calls["sources"] = sources
        calls["start"] = start
        calls["end"] = end
        calls["asof_tolerance"] = asof_tolerance
        return SimpleNamespace(observations=observations)

    monkeypatch.setattr(run_tuner, "prepare_reservoir_data", fake_prepare)

    def fake_tune(**kwargs):
        calls.update(kwargs)
        return SimpleNamespace(
            selected_config=run_tuner.build_base_config("Lexington"),
            selected_prior_hourly_increment_sd=5.0,
            selected_q_inflow=25.0 / 3600.0,
            selection_threshold=0.02,
            selection_reason="smallest competitive candidate",
            competitive_candidates=(25.0 / 3600.0,),
            warnings=(),
            timing_seconds={"total_seconds": 0.1},
            candidate_summary=pd.DataFrame({"q_inflow": [25.0 / 3600.0]}),
            window_diagnostics=pd.DataFrame({"window": ["validation-1"]}),
            regime_diagnostics=pd.DataFrame(),
            horizon_diagnostics=pd.DataFrame(),
            r_sensitivity=None,
        )

    monkeypatch.setattr(run_tuner, "tune_inflow_process_noise", fake_tune)

    output_path = tmp_path / "tuning-report.json"
    result = run_tuner.run_tuner(
        project_root=".",
        reservoir="Lexington",
        data_start="2025-01-01T00:00:00-08:00",
        data_end="2025-01-07T00:00:00-08:00",
        output_path=output_path,
    )

    assert result.selected_prior_hourly_increment_sd == 5.0
    assert calls["sources"] == (run_tuner.PROJECT_ROOT, "Lexington")
    assert calls["start"] == pd.Timestamp("2025-01-01T08:00:00Z")
    assert calls["end"] == pd.Timestamp("2025-01-07T08:00:00Z")
    windows = calls["validation_windows"]
    assert [int(window.mask(index).sum()) for window in windows] == [48, 48, 48]
    assert calls["candidate_prior_hourly_increment_sd"] == (
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
    )
    output = capsys.readouterr().out
    assert "Selected q_inflow" in output
    assert "Candidate summary" in output
    assert "Per-window summary" in output
    assert str(output_path.resolve()) in output
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["selection"]["selected_q_inflow"] == pytest.approx(25.0 / 3600.0)
    assert exported["selected_config"]["reservoir_name"] == "Lexington"
