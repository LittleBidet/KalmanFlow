import io
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from applications import run_noise_tuner
from kalmanflow import NoiseTuningProgress


def test_split_validation_windows_reject_short_or_naive_indexes() -> None:
    with pytest.raises(ValueError, match="at least 6 observations"):
        run_noise_tuner.split_validation_windows(
            pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        run_noise_tuner.split_validation_windows(
            pd.date_range("2025-01-01", periods=6, freq="h")
        )


def test_window_capacity_uses_second_finite_storage_for_warmup() -> None:
    index = pd.date_range("2025-01-01", periods=288, freq="h", tz="UTC")
    storage = np.arange(288.0)
    storage[1:20] = np.nan
    observations = pd.DataFrame(
        {"storage": storage, "outflow": 4.0},
        index=index,
    )
    windows = run_noise_tuner.split_validation_windows(index)

    with pytest.raises(ValueError, match="validation-1 has 4"):
        run_noise_tuner._validate_window_capacity(
            observations,
            windows,
            run_noise_tuner.build_evaluation_settings(),
        )


def test_runner_rejects_invalid_dates_before_loading_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="DATA_START must be timezone-aware"):
        run_noise_tuner.run_noise_tuner(
            project_root=tmp_path,
            data_start="2025-01-01",
            data_end="2025-01-02T00:00:00Z",
        )
    with pytest.raises(ValueError, match="DATA_END must be after"):
        run_noise_tuner.run_noise_tuner(
            project_root=tmp_path,
            data_start="2025-01-02T00:00:00Z",
            data_end="2025-01-01T00:00:00Z",
        )


def test_runner_rejects_unsupported_reservoir(tmp_path) -> None:
    with pytest.raises(ValueError, match="Unsupported reservoir"):
        run_noise_tuner.run_noise_tuner(
            project_root=tmp_path,
            reservoir="Unknown",
            data_start="2025-01-01T00:00:00Z",
            data_end="2025-01-02T00:00:00Z",
        )


def test_runner_rejects_identical_resolved_output_paths(tmp_path) -> None:
    output_path = tmp_path / "same.json"
    with pytest.raises(ValueError, match="must be different files"):
        run_noise_tuner.run_noise_tuner(
            project_root=tmp_path,
            output_path=output_path,
            report_output_path=output_path,
        )


def test_json_export_replaces_nonfinite_values_with_null(tmp_path) -> None:
    output_path = tmp_path / "nonfinite.json"

    run_noise_tuner._write_json(
        {"nan": np.nan, "positive_infinity": np.inf}, output_path
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "nan": None,
        "positive_infinity": None,
    }


def test_runner_preserves_current_tuner_inputs_and_exports_two_artifacts(
    monkeypatch, capsys, tmp_path
) -> None:
    index = pd.date_range("2025-01-01", periods=288, freq="h", tz="UTC")
    observations = pd.DataFrame(
        {
            "storage": np.linspace(100.0, 200.0, len(index)),
            "outflow": np.full(len(index), 4.0),
        },
        index=index,
    )
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        run_noise_tuner,
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

    monkeypatch.setattr(run_noise_tuner, "prepare_reservoir_data", fake_prepare)
    config = run_noise_tuner.build_base_config("Lexington")

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
            base_objective=0.5,
            improvement_standard_error=0.01,
            identifiability=pd.DataFrame(),
            storage_noise=pd.DataFrame(),
            warnings=(),
            timing_seconds={"total_seconds": 0.1},
            candidate_summary=pd.DataFrame(
                {"candidate": ["search-1"], "selected": [True]}
            ),
            window_diagnostics=pd.DataFrame({"window": ["validation-1"]}),
            proxy_diagnostics=pd.DataFrame(),
        )

    monkeypatch.setattr(run_noise_tuner, "tune_inflow_noise", fake_tune)
    output_path = tmp_path / "noise-config.json"
    report_output_path = tmp_path / "noise-report.json"
    result = run_noise_tuner.run_noise_tuner(
        project_root=tmp_path,
        reservoir="Lexington",
        data_start="2025-01-01T00:00:00-08:00",
        data_end="2025-01-13T00:00:00-08:00",
        output_path=output_path,
        report_output_path=report_output_path,
    )

    default_result = run_noise_tuner.run_noise_tuner(
        project_root=tmp_path,
        reservoir="Lexington",
        data_start="2025-01-01T00:00:00-08:00",
        data_end="2025-01-13T00:00:00-08:00",
    )

    assert result.selected_objective == default_result.selected_objective == 0.02
    assert calls["start"] == pd.Timestamp("2025-01-01T08:00:00Z")
    assert calls["end"] == pd.Timestamp("2025-01-13T08:00:00Z")
    assert callable(calls["progress"])
    assert calls["inflow_increment_sd_seeds"] == (
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
    )
    assert len(calls["validation_windows"]) == 6
    exported_config = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported_config["reservoir_id"] == "lexington"
    assert exported_config["configuration_version"] == config.configuration_version
    assert exported_config["metadata"]["status"] == "reviewed base"
    assert exported_config["q"] == config.q.tolist()
    assert exported_config["r"] == config.r.tolist()
    assert exported_config["p0"] == config.p0.tolist()
    assert exported_config["smoothing_lag_seconds"] == 21600.0
    assert "initialization_strategy" not in exported_config
    assert "inflow_units" not in exported_config
    assert exported_config["unit_system"]["flow_to_volume_per_second"] > 0.0
    assert "candidate_summary" not in exported_config

    exported_report = json.loads(report_output_path.read_text(encoding="utf-8"))
    assert exported_report["selection"]["selected_objective"] == 0.02
    assert exported_report["selection"]["base_objective"] == 0.5
    assert "detailed_regime_report" not in exported_report
    assert "forecast_horizon_report" not in exported_report
    assert "identifiability" in exported_report
    assert exported_report["storage_noise"] == []
    assert "selected_config" not in exported_report
    assert exported_report["selected_config_reference"]["reservoir_id"] == "lexington"
    assert exported_report["selected_config_reference"]["path"] == str(
        output_path.resolve()
    )
    default_output_path = (
        tmp_path
        / "Outputs"
        / "noise_tuning"
        / "lexington-2026-11-ml-noise-candidate.json"
    )
    assert default_output_path.is_file()
    assert "candidate_summary" not in json.loads(
        default_output_path.read_text(encoding="utf-8")
    )
    assert list(default_output_path.parent.glob("*.json")) == [default_output_path]
    output = capsys.readouterr().out
    assert str(output_path.resolve()) in output
    assert str(report_output_path.resolve()) in output


class _FakeStream(io.StringIO):
    def __init__(self, interactive: bool, encoding: str | None = None) -> None:
        super().__init__()
        self.interactive = interactive
        self.stream_encoding = encoding
        self.flushes = 0

    @property
    def encoding(self) -> str | None:  # type: ignore[override]
        return self.stream_encoding

    def isatty(self) -> bool:
        return self.interactive

    def flush(self) -> None:
        self.flushes += 1
        super().flush()


def _update(
    phase: str,
    evaluations: int,
    limit: int = 20,
    best: float = 4.5,
    number: int = 1,
    count: int = 4,
) -> NoiseTuningProgress:
    return NoiseTuningProgress(
        phase=phase,
        phase_number=number,
        phase_count=count,
        evaluations=evaluations,
        evaluation_limit=limit,
        best_objective=best,
        elapsed_seconds=3725.0,
    )


def _screen(output: str) -> list[str]:
    """Replay cursor-up, carriage-return and clear-line codes into screen rows."""

    rows = [""]
    row = column = 0
    for token in re.split(r"(\x1b\[1A|\x1b\[K|\r|\n)", output):
        if token == "\x1b[1A":
            row -= 1
        elif token == "\r":
            column = 0
        elif token == "\n":
            row, column = row + 1, 0
            if row == len(rows):
                rows.append("")
        elif token == "\x1b[K":
            rows[row] = rows[row][:column]
        elif token:
            line = rows[row].ljust(column)
            rows[row] = line[:column] + token + line[column + len(token) :]
            column += len(token)
    return [line for line in rows if line]


def _bar(line: str) -> str:
    return line.split("|")[1]


def test_total_and_phase_bars_fill_left_to_right_in_a_terminal() -> None:
    stream = _FakeStream(interactive=True)
    printer = run_noise_tuner._ProgressPrinter(stream)
    printer.close()  # Nothing drawn yet, so nothing to end.
    assert stream.getvalue() == ""

    printer(_update("screening", 0, best=np.inf))
    total, phase = _screen(stream.getvalue())
    assert total.startswith("total")
    assert _bar(total) == "░" * 30
    assert _bar(phase) == "░" * 30

    printer(_update("screening", 10))
    total, phase = _screen(stream.getvalue())
    # Half of the first of four phases is one eighth of the run.
    assert _bar(phase) == "█" * 15 + "░" * 15
    assert _bar(total) == "█" * 4 + "░" * 26
    assert "12%  phase 1/4" in total  # 12.5% rounds to even
    assert phase.endswith("1:02:05")


def test_finished_phases_stay_complete_above_the_live_bars() -> None:
    stream = _FakeStream(interactive=True)
    printer = run_noise_tuner._ProgressPrinter(stream)
    printer(_update("search-1", 3, limit=201, number=2))
    printer(_update("search-2", 0, limit=201, number=3))
    finished, total, live = _screen(stream.getvalue())
    # The early-stopping search is redrawn as full, and the total has moved
    # past both finished phases.
    assert finished.startswith("search-1") and _bar(finished) == "█" * 30
    assert "  3/3" in finished
    assert "50%  phase 3/4" in total
    assert live.startswith("search-2")

    printer.close()
    rows = _screen(stream.getvalue())
    assert [row.split()[0] for row in rows] == ["search-1", "search-2", "total"]
    assert _bar(rows[-1]) == "█" * 30
    assert rows[-1].endswith("100%  done")
    assert stream.getvalue().endswith("\n")
    assert stream.flushes == 3


def test_progress_bars_write_periodic_lines_to_a_log() -> None:
    stream = _FakeStream(interactive=False)
    printer = run_noise_tuner._ProgressPrinter(stream)
    for evaluations in range(12):
        printer(_update("search-1", evaluations, number=2))
    printer(_update("identifiability", 0, number=4))
    printer.close()

    lines = stream.getvalue().splitlines()
    counts = [line.split("|")[2].split()[0] for line in lines]
    assert counts == ["0/20", "10/20", "0/20"]
    assert lines[1].endswith("total 38%")
    assert lines[-1].startswith("identifiability")
    assert "\r" not in stream.getvalue()


def test_progress_bars_fall_back_to_ascii() -> None:
    stream = _FakeStream(interactive=False, encoding="ascii")
    printer = run_noise_tuner._ProgressPrinter(stream)
    printer(_update("screening", 10))
    assert "|###############---------------|" in stream.getvalue()


def test_progress_printer_defaults_to_stderr() -> None:
    assert run_noise_tuner._ProgressPrinter()._stream is sys.stderr
