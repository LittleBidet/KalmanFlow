"""Focused edge-case coverage for the application-layer helpers.

The repository's regular tests exercise the normal data path.  These tests
cover the validation and error paths that are part of the public contracts as
well, so coverage does not hide untested input handling.
"""

from __future__ import annotations

import importlib
import json
import runpy
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications import preparation, run_bayesian_tuner  # noqa: E402


def _write_aquarius(path: Path, index: pd.DatetimeIndex, values: list[object]) -> None:
    pd.DataFrame({"ISO 8601 UTC": index, "Value": values}).to_csv(path, index=False)


def test_sources_for_reservoir_validates_supported_files_and_optional_spillway(
    tmp_path: Path,
) -> None:
    reservoir_root = tmp_path / "Reservoirs" / "Lexington"
    reservoir_root.mkdir(parents=True)
    for name in ("Total_Storage.csv", "Discharge.csv", "Upstream.csv"):
        (reservoir_root / name).touch()

    sources = preparation.sources_for_reservoir(tmp_path, "LEXINGTON")
    assert sources.spillway is None
    assert sources.combine_spillway is False

    (reservoir_root / "Spillway_Flow.csv").touch()
    sources = preparation.sources_for_reservoir(tmp_path, "LEXINGTON")
    assert sources.spillway is not None
    assert sources.combine_spillway is True

    with pytest.raises(ValueError, match="Unsupported reservoir"):
        preparation.sources_for_reservoir(tmp_path, "unknown")

    (reservoir_root / "Discharge.csv").unlink()
    with pytest.raises(FileNotFoundError, match="Missing reservoir source files"):
        preparation.sources_for_reservoir(tmp_path, "lexington")


def test_read_aquarius_series_parses_metadata_duplicates_and_bad_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "series.csv"
    path.write_text(
        "# Time-series identifier: TS-1\n"
        "# Location: Test reservoir\n"
        "# Value units: cfs\n"
        "# this line has no colon\n"
        "ISO 8601 UTC,Value\n"
        "2025-01-01T01:00:00Z,2\n"
        "2025-01-01T00:00:00Z,1\n"
        "2025-01-01T01:00:00Z,3\n"
        "not-a-time,not-a-number\n",
        encoding="utf-8",
    )
    series, audit = preparation.read_aquarius_series(path, "flow")
    assert list(series.index) == list(
        pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T01:00:00Z"])
    )
    assert series.iloc[-1] == 3.0
    assert audit["timestamp_parse_failures"] == 1
    assert audit["duplicates_removed"] == 1
    assert audit["aquarius_identifier"] == "TS-1"
    assert audit["aquarius_location"] == "Test reservoir"
    assert audit["aquarius_units"] == "cfs"

    bad = tmp_path / "bad.csv"
    bad.write_text("ISO 8601 UTC,Wrong\n2025-01-01T00:00:00Z,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing columns"):
        preparation.read_aquarius_series(bad, "flow")


def test_preparation_rejects_invalid_windows_and_no_joint_observation(
    tmp_path: Path,
) -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    for name, values in (
        ("storage", [np.nan, np.nan]),
        ("outlet", [np.nan, np.nan]),
        ("upstream", [1.0, 1.0]),
    ):
        _write_aquarius(tmp_path / f"{name}.csv", index, values)
    sources = preparation.ReservoirSources(
        storage=tmp_path / "storage.csv",
        outlet=tmp_path / "outlet.csv",
        spillway=None,
        upstream=tmp_path / "upstream.csv",
        combine_spillway=False,
    )
    with pytest.raises(ValueError, match="start must be earlier"):
        preparation.prepare_reservoir_data(sources, start=index[-1], end=index[0])
    with pytest.raises(ValueError, match="asof_tolerance must be positive"):
        preparation.prepare_reservoir_data(
            sources, start=index[0], end=index[-1], asof_tolerance=0
        )
    with pytest.raises(ValueError, match="start must be timezone-aware"):
        preparation.prepare_reservoir_data(sources, start="2025-01-01", end=index[-1])
    with pytest.raises(ValueError, match="No joint finite"):
        preparation.prepare_reservoir_data(sources, start=index[0], end=index[-1])


def test_preparation_validation_frame_checks_all_invariants() -> None:
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    with pytest.raises(ValueError, match="no observations"):
        preparation._validate_pandas_api_frame(pd.DataFrame(index=index))
    with pytest.raises(AssertionError, match="timezone-aware"):
        preparation._validate_pandas_api_frame(
            pd.DataFrame(
                {"storage": [1.0], "outflow": [1.0]},
                index=pd.date_range("2025-01-01", periods=1, freq="h"),
            )
        )
    with pytest.raises(AssertionError, match="increasing"):
        preparation._validate_pandas_api_frame(
            pd.DataFrame(
                {"storage": [1.0, 2.0], "outflow": [1.0, 1.0]},
                index=index[::-1],
            )
        )
    with pytest.raises(AssertionError, match="duplicate-free"):
        preparation._validate_pandas_api_frame(
            pd.DataFrame(
                {"storage": [1.0, 2.0], "outflow": [1.0, 1.0]},
                index=pd.DatetimeIndex([index[0], index[0]]),
            )
        )
    with pytest.raises(ValueError, match="At least two"):
        preparation._validate_pandas_api_frame(
            pd.DataFrame({"storage": [1.0, np.nan], "outflow": [1.0, 1.0]}, index=index)
        )


def test_tuner_validates_edges_and_serializes_all_supported_values(
    tmp_path: Path,
) -> None:
    index = pd.date_range("2025-01-01", periods=6, freq="h", tz="UTC")
    with pytest.raises(TypeError, match="DatetimeIndex"):
        run_bayesian_tuner.split_validation_windows(list(index))
    with pytest.raises(ValueError, match="increasing and unique"):
        run_bayesian_tuner.split_validation_windows(index[[0, 1, 1, 3, 4, 5]])
    with pytest.raises(ValueError, match="reservoir must not be empty"):
        run_bayesian_tuner.build_base_config("  ")
    with pytest.raises(ValueError, match="valid timestamp"):
        run_bayesian_tuner._timestamp("not-a-date", "when")

    observations = pd.DataFrame({"storage": [np.nan] * 6, "outflow": 1.0}, index=index)
    with pytest.raises(ValueError, match="at least two finite"):
        run_bayesian_tuner._validate_window_capacity(
            observations,
            run_bayesian_tuner.split_validation_windows(index),
            run_bayesian_tuner.build_evaluation_settings(),
        )

    value = run_bayesian_tuner._json_value(
        {
            "timestamp": pd.Timestamp("2025-01-01T00:00:00Z"),
            "duration": timedelta(seconds=2),
            "array": np.array([np.float64(1.5)]),
            "scalar": np.float64(2.5),
            "mapping": {1: (np.int64(3),)},
            "nan": float("nan"),
        }
    )
    assert value["timestamp"] == "2025-01-01T00:00:00+00:00"
    assert value["duration"] == 2.0
    assert value["array"] == [1.5]
    assert value["scalar"] == 2.5
    assert value["mapping"] == {"1": [3]}
    assert value["nan"] is None

    assert run_bayesian_tuner._frame_records(None) == []
    config = run_bayesian_tuner.build_base_config("Chesbro")
    result = SimpleNamespace(
        selected_config=config,
        selected_parameters={},
        selected_objective=float("nan"),
        selection_threshold=float("inf"),
        selection_reason="test",
        competitive_trial_ids=(),
        warnings=(),
        timing_seconds={},
        candidate_summary=pd.DataFrame(),
        window_diagnostics=pd.DataFrame(),
        proxy_diagnostics=pd.DataFrame(),
    )
    report = run_bayesian_tuner.export_bayesian_tuning_report(
        result, tmp_path / "report.json"
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["selected_config_reference"]["reservoir_id"] == "chesbro"
    assert payload["selection"]["selected_objective"] is None

    result.warnings = ("warning",)
    run_bayesian_tuner._print_result(result)


def test_tuner_import_path_setup_and_main_entrypoint(monkeypatch) -> None:
    module_name = "applications.run_bayesian_tuner"
    root_text = str(run_bayesian_tuner.PROJECT_ROOT)
    src_text = str(run_bayesian_tuner.SRC_ROOT)
    original = sys.path[:]
    try:
        sys.path[:] = [
            entry for entry in sys.path if entry not in {root_text, src_text}
        ]
        importlib.reload(sys.modules[module_name])
    finally:
        sys.path[:] = original
        importlib.reload(sys.modules[module_name])

    called: list[bool] = []
    monkeypatch.setattr(
        run_bayesian_tuner,
        "run_bayesian_tuner",
        lambda: called.append(True),
    )
    run_bayesian_tuner.main()
    assert called == [True]


def test_tuner_runner_handles_empty_short_data_and_upstream_proxy(
    monkeypatch, tmp_path: Path
) -> None:
    index = pd.date_range("2025-01-01", periods=144, freq="h", tz="UTC")
    full = pd.DataFrame(
        {"storage": np.linspace(100.0, 200.0, len(index)), "outflow": 4.0},
        index=index,
    )
    monkeypatch.setattr(
        run_bayesian_tuner,
        "sources_for_reservoir",
        lambda root, reservoir: (root, reservoir),
    )
    monkeypatch.setattr(
        run_bayesian_tuner,
        "prepare_reservoir_data",
        lambda *args, **kwargs: SimpleNamespace(observations=pd.DataFrame()),
    )
    with pytest.raises(ValueError, match="produced no observations"):
        run_bayesian_tuner.run_bayesian_tuner(
            project_root=tmp_path,
            data_start=index[0],
            data_end=index[-1],
        )

    monkeypatch.setattr(
        run_bayesian_tuner,
        "prepare_reservoir_data",
        lambda *args, **kwargs: SimpleNamespace(observations=full.iloc[:2]),
    )
    with pytest.raises(ValueError, match="at least three"):
        run_bayesian_tuner.run_bayesian_tuner(
            project_root=tmp_path,
            data_start=index[0],
            data_end=index[-1],
        )

    config = run_bayesian_tuner.build_base_config("Chesbro")
    fake_result = SimpleNamespace(
        selected_config=config,
        selected_parameters={},
        selected_objective=0.1,
        selection_threshold=0.1,
        selection_reason="test",
        competitive_trial_ids=(),
        warnings=(),
        timing_seconds={},
        candidate_summary=pd.DataFrame(),
        window_diagnostics=pd.DataFrame(),
        proxy_diagnostics=pd.DataFrame(),
    )
    calls: dict[str, object] = {}

    def fake_tune(**kwargs):
        calls.update(kwargs)
        return fake_result

    monkeypatch.setattr(
        run_bayesian_tuner,
        "prepare_reservoir_data",
        lambda *args, **kwargs: SimpleNamespace(
            observations=full,
            diagnostics=pd.DataFrame(
                {"upstream_flow": np.ones(len(index))}, index=index
            ),
        ),
    )
    monkeypatch.setattr(run_bayesian_tuner, "tune_inflow_noise_bayesian", fake_tune)
    run_bayesian_tuner.run_bayesian_tuner(
        project_root=tmp_path,
        output_path=tmp_path / "config.json",
        report_output_path=tmp_path / "report.json",
        data_start=index[0],
        data_end=index[-1],
    )
    assert isinstance(calls["upstream_proxy"], pd.Series)
    assert calls["upstream_proxy"].equals(
        pd.Series(np.ones(len(index)), index=index, name="upstream_flow")
    )

    # A diagnostics frame without the optional proxy exercises the runner's
    # normal no-proxy path after the diagnostics frame has been recognized.
    monkeypatch.setattr(
        run_bayesian_tuner,
        "prepare_reservoir_data",
        lambda *args, **kwargs: SimpleNamespace(
            observations=full,
            diagnostics=pd.DataFrame({"other": np.ones(len(index))}, index=index),
        ),
    )
    run_bayesian_tuner.run_bayesian_tuner(
        project_root=tmp_path,
        output_path=tmp_path / "config-no-proxy.json",
        data_start=index[0],
        data_end=index[-1],
    )
    assert calls["upstream_proxy"] is None


def test_notebook_compatibility_import_and_main_script(capsys) -> None:
    importlib.import_module("main")
    prepare_module = importlib.import_module("Notebooks.prepare_reservoir_data")
    assert "PreparedReservoirData" in prepare_module.__all__
    assert prepare_module.ReservoirSources is preparation.ReservoirSources

    namespace = runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    assert "run_inflow_model" in namespace
    result = namespace["result"]
    expected_index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    assert result.index.equals(expected_index)
    assert np.isfinite(result["estimated_inflow"]).all()
    assert np.isfinite(result["revised_inflow"].iloc[:-1]).all()
    assert pd.isna(result["revised_inflow"].iloc[-1])
    assert result.loc[expected_index[2], "estimated_inflow_flag"] == "PREDICTED"
    assert result.loc[expected_index[2], "revised_inflow_flag"] == "PREDICTED"
    assert pd.isna(result.loc[expected_index[-1], "revised_inflow_flag"])
    assert "estimated_inflow" in capsys.readouterr().out


def test_bayesian_tuner_module_script_entrypoint(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    """Run the real file as a script with a deterministic in-memory dataset."""

    index = pd.date_range("2022-10-01", periods=144, freq="h", tz="UTC")
    observations = pd.DataFrame(
        {"storage": np.linspace(100.0, 200.0, len(index)), "outflow": 4.0},
        index=index,
    )
    config = run_bayesian_tuner.build_base_config("Chesbro")
    fake_result = SimpleNamespace(
        selected_config=config,
        selected_parameters={},
        selected_objective=0.1,
        selection_threshold=0.1,
        selection_reason="script test",
        competitive_trial_ids=(),
        warnings=(),
        timing_seconds={},
        candidate_summary=pd.DataFrame(),
        window_diagnostics=pd.DataFrame(),
        proxy_diagnostics=pd.DataFrame(),
    )
    monkeypatch.setattr(
        preparation,
        "sources_for_reservoir",
        lambda root, reservoir: (root, reservoir),
    )
    monkeypatch.setattr(
        preparation,
        "prepare_reservoir_data",
        lambda *args, **kwargs: SimpleNamespace(
            observations=observations,
            diagnostics=pd.DataFrame(
                {"upstream_flow": np.ones(len(index))}, index=index
            ),
        ),
    )
    monkeypatch.setattr(
        importlib.import_module("kalmanflow"),
        "tune_inflow_noise_bayesian",
        lambda **kwargs: fake_result,
    )
    source_path = ROOT / "applications" / "run_bayesian_tuner.py"
    temporary_file = tmp_path / "applications" / source_path.name
    temporary_file.parent.mkdir(parents=True)
    temporary_root = str(tmp_path)
    temporary_src = str(tmp_path / "src")
    source = source_path.read_text(encoding="utf-8")
    # Compile with the real source filename so coverage records the script's
    # entry-point branch against applications/run_bayesian_tuner.py, while
    # __file__ points into pytest's temporary directory for all output paths.
    try:
        exec(
            compile(source, str(source_path), "exec"),
            {"__name__": "__main__", "__file__": str(temporary_file)},
        )
    finally:
        sys.path[:] = [
            entry for entry in sys.path if entry not in {temporary_root, temporary_src}
        ]
    output_path = (
        tmp_path
        / "Outputs"
        / "bayesian_tuning"
        / ("chesbro-2026-08-bayesian-noise-candidate.json")
    )
    assert output_path.is_file()
    assert "Bayesian tuning configuration exported" in capsys.readouterr().out


def test_notebook_compatibility_import_path_branch() -> None:
    preparation_module = importlib.import_module("applications.preparation")
    original = sys.path[:]
    root_text = str(ROOT)
    try:
        sys.path[:] = [entry for entry in sys.path if entry != root_text]
        # The application module is cached, so the compatibility import can
        # exercise its path insertion branch without requiring a second import
        # of the package itself.
        namespace = runpy.run_path(
            str(ROOT / "Notebooks" / "prepare_reservoir_data.py"),
            run_name="coverage_prepare_reservoir_data",
        )
    finally:
        sys.path[:] = original
    assert namespace["ReservoirSources"] is preparation_module.ReservoirSources
