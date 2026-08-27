import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))

from applications.preparation import ReservoirSources, prepare_reservoir_data


def _write_source(path: Path, index: pd.DatetimeIndex, values: list[float]) -> None:
    pd.DataFrame({"ISO 8601 UTC": index, "Value": values}).to_csv(path, index=False)


def test_spillway_unknown_is_not_silently_zero(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    _write_source(tmp_path / "storage.csv", index, [100, 101, 102, 103])
    _write_source(tmp_path / "outlet.csv", index, [10, 10, 10, 10])
    _write_source(tmp_path / "upstream.csv", index, [4, 4, 4, 4])
    _write_source(tmp_path / "spillway.csv", index, [np.nan, 5, -1, 2])
    prepared = prepare_reservoir_data(
        ReservoirSources(
            storage=tmp_path / "storage.csv",
            outlet=tmp_path / "outlet.csv",
            upstream=tmp_path / "upstream.csv",
            spillway=tmp_path / "spillway.csv",
        ),
        start=index[0],
        end=index[-1],
    )
    assert prepared.observations.loc[index[1], "outflow"] == 15
    assert np.isnan(prepared.observations.loc[index[2], "outflow"])
    assert prepared.observations.loc[index[3], "outflow"] == 12
    assert int(prepared.window_audit.loc[0, "spillway_unknown"]) == 1
    assert int(prepared.window_audit.loc[0, "spillway_invalid"]) == 1
    assert int(prepared.window_audit.loc[0, "spillway_used"]) == 2


def test_missing_spillway_file_preserves_outlet_only_behavior(tmp_path: Path) -> None:
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    _write_source(tmp_path / "storage.csv", index, [100, 101, 102])
    _write_source(tmp_path / "outlet.csv", index, [10, 11, 12])
    _write_source(tmp_path / "upstream.csv", index, [4, 4, 4])
    prepared = prepare_reservoir_data(
        ReservoirSources(
            storage=tmp_path / "storage.csv",
            outlet=tmp_path / "outlet.csv",
            upstream=tmp_path / "upstream.csv",
            spillway=None,
            combine_spillway=False,
        ),
        start=index[0],
        end=index[-1],
    )
    np.testing.assert_allclose(prepared.observations["outflow"], [10, 11, 12])
    assert prepared.window_audit.loc[0, "outflow_definition"] == (
        "downstream discharge only; spillway unavailable in requested data"
    )
