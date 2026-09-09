"""Application-specific preparation of Aquarius reservoir source series.

This module parses the checked-in Aquarius exports used by the offline Bayesian
calibration workflow. It intentionally lives outside the generalizable :mod:`kalmanflow`
package, whose APIs accept already-cleaned and aligned observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReservoirSources:
    """Aquarius exports and the reservoir-specific outflow definition.

    ``spillway`` is optional because older exports may not contain a spillway
    record at all.  When a spillway file exists, finite values are included in
    outflow; missing values are retained as unknown rather than replaced by
    zero.
    """

    storage: Path
    outlet: Path
    spillway: Path | None
    upstream: Path
    combine_spillway: bool = True


@dataclass(frozen=True)
class PreparedReservoirData:
    """Clean model inputs, aligned diagnostics, and cleaning audit tables."""

    observations: pd.DataFrame
    diagnostics: pd.DataFrame
    source_audit: pd.DataFrame
    window_audit: pd.DataFrame


def sources_for_reservoir(project_root: Path, reservoir: str) -> ReservoirSources:
    """Return the checked-in Aquarius source files for a supported reservoir."""

    data_root = Path(project_root) / "Reservoirs"
    source_options = {
        "lexington": ReservoirSources(
            storage=data_root / "Lexington" / "Total_Storage.csv",
            outlet=data_root / "Lexington" / "Discharge.csv",
            upstream=data_root / "Lexington" / "Upstream.csv",
            spillway=data_root / "Lexington" / "Spillway_Flow.csv",
            combine_spillway=True,
        ),
        "chesbro": ReservoirSources(
            storage=data_root / "Chesbro" / "Total_Storage.csv",
            outlet=data_root / "Chesbro" / "Discharge.csv",
            upstream=data_root / "Chesbro" / "Upstream.csv",
            spillway=data_root / "Chesbro" / "Spillway_Flow.csv",
            combine_spillway=True,
        ),
    }
    key = str(reservoir).casefold()
    if key not in source_options:
        supported = ", ".join(name.title() for name in source_options)
        raise ValueError(f"Unsupported reservoir {reservoir!r}; choose {supported}")
    sources = source_options[key]
    required = _source_paths(sources)
    missing = [
        path
        for name, path in required.items()
        if name != "spillway" and not path.is_file()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing reservoir source files: {missing_text}")
    # A missing spillway is a legitimate historical case.  Keep that fact in
    # the source object so preparation can preserve outlet-only behavior.
    if sources.spillway is not None and not sources.spillway.is_file():
        sources = ReservoirSources(
            storage=sources.storage,
            outlet=sources.outlet,
            upstream=sources.upstream,
            spillway=None,
            combine_spillway=False,
        )
    return sources


def read_aquarius_series(path: Path, name: str) -> tuple[pd.Series, dict[str, object]]:
    """Read one Aquarius CSV into a numeric, duplicate-free UTC series."""

    path = Path(path)
    metadata: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            metadata[key.strip()] = value.strip()
    raw = pd.read_csv(path, comment="#")
    required = {"ISO 8601 UTC", "Value"}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")
    parsed = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw["ISO 8601 UTC"], utc=True, errors="coerce"),
            name: pd.to_numeric(raw["Value"], errors="coerce"),
        }
    )
    timestamp_parse_failures = int(parsed["timestamp"].isna().sum())
    parsed = parsed.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    duplicates_removed = int(parsed["timestamp"].duplicated(keep="last").sum())
    parsed = parsed.drop_duplicates("timestamp", keep="last")
    series = parsed.set_index("timestamp")[name].sort_index().rename(name)
    intervals = series.index.to_series().diff().dt.total_seconds()
    audit = {
        "name": name,
        "path": str(path),
        "source_rows": len(raw),
        "clean_rows": len(series),
        "timestamp_parse_failures": timestamp_parse_failures,
        "duplicates_removed": duplicates_removed,
        "finite_value_fraction": float(series.notna().mean()),
        "first_utc": series.index.min(),
        "last_utc": series.index.max(),
        "median_interval_min": float(intervals.median() / 60.0),
        "gaps_over_20min": int((intervals > 20 * 60).sum()),
        "aquarius_identifier": metadata.get("Time-series identifier", ""),
        "aquarius_location": metadata.get("Location", ""),
        "aquarius_units": metadata.get("Value units", ""),
    }
    return series, audit


def prepare_reservoir_data(
    sources: ReservoirSources,
    *,
    start: object,
    end: object,
    asof_tolerance: object = "20min",
) -> PreparedReservoirData:
    """Causally align storage, discharge, spillway, and upstream series.

    Storage timestamps define the output clock.  Every source is matched to
    the most recent available value, bounded by ``asof_tolerance``.  If a
    spillway record is present but missing at a timestamp, combined outflow is
    missing at that timestamp; it is never silently interpreted as zero.
    """

    start_utc = _as_utc_timestamp(start, "start")
    end_utc = _as_utc_timestamp(end, "end")
    if start_utc > end_utc:
        raise ValueError("start must be earlier than or equal to end")
    tolerance = pd.Timedelta(asof_tolerance)
    if tolerance <= pd.Timedelta(0):
        raise ValueError("asof_tolerance must be positive")
    cleaned: dict[str, pd.Series] = {}
    audits: list[dict[str, object]] = []
    for name, path in _source_paths(sources).items():
        if path is None:
            continue
        cleaned[name], audit = read_aquarius_series(path, name)
        audits.append(audit)
    storage = cleaned["storage"]
    clock = storage.index
    outlet = _backward_asof(clock, cleaned["outlet"], "outlet_discharge", tolerance)
    upstream = _backward_asof(clock, cleaned["upstream"], "upstream_flow", tolerance)
    if "spillway" in cleaned:
        spillway = _backward_asof(
            clock, cleaned["spillway"], "spillway_flow", tolerance
        )
    else:
        spillway = pd.Series(np.nan, index=clock, name="spillway_flow", dtype=float)
    diagnostics = pd.concat(
        [storage.rename("storage"), outlet, spillway, upstream], axis=1
    ).sort_index()
    diagnostics = diagnostics[~diagnostics.index.duplicated(keep="last")]
    window_mask = (diagnostics.index >= start_utc) & (diagnostics.index <= end_utc)
    finite_spillway = np.isfinite(diagnostics["spillway_flow"]) & (
        diagnostics["spillway_flow"] >= 0.0
    )
    has_spillway_in_window = bool(finite_spillway.loc[window_mask].any())
    if sources.combine_spillway and "spillway" in cleaned and has_spillway_in_window:
        diagnostics["outflow"] = diagnostics["outlet_discharge"].where(
            finite_spillway,
            np.nan,
        ) + diagnostics["spillway_flow"].where(finite_spillway, np.nan)
        diagnostics["spillway_used"] = finite_spillway
        outflow_definition = (
            "outlet discharge + finite spillway flow; unknown spillway is missing"
        )
    else:
        diagnostics["outflow"] = diagnostics["outlet_discharge"]
        diagnostics["spillway_used"] = False
        outflow_definition = (
            "downstream discharge only; spillway unavailable in requested data"
        )
    diagnostics["spillway_available"] = finite_spillway
    diagnostics["spillway_unknown"] = ~finite_spillway
    diagnostics = diagnostics.loc[window_mask].replace([np.inf, -np.inf], np.nan)
    first_joint = diagnostics[["storage", "outflow"]].dropna().index.min()
    if pd.isna(first_joint):
        raise ValueError("No joint finite storage/outflow observation in the window")
    diagnostics = diagnostics.loc[first_joint:].copy()
    observations = diagnostics[["storage", "outflow"]].copy()
    _validate_pandas_api_frame(observations)
    intervals = observations.index.to_series().diff().dt.total_seconds()
    window_audit = pd.DataFrame(
        [
            {
                "requested_start": start_utc,
                "requested_end": end_utc,
                "actual_start": observations.index.min(),
                "actual_end": observations.index.max(),
                "rows": len(observations),
                "storage_missing": int(observations["storage"].isna().sum()),
                "outflow_missing": int(observations["outflow"].isna().sum()),
                "upstream_missing": int(diagnostics["upstream_flow"].isna().sum()),
                "spillway_unknown": int(diagnostics["spillway_unknown"].sum()),
                "spillway_invalid": int(
                    (
                        np.isfinite(diagnostics["spillway_flow"])
                        & (diagnostics["spillway_flow"] < 0.0)
                    ).sum()
                ),
                "spillway_used": int(diagnostics["spillway_used"].sum()),
                "median_interval_min": float(intervals.median() / 60.0),
                "intervals_over_20min": int((intervals > 20 * 60).sum()),
                "asof_tolerance": tolerance,
                "outflow_definition": outflow_definition,
            }
        ]
    )
    source_audit = pd.DataFrame(audits).set_index("name")
    return PreparedReservoirData(observations, diagnostics, source_audit, window_audit)


def _source_paths(sources: ReservoirSources) -> dict[str, Path | None]:
    return {
        "storage": Path(sources.storage),
        "outlet": Path(sources.outlet),
        "spillway": None if sources.spillway is None else Path(sources.spillway),
        "upstream": Path(sources.upstream),
    }


def _as_utc_timestamp(value: object, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _backward_asof(
    clock: pd.DatetimeIndex,
    source: pd.Series,
    name: str,
    tolerance: pd.Timedelta,
) -> pd.Series:
    left = pd.DataFrame({"timestamp": clock})
    right = source.rename(name).rename_axis("timestamp").reset_index()
    aligned = pd.merge_asof(
        left.sort_values("timestamp"),
        right.sort_values("timestamp", kind="stable"),
        on="timestamp",
        direction="backward",
        tolerance=tolerance,
    )
    return aligned.set_index("timestamp")[name].rename(name)


def _validate_pandas_api_frame(observations: pd.DataFrame) -> None:
    if observations.empty:
        raise ValueError("The requested window produced no observations")
    if observations.index.tz is None:
        raise AssertionError("pandas_api input index must be timezone-aware")
    if not observations.index.is_monotonic_increasing:
        raise AssertionError("pandas_api input index must be increasing")
    if observations.index.has_duplicates:
        raise AssertionError("pandas_api input index must be duplicate-free")
    if len(observations) < 2 or observations["storage"].notna().sum() < 2:
        raise ValueError("At least two finite storage observations are required")
