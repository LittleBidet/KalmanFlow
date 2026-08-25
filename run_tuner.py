"""Run the inflow process-noise tuner for one reservoir data segment.

Edit ``RESERVOIR``, ``DATA_START``, and ``DATA_END`` below, then run this file
from the project root. The script writes a JSON tuning report with the proposed
configuration and diagnostics; it does not change an operational configuration.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# The project uses a src layout, so make direct execution from a checkout work
# without requiring an editable package install.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from kalmone import (  # noqa: E402
    InflowTuningSettings,
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    TuningWindow,
    UnitSystem,
    tune_inflow_process_noise,
)
from Notebooks.prepare_reservoir_data import (  # noqa: E402
    prepare_reservoir_data,
    sources_for_reservoir,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


# ========================= USER PARAMETERS =========================
RESERVOIR = "Chesbro"
DATA_START = pd.Timestamp("2022-01-01T00:00:00Z")
DATA_END = pd.Timestamp("2025-01-01T00:00:00Z")
# =====================================================================

ASOF_TOLERANCE = pd.Timedelta("20min")
BASE_CONFIGURATION_VERSION = "reviewed-base-v1"
PROPOSED_CONFIGURATION_VERSION = "2026-08-q-inflow-candidate"

# These are prior standard deviations for one-hour latent inflow increments.
# The tuner converts them to continuous-time q_inflow values.
CANDIDATE_PRIOR_HOURLY_INCREMENT_SD = (2.0, 5.0, 10.0, 20.0, 40.0)

# The remaining values match the defaults used by the process-noise tuning
# notebook and remain fixed while q_inflow is tuned.
Q_STORAGE = 0.002
Q_OUTFLOW = 0.02
R_STORAGE = 0.25
R_OUTFLOW = 4.0
P0_STORAGE = 100.0
P0_INFLOW = 400.0
P0_OUTFLOW = 25.0
SMOOTHING_LAG = timedelta(hours=12)
WARMUP = timedelta(hours=24)
INNOVATION_MAX_LAG = timedelta(hours=24)
FORECAST_HORIZONS = (
    timedelta(hours=1),
    timedelta(hours=3),
    timedelta(hours=6),
)
MIN_SCORED_STORAGE_OBSERVATIONS = 12
PRACTICAL_EQUIVALENCE_TOLERANCE = 0.02
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 20260820
MAX_JITTER_FRACTION = 1e-9
MAX_REGULARIZED_STEPS = 0
R_SENSITIVITY_MULTIPLIERS = (0.75, 1.0, 1.25)
OUTPUT_DIRECTORY = Path("Outputs") / "tuning"


def split_validation_windows(
    index: pd.DatetimeIndex,
) -> tuple[TuningWindow, ...]:
    """Split ``index`` into three contiguous, non-overlapping windows.

    The row counts differ by at most one.  The final end is one nanosecond
    after the final observation because ``TuningWindow`` uses a half-open
    interval.
    """

    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("observations must use a pandas DatetimeIndex")
    if len(index) < 3:
        raise ValueError("at least three observations are required for three windows")
    if index.tz is None:
        raise ValueError("observation timestamps must be timezone-aware")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("observation timestamps must be increasing and unique")

    positions = np.array_split(np.arange(len(index)), 3)
    windows: list[TuningWindow] = []
    for number, rows in enumerate(positions, start=1):
        first = int(rows[0])
        last = int(rows[-1])
        end = (
            index[last + 1]
            if last + 1 < len(index)
            else index[last] + pd.Timedelta(nanoseconds=1)
        )
        windows.append(
            TuningWindow(
                name=f"validation-{number}",
                start=index[first],
                end=end,
            )
        )
    return tuple(windows)


def build_base_config(reservoir: str) -> ReservoirConfig:
    """Build the fixed base configuration used by the tuner."""

    name = str(reservoir).strip()
    if not name:
        raise ValueError("reservoir must not be empty")
    return ReservoirConfig(
        reservoir_id=name.casefold(),
        reservoir_name=name,
        q=np.diag([Q_STORAGE, 1.0, Q_OUTFLOW]),
        r=np.diag([R_STORAGE, R_OUTFLOW]),
        p0=np.diag([P0_STORAGE, P0_INFLOW, P0_OUTFLOW]),
        smoothing_lag=SMOOTHING_LAG,
        initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
        inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
        model_version="physical-rate-v1",
        configuration_version=BASE_CONFIGURATION_VERSION,
        metadata={
            "source": f"{name}: prepare_reservoir_data",
            "status": "reviewed base",
        },
        unit_system=UnitSystem.us_customary(),
    )


def build_tuning_settings() -> InflowTuningSettings:
    """Return the fixed settings from the tuning notebook."""

    return InflowTuningSettings(
        warmup=WARMUP,
        innovation_max_lag=INNOVATION_MAX_LAG,
        forecast_horizons=FORECAST_HORIZONS,
        min_scored_storage_observations=MIN_SCORED_STORAGE_OBSERVATIONS,
        practical_equivalence_tolerance=PRACTICAL_EQUIVALENCE_TOLERANCE,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_seed=RANDOM_SEED,
        max_jitter_fraction=MAX_JITTER_FRACTION,
        max_regularized_steps=MAX_REGULARIZED_STEPS,
        r_sensitivity_multipliers=R_SENSITIVITY_MULTIPLIERS,
    )


def _validate_window_capacity(
    observations: pd.DataFrame,
    windows: Sequence[TuningWindow],
    settings: InflowTuningSettings,
) -> None:
    """Fail before tuning when a third cannot meet the score-count gate."""

    index = observations.index
    storage = observations["storage"].to_numpy(dtype=float, copy=False)
    warmup_end = index[1] + pd.Timedelta(seconds=settings.warmup.total_seconds())
    scoreable = (
        (np.arange(len(index)) >= 2)
        & (index >= warmup_end)
        & np.isfinite(storage)
    )
    counts = [
        int(np.count_nonzero(scoreable & window.mask(index))) for window in windows
    ]
    minimum = settings.min_scored_storage_observations
    insufficient = [
        f"{window.name} has {count} scored storage observations"
        for window, count in zip(windows, counts, strict=True)
        if count < minimum
    ]
    if insufficient:
        details = "; ".join(insufficient)
        raise ValueError(
            "the three validation windows cannot meet tuner requirements "
            f"(at least {minimum} scored storage observations per window): {details}"
        )


def _timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _print_result(result: object) -> None:
    """Print the small set of tuning outputs useful for review."""

    print(
        "Selected prior hourly increment SD: "
        f"{result.selected_prior_hourly_increment_sd}"
    )
    print(f"Selected q_inflow: {result.selected_q_inflow}")
    print(f"Selection reason: {result.selection_reason}")
    print("Warnings:")
    if result.warnings:
        for warning in result.warnings:
            print(f" - {warning}")
    else:
        print(" - none")
    print("\nCandidate summary:")
    print(result.candidate_summary.to_string(index=False))
    print("\nPer-window summary:")
    print(result.window_diagnostics.to_string(index=False))


def _json_value(value: Any) -> Any:
    """Convert common scientific/Pandas values to standards-compliant JSON."""

    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _frame_records(frame: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Return a JSON-safe records representation, including nulls for NaNs."""

    if frame is None:
        return []
    return _json_value(frame.to_dict(orient="records"))


def export_tuning_report(result: object, output_path: Path) -> Path:
    """Write the proposed configuration and tuning diagnostics to ``output_path``."""

    config = result.selected_config
    payload = {
        "selected_config": {
            "reservoir_id": config.reservoir_id,
            "reservoir_name": config.reservoir_name,
            "q": config.q,
            "r": config.r,
            "p0": config.p0,
            "smoothing_lag_seconds": config.smoothing_lag,
            "initialization_strategy": config.initialization_strategy,
            "inflow_units": config.inflow_units,
            "model_version": config.model_version,
            "configuration_version": config.configuration_version,
            "metadata": config.metadata,
            "unit_system": {
                "volume_label": config.unit_system.volume_label,
                "flow_label": config.unit_system.flow_label,
                "flow_to_volume_per_second": (
                    config.unit_system.flow_to_volume_per_second
                ),
            },
        },
        "selection": {
            "selected_prior_hourly_increment_sd": (
                result.selected_prior_hourly_increment_sd
            ),
            "selected_q_inflow": result.selected_q_inflow,
            "selection_threshold": result.selection_threshold,
            "selection_reason": result.selection_reason,
            "competitive_candidates": result.competitive_candidates,
        },
        "warnings": result.warnings,
        "timing_seconds": result.timing_seconds,
        "candidate_summary": _frame_records(result.candidate_summary),
        "window_diagnostics": _frame_records(result.window_diagnostics),
        "regime_diagnostics": _frame_records(result.regime_diagnostics),
        "horizon_diagnostics": _frame_records(result.horizon_diagnostics),
        "r_sensitivity": _frame_records(result.r_sensitivity),
    }
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


def run_tuner(
    *,
    project_root: Path | None = None,
    reservoir: str | None = None,
    data_start: object | None = None,
    data_end: object | None = None,
    output_path: Path | None = None,
) -> object:
    """Prepare, tune, export a JSON report, print diagnostics, and return it."""

    root = (
        PROJECT_ROOT
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    selected_reservoir = RESERVOIR if reservoir is None else reservoir
    start = _timestamp(DATA_START if data_start is None else data_start, "DATA_START")
    end = _timestamp(DATA_END if data_end is None else data_end, "DATA_END")
    if end < start:
        raise ValueError("DATA_END must be after or equal to DATA_START")

    sources = sources_for_reservoir(root, selected_reservoir)
    prepared = prepare_reservoir_data(
        sources,
        start=start,
        end=end,
        asof_tolerance=ASOF_TOLERANCE,
    )
    observations = prepared.observations
    if observations.empty:
        raise ValueError("the requested segment produced no observations")
    if len(observations) < 3:
        raise ValueError("the requested segment needs at least three observations")

    windows = split_validation_windows(observations.index)
    settings = build_tuning_settings()
    _validate_window_capacity(observations, windows, settings)
    base_config = build_base_config(selected_reservoir)
    result = tune_inflow_process_noise(
        storage=observations["storage"],
        discharge=observations["outflow"],
        base_config=base_config,
        candidate_prior_hourly_increment_sd=CANDIDATE_PRIOR_HOURLY_INCREMENT_SD,
        validation_windows=windows,
        settings=settings,
        proposed_configuration_version=PROPOSED_CONFIGURATION_VERSION,
    )
    destination = (
        root
        / OUTPUT_DIRECTORY
        / f"{base_config.reservoir_id}-{PROPOSED_CONFIGURATION_VERSION}.json"
        if output_path is None
        else Path(output_path)
    )
    exported_path = export_tuning_report(result, destination)
    _print_result(result)
    print(f"\nTuning report exported to: {exported_path}")
    return result


def main() -> None:
    """Run the tuner using the constants configured at the top of this file."""

    run_tuner()


if __name__ == "__main__":
    main()
