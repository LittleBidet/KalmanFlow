"""Run the Bayesian five-parameter innovation tuner."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

# This file lives in the application layer, one directory below the project
# root.  Add both roots so direct-file execution works from a checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from applications.preparation import (  # noqa: E402
    prepare_reservoir_data,
    sources_for_reservoir,
)
from kalmanflow import (  # noqa: E402
    BayesianEvaluationSettings,
    BayesianTuningResult,
    BayesianTuningSettings,
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    UnitSystem,
    ValidationWindow,
    tune_inflow_noise_bayesian,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


RESERVOIR = "Chesbro"
DATA_START = pd.Timestamp("2022-10-01T00:00:00Z")
DATA_END = pd.Timestamp("2023-02-01T00:00:00Z")
ASOF_TOLERANCE = pd.Timedelta("20min")
BASE_CONFIGURATION_VERSION = "reviewed-base-v1"
PROPOSED_CONFIGURATION_VERSION = "2026-08-bayesian-noise-candidate"
INFLOW_INCREMENT_SD_SEEDS = (2.0, 5.0, 10.0, 20.0, 40.0)
OUTPUT_DIRECTORY = Path("Outputs") / "bayesian_tuning"
Q_STORAGE = 0.002
Q_OUTFLOW = 0.02
R_STORAGE = 0.25
R_OUTFLOW = 4.0
P0_STORAGE = 100.0
P0_INFLOW = 400.0
P0_OUTFLOW = 25.0
SMOOTHING_LAG = timedelta(hours=6)
WARMUP = timedelta(hours=24)
INNOVATION_MAX_LAG = timedelta(hours=24)
MIN_SCORED_STORAGE_OBSERVATIONS = 12
PRACTICAL_EQUIVALENCE_TOLERANCE = 0.02
BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 20260820
MAX_JITTER_FRACTION = 1e-9
MAX_REGULARIZED_STEPS = 0


def split_validation_windows(index: pd.DatetimeIndex) -> tuple[ValidationWindow, ...]:
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("observations must use a pandas DatetimeIndex")
    if len(index) < 3:
        raise ValueError("at least three observations are required for three windows")
    if index.tz is None:
        raise ValueError("observation timestamps must be timezone-aware")
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("observation timestamps must be increasing and unique")
    positions = np.array_split(np.arange(len(index)), 3)
    windows: list[ValidationWindow] = []
    for number, rows in enumerate(positions, start=1):
        first, last = int(rows[0]), int(rows[-1])
        end = (
            index[last + 1]
            if last + 1 < len(index)
            else index[last] + pd.Timedelta(nanoseconds=1)
        )
        windows.append(ValidationWindow(f"validation-{number}", index[first], end))
    return tuple(windows)


def build_base_config(reservoir: str) -> ReservoirConfig:
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


def build_evaluation_settings() -> BayesianEvaluationSettings:
    return BayesianEvaluationSettings(
        warmup=WARMUP,
        innovation_max_lag=INNOVATION_MAX_LAG,
        min_scored_storage_observations=MIN_SCORED_STORAGE_OBSERVATIONS,
        practical_equivalence_tolerance=PRACTICAL_EQUIVALENCE_TOLERANCE,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        random_seed=RANDOM_SEED,
        max_jitter_fraction=MAX_JITTER_FRACTION,
        max_regularized_steps=MAX_REGULARIZED_STEPS,
    )


def _validate_window_capacity(
    observations: pd.DataFrame,
    windows: Sequence[ValidationWindow],
    settings: BayesianEvaluationSettings,
) -> None:
    index = observations.index
    storage = observations["storage"].to_numpy(dtype=float, copy=False)
    finite_storage = np.flatnonzero(np.isfinite(storage))
    if len(finite_storage) < 2:
        raise ValueError("at least two finite storage observations are required")
    second_storage = int(finite_storage[1])
    warmup_end = index[second_storage] + pd.Timedelta(
        seconds=settings.warmup.total_seconds()
    )
    scoreable = (
        (np.arange(len(index)) > second_storage)
        & (index >= warmup_end)
        & np.isfinite(storage)
    )
    counts = [
        int(np.count_nonzero(scoreable & window.mask(index))) for window in windows
    ]
    insufficient = [
        f"{window.name} has {count} scored storage observations"
        for window, count in zip(windows, counts, strict=True)
        if count < settings.min_scored_storage_observations
    ]
    if insufficient:
        raise ValueError(
            "the three validation windows cannot meet Bayesian requirements "
            + "; ".join(insufficient)
        )


def _timestamp(value: object, label: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a valid timestamp") from error
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def build_bayesian_settings() -> BayesianTuningSettings:
    """Return conservative defaults for the Bayesian search."""

    return BayesianTuningSettings(
        total_trials=40,
        initial_trials=12,
        random_seed=RANDOM_SEED,
        q_storage_multiplier_bounds=(0.5, 1.0),
        q_outflow_multiplier_bounds=(0.1, 10.0),
        r_storage_multiplier_bounds=(0.5, 1.0),
        r_outflow_multiplier_bounds=(0.25, 4.0),
        one_standard_error_weight=1.0,
        max_abs_elapsed_lag_autocorrelation=0.25,
    )


def _json_value(value: Any) -> Any:
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
    if frame is None:
        return []
    return _json_value(frame.to_dict(orient="records"))


def _config_payload(config: ReservoirConfig) -> dict[str, Any]:
    """Return the JSON representation of a complete reservoir configuration."""

    return {
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
            "flow_to_volume_per_second": config.unit_system.flow_to_volume_per_second,
        },
    }


def _write_json(payload: object, output_path: Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


def export_reservoir_config(config: ReservoirConfig, output_path: Path) -> Path:
    """Write a complete, compact ``ReservoirConfig`` JSON artifact."""

    return _write_json(_config_payload(config), output_path)


def export_bayesian_tuning_report(
    result: BayesianTuningResult,
    output_path: Path,
    *,
    configuration_path: Path | None = None,
) -> Path:
    """Write the detailed trial, selection, and efficacy report as JSON.

    The selected configuration is written by :func:`export_reservoir_config`.
    This report carries its identity and optional artifact path only, avoiding
    a second copy of the complete configuration.
    """

    config = result.selected_config
    payload = {
        "selected_config_reference": {
            "reservoir_id": config.reservoir_id,
            "reservoir_name": config.reservoir_name,
            "configuration_version": config.configuration_version,
            "model_version": config.model_version,
        },
        "selection": {
            "selected_parameters": result.selected_parameters,
            "selected_objective": result.selected_objective,
            "selection_threshold": result.selection_threshold,
            "selection_reason": result.selection_reason,
            "competitive_trial_ids": result.competitive_trial_ids,
        },
        "warnings": result.warnings,
        "timing_seconds": result.timing_seconds,
        "candidate_summary": _frame_records(result.candidate_summary),
        "window_diagnostics": _frame_records(result.window_diagnostics),
        "proxy_diagnostics": _frame_records(result.proxy_diagnostics),
    }
    if configuration_path is not None:
        payload["selected_config_reference"]["path"] = str(
            Path(configuration_path).expanduser().resolve()
        )
    return _write_json(payload, output_path)


def _print_result(result: BayesianTuningResult) -> None:
    print("Selected Bayesian parameters:")
    for name, value in result.selected_parameters.items():
        print(f"  {name}: {value}")
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


def run_bayesian_tuner(
    *,
    project_root: Path | None = None,
    reservoir: str | None = None,
    data_start: object | None = None,
    data_end: object | None = None,
    output_path: Path | None = None,
    report_output_path: Path | None = None,
) -> BayesianTuningResult:
    """Prepare data, run Bayesian tuning, export artifacts, and return the result.

    ``output_path`` receives the compact selected configuration. A detailed
    tuning report is written only when ``report_output_path`` is supplied.
    """

    root = (
        PROJECT_ROOT
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )
    selected_reservoir = RESERVOIR if reservoir is None else reservoir
    default_output_path = (
        root / OUTPUT_DIRECTORY / f"{str(selected_reservoir).strip().casefold()}-"
        f"{PROPOSED_CONFIGURATION_VERSION}.json"
    )
    destination = (
        (default_output_path if output_path is None else Path(output_path))
        .expanduser()
        .resolve()
    )
    report_destination = (
        None
        if report_output_path is None
        else Path(report_output_path).expanduser().resolve()
    )
    if report_destination is not None and report_destination == destination:
        raise ValueError("output_path and report_output_path must be different files")
    start = _timestamp(DATA_START if data_start is None else data_start, "DATA_START")
    end = _timestamp(DATA_END if data_end is None else data_end, "DATA_END")
    if end < start:
        raise ValueError("DATA_END must be after or equal to DATA_START")
    sources = sources_for_reservoir(root, selected_reservoir)
    prepared = prepare_reservoir_data(
        sources, start=start, end=end, asof_tolerance=ASOF_TOLERANCE
    )
    observations = prepared.observations
    if observations.empty:
        raise ValueError("the requested segment produced no observations")
    if len(observations) < 3:
        raise ValueError("the requested segment needs at least three observations")
    windows = split_validation_windows(observations.index)
    settings = build_evaluation_settings()
    _validate_window_capacity(observations, windows, settings)
    base_config = build_base_config(selected_reservoir)
    upstream_proxy = None
    if isinstance(getattr(prepared, "diagnostics", None), pd.DataFrame):
        diagnostics = prepared.diagnostics
        if "upstream_flow" in diagnostics:
            upstream_proxy = diagnostics["upstream_flow"]
    result = tune_inflow_noise_bayesian(
        storage=observations["storage"],
        discharge=observations["outflow"],
        base_config=base_config,
        inflow_increment_sd_seeds=INFLOW_INCREMENT_SD_SEEDS,
        validation_windows=windows,
        settings=settings,
        bayesian_settings=build_bayesian_settings(),
        upstream_proxy=upstream_proxy,
        proposed_configuration_version=PROPOSED_CONFIGURATION_VERSION,
    )
    exported_config_path = export_reservoir_config(result.selected_config, destination)
    exported_report_path = None
    if report_destination is not None:
        exported_report_path = export_bayesian_tuning_report(
            result,
            report_destination,
            configuration_path=exported_config_path,
        )
    _print_result(result)
    print(f"\nBayesian tuning configuration exported to: {exported_config_path}")
    if exported_report_path is not None:
        print(f"Bayesian tuning report exported to: {exported_report_path}")
    return result


def main() -> None:
    run_bayesian_tuner()


if __name__ == "__main__":
    main()
