# Kalmone

Kalmone estimates reservoir inflow from noisy storage and measured discharge. It provides a three-state physical water-balance model, causal Kalman filtering, and fixed-lag Rauch–Tung–Striebel (RTS) smoothing for delayed inflow revisions.

The package is intentionally data-source agnostic: applications are responsible for parsing, cleaning, aligning, and persisting reservoir data.

## Overview

Reservoir inflow is important for water-supply planning and flood-response
operations, but it is often difficult to measure directly. A reservoir may
receive water from many tributaries, drainage areas, or stormwater inputs, so
installing and maintaining flow sensors at every inflow point is not practical.

Timestamped reservoir storage and outflow data can be used to infer inflow
with reverse level-pool routing: if storage and outflow are known, inflow can
be recovered from the water balance. Direct back-calculation, however, is
sensitive to sensor noise and timing differences. Small storage or outflow
errors can yield unrealistic inflow spikes or negative values.

Kalmone estimates a more stable, physically reasonable inflow time series from
those noisy observations. It supports real-time use: a causal inflow is first
published immediately, then a later observation can provide an absolute,
fixed-lag-smoothed replacement for that same timestamp. It also provides
documented assumptions and noise-tuning tools for a measurable, defensible
deployment.

## Install

```bash
pip install kalmone
# Optional Q/R tuning support
pip install 'kalmone[tuning]'
```

For development in this repository:

```bash
uv sync --all-groups
```

## Quick start

Use `OnlineReservoirInflow` when observations arrive one at a time. Timestamps must be timezone-aware; the first completed initialization emits estimates for the first two valid storage observations.

```python
from datetime import UTC, datetime, timedelta
from kalmone import Observation, OnlineReservoirInflow

stream = OnlineReservoirInflow(
    q_storage=1.0,
    q_inflow=1.0,
    q_outflow=1.0,
    r_storage=100.0,
    r_outflow=25.0,
    smoothing_lag=timedelta(hours=12),
)

stream.process(
    Observation(datetime(2026, 1, 1, tzinfo=UTC), storage=10_000.0, discharge=25.0)
)
update = stream.process(
    Observation(
        datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
        storage=10_001.0,
        discharge=25.5,
    )
)

for estimate in update.filtered_inflows:
    print(estimate.timestamp, estimate.value, estimate.prediction_flag)

for revision in update.revised_inflows:
    # Replace the causal value at revision.timestamp; do not add a delta.
    print(revision.timestamp, revision.value, revision.smoothing_flag)
```

The scalar values above only make the example runnable; select and validate noise parameters for each reservoir. See [configuration and tuning](Documentation/CONFIGURATION_AND_TUNING.md).

## Primary APIs

| API | Use it when |
| --- | --- |
| `get_reservoir_inflow` | You have aligned pandas storage and discharge series and use the default acre-ft/cfs model. |
| `get_reservoir_inflow_from_config` | You have batch data and a validated `ReservoirConfig`. |
| `OnlineReservoirInflow` | You process one reservoir’s observations as they arrive. |
| `OnlineReservoirInflow.from_config` | You need a configured, checkpoint-capable streaming estimator. |
| `OnlineInflowPipeline` | You are integrating a custom backend or smoother. |
| `tune_noise` | You want to optimize Q/R from a pre-cleaned historical record (requires the `tuning` extra). |
| Notebook-local `validation.py` | You want hourly proxy-agreement, lag, and storage-closure validation tables and plots. |

## Documentation

- [Architecture](Documentation/ARCHITECTURE.md): modules, state-space model, units, and library boundaries.
- [Model behavior](Documentation/INFLOW_MODEL_BEHAVIOR.md): input contract, initialization, missing values, and batch/streaming outputs.
- [Online pipeline](Documentation/ONLINE_INFLOW_PIPELINE.md): lifecycle, checkpoints, and failure behavior.
- [Configuration and tuning](Documentation/CONFIGURATION_AND_TUNING.md): complete configuration example, unit conventions, parameter selection, and tuning workflow.
- [Validation](Documentation/VALIDATION.md): regular comparison frames, upstream-proxy metrics, lag selection, and storage closure.

## Requirements and units

Inputs must be pre-cleaned and timestamped with timezone-aware, strictly increasing values. The default model uses acre-feet for storage and cfs for flow rates. Missing storage or discharge is represented by `NaN`; available components still participate in a partial update.

The default state is `[storage, inflow_rate, true_outflow_rate]`. Storage and
measured outflow are model inputs, and true outflow is an internal state. The
public API returns causal inflow followed by absolute revised inflow values;
it never returns outflow estimates.

## Development

```bash
uv run pytest
uv run ruff check .
```
