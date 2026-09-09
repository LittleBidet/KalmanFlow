# KalmanFlow

| | Status |
| --- | --- |
| Testing | [![Tests](https://github.com/LittleBidet/KalmanFlow/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/LittleBidet/KalmanFlow/actions/workflows/tests.yml) |
| Package | [![PyPI](https://img.shields.io/pypi/v/kalmanflow)](https://pypi.org/project/kalmanflow/) [![Python versions](https://img.shields.io/pypi/pyversions/kalmanflow)](https://pypi.org/project/kalmanflow/) |
| License | [![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/LittleBidet/KalmanFlow/blob/main/LICENSE) |

KalmanFlow estimates an inflow contribution from noisy storage and measured discharge. It provides a three-state physical water-balance model, causal Kalman filtering, and fixed-lag Rauch–Tung–Striebel (RTS) smoothing for delayed inflow revisions.

The package is intentionally data-source agnostic: applications are responsible for parsing, cleaning, aligning, and persisting reservoir data.

## Overview

Reservoir inflow is important for water-supply planning and flood-response operations, but it is often difficult to measure directly. A reservoir may receive water from many tributaries, drainage areas, or stormwater inputs, so installing and maintaining flow sensors at every inflow point is not practical.

Timestamped reservoir storage and outflow data can be used to infer a net balance contribution with reverse level-pool routing: if storage and accounted outflow are known, the residual contribution can be recovered from the water balance. Direct back-calculation, however, is sensitive to sensor noise and timing differences. Small storage or outflow errors can yield unrealistic inflow spikes or negative values.

KalmanFlow estimates a more stable inflow time series from those noisy observations. After the two-observation startup phase, a causal inflow is published as observations arrive; a later observation can provide an absolute, fixed-lag-smoothed replacement for that same timestamp. It also provides documented assumptions and reviewed noise parameters for a measurable, defensible deployment.

The reported inflow is the residual term in the supplied storage and outflow balance. It is a net balance contribution, not automatically gross watershed inflow. Measured outflow should cover outlet releases, spills, and outward diversions or withdrawals as applicable. Precipitation, evaporation, seepage, and other water exchanges are not separate model terms; account for them with separate justified inputs or treat them as model mismatch. Sensor bias, storage-datum changes, and rating-curve changes are also possible mismatch sources. The linear-Gaussian model has no nonnegativity constraint, so negative estimates remain possible.

## Install

KalmanFlow requires Python 3.12 or newer.

```bash
pip install kalmanflow
```

For experimental Bayesian noise tuning, install the optional extra:

```bash
pip install "kalmanflow[tuning]"
```

The base installation supports filtering, smoothing, and frozen-configuration evaluation without SciPy or scikit-learn. Bayesian tuning is **experimental**: its API, selection rules, and result schema may change between releases.

For development in this repository:

```bash
uv sync --all-groups --extra tuning
```

The source distribution includes the library and its Documentation pages. The
full test suite, application workflows, notebooks, and reservoir data remain
in the GitHub development checkout; run the test commands there.



## Quick start

Use `OnlineReservoirInflow` when observations arrive one at a time. Timestamps must be timezone-aware and no finer than microsecond precision; the first completed initialization emits estimates for its first finite-storage/discharge anchor and the next finite storage observation. The anchor estimate uses a steady-state inflow prior based only on its own outflow, so it never looks ahead to the second storage value.

```python
from datetime import UTC, datetime, timedelta
from kalmanflow import Observation, OnlineReservoirInflow

stream = OnlineReservoirInflow(
    # Parameters are for demonstration only. (See Configuration docs)
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

### Optional inflow uncertainty

Inflow uncertainty is opt-in. The default `include_uncertainty=False` keeps
the existing six-column batch result, and streaming estimates expose
`standard_deviation=None`. Set `include_uncertainty=True` on a batch adapter
or stream constructor to include the inflow standard deviation calculated from
the filter covariance:

```python
from datetime import UTC, datetime, timedelta

import pandas as pd

from kalmanflow import (
    Observation,
    OnlineReservoirInflow,
    add_inflow_uncertainty_intervals,
    get_reservoir_inflow,
)

stream = OnlineReservoirInflow(
    q_storage=1.0,
    q_inflow=1.0,
    q_outflow=1.0,
    r_storage=100.0,
    r_outflow=25.0,
    smoothing_lag=timedelta(minutes=15),
    include_uncertainty=True,
)
stream.process(
    Observation(
        datetime(2026, 1, 1, tzinfo=UTC),
        storage=10_000.0,
        discharge=25.0,
    )
)
update = stream.process(
    Observation(
        datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
        storage=10_001.0,
        discharge=25.5,
    )
)
for estimate in update.filtered_inflows:
    print(estimate.value, estimate.standard_deviation)
    print(estimate.uncertainty_interval(level=0.95))

index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
storage = pd.Series([10_000.0, 10_001.0, 10_003.0], index=index)
discharge = pd.Series([25.0, 25.5, 26.0], index=index)
result = get_reservoir_inflow(
    storage,
    discharge,
    q_storage=1.0,
    q_inflow=1.0,
    q_outflow=1.0,
    r_storage=100.0,
    r_outflow=25.0,
    smoothing_lag=timedelta(minutes=15),
    include_uncertainty=True,
)
result_with_intervals = add_inflow_uncertainty_intervals(result, level=0.95)
```

The opt-in batch result appends `estimated_inflow_standard_deviation` and
`revised_inflow_standard_deviation`. `add_inflow_uncertainty_intervals`
returns a copy with lower and upper columns for both estimates. These are
pointwise, model-based intervals for the net inflow contribution, in the same
flow units as the estimate. Bounds are not clipped at zero. A revised record
replaces both the earlier causal value and its standard deviation; unreleased
revisions, including the trailing row, remain `NaN`.



## Primary APIs


| API                                         | Use it when                                                                                                         |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `get_reservoir_inflow`                      | You have aligned pandas storage and discharge series and use the default acre-ft/cfs model.                         |
| `get_reservoir_inflow_from_config`          | You have batch data and a validated `ReservoirConfig`.                                                              |
| `OnlineReservoirInflow`                     | You process one reservoir’s observations as they arrive.                                                            |
| `OnlineReservoirInflow.from_config`         | You need a configured, checkpoint-capable streaming estimator.                                                      |
| `OnlineInflowPipeline`                      | You are integrating a custom backend or smoother.                                                                   |
| `add_inflow_uncertainty_intervals`          | You want normal-theory lower and upper interval columns for an opt-in batch result.                                 |
| `tune_inflow_noise_bayesian` (experimental) | You want to propose five diagonal `q`/`r` noise terms from causal innovation scores; requires `kalmanflow[tuning]`. |
| `evaluate_configuration`                    | You want to assess one frozen configuration on an untouched period without searching.                               |
| Notebook-local `validation.py`              | You want hourly proxy-agreement, lag, and storage-closure validation tables and plots.                              |




## Documentation

- [Architecture](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/ARCHITECTURE.md): modules, state-space model, units, and library boundaries.
- [Model behavior](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/INFLOW_MODEL_BEHAVIOR.md): input contract, initialization, missing values, and batch/streaming outputs.
- [Online pipeline](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/ONLINE_INFLOW_PIPELINE.md): lifecycle, checkpoints, and failure behavior.
- [Configuration](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/CONFIGURATION.md): complete configuration example, unit conventions, and parameter selection.
- [Validation](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/VALIDATION.md): regular comparison frames, upstream-proxy metrics, lag selection, and storage closure.
- [Bayesian innovation noise tuning](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/BAYESIAN_NOISE_TUNING.md): five-parameter causal covariance search without true inflow labels.
- [API reference](https://github.com/LittleBidet/KalmanFlow/blob/main/Documentation/API_REFERENCE.md): every package-level export, plus the advanced module interfaces.



## Requirements and units

Inputs must be pre-cleaned and indexed by unique, timezone-aware, strictly increasing timestamps with at most microsecond precision. Batch storage and discharge series must have exactly matching indexes. The default model uses acre-feet for storage and cfs for flow rates. Missing storage or discharge is represented by `NaN`; available
components still participate in a partial update.

The default state is `[storage, inflow_rate, true_outflow_rate]`. Storage and measured outflow are model inputs, and true outflow is an internal state. The public API returns causal inflow followed by absolute revised inflow values; it never returns outflow estimates.

## Development

```bash
uv run --extra tuning pytest
uv run ruff check .
```
