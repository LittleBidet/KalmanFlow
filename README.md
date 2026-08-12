# Kalman filtering for reservoir inflow calculation

## Problem

Reservoir inflow is important for water-supply planning and flood-response
operations, but it is often difficult to measure directly. A reservoir may
receive water from many tributaries, drainage areas, or stormwater inputs, so
installing and maintaining flow sensors at every inflow point is not practical.

Valley Water has multiple years of timestamped reservoir storage and outflow data.
In principle, that data can be used to back-calculate inflow with reverse
level-pool routing: if storage and outflow are known, inflow can be inferred
from the water balance. In practice, direct back-calculation is sensitive to
sensor noise and timing differences. Small errors in storage or outflow can
produce unrealistic inflow estimates, including large spikes or negative values.

The goal of this project is to use Kalman filtering to estimate a more stable,
physically reasonable inflow time series from noisy storage and outflow
observations. The method should support real-time use, where the most recent
inflow estimates may continue to adjust as new observations arrive.

The method also needs to be measurable and defensible. That means quantifying
performance, documenting the assumptions, and justifying parameter choices such
as process variance and sensor measurement variance.

## Package contract

`get_reservoir_inflow` is the batch API. It accepts pre-cleaned, timezone-aware,
sorted, deduplicated storage and measured outflow `Series` and returns a
`DataFrame` with `estimated_inflow` and `estimated_outflow` columns plus
prediction and smoothing flag columns. The `*_flag` columns contain `NORMAL`
or `PREDICTED`; any single- or double-missing observation produces
`PREDICTED`. The `*_smoothing_flag` columns contain `SMOOTHED` or
`NON_SMOOTHED`. Inflow is the causal filtered estimate available when each
observation is processed, so it is `NON_SMOOTHED`; outflow is populated only
after its fixed-lag estimate is finalized, so released values are `SMOOTHED`.
Unfinished trailing outflow values are `NaN` and are marked `NON_SMOOTHED`.
Storage remains an internal model state and is not included in the public
frame.

`OnlineReservoirInflow` is the reservoir-specific streaming API. Each
`process` call returns a `ReservoirFlowUpdate`:

- `filtered_inflows` contains timestamped inflow estimates available from the
  causal filter. Each estimate exposes `prediction_flag` (`NORMAL` or
  `PREDICTED`) and `smoothing_flag` (`NON_SMOOTHED`). The first valid storage
  timestamp is emitted when the second valid storage observation completes
  initialization.
- `estimated_outflows` contains only timestamped outflow estimates finalized by
  the smoothing lag. Each estimate exposes `prediction_flag` and
  `smoothing_flag=SMOOTHED`. Provisional lag-window estimates are not exposed.

Validated deployments can construct the streaming estimator with
`OnlineReservoirInflow.from_config(config)` or use the corresponding batch
adapter `get_reservoir_inflow_from_config(storage, outflow, config)`. These
paths honor the configuration's initial covariance, unit system, noise
covariances, and smoothing lag. The scalar-parameter batch API remains the
convenience path for the default acre-ft/cfs model.

`OnlineInflowPipeline` is the streaming API. Each `process` call returns a
`PipelineUpdate`:

- `filtered_state` is the current causal Kalman output that was just processed.
- `filtered_states` contains every new causal filter state created by the call,
  including both initialization states.
- `smoothed_states` contains states finalized after the smoothing lag.

`OnlineInflowPipeline` remains available as the lower-level generic coordinator;
use `OnlineReservoirInflow` when the public output should contain only the
staggered reservoir flow estimates.

The pipeline requires timezone-aware, strictly increasing timestamps. Input
series must be pre-cleaned by the caller: parsed, aligned, sorted, and
deduplicated. The default reservoir state is
`[storage, inflow_rate, true_outflow_rate]`, where the public batch adapter reports both flow-rate states as cfs. Missing components are omitted from the Kalman update and if both are missing, the step is predict-only.

This package does not parse, align, deduplicate, impute, or otherwise clean
input data beyond those runtime missing-value rules.

## Assumptions

- Storage and outflow inputs have matching, timezone-aware `DatetimeIndex`
values in the same order. The caller provides parsed, strictly increasing,
duplicate-free data; the package does not sort, align, or deduplicate it.
- Elapsed-time calculations are normalized to UTC, so timezone-aware timestamps
may use named time zones and still cross daylight-saving transitions safely.
- State order is `[storage, inflow_rate, true_outflow_rate]`. Storage uses the
configured volume unit and both flow-rate states use the configured flow-rate
unit; the public batch adapter reports both flow-rate outputs in cfs.
- Storage and measured outflow are observations with separate measurement
variances. Outflow is not a known control input.
- Online initialization waits for two finite storage observations and requires
finite outflow for the first valid storage. The second outflow may be missing.
After initialization, missing storage or outflow is handled as a partial
observation.
- Tuning requires two finite initial storage values and a finite first outflow.
Later missing storage or outflow values are omitted from their respective
innovation channels.
- The batch `estimated_inflow` column contains causal filtered values. The
  batch `estimated_outflow` column contains only finalized fixed-lag values,
  with `NaN` for the unfinished trailing lag window. Storage is internal and
  is not part of the public batch or reservoir-stream output.
- Process diffusion covariances and initial covariance must be finite, symmetric,
and positive semidefinite. The 3x3 `q` and `p0` matrices use the state order
above. The 2x2 measurement covariance `r` uses storage/outflow order and has
strictly positive diagonal entries. The physical-rate model derives each
discrete process covariance from its continuous-time diffusion covariance and
actual elapsed seconds.
- `ReservoirConfig` supplies physical-state model and tuning parameters. `q`
uses the continuous-time diffusion convention and `p0` uses
`[storage, inflow_rate, true_outflow_rate]` units. Existing two-state or
reference-interval configurations require explicit migration; their numeric
values must not be reused without re-tuning.



## Optional tuning

`tune_noise` optimizes `q_storage`, `q_inflow`, `q_outflow`, `r_storage`, and
`r_outflow` in log space. It imports SciPy only when called; the core estimator
does not require SciPy. Install the optional tuning dependency with
`pip install -e '.[tuning]'`. Tuning returns a result object and never mutates
the active configuration. The `loglik` objective uses joint filter log
likelihood; the `rmse` objective uses standardized finite one-step innovations
from both observation channels.

Artifact persistence remains outside this package.

## Unit migration

The physical-rate model uses actual timestamp differences for every transition.
The public `estimated_inflow` and `estimated_outflow` results are cfs, regardless
of whether samples arrive every 5, 10, 15, or 60 minutes. Existing two-state
configurations, scalar measurement variances, and tuned artifacts are a
different model version and must be converted and revalidated before use.
