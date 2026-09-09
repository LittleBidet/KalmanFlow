# KalmanFlow model behavior

## Input contract

`get_reservoir_inflow` and `get_reservoir_inflow_from_config` accept pre-cleaned storage and discharge `pandas.Series` with exactly equal, timezone-aware, strictly increasing indexes. Observation timestamps support microsecond precision; finer pandas timestamps are rejected before computation. The default adapter expects storage in acre-feet and discharge in cfs. The configured adapter uses the labels in `ReservoirConfig.unit_system`.

Inflow uncertainty is optional and is disabled by default. Pass
`include_uncertainty=True` to a batch adapter or streaming constructor when
standard deviations are needed. A checkpoint restores model state only; the
caller chooses the uncertainty output option again with
`from_checkpoint(..., include_uncertainty=...)`.

`OnlineReservoirInflow.process` accepts either an `Observation` or `timestamp`, `storage`, and `discharge` keywords. Timestamps must be timezone-aware, no finer than microseconds, and strictly later than the previously accepted observation. All elapsed-time comparisons are normalized to UTC. Validation-window boundaries may retain nanosecond precision because they are interval labels rather than observations.

The package intentionally does not sort, align, deduplicate, interpolate, or impute inputs.

## Initialization and missing values

The stream ignores leading rows with missing storage. Its first finite storage
reading must also have finite discharge; that row becomes the initialization
anchor. The next finite storage reading completes initialization, even if its
discharge is missing. At the anchor timestamp, the model uses measured outflow
as a steady-state inflow prior (zero initial storage-change assumption). That
first filtered value therefore depends only on the anchor observation, not on
the later storage reading. The second storage reading then updates inflow
through the normal causal predict/update step. A finite storage row with
missing discharge before an anchor is invalid rather than silently skipped.

After initialization, either storage or discharge may be `NaN`. A finite component is still used as a partial Kalman observation. If both are `NaN`, the step is predict-only. Positive and negative infinity are invalid; `NaN` is the only missing-value marker. A public estimate carrying any missing observation component receives the `PREDICTED` flag; fully observed steps are `NORMAL`.

## Batch outputs

The batch adapters return a `DataFrame`, indexed like the inputs, with:

| Column | Meaning |
| --- | --- |
| `estimated_inflow` | Causal filtered inflow rate. |
| `revised_inflow` | Absolute fixed-lag-smoothed inflow replacement, or `NaN` until final. |
| `estimated_inflow_flag` / `revised_inflow_flag` | `NORMAL` or `PREDICTED`. |
| `estimated_inflow_smoothing_flag` | Always `NON_SMOOTHED`: inflow is causal. |
| `revised_inflow_smoothing_flag` | `SMOOTHED` when released; otherwise `NON_SMOOTHED`. |

When a revised inflow becomes available, it replaces `estimated_inflow` at the
same timestamp; it is not a delta to add to that value. Storage and outflow
are internal states and are not public batch outputs.

### Optional uncertainty outputs

With `include_uncertainty=True`, the batch adapters append these columns after
the six columns above:

| Column | Meaning |
| --- | --- |
| `estimated_inflow_standard_deviation` | Pointwise standard deviation of the causal filtered inflow. |
| `revised_inflow_standard_deviation` | Pointwise standard deviation of the fixed-lag-smoothed inflow replacement. |

The values come from the inflow-rate entry `[1, 1]` of the corresponding
filtered or smoothed covariance. They use the same flow units as the inflow
estimate. Without opt-in, the batch schema remains unchanged and streaming
estimates carry `standard_deviation=None`.

Use `add_inflow_uncertainty_intervals(result, level=0.95)` to add
`estimated_inflow_lower`, `estimated_inflow_upper`, `revised_inflow_lower`,
and `revised_inflow_upper` to a result created with uncertainty enabled. The
helper returns a copy and uses a central normal-theory interval. It preserves
negative bounds. A row without a published estimate has `NaN` bounds; the
trailing revised row is therefore `NaN` until a later observation finalizes it.
The same rule applies to `ReservoirFlowEstimate.uncertainty_interval()` in
the streaming API, which raises if that estimate has no standard deviation.

These are pointwise, model-based uncertainty intervals for the net inflow
contribution, not empirical coverage guarantees. They include the uncertainty
represented by the selected filter covariance and exclude uncertainty in
configuration or noise-parameter tuning, model bias, and other unmodeled water
exchanges. The startup covariance is an engineering approximation rather than
a measurement-conditioned calibration; interpret early uncertainty using that
existing startup caveat.

## Streaming outputs

Each call to `OnlineReservoirInflow.process` returns a `ReservoirFlowUpdate`.
`filtered_inflows` contains newly available causal inflow estimates;
`revised_inflows` contains only newly finalized, absolute smoothed
replacements at their original timestamps. With `include_uncertainty=True`,
each estimate also carries its pointwise `standard_deviation`; with the
default setting the field is `None`. A revised estimate replaces both the
previous causal value and its uncertainty. The initial successful call that
completes initialization can emit two filtered inflow estimates. The active
smoothing window is retained internally and is bounded by `max_window_steps`.
Although both startup estimates are emitted together, the anchor estimate was
computed from the anchor observation alone.

The anchor state mean is seeded from the anchor storage and measured outflow,
then that same anchor observation is assimilated by the filter. Its innovation
is therefore zero by construction and the covariance can be reduced. This startup
covariance is an engineering approximation rather than a measurement-
conditioned uncertainty calibration; interpret early uncertainty and early
calibration diagnostics accordingly.

`process_many` is transactional: if any item is invalid, the stream returns to its entry state and produces no partial group result.

## Configuration

`ReservoirConfig` is immutable. It holds a reservoir identifier, model and configuration versions, continuous-time `q` (3×3), measurement covariance `r` (2×2), initial covariance `p0` (3×3), a positive smoothing lag, units, and metadata. Covariances must be finite, symmetric, positive semidefinite; the diagonal of `r` must be strictly positive.

Select and review continuous-time process noise (`q`), observation noise (`r`),
and initial covariance (`p0`) for each reservoir before operational use.
Document the rationale and configuration version in the configuration metadata.

## Bayesian causal evaluation

`tune_inflow_noise_bayesian` is an offline calibration aid, not an adaptive
streaming mode. It uses the forward Kalman filter and never calls the RTS
smoother. Initialization rows and the configured warm-up period are excluded
from scores. Only rows inside the declared validation windows contribute to
the objective. Earlier observations outside a window may still condition its
causal starting state, and validation observations can condition later
predictions, as they would in operation; no observation can affect its own or
an earlier score.

Use `evaluate_configuration` once a proposed configuration is frozen for a
separate, untouched test period. Test results must not be used to change the
Bayesian search, windows, thresholds, or selected parameters. Filtered storage
closure is a reconstruction diagnostic; it is not an independent predictive
score when the ending storage observation has already been assimilated.
