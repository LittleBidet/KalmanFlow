# API reference

This page covers the complete supported package surface in KalmanFlow 0.1.0.
Import application-facing names from `kalmanflow`. The advanced types below are
also package exports, but are intended for custom models and integrations.
Names in modules beginning with `_` are internal and may change without a
compatibility promise.

`kalmanflow.__version__` reports the installed package version.

## Reservoir estimation

| Export | Purpose |
| --- | --- |
| `Observation(timestamp, storage, discharge)` | Immutable streaming input. Timestamp must be timezone-aware and no finer than microsecond precision; use `NaN` for a missing value after initialization. |
| `OnlineReservoirInflow` | Default acre-ft/cfs streaming estimator. Construct with the five scalar diagonal noise values; use `process`, `process_many`, `initialized`, and `pending_count`. Set keyword-only `include_uncertainty=True` to publish inflow standard deviations. |
| `OnlineReservoirInflow.from_config(config, include_uncertainty=False)` | Creates a stream from reviewed configuration, including its units and smoothing lag. Set `include_uncertainty=True` to publish inflow standard deviations. |
| `OnlineReservoirInflow.from_checkpoint(checkpoint, config=..., include_uncertainty=False)` | Restores a configured stream. The reservoir ID and fingerprinted model, covariance, lag, and unit settings must match. Choose the uncertainty output option again when restoring. |
| `OnlineReservoirInflow.checkpoint()` | Produces resumable state after a successful call. It requires a reservoir ID; streams made with `from_config` have one. |
| `ReservoirFlowEstimate` | One timestamped inflow value with `prediction_flag`, `smoothing_flag`, and optional `standard_deviation`. Call `uncertainty_interval(level=0.95)` when standard deviation is available. |
| `ReservoirFlowUpdate` | The streaming return value: `filtered_inflows` are causal and `revised_inflows` are absolute, finalized replacements. |
| `get_reservoir_inflow(storage, outflow, ..., include_uncertainty=False)` | Default acre-ft/cfs batch estimator. It returns the six documented inflow and provenance columns by default; opt in to append two standard-deviation columns. |
| `get_reservoir_inflow_from_config(storage, outflow, config, ..., include_uncertainty=False)` | Batch estimator using a `ReservoirConfig` and its unit system, with optional inflow standard deviations. |
| `run_inflow_model(observations, ..., include_uncertainty=False)` | DataFrame convenience form of `get_reservoir_inflow`; the frame must have `storage` and `outflow` columns. |
| `add_inflow_uncertainty_intervals(result, level=0.95)` | Returns a copy of an opt-in batch result with lower and upper normal-theory interval columns for causal and revised inflow. |
| `OutputFlag` | `NORMAL` or `PREDICTED` describes observation completeness; `SMOOTHED` or `NON_SMOOTHED` describes estimate provenance. |

Both batch series must share an exactly equal timezone-aware, strictly
increasing index with at most microsecond timestamp precision. The package
neither aligns nor cleans inputs. A revised
inflow always replaces the causal value at the same timestamp; it is never a
delta. See [model behavior](INFLOW_MODEL_BEHAVIOR.md) for initialization,
partial observations, and output columns.

### Optional inflow uncertainty

Uncertainty output is disabled by default. When `include_uncertainty=True`,
the streaming `ReservoirFlowEstimate.standard_deviation` is derived from the
inflow entry of the filtered or fixed-lag smoothed covariance. A disabled
stream sets the field to `None`. The opt-in batch schema appends
`estimated_inflow_standard_deviation` and
`revised_inflow_standard_deviation` after the existing six columns, so callers
that use the default keep the original schema.

Use `estimate.uncertainty_interval(level=0.95)` for one streaming estimate or
`add_inflow_uncertainty_intervals(result, level=0.95)` for a batch result. The
batch helper requires both standard-deviation columns, returns a copy, and
adds `estimated_inflow_lower`, `estimated_inflow_upper`,
`revised_inflow_lower`, and `revised_inflow_upper`. Unpublished revised values
and their uncertainty remain `NaN`, including the trailing row. A revision
replaces the causal value and standard deviation at its timestamp.

`level` must be finite and strictly between zero and one. The streaming
helper raises `ValueError` when standard deviation is unavailable; the batch
helper raises `ValueError` when uncertainty columns are missing or invalid.
Calling the batch helper again replaces its four interval columns using the
new level.

Standard deviations use `sqrt(P[1, 1])`. A negative inflow variance within
`1e-12 * max(1, abs(P[1, 1]))` of zero is treated as numerical roundoff and
clamped to zero; more negative or nonfinite covariance values raise
`ValueError`. This tolerance uses the inflow variance in the configured
squared flow units, without scaling it by storage covariance.

The intervals are pointwise, model-based normal-theory intervals for the net
inflow contribution. Bounds use the same flow units as the estimate and are
not clipped at zero. They reflect the selected filter covariance; uncertainty
in configuration or noise-parameter tuning, model bias, and other unmodeled
exchanges is excluded. The default startup covariance is an engineering
approximation, so early uncertainty should be interpreted with that caveat.

The reservoir inflow state is an inferred net balance contribution from the
supplied storage and accounted outflow. Measured outflow should cover outlet
releases, spills, and outward diversions or withdrawals as applicable.
Precipitation, evaporation, seepage, and other water exchanges are not
separate API terms. The linear-Gaussian estimate is not constrained to be
nonnegative; sensor bias, storage-datum changes, and rating-curve changes can
also appear as model mismatch.

## Configuration and units

| Export | Purpose |
| --- | --- |
| `ReservoirConfig` | Immutable reservoir identity, covariance, lag, units, version, and metadata. `q`, `r`, and `p0` are 3×3, 2×2, and 3×3 covariance matrices respectively. |
| `InitializationStrategy` | Initialization policy enum. The currently supported value is `FIRST_TWO_VALID_STORAGE`. |
| `InflowUnits` | Metadata enum: `CUBIC_FEET_PER_SECOND` or `SYSTEM_FLOW_RATE`. CFS requires a unit system whose flow label is `cfs`. |
| `UnitSystem` | Volume/flow labels and rate-to-volume-per-second conversion. Use `us_customary()`, `si()`, `flow_to_volume()`, and `volume_to_flow_rate()` as needed. |
| `CFS_TO_ACRE_FEET_PER_SECOND` | Conversion constant used by the default acre-ft/cfs model. |

`q` is continuous-time diffusion covariance, not a discrete fixed-interval Q
matrix. All covariance inputs must be finite, symmetric, and positive
semidefinite; `r` must have a strictly positive diagonal. See
[configuration](CONFIGURATION.md) for a complete example.

## Bayesian noise evaluation and tuning

The Bayesian search API is **experimental** and requires
`pip install "kalmanflow[tuning]"`. Its API, selection rules, and result schema may
change between releases. Review and independently validate any proposed
configuration. `evaluate_configuration` is available with the base installation;
the SciPy/scikit-learn optimizer dependencies load only when search is invoked.

| Export | Purpose |
| --- | --- |
| `ValidationWindow(name, start, end, weight=None)` | Named, timezone-aware half-open interval `[start, end)` for causal scoring. Window boundaries may use nanosecond precision even though scored observation timestamps are limited to microseconds. |
| `BayesianEvaluationSettings` | Shared warmup, innovation, calibration-warning, and numerical-stability settings. |
| `BayesianTuningSettings` | Trial counts, covariance multiplier bounds, acquisition settings, and optional upstream-proxy shape/timing gate. |
| `tune_inflow_noise_bayesian(...)` | Proposes five diagonal `q`/`r` terms from causal innovation scores. It needs at least three windows and a new proposed configuration version. |
| `BayesianTuningResult` | Proposed `selected_config`, parameters, trial summary, selected-window diagnostics, optional proxy diagnostics, warnings, and timing. |
| `evaluate_configuration(...)` | Scores one already-frozen configuration without searching or modifying it. |
| `ConfigEvaluationResult` | Frozen configuration, compact candidate summary, window diagnostics, and warnings. |
| `BayesianTuningError` | Raised when no candidate meets the tuning workflow's hard requirements. |

Tuning requires diagonal `q`, `r`, and `p0`, clean matched pandas series, at
least three positive inflow-increment seeds, and at least three distinct,
non-overlapping windows. Every seed is evaluated, so `total_trials` must be at
least the number of seeds. An optional upstream series is a shape/timing
diagnostic only, not a total-inflow label or Kalman observation. See
[Bayesian noise tuning](BAYESIAN_NOISE_TUNING.md) for selection and report
details.

## Advanced state-space interfaces

These exports make it possible to use KalmanFlow's filtering and smoothing with
a different state-space model. They operate on NumPy arrays and do not impose
reservoir units.

| Export | Purpose |
| --- | --- |
| `kalman_filter(...)` | Batch linear-Gaussian filter. Transition, process, offset, and control arrays may be shared or have exactly `n_times - 1` entries; observation arrays may be shared or have exactly `n_times` entries. Model arrays must be finite, covariance arrays symmetric positive semidefinite, and `NaN` is the only omitted observation value. |
| `predict_state(...)` | Predicts one state mean and covariance, optionally with a control offset. |
| `initial_filter_step(...)` / `kalman_step(...)` | Create immutable timestamped filter records for the initial or a later update. |
| `FilterStep` / `KalmanFilterResult` | Filter records and complete batch outputs, including innovations, innovation covariance, update mask, transitions, and log likelihood. |
| `smooth_filter_steps(steps)` | Runs full-record RTS smoothing and returns one `SmoothedStep` per input step. Empty input returns an empty tuple. |
| `OnlineFixedLagRTS` | Time-lagged RTS smoother with `add_step`, `provisional`, active-step inspection, and a bounded memory window. |
| `SmoothedStep` | Immutable smoothed timestamp, state mean/covariance, and provenance flags. |
| `StateSpaceModel` / `ReservoirStateSpaceModel` | Protocol and physical three-state reservoir implementation. The reservoir state order is `[storage, inflow_rate, true_outflow_rate]`. |
| `ReservoirBackend` | Connects `ReservoirStateSpaceModel` to the generic pipeline. |
| `OnlineInflowPipeline` / `PipelineUpdate` | Generic ordered-stream coordinator and its per-call results. Use when supplying a compatible custom backend and smoother. |

`OnlineFixedLagRTS` releases a state once the elapsed-time lag has passed and
raises `OverflowError` if its active window reaches `max_window_steps` before
release. Generic pipeline state and protocol types are available from
`kalmanflow.pipeline` for custom checkpoint/replay implementations, but the
reservoir checkpoint byte format remains internal.
