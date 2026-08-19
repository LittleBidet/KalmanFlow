# Kalmone model behavior

## Input contract

`get_reservoir_inflow` and `get_reservoir_inflow_from_config` accept pre-cleaned storage and discharge `pandas.Series` with exactly equal, timezone-aware, strictly increasing indexes. The default adapter expects storage in acre-feet and discharge in cfs. The configured adapter uses the labels in `ReservoirConfig.unit_system`.

`OnlineReservoirInflow.process` accepts either an `Observation` or `timestamp`, `storage`, and `discharge` keywords. Timestamps must be timezone-aware and strictly later than the previously accepted observation. All elapsed-time comparisons are normalized to UTC.

The package intentionally does not sort, align, deduplicate, interpolate, or impute inputs.

## Initialization and missing values

Initialization waits for two finite storage readings and a finite discharge at the first of those readings. The first inflow rate is calculated with the water balance over those two storage samples. The second discharge may be missing.

After initialization, either storage or discharge may be `NaN`. A finite component is still used as a partial Kalman observation. If both are `NaN`, the step is predict-only. A public estimate carrying any missing observation component receives the `PREDICTED` flag; fully observed steps are `NORMAL`.

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

## Streaming outputs

Each call to `OnlineReservoirInflow.process` returns a `ReservoirFlowUpdate`.
`filtered_inflows` contains newly available causal inflow estimates;
`revised_inflows` contains only newly finalized, absolute smoothed
replacements at their original timestamps. The initial successful call that
completes initialization can emit two filtered inflow estimates. The active
smoothing window is retained internally and is bounded by `max_window_steps`.

`process_many` is transactional: if any item is invalid, the stream returns to its entry state and produces no partial group result.

## Configuration and tuning

`ReservoirConfig` is immutable. It holds a reservoir identifier, model and configuration versions, continuous-time `q` (3×3), measurement covariance `r` (2×2), initial covariance `p0` (3×3), a positive smoothing lag, units, and metadata. Covariances must be finite, symmetric, positive semidefinite; the diagonal of `r` must be strictly positive.

`tune_inflow_model` selects diagonal continuous-time process noise (`q_storage`,
`q_inflow`, `q_outflow`) and observation noise (`r_storage`, `r_outflow`) from
storage and outflow only. One-step Gaussian likelihood can reward very small
measurement noise and large inflow process noise because the filter can follow
sensor noise immediately. The dataframe-first tuner therefore scores causal
one-, six-, and 24-hour forecasts with a blocked multi-horizon Student-t
predictive negative log-likelihood. The default horizon weights are 0.50,
0.30, and 0.20; the horizons and weights are configurable.

At every forecast origin, the tuner first assimilates observations through that
origin, then propagates the filtered state to each target without assimilating
the intervening observations. The target is the first future observation within
half the reservoir's median cadence of the requested lead. Irregular timestamps
use their actual elapsed seconds for every propagation. Missing storage or
outflow components are omitted from the corresponding predictive likelihood.
Student-t degrees of freedom default to 5.0 so isolated sensor spikes have less
influence than under a Gaussian likelihood. Validation uses four contiguous
blocks distributed through the tuning record, and candidate performance is the
median block loss.

All objective forecasts and inflow-behavior diagnostics use causal filtered
inflows only; revised, centered, and fixed-lag-smoothed estimates are never
used for tuning. Sensor-derived parameter bounds remain recommended because
forecast scoring does not fully identify all five noise parameters. The tuner
never mutates the input dataframe, does not require SciPy, and
`tune_reservoirs` runs the same isolated process for a mapping of reservoirs
while retaining independent failures.
