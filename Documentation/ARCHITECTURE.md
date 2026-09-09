# KalmanFlow architecture

`kalmanflow` is a Python package for estimating reservoir inflow from measured storage and discharge. It implements a three-state, linear-Gaussian water-balance model plus a causal Kalman filter and fixed-lag RTS smoothing. It is a library: callers provide clean, aligned data and select or supply the reservoir configuration.

## Package layout

| Module | Responsibility |
| --- | --- |
| `kalmanflow.kalman` | Generic predict, update, and batch filtering primitives. |
| `kalmanflow.rts` | Full-record and bounded online fixed-lag RTS smoothing. |
| `kalmanflow.models` | The physical three-state reservoir model and model protocol. |
| `kalmanflow.reservoir_backend` | Adapts the physical model to the streaming pipeline. |
| `kalmanflow.pipeline` | Generic initialization, ordering, replay, and delayed-release coordinator. |
| `kalmanflow.core` | Public batch and reservoir-streaming adapters. |
| `kalmanflow.pandas_api` | DataFrame convenience wrapper for the default batch adapter. |
| `kalmanflow.bayesian_tuning` | Public façade for Bayesian diagonal-noise search and compact frozen-configuration evaluation; implementation is split across preparation, diagnostics, candidate, optimization, proxy, and workflow modules. |
| `kalmanflow.reservoir_config` | Immutable, validated per-reservoir configuration. |
| `kalmanflow.observations` / `flags` | Public input and output-provenance types. |
| `kalmanflow.units` | Volume/flow-rate conversion systems. |
| `kalmanflow.time_utils` | UTC normalization and elapsed-time validation helpers. |

The package-level imports are the supported starting point for applications.
The generic Kalman, RTS, model, backend, and pipeline types are also exported
for advanced integrations; their contracts are listed in the
[API reference](API_REFERENCE.md). Checkpoint encoding and low-level validation
modules are internal implementation details and are not public serialization
formats or extension points.

Repository-specific workflows live outside the package in `applications/`:
`applications/preparation.py` parses and aligns the checked-in Aquarius
exports, while `applications/run_bayesian_tuner.py` provides the offline
Bayesian calibration entry point.
`Notebooks/prepare_reservoir_data.py` remains a compatibility import for
existing notebooks.

## Reservoir state-space model

The default state is `[storage, inflow_rate, true_outflow_rate]`. Storage and measured discharge are observations; discharge is not treated as a known control. For an interval `dt` seconds, the first state row is:

```text
storage(t + dt) = storage(t)
                + dt × flow_to_volume_per_second
                    × (inflow_rate(t) - true_outflow_rate(t))
```

The rate states are random walks. `ReservoirStateSpaceModel` derives the transition and discrete process covariance from the actual elapsed time and the continuous-time 3×3 diffusion covariance `q_continuous`.

The `inflow_rate` state is the residual net balance contribution required by
the supplied storage and accounted outflow. Measured outflow should include
outlet releases, spills, and outward diversions or withdrawals as applicable.
The API has no separate terms for precipitation, evaporation, seepage, or
other water exchanges; applications must account for those separately with
justified inputs or interpret them as model mismatch. Sensor bias, storage
datum changes, and rating-curve changes are additional mismatch sources.
Estimates are unconstrained by sign, so a negative value is mathematically
valid even when a physical interpretation would prompt data or model review.

`UnitSystem.us_customary()` is the default: storage is acre-feet and flow rates are cfs. `UnitSystem.si()` supports cubic metres and cubic metres per second; custom unit systems supply labels and a rate-to-volume-per-second conversion.

## Public layers

Use `get_reservoir_inflow` for a pandas batch result, or `OnlineReservoirInflow` for a reservoir-specific stream. The latter is built on the generic `OnlineInflowPipeline`, which can also be used with another backend and smoother implementation.

The public reservoir layer exposes causal filtered inflow after its
two-observation startup phase and later exposes an absolute fixed-lag RTS
revised inflow for the same timestamp. The first finite-storage observation
anchors the stream; it does not produce an inflow output by itself.
Storage, measured outflow, and latent true outflow remain model inputs or
internal state; no public outflow result is exposed. Generic filter and
smoothing types remain available for advanced integrations.

Offline calibration uses `tune_inflow_noise_bayesian` with clean, aligned
storage and discharge series. It proposes a new immutable configuration but
does not persist or activate it. `evaluate_configuration` scores one frozen
configuration on a separate period without performing another search.

## Data ownership and boundaries

KalmanFlow does not read files, fetch or clean source data, align series, or
deduplicate records. Callers manage reviewed configuration artifacts; the
package validates the runtime contract—including timestamp indexes,
matrix shapes, covariance properties, and missing observations—at the library
boundary.
