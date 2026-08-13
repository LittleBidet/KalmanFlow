# Kalmone architecture

`kalmone` is a Python package for estimating reservoir inflow from measured storage and discharge. It implements a three-state, linear-Gaussian water-balance model plus a causal Kalman filter and fixed-lag RTS smoothing. It is a library: callers provide clean, aligned data and select or supply the reservoir configuration.

## Package layout

| Module | Responsibility |
| --- | --- |
| `kalmone.kalman` | Generic predict, update, and batch filtering primitives. |
| `kalmone.rts` | Full-record and bounded online fixed-lag RTS smoothing. |
| `kalmone.models` | The physical three-state reservoir model and model protocol. |
| `kalmone.reservoir_backend` | Adapts the physical model to the streaming pipeline. |
| `kalmone.pipeline` | Generic initialization, ordering, replay, and delayed-release coordinator. |
| `kalmone.core` | Public batch and reservoir-streaming adapters. |
| `kalmone.reservoir_config` | Immutable, validated per-reservoir configuration. |
| `kalmone.observations` / `flags` | Public input and output-provenance types. |
| `kalmone.tuning` | Optional SciPy-backed Q/R optimization. |
| `kalmone.units` | Volume/flow-rate conversion systems. |

## Reservoir state-space model

The default state is `[storage, inflow_rate, true_outflow_rate]`. Storage and measured discharge are observations; discharge is not treated as a known control. For an interval `dt` seconds, the first state row is:

```text
storage(t + dt) = storage(t)
                + dt × flow_to_volume_per_second
                    × (inflow_rate(t) - true_outflow_rate(t))
```

The rate states are random walks. `ReservoirStateSpaceModel` derives the transition and discrete process covariance from the actual elapsed time and the continuous-time 3×3 diffusion covariance `q_continuous`.

`UnitSystem.us_customary()` is the default: storage is acre-feet and flow rates are cfs. `UnitSystem.si()` supports cubic metres and cubic metres per second; custom unit systems supply labels and a rate-to-volume-per-second conversion.

## Public layers

Use `get_reservoir_inflow` for a pandas batch result, or `OnlineReservoirInflow` for a reservoir-specific stream. The latter is built on the generic `OnlineInflowPipeline`, which can also be used with another backend and smoother implementation.

The public reservoir layer exposes causal filtered inflow immediately and only exposes outflow when its fixed-lag RTS estimate is final. Generic filter and smoothing types remain available for advanced integrations.

## Data ownership and boundaries

Kalmone does not read files, fetch data, parse timestamps, align series, deduplicate records, or persist tuning artifacts. Those concerns stay in the calling application. The package validates the runtime contract—including timestamps, matrix shapes, covariance properties, and missing observations—at the library boundary.

