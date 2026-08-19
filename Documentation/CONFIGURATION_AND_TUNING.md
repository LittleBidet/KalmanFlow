# Configuration and tuning

## Complete configuration example

Use `ReservoirConfig` when a reservoir has reviewed noise settings, units, and metadata. The values below are structurally valid examples, not recommended production parameters.

```python
from datetime import timedelta

import numpy as np

from kalmone import (
    InflowUnits,
    InitializationStrategy,
    ReservoirConfig,
    UnitSystem,
)

config = ReservoirConfig(
    reservoir_id="lexington",
    reservoir_name="Lexington Reservoir",
    q=np.diag([1.0, 0.01, 0.01]),
    r=np.diag([100.0, 25.0]),
    p0=np.diag([1_000.0, 100.0, 100.0]),
    smoothing_lag=timedelta(hours=12),
    initialization_strategy=InitializationStrategy.FIRST_TWO_VALID_STORAGE,
    inflow_units=InflowUnits.CUBIC_FEET_PER_SECOND,
    model_version="physical-rate-v1",
    configuration_version="2026-01",
    tuning_metadata={"source": "initial calibration"},
    unit_system=UnitSystem.us_customary(),
)
```

Then pass it to either adapter:

```python
from kalmone import OnlineReservoirInflow, get_reservoir_inflow_from_config

stream = OnlineReservoirInflow.from_config(config)
batch_result = get_reservoir_inflow_from_config(storage, discharge, config)
```

Both adapters use discharge as an observation input, but only return causal
and revised inflow. A revised inflow is an absolute fixed-lag-smoothed value
that replaces the causal inflow at its timestamp; it is not an adjustment to
add. The latent true outflow state is never a public result.

## Matrix and unit conventions

The state order is `[storage, inflow_rate, true_outflow_rate]`.

| Field | Shape | Meaning |
| --- | --- | --- |
| `q` | 3×3 | Continuous-time process-diffusion covariance in state units. |
| `r` | 2×2 | Measurement covariance for `[storage, measured_outflow]`. |
| `p0` | 3×3 | Initial covariance in state units. |

All covariance matrices must be finite, symmetric, and positive semidefinite. The two diagonal elements of `r` must be strictly positive. Kalmone derives each discrete process covariance from `q` and the actual elapsed time between observations, so never reuse a discrete, fixed-interval Q matrix as `q`.

`UnitSystem.us_customary()` uses acre-feet and cfs; `UnitSystem.si()` uses m³ and m³/s. A custom `UnitSystem` must supply a positive `flow_to_volume_per_second` conversion that matches the storage unit.

## Tuning workflow

1. Prepare a representative dataframe with a timezone-aware, strictly increasing `DatetimeIndex` and `storage` and `outflow` columns. Missing components may remain `NaN`.
2. Run the dataframe-first tuner. It estimates robust starting values from storage and outflow differences, raw water-balance inflow, cadence, and variability; raw inflow is never treated as a measured target.
3. Review the independent configuration and its predictive score before promotion.
4. Explicitly save the reviewed configurations. The file contains configuration and tuning metadata only, never observations or inflow output.

```python
from kalmone import (
    OnlineReservoirInflow,
    load_reservoir_configs,
    tune_reservoirs,
)

batch = tune_reservoirs(
    {
        "lexington": lexington_dataframe,
        "anderson": anderson_dataframe,
    },
    max_evaluations=64,
    max_tuning_rows=5_000,
    workers=2,
)
batch.save("reservoir-configurations.json")
configs = load_reservoir_configs("reservoir-configurations.json")
models = {
    reservoir_id: OnlineReservoirInflow.from_config(config)
    for reservoir_id, config in configs.items()
}
```

The default dataframe objective is
`robust_multihorizon_student_t_predictive_negative_log_likelihood`. It uses
one-, six-, and 24-hour causal forecast horizons with weights 0.50, 0.30, and
0.20, Student-t degrees of freedom 5.0, and four validation blocks distributed
through the record. Configure these with `forecast_horizons` (numeric values
are hours, or use `timedelta`), `horizon_weights`,
`student_t_degrees_of_freedom`, and `validation_blocks` on either tuning API.
Each target is matched to the first future observation within half the median
cadence. A forecast starts from the filtered state at its origin and does not
assimilate observations between the origin and target, so neither tuning nor
its inflow diagnostics use revised or centered estimates.

The Student-t predictive NLL for observed component vector `e`, predictive
covariance `S`, dimension `d`, and degrees of freedom `ν` is:

```text
lgamma(ν / 2) - lgamma((ν + d) / 2)
+ 0.5 logdet(S) + (d / 2) log(νπ)
+ ((ν + d) / 2) log1p(eᵀ S⁻¹ e / ν)
```

Each horizon is normalized by its usable observed scalar components, and
weights are renormalized when a horizon is unavailable. The aggregate score is
the median block loss, which limits the influence of a single storm or sensor
failure period. Review the returned per-horizon counts and losses, per-block
losses, skipped-forecast counts, and causal inflow roughness diagnostics.

`tune_inflow_model` is the equivalent single-reservoir entry point. It returns
the five selected parameters, immutable `ReservoirConfig`, score, evaluation
count, diagnostics, and full model output. Tuning occurs offline; production
runs should load versioned reviewed configurations.

The legacy `tune_noise(..., objective="loglik")` adapter deliberately retains a
separate one-step Gaussian scoring path for compatibility; its result must not
be interpreted as the new dataframe objective. The dataframe-tuned
configuration version is bumped when these tuning semantics change.
