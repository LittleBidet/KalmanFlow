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

## Choosing initial values

Start from data quality and expected variability, then validate against historical periods that include stable conditions and storms.

- Set `r[0, 0]` near the variance of storage measurement error, in squared storage units.
- Set `r[1, 1]` near the variance of measured discharge error, in squared flow-rate units.
- Use `q[1, 1]` and `q[2, 2]` to control how quickly the estimated inflow and true outflow rates can change. Larger values react faster but can follow noise.
- Use `q[0, 0]` only when unmodelled storage movement or storage-model error needs explicit process uncertainty.
- Set `p0` large enough to reflect uncertainty at stream startup; avoid treating arbitrary initial rate estimates as precise.

There is no portable numeric range for these terms: their magnitudes depend on the selected units, sampling cadence, reservoir scale, sensor precision, and operating regime. Record the data interval, units, objective, and validation result in `tuning_metadata`.

## Tuning workflow

1. Prepare a representative historical interval with timezone-aware, strictly increasing timestamps. The first two storage readings and first discharge reading must be finite; later storage or discharge samples may be `NaN`.
2. Create `NoiseTuningData` and a baseline `ReservoirConfig`.
3. Run `tune_noise` using `loglik` for the joint likelihood fit, or `rmse` for standardized one-step innovation RMSE.
4. Review the returned values and behavior on a separate holdout period before promoting them.
5. Create a new immutable configuration using `config.with_tuned_noise(...)`; persist the configuration and its metadata in the calling application.

```python
import numpy as np
from kalmone import NoiseTuningData, tune_noise

data = NoiseTuningData(
    timestamps=tuple(storage.index.to_pydatetime()),
    storage=storage.to_numpy(dtype=float),
    discharge=discharge.to_numpy(dtype=float),
)

result = tune_noise(config, data, objective="loglik", maxiter=500)
candidate = config.with_tuned_noise(
    q=np.diag([result.q_storage, result.q_inflow, result.q_outflow]),
    r=np.diag([result.r_storage, result.r_outflow]),
    tuning_metadata={
        "objective": result.objective,
        "objective_value": result.objective_value,
        "method": result.method,
        "iterations": result.iterations,
    },
)
```

`tune_noise` never changes `config` and does not save results. The caller owns review, versioning, approval, and persistence.
