# Configuration

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
    metadata={"source": "initial calibration"},
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

## Selecting parameters

Choose `q`, `r`, and `p0` through an engineering review of each reservoir's
sensor accuracy, operating conditions, and historical data. Record the
rationale in `metadata`, version the reviewed configuration, and validate it
against a separate period before operational use.

## Conservative inflow-noise tuning

For an offline proposal, use `tune_inflow_process_noise` with a fixed
`ReservoirConfig` and predeclared `TuningWindow` intervals. Candidates are
specified as `prior_hourly_increment_sd`; the tuner converts each value to the
continuous random-walk diffusion `q_inflow = sd**2 / 3600`. Only `q[1, 1]`
changes. `q_storage`, `q_outflow`, `r`, and `p0` remain fixed, and off-diagonal
covariance matrices are rejected.

The tuner runs one causal chronology per candidate and scores storage before
each observation is assimilated. Equal validation-window weights and a paired
uncertainty rule select the smallest candidate whose loss is practically
equivalent to the best candidate. The returned configuration is proposed only;
it is never persisted automatically. Supply a new
`proposed_configuration_version` and review the warning, regime, horizon, and
optional measurement-noise sensitivity tables before operational approval.
