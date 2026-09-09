# Configuration

## Complete configuration example

Use `ReservoirConfig` when a reservoir has reviewed noise settings, units, and metadata. The values below are structurally valid examples, not recommended production parameters.

```python
from datetime import timedelta

import numpy as np

from kalmanflow import (
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
from kalmanflow import OnlineReservoirInflow, get_reservoir_inflow_from_config

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
| `q` | 3×3 | Continuous-time process-diffusion covariance; `q[i,j]` has units `state_i × state_j / second`. |
| `r` | 2×2 | Measurement covariance for `[storage, measured_outflow]`; `r[i,j]` has units `observation_i × observation_j`. |
| `p0` | 3×3 | Initial covariance; `p0[i,j]` has units `state_i × state_j`. |

All covariance matrices must be finite, symmetric, and positive semidefinite. The two diagonal elements of `r` must be strictly positive. KalmanFlow derives each discrete process covariance from `q` and the actual elapsed time between observations, so never reuse a discrete, fixed-interval Q matrix as `q`.

The balance model infers a net storage balance contribution. Measured outflow
should include outlet releases, spills, and outward diversions or withdrawals
as applicable. Precipitation, evaporation, seepage, and other water exchanges
are not separate model terms; account for them with separate justified inputs
or document them as model mismatch. Sensor bias, storage-datum changes, and
rating-curve changes are additional mismatch sources. Estimates are not
constrained to be nonnegative.

`UnitSystem.us_customary()` uses acre-feet and cfs; `UnitSystem.si()` uses m³ and m³/s. A custom `UnitSystem` must supply a positive `flow_to_volume_per_second` conversion that matches the storage unit.

## Selecting parameters

Choose `q`, `r`, and `p0` through an engineering review of each reservoir's
sensor accuracy, operating conditions, and historical data. Record the
rationale in `metadata`, version the reviewed configuration, and validate it
against a separate period before operational use.

## Bayesian inflow-noise tuning

For an offline proposal, use `tune_inflow_noise_bayesian` with a fixed
`ReservoirConfig` and predeclared `ValidationWindow` intervals. The supplied
`inflow_increment_sd_seeds` initialize the bounded Bayesian search for the five
diagonal covariance terms. Bayesian evaluation requires `q`, `r`, and `p0` to
be diagonal. The tuner runs a causal chronology per trial and scores the joint
predictive innovation for whichever storage and outflow components are
observed, before that row is assimilated. The returned configuration is
proposed only; it is never persisted automatically. Supply a new
`proposed_configuration_version`, review the compact report, and use
`evaluate_configuration` with an explicit untouched `ValidationWindow` before
operational approval.
