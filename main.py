from datetime import timedelta

import pandas as pd

from kalmone import get_reservoir_inflow

index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
my_inflow = get_reservoir_inflow(
    reservoir_storage=pd.Series(
        [100.0, 100.9, float("nan"), 102.0, 103.2], index=index
    ),
    reservoir_outflow=pd.Series([4.0, 6.0, 6.5, 5.0, 2.0], index=index),
    # Example continuous-time diffusion densities and measurement variances.
    q_storage=0.003270083668152831,
    q_inflow=0.00007840696223878258,
    q_outflow=0.0001,
    r_storage=6.8218763002709935,
    r_outflow=4.0,
    smoothing_lag=timedelta(hours=4),
)
print(my_inflow)
