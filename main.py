"""Convenient pandas entry point for estimating reservoir inflow.

Pass a timezone-aware, increasing time series with ``storage`` (acre-ft) and
``outflow`` (cfs) columns. The returned DataFrame is indexed by the same
timestamps and includes causal/revised inflow values and quality flags.
"""

from datetime import timedelta

import pandas as pd

from kalmanflow.pandas_api import run_inflow_model

if __name__ == "__main__":
    index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    sample = pd.DataFrame(
        {
            "storage": [100.0, 100.9, float("nan"), 102.0, 103.2],
            "outflow": [4.0, 6.0, 6.5, 5.0, 2.0],
        },
        index=index,
    )

    result = run_inflow_model(
        sample,
        q_storage=0.003270083668152831,
        q_inflow=0.00007840696223878258,
        q_outflow=0.0001,
        r_storage=6.8218763002709935,
        r_outflow=4.0,
        smoothing_lag=timedelta(hours=0.25),
    )
    print(result)
