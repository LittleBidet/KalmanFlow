"""Check an installed kalmanflow build from outside the source checkout.

Run with the interpreter of a clean environment, for example
``.venv-smoke/bin/python -I .github/scripts/smoke_test.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
from datetime import timedelta
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-version", help="required installed version")
    parser.add_argument(
        "--allow-scipy",
        action="store_true",
        help="skip the check that the core install works without SciPy",
    )
    arguments = parser.parse_args()

    import numpy as np
    import pandas as pd

    import kalmanflow

    package = Path(kalmanflow.__file__).resolve()
    checkout_source = Path(__file__).resolve().parents[2] / "src"
    if checkout_source in package.parents:
        raise SystemExit(f"kalmanflow was imported from the checkout: {package}")

    installed = version("kalmanflow")
    if arguments.expect_version not in (None, installed):
        raise SystemExit(
            f"expected kalmanflow {arguments.expect_version}, found {installed}"
        )

    if not arguments.allow_scipy and importlib.util.find_spec("scipy") is not None:
        raise SystemExit("SciPy is installed, so the core-only check proves nothing")

    count = 48
    index = pd.date_range("2026-01-01", periods=count, freq="15min", tz="UTC")
    storage = pd.Series(10_000.0 + 0.5 * np.arange(count), index=index)
    outflow = pd.Series(25.0, index=index)
    result = kalmanflow.get_reservoir_inflow(
        storage,
        outflow,
        q_storage=1.0,
        q_inflow=1.0,
        q_outflow=1.0,
        r_storage=100.0,
        r_outflow=25.0,
        smoothing_lag=timedelta(hours=1),
    )
    if not result.index.equals(index):
        raise SystemExit("the batch result does not follow the input timestamps")
    if not np.isfinite(result["estimated_inflow"].to_numpy(dtype=float)).all():
        raise SystemExit("estimated_inflow contains non-finite values")
    # Revisions are NaN until the smoothing lag has passed, so only the
    # released rows must be finite.
    released = result["revised_inflow_smoothing_flag"] == "SMOOTHED"
    revised = np.isfinite(result["revised_inflow"].to_numpy(dtype=float))
    if not released.any() or not (revised == released.to_numpy()).all():
        raise SystemExit("revised_inflow does not match its smoothing flags")

    print(f"kalmanflow {installed} from {package.parent} passed the smoke test")


if __name__ == "__main__":
    main()
