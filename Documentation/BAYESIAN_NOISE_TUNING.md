# Bayesian innovation noise tuning

## Scope and inputs

`applications/run_bayesian_tuner.py` is the offline tuning workflow. Its
`run_bayesian_tuner(...)` function accepts `project_root`, `reservoir`,
`data_start`, `data_end`, `output_path`, and the optional
`report_output_path`. It searches five diagonal covariance terms informed by
the observed storage and outflow innovations:

* `q_storage`, `q_inflow`, and `q_outflow`;
* `r_storage` and `r_outflow`.

The package API accepts already-cleaned `pandas.Series` inputs. Storage and
discharge indexes must match exactly and must be timezone-aware, strictly
increasing, unique, and nonmissing. The base configuration must use diagonal
`q`, `r`, and `p0` matrices with positive finite values for all five searched
diagonal `q`/`r` terms. At least three distinct positive
`inflow_increment_sd_seeds` and three uniquely named, non-overlapping,
half-open `ValidationWindow` intervals are required.
Every supplied seed is evaluated. Consequently, `total_trials` must be at
least the number of seeds; when `initial_trials` is smaller, the initial design
is expanded to include them all.

## Objective and selection

The tuner scores the causal, pre-update predictive innovation distribution. It
does not use true inflow values or smoothed states. Only rows inside the
declared validation windows contribute to the objective, although earlier
observations outside a window may causally condition its starting state. The
supplied hourly inflow increment seeds are used as deterministic initial trials
and define the `q_inflow` bounds.
Remaining trials are selected by expected improvement from a Gaussian-process
surrogate in normalized log-parameter space. Proxy diagnostics use only the
same warmup-adjusted validation rows as the innovation objective.

Each trial's objective is the validation-window-weighted joint predictive
negative log density per observed component, plus the configured multiple of
its between-window standard error. Hard validity checks exclude failed or
insufficiently scored trials. Statistically competitive trials are identified
with paired window differences; final selection prefers lower calibration
violation, then proximity to the reviewed base configuration, then objective.

## Output files and held-out evaluation

The runner writes the proposed immutable `ReservoirConfig` to `output_path`.
This compact configuration JSON is the default artifact and contains the
complete configuration needed by a downstream application. Without an
explicit `output_path`, it is written beneath `Outputs/bayesian_tuning`.

Pass `report_output_path` to additionally write a detailed tuning audit. The
report contains the compact candidate table (trial, acquisition source, five
parameters, objective, eligibility, and selection flags), selected-trial
per-window efficacy, compact upstream-proxy diagnostics when supplied, and a
reference to the selected configuration artifact. It does not duplicate the
full configuration JSON. The two resolved output paths must be different.

A separate final evaluation interval should be reserved and evaluated with
`evaluate_configuration` and an explicit `ValidationWindow` after tuning. The
application runner divides its requested segment into three calibration
windows; it does not reserve a final test segment automatically.

Runtime diagnostics distinguish Gaussian-process `acquisition_seconds` from
`candidate_evaluation_seconds`, which covers each compact causal evaluation.

By default, the Bayesian search cannot increase `q_storage` or `r_storage`
above the reviewed base configuration (`0.5x` to `1.0x`). This prevents an
innovation-only objective from explaining storm-driven storage changes as
extra storage process or measurement noise. These bounds remain configurable
through `BayesianTuningSettings` when independent evidence supports a wider
range. The inflow and outflow terms retain wider base-relative bounds.

## Optional upstream proxy

An upstream gauge may be supplied as an optional diagnostic proxy:

```python
from datetime import timedelta

from kalmone import (
    BayesianEvaluationSettings,
    BayesianTuningSettings,
    tune_inflow_noise_bayesian,
)

evaluation_settings = BayesianEvaluationSettings()

result = tune_inflow_noise_bayesian(
    storage,
    discharge,
    base_config,
    inflow_increment_sd_seeds,
    validation_windows,
    upstream_proxy=upstream_flow,
    settings=evaluation_settings,
    bayesian_settings=BayesianTuningSettings(
        proxy_max_lag=timedelta(hours=12),
        proxy_diagnostic_frequency="1h",
        proxy_min_shape_correlation=0.20,
        proxy_min_change_correlation=0.10,
    ),
    proposed_configuration_version="reviewed-bayesian-v1",
)
```

Proxy checks first match the gauge to the model timestamps, then perform a
bounded lag search using standardized level and change shape. They use only
the warmup-adjusted validation rows and aggregate at an hourly cadence by
default. They report best lag, level/change correlation, normalized shape
RMSE, and a gate flag in the result and JSON report. Proxy magnitude is
deliberately not matched and the proxy is never added to the Kalman observation
vector or treated as true total inflow. If at least one innovation-valid trial
passes the configured gate, selection is restricted to those trials; if none
passes, the result is returned with an explicit diagnostic-only warning.

The runtime preparation helper is `applications.preparation`. It retains
`upstream_flow` and spillway audit columns in `PreparedReservoirData`. When the
requested window contains at least one finite, nonnegative spillway value,
finite spillway values are added to outlet discharge; missing or negative
spillway values make combined outflow missing at those timestamps instead of
being treated as zero. If the spillway file is absent or the requested window
contains no usable spillway value, outlet-only outflow is preserved and the
window audit records that limitation. The application tuner uses this helper;
`Notebooks.prepare_reservoir_data` is retained only as a compatibility import
for existing notebooks.

## Application runner

Run the workflow from the project root:

```bash
uv run python applications/run_bayesian_tuner.py
```

Direct execution uses the reviewed constants near the top of the runner;
edit those constants first or call `run_bayesian_tuner(...)` from Python with
explicit arguments. Without an explicit `output_path`, the JSON configuration is
written beneath `Outputs/bayesian_tuning` as the compact selected
configuration. Pass `report_output_path` to additionally write the detailed
tuning audit report; it must resolve to a different file.

For example, to produce both artifacts explicitly:

```python
from pathlib import Path

from applications.run_bayesian_tuner import run_bayesian_tuner

run_bayesian_tuner(
    output_path=Path("Outputs/bayesian_tuning/chesbro-config.json"),
    report_output_path=Path("Outputs/bayesian_tuning/chesbro-report.json"),
)
```

The package API is:

```python
from kalmone import (
    BayesianEvaluationSettings,
    BayesianTuningSettings,
    ValidationWindow,
    evaluate_configuration,
    tune_inflow_noise_bayesian,
)

result = tune_inflow_noise_bayesian(
    storage,
    discharge,
    base_config,
    inflow_increment_sd_seeds,
    validation_windows,
    settings=BayesianEvaluationSettings(),
    bayesian_settings=BayesianTuningSettings(total_trials=40, initial_trials=12),
    proposed_configuration_version="reviewed-bayesian-v1",
)

test_result = evaluate_configuration(
    test_storage,
    test_discharge,
    result.selected_config,
    evaluation_window=ValidationWindow("final-test", test_start, test_end),
    settings=BayesianEvaluationSettings(),
)
```

`evaluate_configuration` performs no search and does not modify the supplied
configuration. It returns a compact `ConfigEvaluationResult` containing one
configuration summary, per-window causal diagnostics, and warnings. When
`evaluation_window` is omitted it evaluates the full supplied series, but an
explicit untouched window is recommended for final approval.
