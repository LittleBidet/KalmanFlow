# Experimental maximum-likelihood noise tuning

Noise tuning is experimental. Its API, selection rules, and result schema
may change between releases. Treat selected settings as proposals requiring
independent validation before operational use.

Install the optional search dependency with `pip install "kalmanflow[tuning]"`.
SciPy loads only when `tune_inflow_noise` is called. Importing KalmanFlow,
filtering, smoothing, and `evaluate_configuration` work with the base
installation.

## Scope and inputs

`tune_inflow_noise` estimates five diagonal covariance terms from observed
storage and outflow innovations, without true inflow labels:

* `q_storage`, `q_inflow`, and `q_outflow`;
* `r_storage` and `r_outflow`.

The package API accepts already-cleaned `pandas.Series` inputs. Storage and
discharge indexes must match exactly and must be timezone-aware, strictly
increasing, unique, and nonmissing. The base configuration must use diagonal
`q`, `r`, and `p0` matrices with positive finite values for all five tuned
terms. Supply at least one positive `inflow_increment_sd_seeds` value (an
hourly inflow increment standard deviation) and at least three uniquely named,
non-overlapping, half-open `ValidationWindow` intervals. Five or more windows
are recommended; with fewer, the result carries a weak-uncertainty warning.

## Choosing a tuning period

The tuner fits one set of constant noise terms, so tune on a period where
the gauge and its processing did not change. Before tuning, scan the record
month by month:

```python
import pandas as pd
from kalmanflow import ValidationWindow, estimate_storage_noise

months = pd.date_range(storage.index[0].floor("D"), storage.index[-1], freq="MS")
scan = estimate_storage_noise(
    storage,
    [ValidationWindow(f"{m:%Y-%m}", m, m + pd.offsets.MonthBegin(1)) for m in months],
)
```

* **Stable:** a run of months whose `r_storage` values stay within about a
  factor of 4 of each other, with a similar `unchanged_share`.
* **Regime change:** a lasting step in `r_storage`, usually together with a
  change in how storage is recorded (value precision, step size, or the
  approval and grade codes in the source export). Tune each regime
  separately, and apply each configuration only to data from its own regime.
* **A single zero month** in a wet season is usually storm-dominated rather
  than a gauge change. Only a run of zeros or a lasting shift is evidence of
  a new regime.
* **Cover the conditions you will estimate.** A configuration tuned on a
  dry period can respond slowly to storms in real time, so include storm
  events if causal storm response matters.

Keep a final test period out of the tuning period and check the frozen
proposal on it with `evaluate_configuration`.

## Objective and search

The objective is the validation-window-weighted causal joint predictive
negative log density (NLPD) per observed component. It is the negative
log-likelihood of the pre-update innovations, so minimizing it is
maximum-likelihood estimation. Only rows inside the declared windows, after
warmup, contribute; earlier observations may still condition a window's
starting state. Smoothed states are never used.

Each seed gives one starting point: the base configuration's terms with
`q_inflow` set from the seed. All starts are scored, and the best
`local_searches` of them are refined with bounded L-BFGS-B in log-parameter
space, each with at most `max_evaluations_per_search` objective evaluations.
The valid local optimum with the lowest objective is proposed, unless the
base configuration scores lower: then the proposal keeps the base terms and
a warning says so. Results are deterministic.

Pass `progress=callback` to follow a long run. The callback receives a
`NoiseTuningProgress` at the start of each phase (`screening`, `search-1` and
so on, then `identifiability`) and after every filtering pass; each phase's
`evaluation_limit` is a bound, not an exact total, because searches often
finish early. Exceptions raised by the callback propagate and cancel the run.

Each search bound is a multiple of the matching base term
(`*_multiplier_bounds` in `NoiseTuningSettings`), so the base `q_inflow`
must be realistic, and every seed must fall inside the `q_inflow` bounds. By
default `q_storage` cannot exceed its reviewed base value: the innovation
likelihood alone cannot tell a rapid inflow event from an over-dispersed
storage channel. A selected term on a bound is reported, because the
likelihood optimum may lie outside the box.

Storage reading noise is measured from the data before the search. Reading
errors make consecutive storage changes negatively correlated, so
`r_storage` is estimated as minus the lag-1 autocovariance of storage
changes. Storm-driven changes are persistent and can hide that signal, so
the estimate is taken in one-week blocks over the scored rows and the median
across weeks is used. The search then keeps `r_storage` within a factor of
`r_storage_estimate_factor` (default 2) of that estimate. When the data give
no positive estimate (fewer than 100 evenly spaced change pairs, or no
reading-noise signal), the base-relative `r_storage_multiplier_bounds` apply
instead, with a warning that says which. Set `estimate_r_storage=False` to
always use those bounds. The per-window estimates are in the result's
`storage_noise` table. A window that gives no estimate is named in a warning
and left out of the comparison. If the remaining windows disagree by more
than a factor of 4, which often points to a gauge or processing change, a
warning suggests comparing periods with
`estimate_storage_noise(storage, validation_windows)` and tuning them
separately. That function needs no SciPy and can be run before choosing
windows.

## Diagnostics and warnings

Diagnostics never change the selection. The result reports:

* **Improvement over base.** `base_objective`, `selected_objective`, and the
  paired per-window standard error `improvement_standard_error`. A warning is
  raised when the improvement does not exceed that error (or
  `practical_equivalence_tolerance`).
* **Calibration.** Per-channel NIS mean (ideal 1) and variance (ideal 2),
  innovation bias, the share of observations inside the central 50% and 90%
  predictive intervals, and the largest elapsed-lag innovation
  autocorrelation. Thresholds in `NoiseEvaluationSettings` turn failures into
  warnings; set a threshold to `None` to disable it. A heavy-tailed or
  serially correlated innovation sequence fails these checks at any noise
  setting, which points at the model structure rather than the tuning.
* **Identifiability.** For each selected term not on a bound: the curvature
  of the total negative log-likelihood in log space, an approximate standard
  error of the log value, and the correlations implied by the inverse
  Hessian. Window weights and sizes do not change these figures. Terms
  uncertain by more than a factor of e, and pairs correlated beyond ±0.95,
  are named in warnings, and so is a term whose curvature could not be
  measured because a nearby step failed the validity checks; its standard
  error is NaN. The standard errors assume a correct Gaussian model.
  The check adds about 2n² objective evaluations for n free terms; disable it
  with `check_identifiability=False`.
* **Search status.** Each local search's evaluation count, convergence flag,
  and stopping message.

`eligible` means a configuration has the required number of finite scored
storage observations in every window, finite joint NLPD values, and no
numerical failure. `calibrated` means it is eligible and raises no
calibration warning. Both tuning and `evaluate_configuration` use the same
checks.

## Optional upstream proxy

An upstream gauge may be passed as `upstream_proxy`. It is a shape and timing
diagnostic only: it is never added to the Kalman observation vector, never
treated as true total inflow, and never affects the selection. The proxy is
matched to the model timestamps, aggregated to `proxy_diagnostic_frequency`
over the scored validation rows, and compared with the causal inflow
estimate by level and change correlation over whole-cadence shifts within
`proxy_max_lag`. Correlation is scale-free, so amplitudes need not match. A
positive reported lag means the proxy is shifted later relative to the causal
estimate. A selected configuration that fails the correlation thresholds
gets a warning.

## Output and held-out evaluation

`NoiseTuningResult.candidate_summary` has one row for the base configuration
and one per local search. `window_diagnostics` covers the base and selected
configurations. The proposed `selected_config` records the method, tuned
terms, and objectives in its metadata; it is never persisted automatically.
Supply a new `proposed_configuration_version`.

Reserve a separate final period and evaluate the frozen proposal with
`evaluate_configuration` and an explicit `ValidationWindow`.
`evaluate_configuration` performs no search and does not modify the supplied
configuration. It returns a compact `ConfigEvaluationResult` with one
configuration summary, per-window causal diagnostics, and warnings. Test
results must not feed back into the search, windows, or thresholds.

## Application runner

`applications/run_noise_tuner.py` is the repository's offline workflow. It
prepares the checked-in Aquarius data, splits the requested segment into six
validation windows, and writes the proposed `ReservoirConfig` JSON beneath
`Outputs/noise_tuning` unless `output_path` is given. Pass
`report_output_path` to also write the detailed audit report; the two paths
must differ. Progress is written to stderr: in a terminal, a total bar and a
bar for the current phase; when redirected to a file, a line every ten
filtering passes. Run it from the project root:

```bash
uv run --extra tuning python applications/run_noise_tuner.py
```

Direct execution uses the reviewed constants near the top of the runner; edit
them or call `run_noise_tuner(...)` from Python with explicit arguments.

The runtime preparation helper is `applications.preparation`. It retains
`upstream_flow` and spillway audit columns in `PreparedReservoirData`. When the
requested window contains at least one finite, nonnegative spillway value,
finite spillway values are added to outlet discharge; missing or negative
spillway values make combined outflow missing at those timestamps instead of
being treated as zero. If the spillway file is absent or the requested window
contains no usable spillway value, outlet-only outflow is preserved and the
window audit records that limitation. `Notebooks.prepare_reservoir_data` is
retained only as a compatibility import for existing notebooks.

## Package API

```python
from kalmanflow import (
    NoiseEvaluationSettings,
    NoiseTuningSettings,
    ValidationWindow,
    evaluate_configuration,
    tune_inflow_noise,
)

result = tune_inflow_noise(
    storage,
    discharge,
    base_config,
    inflow_increment_sd_seeds=(2.0, 5.0, 10.0, 20.0, 40.0),
    validation_windows=validation_windows,
    settings=NoiseEvaluationSettings(),
    tuning_settings=NoiseTuningSettings(local_searches=2),
    upstream_proxy=upstream_flow,  # optional
    proposed_configuration_version="reviewed-ml-v1",
)

test_result = evaluate_configuration(
    test_storage,
    test_discharge,
    result.selected_config,
    evaluation_window=ValidationWindow("final-test", test_start, test_end),
    settings=NoiseEvaluationSettings(),
)
```
