# Implementation Plan: Conservative Kalman Inflow Process-Noise Tuning

## 1. Objective

Add an offline calibration workflow that proposes `q_inflow` using causal predictive performance while favoring smoother models when several candidates perform similarly.

The first release will:

- tune only `q_inflow`;
- keep `R`, `P0`, `q_storage`, and `q_outflow` fixed;
- require constant diagonal covariance inputs;
- run every candidate causally through the same chronology;
- score only predeclared validation windows after warm-up;
- use storage-targeted one-step predictive density as the primary score;
- report joint predictive density and innovation-consistency diagnostics;
- use a paired, window-block uncertainty rule to identify competitive candidates;
- select the smallest competitive `q_inflow`;
- report multi-horizon, physical-behavior, and measurement-noise sensitivity checks;
- return a proposed `ReservoirConfig` without changing operational state or configuration files.

The tuner will not:

- tune against RTS-smoothed results;
- treat an upstream gauge as true inflow;
- optimize filtered storage closure;
- tune `Q`, `R`, and `P0` simultaneously;
- adapt covariance values during streaming operation;
- use the final test period to change the grid, thresholds, windows, or selected parameter.

The statistical workflow is prequential, or rolling-origin, validation: each observation is scored from the distribution available immediately before that observation is assimilated. Validation observations may affect later predictions, as they would during live operation, but they never affect their own or earlier scores.

## 2. Model parameterization

Users specify a prior standard deviation for a one-hour increment of the latent inflow random walk:

\[
\sigma_{I,1h}
=
\operatorname{SD}(I_{t+1h}-I_t).
\]

For the model

\[
dI_t = \sqrt{q_\text{inflow}}\,dW_t,
\]

the continuous-time diffusion parameter is

\[
q_\text{inflow}
=
\frac{\sigma_{I,1h}^2}{3600}.
\]

The public API should call this quantity `prior_hourly_increment_sd` to distinguish it from the empirical standard deviation of filtered inflow changes.

Example:

```python
prior_hourly_increment_sd = np.array([2, 5, 10, 20, 50, 100], dtype=float)
q_inflow_candidates = prior_hourly_increment_sd**2 / 3600.0
```

Candidates must be positive, finite, unique after conversion, and sorted internally. Candidate grids should normally be approximately log-spaced and based on reservoir scale and engineering judgment. Selection at either grid boundary must be flagged. The grid may be expanded using development data, but never after inspecting the untouched final test period.

## 3. Proposed public API

Create `src/kalmone/tuning.py` with an API resembling:

```python
result = tune_inflow_process_noise(
    storage,
    discharge,
    base_config,
    candidate_prior_hourly_increment_sd=[2, 5, 10, 20, 50, 100],
    validation_windows=[
        TuningWindow("storm-1", start_1, end_1, regime="storm"),
        TuningWindow("storm-2", start_2, end_2, regime="storm"),
        TuningWindow("dry-1", start_3, end_3, regime="dry"),
    ],
    settings=InflowTuningSettings(
        warmup=timedelta(hours=24),
        innovation_max_lag=timedelta(hours=24),
        forecast_horizons=(
            timedelta(hours=1),
            timedelta(hours=3),
            timedelta(hours=6),
        ),
    ),
    proposed_configuration_version="2026-08-q-inflow-candidate",
)
```

Return:

```python
@dataclass(frozen=True)
class InflowTuningResult:
    selected_config: ReservoirConfig
    selected_prior_hourly_increment_sd: float
    selected_q_inflow: float
    candidate_summary: pandas.DataFrame
    window_diagnostics: pandas.DataFrame
    regime_diagnostics: pandas.DataFrame
    horizon_diagnostics: pandas.DataFrame
    r_sensitivity: pandas.DataFrame | None
    competitive_candidates: tuple[float, ...]
    selection_threshold: float
    selection_reason: str
    warnings: tuple[str, ...]
```

The result must be inspectable before the proposed configuration is accepted or persisted. Final-test evaluation should be a separate function so test data cannot accidentally enter candidate selection:

```python
test_result = evaluate_inflow_config(
    test_storage,
    test_discharge,
    result.selected_config,
    evaluation_window=final_test_window,
    settings=evaluation_settings,
)
```

## 4. Configuration and input rules

The first implementation requires:

- constant diagonal `Q`;
- constant diagonal `R`;
- constant diagonal `P0`;
- fixed `q_storage` and `q_outflow`;
- fixed, independently justified `R`;
- fixed `P0`;
- at least three physically plausible `q_inflow` candidates;
- equal, timezone-aware, strictly increasing storage and discharge indexes;
- nonoverlapping validation windows with explicit inclusive/exclusive boundary semantics;
- a minimum count of scored storage observations in every required window.

Reject off-diagonal inputs with a clear explanation rather than silently discarding them. Normally, `q_storage` should remain zero or have an independent physical justification so it does not compete with inflow process noise to explain storage fluctuations.

All selection settings, windows, regime labels, weights, diagnostic limits, and physical gates must be fixed before candidate results are inspected.

## 5. Filter execution and chronology

Refactor the existing batch preparation in `src/kalmone/core.py` into a reusable internal function that returns:

- the complete `KalmanFilterResult`;
- source positions and timestamps;
- observations;
- elapsed intervals;
- initialization positions.

For each candidate:

1. Copy the base configuration and change only `q[1, 1]`.
2. Use the actual elapsed intervals to generate transition matrices.
3. Generate every discrete process covariance through the same continuous-time model used operationally.
4. Run the existing public `kalman_filter` once through the full available chronology.
5. Retain predicted states, innovations, innovation covariances, filtered states, and update masks.
6. Score only timestamps inside the predeclared validation mask and outside initialization or warm-up exclusions.
7. Do not run RTS smoothing.

Do not restart the filter independently at every validation window. Restart only at genuine disconnected data segments when operational processing would also restart. Apply initialization and warm-up exclusions after each genuine restart.

Changing an observation after timestamp `t` must never change any score at or before `t`.

The operational batch and streaming APIs must remain numerically unchanged.

## 6. Efficient covariance preparation

Transition matrices and timestamp geometry are identical for all candidates and should be prepared once.

The exact discrete process covariance is linear in the continuous covariance. For every elapsed interval, precompute:

```text
Q_discrete(dt, candidate)
    = Q_fixed_discrete(dt)
    + q_inflow_candidate * Q_inflow_unit_discrete(dt)
```

This avoids repeating the full covariance derivation for every candidate while preserving numerical equivalence with `ReservoirStateSpaceModel.process_covariance`. Add a test comparing cached construction with direct model construction over irregular intervals.

Candidate filters may later be evaluated in parallel, but the first implementation should prefer simple deterministic execution unless profiling shows a need.

## 7. Primary one-step predictive score

### 7.1 Joint predictive density

For each timestamp with at least one observation, calculate joint one-step negative log predictive density:

\[
\operatorname{NLPD}_k
=
\frac{1}{2}
\left[
m_k\log(2\pi)
+\log|S_k|
+\nu_k^\mathsf{T}S_k^{-1}\nu_k
\right].
\]

Use only the finite observation block. Skip timestamps with no update. Record the number of observed scalar components `m_k`.

Joint NLPD is an important whole-model diagnostic, but it is not the primary selection score because the outflow likelihood can dilute differences attributable to `q_inflow`.

### 7.2 Storage-targeted predictive density

Use storage predictive density as the primary score.

When storage is the only finite observation, use its marginal predictive distribution. When storage and outflow are both finite, score storage conditional on the simultaneous outflow measurement using the joint predictive covariance:

\[
\nu_{s\mid o}
=
\nu_s - S_{so}S_{oo}^{-1}\nu_o,
\]

\[
S_{s\mid o}
=
S_{ss}-S_{so}S_{oo}^{-1}S_{os}.
\]

Then:

\[
\operatorname{NLPD}^{storage}_k
=
\frac{1}{2}
\left[
\log(2\pi)
+\log S_{s\mid o}
+\frac{\nu_{s\mid o}^2}{S_{s\mid o}}
\right].
\]

If storage is missing, the timestamp contributes to joint and outflow diagnostics but not to the primary selection score.

This factorization targets the observation that directly identifies inflow while still accounting for current outflow-measurement uncertainty.

### 7.3 Numerical calculation

Use Cholesky factorization or stable linear solves; never form a matrix inverse. Permit only a small, explicitly bounded diagonal jitter. Record the number and magnitude of regularized steps per candidate.

Do not use a pseudoinverse to assign a valid probability density to a singular covariance. If bounded jitter cannot produce a positive-definite scored covariance, mark that candidate-window result invalid.

## 8. Window aggregation and weighting

For candidate `c` and window `w`, calculate:

\[
L_{c,w}
=
\frac{\sum_{k\in w}\operatorname{NLPD}^{storage}_{c,k}}
{n^{storage}_{c,w}}.
\]

Use predeclared equal window weights by default so long dry periods do not automatically overwhelm shorter event windows. Allow explicit weights when there is a documented operational reason. Normalize supplied weights to sum to one.

The primary aggregate is:

\[
L_c = \sum_w a_w L_{c,w}.
\]

Also report:

- globally observation-weighted storage NLPD;
- joint NLPD per observed scalar component;
- window-level and regime-level scores;
- scored counts and coverage.

Missingness is identical across candidates, but every summary must expose counts so comparisons remain auditable.

## 9. Initialization and warm-up

The existing initial inflow uses the first two finite storage observations, which are subsequently assimilated. Their likelihood contributions must never be scored.

For every genuine filter segment:

- exclude both initialization observations;
- exclude a configurable warm-up period after initialization;
- begin scoring only after both exclusions have passed;
- require a minimum number of scored storage observations per window.

The default warm-up should be the greater of:

- 24 hours; or
- a reviewed multiple of the expected filter response time.

Report sensitivity to a longer warm-up during engineering review. Warm-up data may initialize and condition the filter but never contribute to the validation score.

## 10. Validation-window design

Use complete operational regimes rather than randomized timestamps. Recommended coverage includes:

- multiple independent storm events;
- rising and falling limbs;
- dry-weather periods;
- large discharge changes;
- high and low reservoir elevations;
- missing-data episodes;
- more than one season where data permits;
- known sensor or rating-curve eras where applicable.

Windows should be nonoverlapping and separated enough to behave approximately as independent event blocks. Assign each window a regime label and report regime-stratified results.

Five or more reasonably independent windows are recommended for automatic selection. With fewer than five, produce a proposed result but require manual approval and label uncertainty as weak. Three windows are a minimum for calculating a rough block uncertainty estimate, not evidence of reliable statistical equivalence.

Reserve one final contiguous test period that is never supplied to the tuner.

## 11. Innovation-consistency diagnostics

For each candidate, window, and regime, calculate the following from pre-update innovations.

### 11.1 Normalized NIS

Joint:

\[
\overline{\operatorname{NIS}}_{joint}
=
\frac{\sum_k \nu_k^\mathsf{T}S_k^{-1}\nu_k}
{\sum_k m_k}.
\]

The theoretical target is approximately 1.

For individual observations, use marginal contributions:

\[
\overline{\operatorname{NIS}}_j
=
\frac{1}{n_j}\sum_k\frac{\nu_{k,j}^2}{S_{k,jj}}.
\]

Do not imply that marginal storage and outflow NIS values add to joint NIS when the joint covariance is correlated.

### 11.2 Innovation bias

Report the mean standardized innovation for storage, outflow, and conditional storage:

\[
z_{k,j}=\frac{\nu_{k,j}}{\sqrt{S_{k,jj}}}.
\]

Persistent nonzero bias indicates structural or sensor bias that should not be repaired by increasing process noise.

### 11.3 Innovation autocorrelation

For storage, outflow, and conditional-storage standardized innovations, report:

- autocorrelation in elapsed-time lag bins through `innovation_max_lag`;
- maximum absolute nonzero-lag autocorrelation over predeclared material lags;
- the elapsed-time lag where it occurs;
- pair count and coverage at every lag.

Do not interpret lag as row count when timestamps are irregular. Do not interpolate missing innovations. Use pairwise-finite values whose timestamp separation falls within a documented lag tolerance.

Because testing many lags creates a multiple-comparison problem, autocorrelation should initially be a warning and diagnostic rather than a universal automatic rejection gate.

## 12. Candidate eligibility

Separate hard validity gates from review diagnostics.

### Hard gates

A candidate is ineligible if any of the following occurs:

- nonfinite primary NLPD;
- scored covariance that remains non-positive-definite after bounded jitter;
- insufficient scored storage observations in a required window;
- nonfinite filtered or predicted states;
- a predeclared physical safety bound is violated;
- excessive numerical regularization beyond a configured limit.

### Configurable engineering gates

The following may be hard gates only when thresholds were independently justified and fixed before tuning:

- normalized NIS range;
- absolute standardized innovation bias;
- physically implausible negative-inflow frequency;
- robust upper quantile of time-normalized inflow changes;
- other reservoir-specific operating limits.

Initial review values such as normalized NIS between 0.7 and 1.5 or absolute mean standardized innovation below 0.25 are not universal scientific thresholds. By default, report them as warnings until reservoir-specific behavior supports using them as gates.

Store every rejection and warning as a machine-readable reason.

## 13. Paired conservative selection rule

Among eligible candidates:

1. Find the candidate with the minimum weighted mean primary score `L_best`.
2. For every candidate `c`, calculate paired window differences:

   \[
   d_{c,w}=L_{c,w}-L_{best,w}.
   \]

3. Estimate uncertainty in the weighted mean paired difference using the validation windows as resampling blocks. Use a deterministic-seed block bootstrap when at least five windows are available. With three or four windows, use the analytic paired standard error and mark it as weak.
4. Define a candidate as competitive when its mean excess loss is no greater than the larger of:

   - one paired standard error; or
   - a predeclared practical-equivalence tolerance in NLPD per storage observation.

5. Select the competitive candidate with the smallest `q_inflow`.
6. Break exact ties deterministically toward smaller `q_inflow`.

Using paired differences removes much of the event-to-event variation shared by all candidates. The practical-equivalence tolerance prevents extremely large samples from treating operationally negligible score differences as decisive.

If fewer than three valid windows remain, do not automatically select a production candidate. Return diagnostics and require the caller to revise the data or window design.

## 14. Multi-horizon predictive diagnostics

One-step likelihood can reward excessive responsiveness. For the selected candidate and nearby competitive candidates, calculate open-loop predictive diagnostics at predeclared elapsed-time horizons such as 1, 3, and 6 hours.

At each forecast origin:

1. Start from the causal filtered state at the origin.
2. Propagate with actual transition intervals and process covariances.
3. Do not assimilate intervening measurements.
4. Score the first observation falling within a configured tolerance of each horizon.

Report storage NLPD, bias, RMSE, coverage, and NIS by horizon. These metrics are initially diagnostic rather than optimized. Flag a selected candidate that wins at one step but degrades materially relative to competitive smoother candidates at operational horizons.

## 15. Measurement-noise sensitivity

Fix `R` during primary tuning, but evaluate whether the result is robust to plausible independently specified alternatives.

Allow optional reviewed multipliers or alternate matrices, for example:

```python
r_sensitivity_multipliers = (0.75, 1.0, 1.25)
```

For each alternative `R`, rerun the candidate comparison without changing the primary selected result. Report:

- the best and conservatively selected candidate under that scenario;
- movement in grid positions relative to the base result;
- score and NIS changes;
- whether the base selected candidate remains competitive.

Material movement under modest `R` changes should require engineering review. The sensitivity analysis is not permission to choose whichever `R` produces the preferred inflow.

## 16. Causal physical-behavior checks

Report, but do not initially optimize:

- negative filtered-inflow frequency;
- filtered-inflow first-difference median and robust upper quantiles;
- changes normalized for actual elapsed time;
- event peak timing and magnitude;
- causal one-step storage prediction RMSE and bias;
- missing-observation coverage;
- boundary-selection warning.

Do not use filtered inflow at timestamp `t` to reconstruct the storage change ending at `t`, because that estimate has already assimilated `storage[t]`. Define causal closure using either:

- the pre-update storage innovation already produced by the filter; or
- the state available at the beginning of the interval propagated to its end without the ending observation.

The existing filtered storage-closure output may still be reported, but must be labeled as a reconstruction diagnostic rather than independent predictive validation.

Upstream-gauge metrics must remain separate and labeled as timing/agreement diagnostics. Peak magnitude against a partial-catchment gauge is not a truth metric for total reservoir inflow.

## 17. Candidate result tables

The candidate summary should contain:

- prior hourly inflow-increment standard deviation;
- `q_inflow`;
- weighted mean storage-targeted NLPD;
- globally observation-weighted storage NLPD;
- joint NLPD per observed component;
- delta from the minimum primary score;
- paired standard error or bootstrap uncertainty;
- practical-equivalence tolerance;
- normalized joint, storage, outflow, and conditional-storage NIS;
- innovation bias;
- maximum material elapsed-lag autocorrelation;
- number of windows passed;
- scored storage count and joint observed-component count;
- numerical regularization count and maximum jitter;
- physical-behavior metrics;
- eligible flag;
- rejection reasons and warnings;
- competitive flag;
- selected flag.

The per-window and per-regime tables should contain the same core diagnostics indexed by candidate and window or regime.

Sort output with the selected candidate first, then other competitive candidates in increasing `q_inflow`, then remaining candidates in increasing `q_inflow`.

## 18. Final test evaluation

After selection:

1. Freeze the proposed configuration.
2. Evaluate it once on the untouched test period with the separate evaluation API.
3. Calculate causal one-step primary and joint NLPD.
4. Calculate all innovation, regime, physical, and multi-horizon diagnostics.
5. Generate existing filtered closure and upstream-proxy tables with their limitations clearly labeled.
6. Compare validation and test metrics using predeclared degradation criteria.
7. Flag material degradation and require engineering approval before assigning a production version.

The test period must never be used to change the candidate grid, thresholds, window weights, warm-up, model structure, or selected parameter. If any of those change after test inspection, the period becomes development data and a new untouched test period is required.

A failed test does not authorize selecting the second-best candidate using the same test results.

## 19. Configuration provenance

Populate proposed configuration metadata with enough information to reproduce the result:

```python
metadata={
    "calibration_method": (
        "causal-storage-conditional-nlpd-"
        "paired-block-conservative-selection"
    ),
    "tuned_parameters": ["q_inflow"],
    "fixed_parameters": ["q_storage", "q_outflow", "r", "p0"],
    "candidate_prior_hourly_increment_sd": [...],
    "candidate_q_inflow": [...],
    "validation_windows": [...],
    "window_weights": [...],
    "warmup_seconds": ...,
    "forecast_horizons_seconds": [...],
    "selection_settings": {...},
    "eligibility_settings": {...},
    "selected_prior_hourly_increment_sd": ...,
    "selected_q_inflow": ...,
    "best_mean_storage_nlpd": ...,
    "selected_mean_storage_nlpd": ...,
    "paired_selection_threshold": ...,
    "competitive_candidates": [...],
    "innovation_diagnostics": {...},
    "r_sensitivity_summary": {...},
    "code_version": ...,
    "data_fingerprint": ...,
}
```

The tuner must require a new proposed configuration version and must never overwrite an existing version.

## 20. File changes

### New package module

`src/kalmone/tuning.py`

Contains:

- `TuningWindow`;
- `InflowTuningSettings`;
- `InflowTuningResult`;
- `InflowConfigEvaluationResult`;
- `tune_inflow_process_noise`;
- `evaluate_inflow_config`;
- predictive-density and innovation helpers;
- elapsed-lag autocorrelation helpers;
- paired block-uncertainty and conservative-selection logic.

### Package exports

`src/kalmone/__init__.py`

Export only stable public tuning and evaluation objects.

### Batch internals

`src/kalmone/core.py`

Extract reusable filter-input preparation and raw filter execution without changing public inflow results.

### Validation support

Move or duplicate the necessary causal validation primitives into package code rather than importing runtime logic from `Notebooks/validation.py`. Preserve the existing notebook-facing validation outputs.

### Documentation

Update:

- `Documentation/CONFIGURATION.md`;
- `Documentation/VALIDATION.md`;
- `Documentation/INFLOW_MODEL_BEHAVIOR.md`;
- `README.md`.

Document physical parameterization, prequential scoring, conditional-storage likelihood, window weighting, uncertainty, causal closure, `R` sensitivity, multi-horizon checks, and the untouched-test protocol.

## 21. Required tests

### Metric tests

- Hand-calculated scalar NLPD.
- Hand-calculated multivariate joint NLPD.
- Hand-calculated storage-conditional-on-outflow NLPD.
- Correct marginal storage score when outflow is missing.
- No primary score when storage is missing.
- No-observation timestamps skipped.
- Correct per-observation and per-component normalization.
- NIS target for known synthetic innovations.
- Bounded jitter handling for nearly singular covariance.
- Rejection when a scored covariance cannot be stabilized within the bound.
- No pseudoinverse-based probability density.

### Selection tests

- Exact best candidate selected when clearly superior.
- Smoother candidate selected under the paired conservative rule.
- Practical-equivalence tolerance behaves as configured.
- Shared window difficulty is removed by paired differences.
- Window weights are normalized and applied correctly.
- Ineligible candidate is never selected.
- Deterministic tie-breaking toward lower `q_inflow`.
- Boundary selection produces a warning.
- Clear failure when no candidate is eligible.
- Manual-approval warning with fewer than five valid windows.

### Causality tests

- Altering future observations does not change prior scores.
- A validation observation is scored before it is assimilated.
- Earlier validation observations may causally affect later predictions.
- RTS smoothing is never called by the tuner.
- Initialization and warm-up rows are excluded.
- Windows use one continuous run rather than independent restarts.
- Test-period data cannot influence candidate selection.
- Causal closure does not use a state updated by the ending storage observation.
- Multi-horizon forecasts do not assimilate intervening observations.

### Model and integration tests

- Prior hourly increment standard deviation converts correctly to continuous `q_inflow`.
- Irregular intervals use exact continuous-time covariance calculation.
- Cached covariance-basis construction matches direct model construction.
- Base configuration is not mutated.
- Only `q[1, 1]` changes in the proposed configuration.
- Off-diagonal covariance is rejected in the first release.
- Existing batch and streaming outputs remain numerically unchanged.
- Missing storage and outflow follow the existing partial-update behavior.
- `R` sensitivity results cannot replace the primary selected result.

### Synthetic recovery tests

Generate deterministic-seed data from the reservoir state-space model with known noise values. Verify that:

- grossly too-small and too-large `q_inflow` values score worse on average;
- the true or neighboring candidate is competitive;
- the conservative rule selects a smooth competitive candidate;
- partial missing observations do not destabilize results;
- irregular sampling is handled correctly;
- results remain deterministic;
- understated `R` tends to push selection toward larger `q_inflow`, demonstrating the purpose of the sensitivity report;
- isolated outliers do not silently pass without diagnostic consequences.

Avoid brittle assertions that a stochastic simulation must always choose one exact grid point.

## 22. Acceptance criteria

The feature is complete when:

- candidate evaluation is entirely causal and pre-update;
- each candidate uses the same continuous chronology and scoring mask;
- no development, validation, or test leakage is present;
- measurement noise remains fixed during primary tuning;
- `R` sensitivity is reported separately when requested;
- results are deterministic for identical inputs and settings;
- the selected candidate follows the paired conservative rule;
- window weighting and practical equivalence are explicit;
- filtered reconstruction closure is not presented as independent prediction;
- irregular-time innovation diagnostics use elapsed time rather than row lag;
- all warnings, diagnostics, counts, and rejection reasons are visible;
- current batch and streaming tests remain unchanged;
- no new runtime dependency is required;
- configuration provenance is sufficient to reproduce selection;
- documentation states that the result is proposed and requires engineering approval.

## 23. Suggested delivery sequence

### Phase 1: Causal scoring foundation

- Extract reusable raw batch-filter preparation and execution.
- Implement joint and storage-targeted predictive density.
- Implement bounded numerical stabilization and audit fields.
- Add missing-data, initialization, continuous-chronology, and causality tests.

### Phase 2: Candidate evaluation

- Add physical candidate conversion and validation.
- Precompute transition and covariance bases.
- Run the candidate grid.
- Produce per-window, per-regime, and aggregate tables.

### Phase 3: Conservative selection

- Add hard validity gates and configurable engineering warnings.
- Implement paired block uncertainty and practical equivalence.
- Select the smoothest competitive candidate.
- Construct proposed configuration and provenance.

### Phase 4: Operational robustness

- Add elapsed-time innovation bias and autocorrelation diagnostics.
- Add causal closure and physical-behavior metrics.
- Add multi-horizon open-loop diagnostics.
- Add optional `R` sensitivity reporting.

### Phase 5: Independent evaluation and documentation

- Add the separate untouched-test evaluation workflow.
- Connect existing filtered closure and upstream-proxy reporting with clear labels.
- Add reproducible synthetic examples.
- Document the complete calibration and approval protocol.
- Verify numerical equivalence of all existing operational APIs.

## 24. Expected real-data interpretation

The selected `q_inflow` should be treated as the best conservative constant-diffusion approximation for the reviewed operating regimes, not as a universal physical constant.

Expected failure patterns are:

- too-small `q_inflow`: delayed event response, NIS above one, and persistent innovation autocorrelation;
- too-large `q_inflow`: rough or negative inflow, NIS below one, and strong sensitivity to storage spikes;
- misspecified `R`: systematic movement of selected `q_inflow` under modest `R` sensitivity scenarios;
- structural model error: regime-dependent bias or autocorrelation that no single `q_inflow` resolves.

If storm and dry regimes require materially different candidates, do not hide the conflict in an aggregate score. Approve a documented compromise for the first release or begin a separately validated regime-dependent process-noise design.
