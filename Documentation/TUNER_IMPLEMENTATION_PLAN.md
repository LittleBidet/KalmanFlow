# Dataframe-First Tuner Plan

## Scope and input contract

The tuner accepts pre-cleaned pandas dataframes containing only `storage` and
`outflow`. Callers provide a timezone-aware, strictly increasing
`DatetimeIndex`; the tuner does not parse, sort, align, deduplicate, or repair
input data.

Each tuning run selects positive, finite values for `q_storage`, `q_inflow`,
`q_outflow`, `r_storage`, and `r_outflow`. It returns an immutable,
reservoir-specific configuration, predictive score, diagnostics, evaluation
count, and a full output dataframe. Input dataframes remain unchanged.

## Search and scoring

Tuning uses a bounded logarithmic search with a hard evaluation limit and a
deterministic seed. Robust storage and outflow statistics, water-balance raw
inflow, and observation cadence provide automatic initial values. Raw inflow
is an initialization aid only, never a measured observation or optimization
target.

The dataframe objective is robust blocked multi-horizon Student-t predictive
negative log-likelihood. Default horizons are one, six, and 24 hours with
weights 0.50, 0.30, and 0.20 and degrees of freedom 5.0. One-step Gaussian
likelihood was too eager to reward immediate sensor-noise tracking; the
longer causal leads and heavy-tailed likelihood make candidate selection more
stable. Missing components are omitted from the score, each horizon is
normalized by usable observed scalar components, unavailable horizons have
their weights renormalized, and invalid predictive covariances score infinity.

Forecast origins are selected in four contiguous validation blocks distributed
through the tuning record. Each forecast starts from the filtered state using
observations through that origin and propagates to its target without
assimilating intervening observations. Actual elapsed seconds are used for
regular and irregular timestamp indexes. Candidate performance is the median
block loss, and all forecasts and inflow diagnostics are causal; revised or
centered estimates are not used.

Transition matrices, covariance bases, observations, masks, initial state, and
initial covariance are prepared once per reservoir. Progressive candidate
stages use bounded contiguous prefixes that share those arrays; finalists are
evaluated on the complete record. Full production output is generated only for
the winner.

## Batch execution and persistence

`tune_reservoirs` gives every reservoir an independent configuration and seed,
and may parallelize between reservoirs. Failures are isolated unless
`fail_fast=True` is requested.

Configurations are saved and loaded as JSON without observations or generated
output. Saved metadata includes the tuning-time timestamp, reservoir identity,
parameters, initial covariance, units, smoothing lag, score, evaluation count,
data interval, missing-observation counts, random seed, and version fields.

## Legacy entry point

`tune_noise` remains available with a separate retained one-step Gaussian
scoring path for `objective="loglik"`. It accepts the legacy array container,
uses the same storage-and-outflow-only model, and does not require SciPy. The
dataframe tuner still uses a bounded log-space search with a hard candidate
evaluation budget. Sensor-derived bounds remain recommended because forecast
scoring does not fully identify all five noise parameters.
