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

The objective is mean one-step predictive negative log-likelihood for observed
storage and outflow after burn-in. Missing components are omitted from the
score, invalid candidates score infinity, and uncertainty is included through
the predictive likelihood.

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

`tune_noise` remains available as a log-likelihood adapter to the bounded
dataframe-first search. It accepts the legacy array container but uses the same
storage-and-outflow-only model and does not require SciPy.
