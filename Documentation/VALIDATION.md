# Inflow validation

```python
from validation import (
    ValidationSettings,
    generate_validation_outputs,
)

settings = ValidationSettings(
    evaluation_frequency="1h",
    cross_correlation_range=48,
    minimum_paired_observations=24,
    training_start="2024-01-01 00:00Z",
    training_end="2024-01-14 23:00Z",
    evaluation_start="2024-01-15 00:00Z",
    evaluation_end="2024-01-31 23:00Z",
)
outputs = generate_validation_outputs(comparison, settings=settings)
outputs.upstream_proxy_agreement
outputs.best_lag_summary
outputs.storage_closure
```

The comparison frame is resampled to the configured regular frequency using
means. Missing bins remain missing; validation never interpolates. The frame
contains `raw_inflow`, `centered_rolling_inflow`, `estimated_inflow`,
`delayed_revised_inflow`, and `upstream_flow`, in addition to `storage` and
`outflow`. The revised estimate is deliberately renamed to
`delayed_revised_inflow` in this workflow because it is a fixed-lag diagnostic,
not a causal estimate. The centered rolling mean is also acausal because it
uses values before and after each timestamp.

The upstream table reports Pearson and Spearman correlation, KGE and its
correlation/variability/mean-flow components, percent bias, NSE, normalized
RMSE, paired observations, and coverage. The upstream gauge is a
partial-catchment proxy: Pearson correlation and KGE measure agreement with
that proxy, not total-inflow accuracy. KGE bias and variability can therefore
be poor even when timing is correct because the gauge represents only part of
the contributing flow.

Storage closure remains the stronger internal-consistency check. Its one-step  
residual is predicted storage change minus observed storage change, using the  
estimate and outflow at the current regular timestamp. The closure table  
reports RMSE, MAE, bias, paired observations, and coverage.

## Offline process-noise tuning

The package-level `tune_inflow_process_noise` workflow is separate from these
notebook comparison tables. It uses timezone-aware, non-overlapping operational
windows and elapsed-time innovation diagnostics. The primary score is causal
one-step storage NLPD, conditional on simultaneous outflow when available;
joint NLPD, NIS, bias, physical behavior, and open-loop horizon diagnostics
are reported separately. This avoids optimizing the filtered storage-closure
reconstruction.

Keep a contiguous final test period out of tuning. After selection, call
`evaluate_inflow_config` with the frozen proposed configuration and compare
test diagnostics against validation using criteria agreed before test
inspection. Measurement-noise sensitivity scenarios are review diagnostics and
cannot replace the primary selected result.
