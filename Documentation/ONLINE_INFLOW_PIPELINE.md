# Online inflow pipeline

`OnlineInflowPipeline` coordinates an ordered observation stream with a model-specific backend and a fixed-lag smoother. `OnlineReservoirInflow` is the reservoir-facing wrapper and is the normal choice for application code.

## Lifecycle

1. The pipeline accepts timezone-aware, strictly increasing observations with at most microsecond timestamp precision. Nanosecond-level `ValidationWindow` boundaries remain valid for half-open scoring intervals.
2. It ignores leading missing-storage rows, then stores the first finite-storage observation (which must have finite discharge) and waits for the next finite storage sample to initialize the backend.
3. Initialization creates two forward filter steps. The first uses outflow as a steady-state inflow prior and does not inspect the second observation; the second is a normal causal update. Each later input creates one forward step, using the actual elapsed seconds since its predecessor.
4. Each step enters the fixed-lag RTS smoother. A state is released only after the configured elapsed-time lag has passed.

`PipelineUpdate.filtered_state` is the latest forward step; `filtered_states` contains every step created by that call; and `smoothed_states` contains just the newly finalized states. `provisional_states` may inspect the active lag window without releasing it.

## Reservoir API

Create `OnlineReservoirInflow` with scalar noise parameters for the default acre-ft/cfs model, or use `OnlineReservoirInflow.from_config(config)` for a validated configuration:

```python
from kalmanflow import Observation, OnlineReservoirInflow

stream = OnlineReservoirInflow(
    q_storage=1.0, q_inflow=1.0, q_outflow=1.0,
    r_storage=1.0, r_outflow=1.0,
)
update = stream.process(Observation(timestamp, storage, discharge))
```

`filtered_inflows` are causal and marked `NON_SMOOTHED`. `revised_inflows`
are released only after smoothing and marked `SMOOTHED`; each is an absolute
inflow value that replaces the causal estimate at the same timestamp.
Only `NaN` denotes a missing storage or discharge value; infinities are
rejected before pipeline state changes.

## Checkpoints

Configured reservoir streams support compact checkpoints after a successful `process` or `process_many` call:

```python
checkpoint = stream.checkpoint()
restored = OnlineReservoirInflow.from_checkpoint(checkpoint, config=config)
```

A checkpoint is bound to one `reservoir_id` and a fingerprint of the model,
covariance, smoothing-lag, and unit settings. Restore rejects a configuration
that does not match. Restoration rebuilds the active smoothing window from its
retained observations without re-emitting already released records.

Checkpoints are version-bound. Checkpoints created before the package rename
are rejected during restore and must be recreated.

Checkpoint timestamps preserve microsecond precision. Inputs with a nonzero
nanosecond remainder are rejected before pipeline state changes, so restoring a
checkpoint cannot silently change an accepted observation label.

Checkpoints are unavailable during processing, before a successful `process` or
`process_many` result, after a failed processing call, or for streams created
without a reservoir ID. The serialized bytes are intentionally an internal
format; retain the compatible configuration and use the package's restore API
instead of decoding them yourself.

## Resource bounds and failures

`max_window_steps` must be at least two and bounds the active smoother window. Exceeding that bound, invalid timestamps, incompatible checkpoint data, or a backend failure raises an exception. `process_many` rolls back its entire input group on failure; single `process` validates before forwarding values to the pipeline.
