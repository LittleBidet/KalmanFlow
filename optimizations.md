# Optimizations

## Array-native reservoir batch processing

`OnlineReservoirInflow` is a one-observation streaming adapter.
It creates immutable `ReservoirFlowUpdate` and `ReservoirFlowEstimate` objects
so callers can consume each filtered inflow and finalized fixed-lag inflow
revision as they become available. A revision is an absolute replacement for
the causal value at its timestamp. However, it adds allocation and remapping
when the caller already has a complete pair of aligned Pandas series.

The batch APIs therefore bypass `OnlineReservoirInflow` and run a private
array-native kernel. It builds the same `ReservoirBackend`, calculates the
time-varying Kalman transitions and covariances in arrays, stores filtered
state in the existing array-based filter result, and writes causal inflow,
revised inflow, and provenance flags into preallocated NumPy columns. Pandas receives those
finished columns once to construct the public `DataFrame`.

The numerical recursion remains ordered: each Kalman update depends on the
previous state, sample intervals may be irregular, observations may be
partially missing, and fixed-lag RTS output is released only when its lag has
elapsed. The optimization removes adapter and dataclass overhead without
changing those semantics. Streaming behavior, checkpointing, and the public
batch DataFrame contract remain separate and unchanged.
