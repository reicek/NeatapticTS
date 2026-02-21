# architecture/network/prune

## architecture/network/prune/network.prune.utils.ts

### getCurrentSparsity

`() => number`

Current sparsity fraction relative to the training-time pruning baseline.

### maybePrune

`(iteration: number) => void`

Structured and dynamic pruning utilities for networks.

Features:
 - Scheduled pruning during gradient-based training ({@link maybePrune}) with linear sparsity ramp.
 - Evolutionary generation pruning toward a target sparsity ({@link pruneToSparsity}).
 - Two ranking heuristics:
     magnitude: |w|
     snip: |w * g| approximation (g approximated via accumulated delta stats; falls back to |w|)
 - Optional stochastic regrowth during scheduled pruning (dynamic sparse training), preserving acyclic constraints.

Internal State Fields (attached to Network via `any` casting):
 - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 - _rand: deterministic RNG function
 - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting

### pruneToSparsity

`(targetSparsity: number, method: import("C:/NeatapticTS/src/architecture/network/network.types").PruningMethod) => void`

Evolutionary (generation-based) pruning toward a target sparsity baseline.
Unlike maybePrune this operates immediately relative to the first invocation's connection count
(stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.
