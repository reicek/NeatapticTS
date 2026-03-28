# architecture/network/prune

Structured and dynamic pruning utilities for networks.

Features:
 - Scheduled pruning during gradient-based training ({@link maybePrune}) with linear sparsity ramp.
 - Evolutionary generation pruning toward a target sparsity ({@link pruneToSparsity}).
 - Two ranking heuristics:
     magnitude: |w|
     snip: |w * g| approximation (g approximated via accumulated delta stats; falls back to |w|)
 - Optional stochastic regrowth during scheduled pruning (dynamic sparse training), preserving acyclic constraints.

Internal state fields (attached to Network through a loose internal bridge):
 - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 - _rand: deterministic RNG function
 - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting

Opportunistically perform scheduled pruning during gradient-based training.

Scheduling model:
 - start / end define an iteration window (inclusive) during which pruning may occur
 - frequency defines cadence (every N iterations inside the window)
 - targetSparsity is linearly annealed from 0 to its final value across the window
 - method chooses ranking heuristic (magnitude | snip)
 - optional regrowFraction allows dynamic sparse training: after removing edges we probabilistically regrow
   a fraction of them at random unused positions (respecting acyclic constraint if enforced)

SNIP heuristic:
 - Uses |w * grad| style saliency approximation (here reusing stored delta stats as gradient proxy)
 - Falls back to pure magnitude if gradient stats absent.

## architecture/network/prune/network.prune.utils.types.ts

### PRUNING_METHOD_MAGNITUDE

Pruning method identifier for absolute-weight ranking.

### PRUNING_METHOD_SNIP

Pruning method identifier for SNIP-like saliency ranking.

### MIN_REMAINING_CONNECTION_COUNT

Lower bound to ensure at least one connection remains after pruning.

### DEFAULT_PRUNE_FREQUENCY

Fallback prune cadence when schedule frequency is omitted or invalid.

### MIN_PROGRESS_FRACTION

Minimum normalized schedule progress value.

### MAX_PROGRESS_FRACTION

Maximum normalized schedule progress value.

### MAX_EVOLUTIONARY_TARGET_SPARSITY

Safety cap below full sparsity to avoid degenerate zero-connection networks.

### REGROW_ATTEMPT_MULTIPLIER

Retry multiplier to convert intended regrowth count into max attempts.

### ActivePruningConfig

Non-nullable pruning schedule configuration shape used by helpers.

## architecture/network/prune/network.prune.utils.ts

### maybePrune

```ts
maybePrune(
  iteration: number,
): void
```

Perform scheduled pruning at a given training iteration if conditions are met.

Scheduling fields (cfg): start, end, frequency, targetSparsity, method ('magnitude' | 'snip'), regrowFraction.
The target sparsity ramps linearly from 0 at start to cfg.targetSparsity at end.

Parameters:
- `iteration` - Current (0-based or 1-based) training iteration counter used for scheduling.

### pruneToSparsity

```ts
pruneToSparsity(
  targetSparsity: number,
  method: PruningMethod,
): void
```

Evolutionary (generation-based) pruning toward a target sparsity baseline.
Unlike maybePrune this operates immediately relative to the first invocation's connection count
(stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.

Parameters:
- `targetSparsity` - - Requested target sparsity.
- `method` - - Connection ranking heuristic.

Returns: Nothing.

### getCurrentSparsity

```ts
getCurrentSparsity(): number
```

Current sparsity fraction relative to the training-time pruning baseline.

Returns: Current sparsity in the [0,1] range when baseline is available.

## architecture/network/prune/network.prune.regrowth.utils.ts

### maybeRunRegrowth

```ts
maybeRunRegrowth(
  currentNetwork: default,
  context: RegrowthPlanContext,
): void
```

Build and execute a regrowth plan when enabled.

Parameters:
- `currentNetwork` - - Network to regrow.
- `context` - - Inputs describing regrowth intent.

Returns: Nothing.

### buildRegrowthPlan

```ts
buildRegrowthPlan(
  context: RegrowthPlanContext,
): RegrowthPlan | null
```

Convert regrowth intent into a bounded execution plan.

Parameters:
- `context` - - Regrowth planning inputs.

Returns: A plan when regrowth is meaningful; otherwise null.

### executeRegrowthAttempts

```ts
executeRegrowthAttempts(
  context: RegrowthExecutionContext,
): void
```

Execute bounded stochastic regrowth attempts.

Parameters:
- `context` - - Regrowth execution settings.

Returns: Nothing.

### shouldContinueRegrowth

```ts
shouldContinueRegrowth(
  currentNetwork: default,
  desiredRemainingConnections: number,
  attemptedRegrowthCount: number,
  maxAttempts: number,
): boolean
```

Decide whether another regrowth attempt is allowed.

Parameters:
- `currentNetwork` - - Network being regrown.
- `desiredRemainingConnections` - - Target remaining connection count.
- `attemptedRegrowthCount` - - Number of attempts already used.
- `maxAttempts` - - Maximum attempts allowed.

Returns: True when another attempt should run.

### tryRegrowConnection

```ts
tryRegrowConnection(
  currentNetwork: default,
): void
```

Attempt one random valid connection addition.

Parameters:
- `currentNetwork` - - Network being regrown.

Returns: Nothing.

### buildRegrowthCandidatePair

```ts
buildRegrowthCandidatePair(
  currentNetwork: default,
): { sourceNode: default; targetNode: default; } | null
```

Build one random regrowth candidate pair if valid.

Parameters:
- `currentNetwork` - - Network being regrown.

Returns: Candidate node pair or null when invalid.

### pickRandomNode

```ts
pickRandomNode(
  currentNetwork: default,
): default | undefined
```

Pick a random node using the network RNG.

Parameters:
- `currentNetwork` - - Network providing node set and RNG.

Returns: Random node or undefined when the node list is empty.

### isInvalidRegrowthPair

```ts
isInvalidRegrowthPair(
  currentNetwork: default,
  sourceNode: default,
  targetNode: default,
): boolean
```

Validate whether a candidate regrowth pair is acceptable.

Parameters:
- `currentNetwork` - - Network being regrown.
- `sourceNode` - - Proposed source node.
- `targetNode` - - Proposed target node.

Returns: True when the pair must be rejected.

### connectionAlreadyExists

```ts
connectionAlreadyExists(
  currentNetwork: default,
  sourceNode: default,
  targetNode: default,
): boolean
```

Check whether a connection already exists.

Parameters:
- `currentNetwork` - - Network being regrown.
- `sourceNode` - - Proposed source node.
- `targetNode` - - Proposed target node.

Returns: True when the edge already exists.

### violatesAcyclicConstraint

```ts
violatesAcyclicConstraint(
  currentNetwork: default,
  sourceNode: default,
  targetNode: default,
): boolean
```

Check whether a pair violates forward-only acyclic ordering.

Parameters:
- `currentNetwork` - - Network being regrown.
- `sourceNode` - - Proposed source node.
- `targetNode` - - Proposed target node.

Returns: True when acyclic ordering would be violated.

## architecture/network/prune/network.prune.schedule.utils.ts

### getPruningConfig

```ts
getPruningConfig(
  currentNetwork: default,
): { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; } | undefined
```

Read the active pruning schedule from network internals.

Parameters:
- `currentNetwork` - - Network instance to inspect.

Returns: Pruning configuration when enabled; otherwise undefined.

### shouldRunScheduledPrune

```ts
shouldRunScheduledPrune(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Determine whether scheduled pruning should run at this iteration.

Parameters:
- `currentIteration` - - Training iteration being processed.
- `currentPruningConfig` - - Active pruning schedule.

Returns: True when pruning should execute now.

### getInitialConnectionBaseline

```ts
getInitialConnectionBaseline(
  currentNetwork: default,
): number | undefined
```

Read the scheduled-pruning baseline connection count.

Parameters:
- `currentNetwork` - - Network instance to inspect.

Returns: Baseline count when captured; otherwise undefined.

### buildScheduledTarget

```ts
buildScheduledTarget(
  context: ScheduledTargetContext,
  currentConnectionCount: number,
): ScheduledTargetResult
```

Build current scheduled pruning targets from schedule context.

Parameters:
- `context` - - Inputs required to compute desired remaining connections.
- `currentConnectionCount` - - Current number of network connections.

Returns: Desired remaining connections and current excess.

### buildPruneSelection

```ts
buildPruneSelection(
  context: PruneSelectionContext,
): PruneSelectionResult
```

Build a connection removal selection from current ranking context.

Parameters:
- `context` - - Inputs for ranking and slicing removable connections.

Returns: Connections selected for pruning.

### disconnectConnections

```ts
disconnectConnections(
  currentNetwork: default,
  connectionsToDisconnect: default[],
): void
```

Disconnect all selected connections from the network.

Parameters:
- `currentNetwork` - - Network to mutate.
- `connectionsToDisconnect` - - Connections to remove.

Returns: Nothing.

### resolvePruningMethod

```ts
resolvePruningMethod(
  method: PruningMethod | undefined,
): PruningMethod
```

Normalize optional pruning method to a concrete value.

Parameters:
- `method` - - Optional configured pruning method.

Returns: Concrete pruning method.

### markPruneIteration

```ts
markPruneIteration(
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
  currentIteration: number,
): void
```

Persist the iteration that last performed pruning.

Parameters:
- `currentPruningConfig` - - Active pruning configuration.
- `currentIteration` - - Iteration to record.

Returns: Nothing.

### markTopologyDirty

```ts
markTopologyDirty(
  currentNetwork: default,
): void
```

Mark topology cache as dirty after structural updates.

Parameters:
- `currentNetwork` - - Network with modified connectivity.

Returns: Nothing.

### isOutsidePruningWindow

```ts
isOutsidePruningWindow(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Check whether an iteration is outside the pruning window.

Parameters:
- `currentIteration` - - Iteration to evaluate.
- `currentPruningConfig` - - Active pruning schedule.

Returns: True when the iteration is out of range.

### alreadyPrunedThisIteration

```ts
alreadyPrunedThisIteration(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Check whether this iteration was already pruned.

Parameters:
- `currentIteration` - - Iteration to evaluate.
- `currentPruningConfig` - - Active pruning schedule.

Returns: True when pruning already happened for this iteration.

### isScheduledPruningIteration

```ts
isScheduledPruningIteration(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Check frequency cadence for scheduled pruning.

Parameters:
- `currentIteration` - - Iteration to evaluate.
- `currentPruningConfig` - - Active pruning schedule.

Returns: True when this iteration matches the schedule cadence.

### calculateProgressFraction

```ts
calculateProgressFraction(
  currentIteration: number,
  scheduleStart: number,
  scheduleEnd: number,
): number
```

Compute clamped schedule progress in the [0,1] range.

Parameters:
- `currentIteration` - - Iteration to evaluate.
- `scheduleStart` - - Start iteration of schedule window.
- `scheduleEnd` - - End iteration of schedule window.

Returns: Clamped normalized progress.

### clamp

```ts
clamp(
  value: number,
  minimum: number,
  maximum: number,
): number
```

Clamp a number into an inclusive range.

Parameters:
- `value` - - Raw value to clamp.
- `minimum` - - Inclusive lower bound.
- `maximum` - - Inclusive upper bound.

Returns: Clamped value.

### rankConnectionsByRemovalPriority

```ts
rankConnectionsByRemovalPriority(
  connections: default[],
  method: PruningMethod,
): default[]
```

Route ranking to the configured pruning heuristic.

Parameters:
- `connections` - - Candidate connections to rank.
- `method` - - Ranking method to apply.

Returns: Connections sorted by ascending removal priority.

### rankConnectionsByMagnitude

```ts
rankConnectionsByMagnitude(
  connections: default[],
): default[]
```

Rank connections by absolute weight magnitude.

Parameters:
- `connections` - - Candidate connections to rank.

Returns: Connections sorted by ascending absolute weight.

### rankConnectionsBySnipSaliency

```ts
rankConnectionsBySnipSaliency(
  connections: default[],
): default[]
```

Rank connections by SNIP-like saliency approximation.

Parameters:
- `connections` - - Candidate connections to rank.

Returns: Connections sorted by ascending saliency.

### calculateSnipSaliency

```ts
calculateSnipSaliency(
  connection: default,
): number
```

Compute saliency for SNIP-like ranking.

Parameters:
- `connection` - - Connection to score.

Returns: Saliency value used for sorting.

### resolveGradientMagnitude

```ts
resolveGradientMagnitude(
  connection: default,
): number
```

Resolve a stable gradient-magnitude proxy from connection delta statistics.

Parameters:
- `connection` - - Connection containing accumulated delta history.

Returns: Absolute gradient magnitude proxy.

## architecture/network/prune/network.prune.sparsity.utils.ts

### readInitialSparsityBaseline

```ts
readInitialSparsityBaseline(
  currentNetwork: default,
): number | undefined
```

Read baseline used for sparsity reporting.

Parameters:
- `currentNetwork` - - Network to inspect.

Returns: Baseline connection count when available.

### calculateSparsityFromBaseline

```ts
calculateSparsityFromBaseline(
  currentConnectionCount: number,
  baselineConnectionCount: number,
): number
```

Convert current density into sparsity ratio.

Parameters:
- `currentConnectionCount` - - Current connection count.
- `baselineConnectionCount` - - Baseline connection count.

Returns: Sparsity ratio in [0,1] for valid baselines.

## architecture/network/prune/network.prune.evolutionary.utils.ts

### normalizeEvolutionaryTargetSparsity

```ts
normalizeEvolutionaryTargetSparsity(
  rawTargetSparsity: number,
): number
```

Clamp evolutionary target sparsity to safe operational bounds.

Parameters:
- `rawTargetSparsity` - - Requested target sparsity.

Returns: Normalized target sparsity.

### getOrCaptureEvolutionaryBaseline

```ts
getOrCaptureEvolutionaryBaseline(
  currentNetwork: default,
): number
```

Capture evolutionary baseline once and reuse it for subsequent pruning calls.

Parameters:
- `currentNetwork` - - Network to inspect and possibly initialize.

Returns: Evolutionary baseline connection count.

### buildEvolutionaryTarget

```ts
buildEvolutionaryTarget(
  context: EvolutionaryTargetContext,
  currentConnectionCount: number,
): EvolutionaryTargetResult
```

Compute evolutionary pruning target counts.

Parameters:
- `context` - - Inputs for sparsity-to-count conversion.
- `currentConnectionCount` - - Current number of network connections.

Returns: Desired remaining and excess connection counts.

### buildEvolutionaryPruneSelection

```ts
buildEvolutionaryPruneSelection(
  context: PruneSelectionContext,
): PruneSelectionResult
```

Build evolutionary pruning connection selection.

Parameters:
- `context` - - Inputs for ranking and slicing.

Returns: Connections selected for removal.

### disconnectEvolutionaryConnections

```ts
disconnectEvolutionaryConnections(
  currentNetwork: default,
  connectionsToDisconnect: default[],
): void
```

Disconnect selected evolutionary pruning edges.

Parameters:
- `currentNetwork` - - Network to mutate.
- `connectionsToDisconnect` - - Edges to remove.

Returns: Nothing.

### markEvolutionaryTopologyDirty

```ts
markEvolutionaryTopologyDirty(
  currentNetwork: default,
): void
```

Mark topology cache as dirty after evolutionary pruning.

Parameters:
- `currentNetwork` - - Network with changed structure.

Returns: Nothing.

### rankEvolutionaryConnections

```ts
rankEvolutionaryConnections(
  connections: default[],
  pruningMethod: PruningMethod,
): default[]
```

Route evolutionary ranking to selected heuristic.

Parameters:
- `connections` - - Candidate connections.
- `pruningMethod` - - Ranking heuristic.

Returns: Connections sorted by ascending removal priority.

### rankEvolutionaryConnectionsByMagnitude

```ts
rankEvolutionaryConnectionsByMagnitude(
  connections: default[],
): default[]
```

Rank connections by magnitude for evolutionary pruning.

Parameters:
- `connections` - - Candidate connections.

Returns: Connections sorted by ascending absolute weight.

### rankEvolutionaryConnectionsBySnip

```ts
rankEvolutionaryConnectionsBySnip(
  connections: default[],
): default[]
```

Rank connections by SNIP-like saliency for evolutionary pruning.

Parameters:
- `connections` - - Candidate connections.

Returns: Connections sorted by ascending saliency.

### calculateEvolutionarySnipSaliency

```ts
calculateEvolutionarySnipSaliency(
  connection: default,
): number
```

Compute evolutionary SNIP-like saliency for one connection.

Parameters:
- `connection` - - Connection to score.

Returns: Saliency score.

### resolveEvolutionaryGradientMagnitude

```ts
resolveEvolutionaryGradientMagnitude(
  connection: default,
): number
```

Resolve gradient proxy for evolutionary SNIP ranking.

Parameters:
- `connection` - - Connection containing delta history.

Returns: Absolute gradient magnitude proxy.
