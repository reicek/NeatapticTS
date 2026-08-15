# architecture/network/prune

Error raised when a sparsity-budget max-connection count cap is invalid.

## architecture/network/prune/network.prune.budget.errors.ts

### NetworkPruneBudgetGrowthGraceFractionError

Error raised when the sparsity-budget growth-grace fraction configuration is invalid.

### NetworkPruneBudgetMaxConnectionsError

Error raised when a sparsity-budget max-connection count cap is invalid.

## architecture/network/prune/network.prune.utils.types.ts

### ActivePruningConfig

Normalized pruning-config shape consumed by internal helpers after boundary
guards confirm that runtime options are present.

### DEFAULT_PRUNE_FREQUENCY

Default prune cadence used when schedule frequency is absent or invalid.

### MAX_EVOLUTIONARY_TARGET_SPARSITY

Safety cap below full sparsity to avoid degenerate zero-connection networks.

### MAX_PROGRESS_FRACTION

Maximum normalized schedule progress value used by clamp logic before
deriving sparsity targets inside schedule helpers.

### MIN_PROGRESS_FRACTION

Minimum normalized schedule progress value used by clamp logic before
deriving sparsity targets inside schedule helpers.

### MIN_REMAINING_CONNECTION_COUNT

Lower bound that guarantees at least one connection remains after pruning.

### PRUNING_METHOD_MAGNITUDE

Pruning method identifier for absolute-weight ranking (`|w|`) used by
default scheduled and evolutionary pruning selection helpers.

### PRUNING_METHOD_SNIP

Pruning method identifier for SNIP-like saliency ranking (`|w * g|` proxy).

### REGROW_ATTEMPT_MULTIPLIER

Retry multiplier used to translate desired regrowth count into max attempts.

## architecture/network/prune/network.prune.utils.ts

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

### configureSparsityBudget

```ts
configureSparsityBudget(
  configuration: SparsityBudgetConfiguration,
): void
```

Configure a total-connection growth sparsity budget on one network.

The budget is expressed as an absolute cap across forward and self
connections plus an optional grace fraction. Growth helpers can then prune
before mutation or deny the request when the graph cannot stay within the
allowed envelope.

Parameters:
- `this` - Target network instance.
- `configuration` - Budget settings for future structural growth.

Returns: Nothing.

### getCurrentSparsity

```ts
getCurrentSparsity(): number
```

Return current sparsity relative to the captured pruning baseline connection count.

Parameters:
- `this` - Network instance whose pruning baseline is inspected.

Returns: Current sparsity in the [0,1] range when baseline is available.

### getSparsityBudgetSnapshot

```ts
getSparsityBudgetSnapshot(
  currentNetwork: default,
): NetworkSparsityBudgetSnapshot | undefined
```

Return the latest recorded sparsity-budget decision snapshot for diagnostics and telemetry.

Parameters:
- `currentNetwork` - Network to inspect.

Returns: Snapshot clone when one exists; otherwise undefined.

### maybePrune

```ts
maybePrune(
  iteration: number,
): void
```

Perform scheduled pruning at a given training iteration if conditions are met.

Uses schedule fields from `_pruningConfig` (`start`, `end`, `frequency`,
`targetSparsity`, `method`, and optional `regrowFraction`) to decide whether
this iteration should prune, then removes low-ranked connections and can
optionally regrow a bounded subset.

Parameters:
- `this` - Network instance bound by method call.
- `iteration` - Current (0-based or 1-based) training iteration counter used for scheduling.

Returns: Nothing.

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
- `this` - Network instance bound by method call.
- `targetSparsity` - Requested target sparsity.
- `method` - Connection ranking heuristic.

Returns: Nothing.

## architecture/network/prune/network.prune.budget.utils.ts

### asSparsityBudgetProps

```ts
asSparsityBudgetProps(
  currentNetwork: default,
): NetworkSparsityBudgetProps
```

Interpret one network as a sparsity-budget host.

Parameters:
- `currentNetwork` - Network being inspected.

Returns: Runtime budget props bridge.

### asSparsityBudgetRuntimeProps

```ts
asSparsityBudgetRuntimeProps(
  currentNetwork: default,
): NetworkSparsityBudgetRuntimeProps
```

Interpret one network as a sparsity-budget host with internal retry state.

Parameters:
- `currentNetwork` - Network being inspected.

Returns: Runtime budget props bridge including deny-backoff state.

### clearDeniedGrowthBackoffState

```ts
clearDeniedGrowthBackoffState(
  currentNetwork: default,
): void
```

Reset any deny-backoff state after growth becomes viable again.

Parameters:
- `currentNetwork` - Network whose retry state should be cleared.

Returns: Nothing.

### collectBudgetedConnections

```ts
collectBudgetedConnections(
  currentNetwork: default,
): default[]
```

Collect all live connection objects that may be pruned to free budget.

Parameters:
- `currentNetwork` - Network being inspected.

Returns: Forward and self connections.

### configureSparsityBudget

```ts
configureSparsityBudget(
  configuration: SparsityBudgetConfiguration,
): void
```

Configure a total-connection growth sparsity budget on one network.

The budget is expressed as an absolute cap across forward and self
connections plus an optional grace fraction. Growth helpers can then prune
before mutation or deny the request when the graph cannot stay within the
allowed envelope.

Parameters:
- `this` - Target network instance.
- `configuration` - Budget settings for future structural growth.

Returns: Nothing.

### convertMegabytesToBytes

```ts
convertMegabytesToBytes(
  megabytes: number,
): number
```

Convert megabytes to bytes for runtime-heap comparisons.

Parameters:
- `megabytes` - Soft target expressed in megabytes.

Returns: Equivalent byte count.

### countBudgetedConnections

```ts
countBudgetedConnections(
  currentNetwork: default,
): number
```

Count all connection objects that contribute to structural sparsity.

Parameters:
- `currentNetwork` - Network being inspected.

Returns: Total number of forward and self connections.

### createDeniedGrowthBackoffFingerprint

```ts
createDeniedGrowthBackoffFingerprint(
  input: { allowedConnectionLimit: number; budgetConfig: { maxConnections: number; growthGraceFraction: number; method: PruningMethod; }; connectionCountBeforeDecision: number; requiredAdditionalConnections: number; triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined; },
): DeniedGrowthBackoffFingerprint
```

Build the fingerprint used to decide whether one deny state still matches.

Parameters:
- `input` - Current decision inputs that define one growth dead-end.

Returns: Stable fingerprint for deny-backoff reuse.

### denyIfPruneInfeasible

```ts
denyIfPruneInfeasible(
  currentNetwork: default,
  params: { allowedConnectionLimit: number; connectionCountBeforeDecision: number; desiredConnectionCountBeforeGrowth: number; deniedGrowthBackoffFingerprint: DeniedGrowthBackoffFingerprint; maxPrunableConnectionCount: number; plannedPruneCount: number; projectedConnectionCount: number; requiredAdditionalConnections: number; triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined; },
): boolean
```

Deny growth when pruning cannot free enough connections to fit the budget.

Parameters:
- `currentNetwork` - Network about to grow.
- `params` - Decision inputs for the feasibility check.

Returns: True when growth was denied because pruning is infeasible.

### ensureGrowthBudget

```ts
ensureGrowthBudget(
  currentNetwork: default,
  requiredAdditionalConnections: number,
): boolean
```

Ensure enough total-connection budget remains before a growth mutation writes.

Behavior:
- allow immediately when the projected total connection count fits the budget,
- prune lowest-priority connections first when the budget can be satisfied by
  freeing space,
- temporarily stop net-new growth when the active Node/browser heap already exceeds
  its soft memory target,
- deny without structural writes when the request cannot stay within the
  minimum remaining-connection invariant.

Parameters:
- `currentNetwork` - Network about to grow.
- `requiredAdditionalConnections` - Net total-connection increase requested by the caller.

Returns: True when growth may proceed.

### executePruneAndAllow

```ts
executePruneAndAllow(
  currentNetwork: default,
  params: { allowedConnectionLimit: number; budgetConfig: { maxConnections: number; growthGraceFraction: number; method: PruningMethod; }; budgetedConnections: default[]; connectionCountBeforeDecision: number; deniedGrowthBackoffFingerprint: DeniedGrowthBackoffFingerprint; desiredConnectionCountBeforeGrowth: number; plannedPruneCount: number; projectedConnectionCount: number; requiredAdditionalConnections: number; triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined; },
): boolean
```

Execute the prune-then-allow path: prune connections, check whether growth
now fits, record the decision snapshot, and return the verdict.

Parameters:
- `currentNetwork` - Network about to grow.
- `params` - Decision inputs for the prune-and-allow path.

Returns: True when growth may proceed after pruning.

### getSoftBudgetEnvironment

```ts
getSoftBudgetEnvironment(
  triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined,
): "browser" | "node" | undefined
```

Resolve the soft-budget environment label from the triggered state, if any.

Parameters:
- `triggeredSoftBudgetState` - Active soft-budget pressure, if any.

Returns: Environment label when a soft budget is active, otherwise undefined.

### getSparsityBudgetSnapshot

```ts
getSparsityBudgetSnapshot(
  currentNetwork: default,
): NetworkSparsityBudgetSnapshot | undefined
```

Return the latest recorded sparsity-budget decision snapshot for diagnostics and telemetry.

Parameters:
- `currentNetwork` - Network to inspect.

Returns: Snapshot clone when one exists; otherwise undefined.

### isSameDeniedGrowthFingerprint

```ts
isSameDeniedGrowthFingerprint(
  deniedGrowthBackoffState: DeniedGrowthBackoffState,
  fingerprint: DeniedGrowthBackoffFingerprint,
): boolean
```

Compare the current deny fingerprint against the stored backoff state.

Parameters:
- `deniedGrowthBackoffState` - Previously recorded deny-backoff state.
- `fingerprint` - Current decision fingerprint.

Returns: True when the deny state is unchanged.

### isSoftBudgetExceeded

```ts
isSoftBudgetExceeded(
  measuredBytes: number | undefined,
  configuredLimitMegabytes: number | undefined,
): boolean
```

Compare one runtime memory reading against a configured soft target.

Parameters:
- `measuredBytes` - Heap bytes reported by the runtime.
- `configuredLimitMegabytes` - Soft target in megabytes.

Returns: True when the runtime is already over the configured soft target.

### normalizeGrowthGraceFraction

```ts
normalizeGrowthGraceFraction(
  growthGraceFraction: number | undefined,
): number
```

Validate and normalize the configured growth-grace fraction.

Parameters:
- `growthGraceFraction` - Raw configured grace fraction.

Returns: Safe non-negative fraction.

### normalizeMaxConnections

```ts
normalizeMaxConnections(
  maxConnections: number,
): number
```

Validate and normalize the configured max-connection cap.

Parameters:
- `maxConnections` - Raw configured connection cap.

Returns: Safe integer cap.

### recordBudgetSnapshot

```ts
recordBudgetSnapshot(
  currentNetwork: default,
  snapshot: Omit<NetworkSparsityBudgetSnapshot, "softBudgetEnvironment" | "softBudgetTriggered">,
  triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined,
): void
```

Persist a budget snapshot with the soft-budget fields filled automatically.

Parameters:
- `currentNetwork` - Network being updated.
- `snapshot` - Snapshot fields excluding soft-budget environment/trigger.
- `triggeredSoftBudgetState` - Active soft-budget pressure, if any.

### recordSparsityBudgetSnapshot

```ts
recordSparsityBudgetSnapshot(
  currentNetwork: default,
  snapshot: NetworkSparsityBudgetSnapshot,
): void
```

Persist the latest read-only decision snapshot.

Parameters:
- `currentNetwork` - Network being updated.
- `snapshot` - Snapshot to store.

Returns: Nothing.

### registerDeniedGrowthBackoff

```ts
registerDeniedGrowthBackoff(
  currentNetwork: default,
  fingerprint: DeniedGrowthBackoffFingerprint,
): void
```

Record one evaluated deny and expand the retry window for unchanged future attempts.

Parameters:
- `currentNetwork` - Network that just denied growth.
- `fingerprint` - Current deny fingerprint.

Returns: Nothing.

### resolveAllowedConnectionLimit

```ts
resolveAllowedConnectionLimit(
  budgetConfig: { maxConnections: number; growthGraceFraction: number; method: PruningMethod; },
): number
```

Resolve the effective connection budget including grace headroom.

Parameters:
- `budgetConfig` - Normalized sparsity-budget configuration.

Returns: Effective allowed connection limit.

### resolveDeniedGrowthBackoffWindow

```ts
resolveDeniedGrowthBackoffWindow(
  consecutiveEvaluatedDenials: number,
): number
```

Resolve how many repeated requests to skip after one evaluated deny.

Parameters:
- `consecutiveEvaluatedDenials` - Number of full reevaluated denies in the same state.

Returns: Remaining retry slots to skip before the next reevaluation.

### resolveEffectiveAllowedConnectionLimit

```ts
resolveEffectiveAllowedConnectionLimit(
  allowedConnectionLimit: number,
  connectionCountBeforeDecision: number,
  triggeredSoftBudgetState: TriggeredSoftBudgetState | undefined,
): number
```

Tighten the effective connection cap when the runtime is already over a soft heap target.

Parameters:
- `allowedConnectionLimit` - Hard connection cap resolved from the network budget.
- `connectionCountBeforeDecision` - Current total connection count.
- `triggeredSoftBudgetState` - Active soft-budget pressure, if any.

Returns: Effective cap for this growth decision.

### resolveTriggeredSoftBudgetState

```ts
resolveTriggeredSoftBudgetState(): TriggeredSoftBudgetState | undefined
```

Detect whether the current runtime heap already exceeds one active soft-memory target.

Returns: Triggered soft-budget details when one environment is over budget.

### shouldSkipDeniedGrowthAttempt

```ts
shouldSkipDeniedGrowthAttempt(
  currentNetwork: default,
  fingerprint: DeniedGrowthBackoffFingerprint,
): boolean
```

Skip one repeated growth attempt when the network is still in the same dead-end state.

Parameters:
- `currentNetwork` - Network about to retry growth.
- `fingerprint` - Current deny fingerprint.

Returns: True when the retry should be denied without reevaluating prune work.

## architecture/network/prune/network.prune.regrowth.utils.ts

### buildRegrowthCandidatePair

```ts
buildRegrowthCandidatePair(
  currentNetwork: default,
): { sourceNode: default; targetNode: default; } | null
```

Build one random regrowth candidate pair if valid.

Parameters:
- `currentNetwork` - Network being regrown.

Returns: Candidate node pair or null when invalid.

### buildRegrowthPlan

```ts
buildRegrowthPlan(
  context: RegrowthPlanContext,
): RegrowthPlan | null
```

Convert regrowth intent into a bounded execution plan.

Parameters:
- `context` - Regrowth planning inputs.

Returns: A plan when regrowth is meaningful; otherwise null.

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
- `currentNetwork` - Network being regrown.
- `sourceNode` - Proposed source node.
- `targetNode` - Proposed target node.

Returns: True when the edge already exists.

### executeRegrowthAttempts

```ts
executeRegrowthAttempts(
  context: RegrowthExecutionContext,
): void
```

Execute bounded stochastic regrowth attempts.

Parameters:
- `context` - Regrowth execution settings.

Returns: Nothing.

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
- `currentNetwork` - Network being regrown.
- `sourceNode` - Proposed source node.
- `targetNode` - Proposed target node.

Returns: True when the pair must be rejected.

### maybeRunRegrowth

```ts
maybeRunRegrowth(
  currentNetwork: default,
  context: RegrowthPlanContext,
): void
```

Build and execute a bounded connection-regrowth plan when regrowth is enabled.

Parameters:
- `currentNetwork` - Network to regrow.
- `context` - Inputs describing regrowth intent.

Returns: Nothing.

### pickRandomNode

```ts
pickRandomNode(
  currentNetwork: default,
): default | undefined
```

Pick a random node using the network RNG.

Parameters:
- `currentNetwork` - Network providing node set and RNG.

Returns: Random node or undefined when the node list is empty.

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
- `currentNetwork` - Network being regrown.
- `desiredRemainingConnections` - Target remaining connection count.
- `attemptedRegrowthCount` - Number of attempts already used.
- `maxAttempts` - Maximum attempts allowed.

Returns: True when another attempt should run.

### tryRegrowConnection

```ts
tryRegrowConnection(
  currentNetwork: default,
): void
```

Attempt one random valid connection addition.

Parameters:
- `currentNetwork` - Network being regrown.

Returns: Nothing.

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
- `currentNetwork` - Network being regrown.
- `sourceNode` - Proposed source node.
- `targetNode` - Proposed target node.

Returns: True when acyclic ordering would be violated.

## architecture/network/prune/network.prune.schedule.utils.ts

### alreadyPrunedThisIteration

```ts
alreadyPrunedThisIteration(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Check whether this iteration was already pruned.

Parameters:
- `currentIteration` - Iteration to evaluate.
- `currentPruningConfig` - Active pruning schedule.

Returns: True when pruning already happened for this iteration.

### buildPruneSelection

```ts
buildPruneSelection(
  context: PruneSelectionContext,
): PruneSelectionResult
```

Build a connection removal selection from ranking context so pruning removes the lowest-priority edges first.
This helper isolates ordering and slicing rules from orchestration code that manages structural side effects.

Parameters:
- `context` - Inputs for ranking and slicing removable connections.

Returns: Connections selected for pruning.

### buildScheduledTarget

```ts
buildScheduledTarget(
  context: ScheduledTargetContext,
  currentConnectionCount: number,
): ScheduledTargetResult
```

Build current scheduled pruning targets from schedule context and current connection totals for this iteration.
The output combines desired remaining edges and immediate excess so callers can prune deterministically.

Parameters:
- `context` - Inputs required to compute desired remaining connections.
- `currentConnectionCount` - Current number of network connections.

Returns: Desired remaining connections and current excess.

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
- `currentIteration` - Iteration to evaluate.
- `scheduleStart` - Start iteration of schedule window.
- `scheduleEnd` - End iteration of schedule window.

Returns: Clamped normalized progress.

### calculateSnipSaliency

```ts
calculateSnipSaliency(
  connection: default,
): number
```

Compute saliency for SNIP-like ranking.

Parameters:
- `connection` - Connection to score.

Returns: Saliency value used for sorting.

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
- `value` - Raw value to clamp.
- `minimum` - Inclusive lower bound.
- `maximum` - Inclusive upper bound.

Returns: Clamped value.

### disconnectConnections

```ts
disconnectConnections(
  currentNetwork: default,
  connectionsToDisconnect: default[],
): void
```

Disconnect all selected connections from the network and schedule post-prune activation-pool compaction when needed.
Grouping removal side effects here keeps pruning orchestration compact and consistent across pruning strategies.

Parameters:
- `currentNetwork` - Network to mutate.
- `connectionsToDisconnect` - Connections to remove.

Returns: Nothing.

### getInitialConnectionBaseline

```ts
getInitialConnectionBaseline(
  currentNetwork: default,
): number | undefined
```

Read the scheduled-pruning baseline connection count used to compute progressive sparsity targets over time.
Keeping this baseline explicit prevents schedule drift when connection totals fluctuate across pruning iterations.

Parameters:
- `currentNetwork` - Network instance to inspect.

Returns: Baseline count when captured; otherwise undefined.

### getPruningConfig

```ts
getPruningConfig(
  currentNetwork: default,
): { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; } | undefined
```

Read the active pruning schedule from network internals so scheduled pruning logic can run against current runtime policy.
Returning the optional config directly keeps orchestration code declarative and avoids repeated unsafe internal casts.

Parameters:
- `currentNetwork` - Network instance to inspect.

Returns: Pruning configuration when enabled; otherwise undefined.

### isOutsidePruningWindow

```ts
isOutsidePruningWindow(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Check whether an iteration is outside the pruning window.

Parameters:
- `currentIteration` - Iteration to evaluate.
- `currentPruningConfig` - Active pruning schedule.

Returns: True when the iteration is out of range.

### isScheduledPruningIteration

```ts
isScheduledPruningIteration(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Check frequency cadence for scheduled pruning.

Parameters:
- `currentIteration` - Iteration to evaluate.
- `currentPruningConfig` - Active pruning schedule.

Returns: True when this iteration matches the schedule cadence.

### markPruneIteration

```ts
markPruneIteration(
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
  currentIteration: number,
): void
```

Persist the iteration that last performed pruning so duplicate schedule triggers within the same step are ignored.
This marker is essential for idempotent training loops that may re-enter pruning checks.

Parameters:
- `currentPruningConfig` - Active pruning configuration.
- `currentIteration` - Iteration to record.

Returns: Nothing.

### markTopologyDirty

```ts
markTopologyDirty(
  currentNetwork: default,
): void
```

Mark the topology cache as dirty after scheduled pruning structural updates.

Parameters:
- `currentNetwork` - Network with modified connectivity.

Returns: Nothing.

### rankConnectionsByMagnitude

```ts
rankConnectionsByMagnitude(
  connections: default[],
): default[]
```

Rank connections by absolute weight magnitude.

Parameters:
- `connections` - Candidate connections to rank.

Returns: Connections sorted by ascending absolute weight.

### rankConnectionsByRemovalPriority

```ts
rankConnectionsByRemovalPriority(
  connections: default[],
  method: PruningMethod,
): default[]
```

Route ranking to the configured pruning heuristic.

Parameters:
- `connections` - Candidate connections to rank.
- `method` - Ranking method to apply.

Returns: Connections sorted by ascending removal priority.

### rankConnectionsBySnipSaliency

```ts
rankConnectionsBySnipSaliency(
  connections: default[],
): default[]
```

Rank connections by SNIP-like saliency approximation.

Parameters:
- `connections` - Candidate connections to rank.

Returns: Connections sorted by ascending saliency.

### resolveGradientMagnitude

```ts
resolveGradientMagnitude(
  connection: default,
): number
```

Resolve a stable gradient-magnitude proxy from connection delta statistics.

Parameters:
- `connection` - Connection containing accumulated delta history.

Returns: Absolute gradient magnitude proxy.

### resolvePruningMethod

```ts
resolvePruningMethod(
  method: PruningMethod | undefined,
): PruningMethod
```

Normalize an optional pruning method to a concrete default value.

Parameters:
- `method` - Optional configured pruning method.

Returns: Concrete pruning method.

### shouldRunScheduledPrune

```ts
shouldRunScheduledPrune(
  currentIteration: number,
  currentPruningConfig: { start: number; end: number; frequency: number; targetSparsity: number; method: PruningMethod; regrowFraction: number; lastPruneIter?: number | undefined; },
): boolean
```

Determine whether scheduled pruning should run at this iteration using configured window bounds and cadence guards.
The decision also prevents duplicate pruning passes within the same iteration when callers retry training steps.

Parameters:
- `currentIteration` - Training iteration being processed.
- `currentPruningConfig` - Active pruning schedule.

Returns: True when pruning should execute now.

## architecture/network/prune/network.prune.sparsity.utils.ts

### calculateSparsityFromBaseline

```ts
calculateSparsityFromBaseline(
  currentConnectionCount: number,
  baselineConnectionCount: number,
): number
```

Convert current connection density into a normalized sparsity ratio value.

Parameters:
- `currentConnectionCount` - Current connection count.
- `baselineConnectionCount` - Baseline connection count.

Returns: Sparsity ratio in [0,1] for valid baselines.

### readInitialSparsityBaseline

```ts
readInitialSparsityBaseline(
  currentNetwork: default,
): number | undefined
```

Read the initial connection-count baseline value used for sparsity ratio reporting.

Parameters:
- `currentNetwork` - Network to inspect.

Returns: Baseline connection count when available.

## architecture/network/prune/network.prune.evolutionary.utils.ts

### buildEvolutionaryPruneSelection

```ts
buildEvolutionaryPruneSelection(
  context: PruneSelectionContext,
): PruneSelectionResult
```

Build evolutionary pruning connection selection.
Ranking and slicing happen in one deterministic pass so stochastic training variance does not reorder equal-score candidates across repeated pruning runs.

Parameters:
- `context` - Inputs for ranking and slicing.

Returns: Connections selected for removal.

### buildEvolutionaryTarget

```ts
buildEvolutionaryTarget(
  context: EvolutionaryTargetContext,
  currentConnectionCount: number,
): EvolutionaryTargetResult
```

Compute evolutionary pruning target counts.
This conversion translates a normalized sparsity objective into concrete connection counts while enforcing the minimum remaining edge safety floor.

Parameters:
- `context` - Inputs for sparsity-to-count conversion.
- `currentConnectionCount` - Current number of network connections.

Returns: Desired remaining and excess connection counts.

### calculateEvolutionarySnipSaliency

```ts
calculateEvolutionarySnipSaliency(
  connection: default,
): number
```

Compute evolutionary SNIP-like saliency for one connection.

Parameters:
- `connection` - Connection to score.

Returns: Saliency score.

### disconnectEvolutionaryConnections

```ts
disconnectEvolutionaryConnections(
  currentNetwork: default,
  connectionsToDisconnect: default[],
): void
```

Disconnect selected evolutionary pruning edges.
Removal is followed by deferred activation-pool compaction scheduling so large structural contractions can reclaim memory without forcing immediate synchronous pool reshaping.

Parameters:
- `currentNetwork` - Network to mutate.
- `connectionsToDisconnect` - Edges to remove.

Returns: Nothing.

### getOrCaptureEvolutionaryBaseline

```ts
getOrCaptureEvolutionaryBaseline(
  currentNetwork: default,
): number
```

Capture evolutionary baseline once and reuse it for subsequent pruning calls.
The baseline anchors target sparsity to the original connection budget so repeated prune cycles converge predictably instead of drifting with the current graph size.

Parameters:
- `currentNetwork` - Network to inspect and possibly initialize.

Returns: Evolutionary baseline connection count.

### markEvolutionaryTopologyDirty

```ts
markEvolutionaryTopologyDirty(
  currentNetwork: default,
): void
```

Mark the topology cache as dirty after evolutionary pruning removes connections.

Parameters:
- `currentNetwork` - Network with changed structure.

Returns: Nothing.

### normalizeEvolutionaryTargetSparsity

```ts
normalizeEvolutionaryTargetSparsity(
  rawTargetSparsity: number,
): number
```

Clamp evolutionary target sparsity to safe operational bounds for pruning.

Parameters:
- `rawTargetSparsity` - Requested target sparsity.

Returns: Normalized target sparsity.

### rankEvolutionaryConnections

```ts
rankEvolutionaryConnections(
  connections: default[],
  pruningMethod: PruningMethod,
): default[]
```

Route evolutionary ranking to selected heuristic.

Parameters:
- `connections` - Candidate connections.
- `pruningMethod` - Ranking heuristic.

Returns: Connections sorted by ascending removal priority.

### rankEvolutionaryConnectionsByMagnitude

```ts
rankEvolutionaryConnectionsByMagnitude(
  connections: default[],
): default[]
```

Rank connections by magnitude for evolutionary pruning.

Parameters:
- `connections` - Candidate connections.

Returns: Connections sorted by ascending absolute weight.

### rankEvolutionaryConnectionsBySnip

```ts
rankEvolutionaryConnectionsBySnip(
  connections: default[],
): default[]
```

Rank connections by SNIP-like saliency for evolutionary pruning.

Parameters:
- `connections` - Candidate connections.

Returns: Connections sorted by ascending saliency.

### resolveEvolutionaryGradientMagnitude

```ts
resolveEvolutionaryGradientMagnitude(
  connection: default,
): number
```

Resolve gradient proxy for evolutionary SNIP ranking.

Parameters:
- `connection` - Connection containing delta history.

Returns: Absolute gradient magnitude proxy.
