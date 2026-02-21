import type Network from '../../network';
import Connection from '../../connection';
import type Node from '../../node';
import type {
  EvolutionaryTargetContext,
  EvolutionaryTargetResult,
  NetworkPruningProps,
  PruneSelectionContext,
  PruneSelectionResult,
  PruningMethod,
  RegrowthExecutionContext,
  RegrowthPlan,
  RegrowthPlanContext,
  ScheduledTargetContext,
  ScheduledTargetResult,
} from '../network.types';

/** Pruning method identifier for absolute-weight ranking. */
const PRUNING_METHOD_MAGNITUDE = 'magnitude' as const;
/** Pruning method identifier for SNIP-like saliency ranking. */
const PRUNING_METHOD_SNIP = 'snip' as const;
/** Lower bound to ensure at least one connection remains after pruning. */
const MIN_REMAINING_CONNECTION_COUNT = 1;
/** Fallback prune cadence when schedule frequency is omitted or invalid. */
const DEFAULT_PRUNE_FREQUENCY = 1;
/** Minimum normalized schedule progress value. */
const MIN_PROGRESS_FRACTION = 0;
/** Maximum normalized schedule progress value. */
const MAX_PROGRESS_FRACTION = 1;
/** Safety cap below full sparsity to avoid degenerate zero-connection networks. */
const MAX_EVOLUTIONARY_TARGET_SPARSITY = 0.999;
/** Retry multiplier to convert intended regrowth count into max attempts. */
const REGROW_ATTEMPT_MULTIPLIER = 10;

/**
 * Structured and dynamic pruning utilities for networks.
 *
 * Features:
 *  - Scheduled pruning during gradient-based training ({@link maybePrune}) with linear sparsity ramp.
 *  - Evolutionary generation pruning toward a target sparsity ({@link pruneToSparsity}).
 *  - Two ranking heuristics:
 *      magnitude: |w|
 *      snip: |w * g| approximation (g approximated via accumulated delta stats; falls back to |w|)
 *  - Optional stochastic regrowth during scheduled pruning (dynamic sparse training), preserving acyclic constraints.
 *
 * Internal State Fields (attached to Network via `any` casting):
 *  - _pruningConfig: user-specified schedule & options (start, end, frequency, targetSparsity, method, regrowFraction, lastPruneIter)
 *  - _initialConnectionCount: baseline connection count captured outside (first training iteration)
 *  - _evoInitialConnCount: baseline for evolutionary pruning (first invocation of pruneToSparsity)
 *  - _rand: deterministic RNG function
 *  - _enforceAcyclic: boolean flag enforcing forward-only connectivity ordering
 *  - _topoDirty: topology order invalidation flag consumed by activation fast path / topological sorting
 */

/**
 * Opportunistically perform scheduled pruning during gradient-based training.
 *
 * Scheduling model:
 *  - start / end define an iteration window (inclusive) during which pruning may occur
 *  - frequency defines cadence (every N iterations inside the window)
 *  - targetSparsity is linearly annealed from 0 to its final value across the window
 *  - method chooses ranking heuristic (magnitude | snip)
 *  - optional regrowFraction allows dynamic sparse training: after removing edges we probabilistically regrow
 *    a fraction of them at random unused positions (respecting acyclic constraint if enforced)
 *
 * SNIP heuristic:
 *  - Uses |w * grad| style saliency approximation (here reusing stored delta stats as gradient proxy)
 *  - Falls back to pure magnitude if gradient stats absent.
 */
/**
 * Perform scheduled pruning at a given training iteration if conditions are met.
 *
 * Scheduling fields (cfg): start, end, frequency, targetSparsity, method ('magnitude' | 'snip'), regrowFraction.
 * The target sparsity ramps linearly from 0 at start to cfg.targetSparsity at end.
 *
 * @param iteration Current (0-based or 1-based) training iteration counter used for scheduling.
 */
export function maybePrune(this: Network, iteration: number): void {
  // Step 1: Collect required schedule and baseline context.
  const network = this;
  const pruningConfig = getPruningConfig(network);
  if (!pruningConfig) return;

  // Step 2: Exit early when this iteration is not eligible for pruning.
  if (!shouldRunScheduledPrune(iteration, pruningConfig)) return;

  const initialConnectionBaseline = getInitialConnectionBaseline(network);
  if (!initialConnectionBaseline) return;

  // Step 3: Compute current target density and required removals.
  const scheduledTarget = buildScheduledTarget({
    iteration,
    scheduleStart: pruningConfig.start,
    scheduleEnd: pruningConfig.end,
    targetSparsity: pruningConfig.targetSparsity,
    baselineConnectionCount: initialConnectionBaseline,
  });

  if (scheduledTarget.excessConnectionCount <= 0) {
    markPruneIteration(pruningConfig, iteration);
    return;
  }

  // Step 4: Select and remove the least important connections.
  const pruneSelection = buildPruneSelection({
    connections: network.connections,
    removalCount: scheduledTarget.excessConnectionCount,
    method: resolvePruningMethod(pruningConfig.method),
  });

  disconnectConnections(network, pruneSelection.connectionsToPrune);

  // Step 5: Optionally regrow random valid edges.
  maybeRunRegrowth(network, {
    prunedConnectionCount: pruneSelection.connectionsToPrune.length,
    regrowFraction: pruningConfig.regrowFraction,
    desiredRemainingConnections: scheduledTarget.desiredRemainingConnections,
  });

  // Step 6: Persist bookkeeping for topology and schedule state.
  markPruneIteration(pruningConfig, iteration);
  markTopologyDirty(network);

  /**
   * Read the active pruning schedule from network internals.
   * @param currentNetwork - Network instance to inspect.
   * @returns Pruning configuration when enabled; otherwise undefined.
   */
  function getPruningConfig(
    currentNetwork: Network,
  ): NetworkPruningProps['_pruningConfig'] | undefined {
    return (currentNetwork as unknown as NetworkPruningProps)._pruningConfig;
  }

  /**
   * Determine whether scheduled pruning should run at this iteration.
   * @param currentIteration - Training iteration being processed.
   * @param currentPruningConfig - Active pruning schedule.
   * @returns True when pruning should execute now.
   */
  function shouldRunScheduledPrune(
    currentIteration: number,
    currentPruningConfig: NonNullable<NetworkPruningProps['_pruningConfig']>,
  ): boolean {
    // Step 1: Reject iterations outside the configured window.
    if (isOutsidePruningWindow(currentIteration, currentPruningConfig)) {
      return false;
    }

    // Step 2: Reject duplicate pruning within the same iteration.
    if (alreadyPrunedThisIteration(currentIteration, currentPruningConfig)) {
      return false;
    }

    // Step 3: Check frequency cadence alignment.
    return isScheduledPruningIteration(currentIteration, currentPruningConfig);
  }

  /**
   * Check whether an iteration is outside the pruning window.
   * @param currentIteration - Iteration to evaluate.
   * @param currentPruningConfig - Active pruning schedule.
   * @returns True when the iteration is out of range.
   */
  function isOutsidePruningWindow(
    currentIteration: number,
    currentPruningConfig: NonNullable<NetworkPruningProps['_pruningConfig']>,
  ): boolean {
    return (
      currentIteration < currentPruningConfig.start ||
      currentIteration > currentPruningConfig.end
    );
  }

  /**
   * Check whether this iteration was already pruned.
   * @param currentIteration - Iteration to evaluate.
   * @param currentPruningConfig - Active pruning schedule.
   * @returns True when pruning already happened for this iteration.
   */
  function alreadyPrunedThisIteration(
    currentIteration: number,
    currentPruningConfig: NonNullable<NetworkPruningProps['_pruningConfig']>,
  ): boolean {
    return (
      currentPruningConfig.lastPruneIter != null &&
      currentPruningConfig.lastPruneIter === currentIteration
    );
  }

  /**
   * Check frequency cadence for scheduled pruning.
   * @param currentIteration - Iteration to evaluate.
   * @param currentPruningConfig - Active pruning schedule.
   * @returns True when this iteration matches the schedule cadence.
   */
  function isScheduledPruningIteration(
    currentIteration: number,
    currentPruningConfig: NonNullable<NetworkPruningProps['_pruningConfig']>,
  ): boolean {
    // Step 1: Resolve a safe cadence value.
    const pruningFrequency =
      currentPruningConfig.frequency || DEFAULT_PRUNE_FREQUENCY;

    // Step 2: Evaluate start-offset divisibility.
    return (
      (currentIteration - currentPruningConfig.start) % pruningFrequency === 0
    );
  }

  /**
   * Read the scheduled-pruning baseline connection count.
   * @param currentNetwork - Network instance to inspect.
   * @returns Baseline count when captured; otherwise undefined.
   */
  function getInitialConnectionBaseline(
    currentNetwork: Network,
  ): number | undefined {
    return (currentNetwork as unknown as NetworkPruningProps)
      ._initialConnectionCount;
  }

  /**
   * Build current scheduled pruning targets from schedule context.
   * @param context - Inputs required to compute desired remaining connections.
   * @returns Desired remaining connections and current excess.
   */
  function buildScheduledTarget(
    context: ScheduledTargetContext,
  ): ScheduledTargetResult {
    // Step 1: Compute normalized schedule progress.
    const progressFraction = calculateProgressFraction(
      context.iteration,
      context.scheduleStart,
      context.scheduleEnd,
    );

    // Step 2: Convert progress to instantaneous target sparsity.
    const targetSparsityNow = context.targetSparsity * progressFraction;

    // Step 3: Convert sparsity into required remaining connection count.
    const desiredRemainingConnections = Math.max(
      MIN_REMAINING_CONNECTION_COUNT,
      Math.floor(context.baselineConnectionCount * (1 - targetSparsityNow)),
    );

    // Step 4: Compute how many connections must be removed now.
    const excessConnectionCount =
      network.connections.length - desiredRemainingConnections;

    return {
      desiredRemainingConnections,
      excessConnectionCount,
    };
  }

  /**
   * Compute clamped schedule progress in the [0,1] range.
   * @param currentIteration - Iteration to evaluate.
   * @param scheduleStart - Start iteration of schedule window.
   * @param scheduleEnd - End iteration of schedule window.
   * @returns Clamped normalized progress.
   */
  function calculateProgressFraction(
    currentIteration: number,
    scheduleStart: number,
    scheduleEnd: number,
  ): number {
    // Step 1: Protect denominator from zero-span schedules.
    const scheduleSpan = Math.max(1, scheduleEnd - scheduleStart);

    // Step 2: Compute raw fractional progress.
    const rawProgressFraction =
      (currentIteration - scheduleStart) / scheduleSpan;

    // Step 3: Clamp progress to stable bounds.
    return clamp(
      rawProgressFraction,
      MIN_PROGRESS_FRACTION,
      MAX_PROGRESS_FRACTION,
    );
  }

  /**
   * Clamp a number into an inclusive range.
   * @param value - Raw value to clamp.
   * @param minimum - Inclusive lower bound.
   * @param maximum - Inclusive upper bound.
   * @returns Clamped value.
   */
  function clamp(value: number, minimum: number, maximum: number): number {
    return Math.min(maximum, Math.max(minimum, value));
  }

  /**
   * Build a connection removal selection from current ranking context.
   * @param context - Inputs for ranking and slicing removable connections.
   * @returns Connections selected for pruning.
   */
  function buildPruneSelection(
    context: PruneSelectionContext,
  ): PruneSelectionResult {
    // Step 1: Rank from lowest to highest importance.
    const rankedConnections = rankConnectionsByRemovalPriority(
      context.connections,
      context.method,
    );

    // Step 2: Select the exact number of removable edges.
    const connectionsToPrune = rankedConnections.slice(0, context.removalCount);
    return { connectionsToPrune };
  }

  /**
   * Route ranking to the configured pruning heuristic.
   * @param connections - Candidate connections to rank.
   * @param method - Ranking method to apply.
   * @returns Connections sorted by ascending removal priority.
   */
  function rankConnectionsByRemovalPriority(
    connections: Connection[],
    method: PruningMethod,
  ): Connection[] {
    if (method === PRUNING_METHOD_SNIP) {
      return rankConnectionsBySnipSaliency(connections);
    }
    return rankConnectionsByMagnitude(connections);
  }

  /**
   * Rank connections by absolute weight magnitude.
   * @param connections - Candidate connections to rank.
   * @returns Connections sorted by ascending absolute weight.
   */
  function rankConnectionsByMagnitude(connections: Connection[]): Connection[] {
    return connections.toSorted(
      (leftConnection, rightConnection) =>
        Math.abs(leftConnection.weight) - Math.abs(rightConnection.weight),
    );
  }

  /**
   * Rank connections by SNIP-like saliency approximation.
   * @param connections - Candidate connections to rank.
   * @returns Connections sorted by ascending saliency.
   */
  function rankConnectionsBySnipSaliency(
    connections: Connection[],
  ): Connection[] {
    return connections.toSorted(
      (leftConnection, rightConnection) =>
        calculateSnipSaliency(leftConnection) -
        calculateSnipSaliency(rightConnection),
    );
  }

  /**
   * Compute saliency for SNIP-like ranking.
   * @param connection - Connection to score.
   * @returns Saliency value used for sorting.
   */
  function calculateSnipSaliency(connection: Connection): number {
    // Step 1: Estimate gradient magnitude from stored deltas.
    const gradientMagnitude = resolveGradientMagnitude(connection);

    // Step 2: Fall back to magnitude-only saliency when gradient is unavailable.
    if (gradientMagnitude === 0) {
      return Math.abs(connection.weight);
    }
    return Math.abs(connection.weight) * gradientMagnitude;
  }

  /**
   * Resolve a stable gradient-magnitude proxy from connection delta statistics.
   * @param connection - Connection containing accumulated delta history.
   * @returns Absolute gradient magnitude proxy.
   */
  function resolveGradientMagnitude(connection: Connection): number {
    return (
      Math.abs(connection.totalDeltaWeight) ||
      Math.abs(connection.previousDeltaWeight) ||
      0
    );
  }

  /**
   * Disconnect all selected connections from the network.
   * @param currentNetwork - Network to mutate.
   * @param connectionsToDisconnect - Connections to remove.
   * @returns Nothing.
   */
  function disconnectConnections(
    currentNetwork: Network,
    connectionsToDisconnect: Connection[],
  ): void {
    // Step 1: Remove each selected edge.
    connectionsToDisconnect.forEach((connection) => {
      currentNetwork.disconnect(connection.from, connection.to);
    });
  }

  /**
   * Build and execute a regrowth plan when enabled.
   * @param currentNetwork - Network to regrow.
   * @param context - Inputs describing regrowth intent.
   * @returns Nothing.
   */
  function maybeRunRegrowth(
    currentNetwork: Network,
    context: RegrowthPlanContext,
  ): void {
    // Step 1: Build a bounded regrowth execution plan.
    const regrowthPlan = buildRegrowthPlan(context);
    if (!regrowthPlan) return;

    // Step 2: Execute random valid edge additions.
    executeRegrowthAttempts({
      network: currentNetwork,
      desiredRemainingConnections: regrowthPlan.desiredRemainingConnections,
      maxAttempts: regrowthPlan.maxAttempts,
    });
  }

  /**
   * Convert regrowth intent into a bounded execution plan.
   * @param context - Regrowth planning inputs.
   * @returns A plan when regrowth is meaningful; otherwise null.
   */
  function buildRegrowthPlan(
    context: RegrowthPlanContext,
  ): RegrowthPlan | null {
    // Step 1: Skip disabled regrowth.
    if (context.regrowFraction <= 0) return null;

    // Step 2: Compute requested number of regrown connections.
    const intendedRegrowCount = Math.floor(
      context.prunedConnectionCount * context.regrowFraction,
    );
    if (intendedRegrowCount <= 0) return null;

    // Step 3: Translate requested count into attempt budget.
    return {
      desiredRemainingConnections: context.desiredRemainingConnections,
      maxAttempts: intendedRegrowCount * REGROW_ATTEMPT_MULTIPLIER,
    };
  }

  /**
   * Execute bounded stochastic regrowth attempts.
   * @param context - Regrowth execution settings.
   * @returns Nothing.
   */
  function executeRegrowthAttempts(context: RegrowthExecutionContext): void {
    // Step 1: Track how many tries have been consumed.
    let attemptedRegrowthCount = 0;

    // Step 2: Keep trying until target density or attempt cap is reached.
    while (
      shouldContinueRegrowth(
        context.network,
        context.desiredRemainingConnections,
        attemptedRegrowthCount,
        context.maxAttempts,
      )
    ) {
      // Step 3: Consume one attempt and try to add one valid edge.
      attemptedRegrowthCount += 1;
      tryRegrowConnection(context.network);
    }
  }

  /**
   * Decide whether another regrowth attempt is allowed.
   * @param currentNetwork - Network being regrown.
   * @param desiredRemainingConnections - Target remaining connection count.
   * @param attemptedRegrowthCount - Number of attempts already used.
   * @param maxAttempts - Maximum attempts allowed.
   * @returns True when another attempt should run.
   */
  function shouldContinueRegrowth(
    currentNetwork: Network,
    desiredRemainingConnections: number,
    attemptedRegrowthCount: number,
    maxAttempts: number,
  ): boolean {
    return (
      currentNetwork.connections.length < desiredRemainingConnections &&
      attemptedRegrowthCount < maxAttempts
    );
  }

  /**
   * Attempt one random valid connection addition.
   * @param currentNetwork - Network being regrown.
   * @returns Nothing.
   */
  function tryRegrowConnection(currentNetwork: Network): void {
    // Step 1: Build a valid random source-target pair.
    const regrowthPair = buildRegrowthCandidatePair(currentNetwork);
    if (!regrowthPair) return;

    // Step 2: Materialize the new edge.
    currentNetwork.connect(regrowthPair.sourceNode, regrowthPair.targetNode);
  }

  /**
   * Build one random regrowth candidate pair if valid.
   * @param currentNetwork - Network being regrown.
   * @returns Candidate node pair or null when invalid.
   */
  function buildRegrowthCandidatePair(
    currentNetwork: Network,
  ): { sourceNode: Node; targetNode: Node } | null {
    // Step 1: Draw random source and target nodes.
    const sourceNode = pickRandomNode(currentNetwork);
    const targetNode = pickRandomNode(currentNetwork);
    if (!sourceNode || !targetNode) return null;

    // Step 2: Validate candidate constraints.
    if (isInvalidRegrowthPair(currentNetwork, sourceNode, targetNode)) {
      return null;
    }

    return { sourceNode, targetNode };
  }

  /**
   * Pick a random node using the network RNG.
   * @param currentNetwork - Network providing node set and RNG.
   * @returns Random node or undefined when the node list is empty.
   */
  function pickRandomNode(currentNetwork: Network): Node | undefined {
    // Step 1: Resolve deterministic RNG when available.
    const randomSource =
      (currentNetwork as unknown as NetworkPruningProps)._rand ?? Math.random;

    // Step 2: Sample one node index uniformly.
    const randomIndex = Math.floor(
      randomSource() * currentNetwork.nodes.length,
    );
    return currentNetwork.nodes[randomIndex];
  }

  /**
   * Validate whether a candidate regrowth pair is acceptable.
   * @param currentNetwork - Network being regrown.
   * @param sourceNode - Proposed source node.
   * @param targetNode - Proposed target node.
   * @returns True when the pair must be rejected.
   */
  function isInvalidRegrowthPair(
    currentNetwork: Network,
    sourceNode: Node,
    targetNode: Node,
  ): boolean {
    if (sourceNode === targetNode) return true;
    if (connectionAlreadyExists(currentNetwork, sourceNode, targetNode)) {
      return true;
    }
    return violatesAcyclicConstraint(currentNetwork, sourceNode, targetNode);
  }

  /**
   * Check whether a connection already exists.
   * @param currentNetwork - Network being regrown.
   * @param sourceNode - Proposed source node.
   * @param targetNode - Proposed target node.
   * @returns True when the edge already exists.
   */
  function connectionAlreadyExists(
    currentNetwork: Network,
    sourceNode: Node,
    targetNode: Node,
  ): boolean {
    return currentNetwork.connections.some(
      (connection) =>
        connection.from === sourceNode && connection.to === targetNode,
    );
  }

  /**
   * Check whether a pair violates forward-only acyclic ordering.
   * @param currentNetwork - Network being regrown.
   * @param sourceNode - Proposed source node.
   * @param targetNode - Proposed target node.
   * @returns True when acyclic ordering would be violated.
   */
  function violatesAcyclicConstraint(
    currentNetwork: Network,
    sourceNode: Node,
    targetNode: Node,
  ): boolean {
    // Step 1: Skip ordering checks when acyclic enforcement is disabled.
    const networkProperties = currentNetwork as unknown as NetworkPruningProps;
    if (!networkProperties._enforceAcyclic) return false;

    // Step 2: Reject backward edges by node index ordering.
    return (
      currentNetwork.nodes.indexOf(sourceNode) >
      currentNetwork.nodes.indexOf(targetNode)
    );
  }

  /**
   * Normalize optional pruning method to a concrete value.
   * @param method - Optional configured pruning method.
   * @returns Concrete pruning method.
   */
  function resolvePruningMethod(
    method: PruningMethod | undefined,
  ): PruningMethod {
    return method ?? PRUNING_METHOD_MAGNITUDE;
  }

  /**
   * Persist the iteration that last performed pruning.
   * @param currentPruningConfig - Active pruning configuration.
   * @param currentIteration - Iteration to record.
   * @returns Nothing.
   */
  function markPruneIteration(
    currentPruningConfig: NonNullable<NetworkPruningProps['_pruningConfig']>,
    currentIteration: number,
  ): void {
    currentPruningConfig.lastPruneIter = currentIteration;
  }

  /**
   * Mark topology cache as dirty after structural updates.
   * @param currentNetwork - Network with modified connectivity.
   * @returns Nothing.
   */
  function markTopologyDirty(currentNetwork: Network): void {
    (currentNetwork as unknown as NetworkPruningProps)._topoDirty = true;
  }
}

/**
 * Evolutionary (generation-based) pruning toward a target sparsity baseline.
 * Unlike maybePrune this operates immediately relative to the first invocation's connection count
 * (stored separately as _evoInitialConnCount) and does not implement scheduling or regrowth.
 */
export function pruneToSparsity(
  this: Network,
  targetSparsity: number,
  method: PruningMethod = PRUNING_METHOD_MAGNITUDE,
): void {
  // Step 1: Normalize user target and short-circuit no-op requests.
  const network = this;
  const normalizedTargetSparsity =
    normalizeEvolutionaryTargetSparsity(targetSparsity);
  if (normalizedTargetSparsity <= 0) return;

  // Step 2: Resolve baseline and compute required removals.
  const evolutionaryBaseline = getOrCaptureEvolutionaryBaseline(network);
  const evolutionaryTarget = buildEvolutionaryTarget({
    targetSparsity: normalizedTargetSparsity,
    baselineConnectionCount: evolutionaryBaseline,
  });
  if (evolutionaryTarget.excessConnectionCount <= 0) return;

  // Step 3: Select and remove least important connections.
  const pruneSelection = buildEvolutionaryPruneSelection({
    connections: network.connections,
    removalCount: evolutionaryTarget.excessConnectionCount,
    method,
  });

  disconnectEvolutionaryConnections(network, pruneSelection.connectionsToPrune);

  // Step 4: Mark topology cache invalid after structural change.
  markEvolutionaryTopologyDirty(network);

  /**
   * Clamp evolutionary target sparsity to safe operational bounds.
   * @param rawTargetSparsity - Requested target sparsity.
   * @returns Normalized target sparsity.
   */
  function normalizeEvolutionaryTargetSparsity(
    rawTargetSparsity: number,
  ): number {
    if (rawTargetSparsity <= 0) return 0;
    if (rawTargetSparsity >= 1) return MAX_EVOLUTIONARY_TARGET_SPARSITY;
    return rawTargetSparsity;
  }

  /**
   * Capture evolutionary baseline once and reuse it for subsequent pruning calls.
   * @param currentNetwork - Network to inspect and possibly initialize.
   * @returns Evolutionary baseline connection count.
   */
  function getOrCaptureEvolutionaryBaseline(currentNetwork: Network): number {
    // Step 1: Access internal baseline storage.
    const networkProperties = currentNetwork as unknown as NetworkPruningProps;

    // Step 2: Initialize baseline on first call.
    if (!networkProperties._evoInitialConnCount) {
      networkProperties._evoInitialConnCount =
        currentNetwork.connections.length;
    }
    return networkProperties._evoInitialConnCount;
  }

  /**
   * Compute evolutionary pruning target counts.
   * @param context - Inputs for sparsity-to-count conversion.
   * @returns Desired remaining and excess connection counts.
   */
  function buildEvolutionaryTarget(
    context: EvolutionaryTargetContext,
  ): EvolutionaryTargetResult {
    // Step 1: Convert target sparsity to desired retained count.
    const desiredRemainingConnections = Math.max(
      MIN_REMAINING_CONNECTION_COUNT,
      Math.floor(
        context.baselineConnectionCount * (1 - context.targetSparsity),
      ),
    );

    // Step 2: Derive how many edges must be removed.
    const excessConnectionCount =
      network.connections.length - desiredRemainingConnections;
    return { desiredRemainingConnections, excessConnectionCount };
  }

  /**
   * Build evolutionary pruning connection selection.
   * @param context - Inputs for ranking and slicing.
   * @returns Connections selected for removal.
   */
  function buildEvolutionaryPruneSelection(
    context: PruneSelectionContext,
  ): PruneSelectionResult {
    // Step 1: Rank according to selected heuristic.
    const rankedConnections = rankEvolutionaryConnections(
      context.connections,
      context.method,
    );

    // Step 2: Select top removable prefix.
    const connectionsToPrune = rankedConnections.slice(0, context.removalCount);
    return { connectionsToPrune };
  }

  /**
   * Route evolutionary ranking to selected heuristic.
   * @param connections - Candidate connections.
   * @param pruningMethod - Ranking heuristic.
   * @returns Connections sorted by ascending removal priority.
   */
  function rankEvolutionaryConnections(
    connections: Connection[],
    pruningMethod: PruningMethod,
  ): Connection[] {
    if (pruningMethod === PRUNING_METHOD_SNIP) {
      return rankEvolutionaryConnectionsBySnip(connections);
    }
    return rankEvolutionaryConnectionsByMagnitude(connections);
  }

  /**
   * Rank connections by magnitude for evolutionary pruning.
   * @param connections - Candidate connections.
   * @returns Connections sorted by ascending absolute weight.
   */
  function rankEvolutionaryConnectionsByMagnitude(
    connections: Connection[],
  ): Connection[] {
    return connections.toSorted(
      (leftConnection, rightConnection) =>
        Math.abs(leftConnection.weight) - Math.abs(rightConnection.weight),
    );
  }

  /**
   * Rank connections by SNIP-like saliency for evolutionary pruning.
   * @param connections - Candidate connections.
   * @returns Connections sorted by ascending saliency.
   */
  function rankEvolutionaryConnectionsBySnip(
    connections: Connection[],
  ): Connection[] {
    return connections.toSorted(
      (leftConnection, rightConnection) =>
        calculateEvolutionarySnipSaliency(leftConnection) -
        calculateEvolutionarySnipSaliency(rightConnection),
    );
  }

  /**
   * Compute evolutionary SNIP-like saliency for one connection.
   * @param connection - Connection to score.
   * @returns Saliency score.
   */
  function calculateEvolutionarySnipSaliency(connection: Connection): number {
    // Step 1: Resolve gradient proxy from delta history.
    const gradientMagnitude = resolveEvolutionaryGradientMagnitude(connection);

    // Step 2: Fall back to pure magnitude when gradient info is absent.
    if (gradientMagnitude === 0) {
      return Math.abs(connection.weight);
    }
    return Math.abs(connection.weight) * gradientMagnitude;
  }

  /**
   * Resolve gradient proxy for evolutionary SNIP ranking.
   * @param connection - Connection containing delta history.
   * @returns Absolute gradient magnitude proxy.
   */
  function resolveEvolutionaryGradientMagnitude(
    connection: Connection,
  ): number {
    return (
      Math.abs(connection.totalDeltaWeight) ||
      Math.abs(connection.previousDeltaWeight) ||
      0
    );
  }

  /**
   * Disconnect selected evolutionary pruning edges.
   * @param currentNetwork - Network to mutate.
   * @param connectionsToDisconnect - Edges to remove.
   * @returns Nothing.
   */
  function disconnectEvolutionaryConnections(
    currentNetwork: Network,
    connectionsToDisconnect: Connection[],
  ): void {
    // Step 1: Remove each selected connection.
    connectionsToDisconnect.forEach((connection) => {
      currentNetwork.disconnect(connection.from, connection.to);
    });
  }

  /**
   * Mark topology cache as dirty after evolutionary pruning.
   * @param currentNetwork - Network with changed structure.
   * @returns Nothing.
   */
  function markEvolutionaryTopologyDirty(currentNetwork: Network): void {
    (currentNetwork as unknown as NetworkPruningProps)._topoDirty = true;
  }
}

/** Current sparsity fraction relative to the training-time pruning baseline. */
export function getCurrentSparsity(this: Network): number {
  // Step 1: Resolve baseline and return dense default when unavailable.
  const network = this;
  const initialBaseline = readInitialSparsityBaseline(network);
  if (!initialBaseline) return 0;

  // Step 2: Compute current sparsity from baseline.
  return calculateSparsityFromBaseline(
    network.connections.length,
    initialBaseline,
  );

  /**
   * Read baseline used for sparsity reporting.
   * @param currentNetwork - Network to inspect.
   * @returns Baseline connection count when available.
   */
  function readInitialSparsityBaseline(
    currentNetwork: Network,
  ): number | undefined {
    return (currentNetwork as unknown as NetworkPruningProps)
      ._initialConnectionCount;
  }

  /**
   * Convert current density into sparsity ratio.
   * @param currentConnectionCount - Current connection count.
   * @param baselineConnectionCount - Baseline connection count.
   * @returns Sparsity ratio in [0,1] for valid baselines.
   */
  function calculateSparsityFromBaseline(
    currentConnectionCount: number,
    baselineConnectionCount: number,
  ): number {
    return 1 - currentConnectionCount / baselineConnectionCount;
  }
}

// Explicit export object to keep module side-effects clear (tree-shaking friendliness)
export {};
