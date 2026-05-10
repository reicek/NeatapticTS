import Connection from '../../connection';
import { activationArrayPool } from '../../activationArrayPool/activationArrayPool';
import type Network from '../../network/network';
import type {
  NetworkPruningProps,
  PruneSelectionContext,
  PruneSelectionResult,
  PruningMethod,
  ScheduledTargetContext,
  ScheduledTargetResult,
} from '../network.types';
import {
  DEFAULT_PRUNE_FREQUENCY,
  MAX_PROGRESS_FRACTION,
  MIN_PROGRESS_FRACTION,
  MIN_REMAINING_CONNECTION_COUNT,
  PRUNING_METHOD_MAGNITUDE,
  PRUNING_METHOD_SNIP,
  type ActivePruningConfig,
} from './network.prune.utils.types';

/**
 * Read the active pruning schedule from network internals.
 * @param currentNetwork - Network instance to inspect.
 * @returns Pruning configuration when enabled; otherwise undefined.
 */
export function getPruningConfig(
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
export function shouldRunScheduledPrune(
  currentIteration: number,
  currentPruningConfig: ActivePruningConfig,
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
 * Read the scheduled-pruning baseline connection count.
 * @param currentNetwork - Network instance to inspect.
 * @returns Baseline count when captured; otherwise undefined.
 */
export function getInitialConnectionBaseline(
  currentNetwork: Network,
): number | undefined {
  return (currentNetwork as unknown as NetworkPruningProps)
    ._initialConnectionCount;
}

/**
 * Build current scheduled pruning targets from schedule context.
 * @param context - Inputs required to compute desired remaining connections.
 * @param currentConnectionCount - Current number of network connections.
 * @returns Desired remaining connections and current excess.
 */
export function buildScheduledTarget(
  context: ScheduledTargetContext,
  currentConnectionCount: number,
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
    currentConnectionCount - desiredRemainingConnections;

  return {
    desiredRemainingConnections,
    excessConnectionCount,
  };
}

/**
 * Build a connection removal selection from current ranking context.
 * @param context - Inputs for ranking and slicing removable connections.
 * @returns Connections selected for pruning.
 */
export function buildPruneSelection(
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
 * Disconnect all selected connections from the network.
 * @param currentNetwork - Network to mutate.
 * @param connectionsToDisconnect - Connections to remove.
 * @returns Nothing.
 */
export function disconnectConnections(
  currentNetwork: Network,
  connectionsToDisconnect: Connection[],
): void {
  // Step 1: Remove each selected edge.
  connectionsToDisconnect.forEach((connection) => {
    currentNetwork.disconnect(connection.from, connection.to);
  });

  // Step 2: Schedule a deferred activation-pool trim after large prune bursts.
  activationArrayPool.scheduleCompactionAfterLargePrune(
    connectionsToDisconnect.length,
  );
}

/**
 * Normalize optional pruning method to a concrete value.
 * @param method - Optional configured pruning method.
 * @returns Concrete pruning method.
 */
export function resolvePruningMethod(
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
export function markPruneIteration(
  currentPruningConfig: ActivePruningConfig,
  currentIteration: number,
): void {
  currentPruningConfig.lastPruneIter = currentIteration;
}

/**
 * Mark topology cache as dirty after structural updates.
 * @param currentNetwork - Network with modified connectivity.
 * @returns Nothing.
 */
export function markTopologyDirty(currentNetwork: Network): void {
  (currentNetwork as unknown as NetworkPruningProps)._topoDirty = true;
}

/**
 * Check whether an iteration is outside the pruning window.
 * @param currentIteration - Iteration to evaluate.
 * @param currentPruningConfig - Active pruning schedule.
 * @returns True when the iteration is out of range.
 */
function isOutsidePruningWindow(
  currentIteration: number,
  currentPruningConfig: ActivePruningConfig,
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
  currentPruningConfig: ActivePruningConfig,
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
  currentPruningConfig: ActivePruningConfig,
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
  const rawProgressFraction = (currentIteration - scheduleStart) / scheduleSpan;

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
