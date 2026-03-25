import Connection from '../../connection';
import type Network from '../../network/network';
import type {
  EvolutionaryTargetContext,
  EvolutionaryTargetResult,
  NetworkPruningProps,
  PruneSelectionContext,
  PruneSelectionResult,
  PruningMethod,
} from '../network.types';
import {
  MAX_EVOLUTIONARY_TARGET_SPARSITY,
  MIN_REMAINING_CONNECTION_COUNT,
  PRUNING_METHOD_SNIP,
} from './network.prune.utils.types';

/**
 * Clamp evolutionary target sparsity to safe operational bounds.
 * @param rawTargetSparsity - Requested target sparsity.
 * @returns Normalized target sparsity.
 */
export function normalizeEvolutionaryTargetSparsity(
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
export function getOrCaptureEvolutionaryBaseline(
  currentNetwork: Network,
): number {
  // Step 1: Access internal baseline storage.
  const networkProperties = currentNetwork as unknown as NetworkPruningProps;

  // Step 2: Initialize baseline on first call.
  if (!networkProperties._evoInitialConnCount) {
    networkProperties._evoInitialConnCount = currentNetwork.connections.length;
  }
  return networkProperties._evoInitialConnCount;
}

/**
 * Compute evolutionary pruning target counts.
 * @param context - Inputs for sparsity-to-count conversion.
 * @param currentConnectionCount - Current number of network connections.
 * @returns Desired remaining and excess connection counts.
 */
export function buildEvolutionaryTarget(
  context: EvolutionaryTargetContext,
  currentConnectionCount: number,
): EvolutionaryTargetResult {
  // Step 1: Convert target sparsity to desired retained count.
  const desiredRemainingConnections = Math.max(
    MIN_REMAINING_CONNECTION_COUNT,
    Math.floor(context.baselineConnectionCount * (1 - context.targetSparsity)),
  );

  // Step 2: Derive how many edges must be removed.
  const excessConnectionCount =
    currentConnectionCount - desiredRemainingConnections;
  return { desiredRemainingConnections, excessConnectionCount };
}

/**
 * Build evolutionary pruning connection selection.
 * @param context - Inputs for ranking and slicing.
 * @returns Connections selected for removal.
 */
export function buildEvolutionaryPruneSelection(
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
 * Disconnect selected evolutionary pruning edges.
 * @param currentNetwork - Network to mutate.
 * @param connectionsToDisconnect - Edges to remove.
 * @returns Nothing.
 */
export function disconnectEvolutionaryConnections(
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
export function markEvolutionaryTopologyDirty(currentNetwork: Network): void {
  (currentNetwork as unknown as NetworkPruningProps)._topoDirty = true;
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
function resolveEvolutionaryGradientMagnitude(connection: Connection): number {
  return (
    Math.abs(connection.totalDeltaWeight) ||
    Math.abs(connection.previousDeltaWeight) ||
    0
  );
}
