import type {
  TopologyBuildContext,
  TopologyNetwork,
  TopologyNetworkProps,
} from './network.topology.utils.types';
import {
  IN_DEGREE_DECREMENT,
  ZERO_COUNT,
} from './network.topology.utils.types';

/**
 * Cast network to internal topology props view.
 *
 * @param network Network instance.
 * @returns Internal topology props view.
 */
export function asTopologyProps(
  network: TopologyNetwork,
): TopologyNetworkProps {
  return network as unknown as TopologyNetworkProps;
}

/**
 * Determine whether topological order should be bypassed.
 *
 * @param internalTopologyProps Internal topology props view.
 * @returns True when acyclic mode is disabled.
 */
export function shouldUseRawNodeOrder(
  internalTopologyProps: TopologyNetworkProps,
): boolean {
  return !internalTopologyProps._enforceAcyclic;
}

/**
 * Clear cached topological order state.
 *
 * @param internalTopologyProps Internal topology props view.
 * @returns Void.
 */
export function clearCachedTopoOrder(
  internalTopologyProps: TopologyNetworkProps,
): void {
  internalTopologyProps._topoOrder = null;
  internalTopologyProps._topoDirty = false;
}

/**
 * Create mutable build context for Kahn traversal.
 *
 * @param network Network instance.
 * @param internalTopologyProps Internal topology props view.
 * @returns Initialized build context.
 */
export function createTopologyBuildContext(
  network: TopologyNetwork,
  internalTopologyProps: TopologyNetworkProps,
): TopologyBuildContext {
  return {
    network,
    internalTopologyProps,
    inDegreeByNode: new Map(),
    processingQueue: [],
    topoOrder: [],
  };
}

/**
 * Initialize all nodes with zero in-degree.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function initializeAllNodeInDegreeCounts(
  buildContext: TopologyBuildContext,
): void {
  for (const node of buildContext.network.nodes) {
    buildContext.inDegreeByNode.set(node, ZERO_COUNT);
  }
}

/**
 * Apply in-degree increments from non-self connections.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function applyIncomingEdgeCounts(
  buildContext: TopologyBuildContext,
): void {
  for (const connection of buildContext.network.connections) {
    if (isSelfConnection(connection.from, connection.to)) {
      continue;
    }

    incrementNodeInDegree(buildContext, connection.to);
  }
}

/**
 * Finalize cached order, falling back to raw node order on cycle detection.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function finalizeTopoOrder(buildContext: TopologyBuildContext): void {
  buildContext.internalTopologyProps._topoOrder =
    resolveFinalOrder(buildContext);
  buildContext.internalTopologyProps._topoDirty = false;
}

/**
 * Resolve final topological order with cycle fallback.
 *
 * @param buildContext Mutable build context.
 * @returns Fully valid topological order or raw node order fallback.
 */
function resolveFinalOrder(buildContext: TopologyBuildContext) {
  const isCompleteOrder =
    buildContext.topoOrder.length === buildContext.network.nodes.length;

  if (isCompleteOrder) {
    return buildContext.topoOrder;
  }

  return [...buildContext.network.nodes];
}

/**
 * Test whether a connection is a self-loop.
 *
 * @param from Source node.
 * @param to Target node.
 * @returns True when source and target are the same node.
 */
function isSelfConnection(
  from: TopologyBuildContext['network']['nodes'][number],
  to: TopologyBuildContext['network']['nodes'][number],
): boolean {
  return from === to;
}

/**
 * Increment in-degree for a node in the tally map.
 *
 * @param buildContext Mutable build context.
 * @param node Target node.
 * @returns Void.
 */
function incrementNodeInDegree(
  buildContext: TopologyBuildContext,
  node: TopologyBuildContext['network']['nodes'][number],
): void {
  const currentInDegree = buildContext.inDegreeByNode.get(node) ?? ZERO_COUNT;
  buildContext.inDegreeByNode.set(node, currentInDegree + IN_DEGREE_DECREMENT);
}
