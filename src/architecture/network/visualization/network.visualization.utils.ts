/**
 * Pure helper utilities for deterministic network visualization export.
 *
 * These helpers are intentionally stateless and accept only primitive inputs so
 * they can be tested without constructing a full `Network` instance.
 */

import type Node from '../../node';
import type Connection from '../../connection';
import type {
  VisualizationEdgeV1,
  VisualizationNodeV1,
} from './network.visualization.types';

/**
 * Infers the human-readable activation name from a squash function.
 *
 * Falls back to `'unknown'` when the function reference does not expose a name.
 *
 * @param squash - Activation function attached to the node.
 * @returns Name string, e.g. `'LOGISTIC'` or `'TANH'`.
 */
export function resolveActivationName(
  squash: ((x: number, derivate?: boolean) => number) & { name?: string },
): string {
  return squash?.name ?? 'unknown';
}

/**
 * Infers the edge kind for a directed connection.
 *
 * - Returns `'self'` when source and target are the same node identity.
 * - Returns `'recurrent'` when the source node's stable position in the
 *   sorted node list comes **after** the target's position (backward edge).
 * - Returns `'forward'` otherwise.
 *
 * @param from - Source node.
 * @param to - Target node.
 * @param nodePositionByGeneId - Map from geneId to sorted position index.
 * @returns Inferred connection kind.
 */
export function inferEdgeKind(
  from: Node,
  to: Node,
  nodePositionByGeneId: Map<number, number>,
): 'forward' | 'recurrent' | 'self' {
  // Step 1: Self-connections are detected by node identity.
  if (from === to) {
    return 'self';
  }

  // Step 2: Backward edges are recurrent.
  const fromPosition = nodePositionByGeneId.get(from.geneId) ?? 0;
  const toPosition = nodePositionByGeneId.get(to.geneId) ?? 0;

  return fromPosition > toPosition ? 'recurrent' : 'forward';
}

/**
 * Builds a lookup map from stable gene id to sorted positional index.
 *
 * The sorted order is by gene id ascending, which is the same order used in
 * the exported `nodes` array. This map is used by {@link inferEdgeKind} to
 * detect backward (recurrent) edges without inspecting the full node list on
 * each call.
 *
 * @param sortedNodes - Node array already sorted by geneId ascending.
 * @returns Map from geneId to zero-based position.
 */
export function buildNodePositionMap(sortedNodes: Node[]): Map<number, number> {
  return new Map<number, number>(
    sortedNodes.map((node, position) => [node.geneId, position]),
  );
}

/**
 * Convert a runtime node into a deterministic visualization node descriptor.
 *
 * @param node - Source node instance.
 * @param role - Resolved semantic role for this node.
 * @param includeBias - Whether to include the bias field.
 * @returns Immutable node descriptor for the visualization schema.
 */
export function nodeToVisualizationDescriptor(
  node: Node,
  role: 'input' | 'output' | 'hidden',
  includeBias: boolean,
): VisualizationNodeV1 {
  const descriptor: VisualizationNodeV1 = {
    id: node.geneId,
    role,
    activation: resolveActivationName(node.squash),
  };

  // Step 1: Include optional label when set.
  if (node.label != null) {
    descriptor.label = node.label;
  }

  // Step 2: Include bias when requested.
  if (includeBias) {
    descriptor.bias = node.bias;
  }

  return descriptor;
}

/**
 * Convert a runtime connection into the exported edge schema with deterministic edge-kind inference.
 *
 * @param connection - Source connection instance.
 * @param nodePositionByGeneId - Sorted position lookup built by {@link buildNodePositionMap}.
 * @param includeWeight - Whether the emitted descriptor should preserve the runtime weight.
 * @returns Immutable edge descriptor for the visualization schema.
 */
export function connectionToVisualizationDescriptor(
  connection: Connection,
  nodePositionByGeneId: Map<number, number>,
  includeWeight: boolean,
): VisualizationEdgeV1 {
  const kind = inferEdgeKind(
    connection.from,
    connection.to,
    nodePositionByGeneId,
  );

  const descriptor: VisualizationEdgeV1 = {
    from: connection.from.geneId,
    to: connection.to.geneId,
    weight: includeWeight ? connection.weight : 0,
    enabled: connection.enabled,
    kind,
  };

  return descriptor;
}

/**
 * Resolve node role by explicit stable-id membership, defaulting to hidden for all remaining nodes.
 *
 * @param node - Node to classify.
 * @param inputIdSet - Set of stable gene ids for input-role nodes.
 * @param outputIdSet - Set of stable gene ids for output-role nodes.
 * @returns `'input'`, `'output'`, or `'hidden'`.
 */
export function resolveNodeRole(
  node: Node,
  inputIdSet: Set<number>,
  outputIdSet: Set<number>,
): 'input' | 'output' | 'hidden' {
  if (inputIdSet.has(node.geneId)) {
    return 'input';
  }
  if (outputIdSet.has(node.geneId)) {
    return 'output';
  }
  return 'hidden';
}

/**
 * Collects and sorts a connection list for deterministic export.
 *
 * Primary sort key: `from` gene id ascending.
 * Secondary sort key: `to` gene id ascending.
 *
 * @param connections - Flat connection list (regular + self-connections merged by caller).
 * @param includeDisabled - Whether to retain disabled connections.
 * @returns Sorted, optionally filtered connection array.
 */
export function collectSortedConnections(
  connections: Connection[],
  includeDisabled: boolean,
): Connection[] {
  const filtered = includeDisabled
    ? connections
    : connections.filter((conn) => conn.enabled);

  return filtered.toSorted((firstConn, secondConn) => {
    const fromDelta = firstConn.from.geneId - secondConn.from.geneId;
    if (fromDelta !== 0) {
      return fromDelta;
    }
    return firstConn.to.geneId - secondConn.to.geneId;
  });
}
