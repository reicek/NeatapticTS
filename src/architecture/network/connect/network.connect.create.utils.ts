import type Network from '../../network/network';
import Node from '../../node';
import Connection from '../../connection';
import type { NetworkInternals } from './network.connect.utils.types';

/**
 * Determine whether an edge must be rejected to preserve acyclic ordering.
 *
 * @param network - Network instance owning node ordering.
 * @param internalState - Runtime network internals used by connection pipeline.
 * @param sourceNode - Candidate source node.
 * @param targetNode - Candidate target node.
 * @returns True when edge should be rejected.
 */
export function shouldRejectConnectionForAcyclicMode(
  network: Network,
  internalState: NetworkInternals,
  sourceNode: Node,
  targetNode: Node,
): boolean {
  if (!internalState._enforceAcyclic) return false;
  return network.nodes.indexOf(sourceNode) > network.nodes.indexOf(targetNode);
}

/**
 * Build one or more low-level connection objects from source node to target node.
 *
 * @param sourceNode - Source node.
 * @param targetNode - Target node.
 * @param initialWeight - Optional explicit initial weight.
 * @returns Created low-level connection objects.
 */
export function createConnectionsFromSourceNode(
  sourceNode: Node,
  targetNode: Node,
  initialWeight?: number,
): Connection[] {
  return sourceNode.connect(targetNode, initialWeight);
}

/**
 * Register created connections in either normal-connection or self-connection storage.
 *
 * @param network - Network instance owning connection collections.
 * @param internalState - Runtime network internals used by connection pipeline.
 * @param sourceNode - Source node used during connection creation.
 * @param targetNode - Target node used during connection creation.
 * @param createdConnections - Created low-level connection objects.
 * @returns Nothing.
 */
export function registerCreatedConnections(
  network: Network,
  internalState: NetworkInternals,
  sourceNode: Node,
  targetNode: Node,
  createdConnections: Connection[],
): void {
  const isSelfConnection = sourceNode === targetNode;

  createdConnections.forEach((createdConnection) => {
    registerSingleCreatedConnection(
      network,
      internalState,
      isSelfConnection,
      createdConnection,
    );
  });
}

/**
 * Mark topology and slab caches dirty when connection creation occurred.
 *
 * @param internalState - Runtime network internals used by connection pipeline.
 * @param createdConnectionCount - Number of created low-level connections.
 * @returns Nothing.
 */
export function markConnectionCachesDirtyWhenNeeded(
  internalState: NetworkInternals,
  createdConnectionCount: number,
): void {
  if (!createdConnectionCount) return;
  internalState._topoDirty = true;
  internalState._slabDirty = true;
}

/**
 * Register one created connection in the appropriate collection.
 *
 * @param network - Network instance owning connection collections.
 * @param internalState - Runtime network internals used by connection pipeline.
 * @param isSelfConnection - Whether source and target nodes are the same.
 * @param createdConnection - Created low-level connection object.
 * @returns Nothing.
 */
function registerSingleCreatedConnection(
  network: Network,
  internalState: NetworkInternals,
  isSelfConnection: boolean,
  createdConnection: Connection,
): void {
  if (!isSelfConnection) {
    network.connections.push(createdConnection);
    return;
  }

  if (internalState._enforceAcyclic) return;
  network.selfconns.push(createdConnection);
}
