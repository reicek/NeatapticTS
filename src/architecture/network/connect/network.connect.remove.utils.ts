import type Network from '../../network';
import Node from '../../node';
import Connection from '../../connection';
import type { NetworkInternals } from './network.connect.utils.types';

/**
 * Select the relevant collection to search for the edge.
 *
 * @param network - Network instance owning connection collections.
 * @param sourceNode - Source node.
 * @param targetNode - Target node.
 * @returns Candidate connection collection.
 */
export function selectConnectionCollection(
  network: Network,
  sourceNode: Node,
  targetNode: Node,
): Connection[] {
  return sourceNode === targetNode ? network.selfconns : network.connections;
}

/**
 * Remove first connection that matches source and target nodes.
 *
 * @param network - Network instance used for ungating.
 * @param candidateConnections - Candidate collection to search.
 * @param sourceNode - Source node.
 * @param targetNode - Target node.
 * @returns Nothing.
 */
export function removeFirstMatchingConnection(
  network: Network,
  candidateConnections: Connection[],
  sourceNode: Node,
  targetNode: Node,
): void {
  const targetConnectionIndex = findConnectionIndex(
    candidateConnections,
    sourceNode,
    targetNode,
  );

  if (targetConnectionIndex < 0) return;
  removeConnectionAtIndex(network, candidateConnections, targetConnectionIndex);
}

/**
 * Delegate per-node disconnect cleanup.
 *
 * @param sourceNode - Source node.
 * @param targetNode - Target node.
 * @returns Nothing.
 */
export function disconnectNodes(sourceNode: Node, targetNode: Node): void {
  sourceNode.disconnect(targetNode);
}

/**
 * Mark topology/slab caches dirty after structural mutation.
 *
 * @param internalState - Runtime network internals used by connection pipeline.
 * @returns Nothing.
 */
export function markStructureCachesDirty(
  internalState: NetworkInternals,
): void {
  internalState._topoDirty = true;
  internalState._slabDirty = true;
}

/**
 * Find index of the first connection matching source and target nodes.
 *
 * @param candidateConnections - Candidate collection to search.
 * @param sourceNode - Source node.
 * @param targetNode - Target node.
 * @returns Matching index or -1 when no edge is found.
 */
function findConnectionIndex(
  candidateConnections: Connection[],
  sourceNode: Node,
  targetNode: Node,
): number {
  return candidateConnections.findIndex(
    (candidateConnection) =>
      candidateConnection.from === sourceNode &&
      candidateConnection.to === targetNode,
  );
}

/**
 * Remove one connection by index, ungating first if required.
 *
 * @param network - Network instance used for ungating.
 * @param candidateConnections - Candidate collection containing target index.
 * @param targetConnectionIndex - Index to remove.
 * @returns Nothing.
 */
function removeConnectionAtIndex(
  network: Network,
  candidateConnections: Connection[],
  targetConnectionIndex: number,
): void {
  const targetConnection = candidateConnections[targetConnectionIndex];
  if (targetConnection.gater) network.ungate(targetConnection);
  candidateConnections.splice(targetConnectionIndex, 1);
}
