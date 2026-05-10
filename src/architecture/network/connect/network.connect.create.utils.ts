import type Network from '../../network/network';
import Node from '../../node';
import Connection from '../../connection';
import type { NetworkInternals } from './network.connect.utils.types';

type CreatedConnectionBatch = {
  sourceNode: Node;
  targetNode: Node;
  createdConnections: readonly Connection[];
};

type ConnectionStoragePlan = {
  totalCreatedConnectionCount: number;
  standardConnectionCount: number;
  selfConnectionCount: number;
};

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
 * @param randomValue - Network-owned RNG used when the caller did not provide a weight.
 * @returns Created low-level connection objects.
 */
export function createConnectionsFromSourceNode(
  sourceNode: Node,
  targetNode: Node,
  initialWeight?: number,
  randomValue?: () => number,
): Connection[] {
  const resolvedWeight =
    initialWeight ?? (randomValue ? randomValue() * 0.2 - 0.1 : undefined);

  return sourceNode.connect(targetNode, resolvedWeight);
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
 * Register many created connection groups while reserving network-level storage once.
 *
 * This preserves the same registration semantics as repeated
 * {@link registerCreatedConnections} calls, but it grows the top-level
 * `connections` and `selfconns` arrays one time for the whole batch.
 *
 * @param network - Network instance owning connection collections.
 * @param internalState - Runtime network internals used by connection pipeline.
 * @param createdConnectionBatches - Ordered connection groups produced from one batch request shelf.
 * @returns Flattened created connections in request order.
 */
export function registerCreatedConnectionBatches(
  network: Network,
  internalState: NetworkInternals,
  createdConnectionBatches: readonly CreatedConnectionBatch[],
): Connection[] {
  const connectionStoragePlan = resolveConnectionStoragePlan(
    createdConnectionBatches,
    internalState,
  );
  const orderedCreatedConnections = new Array<Connection>(
    connectionStoragePlan.totalCreatedConnectionCount,
  );
  const standardConnectionStartIndex = network.connections.length;
  const selfConnectionStartIndex = network.selfconns.length;

  network.connections.length =
    standardConnectionStartIndex + connectionStoragePlan.standardConnectionCount;
  network.selfconns.length =
    selfConnectionStartIndex + connectionStoragePlan.selfConnectionCount;

  let nextOrderedConnectionIndex = 0;
  let nextStandardConnectionIndex = standardConnectionStartIndex;
  let nextSelfConnectionIndex = selfConnectionStartIndex;

  for (const createdConnectionBatch of createdConnectionBatches) {
    const isSelfConnection =
      createdConnectionBatch.sourceNode === createdConnectionBatch.targetNode;

    for (const createdConnection of createdConnectionBatch.createdConnections) {
      orderedCreatedConnections[nextOrderedConnectionIndex] = createdConnection;
      nextOrderedConnectionIndex += 1;

      if (!isSelfConnection) {
        network.connections[nextStandardConnectionIndex] = createdConnection;
        nextStandardConnectionIndex += 1;
        continue;
      }

      if (internalState._enforceAcyclic) {
        continue;
      }

      network.selfconns[nextSelfConnectionIndex] = createdConnection;
      nextSelfConnectionIndex += 1;
    }
  }

  return orderedCreatedConnections;
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

/**
 * Resolve how much top-level connection storage one batch must reserve.
 *
 * @param createdConnectionBatches - Ordered connection groups produced from one batch request shelf.
 * @param internalState - Runtime network internals used by connection pipeline.
 * @returns Planned storage counts for flattened, standard, and self connections.
 */
function resolveConnectionStoragePlan(
  createdConnectionBatches: readonly CreatedConnectionBatch[],
  internalState: NetworkInternals,
): ConnectionStoragePlan {
  let totalCreatedConnectionCount = 0;
  let standardConnectionCount = 0;
  let selfConnectionCount = 0;

  for (const createdConnectionBatch of createdConnectionBatches) {
    const createdConnectionCount =
      createdConnectionBatch.createdConnections.length;

    totalCreatedConnectionCount += createdConnectionCount;

    if (createdConnectionBatch.sourceNode !== createdConnectionBatch.targetNode) {
      standardConnectionCount += createdConnectionCount;
      continue;
    }

    if (!internalState._enforceAcyclic) {
      selfConnectionCount += createdConnectionCount;
    }
  }

  return {
    totalCreatedConnectionCount,
    standardConnectionCount,
    selfConnectionCount,
  };
}
