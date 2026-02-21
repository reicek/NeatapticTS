import type Network from '../../network';
import type Node from '../../node';
import type Connection from '../../connection';
import { releaseNode as _releaseNode } from '../../nodePool';
import { config } from '../../../config';
import type {
  NetworkRemoveProps,
  NodeConnectionSnapshotContext,
  NodeRemovalContext,
  ReconnectEndpointPairContext,
} from '../network.types';

/**
 * Node type literal for input anchors.
 */
const NODE_TYPE_INPUT: Node['type'] = 'input';

/**
 * Node type literal for output anchors.
 */
const NODE_TYPE_OUTPUT: Node['type'] = 'output';

/**
 * Error emitted when target node is not part of the network.
 */
const ERROR_NODE_NOT_IN_NETWORK = 'Node not in network';

/**
 * Error emitted when trying to remove structural anchor nodes.
 */
const ERROR_CANNOT_REMOVE_ANCHOR_NODE =
  'Cannot remove input or output node from the network.';

/**
 * Sentinel index used when node is not found.
 */
const NODE_NOT_FOUND_INDEX = -1;

/**
 * Index for selecting first spliced node.
 */
const FIRST_REMOVED_NODE_INDEX = 0;

/**
 * Node removal utilities.
 *
 * This module provides a focused implementation for removing a single hidden node from a network
 * while attempting to preserve overall functional connectivity. The removal procedure mirrors the
 * legacy Neataptic logic but augments it with clearer documentation and explicit invariants.
 *
 * High‑level algorithm (removeNode):
 *  1. Guard: ensure the node exists and is not an input or output (those are structural anchors).
 *  2. Ungate: detach any connections gated BY the node (we don't currently reassign gater roles).
 *  3. Snapshot inbound / outbound connections (before mutation of adjacency lists).
 *  4. Disconnect all inbound, outbound, and self connections.
 *  5. Physically remove the node from the network's node array.
 *  6. Simple path repair heuristic: for every former inbound source and outbound target, add a
 *     direct connection if (a) both endpoints still exist, (b) they are distinct, and (c) no
 *     direct connection already exists. This keeps forward information flow possibilities.
 *  7. Mark topology / caches dirty so that subsequent activation / ordering passes rebuild state.
 *
 * Notes / Limitations:
 *  - We do NOT attempt to clone weights or distribute the removed node's function across new
 *    connections (more sophisticated strategies could average or compose weights).
 *  - Gating effects involving the removed node as a gater are dropped; downstream behavior may
 *    change—callers relying heavily on gating may want a custom remap strategy.
 *  - Self connections are simply removed; no attempt is made to emulate recursion via alternative
 *    structures.
 */

/**
 * Remove a hidden node from the network while minimally repairing connectivity.
 *
 * @param this Network instance (bound implicitly via method-style call).
 * @param node The node object to remove (must be of type 'hidden').
 * @throws If the node is not present or is an input / output node.
 *
 * Side Effects:
 *  - Mutates network.nodes, network.connections (via disconnect/connect calls), and network.gates.
 *  - Marks internal dirty flags so that future activation / ordering passes recompute derived state.
 */
export function removeNode(this: Network, node: Node) {
  const removalContext = createValidatedNodeRemovalContext(this, node);

  // Step 1: Remove gate assignments owned by the removed node.
  detachGatesOwnedByNode(removalContext);

  // Step 2: Snapshot connection neighborhoods before mutating graph edges.
  const snapshotContext = createNodeConnectionSnapshot(removalContext);

  // Step 3: Disconnect all edges touching the removed node.
  disconnectAllNodeConnections(removalContext, snapshotContext);

  // Step 4: Remove the node from storage and release pooled instance when enabled.
  removeNodeFromNetworkStorage(removalContext);

  // Step 5: Reconnect source-to-target paths that were bridged through the removed node.
  reconnectBridgedPaths(removalContext, snapshotContext);

  // Step 6: Mark all cached derived graph structures dirty.
  markNetworkRemovalDirtyFlags(removalContext.internalNetwork);
}

/**
 * Creates validated immutable context for a node-removal operation.
 *
 * @param network - Target network.
 * @param targetNode - Node requested for removal.
 * @returns Validated removal context.
 */
function createValidatedNodeRemovalContext(
  network: Network,
  targetNode: Node,
): NodeRemovalContext {
  const targetNodeIndex = resolveNodeIndexOrThrow(network, targetNode);
  ensureNodeIsNotStructuralAnchor(targetNode);

  return {
    network,
    internalNetwork: network as unknown as NetworkRemoveProps,
    targetNode,
    targetNodeIndex,
  };
}

/**
 * Resolves node index and throws when missing.
 *
 * @param network - Target network.
 * @param targetNode - Node being removed.
 * @returns Node index inside network list.
 */
function resolveNodeIndexOrThrow(network: Network, targetNode: Node): number {
  const targetNodeIndex = network.nodes.indexOf(targetNode);
  if (targetNodeIndex === NODE_NOT_FOUND_INDEX) {
    throw new Error(ERROR_NODE_NOT_IN_NETWORK);
  }
  return targetNodeIndex;
}

/**
 * Ensures removal target is not an input/output anchor node.
 *
 * @param targetNode - Node under validation.
 * @returns Nothing.
 */
function ensureNodeIsNotStructuralAnchor(targetNode: Node): void {
  if (isStructuralAnchorNode(targetNode)) {
    throw new Error(ERROR_CANNOT_REMOVE_ANCHOR_NODE);
  }
}

/**
 * Checks whether node is an input/output structural anchor.
 *
 * @param targetNode - Node under evaluation.
 * @returns True when node is an anchor.
 */
function isStructuralAnchorNode(targetNode: Node): boolean {
  return (
    targetNode.type === NODE_TYPE_INPUT || targetNode.type === NODE_TYPE_OUTPUT
  );
}

/**
 * Removes gate records gated by target node and nulls their gater field.
 *
 * @param removalContext - Immutable removal context.
 * @returns Nothing.
 */
function detachGatesOwnedByNode(removalContext: NodeRemovalContext): void {
  removalContext.network.gates = removalContext.network.gates.filter(
    (candidateConnection) =>
      keepGateConnectionAfterNodeRemoval(
        candidateConnection,
        removalContext.targetNode,
      ),
  );
}

/**
 * Filters one gate connection while clearing removed-node gater ownership.
 *
 * @param candidateConnection - Gate candidate.
 * @param removedNode - Removed node reference.
 * @returns True when gate should remain in list.
 */
function keepGateConnectionAfterNodeRemoval(
  candidateConnection: Connection,
  removedNode: Node,
): boolean {
  if (!isGatedByRemovedNode(candidateConnection, removedNode)) {
    return true;
  }

  clearConnectionGater(candidateConnection);
  return false;
}

/**
 * Checks whether a gate candidate is currently gated by removed node.
 *
 * @param candidateConnection - Gate candidate.
 * @param removedNode - Removed node reference.
 * @returns True when removed node is gater.
 */
function isGatedByRemovedNode(
  candidateConnection: Connection,
  removedNode: Node,
): boolean {
  return candidateConnection.gater === removedNode;
}

/**
 * Clears gater reference so legacy checks treat connection as ungated.
 *
 * @param candidateConnection - Connection to clear.
 * @returns Nothing.
 */
function clearConnectionGater(candidateConnection: Connection): void {
  candidateConnection.gater = null;
}

/**
 * Creates immutable snapshots of node adjacency lists before mutation.
 *
 * @param removalContext - Immutable removal context.
 * @returns Snapshot context.
 */
function createNodeConnectionSnapshot(
  removalContext: NodeRemovalContext,
): NodeConnectionSnapshotContext {
  return {
    inboundConnections: cloneInboundConnections(removalContext.targetNode),
    outboundConnections: cloneOutboundConnections(removalContext.targetNode),
    selfConnectionCount: countSelfConnections(removalContext.targetNode),
  };
}

/**
 * Clones inbound connections for safe traversal after mutation.
 *
 * @param targetNode - Node being removed.
 * @returns Inbound connection snapshot.
 */
function cloneInboundConnections(targetNode: Node): Connection[] {
  return targetNode.connections.in.slice();
}

/**
 * Clones outbound connections for safe traversal after mutation.
 *
 * @param targetNode - Node being removed.
 * @returns Outbound connection snapshot.
 */
function cloneOutboundConnections(targetNode: Node): Connection[] {
  return targetNode.connections.out.slice();
}

/**
 * Counts self-loop connections currently attached to node.
 *
 * @param targetNode - Node being removed.
 * @returns Self-loop count.
 */
function countSelfConnections(targetNode: Node): number {
  return targetNode.connections.self.length;
}

/**
 * Disconnects all inbound, outbound, and self-loop edges for removed node.
 *
 * @param removalContext - Immutable removal context.
 * @param snapshotContext - Immutable adjacency snapshot.
 * @returns Nothing.
 */
function disconnectAllNodeConnections(
  removalContext: NodeRemovalContext,
  snapshotContext: NodeConnectionSnapshotContext,
): void {
  disconnectConnectionGroup(
    removalContext.network,
    snapshotContext.inboundConnections,
  );
  disconnectConnectionGroup(
    removalContext.network,
    snapshotContext.outboundConnections,
  );
  disconnectSelfLoops(
    removalContext.network,
    removalContext.targetNode,
    snapshotContext.selfConnectionCount,
  );
}

/**
 * Disconnects each connection in a single connection list.
 *
 * @param network - Target network.
 * @param connectionsToDisconnect - Connection list.
 * @returns Nothing.
 */
function disconnectConnectionGroup(
  network: Network,
  connectionsToDisconnect: Connection[],
): void {
  connectionsToDisconnect.forEach((candidateConnection) => {
    network.disconnect(candidateConnection.from, candidateConnection.to);
  });
}

/**
 * Disconnects node self-loop connections using deterministic count traversal.
 *
 * @param network - Target network.
 * @param targetNode - Node whose self-loop is removed.
 * @param selfConnectionCount - Number of self-loops to remove.
 * @returns Nothing.
 */
function disconnectSelfLoops(
  network: Network,
  targetNode: Node,
  selfConnectionCount: number,
): void {
  for (
    let selfConnectionIndex = 0;
    selfConnectionIndex < selfConnectionCount;
    selfConnectionIndex++
  ) {
    network.disconnect(targetNode, targetNode);
  }
}

/**
 * Removes node from network storage and conditionally releases it to pool.
 *
 * @param removalContext - Immutable removal context.
 * @returns Nothing.
 */
function removeNodeFromNetworkStorage(
  removalContext: NodeRemovalContext,
): void {
  const removedNode = spliceNodeFromNetwork(removalContext);
  releaseRemovedNodeWhenPoolingEnabled(removedNode);
}

/**
 * Splices node out of network list using validated index.
 *
 * @param removalContext - Immutable removal context.
 * @returns Removed node or undefined.
 */
function spliceNodeFromNetwork(
  removalContext: NodeRemovalContext,
): Node | undefined {
  return removalContext.network.nodes.splice(removalContext.targetNodeIndex, 1)[
    FIRST_REMOVED_NODE_INDEX
  ];
}

/**
 * Releases removed node to object pool when pooling is enabled.
 *
 * @param removedNode - Removed node instance.
 * @returns Nothing.
 */
function releaseRemovedNodeWhenPoolingEnabled(
  removedNode: Node | undefined,
): void {
  if (!config.enableNodePooling || !removedNode) {
    return;
  }
  _releaseNode(removedNode);
}

/**
 * Reconnects paths from former inbound sources to former outbound targets.
 *
 * @param removalContext - Immutable removal context.
 * @param snapshotContext - Immutable adjacency snapshot.
 * @returns Nothing.
 */
function reconnectBridgedPaths(
  removalContext: NodeRemovalContext,
  snapshotContext: NodeConnectionSnapshotContext,
): void {
  const reconnectCandidates = collectReconnectEndpointPairs(snapshotContext);
  reconnectCandidates.forEach((reconnectPair) => {
    connectPairWhenMissing(removalContext.network, reconnectPair);
  });
}

/**
 * Collects all valid source/target reconnect endpoint pairs.
 *
 * @param snapshotContext - Immutable adjacency snapshot.
 * @returns Valid reconnect endpoint pairs.
 */
function collectReconnectEndpointPairs(
  snapshotContext: NodeConnectionSnapshotContext,
): ReconnectEndpointPairContext[] {
  const reconnectCandidates: ReconnectEndpointPairContext[] = [];

  snapshotContext.inboundConnections.forEach((inboundConnection) => {
    snapshotContext.outboundConnections.forEach((outboundConnection) => {
      const reconnectPair = createReconnectEndpointPair(
        inboundConnection,
        outboundConnection,
      );
      if (!reconnectPair) {
        return;
      }
      reconnectCandidates.push(reconnectPair);
    });
  });

  return reconnectCandidates;
}

/**
 * Creates one reconnect endpoint pair when endpoints are valid.
 *
 * @param inboundConnection - Inbound edge from snapshot.
 * @param outboundConnection - Outbound edge from snapshot.
 * @returns Reconnect pair or undefined.
 */
function createReconnectEndpointPair(
  inboundConnection: Connection,
  outboundConnection: Connection,
): ReconnectEndpointPairContext | undefined {
  if (!isReconnectPairValid(inboundConnection, outboundConnection)) {
    return undefined;
  }

  return {
    sourceNode: inboundConnection.from,
    targetNode: outboundConnection.to,
  };
}

/**
 * Validates reconnect pair endpoints.
 *
 * @param inboundConnection - Inbound edge from snapshot.
 * @param outboundConnection - Outbound edge from snapshot.
 * @returns True when reconnect pair should be attempted.
 */
function isReconnectPairValid(
  inboundConnection: Connection,
  outboundConnection: Connection,
): boolean {
  if (!inboundConnection.from || !outboundConnection.to) {
    return false;
  }

  return inboundConnection.from !== outboundConnection.to;
}

/**
 * Connects one endpoint pair only when direct edge does not already exist.
 *
 * @param network - Target network.
 * @param reconnectPair - Source/target pair.
 * @returns Nothing.
 */
function connectPairWhenMissing(
  network: Network,
  reconnectPair: ReconnectEndpointPairContext,
): void {
  if (doesDirectConnectionExist(network, reconnectPair)) {
    return;
  }

  network.connect(reconnectPair.sourceNode, reconnectPair.targetNode);
}

/**
 * Checks whether a direct connection already exists for reconnect pair.
 *
 * @param network - Target network.
 * @param reconnectPair - Source/target pair.
 * @returns True when direct edge already exists.
 */
function doesDirectConnectionExist(
  network: Network,
  reconnectPair: ReconnectEndpointPairContext,
): boolean {
  return network.connections.some(
    (candidateConnection) =>
      candidateConnection.from === reconnectPair.sourceNode &&
      candidateConnection.to === reconnectPair.targetNode,
  );
}

/**
 * Marks all cached removal-sensitive structures as dirty.
 *
 * @param internalNetwork - Internal mutable network props.
 * @returns Nothing.
 */
function markNetworkRemovalDirtyFlags(
  internalNetwork: NetworkRemoveProps,
): void {
  internalNetwork._topoDirty = true;
  internalNetwork._nodeIndexDirty = true;
  internalNetwork._slabDirty = true;
  internalNetwork._adjDirty = true;
}

export default { removeNode };
