import type Network from '../../network/network';
import type Node from '../../node';
import type Connection from '../../connection';
import type {
  NodeConnectionSnapshotContext,
  NodeRemovalContext,
} from './network.remove.utils.types';

/**
 * Creates immutable snapshots of node adjacency lists before mutation.
 *
 * @param removalContext - Immutable removal context.
 * @returns Snapshot context.
 */
export function createNodeConnectionSnapshot(
  removalContext: NodeRemovalContext,
): NodeConnectionSnapshotContext {
  return {
    inboundConnections: cloneInboundConnections(removalContext.targetNode),
    outboundConnections: cloneOutboundConnections(removalContext.targetNode),
    selfConnectionCount: countSelfConnections(removalContext.targetNode),
  };
}

/**
 * Disconnects all inbound, outbound, and self-loop edges for removed node.
 *
 * @param removalContext - Immutable removal context.
 * @param snapshotContext - Immutable adjacency snapshot.
 * @returns Nothing.
 */
export function disconnectAllNodeConnections(
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
