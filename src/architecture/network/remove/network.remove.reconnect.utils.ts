import type Network from '../../network/network';
import type Connection from '../../connection';
import type {
  NodeConnectionSnapshotContext,
  NodeRemovalContext,
  ReconnectEndpointPairContext,
} from './network.remove.utils.types';

/**
 * Reconnects paths from former inbound sources to former outbound targets.
 *
 * @param removalContext - Immutable removal context.
 * @param snapshotContext - Immutable adjacency snapshot.
 * @returns Nothing.
 */
export function reconnectBridgedPaths(
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
