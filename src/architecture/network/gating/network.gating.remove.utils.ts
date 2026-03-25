import type Network from '../../network/network';
import Node from '../../node';
import Connection from '../../connection';
import mutation from '../../../methods/mutation/mutation';
import type {
  BridgingConnectionList,
  ConnectedNodeList,
  MutableNetworkGatingProps,
  NodeRemovalMutationConfig,
  PreservedGaters,
} from './network.gating.utils.types';

/**
 * Ensure a node can be removed and return its index in the network node list.
 *
 * @param network - Network containing the node.
 * @param node - Node to validate.
 * @returns Index of the node in the network node list.
 * @throws If the node is an input/output node or absent from the network.
 */
export function assertNodeRemovableAndGetIndex(
  network: Network,
  node: Node,
): number {
  if (node.type === 'input' || node.type === 'output') {
    throw new Error('Cannot remove input or output node from the network.');
  }

  const nodeIndex = network.nodes.indexOf(node);
  if (nodeIndex === -1) {
    throw new Error('Node not found in the network for removal.');
  }

  return nodeIndex;
}

/**
 * Resolve the active SUB_NODE mutation configuration shape.
 *
 * @returns Normalized SUB_NODE config when available; otherwise undefined.
 */
export function resolveSubNodeMutationConfig(): NodeRemovalMutationConfig {
  const subNodeMutationConfig = Array.isArray(mutation.SUB_NODE)
    ? mutation.SUB_NODE[0]
    : mutation.SUB_NODE;
  return subNodeMutationConfig as NodeRemovalMutationConfig;
}

/**
 * Disconnect a node self-loop before broader edge rewiring.
 *
 * @param network - Network being updated.
 * @param node - Node whose self-loop should be removed.
 * @returns Nothing.
 */
export function disconnectNodeSelfLoop(network: Network, node: Node): void {
  network.disconnect(node, node);
}

/**
 * Disconnect all inbound connections for a node while collecting predecessors.
 *
 * @param network - Network being updated.
 * @param node - Node being removed.
 * @param preservedGaters - Mutable collection for gaters that should be reassigned.
 * @param subNodeConfig - Current SUB_NODE mutation settings.
 * @returns Predecessor nodes that previously projected into the removed node.
 */
export function disconnectInboundConnections(
  network: Network,
  node: Node,
  preservedGaters: PreservedGaters,
  subNodeConfig: NodeRemovalMutationConfig,
): ConnectedNodeList {
  const predecessorNodes: ConnectedNodeList = [];
  const inboundConnections =
    node.connections.in.toReversed?.() ?? [...node.connections.in].reverse();

  for (const inboundConnection of inboundConnections) {
    preserveGaterForReassignment(
      inboundConnection,
      node,
      preservedGaters,
      subNodeConfig,
    );
    predecessorNodes.push(inboundConnection.from);
    network.disconnect(inboundConnection.from, node);
  }

  return predecessorNodes;
}

/**
 * Disconnect all outbound connections for a node while collecting successors.
 *
 * @param network - Network being updated.
 * @param node - Node being removed.
 * @param preservedGaters - Mutable collection for gaters that should be reassigned.
 * @param subNodeConfig - Current SUB_NODE mutation settings.
 * @returns Successor nodes that were previously targeted by the removed node.
 */
export function disconnectOutboundConnections(
  network: Network,
  node: Node,
  preservedGaters: PreservedGaters,
  subNodeConfig: NodeRemovalMutationConfig,
): ConnectedNodeList {
  const successorNodes: ConnectedNodeList = [];
  const outboundConnections =
    node.connections.out.toReversed?.() ?? [...node.connections.out].reverse();

  for (const outboundConnection of outboundConnections) {
    preserveGaterForReassignment(
      outboundConnection,
      node,
      preservedGaters,
      subNodeConfig,
    );
    successorNodes.push(outboundConnection.to);
    network.disconnect(node, outboundConnection.to);
  }

  return successorNodes;
}

/**
 * Create bridging connections from each predecessor to each successor when valid.
 *
 * @param network - Network where bridge connections are created.
 * @param predecessorNodes - Source nodes collected from inbound edges.
 * @param successorNodes - Target nodes collected from outbound edges.
 * @returns Newly created bridge connections.
 */
export function createBridgingConnections(
  network: Network,
  predecessorNodes: ConnectedNodeList,
  successorNodes: ConnectedNodeList,
): BridgingConnectionList {
  const bridgingConnections: BridgingConnectionList = [];

  for (const predecessorNode of predecessorNodes) {
    for (const successorNode of successorNodes) {
      if (!shouldCreateBridgeConnection(predecessorNode, successorNode)) {
        continue;
      }

      const createdConnections = network.connect(
        predecessorNode,
        successorNode,
      );
      const createdConnection = createdConnections.at(0);
      if (createdConnection) {
        bridgingConnections.push(createdConnection);
      }
    }
  }

  return bridgingConnections;
}

/**
 * Reattach preserved gaters to randomly selected newly-created bridge connections.
 *
 * @param network - Network performing reassignment.
 * @param preservedGaters - Gaters retained during node removal.
 * @param bridgingConnections - Available bridge connections for reassignment.
 * @returns Nothing.
 */
export function reassignPreservedGaters(
  network: Network,
  preservedGaters: PreservedGaters,
  bridgingConnections: BridgingConnectionList,
): void {
  for (const gaterNode of preservedGaters) {
    if (bridgingConnections.length === 0) {
      return;
    }

    const connectionIndex = selectRandomIndex(bridgingConnections.length);
    const connectionToGate = bridgingConnections.at(connectionIndex);
    if (!connectionToGate) {
      continue;
    }

    network.gate(gaterNode, connectionToGate);
    bridgingConnections.splice(connectionIndex, 1);
  }
}

/**
 * Ungate all connections that are currently gated by the removed node.
 *
 * @param network - Network performing ungate operations.
 * @param node - Node whose gated connections should be released.
 * @returns Nothing.
 */
export function ungateConnectionsGatedByNode(
  network: Network,
  node: Node,
): void {
  const gatedConnections =
    node.connections.gated.toReversed?.() ??
    [...node.connections.gated].reverse();

  for (const gatedConnection of gatedConnections) {
    network.ungate(gatedConnection);
  }
}

/**
 * Remove a node from the network list and mark node indexing as dirty.
 *
 * @param network - Network being mutated.
 * @param nodeIndex - Index of the node to remove.
 * @returns Nothing.
 */
export function removeNodeAtIndex(network: Network, nodeIndex: number): void {
  network.nodes.splice(nodeIndex, 1);
  (network as unknown as MutableNetworkGatingProps)._nodeIndexDirty = true;
}

/**
 * Preserve a gater for later reassignment when gate retention is enabled.
 *
 * @param connection - Connection being detached.
 * @param removedNode - Node currently being removed.
 * @param preservedGaters - Mutable list of gaters to keep.
 * @param subNodeConfig - Current SUB_NODE mutation settings.
 * @returns Nothing.
 */
function preserveGaterForReassignment(
  connection: Connection,
  removedNode: Node,
  preservedGaters: PreservedGaters,
  subNodeConfig: NodeRemovalMutationConfig,
): void {
  if (!subNodeConfig?.keep_gates || !connection.gater) {
    return;
  }

  if (connection.gater !== removedNode) {
    preservedGaters.push(connection.gater);
  }
}

/**
 * Decide whether a predecessor-successor pair should receive a bridge connection.
 *
 * @param predecessorNode - Candidate source node.
 * @param successorNode - Candidate target node.
 * @returns True when the pair is distinct and no projection already exists.
 */
function shouldCreateBridgeConnection(
  predecessorNode: Node,
  successorNode: Node,
): boolean {
  if (predecessorNode === successorNode) {
    return false;
  }

  return !predecessorNode.isProjectingTo(successorNode);
}

/**
 * Select a uniformly random integer index in the range [0, length).
 *
 * @param length - Upper bound (exclusive).
 * @returns Random zero-based index.
 */
function selectRandomIndex(length: number): number {
  return Math.floor(Math.random() * length);
}
