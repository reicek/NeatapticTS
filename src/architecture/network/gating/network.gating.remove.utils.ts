import type Network from '../../network/network';
import Node from '../../node';
import Connection from '../../connection';
import mutation from '../../../methods/mutation/mutation';
import {
  NetworkGatingRemovalNodeNotFoundError,
  NetworkGatingStructuralAnchorRemovalError,
} from './network.gating.errors';
import type {
  BridgingConnectionList,
  ConnectedNodeList,
  MutableNetworkGatingProps,
  NodeRemovalMutationConfig,
  PreservedGaters,
} from './network.gating.utils.types';

/**
 * Ensure a node is eligible for removal and return its index in the network node list so structural anchors and missing nodes fail fast with explicit diagnostics.
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
    throw new NetworkGatingStructuralAnchorRemovalError(
      'Cannot remove input or output node from the network.',
    );
  }

  const nodeIndex = network.nodes.indexOf(node);
  if (nodeIndex === -1) {
    throw new NetworkGatingRemovalNodeNotFoundError(
      'Node not found in the network for removal.',
    );
  }

  return nodeIndex;
}

/**
 * Resolve the active SUB_NODE mutation configuration shape for node removal.
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
 * Disconnect a node self-loop before broader edge rewiring so self-referential activation state does not survive node-removal mutation flows.
 * This keeps removal semantics consistent with later bridge reconstruction and gater reassignment stages.
 *
 * @param network - Network being updated.
 * @param node - Node whose self-loop should be removed.
 * @returns Nothing.
 */
export function disconnectNodeSelfLoop(network: Network, node: Node): void {
  network.disconnect(node, node);
}

/**
 * Disconnect all inbound connections for a node while collecting predecessor nodes so bridge-connection reconstruction can preserve upstream reachability.
 * The collected predecessor set defines candidate source nodes for post-removal connectivity restoration.
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
 * Disconnect all outbound connections for a node while collecting successor nodes so bridge-connection reconstruction can preserve downstream projection coverage.
 * The collected successor set defines candidate targets for bridging after structural deletion.
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
 * Create bridging connections from each predecessor to each successor when valid so node removal can maintain coarse connectivity without duplicating existing projections.
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
 * Reattach preserved gaters to randomly selected bridge connections so gate ownership can survive node removal when keep-gates mutation policy is enabled.
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
    const connectionToGate = bridgingConnections[connectionIndex];

    network.gate(gaterNode, connectionToGate);
    bridgingConnections.splice(connectionIndex, 1);
  }
}

/**
 * Ungate all connections currently gated by the removed node so detached gating references do not remain after structural mutation completes.
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
 * Remove a node from the runtime node list and mark index caches dirty so later activation and topology helpers rebuild index-based lookup state.
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
