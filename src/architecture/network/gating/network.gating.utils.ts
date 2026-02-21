import type Network from '../../network';
import Node from '../../node';
import Connection from '../../connection';
import mutation from '../../../methods/mutation';
import { config } from '../../../config';
import type {
  GatingNetworkProps as NetworkGatingProps,
  SubNodeMutationConfig,
} from '../network.types';

/**
 * Attach a gater node to a connection so that the connection's effective weight
 * becomes dynamically modulated by the gater's activation (see {@link Node.gate} for exact math).
 *
 * Validation / invariants:
 *  - Throws if the gater node is not part of this network (prevents cross-network corruption).
 *  - If the connection is already gated, function is a no-op (emits warning when enabled).
 *
 * Complexity: O(1)
 *
 * @param this - Bound {@link Network} instance.
 * @param node - Candidate gater node (must belong to network).
 * @param connection - Connection to gate.
 */
export function gate(this: Network, node: Node, connection: Connection) {
  assertGaterNodeBelongsToNetwork(this, node);
  if (isConnectionAlreadyGated(connection)) {
    warnConnectionAlreadyGated();
    return;
  }

  attachGaterToConnection(this, node, connection);
}

/**
 * Remove gating from a connection, restoring its static weight contribution.
 *
 * Idempotent: If the connection is not currently gated, the call performs no structural changes
 * (and optionally logs a warning). After ungating, the connection's weight will be used directly
 * without modulation by a gater activation.
 *
 * Complexity: O(n) where n = number of gated connections (indexOf lookup) – typically small.
 *
 * @param this - Bound {@link Network} instance.
 * @param connection - Connection to ungate.
 */
export function ungate(this: Network, connection: Connection) {
  const gateIndex = findGateIndex(this, connection);
  if (gateIndex === -1) {
    warnMissingGateConnection();
    return;
  }

  removeGateAtIndex(this, gateIndex);
  detachConnectionFromGater(connection);
}

/**
 * Remove a hidden node from the network while attempting to preserve functional connectivity.
 *
 * Algorithm outline:
 *  1. Reject removal if node is input/output (structural invariants) or absent (error).
 *  2. Optionally collect gating nodes (if keep_gates flag) from inbound & outbound connections.
 *  3. Remove self-loop (if present) to simplify subsequent edge handling.
 *  4. Disconnect all inbound edges (record their source nodes) and all outbound edges (record targets).
 *  5. For every (input predecessor, output successor) pair create a new connection unless:
 *       a. input === output (avoid trivial self loops) OR
 *       b. an existing projection already connects them.
 *  6. Reassign preserved gater nodes randomly onto newly created bridging connections.
 *  7. Ungate any connections that were gated BY this node (where node acted as gater).
 *  8. Remove node from network node list and flag node index cache as dirty.
 *
 * Complexity summary:
 *  - Let I = number of inbound edges, O = number of outbound edges.
 *  - Disconnect phase: O(I + O)
 *  - Bridging phase: O(I * O) connection existence checks (isProjectingTo) + potential additions.
 *  - Gater reassignment: O(min(G, newConnections)) where G is number of preserved gaters.
 *
 * Preservation rationale:
 *  - Reassigning gaters maintains some of the dynamic modulation capacity that would otherwise
 *    be lost, aiding continuity during topology simplification.
 *
 * @param this - Bound {@link Network} instance.
 * @param node - Hidden node to remove.
 * @throws If node is input/output or not present in network.
 */
export function removeNode(this: Network, node: Node) {
  const nodeIndex = assertNodeRemovableAndGetIndex(this, node);
  const subNodeConfig = resolveSubNodeMutationConfig();
  const preservedGaters: Node[] = [];

  disconnectNodeSelfLoop(this, node);
  const predecessorNodes = disconnectInboundConnections(
    this,
    node,
    preservedGaters,
    subNodeConfig,
  );
  const successorNodes = disconnectOutboundConnections(
    this,
    node,
    preservedGaters,
    subNodeConfig,
  );
  const bridgingConnections = createBridgingConnections(
    this,
    predecessorNodes,
    successorNodes,
  );

  reassignPreservedGaters(this, preservedGaters, bridgingConnections);
  ungateConnectionsGatedByNode(this, node);
  removeNodeAtIndex(this, nodeIndex);
}

/**
 * Validate that a candidate gater node belongs to the target network.
 *
 * @param network - Network performing the gating operation.
 * @param node - Candidate gater node.
 * @returns Nothing.
 * @throws If the node does not belong to the network.
 */
function assertGaterNodeBelongsToNetwork(network: Network, node: Node): void {
  if (!network.nodes.includes(node)) {
    throw new Error(
      'Gating node must be part of the network to gate a connection!',
    );
  }
}

/**
 * Determine whether a connection already has a gater node assigned.
 *
 * @param connection - Connection candidate.
 * @returns True when a gater is already set; otherwise false.
 */
function isConnectionAlreadyGated(connection: Connection): boolean {
  return Boolean(connection.gater);
}

/**
 * Emit a warning when a gate operation is skipped due to existing gating.
 *
 * @returns Nothing.
 */
function warnConnectionAlreadyGated(): void {
  if (config.warnings) {
    console.warn('Connection is already gated. Skipping.');
  }
}

/**
 * Attach a gater to a connection and track that connection in the network gate list.
 *
 * @param network - Network being updated.
 * @param node - Gater node to attach.
 * @param connection - Connection to gate.
 * @returns Nothing.
 */
function attachGaterToConnection(
  network: Network,
  node: Node,
  connection: Connection,
): void {
  node.gate(connection);
  network.gates.push(connection);
}

/**
 * Find a connection position within the network global gates list.
 *
 * @param network - Network containing global gate references.
 * @param connection - Connection to locate.
 * @returns Zero-based index in the gates list, or -1 when absent.
 */
function findGateIndex(network: Network, connection: Connection): number {
  return network.gates.indexOf(connection);
}

/**
 * Emit a warning when an ungate request targets a non-tracked connection.
 *
 * @returns Nothing.
 */
function warnMissingGateConnection(): void {
  if (config.warnings) {
    console.warn('Attempted to ungate a connection not in the gates list.');
  }
}

/**
 * Remove a gated connection from the network global gate list.
 *
 * @param network - Network being updated.
 * @param gateIndex - Index to remove from the gate list.
 * @returns Nothing.
 */
function removeGateAtIndex(network: Network, gateIndex: number): void {
  network.gates.splice(gateIndex, 1);
}

/**
 * Remove reverse gater bookkeeping from a connection's gater node.
 *
 * @param connection - Connection to detach from its gater.
 * @returns Nothing.
 */
function detachConnectionFromGater(connection: Connection): void {
  connection.gater?.ungate(connection);
}

/**
 * Ensure a node can be removed and return its index in the network node list.
 *
 * @param network - Network containing the node.
 * @param node - Node to validate.
 * @returns Index of the node in the network node list.
 * @throws If the node is an input/output node or absent from the network.
 */
function assertNodeRemovableAndGetIndex(network: Network, node: Node): number {
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
function resolveSubNodeMutationConfig(): SubNodeMutationConfig | undefined {
  const subNodeConfig = Array.isArray(mutation.SUB_NODE)
    ? mutation.SUB_NODE[0]
    : mutation.SUB_NODE;
  return subNodeConfig as SubNodeMutationConfig | undefined;
}

/**
 * Disconnect a node self-loop before broader edge rewiring.
 *
 * @param network - Network being updated.
 * @param node - Node whose self-loop should be removed.
 * @returns Nothing.
 */
function disconnectNodeSelfLoop(network: Network, node: Node): void {
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
function disconnectInboundConnections(
  network: Network,
  node: Node,
  preservedGaters: Node[],
  subNodeConfig: SubNodeMutationConfig | undefined,
): Node[] {
  const predecessorNodes: Node[] = [];
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
function disconnectOutboundConnections(
  network: Network,
  node: Node,
  preservedGaters: Node[],
  subNodeConfig: SubNodeMutationConfig | undefined,
): Node[] {
  const successorNodes: Node[] = [];
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
  preservedGaters: Node[],
  subNodeConfig: SubNodeMutationConfig | undefined,
): void {
  if (!subNodeConfig?.keep_gates || !connection.gater) {
    return;
  }

  if (connection.gater !== removedNode) {
    preservedGaters.push(connection.gater);
  }
}

/**
 * Create bridging connections from each predecessor to each successor when valid.
 *
 * @param network - Network where bridge connections are created.
 * @param predecessorNodes - Source nodes collected from inbound edges.
 * @param successorNodes - Target nodes collected from outbound edges.
 * @returns Newly created bridge connections.
 */
function createBridgingConnections(
  network: Network,
  predecessorNodes: Node[],
  successorNodes: Node[],
): Connection[] {
  const bridgingConnections: Connection[] = [];

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
 * Reattach preserved gaters to randomly selected newly-created bridge connections.
 *
 * @param network - Network performing reassignment.
 * @param preservedGaters - Gaters retained during node removal.
 * @param bridgingConnections - Available bridge connections for reassignment.
 * @returns Nothing.
 */
function reassignPreservedGaters(
  network: Network,
  preservedGaters: Node[],
  bridgingConnections: Connection[],
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
 * Select a uniformly random integer index in the range [0, length).
 *
 * @param length - Upper bound (exclusive).
 * @returns Random zero-based index.
 */
function selectRandomIndex(length: number): number {
  return Math.floor(Math.random() * length);
}

/**
 * Ungate all connections that are currently gated by the removed node.
 *
 * @param network - Network performing ungate operations.
 * @param node - Node whose gated connections should be released.
 * @returns Nothing.
 */
function ungateConnectionsGatedByNode(network: Network, node: Node): void {
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
function removeNodeAtIndex(network: Network, nodeIndex: number): void {
  network.nodes.splice(nodeIndex, 1);
  (network as unknown as NetworkGatingProps)._nodeIndexDirty = true;
}

// Only functions exported; keep module shape predictable for tree-shaking / documentation tooling.
export {};
