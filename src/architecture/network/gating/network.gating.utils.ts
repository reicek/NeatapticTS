/**
 * Connection-gating utilities and gate-aware repair helpers for network edits.
 *
 * Gating is the lightweight way to let one node decide how strongly another
 * connection should matter right now. That makes this chapter about more than
 * a simple `gate()` setter. It also owns the awkward structural case where a
 * hidden node is removed and the network tries to preserve useful gated
 * behavior by reconnecting predecessor and successor nodes, then reassigning
 * preserved gaters onto the new bridge connections.
 *
 * The split inside this folder follows those two jobs. `gate.utils` keeps
 * ordinary gate and ungate bookkeeping small and predictable. `remove.utils`
 * handles the more invasive bridge-repair flow used during gate-aware hidden
 * node removal. The shared types file only carries temporary collections used
 * during that repair work; it is not the right chapter intro.
 *
 * ```mermaid
 * flowchart LR
 *   Gater[Gater node activation] --> Modulate[Modulate connection gain]
 *   Modulate --> Target[Target node input sum]
 *   Remove[Remove hidden node] --> Bridge[Create bridge connections]
 *   Bridge --> Preserve[Reassign preserved gaters]
 * ```
 *
 * A useful mental model is that gating changes when a connection is
 * influential, while bridge repair tries to preserve where information can
 * still travel after topology surgery. Keeping both concerns together makes
 * the public surface easier to learn because callers usually encounter them in
 * the same network-editing workflows.
 *
 * Example: attach a hidden node as a gater for one connection.
 *
 * ```ts
 * network.gate(hiddenNode, connection);
 * ```
 *
 * Example: later remove that modulation and return the connection to static
 * weighting.
 *
 * ```ts
 * network.ungate(connection);
 * ```
 */
import type Network from '../../network/network';
import Node from '../../node';
import Connection from '../../connection';
import { synchronizeTemporalDescriptorExtensions } from '../network.temporal.extensions.utils';
import {
  assertGaterNodeBelongsToNetwork,
  isConnectionAlreadyGated,
  warnConnectionAlreadyGated,
  attachGaterToConnection,
  findGateIndex,
  warnMissingGateConnection,
  removeGateAtIndex,
  detachConnectionFromGater,
} from './network.gating.gate.utils';
import {
  assertNodeRemovableAndGetIndex,
  resolveSubNodeMutationConfig,
  disconnectNodeSelfLoop,
  disconnectInboundConnections,
  disconnectOutboundConnections,
  createBridgingConnections,
  reassignPreservedGaters,
  ungateConnectionsGatedByNode,
  removeNodeAtIndex,
} from './network.gating.remove.utils';

/**
 * Attach a gater node to a connection so that the connection's effective weight
 * becomes dynamically modulated by the gater's activation (see {@link Node.gate} for exact math).
 *
 * Validation / invariants:
 *  - Throws if the gater node is not part of this network (prevents cross-network corruption).
 *  - If the connection is already gated, function is a no-op (emits warning when enabled).
 *  - Successful gate attachment revalidates the explicit temporal descriptor bag so generic gating edits
 *    cannot leave stale module metadata behind.
 *
 * Complexity: O(1)
 *
 * @param this - Bound Network instance.
 * @param node - Candidate gater node (must belong to network).
 * @param connection - Connection to gate.
 */
export function gate(this: Network, node: Node, connection: Connection) {
  // Step 1: Validate the gater node belongs to this network.
  assertGaterNodeBelongsToNetwork(this, node);

  // Step 2: Keep operation idempotent when the connection is already gated.
  if (isConnectionAlreadyGated(connection)) {
    warnConnectionAlreadyGated();
    return;
  }

  // Step 3: Attach and track the gate connection.
  attachGaterToConnection(this, node, connection);

  // Step 4: Revalidate explicit temporal descriptors after gate attachment.
  synchronizeTemporalDescriptorExtensions(this);
}

/**
 * Remove gating from a connection, restoring its static weight contribution.
 *
 * Idempotent: If the connection is not currently gated, the call performs no structural changes
 * (and optionally logs a warning). After ungating, the connection's weight will be used directly
 * without modulation by a gater activation.
 * Successful ungate operations also revalidate the explicit temporal descriptor bag so stale gated-block
 * descriptors do not survive until a later serialize pass.
 *
 * Complexity: O(n) where n = number of gated connections (indexOf lookup) – typically small.
 *
 * @param this - Bound Network instance.
 * @param connection - Connection to ungate.
 */
export function ungate(this: Network, connection: Connection) {
  // Step 1: Locate the connection in the network gates list.
  const gateIndex = findGateIndex(this, connection);

  // Step 2: Warn and stop when ungating a non-tracked connection.
  if (gateIndex === -1) {
    warnMissingGateConnection();
    return;
  }

  // Step 3: Remove gate tracking and detach gater bookkeeping.
  removeGateAtIndex(this, gateIndex);
  detachConnectionFromGater(connection);

  // Step 4: Revalidate explicit temporal descriptors after gate removal.
  synchronizeTemporalDescriptorExtensions(this);
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
 *  7. Ungate connections that were gated BY this node (where node acted as gater).
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
 * @param this - Bound Network instance.
 * @param node - Hidden node to remove.
 * @throws If node is input/output or not present in network.
 */
export function removeNode(this: Network, node: Node) {
  // Step 1: Validate removal preconditions and gather mutable state.
  const nodeIndex = assertNodeRemovableAndGetIndex(this, node);
  const subNodeConfig = resolveSubNodeMutationConfig();
  const preservedGaters: Node[] = [];

  // Step 2: Disconnect old edges and collect predecessor/successor nodes.
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

  // Step 3: Rebuild bridge connectivity and preserve gating behavior.
  reassignPreservedGaters(this, preservedGaters, bridgingConnections);
  ungateConnectionsGatedByNode(this, node);
  removeNodeAtIndex(this, nodeIndex);
}

// Only functions exported; keep module shape predictable for tree-shaking / documentation tooling.
export {};
