import type Network from '../network';
import Node from '../node';
import Connection from '../connection';
import mutation from '../../methods/mutation';
import { config } from '../../config';

/**
 * Gating & node removal utilities for {@link Network}.
 *
 * Gating concept:
 *  - A "gater" node modulates the effective weight of a target connection. Conceptually the raw
 *    connection weight w is multiplied (or otherwise transformed) by a function of the gater node's
 *    activation a_g (actual math lives in {@link Node.gate}). This enables dynamic, context-sensitive
 *    routing (similar in spirit to attention mechanisms or LSTM-style gates) within an evolved topology.
 *
 * Removal strategy (removeNode):
 *  - When excising a hidden node we attempt to preserve overall connectivity by creating bridging
 *    connections from each of its predecessors to each of its successors if such edges do not already
 *    exist. Optional logic reassigns previous gater nodes to these new edges (best-effort) to preserve
 *    modulation diversity.
 *
 * Mutation interplay:
 *  - The flag `mutation.SUB_NODE.keep_gates` determines whether gating nodes associated with edges
 *    passing through the removed node should be retained and reassigned.
 *
 * Determinism note:
 *  - Bridging gate reassignment currently uses Math.random directly; for fully deterministic runs
 *    you may consider replacing with the network's seeded RNG (if provided) in future refactors.
 *
 * Exported functions:
 *  - {@link gate}: Attach a gater to a connection.
 *  - {@link ungate}: Remove gating from a connection.
 *  - {@link removeNode}: Remove a hidden node while attempting to preserve connectivity & gating.
 *
 * @module network.gating
 */

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
  if (!this.nodes.includes(node))
    throw new Error(
      'Gating node must be part of the network to gate a connection!'
    );
  if (connection.gater) {
    if (config.warnings) console.warn('Connection is already gated. Skipping.');
    return;
  }
  node.gate(connection); // Delegate per-node bookkeeping (adds to node.connections.gated & sets connection.gater)
  this.gates.push(connection); // Track globally for fast iteration / serialization.
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
  /** Index of the connection within the global gates list ( -1 if not found ). */
  const index = this.gates.indexOf(connection);
  if (index === -1) {
    if (config.warnings)
      console.warn('Attempted to ungate a connection not in the gates list.');
    return;
  }
  this.gates.splice(index, 1); // Remove from global gated list.
  connection.gater?.ungate(connection); // Remove reverse reference from the gater node.
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
  if (node.type === 'input' || node.type === 'output')
    throw new Error('Cannot remove input or output node from the network.');
  const idx = this.nodes.indexOf(node);
  if (idx === -1) throw new Error('Node not found in the network for removal.');

  // Collected gating nodes to potentially reattach to new bridging connections.
  /** Collection of gater nodes preserved for reassignment onto new bridging connections. */
  const gaters: Node[] = [];

  // Remove self-loop first (simplifies later logic and ensures gating removal handled early).
  this.disconnect(node, node);

  // Gather inbound source nodes and optionally preserve their gaters.
  /** List of source nodes feeding into the node being removed (predecessors). */
  const inputs: Node[] = [];
  for (let i = node.connections.in.length - 1; i >= 0; i--) {
    const c = node.connections.in[i];
    if (mutation.SUB_NODE.keep_gates && c.gater && c.gater !== node)
      gaters.push(c.gater);
    inputs.push(c.from);
    this.disconnect(c.from, node);
  }

  // Gather outbound destination nodes similarly.
  /** List of destination nodes the node being removed projects to (successors). */
  const outputs: Node[] = [];
  for (let i = node.connections.out.length - 1; i >= 0; i--) {
    const c = node.connections.out[i];
    if (mutation.SUB_NODE.keep_gates && c.gater && c.gater !== node)
      gaters.push(c.gater);
    outputs.push(c.to);
    this.disconnect(node, c.to);
  }

  // Create bridging connections between every predecessor and successor (if not already connected).
  /** New bridging connections created to preserve path connectivity after removal. */
  const newConns: Connection[] = [];
  for (const input of inputs) {
    for (const output of outputs) {
      // Skip trivial self-loop & skip if an existing connection already links them.
      if (input !== output && !input.isProjectingTo(output)) {
        const conn = this.connect(input, output);
        if (conn.length) newConns.push(conn[0]); // Only record created connection
      }
    }
  }

  // Reassign preserved gaters randomly to newly formed bridging connections.
  for (const g of gaters) {
    if (!newConns.length) break; // No more candidate connections
    /** Random index into the remaining pool of new bridging connections for gater reassignment. */
    const ci = Math.floor(Math.random() * newConns.length);
    this.gate(g, newConns[ci]);
    newConns.splice(ci, 1); // Avoid double‑gating same connection
  }

  // Ungate connections that were gated by the removed node itself.
  for (let i = node.connections.gated.length - 1; i >= 0; i--) {
    this.ungate(node.connections.gated[i]);
  }

  // Final removal & cache invalidation (indices may be used by fast lookup structures elsewhere).
  this.nodes.splice(idx, 1);
  (this as any)._nodeIndexDirty = true;
}

// Only functions exported; keep module shape predictable for tree-shaking / documentation tooling.
export {};
