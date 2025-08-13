import type Network from '../network';
import Node from '../node';
import Connection from '../connection';

/**
 * Network structural mutation helpers (connect / disconnect).
 *
 * This module centralizes the logic for adding and removing edges (connections) between
 * nodes in a {@link Network}. By isolating the book‑keeping here we keep the primary
 * Network class lean and ensure consistent handling of:
 *  - Acyclic constraints
 *  - Multiple low‑level connections returned by composite node operations
 *  - Gating & self‑connection invariants
 *  - Cache invalidation (topological order + packed activation slabs)
 *
 * Exported functions:
 *  - {@link connect}: Create one or more connections from a source node to a target node.
 *  - {@link disconnect}: Remove (at most) one direct connection from source to target.
 *
 * Key terminology:
 *  - Self‑connection: An edge where from === to (loop). Usually disallowed under acyclicity.
 *  - Gating: A mechanism where a third node modulates (gates) the weight / influence of a connection.
 *  - Slab: Packed typed‑array representation of connections for vectorized forward passes.
 *
 * @module network.connect
 */

/**
 * Create and register one (or multiple) directed connection objects between two nodes.
 *
 * Some node types (or future composite structures) may return several low‑level connections when
 * their {@link Node.connect} is invoked (e.g., expanded recurrent templates). For that reason this
 * function always treats the result as an array and appends each edge to the appropriate collection.
 *
 * Algorithm outline:
 *  1. (Acyclic guard) If acyclicity is enforced and the source node appears after the target node in
 *     the network's node ordering, abort early and return an empty array (prevents back‑edge creation).
 *  2. Delegate to sourceNode.connect(targetNode, weight) to build the raw Connection object(s).
 *  3. For each created connection:
 *       a. If it's a self‑connection: either ignore (acyclic mode) or store in selfconns.
 *       b. Otherwise store in standard connections array.
 *  4. If any connection was added, mark structural caches dirty (_topoDirty & _slabDirty) so lazy
 *     rebuild can occur before the next forward pass.
 *
 * Complexity:
 *  - Time: O(k) where k is the number of low‑level connections returned (typically 1).
 *  - Space: O(k) new Connection instances (delegated to Node.connect).
 *
 * Edge cases & invariants:
 *  - Acyclic mode silently refuses back‑edges instead of throwing (makes evolutionary search easier).
 *  - Self‑connections are skipped entirely when acyclicity is enforced.
 *  - Weight initialization policy is delegated to Node.connect if not explicitly provided.
 *
 * @param this - Bound {@link Network} instance.
 * @param from - Source node (emits signal).
 * @param to - Target node (receives signal).
 * @param weight - Optional explicit initial weight value.
 * @returns Array of created {@link Connection} objects (possibly empty if acyclicity rejected the edge).
 * @example
 * const [edge] = net.connect(nodeA, nodeB, 0.5);
 * @remarks For bulk layer-to-layer wiring see higher-level utilities that iterate groups.
 */
export function connect(
  this: Network,
  from: Node,
  to: Node,
  weight?: number
): Connection[] {
  // Step 1: Acyclic pre‑check – prevents cycles by disallowing edges that point "backwards" in order.
  if (
    (this as any)._enforceAcyclic &&
    this.nodes.indexOf(from) > this.nodes.indexOf(to)
  )
    return [];

  // Step 2: Delegate creation to the node. May return >1 low‑level connections (treat generically).
  /** Array of new connection objects produced by the source node. */
  const connections = from.connect(to, weight);

  // Step 3: Register each new connection in the appropriate collection.
  for (const c of connections) {
    // c: individual low‑level connection
    if (from !== to) {
      // Standard edge (feed‑forward or recurrent) tracked in 'connections'.
      this.connections.push(c);
    } else {
      // Self‑connection: only valid when acyclicity is not enforced.
      if ((this as any)._enforceAcyclic) continue; // Skip silently to preserve invariant.
      this.selfconns.push(c);
    }
  }

  // Step 4: Invalidate caches if we materially changed structure (at least one edge added).
  if (connections.length) {
    (this as any)._topoDirty = true; // Topological ordering must be recomputed lazily.
    (this as any)._slabDirty = true; // Packed connection slab requires rebuild for fast activation path.
  }

  return connections; // Return created edges so caller can inspect / further manipulate (e.g., gating).
}

/**
 * Remove (at most) one directed connection from source 'from' to target 'to'.
 *
 * Only a single direct edge is removed because typical graph configurations maintain at most
 * one logical connection between a given pair of nodes (excluding potential future multi‑edge
 * semantics). If the target edge is gated we first call {@link Network.ungate} to maintain
 * gating invariants (ensuring the gater node's internal gate list remains consistent).
 *
 * Algorithm outline:
 *  1. Choose the correct list (selfconns vs connections) based on whether from === to.
 *  2. Linear scan to find the first edge with matching endpoints.
 *  3. If gated, ungate to detach gater bookkeeping.
 *  4. Splice the edge out; exit loop (only one expected).
 *  5. Delegate per‑node cleanup via from.disconnect(to) (clears reverse references, traces, etc.).
 *  6. Mark structural caches dirty for lazy recomputation.
 *
 * Complexity:
 *  - Time: O(m) where m is length of the searched list (connections or selfconns).
 *  - Space: O(1) extra.
 *
 * Idempotence: If no such edge exists we still perform node-level disconnect and flag caches dirty –
 * this conservative approach simplifies callers (they need not pre‑check existence).
 *
 * @param this - Bound {@link Network} instance.
 * @param from - Source node.
 * @param to - Target node.
 * @example
 * net.disconnect(nodeA, nodeB);
 * @remarks For removing many edges consider higher‑level bulk utilities to avoid repeated scans.
 */
export function disconnect(this: Network, from: Node, to: Node): void {
  // Step 1: Select list to search: selfconns for loops, otherwise normal connections.
  /** Candidate list of connections to inspect for removal. */
  const list = from === to ? this.selfconns : this.connections;

  // Step 2: Linear scan – lists are typically small relative to node count; acceptable trade‑off.
  for (let i = 0; i < list.length; i++) {
    /** Connection currently inspected. */
    const c = list[i];
    if (c.from === from && c.to === to) {
      // Found target edge
      // Step 3: If gated, maintain gating invariants by ungating before removal.
      if (c.gater) this.ungate(c);
      // Step 4: Remove and exit (only one expected between a pair of nodes).
      list.splice(i, 1);
      break;
    }
  }

  // Step 5: Node-level cleanup (clears internal references, derivative / eligibility traces, etc.).
  from.disconnect(to);

  // Step 6: Structural mutation => mark caches dirty so next activation can rebuild fast-path artifacts.
  (this as any)._topoDirty = true;
  (this as any)._slabDirty = true;
}
