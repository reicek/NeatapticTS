import type Network from '../network';
import type Node from '../node';

/**
 * Topology utilities.
 *
 * Provides:
 *  - computeTopoOrder: Kahn-style topological sorting with graceful fallback when cycles detected.
 *  - hasPath: depth-first reachability query (used to prevent cycle introduction when acyclicity enforced).
 *
 * Design Notes:
 *  - We deliberately tolerate cycles by falling back to raw node ordering instead of throwing; this
 *    allows callers performing interim structural mutations to proceed (e.g. during evolve phases)
 *    while signaling that the fast acyclic optimizations should not be used.
 *  - Input nodes are seeded into the queue immediately regardless of in-degree to keep them early in
 *    the ordering even if an unusual inbound edge was added (defensive redundancy).
 *  - Self loops are ignored for in-degree accounting and queue progression (they neither unlock new
 *    nodes nor should they block ordering completion).
 */

/**
 * Compute a topological ordering (Kahn's algorithm) for the current directed acyclic graph.
 * If cycles are detected (order shorter than node count) we fall back to raw node order to avoid breaking callers.
 * In non-acyclic mode we simply clear cached order to signal use of sequential node array.
 */
export function computeTopoOrder(this: Network): void {
  const internalNet = this as any;
  // Fast exit: if acyclicity not enforced we discard any cached order (signals using raw nodes list).
  if (!internalNet._enforceAcyclic) {
    internalNet._topoOrder = null;
    internalNet._topoDirty = false;
    return;
  }
  /** In-degree tally per node (excluding self loops). */
  const inDegree: Map<Node, number> = new Map();
  this.nodes.forEach((node) => inDegree.set(node, 0));
  for (const connection of this.connections) {
    if (connection.from !== connection.to) {
      inDegree.set(connection.to, (inDegree.get(connection.to) || 0) + 1);
    }
  }
  /** Processing queue for Kahn's algorithm. */
  const processingQueue: Node[] = [];
  this.nodes.forEach((node) => {
    if ((node as any).type === 'input' || (inDegree.get(node) || 0) === 0) {
      processingQueue.push(node);
    }
  });
  /** Accumulated topological order under construction. */
  const topoOrder: Node[] = [];
  while (processingQueue.length) {
    /** Next node with satisfied dependencies. */
    const node = processingQueue.shift()!;
    topoOrder.push(node);
    // Decrement in-degree of outgoing targets (ignoring self loops which were excluded earlier).
    for (const outgoing of (node as any).connections.out) {
      if (outgoing.to === node) continue; // Skip self loop.
      const remaining = (inDegree.get(outgoing.to) || 0) - 1;
      inDegree.set(outgoing.to, remaining);
      if (remaining === 0) processingQueue.push(outgoing.to);
    }
  }
  // Fallback: If cycle detected (not all nodes output), revert to raw node ordering to avoid partial order usage.
  internalNet._topoOrder =
    topoOrder.length === this.nodes.length ? topoOrder : this.nodes.slice();
  internalNet._topoDirty = false;
}

/** Depth-first reachability test (avoids infinite loops via visited set). */
export function hasPath(this: Network, from: Node, to: Node): boolean {
  if (from === to) return true; // Trivial reachability.
  /** Visited node set to prevent infinite traversal on cycles. */
  const visited = new Set<Node>();
  /** Stack for explicit depth-first search (iterative to avoid recursion limits). */
  const dfsStack: Node[] = [from];
  while (dfsStack.length) {
    const current = dfsStack.pop()!;
    if (current === to) return true;
    if (visited.has(current)) continue; // Already expanded.
    visited.add(current);
    for (const edge of (current as any).connections.out) {
      if (edge.to !== current) dfsStack.push(edge.to); // Skip self loops.
    }
  }
  return false;
}
