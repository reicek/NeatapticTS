import type Network from '../../network/network';
import type Node from '../../node';
import {
  getTopologyIntent,
  setEnforceAcyclic,
  setTopologyIntent,
} from './network.topology.contract.utils';
export {
  getTopologyIntent,
  setEnforceAcyclic,
  setTopologyIntent,
} from './network.topology.contract.utils';
export {
  createMLP,
  rebuildConnections,
} from './network.topology.factory.utils';
import {
  processKahnQueue,
  seedProcessingQueue,
} from './network.topology.loop.utils';
import {
  createPathSearchContext,
  isSameNode,
  traversePathSearch,
} from './network.topology.path.utils';
import {
  applyIncomingEdgeCounts,
  asTopologyProps,
  clearCachedTopoOrder,
  createTopologyBuildContext,
  finalizeTopoOrder,
  initializeAllNodeInDegreeCounts,
  shouldUseRawNodeOrder,
} from './network.topology.setup.utils';

/**
 * Topology utilities.
 *
 * Provides:
 *  - computeTopoOrder: Kahn-style topological sorting with graceful fallback when cycles detected.
 *  - hasPath: depth-first reachability query (used to prevent cycle introduction when acyclicity enforced).
 *  - topology contract helpers: public intent accessors that keep semantic API state aligned with low-level runtime flags.
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
  // Step 1: Resolve internal topology flags.
  const internalTopologyProps = asTopologyProps(this);

  // Step 2: Handle non-acyclic mode by clearing cached topological order.
  if (shouldUseRawNodeOrder(internalTopologyProps)) {
    clearCachedTopoOrder(internalTopologyProps);
    return;
  }

  // Step 3: Build Kahn traversal context and in-degree model.
  const buildContext = createTopologyBuildContext(this, internalTopologyProps);
  initializeAllNodeInDegreeCounts(buildContext);
  applyIncomingEdgeCounts(buildContext);

  // Step 4: Seed queue, traverse graph, and finalize cache.
  seedProcessingQueue(buildContext);
  processKahnQueue(buildContext);
  finalizeTopoOrder(buildContext);
}

/** Depth-first reachability test (avoids infinite loops via visited set). */
export function hasPath(this: Network, from: Node, to: Node): boolean {
  // Step 1: Handle trivial reachability.
  if (isSameNode(from, to)) {
    return true;
  }

  // Step 2: Traverse from origin to target using iterative DFS.
  const searchContext = createPathSearchContext(from, to);
  return traversePathSearch(searchContext);
}

export default {
  computeTopoOrder,
  getTopologyIntent,
  hasPath,
  setEnforceAcyclic,
  setTopologyIntent,
};
