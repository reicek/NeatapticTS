import type Network from '../../network/network';
import type Node from '../../node';
import {
  getTopologyIntent,
  hasFeedForwardTopologyContract,
  setEnforceAcyclic,
  setTopologyIntent,
} from './network.topology.contract.utils';
export {
  getTopologyIntent,
  hasFeedForwardTopologyContract,
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
  createTopologyBuildContext,
  finalizeRecurrentSchedule,
  finalizeTopoOrder,
  initializeAllNodeInDegreeCounts,
  shouldBuildRecurrentSchedule,
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
 * Compute a deterministic activation schedule for the current topology mode.
 *
 * Acyclic mode uses Kahn traversal with stable waves and still flattens those
 * waves back into the legacy `_topoOrder` cache for callers that depend on one
 * ordered list. Recurrent mode uses the SCC condensation graph to emit
 * deterministic recurrent-component boundaries while leaving the legacy acyclic
 * cache empty until the activation path adopts the richer schedule directly.
 * @param this - Network instance bound by method call.
 */
export function computeTopoOrder(this: Network): void {
  // Step 1: Resolve internal topology flags.
  const internalTopologyProps = asTopologyProps(this);

  // Step 2: Build recurrent schedule when acyclic enforcement is disabled.
  if (shouldBuildRecurrentSchedule(internalTopologyProps)) {
    finalizeRecurrentSchedule(this, internalTopologyProps);
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

/**
 * Depth-first reachability test that avoids infinite loops using a visited set.
 *
 * @param this - Network instance bound by method call.
 * @param from - Source node from which reachability is tested.
 * @param to - Target node to which reachability is tested.
 * @returns True when a path exists from `from` to `to`, false otherwise.
 */
export function hasPath(this: Network, from: Node, to: Node): boolean {
  // Step 1: Handle trivial reachability.
  if (isSameNode(from, to)) {
    return true;
  }

  // Step 2: Traverse from origin to target using iterative DFS.
  const searchContext = createPathSearchContext(from, to);
  return traversePathSearch(searchContext);
}

/**
 * Default export bundle for the topology utilities chapter.
 *
 * Bundles the core topology helpers so the network facade can bind them as methods
 * without importing each function individually.
 */
const networkTopologyUtils = {
  computeTopoOrder,
  getTopologyIntent,
  hasPath,
  hasFeedForwardTopologyContract,
  setEnforceAcyclic,
  setTopologyIntent,
};
export default networkTopologyUtils;
