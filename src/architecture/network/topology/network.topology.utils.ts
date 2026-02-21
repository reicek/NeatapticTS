import type Network from '../../network';
import type Node from '../../node';
import type {
  PathSearchContext,
  TopologyBuildContext,
  TopologyNetworkProps as NetworkTopologyProps,
} from '../network.types';

const INPUT_NODE_TYPE = 'input';
const ZERO_COUNT = 0;
const IN_DEGREE_DECREMENT = 1;

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
// eslint-disable-next-line prefer-arrow/prefer-arrow-functions
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
 * Cast network to internal topology props view.
 *
 * @param network Network instance.
 * @returns Internal topology props view.
 */
function asTopologyProps(network: Network): NetworkTopologyProps {
  return network as unknown as NetworkTopologyProps;
}

/**
 * Determine whether topological order should be bypassed.
 *
 * @param internalTopologyProps Internal topology props view.
 * @returns True when acyclic mode is disabled.
 */
function shouldUseRawNodeOrder(
  internalTopologyProps: NetworkTopologyProps,
): boolean {
  return !internalTopologyProps._enforceAcyclic;
}

/**
 * Clear cached topological order state.
 *
 * @param internalTopologyProps Internal topology props view.
 * @returns Void.
 */
function clearCachedTopoOrder(
  internalTopologyProps: NetworkTopologyProps,
): void {
  internalTopologyProps._topoOrder = null;
  internalTopologyProps._topoDirty = false;
}

/**
 * Create mutable build context for Kahn traversal.
 *
 * @param network Network instance.
 * @param internalTopologyProps Internal topology props view.
 * @returns Initialized build context.
 */
function createTopologyBuildContext(
  network: Network,
  internalTopologyProps: NetworkTopologyProps,
): TopologyBuildContext {
  return {
    network,
    internalTopologyProps,
    inDegreeByNode: new Map<Node, number>(),
    processingQueue: [],
    topoOrder: [],
  };
}

/**
 * Initialize all nodes with zero in-degree.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
function initializeAllNodeInDegreeCounts(
  buildContext: TopologyBuildContext,
): void {
  for (const node of buildContext.network.nodes) {
    buildContext.inDegreeByNode.set(node, ZERO_COUNT);
  }
}

/**
 * Apply in-degree increments from non-self connections.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
function applyIncomingEdgeCounts(buildContext: TopologyBuildContext): void {
  for (const connection of buildContext.network.connections) {
    if (isSelfConnection(connection.from, connection.to)) {
      continue;
    }

    incrementNodeInDegree(buildContext.inDegreeByNode, connection.to);
  }
}

/**
 * Seed Kahn queue with input nodes and zero in-degree nodes.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
function seedProcessingQueue(buildContext: TopologyBuildContext): void {
  for (const node of buildContext.network.nodes) {
    if (isQueueSeedNode(node, buildContext.inDegreeByNode)) {
      buildContext.processingQueue.push(node);
    }
  }
}

/**
 * Process queue until all available nodes are emitted.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
function processKahnQueue(buildContext: TopologyBuildContext): void {
  while (buildContext.processingQueue.length > ZERO_COUNT) {
    const currentNode = takeNextQueueNode(buildContext.processingQueue);
    appendTopoNode(buildContext.topoOrder, currentNode);
    relaxOutgoingEdges(buildContext, currentNode);
  }
}

/**
 * Finalize cached order, falling back to raw node order on cycle detection.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
function finalizeTopoOrder(buildContext: TopologyBuildContext): void {
  buildContext.internalTopologyProps._topoOrder =
    resolveFinalOrder(buildContext);
  buildContext.internalTopologyProps._topoDirty = false;
}

/**
 * Compare node identity.
 *
 * @param leftNode Left node.
 * @param rightNode Right node.
 * @returns True when references are identical.
 */
function isSameNode(leftNode: Node, rightNode: Node): boolean {
  return leftNode === rightNode;
}

/**
 * Create DFS search context.
 *
 * @param from Origin node.
 * @param to Target node.
 * @returns Initialized path-search context.
 */
function createPathSearchContext(from: Node, to: Node): PathSearchContext {
  return {
    targetNode: to,
    visitedNodes: new Set<Node>(),
    nodesToVisitStack: [from],
  };
}

/**
 * Traverse DFS search stack and test reachability.
 *
 * @param searchContext Mutable search context.
 * @returns True when target node is reachable.
 */
function traversePathSearch(searchContext: PathSearchContext): boolean {
  while (searchContext.nodesToVisitStack.length > ZERO_COUNT) {
    const currentNode = takeNextStackNode(searchContext.nodesToVisitStack);

    if (isSameNode(currentNode, searchContext.targetNode)) {
      return true;
    }

    if (hasVisitedNode(searchContext.visitedNodes, currentNode)) {
      continue;
    }

    markVisited(searchContext.visitedNodes, currentNode);
    pushOutgoingTargets(searchContext.nodesToVisitStack, currentNode);
  }

  return false;
}

/**
 * Test whether a connection is a self-loop.
 *
 * @param from Source node.
 * @param to Target node.
 * @returns True when source and target are the same node.
 */
function isSelfConnection(from: Node, to: Node): boolean {
  return from === to;
}

/**
 * Increment in-degree for a node in the tally map.
 *
 * @param inDegreeByNode In-degree tally map.
 * @param node Target node.
 * @returns Void.
 */
function incrementNodeInDegree(
  inDegreeByNode: Map<Node, number>,
  node: Node,
): void {
  const currentInDegree = inDegreeByNode.get(node) ?? ZERO_COUNT;
  inDegreeByNode.set(node, currentInDegree + IN_DEGREE_DECREMENT);
}

/**
 * Determine whether a node belongs in the initial queue.
 *
 * @param node Candidate node.
 * @param inDegreeByNode In-degree tally map.
 * @returns True when node is input-type or has zero in-degree.
 */
function isQueueSeedNode(
  node: Node,
  inDegreeByNode: Map<Node, number>,
): boolean {
  return isInputNode(node) || getInDegree(inDegreeByNode, node) === ZERO_COUNT;
}

/**
 * Test whether a node is an input node.
 *
 * @param node Candidate node.
 * @returns True when node type is input.
 */
function isInputNode(node: Node): boolean {
  return node.type === INPUT_NODE_TYPE;
}

/**
 * Read in-degree for a node with zero fallback.
 *
 * @param inDegreeByNode In-degree tally map.
 * @param node Candidate node.
 * @returns In-degree value.
 */
function getInDegree(inDegreeByNode: Map<Node, number>, node: Node): number {
  return inDegreeByNode.get(node) ?? ZERO_COUNT;
}

/**
 * Shift and return the next queue node.
 *
 * @param processingQueue Queue of pending nodes.
 * @returns Next node.
 */
function takeNextQueueNode(processingQueue: Node[]): Node {
  return processingQueue.shift() as Node;
}

/**
 * Append one node to topological order output.
 *
 * @param topoOrder Accumulated topological order.
 * @param node Node to append.
 * @returns Void.
 */
function appendTopoNode(topoOrder: Node[], node: Node): void {
  topoOrder.push(node);
}

/**
 * Relax outgoing edges for one processed node.
 *
 * @param buildContext Mutable build context.
 * @param currentNode Processed node.
 * @returns Void.
 */
function relaxOutgoingEdges(
  buildContext: TopologyBuildContext,
  currentNode: Node,
): void {
  for (const outgoingConnection of currentNode.connections.out) {
    if (isSelfConnection(outgoingConnection.to, currentNode)) {
      continue;
    }

    const remainingInDegree = decrementNodeInDegree(
      buildContext.inDegreeByNode,
      outgoingConnection.to,
    );

    if (remainingInDegree === ZERO_COUNT) {
      buildContext.processingQueue.push(outgoingConnection.to);
    }
  }
}

/**
 * Decrement node in-degree and return remaining value.
 *
 * @param inDegreeByNode In-degree tally map.
 * @param node Target node.
 * @returns Remaining in-degree after decrement.
 */
function decrementNodeInDegree(
  inDegreeByNode: Map<Node, number>,
  node: Node,
): number {
  const currentInDegree = getInDegree(inDegreeByNode, node);
  const remainingInDegree = currentInDegree - IN_DEGREE_DECREMENT;
  inDegreeByNode.set(node, remainingInDegree);
  return remainingInDegree;
}

/**
 * Resolve final topological order with cycle fallback.
 *
 * @param buildContext Mutable build context.
 * @returns Fully valid topological order or raw node order fallback.
 */
function resolveFinalOrder(buildContext: TopologyBuildContext): Node[] {
  const isCompleteOrder =
    buildContext.topoOrder.length === buildContext.network.nodes.length;
  if (isCompleteOrder) {
    return buildContext.topoOrder;
  }

  return [...buildContext.network.nodes];
}

/**
 * Pop and return next DFS stack node.
 *
 * @param nodesToVisitStack DFS stack.
 * @returns Next node to process.
 */
function takeNextStackNode(nodesToVisitStack: Node[]): Node {
  return nodesToVisitStack.pop() as Node;
}

/**
 * Test whether a node has already been visited.
 *
 * @param visitedNodes Visited-node set.
 * @param node Candidate node.
 * @returns True when node is already visited.
 */
function hasVisitedNode(visitedNodes: Set<Node>, node: Node): boolean {
  return visitedNodes.has(node);
}

/**
 * Mark a node as visited.
 *
 * @param visitedNodes Visited-node set.
 * @param node Node to mark.
 * @returns Void.
 */
function markVisited(visitedNodes: Set<Node>, node: Node): void {
  visitedNodes.add(node);
}

/**
 * Push non-self outgoing targets to DFS stack.
 *
 * @param nodesToVisitStack DFS stack.
 * @param currentNode Current expanded node.
 * @returns Void.
 */
function pushOutgoingTargets(
  nodesToVisitStack: Node[],
  currentNode: Node,
): void {
  for (const outgoingConnection of currentNode.connections.out) {
    if (isSelfConnection(outgoingConnection.to, currentNode)) {
      continue;
    }

    nodesToVisitStack.push(outgoingConnection.to);
  }
}

export default { computeTopoOrder, hasPath };
