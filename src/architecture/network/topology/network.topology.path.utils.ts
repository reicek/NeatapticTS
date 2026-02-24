import type {
  PathSearchContext,
  TopologyNode,
} from './network.topology.utils.types';
import { ZERO_COUNT } from './network.topology.utils.types';

/**
 * Create DFS search context.
 *
 * @param from Origin node.
 * @param to Target node.
 * @returns Initialized path-search context.
 */
export function createPathSearchContext(
  from: TopologyNode,
  to: TopologyNode,
): PathSearchContext {
  return {
    targetNode: to,
    visitedNodes: new Set<TopologyNode>(),
    nodesToVisitStack: [from],
  };
}

/**
 * Traverse DFS search stack and test reachability.
 *
 * @param searchContext Mutable search context.
 * @returns True when target node is reachable.
 */
export function traversePathSearch(searchContext: PathSearchContext): boolean {
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
 * Compare node identity.
 *
 * @param leftNode Left node.
 * @param rightNode Right node.
 * @returns True when references are identical.
 */
export function isSameNode(
  leftNode: TopologyNode,
  rightNode: TopologyNode,
): boolean {
  return leftNode === rightNode;
}

/**
 * Pop and return next DFS stack node.
 *
 * @param nodesToVisitStack DFS stack.
 * @returns Next node to process.
 */
function takeNextStackNode(nodesToVisitStack: TopologyNode[]): TopologyNode {
  return nodesToVisitStack.pop() as TopologyNode;
}

/**
 * Test whether a node has already been visited.
 *
 * @param visitedNodes Visited-node set.
 * @param node Candidate node.
 * @returns True when node is already visited.
 */
function hasVisitedNode(
  visitedNodes: Set<TopologyNode>,
  node: TopologyNode,
): boolean {
  return visitedNodes.has(node);
}

/**
 * Mark a node as visited.
 *
 * @param visitedNodes Visited-node set.
 * @param node Node to mark.
 * @returns Void.
 */
function markVisited(
  visitedNodes: Set<TopologyNode>,
  node: TopologyNode,
): void {
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
  nodesToVisitStack: TopologyNode[],
  currentNode: TopologyNode,
): void {
  for (const outgoingConnection of currentNode.connections.out) {
    if (isSelfConnection(outgoingConnection.to, currentNode)) {
      continue;
    }

    nodesToVisitStack.push(outgoingConnection.to);
  }
}

/**
 * Test whether a connection is a self-loop.
 *
 * @param from Source node.
 * @param to Target node.
 * @returns True when source and target are the same node.
 */
function isSelfConnection(from: TopologyNode, to: TopologyNode): boolean {
  return from === to;
}
