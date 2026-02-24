import type {
  TopologyBuildContext,
  TopologyNode,
} from './network.topology.utils.types';
import {
  IN_DEGREE_DECREMENT,
  INPUT_NODE_TYPE,
  ZERO_COUNT,
} from './network.topology.utils.types';

/**
 * Seed Kahn queue with input nodes and zero in-degree nodes.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function seedProcessingQueue(buildContext: TopologyBuildContext): void {
  for (const node of buildContext.network.nodes) {
    if (isQueueSeedNode(node, buildContext)) {
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
export function processKahnQueue(buildContext: TopologyBuildContext): void {
  while (buildContext.processingQueue.length > ZERO_COUNT) {
    const currentNode = takeNextQueueNode(buildContext.processingQueue);
    appendTopoNode(buildContext.topoOrder, currentNode);
    relaxOutgoingEdges(buildContext, currentNode);
  }
}

/**
 * Determine whether a node belongs in the initial queue.
 *
 * @param node Candidate node.
 * @param buildContext Mutable build context.
 * @returns True when node is input-type or has zero in-degree.
 */
function isQueueSeedNode(
  node: TopologyNode,
  buildContext: TopologyBuildContext,
): boolean {
  return isInputNode(node) || getInDegree(buildContext, node) === ZERO_COUNT;
}

/**
 * Test whether a node is an input node.
 *
 * @param node Candidate node.
 * @returns True when node type is input.
 */
function isInputNode(node: TopologyNode): boolean {
  return node.type === INPUT_NODE_TYPE;
}

/**
 * Read in-degree for a node with zero fallback.
 *
 * @param buildContext Mutable build context.
 * @param node Candidate node.
 * @returns In-degree value.
 */
function getInDegree(
  buildContext: TopologyBuildContext,
  node: TopologyNode,
): number {
  return buildContext.inDegreeByNode.get(node) ?? ZERO_COUNT;
}

/**
 * Shift and return the next queue node.
 *
 * @param processingQueue Queue of pending nodes.
 * @returns Next node.
 */
function takeNextQueueNode(processingQueue: TopologyNode[]): TopologyNode {
  return processingQueue.shift() as TopologyNode;
}

/**
 * Append one node to topological order output.
 *
 * @param topoOrder Accumulated topological order.
 * @param node Node to append.
 * @returns Void.
 */
function appendTopoNode(topoOrder: TopologyNode[], node: TopologyNode): void {
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
  currentNode: TopologyNode,
): void {
  for (const outgoingConnection of currentNode.connections.out) {
    if (isSelfConnection(outgoingConnection.to, currentNode)) {
      continue;
    }

    const remainingInDegree = decrementNodeInDegree(
      buildContext,
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
 * @param buildContext Mutable build context.
 * @param node Target node.
 * @returns Remaining in-degree after decrement.
 */
function decrementNodeInDegree(
  buildContext: TopologyBuildContext,
  node: TopologyNode,
): number {
  const currentInDegree = getInDegree(buildContext, node);
  const remainingInDegree = currentInDegree - IN_DEGREE_DECREMENT;
  buildContext.inDegreeByNode.set(node, remainingInDegree);
  return remainingInDegree;
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
