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

  buildContext.processingQueue = sortNodesByStableTieBreak(
    buildContext.processingQueue,
  );
}

/**
 * Process queue until all available nodes are emitted.
 *
 * @param buildContext Mutable build context.
 * @returns Void.
 */
export function processKahnQueue(buildContext: TopologyBuildContext): void {
  while (buildContext.processingQueue.length > ZERO_COUNT) {
    const currentStep = takeNextQueueStep(buildContext.processingQueue);
    const nextProcessingQueue: TopologyNode[] = [];
    const activationStep: number[] = [];

    for (const currentNode of currentStep) {
      appendTopoNode(buildContext.topoOrder, currentNode);
      appendActivationStepNode(activationStep, currentNode);
      relaxOutgoingEdges(buildContext, currentNode, nextProcessingQueue);
    }

    appendActivationStep(buildContext.activationSteps, activationStep);
    buildContext.processingQueue = sortNodesByStableTieBreak(
      nextProcessingQueue,
    );
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
 * Take the full current Kahn wave from the processing queue.
 *
 * @param processingQueue Queue of pending nodes.
 * @returns Current zero-in-degree wave in deterministic order.
 */
function takeNextQueueStep(processingQueue: TopologyNode[]): TopologyNode[] {
  return processingQueue.splice(0, processingQueue.length);
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
 * Append one stable node id to the current activation wave.
 *
 * @param activationStep Current activation wave.
 * @param node Node to append.
 * @returns Void.
 */
function appendActivationStepNode(
  activationStep: number[],
  node: TopologyNode,
): void {
  activationStep.push(resolveStableNodeTieBreakValue(node));
}

/**
 * Append one completed activation wave to the cached schedule.
 *
 * @param activationSteps Accumulated activation waves.
 * @param activationStep Current activation wave.
 * @returns Void.
 */
function appendActivationStep(
  activationSteps: number[][],
  activationStep: number[],
): void {
  if (activationStep.length === ZERO_COUNT) {
    return;
  }

  activationSteps.push(activationStep);
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
  nextProcessingQueue: TopologyNode[],
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
      nextProcessingQueue.push(outgoingConnection.to);
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

/**
 * Sort one node collection by the deterministic activation tie-break.
 *
 * @param nodes Candidate nodes.
 * @returns Sorted node collection.
 */
export function sortNodesByStableTieBreak(nodes: TopologyNode[]): TopologyNode[] {
  return nodes.toSorted(compareNodesByStableTieBreak);
}

/**
 * Compare two nodes by stable activation tie-break order.
 *
 * @param leftNode First node.
 * @param rightNode Second node.
 * @returns Negative when left should run first, positive when right should run first.
 */
export function compareNodesByStableTieBreak(
  leftNode: TopologyNode,
  rightNode: TopologyNode,
): number {
  return (
    resolveStableNodeTieBreakValue(leftNode) -
    resolveStableNodeTieBreakValue(rightNode)
  );
}

/**
 * Resolve the deterministic activation tie-break scalar for one node.
 *
 * Stable gene ids are preferred. Node index remains a conservative fallback for
 * unusual fixtures that bypass ordinary node construction.
 *
 * @param node Candidate node.
 * @returns Deterministic scalar used for sorting and schedule emission.
 */
export function resolveStableNodeTieBreakValue(node: TopologyNode): number {
  if (typeof node.geneId === 'number' && Number.isFinite(node.geneId)) {
    return node.geneId;
  }

  if (typeof node.index === 'number' && Number.isFinite(node.index)) {
    return node.index;
  }

  return Number.MAX_SAFE_INTEGER;
}
