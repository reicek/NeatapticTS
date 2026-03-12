import Network from '../../../../../src/architecture/network';
import type {
  VisualNetworkConnectionLike,
  VisualNetworkNodeLike,
} from '../browser-entry.types';

/**
 * Topology resolution helpers for the browser network view.
 *
 * These helpers answer a key visualization question: how should the current
 * network be partitioned into ordered layers so layout and architecture labels
 * stay meaningful even when some metadata is missing?
 */

/**
 * Resolves layered node groups for network-view layout and rendering.
 *
 * Educational note:
 * Layer grouping is a network-view concern because it drives sizing, node
 * placement, and architecture presentation. Visualization code can still reuse
 * the result, but this helper now lives with the module that owns layout.
 *
 * The resolver prefers explicit layer metadata when it exists, then falls back
 * to a topology-derived depth estimate so even loosely structured networks can
 * still be drawn in an intelligible left-to-right order.
 *
 * @param network - Runtime network instance.
 * @param inputSize - Input count fallback.
 * @param outputSize - Output count fallback.
 * @returns Layered nodes for rendering.
 */
export function resolveNetworkVisualizationLayers(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): VisualNetworkNodeLike[][] {
  // Step 1: Build fallback layers when no runtime network is available.
  if (!network) {
    return [
      Array.from({ length: inputSize }, (_unusedValue, inputNodeIndex) => ({
        index: inputNodeIndex,
        type: 'input',
        bias: 0,
      })),
      Array.from({ length: outputSize }, (_unusedValue, outputNodeIndex) => ({
        index: inputSize + outputNodeIndex,
        type: 'output',
        bias: 0,
      })),
    ];
  }

  // Step 2: Normalize runtime node records into the view-layer shape.
  const runtimeNodes = (
    (network.nodes ?? []) as Array<{
      index?: number;
      type?: string;
      bias?: number;
      layer?: number;
    }>
  ).map((runtimeNode, fallbackNodeIndex) => ({
    index:
      typeof runtimeNode.index === 'number'
        ? runtimeNode.index
        : fallbackNodeIndex,
    type: runtimeNode.type ?? 'hidden',
    bias: runtimeNode.bias ?? 0,
    layer: runtimeNode.layer,
  }));

  // Step 3: Resolve input, hidden, and output partitions.
  const inputAndConstantNodes = runtimeNodes
    .filter(
      (runtimeNode) =>
        runtimeNode.type === 'input' || runtimeNode.type === 'constant',
    )
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
  const outputNodes = runtimeNodes
    .filter((runtimeNode) => runtimeNode.type === 'output')
    .toSorted((leftNode, rightNode) => leftNode.index - rightNode.index);
  const hiddenNodes = runtimeNodes.filter(
    (runtimeNode) =>
      runtimeNode.type !== 'input' &&
      runtimeNode.type !== 'constant' &&
      runtimeNode.type !== 'output',
  );

  // Step 4: Prefer layer metadata, then fall back to topology-derived depth.
  const hiddenLayersByMetadata = groupHiddenNodesByLayerMetadata(hiddenNodes);
  const hiddenLayers =
    hiddenLayersByMetadata.length > 0
      ? hiddenLayersByMetadata
      : groupHiddenNodesByTopology(network, runtimeNodes, hiddenNodes);
  const layeredNodes = [
    inputAndConstantNodes,
    ...hiddenLayers,
    outputNodes,
  ].filter((layerNodes) => layerNodes.length > 0);

  return layeredNodes.length > 0 ? layeredNodes : [runtimeNodes];
}

function groupHiddenNodesByLayerMetadata(
  hiddenNodes: VisualNetworkNodeLike[],
): VisualNetworkNodeLike[][] {
  const hiddenNodesWithLayer = hiddenNodes.filter(
    (hiddenNode) => typeof hiddenNode.layer === 'number',
  );
  if (hiddenNodesWithLayer.length === 0) {
    return [];
  }

  const nodesByLayer = new Map<number, VisualNetworkNodeLike[]>();
  hiddenNodesWithLayer.forEach((hiddenNode) => {
    const layerIndex = hiddenNode.layer as number;
    const existingLayerNodes = nodesByLayer.get(layerIndex) ?? [];
    existingLayerNodes.push(hiddenNode);
    nodesByLayer.set(layerIndex, existingLayerNodes);
  });

  return [...nodesByLayer.entries()]
    .toSorted(
      (leftLayerEntry, rightLayerEntry) =>
        leftLayerEntry[0] - rightLayerEntry[0],
    )
    .map((layerEntry) =>
      layerEntry[1].toSorted(
        (leftNode, rightNode) => leftNode.index - rightNode.index,
      ),
    );
}

function groupHiddenNodesByTopology(
  network: Network,
  runtimeNodes: VisualNetworkNodeLike[],
  hiddenNodes: VisualNetworkNodeLike[],
): VisualNetworkNodeLike[][] {
  if (hiddenNodes.length === 0) {
    return [];
  }

  const hiddenDepthByNodeIndex = resolveHiddenNodeDepthByTopology(
    runtimeNodes,
    (network.connections ?? []) as VisualNetworkConnectionLike[],
  );
  const nodesByDepth = new Map<number, VisualNetworkNodeLike[]>();

  hiddenNodes.forEach((hiddenNode) => {
    const depth = hiddenDepthByNodeIndex.get(hiddenNode.index) ?? 1;
    const existingDepthNodes = nodesByDepth.get(depth) ?? [];
    existingDepthNodes.push(hiddenNode);
    nodesByDepth.set(depth, existingDepthNodes);
  });

  return [...nodesByDepth.entries()]
    .toSorted(
      (leftDepthEntry, rightDepthEntry) =>
        leftDepthEntry[0] - rightDepthEntry[0],
    )
    .map((depthEntry) =>
      depthEntry[1].toSorted(
        (leftNode, rightNode) => leftNode.index - rightNode.index,
      ),
    );
}

function resolveHiddenNodeDepthByTopology(
  runtimeNodes: VisualNetworkNodeLike[],
  runtimeConnections: VisualNetworkConnectionLike[],
): Map<number, number> {
  const nodeByIndex = new Map<number, VisualNetworkNodeLike>(
    runtimeNodes.map((runtimeNode) => [runtimeNode.index, runtimeNode]),
  );
  const outgoingTargetsByNode = new Map<number, number[]>();
  const incomingEdgeCountByNode = new Map<number, number>();

  nodeByIndex.forEach((_runtimeNode, runtimeNodeIndex) => {
    outgoingTargetsByNode.set(runtimeNodeIndex, []);
    incomingEdgeCountByNode.set(runtimeNodeIndex, 0);
  });

  runtimeConnections.forEach((runtimeConnection) => {
    if (runtimeConnection.enabled === false) {
      return;
    }

    const fromNodeIndex = runtimeConnection.from?.index;
    const toNodeIndex = runtimeConnection.to?.index;
    if (
      typeof fromNodeIndex !== 'number' ||
      typeof toNodeIndex !== 'number' ||
      !nodeByIndex.has(fromNodeIndex) ||
      !nodeByIndex.has(toNodeIndex) ||
      fromNodeIndex === toNodeIndex
    ) {
      return;
    }

    const outgoingTargets = outgoingTargetsByNode.get(fromNodeIndex) ?? [];
    outgoingTargets.push(toNodeIndex);
    outgoingTargetsByNode.set(fromNodeIndex, outgoingTargets);
    incomingEdgeCountByNode.set(
      toNodeIndex,
      (incomingEdgeCountByNode.get(toNodeIndex) ?? 0) + 1,
    );
  });

  const topologicalQueue = [...incomingEdgeCountByNode.entries()]
    .filter((incomingEntry) => incomingEntry[1] === 0)
    .map((incomingEntry) => incomingEntry[0]);
  const topologicalOrder: number[] = [];

  while (topologicalQueue.length > 0) {
    const currentNodeIndex = topologicalQueue.shift();
    if (typeof currentNodeIndex !== 'number') {
      continue;
    }

    topologicalOrder.push(currentNodeIndex);

    const outgoingTargets = outgoingTargetsByNode.get(currentNodeIndex) ?? [];
    outgoingTargets.forEach((targetNodeIndex) => {
      const remainingIncomingCount =
        (incomingEdgeCountByNode.get(targetNodeIndex) ?? 0) - 1;
      incomingEdgeCountByNode.set(targetNodeIndex, remainingIncomingCount);
      if (remainingIncomingCount === 0) {
        topologicalQueue.push(targetNodeIndex);
      }
    });
  }

  if (topologicalOrder.length !== nodeByIndex.size) {
    return new Map<number, number>(
      runtimeNodes
        .filter((runtimeNode) => runtimeNode.type === 'hidden')
        .map((runtimeNode) => [runtimeNode.index, 1]),
    );
  }

  const depthByNodeIndex = new Map<number, number>();
  runtimeNodes.forEach((runtimeNode) => {
    const baseDepth =
      runtimeNode.type === 'input' || runtimeNode.type === 'constant' ? 0 : 1;
    depthByNodeIndex.set(runtimeNode.index, baseDepth);
  });

  topologicalOrder.forEach((fromNodeIndex) => {
    const fromDepth = depthByNodeIndex.get(fromNodeIndex) ?? 0;
    const outgoingTargets = outgoingTargetsByNode.get(fromNodeIndex) ?? [];
    outgoingTargets.forEach((toNodeIndex) => {
      const nextDepth = Math.max(
        depthByNodeIndex.get(toNodeIndex) ?? 1,
        fromDepth + 1,
      );
      depthByNodeIndex.set(toNodeIndex, nextDepth);
    });
  });

  return new Map<number, number>(
    runtimeNodes
      .filter((runtimeNode) => runtimeNode.type === 'hidden')
      .map((runtimeNode) => [
        runtimeNode.index,
        depthByNodeIndex.get(runtimeNode.index) ?? 1,
      ]),
  );
}
