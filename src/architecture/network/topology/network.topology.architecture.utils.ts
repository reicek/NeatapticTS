import type Network from '../../network';
import type {
  NetworkArchitectureDescriptor,
  NetworkArchitectureSource,
} from '../network.types';

interface RuntimeNodeLike {
  index?: number;
  type?: string;
  layer?: number;
}

interface RuntimeConnectionLike {
  from?: { index?: number };
  to?: { index?: number };
  enabled?: boolean;
}

/**
 * Describes network architecture for diagnostics, telemetry, and UI rendering.
 *
 * Resolution priority is intentionally explicit:
 * 1) node `layer` metadata (factual when present)
 * 2) graph-derived feed-forward depth layering (factual for acyclic graphs)
 * 3) hidden-node count fallback (heuristic inference)
 *
 * @param network - Runtime network instance.
 * @returns Stable architecture descriptor.
 */
export function describeArchitecture(
  network: Network,
): NetworkArchitectureDescriptor {
  const runtimeNodes = (network.nodes ?? []) as RuntimeNodeLike[];
  const runtimeConnections = (network.connections ??
    []) as RuntimeConnectionLike[];

  // Step 1: Resolve hidden-layer widths from explicit per-node layer metadata.
  const layerMetadataHiddenSizes =
    resolveHiddenLayerSizesFromLayerMetadata(runtimeNodes);
  if (layerMetadataHiddenSizes.length > 0) {
    return createArchitectureDescriptor(
      layerMetadataHiddenSizes,
      false,
      'layer-metadata',
      runtimeNodes.length,
      runtimeConnections.length,
    );
  }

  // Step 2: Resolve hidden-layer widths from graph topology when acyclic.
  const graphTopologyResult = resolveHiddenLayerSizesFromGraphTopology(
    runtimeNodes,
    runtimeConnections,
  );
  if (graphTopologyResult.hiddenLayerSizes.length > 0) {
    return createArchitectureDescriptor(
      graphTopologyResult.hiddenLayerSizes,
      graphTopologyResult.hasCycles,
      'graph-topology',
      runtimeNodes.length,
      runtimeConnections.length,
    );
  }

  // Step 3: Fallback to hidden-node count inference.
  const hiddenNodeCount = runtimeNodes.filter(isHiddenNode).length;
  const inferredHiddenSizes = hiddenNodeCount > 0 ? [hiddenNodeCount] : [];

  return createArchitectureDescriptor(
    inferredHiddenSizes,
    graphTopologyResult.hasCycles,
    'inferred',
    runtimeNodes.length,
    runtimeConnections.length,
  );
}

/**
 * @param hiddenLayerSizes - Hidden-layer widths.
 * @param hasCycles - Whether cycles were detected.
 * @param source - Descriptor provenance.
 * @param totalNodes - Node count.
 * @param totalConnections - Connection count.
 * @returns Descriptor object.
 */
function createArchitectureDescriptor(
  hiddenLayerSizes: number[],
  hasCycles: boolean,
  source: NetworkArchitectureSource,
  totalNodes: number,
  totalConnections: number,
): NetworkArchitectureDescriptor {
  return {
    hiddenLayerSizes,
    hasCycles,
    source,
    totalNodes,
    totalConnections,
  };
}

/**
 * @param runtimeNodes - Runtime nodes.
 * @returns Hidden-layer widths from explicit node.layer metadata.
 */
function resolveHiddenLayerSizesFromLayerMetadata(
  runtimeNodes: RuntimeNodeLike[],
): number[] {
  const layerCounts = new Map<number, number>();

  runtimeNodes.forEach((runtimeNode) => {
    const runtimeNodeType = runtimeNode.type ?? '';
    if (
      runtimeNodeType === 'input' ||
      runtimeNodeType === 'output' ||
      runtimeNodeType === 'constant'
    ) {
      return;
    }

    if (typeof runtimeNode.layer !== 'number') {
      return;
    }

    layerCounts.set(
      runtimeNode.layer,
      (layerCounts.get(runtimeNode.layer) ?? 0) + 1,
    );
  });

  return [...layerCounts.entries()]
    .toSorted(
      (leftLayerEntry, rightLayerEntry) =>
        leftLayerEntry[0] - rightLayerEntry[0],
    )
    .map((layerEntry) => layerEntry[1]);
}

/**
 * @param runtimeNodes - Runtime nodes.
 * @param runtimeConnections - Runtime connections.
 * @returns Hidden-layer widths derived from acyclic topology and cycle flag.
 */
function resolveHiddenLayerSizesFromGraphTopology(
  runtimeNodes: RuntimeNodeLike[],
  runtimeConnections: RuntimeConnectionLike[],
): { hiddenLayerSizes: number[]; hasCycles: boolean } {
  const nodeByIndex = createNodeIndexMap(runtimeNodes);
  const directedEdges = createDirectedEdgeList(runtimeConnections, nodeByIndex);

  if (nodeByIndex.size === 0 || directedEdges.length === 0) {
    return { hiddenLayerSizes: [], hasCycles: false };
  }

  const cycleCheck = resolveCycleStateAndTopoOrder(nodeByIndex, directedEdges);
  if (cycleCheck.hasCycles) {
    return { hiddenLayerSizes: [], hasCycles: true };
  }

  const nodeDepthByIndex = resolveNodeDepthByIndex(
    nodeByIndex,
    directedEdges,
    cycleCheck.topologicalOrder,
  );
  const hiddenCountsByDepth = resolveHiddenCountsByDepth(
    nodeByIndex,
    nodeDepthByIndex,
  );

  const hiddenLayerSizes = [...hiddenCountsByDepth.entries()]
    .toSorted(
      (leftDepthEntry, rightDepthEntry) =>
        leftDepthEntry[0] - rightDepthEntry[0],
    )
    .map((depthEntry) => depthEntry[1]);

  return { hiddenLayerSizes, hasCycles: false };
}

/**
 * @param runtimeNodes - Runtime nodes.
 * @returns Node map keyed by stable node index.
 */
function createNodeIndexMap(
  runtimeNodes: RuntimeNodeLike[],
): Map<number, RuntimeNodeLike> {
  const nodeByIndex = new Map<number, RuntimeNodeLike>();

  runtimeNodes.forEach((runtimeNode, runtimeNodeIndex) => {
    const stableNodeIndex =
      typeof runtimeNode.index === 'number'
        ? runtimeNode.index
        : runtimeNodeIndex;
    nodeByIndex.set(stableNodeIndex, runtimeNode);
  });

  return nodeByIndex;
}

/**
 * @param runtimeConnections - Runtime connections.
 * @param nodeByIndex - Indexed nodes.
 * @returns Valid directed edges.
 */
function createDirectedEdgeList(
  runtimeConnections: RuntimeConnectionLike[],
  nodeByIndex: Map<number, RuntimeNodeLike>,
): Array<{ fromIndex: number; toIndex: number }> {
  return runtimeConnections
    .map((runtimeConnection) => {
      if (runtimeConnection.enabled === false) {
        return null;
      }

      const fromNodeIndex = runtimeConnection.from?.index;
      const toNodeIndex = runtimeConnection.to?.index;

      if (
        typeof fromNodeIndex !== 'number' ||
        typeof toNodeIndex !== 'number'
      ) {
        return null;
      }

      if (!nodeByIndex.has(fromNodeIndex) || !nodeByIndex.has(toNodeIndex)) {
        return null;
      }

      if (fromNodeIndex === toNodeIndex) {
        return null;
      }

      return { fromIndex: fromNodeIndex, toIndex: toNodeIndex };
    })
    .filter(
      (edge): edge is { fromIndex: number; toIndex: number } => edge !== null,
    );
}

/**
 * @param nodeByIndex - Indexed nodes.
 * @param directedEdges - Directed edges.
 * @returns Topological order and cycle status.
 */
function resolveCycleStateAndTopoOrder(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  directedEdges: Array<{ fromIndex: number; toIndex: number }>,
): { topologicalOrder: number[]; hasCycles: boolean } {
  const incomingEdgeCountByNode = new Map<number, number>();
  const outgoingTargetsByNode = new Map<number, number[]>();

  nodeByIndex.forEach((_node, nodeIndex) => {
    incomingEdgeCountByNode.set(nodeIndex, 0);
    outgoingTargetsByNode.set(nodeIndex, []);
  });

  directedEdges.forEach((directedEdge) => {
    incomingEdgeCountByNode.set(
      directedEdge.toIndex,
      (incomingEdgeCountByNode.get(directedEdge.toIndex) ?? 0) + 1,
    );
    const outgoingTargets =
      outgoingTargetsByNode.get(directedEdge.fromIndex) ?? [];
    outgoingTargets.push(directedEdge.toIndex);
    outgoingTargetsByNode.set(directedEdge.fromIndex, outgoingTargets);
  });

  const traversalQueue: number[] = [];
  incomingEdgeCountByNode.forEach((incomingCount, nodeIndex) => {
    if (incomingCount === 0) {
      traversalQueue.push(nodeIndex);
    }
  });

  const topologicalOrder: number[] = [];
  while (traversalQueue.length > 0) {
    const currentNodeIndex = traversalQueue.shift();
    if (typeof currentNodeIndex !== 'number') {
      continue;
    }

    topologicalOrder.push(currentNodeIndex);
    const outgoingTargets = outgoingTargetsByNode.get(currentNodeIndex) ?? [];
    outgoingTargets.forEach((targetNodeIndex) => {
      const nextIncomingCount =
        (incomingEdgeCountByNode.get(targetNodeIndex) ?? 0) - 1;
      incomingEdgeCountByNode.set(targetNodeIndex, nextIncomingCount);
      if (nextIncomingCount === 0) {
        traversalQueue.push(targetNodeIndex);
      }
    });
  }

  return {
    topologicalOrder,
    hasCycles: topologicalOrder.length !== nodeByIndex.size,
  };
}

/**
 * @param nodeByIndex - Indexed nodes.
 * @param directedEdges - Directed edges.
 * @param topologicalOrder - Acyclic topological order.
 * @returns Derived depth by node index.
 */
function resolveNodeDepthByIndex(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  directedEdges: Array<{ fromIndex: number; toIndex: number }>,
  topologicalOrder: number[],
): Map<number, number> {
  const incomingSourcesByNode = new Map<number, number[]>();
  nodeByIndex.forEach((_node, nodeIndex) => {
    incomingSourcesByNode.set(nodeIndex, []);
  });

  directedEdges.forEach((directedEdge) => {
    const incomingSources =
      incomingSourcesByNode.get(directedEdge.toIndex) ?? [];
    incomingSources.push(directedEdge.fromIndex);
    incomingSourcesByNode.set(directedEdge.toIndex, incomingSources);
  });

  const depthByNodeIndex = new Map<number, number>();

  topologicalOrder.forEach((nodeIndex) => {
    const node = nodeByIndex.get(nodeIndex);
    if (!node) {
      return;
    }

    if (node.type === 'input') {
      depthByNodeIndex.set(nodeIndex, 0);
      return;
    }

    const incomingSources = incomingSourcesByNode.get(nodeIndex) ?? [];
    const parentDepths = incomingSources
      .map((sourceNodeIndex) => depthByNodeIndex.get(sourceNodeIndex))
      .filter(
        (parentDepth): parentDepth is number => typeof parentDepth === 'number',
      );

    if (parentDepths.length === 0) {
      return;
    }

    depthByNodeIndex.set(nodeIndex, Math.max(...parentDepths) + 1);
  });

  return depthByNodeIndex;
}

/**
 * @param nodeByIndex - Indexed nodes.
 * @param depthByNodeIndex - Derived depths.
 * @returns Hidden-node counts by depth.
 */
function resolveHiddenCountsByDepth(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  depthByNodeIndex: Map<number, number>,
): Map<number, number> {
  const hiddenCountsByDepth = new Map<number, number>();

  nodeByIndex.forEach((node, nodeIndex) => {
    if (!isHiddenNode(node)) {
      return;
    }

    const nodeDepth = depthByNodeIndex.get(nodeIndex);
    if (typeof nodeDepth !== 'number') {
      return;
    }

    hiddenCountsByDepth.set(
      nodeDepth,
      (hiddenCountsByDepth.get(nodeDepth) ?? 0) + 1,
    );
  });

  return hiddenCountsByDepth;
}

/**
 * @param runtimeNode - Candidate node.
 * @returns True when node type is hidden.
 */
function isHiddenNode(runtimeNode: RuntimeNodeLike): boolean {
  return runtimeNode.type === 'hidden';
}
