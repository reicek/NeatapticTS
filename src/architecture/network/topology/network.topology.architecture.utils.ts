import type Network from '../../network/network';
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
 * This function prefers factual sources over heuristics so downstream tooling
 * can rely on the descriptor while still receiving useful output for partially
 * specified runtime graphs.
 *
 * Resolution priority is intentionally explicit:
 * 1) node `layer` metadata (factual when present)
 * 2) graph-derived feed-forward depth layering (factual for acyclic graphs)
 * 3) hidden-node count fallback (heuristic inference)
 *
 * @example
 * ```ts
 * const descriptor = describeArchitecture(network);
 * // descriptor.hiddenLayerSizes -> [8, 4]
 * // descriptor.source -> 'layer-metadata' | 'graph-topology' | 'inferred'
 * ```
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
  const graphTopologyResult = resolveHiddenLayerSizesFromGraphTopology(
    runtimeNodes,
    runtimeConnections,
  );

  // Step 1: Resolve hidden-layer widths from explicit per-node layer metadata.
  const layerMetadataHiddenSizes =
    resolveHiddenLayerSizesFromLayerMetadata(runtimeNodes);
  if (layerMetadataHiddenSizes.length > 0) {
    return createArchitectureDescriptor(
      layerMetadataHiddenSizes,
      graphTopologyResult.hasCycles,
      'layer-metadata',
      runtimeNodes.length,
      runtimeConnections.length,
    );
  }

  // Step 2: Resolve hidden-layer widths from graph topology when acyclic.
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

type HydratedDescriptorNetwork = Network & {
  _serializedArchitectureDescriptor?: NetworkArchitectureDescriptor;
};

/**
 * Resolve the public architecture descriptor, preferring live graph facts and
 * falling back to hydrated serialization metadata only when the live result is
 * still purely inferred.
 *
 * This helper keeps the descriptor ownership story in one chapter: topology
 * owns the live analysis while serialization can optionally hydrate a cached
 * descriptor that remains safe to reuse when the runtime graph shape matches.
 *
 * @param network Runtime network instance.
 * @returns Public architecture descriptor for telemetry and UI consumers.
 */
export function resolveArchitectureDescriptor(
  network: Network,
): NetworkArchitectureDescriptor {
  const liveDescriptor = describeArchitecture(network);
  const hydratedNetwork = network as unknown as HydratedDescriptorNetwork;
  const hydratedDescriptor = hydratedNetwork._serializedArchitectureDescriptor;

  if (liveDescriptor.source !== 'inferred') {
    hydratedNetwork._serializedArchitectureDescriptor = liveDescriptor;
    return liveDescriptor;
  }

  if (isHydratedDescriptorCompatible(network, hydratedDescriptor)) {
    return hydratedDescriptor;
  }

  return liveDescriptor;
}

/**
 * Check whether hydrated descriptor metadata still matches the current graph shape.
 *
 * @param network Runtime network instance.
 * @param hydratedDescriptor Optional hydrated descriptor candidate.
 * @returns True when hydrated descriptor can safely stand in for the inferred result.
 */
function isHydratedDescriptorCompatible(
  network: Network,
  hydratedDescriptor: NetworkArchitectureDescriptor | undefined,
): hydratedDescriptor is NetworkArchitectureDescriptor {
  return (
    hydratedDescriptor != null &&
    hydratedDescriptor.hiddenLayerSizes.length > 0 &&
    hydratedDescriptor.totalNodes === network.nodes.length &&
    hydratedDescriptor.totalConnections === network.connections.length
  );
}

/**
 * Creates the final immutable descriptor shape used by telemetry and UI code.
 *
 * Keeping descriptor assembly in one place ensures every resolution strategy
 * returns the same payload contract and avoids accidental field drift.
 *
 * @param hiddenLayerSizes - Hidden-layer widths.
 * @param hasCycles - Whether cycles were detected.
 * @param source - Descriptor provenance.
 * @param totalNodes - Node count.
 * @param totalConnections - Connection count.
 * @returns Descriptor object.
 * @example
 * ```ts
 * const descriptor = createArchitectureDescriptor([6, 3], false, 'graph-topology', 14, 25);
 * // descriptor.totalNodes === 14
 * ```
 */
function createArchitectureDescriptor(
  hiddenLayerSizes: number[],
  hasCycles: boolean,
  source: NetworkArchitectureSource,
  totalNodes: number,
  totalConnections: number,
): NetworkArchitectureDescriptor {
  // Step 1: Return a stable descriptor payload shared by all resolution paths.
  return {
    hiddenLayerSizes,
    hasCycles,
    source,
    totalNodes,
    totalConnections,
  };
}

/**
 * Resolves hidden-layer widths from explicit `node.layer` metadata.
 *
 * This is treated as the most trustworthy source because layer assignment is
 * usually produced by architecture-aware builders and does not depend on
 * topological reconstruction.
 *
 * @param runtimeNodes - Runtime nodes.
 * @returns Hidden-layer widths from explicit node.layer metadata.
 * @example
 * ```ts
 * // Hidden nodes in layers 1, 1, and 2 -> [2, 1]
 * const sizes = resolveHiddenLayerSizesFromLayerMetadata(nodes);
 * ```
 */
function resolveHiddenLayerSizesFromLayerMetadata(
  runtimeNodes: RuntimeNodeLike[],
): number[] {
  // Step 1: Accumulate counts keyed by hidden layer index.
  const layerCounts = new Map<number, number>();

  runtimeNodes.forEach((runtimeNode) => {
    // Key note: input/output/constant nodes are not part of hidden topology.
    const runtimeNodeType = runtimeNode.type ?? '';
    if (
      runtimeNodeType === 'input' ||
      runtimeNodeType === 'output' ||
      runtimeNodeType === 'constant'
    ) {
      return;
    }

    // Key note: missing numeric layer metadata means the node cannot be grouped here.
    if (typeof runtimeNode.layer !== 'number') {
      return;
    }

    layerCounts.set(
      runtimeNode.layer,
      (layerCounts.get(runtimeNode.layer) ?? 0) + 1,
    );
  });

  // Step 2: Convert counts into a deterministic ordered width vector.
  return [...layerCounts.entries()]
    .toSorted(
      (leftLayerEntry, rightLayerEntry) =>
        leftLayerEntry[0] - rightLayerEntry[0],
    )
    .map((layerEntry) => layerEntry[1]);
}

/**
 * Derives hidden-layer widths from graph topology when no explicit layer
 * metadata is available.
 *
 * The method computes a topological depth model for acyclic graphs; cyclic
 * graphs are flagged and intentionally return no width inference because depth
 * is not well-defined in recurrent loops.
 *
 * @param runtimeNodes - Runtime nodes.
 * @param runtimeConnections - Runtime connections.
 * @returns Hidden-layer widths derived from acyclic topology and cycle flag.
 * @example
 * ```ts
 * const { hiddenLayerSizes, hasCycles } = resolveHiddenLayerSizesFromGraphTopology(nodes, edges);
 * ```
 */
function resolveHiddenLayerSizesFromGraphTopology(
  runtimeNodes: RuntimeNodeLike[],
  runtimeConnections: RuntimeConnectionLike[],
): { hiddenLayerSizes: number[]; hasCycles: boolean } {
  // Step 1: Normalize runtime entities into index-addressable graph data.
  const nodeByIndex = createNodeIndexMap(runtimeNodes);
  const directedEdges = createDirectedEdgeList(runtimeConnections, nodeByIndex);

  // Step 2: Guard against empty or disconnected graph material.
  if (nodeByIndex.size === 0 || directedEdges.length === 0) {
    return { hiddenLayerSizes: [], hasCycles: false };
  }

  // Step 3: Detect cycles and establish a valid topological ordering.
  const cycleCheck = resolveCycleStateAndTopoOrder(nodeByIndex, directedEdges);
  if (cycleCheck.hasCycles) {
    return { hiddenLayerSizes: [], hasCycles: true };
  }

  // Step 4: Compute node depths, then count hidden nodes at each depth.
  const nodeDepthByIndex = resolveNodeDepthByIndex(
    nodeByIndex,
    directedEdges,
    cycleCheck.topologicalOrder,
  );
  const hiddenCountsByDepth = resolveHiddenCountsByDepth(
    nodeByIndex,
    nodeDepthByIndex,
  );

  // Step 5: Emit a deterministic hidden-layer size vector (shallow to deep).
  const hiddenLayerSizes = [...hiddenCountsByDepth.entries()]
    .toSorted(
      (leftDepthEntry, rightDepthEntry) =>
        leftDepthEntry[0] - rightDepthEntry[0],
    )
    .map((depthEntry) => depthEntry[1]);

  return { hiddenLayerSizes, hasCycles: false };
}

/**
 * Builds a node lookup table keyed by stable index.
 *
 * Runtime objects may omit `index`; in that case the current array position is
 * used as a deterministic fallback to keep downstream graph logic total.
 *
 * @param runtimeNodes - Runtime nodes.
 * @returns Node map keyed by stable node index.
 * @example
 * ```ts
 * const nodeByIndex = createNodeIndexMap(nodes);
 * // nodeByIndex.get(0) -> first node or node with explicit index 0
 * ```
 */
function createNodeIndexMap(
  runtimeNodes: RuntimeNodeLike[],
): Map<number, RuntimeNodeLike> {
  // Step 1: Create the index map with explicit-index preference.
  const nodeByIndex = new Map<number, RuntimeNodeLike>();

  runtimeNodes.forEach((runtimeNode, runtimeNodeIndex) => {
    // Key note: explicit indices preserve runtime graph identity if provided.
    const stableNodeIndex =
      typeof runtimeNode.index === 'number'
        ? runtimeNode.index
        : runtimeNodeIndex;
    nodeByIndex.set(stableNodeIndex, runtimeNode);
  });

  return nodeByIndex;
}

/**
 * Produces a validated list of enabled directed edges.
 *
 * Invalid references, disabled connections, and self-loops are removed so the
 * remaining edge list can be consumed safely by cycle and depth algorithms.
 *
 * @param runtimeConnections - Runtime connections.
 * @param nodeByIndex - Indexed nodes.
 * @returns Valid directed edges.
 * @example
 * ```ts
 * const edges = createDirectedEdgeList(runtimeConnections, nodeByIndex);
 * // edges -> [{ fromIndex: 0, toIndex: 3 }, ...]
 * ```
 */
function createDirectedEdgeList(
  runtimeConnections: RuntimeConnectionLike[],
  nodeByIndex: Map<number, RuntimeNodeLike>,
): Array<{ fromIndex: number; toIndex: number }> {
  // Step 1: Normalize each connection into a strongly typed edge or null.
  return (
    runtimeConnections
      .map((runtimeConnection) => {
        // Key note: disabled connections are intentionally excluded from topology.
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
      // Step 2: Keep only validated edges.
      .filter(
        (edge): edge is { fromIndex: number; toIndex: number } => edge !== null,
      )
  );
}

/**
 * Resolves cycle presence and, when possible, returns a topological order
 * using Kahn's algorithm.
 *
 * A complete topological ordering implies an acyclic graph. If some nodes
 * remain unprocessed, at least one cycle exists.
 *
 * @param nodeByIndex - Indexed nodes.
 * @param directedEdges - Directed edges.
 * @returns Topological order and cycle status.
 * @example
 * ```ts
 * const { topologicalOrder, hasCycles } = resolveCycleStateAndTopoOrder(nodeByIndex, edges);
 * ```
 */
function resolveCycleStateAndTopoOrder(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  directedEdges: Array<{ fromIndex: number; toIndex: number }>,
): { topologicalOrder: number[]; hasCycles: boolean } {
  // Step 1: Initialize in-degree and outgoing adjacency structures.
  const incomingEdgeCountByNode = new Map<number, number>();
  const outgoingTargetsByNode = new Map<number, number[]>();

  nodeByIndex.forEach((_node, nodeIndex) => {
    incomingEdgeCountByNode.set(nodeIndex, 0);
    outgoingTargetsByNode.set(nodeIndex, []);
  });

  // Step 2: Materialize graph bookkeeping from validated edges.
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

  // Step 3: Seed traversal with all zero in-degree nodes.
  const traversalQueue: number[] = [];
  incomingEdgeCountByNode.forEach((incomingCount, nodeIndex) => {
    if (incomingCount === 0) {
      traversalQueue.push(nodeIndex);
    }
  });

  // Step 4: Consume queue while decrementing downstream in-degree counters.
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

  // Step 5: Incomplete traversal indicates one or more directed cycles.
  return {
    topologicalOrder,
    hasCycles: topologicalOrder.length !== nodeByIndex.size,
  };
}

/**
 * Computes node depth (feed-forward distance from inputs) for an acyclic graph.
 *
 * Depth assignment is parent-driven: each node depth is one plus the maximum
 * resolved parent depth. Nodes with no resolved parents are skipped.
 *
 * @param nodeByIndex - Indexed nodes.
 * @param directedEdges - Directed edges.
 * @param topologicalOrder - Acyclic topological order.
 * @returns Derived depth by node index.
 * @example
 * ```ts
 * const depthByNodeIndex = resolveNodeDepthByIndex(nodeByIndex, edges, topologicalOrder);
 * ```
 */
function resolveNodeDepthByIndex(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  directedEdges: Array<{ fromIndex: number; toIndex: number }>,
  topologicalOrder: number[],
): Map<number, number> {
  // Step 1: Build incoming adjacency so each node can inspect its parents.
  const incomingSourcesByNode = new Map<number, number[]>();
  nodeByIndex.forEach((_node, nodeIndex) => {
    incomingSourcesByNode.set(nodeIndex, []);
  });

  // Step 2: Populate incoming source lists from edge data.
  directedEdges.forEach((directedEdge) => {
    const incomingSources =
      incomingSourcesByNode.get(directedEdge.toIndex) ?? [];
    incomingSources.push(directedEdge.fromIndex);
    incomingSourcesByNode.set(directedEdge.toIndex, incomingSources);
  });

  // Step 3: Traverse nodes in topological order and assign depths.
  const depthByNodeIndex = new Map<number, number>();

  topologicalOrder.forEach((nodeIndex) => {
    const node = nodeByIndex.get(nodeIndex);
    if (!node) {
      return;
    }

    // Key note: input nodes define depth origin.
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

    // Key note: unresolved parent depths mean this node cannot be reliably layered yet.
    if (parentDepths.length === 0) {
      return;
    }

    depthByNodeIndex.set(nodeIndex, Math.max(...parentDepths) + 1);
  });

  return depthByNodeIndex;
}

/**
 * Aggregates hidden-node counts per derived depth.
 *
 * This is the final transformation before emitting architecture widths:
 * hidden nodes are grouped by depth and counted in insertion-safe maps.
 *
 * @param nodeByIndex - Indexed nodes.
 * @param depthByNodeIndex - Derived depths.
 * @returns Hidden-node counts by depth.
 * @example
 * ```ts
 * const hiddenCountsByDepth = resolveHiddenCountsByDepth(nodeByIndex, depthByNodeIndex);
 * ```
 */
function resolveHiddenCountsByDepth(
  nodeByIndex: Map<number, RuntimeNodeLike>,
  depthByNodeIndex: Map<number, number>,
): Map<number, number> {
  // Step 1: Count only hidden nodes that have a resolved depth.
  const hiddenCountsByDepth = new Map<number, number>();

  nodeByIndex.forEach((node, nodeIndex) => {
    // Key note: non-hidden node classes are excluded by design.
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

  // Step 2: Return depth buckets for deterministic upstream sorting.
  return hiddenCountsByDepth;
}

/**
 * Identifies whether a runtime node should be treated as hidden for topology
 * reconstruction and fallback inference.
 *
 * @param runtimeNode - Candidate node.
 * @returns True when node type is hidden.
 * @example
 * ```ts
 * if (isHiddenNode(node)) {
 *   // Include in hidden-layer counting
 * }
 * ```
 */
function isHiddenNode(runtimeNode: RuntimeNodeLike): boolean {
  // Step 1: Use explicit type tag semantics for hidden-node classification.
  return runtimeNode.type === 'hidden';
}
