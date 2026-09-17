/**
 * Shared level-of-detail (LOD) network-visualization service.
 *
 * The full-detail network view draws every node and edge, which collapses FPS
 * once networks grow past a few hundred nodes. This module replaces that
 * full-detail path for dense networks with a cheap abstraction:
 *
 * - input and output shelves are always rendered in full so tooltips and
 *   output labels keep working;
 * - hidden nodes above a threshold are collapsed into a handful of density
 *   clusters;
 * - hovering a hidden node expands a deterministic 2-hop ego neighborhood so
 *   the user can still inspect local topology without redrawing the whole
 *   graph.
 *
 * The renderer stays host-agnostic: semantic input labels are threaded in as
 * plain {@link InputLabelGroupDefinition} groups, and the orchestrator in
 * `network-view/network-view.ts` branches to this service when the settings
 * bag enables LOD for a dense network.
 */

import type Network from '../../../src/architecture/network';
import {
  NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  NETWORK_LOD_CANVAS_BACKGROUND,
  NETWORK_LOD_CLUSTER_CONNECTION_DEFAULT_OPACITY,
  NETWORK_LOD_CLUSTER_CONNECTION_RGB,
  NETWORK_LOD_CLUSTER_HEIGHT_PX,
  NETWORK_LOD_CLUSTER_WIDTH_PX,
  NETWORK_LOD_CONNECTION_LINE_WIDTH_PX,
  NETWORK_LOD_EGO_CONNECTION_DEFAULT_OPACITY,
  NETWORK_LOD_EGO_CONNECTION_RGB,
  NETWORK_LOD_EGO_DETAIL_FILL,
  NETWORK_LOD_EGO_LABEL_WIDTH_PX,
  NETWORK_LOD_GRAPH_BOTTOM_PADDING_PX,
  NETWORK_LOD_GRAPH_TOP_PADDING_PX,
  NETWORK_LOD_HIDDEN_CLUSTER_COUNT,
  NETWORK_LOD_HIDDEN_CLUSTER_FILL,
  NETWORK_LOD_HOVER_MAX_LOCAL_NODES,
  NETWORK_LOD_INPUT_LABEL_BAND_WIDTH_PX,
  NETWORK_LOD_INPUT_LABEL_LEFT_PADDING_PX,
  NETWORK_LOD_INPUT_LEFT_RESERVE_PX,
  NETWORK_LOD_INPUT_NODE_FILL,
  NETWORK_LOD_NODE_HEIGHT_PX,
  NETWORK_LOD_NODE_STROKE_COLOR,
  NETWORK_LOD_NODE_STROKE_WIDTH_PX,
  NETWORK_LOD_NODE_WIDTH_PX,
  NETWORK_LOD_OUTPUT_NODE_FILL,
  NETWORK_LOD_OUTPUT_RIGHT_RESERVE_PX,
} from './network-visualization.constants';
import type {
  NetworkHiddenColumnLabelScene,
  NetworkInputDescriptionScene,
  NetworkInputGroupLabelBandScene,
  NetworkVisualizationPositionedScene,
  PositionedNetworkNodeLike,
} from './network-visualization.types';
import {
  resolveNetworkArchitectureLabel,
  type NetworkVisualizationResolvedFrame,
} from './network-view/network-view';
import type { InputLabelGroupDefinition } from './network-view/network-view.types';
import { resolveNetworkVisualizationColorScales } from './visualization/visualization.colors.utils';
import type { WeightedConnectionLayerStyle } from './visualization/visualization.draw.service';

/** Node rectangle dimensions used by the LOD shelf nodes. */
const LOD_NODE_DIMENSIONS = {
  widthPx: NETWORK_LOD_NODE_WIDTH_PX,
  heightPx: NETWORK_LOD_NODE_HEIGHT_PX,
};

/** Cluster marker dimensions used by the LOD hidden-density clusters. */
const LOD_CLUSTER_DIMENSIONS = {
  widthPx: NETWORK_LOD_CLUSTER_WIDTH_PX,
  heightPx: NETWORK_LOD_CLUSTER_HEIGHT_PX,
};

/** Branded resolved frame produced by the shared LOD renderer. */
export interface NetworkVisualizationLODResolvedFrame
  extends NetworkVisualizationResolvedFrame {
  __networkLod: true;
  sourceNetwork: Network;
  hoverMaxLocalNodes: number;
}

/** Optional tuning bag for the shared LOD renderer. */
export interface NetworkVisualizationLODFrameOptions {
  /** Number of abstract hidden clusters; defaults to NETWORK_LOD_HIDDEN_CLUSTER_COUNT. */
  clusterCount?: number;
  /** Maximum local nodes when expanding a hovered hidden node; defaults to NETWORK_LOD_HOVER_MAX_LOCAL_NODES. */
  hoverMaxLocalNodes?: number;
  /** Optional semantic input-label groups threaded into the LOD input shelf. */
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[];
  /** Optional canvas background override; defaults to NETWORK_LOD_CANVAS_BACKGROUND. */
  canvasBackground?: string;
}

/** Geometry box shared by LOD scene resolution and drawing. */
interface LodGeometry {
  graphLeftPx: number;
  graphRightPx: number;
  graphTopPx: number;
  graphBottomPx: number;
  graphWidthPx: number;
  graphHeightPx: number;
  centerXPx: number;
}

/** Hovered ego-neighborhood overlay produced for one LOD draw pass. */
interface EgoDetailScene {
  positionedNodes: PositionedNetworkNodeLike[];
  hiddenColumnLabelScenes: NetworkHiddenColumnLabelScene[];
  edges: Array<{ fromIndex: number; toIndex: number }>;
}

/**
 * Decide whether a network is large enough to warrant the LOD abstraction.
 *
 * @example
 * ```ts
 * if (shouldUseNetworkLOD(network, NETWORK_LOD_HIDDEN_NODE_THRESHOLD)) {
 *   frame = resolveNetworkLODFrame(context, network, 70, 2);
 * }
 * ```
 *
 * @param network - Network to inspect, or `undefined`.
 * @param hiddenNodeThreshold - Hidden-node count above which LOD should activate.
 * @returns `true` when LOD should be used.
 */
export function shouldUseNetworkLOD(
  network: Network | undefined,
  hiddenNodeThreshold: number,
): boolean {
  // Step 1: Fall back to the full-detail path when no network is active.
  if (!network) {
    return false;
  }

  // Step 2: Count hidden nodes so the threshold decision stays payload-driven.
  let hiddenNodeCount = 0;
  for (const node of network.nodes) {
    if (node.type === 'hidden') {
      hiddenNodeCount += 1;
    }
  }

  // Step 3: Activate the LOD abstraction above the configured threshold.
  return hiddenNodeCount > hiddenNodeThreshold;
}

/**
 * Resolve a reusable LOD frame for a dense network.
 *
 * The frame keeps input and output shelves fully positioned, collapses hidden
 * nodes into abstract density clusters, and carries the tuning needed by
 * hover-only redraws.
 *
 * @example
 * ```ts
 * const frame = resolveNetworkLODFrame(canvasContext, denseNetwork, 70, 2);
 * ```
 *
 * @param context - Canvas 2D context used for sizing.
 * @param network - Network to abstract.
 * @param inputSize - Expected input node count.
 * @param outputSize - Expected output node count.
 * @param lodOptions - Optional LOD tuning (cluster count, hover budget, labels).
 * @returns Branded LOD frame.
 */
export function resolveNetworkLODFrame(
  context: CanvasRenderingContext2D,
  network: Network,
  inputSize: number,
  outputSize: number,
  lodOptions?: NetworkVisualizationLODFrameOptions,
): NetworkVisualizationLODResolvedFrame {
  // Step 1: Resolve the LOD tuning defaults for this frame.
  const clusterCount =
    lodOptions?.clusterCount ?? NETWORK_LOD_HIDDEN_CLUSTER_COUNT;
  const hoverMaxLocalNodes =
    lodOptions?.hoverMaxLocalNodes ?? NETWORK_LOD_HOVER_MAX_LOCAL_NODES;

  // Step 2: Resolve canvas, label, and color state from the active network.
  const canvasWidthPx = Math.max(1, Math.floor(context.canvas.width));
  const canvasHeightPx = Math.max(1, Math.floor(context.canvas.height));
  const architectureLabel = resolveNetworkArchitectureLabel(
    network,
    inputSize,
    outputSize,
  );
  const colorScales = resolveNetworkVisualizationColorScales(network);

  // Step 3: Resolve the abstract scene with full IO shelves and hidden clusters.
  const abstractScene = resolveAbstractLodPositionedScene(
    network,
    canvasWidthPx,
    canvasHeightPx,
    clusterCount,
    lodOptions?.inputLabelGroupDefinitions,
  );

  // Step 4: Fold the reusable LOD frame fields for hover-only redraws.
  return {
    canvasBackground:
      lodOptions?.canvasBackground ?? NETWORK_LOD_CANVAS_BACKGROUND,
    canvasWidthPx,
    canvasHeightPx,
    architectureLabel,
    colorScales,
    hideNetworkOverlays: false,
    positionedScene: abstractScene,
    positionByNodeIndex: buildPositionByNodeIndex(
      abstractScene.positionedNodes,
    ),
    runtimeConnections: [],
    hoverMaxLocalNodes,
    __networkLod: true,
    sourceNetwork: network,
  };
}

/**
 * Draw a previously resolved LOD frame, optionally expanding a hovered ego
 * graph.
 *
 * @example
 * ```ts
 * const scene = drawNetworkLODFromFrame(context, frame, [hoveredNodeIndex]);
 * ```
 *
 * @param context - Canvas 2D context to paint on.
 * @param resolvedFrame - LOD frame produced by {@link resolveNetworkLODFrame}.
 * @param hoveredNodeIndices - Optional host-owned hovered node ids.
 * @param connectionLayerStyle - Optional override for connection stroke visibility.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 */
export function drawNetworkLODFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationLODResolvedFrame,
  hoveredNodeIndices?: readonly number[],
  connectionLayerStyle?: Partial<WeightedConnectionLayerStyle>,
): NetworkVisualizationPositionedScene {
  // Step 1: Paint the LOD base canvas and the abstract cluster scene.
  const abstractScene = resolvedFrame.positionedScene;
  paintLodCanvasBase(context, resolvedFrame);
  drawAbstractLodNodesAndConnections(
    context,
    abstractScene,
    connectionLayerStyle,
  );

  // Step 2: Expand the hovered hidden node's 2-hop ego neighborhood when present.
  const geometry = resolveLodGeometry(
    resolvedFrame.canvasWidthPx,
    resolvedFrame.canvasHeightPx,
  );
  const detail = hoveredNodeIndices?.length
    ? resolveHoveredEgoDetail(
        resolvedFrame.sourceNetwork,
        hoveredNodeIndices,
        geometry,
        resolvedFrame.hoverMaxLocalNodes,
      )
    : undefined;

  if (detail) {
    // Step 3: Paint the ego overlay and fold the merged hover scene.
    drawEgoDetailOverlay(context, detail, connectionLayerStyle);

    return {
      positionedNodes: abstractScene.positionedNodes.concat(
        detail.positionedNodes,
      ),
      nodeDimensions: abstractScene.nodeDimensions,
      inputDescriptionScenes: abstractScene.inputDescriptionScenes,
      inputGroupLabelBandScenes: abstractScene.inputGroupLabelBandScenes,
      hiddenColumnLabelScenes: (
        abstractScene.hiddenColumnLabelScenes ?? []
      ).concat(detail.hiddenColumnLabelScenes),
    };
  }

  // Step 4: Return the abstract scene for host-owned hover hit testing.
  return abstractScene;
}

/**
 * Type guard for frames produced by {@link resolveNetworkLODFrame}.
 *
 * @example
 * ```ts
 * if (isNetworkLODResolvedFrame(frame)) {
 *   drawNetworkLODFromFrame(context, frame, hoveredNodeIndices);
 * }
 * ```
 *
 * @param resolvedFrame - Any resolved network-visualization frame.
 * @returns `true` when the frame was produced by the LOD renderer.
 */
export function isNetworkLODResolvedFrame(
  resolvedFrame: NetworkVisualizationResolvedFrame,
): resolvedFrame is NetworkVisualizationLODResolvedFrame {
  // Step 1: Check the LOD brand so draw routing stays unambiguous.
  return '__networkLod' in resolvedFrame && resolvedFrame.__networkLod === true;
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

/** Resolves the shared LOD geometry box from the canvas backing store. */
function resolveLodGeometry(
  canvasWidthPx: number,
  canvasHeightPx: number,
): LodGeometry {
  // Step 1: Resolve the vertical graph band inside the canvas padding.
  const graphTopPx = NETWORK_LOD_GRAPH_TOP_PADDING_PX;
  const graphBottomPx = Math.max(
    graphTopPx + 1,
    canvasHeightPx - NETWORK_LOD_GRAPH_BOTTOM_PADDING_PX,
  );
  const graphHeightPx = graphBottomPx - graphTopPx;

  // Step 2: Resolve the horizontal graph band with IO label reserves.
  const graphLeftPx =
    NETWORK_LOD_INPUT_LABEL_LEFT_PADDING_PX + NETWORK_LOD_INPUT_LEFT_RESERVE_PX;
  const graphRightPx = Math.max(
    graphLeftPx + 1,
    canvasWidthPx - NETWORK_LOD_OUTPUT_RIGHT_RESERVE_PX,
  );
  const graphWidthPx = graphRightPx - graphLeftPx;

  // Step 3: Fold the geometry box shared by scene resolution and drawing.
  return {
    graphLeftPx,
    graphRightPx,
    graphTopPx,
    graphBottomPx,
    graphWidthPx,
    graphHeightPx,
    centerXPx: graphLeftPx + graphWidthPx * 0.5,
  };
}

/** Resolves the abstract positioned scene for one dense network frame. */
function resolveAbstractLodPositionedScene(
  network: Network,
  canvasWidthPx: number,
  canvasHeightPx: number,
  clusterCount: number,
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): NetworkVisualizationPositionedScene {
  // Step 1: Resolve shelf and cluster positions from the shared geometry box.
  const geometry = resolveLodGeometry(canvasWidthPx, canvasHeightPx);
  const inputNodes = resolveLodInputPositionedNodes(network, geometry);
  const outputNodes = resolveLodOutputPositionedNodes(network, geometry);
  const hiddenClusterNodes = resolveLodHiddenClusterNodes(
    network,
    geometry,
    clusterCount,
  );

  // Step 2: Bind semantic input labels when the host provides groups.
  const inputDescriptionScenes = resolveLodInputDescriptionScenes(
    inputNodes,
    inputLabelGroupDefinitions,
  );
  const inputGroupLabelBandScenes = resolveLodInputGroupLabelBandScenes(
    inputNodes,
    inputLabelGroupDefinitions,
  );

  // Step 3: Fold the abstract positioned scene for the resolved frame.
  return {
    positionedNodes: inputNodes.concat(outputNodes, hiddenClusterNodes),
    nodeDimensions: LOD_NODE_DIMENSIONS,
    inputDescriptionScenes,
    inputGroupLabelBandScenes,
    hiddenColumnLabelScenes: resolveLodHiddenClusterLabelScenes(
      hiddenClusterNodes,
    ),
  };
}

/** Resolves the fully positioned input shelf for the LOD scene. */
function resolveLodInputPositionedNodes(
  network: Network,
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  // Step 1: Distribute the input shelf vertically along the left graph edge.
  const inputNodes = collectLodNodesByType(network, 'input');
  const inputCount = inputNodes.length;
  const xPx = geometry.graphLeftPx + LOD_NODE_DIMENSIONS.widthPx * 0.5;

  return inputNodes.map((inputNode, inputIndex) => {
    const yPx =
      geometry.graphTopPx +
      (inputCount > 1
        ? (inputIndex / (inputCount - 1)) * geometry.graphHeightPx
        : geometry.graphHeightPx * 0.5);

    return {
      node: {
        index: inputNode.index ?? inputIndex,
        type: inputNode.type,
        bias: Number(inputNode.bias ?? 0),
        activation: Number(inputNode.activation ?? 0),
      },
      xPx,
      yPx,
    };
  });
}

/** Resolves the fully positioned output shelf for the LOD scene. */
function resolveLodOutputPositionedNodes(
  network: Network,
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  // Step 1: Distribute the output shelf vertically along the right graph edge.
  const outputNodes = collectLodNodesByType(network, 'output');
  const outputCount = outputNodes.length;
  const xPx = geometry.graphRightPx - LOD_NODE_DIMENSIONS.widthPx * 0.5;

  return outputNodes.map((outputNode, outputIndex) => {
    const yPx =
      geometry.graphTopPx +
      (outputCount > 1
        ? (outputIndex / (outputCount - 1)) * geometry.graphHeightPx
        : geometry.graphHeightPx * 0.5);

    return {
      node: {
        index: outputNode.index ?? outputIndex,
        type: outputNode.type,
        bias: Number(outputNode.bias ?? 0),
        activation: Number(outputNode.activation ?? 0),
      },
      xPx,
      yPx,
    };
  });
}

/** Resolves one abstract hidden-density cluster marker per hidden slice. */
function resolveLodHiddenClusterNodes(
  network: Network,
  geometry: LodGeometry,
  clusterCount: number,
): PositionedNetworkNodeLike[] {
  // Step 1: Collect the hidden-node indices to aggregate into clusters.
  const hiddenIndices = collectLodNodesByType(network, 'hidden').map(
    (hiddenNode) => hiddenNode.index,
  );
  const hiddenCount = hiddenIndices.length;
  if (hiddenCount === 0) {
    return [];
  }

  // Step 2: Split the hidden indices into equal density slices.
  const resolvedClusterCount = Math.min(clusterCount, Math.max(1, hiddenCount));
  const clusters: PositionedNetworkNodeLike[] = [];

  for (
    let clusterIndex = 0;
    clusterIndex < resolvedClusterCount;
    clusterIndex += 1
  ) {
    const startOffset = Math.floor(
      (clusterIndex / resolvedClusterCount) * hiddenCount,
    );
    const endOffset = Math.floor(
      ((clusterIndex + 1) / resolvedClusterCount) * hiddenCount,
    );
    const clusterNodeIndices = hiddenIndices.slice(startOffset, endOffset);
    if (clusterNodeIndices.length === 0) {
      continue;
    }

    // Step 3: Place one representative marker per density slice.
    const representativeIndex = clusterNodeIndices[0] ?? clusterIndex;
    clusters.push({
      node: {
        index: representativeIndex,
        type: 'hidden',
        bias: 0,
        activation: 0,
        geneId: clusterIndex,
        layer: 1,
      },
      xPx: geometry.centerXPx,
      yPx:
        geometry.graphTopPx +
        ((clusterIndex + 0.5) / resolvedClusterCount) *
          geometry.graphHeightPx,
    });
  }

  return clusters;
}

/** Resolves one horizontal description chip per bound input node row. */
function resolveLodInputDescriptionScenes(
  inputNodes: readonly PositionedNetworkNodeLike[],
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): NetworkInputDescriptionScene[] {
  // Step 1: Skip the description column when the host provides no groups.
  const resolvedInputLabelGroupDefinitions = inputLabelGroupDefinitions ?? [];
  if (resolvedInputLabelGroupDefinitions.length === 0) {
    return [];
  }

  // Step 2: Bind one description chip per group node row.
  const scenes: NetworkInputDescriptionScene[] = [];
  let inputIndex = 0;

  for (const inputGroup of resolvedInputLabelGroupDefinitions) {
    for (
      let groupNodeIndex = 0;
      groupNodeIndex < inputGroup.nodeDescriptionDefinitions.length;
      groupNodeIndex += 1
    ) {
      const positionedNode = inputNodes[inputIndex];
      const nodeDescription =
        inputGroup.nodeDescriptionDefinitions[groupNodeIndex];
      if (positionedNode && nodeDescription) {
        scenes.push({
          labelLines: nodeDescription.labelLines,
          tooltipHeading: nodeDescription.tooltipHeading,
          tooltipBodyParagraphs: nodeDescription.tooltipBodyParagraphs,
          leftPx: 0,
          topPx: positionedNode.yPx - LOD_NODE_DIMENSIONS.heightPx * 0.5,
          widthPx:
            NETWORK_LOD_INPUT_LABEL_LEFT_PADDING_PX -
            NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
          heightPx: LOD_NODE_DIMENSIONS.heightPx,
          nodeIndex: positionedNode.node.index,
        });
      }
      inputIndex += 1;
    }
  }

  return scenes;
}

/** Resolves one vertical semantic band per provided input-label group. */
function resolveLodInputGroupLabelBandScenes(
  inputNodes: readonly PositionedNetworkNodeLike[],
  inputLabelGroupDefinitions?: readonly InputLabelGroupDefinition[],
): NetworkInputGroupLabelBandScene[] {
  // Step 1: Skip the band column when the host provides no groups.
  const resolvedInputLabelGroupDefinitions = inputLabelGroupDefinitions ?? [];
  if (resolvedInputLabelGroupDefinitions.length === 0) {
    return [];
  }

  // Step 2: Bind one vertical band per semantic input group.
  let inputIndex = 0;
  const scenes: NetworkInputGroupLabelBandScene[] = [];

  for (const inputGroup of resolvedInputLabelGroupDefinitions) {
    const groupNodes = inputNodes.slice(
      inputIndex,
      inputIndex + inputGroup.nodeDescriptionDefinitions.length,
    );
    const firstNode = groupNodes[0];
    const lastNode = groupNodes[groupNodes.length - 1];
    inputIndex += inputGroup.nodeDescriptionDefinitions.length;

    if (!firstNode || !lastNode) {
      continue;
    }

    const topPx = firstNode.yPx - LOD_NODE_DIMENSIONS.heightPx * 0.5;
    const bottomPx = lastNode.yPx + LOD_NODE_DIMENSIONS.heightPx * 0.5;

    scenes.push({
      label: inputGroup.label,
      labelLines: inputGroup.labelLines,
      tooltipHeading: inputGroup.tooltipHeading,
      tooltipBodyParagraphs: inputGroup.tooltipBodyParagraphs,
      leftPx: 0,
      topPx,
      widthPx: NETWORK_LOD_INPUT_LABEL_BAND_WIDTH_PX,
      heightPx: bottomPx - topPx,
      backgroundColor: inputGroup.backgroundColor,
      orientation: inputGroup.orientation,
      nodeIndices: groupNodes.map(
        (positionedNode) => positionedNode.node.index,
      ),
    });
  }

  return scenes;
}

/** Resolves one density-cluster chip per abstract hidden marker. */
function resolveLodHiddenClusterLabelScenes(
  hiddenClusterNodes: readonly PositionedNetworkNodeLike[],
): NetworkHiddenColumnLabelScene[] {
  // Step 1: Bind one cluster chip per abstract hidden marker.
  return hiddenClusterNodes.map((clusterNode, clusterIndex) => ({
    labelLines: [`HIDDEN ${clusterIndex + 1}`],
    tooltipHeading: `Hidden density cluster ${clusterIndex + 1}`,
    tooltipBodyParagraphs: [
      'Aggregated hidden-node density bin shown while the network is in level-of-detail mode.',
      'Hover any hidden node to expand its local 2-hop neighborhood.',
    ],
    leftPx: clusterNode.xPx - LOD_CLUSTER_DIMENSIONS.widthPx * 0.5,
    topPx: clusterNode.yPx - LOD_CLUSTER_DIMENSIONS.heightPx * 0.5,
    widthPx: LOD_CLUSTER_DIMENSIONS.widthPx,
    heightPx: LOD_CLUSTER_DIMENSIONS.heightPx,
    backgroundColor: NETWORK_LOD_HIDDEN_CLUSTER_FILL,
    nodeIndices: [clusterNode.node.index],
  }));
}

/** Filters the runtime node list by visual layer type. */
function collectLodNodesByType(
  network: Network,
  nodeType: 'input' | 'hidden' | 'output',
): Network['nodes'] {
  // Step 1: Keep only the nodes belonging to the requested visual layer.
  return network.nodes.filter(
    (networkNode) => networkNode.type === nodeType,
  ) as Network['nodes'];
}

// ---------------------------------------------------------------------------
// Ego-detail helpers
// ---------------------------------------------------------------------------

/** Resolves the hovered hidden node's capped 2-hop ego-neighborhood overlay. */
function resolveHoveredEgoDetail(
  network: Network,
  hoveredNodeIndices: readonly number[],
  geometry: LodGeometry,
  hoverMaxLocalNodes: number,
): EgoDetailScene | undefined {
  // Step 1: Only hidden nodes expand into ego neighborhoods.
  const hoveredIndex = hoveredNodeIndices[0];
  const hoveredNode = network.nodes.find(
    (node) => node.index === hoveredIndex,
  );
  if (!hoveredNode || hoveredNode.type !== 'hidden') {
    return undefined;
  }

  // Step 2: Resolve the deterministic 2-hop neighborhood around the hover.
  const localIndices = resolveTwoHopEgoIndices(
    network,
    hoveredIndex,
    hoverMaxLocalNodes,
  );
  if (localIndices.length === 0) {
    return undefined;
  }

  // Step 3: Keep only hidden nodes so IO hit-testing stays unambiguous.
  const nodeByIndex = new Map<number, Network['nodes'][number]>();
  for (const node of network.nodes) {
    if (typeof node.index === 'number') {
      nodeByIndex.set(node.index, node);
    }
  }
  const hiddenIndices = localIndices.filter((localIndex) => {
    const node = nodeByIndex.get(localIndex);
    return node?.type === 'hidden';
  });
  if (hiddenIndices.length === 0) {
    return undefined;
  }

  // Step 4: Lay the hidden detail nodes out on a deterministic grid.
  const hiddenPositions = positionEgoHiddenNodes(hiddenIndices, geometry);

  // Step 5: Collect deduplicated edges between co-located detail nodes.
  const detailNodeIndices = new Set<number>(hiddenIndices);
  const adjacency = buildLodAdjacencyMap(network);
  const detailEdges: Array<{ fromIndex: number; toIndex: number }> = [];
  for (const fromIndex of hiddenIndices) {
    for (const toIndex of adjacency.get(fromIndex) ?? []) {
      if (fromIndex < toIndex && detailNodeIndices.has(toIndex)) {
        detailEdges.push({ fromIndex, toIndex });
      }
    }
  }

  // Step 6: Fold the ego overlay scene for the current hover state.
  return {
    positionedNodes: hiddenPositions,
    hiddenColumnLabelScenes: [
      {
        labelLines: ['HOVER DETAIL'],
        tooltipHeading: 'Hovered hidden-node neighborhood',
        tooltipBodyParagraphs: [
          `Local 2-hop ego graph centered on hidden node ${hoveredIndex}.`,
          `Showing up to ${hoverMaxLocalNodes} connected nodes.`,
        ],
        leftPx: geometry.centerXPx - NETWORK_LOD_EGO_LABEL_WIDTH_PX * 0.5,
        topPx: geometry.graphTopPx,
        widthPx: NETWORK_LOD_EGO_LABEL_WIDTH_PX,
        heightPx: geometry.graphHeightPx,
        backgroundColor: NETWORK_LOD_EGO_DETAIL_FILL,
        nodeIndices: hiddenIndices,
      },
    ],
    edges: detailEdges,
  };
}

/** Walks a capped 2-hop breadth-first neighborhood from one node index. */
function resolveTwoHopEgoIndices(
  network: Network,
  startIndex: number,
  maxNodes: number,
): number[] {
  // Step 1: Build the bidirectional adjacency map once for the BFS walk.
  const adjacency = buildLodAdjacencyMap(network);
  const visitedOrder: number[] = [];
  const visited = new Set<number>();
  const queue: Array<{ index: number; distance: number }> = [
    { index: startIndex, distance: 0 },
  ];
  visited.add(startIndex);
  visitedOrder.push(startIndex);

  // Step 2: Walk two hops outward from the hovered node, capped at maxNodes.
  while (queue.length > 0 && visitedOrder.length < maxNodes) {
    const { index, distance } = queue.shift() as {
      index: number;
      distance: number;
    };
    if (distance >= 2) {
      continue;
    }

    const neighbors = adjacency.get(index) ?? [];
    for (const neighbor of neighbors) {
      if (visited.has(neighbor)) {
        continue;
      }
      visited.add(neighbor);
      visitedOrder.push(neighbor);
      queue.push({ index: neighbor, distance: distance + 1 });

      if (visitedOrder.length >= maxNodes) {
        break;
      }
    }
  }

  // Step 3: Return the deterministic visit order for grid placement.
  return visitedOrder;
}

/** Builds a bidirectional adjacency map from connections and self-connections. */
function buildLodAdjacencyMap(network: Network): Map<number, number[]> {
  // Step 1: Fold forward and backward edges into one adjacency map.
  const adjacency = new Map<number, number[]>();
  const addEdge = (fromIndex: number, toIndex: number): void => {
    if (fromIndex === toIndex) {
      return;
    }
    let fromNeighbors = adjacency.get(fromIndex);
    if (!fromNeighbors) {
      fromNeighbors = [];
      adjacency.set(fromIndex, fromNeighbors);
    }
    fromNeighbors.push(toIndex);
  };

  // Step 2: Thread explicit connections as bidirectional edges.
  for (const connection of network.connections as Array<{
    from?: { index?: number };
    to?: { index?: number };
  }>) {
    const fromIndex = connection.from?.index;
    const toIndex = connection.to?.index;
    if (typeof fromIndex === 'number' && typeof toIndex === 'number') {
      addEdge(fromIndex, toIndex);
      addEdge(toIndex, fromIndex);
    }
  }

  // Step 3: Thread self-connections so gated loops stay inspectable.
  for (const selfConnection of network.selfconns as Array<{
    from?: { index?: number };
    to?: { index?: number };
  }>) {
    const fromIndex = selfConnection.from?.index;
    const toIndex = selfConnection.to?.index;
    if (typeof fromIndex === 'number' && typeof toIndex === 'number') {
      addEdge(fromIndex, toIndex);
      addEdge(toIndex, fromIndex);
    }
  }

  return adjacency;
}

/** Places hidden ego-detail nodes on a deterministic grid inside the graph. */
function positionEgoHiddenNodes(
  hiddenIndices: readonly number[],
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  // Step 1: Fall back to an empty overlay when no hidden nodes need layout.
  if (hiddenIndices.length === 0) {
    return [];
  }

  // Step 2: Resolve the grid cell size from the graph area.
  const count = hiddenIndices.length;
  const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
  const rows = Math.max(1, Math.ceil(count / columns));
  const cellWidthPx = geometry.graphWidthPx / columns;
  const cellHeightPx = geometry.graphHeightPx / rows;
  const sortedIndices = hiddenIndices.toSorted((a, b) => a - b);

  // Step 3: Place each hidden node row-major so the layout stays deterministic.
  return sortedIndices.map((nodeIndex, sortedIndex) => {
    const column = sortedIndex % columns;
    const row = Math.floor(sortedIndex / columns);
    const xPx = geometry.graphLeftPx + (column + 0.5) * cellWidthPx;
    const yPx = geometry.graphTopPx + (row + 0.5) * cellHeightPx;

    return {
      node: {
        index: nodeIndex,
        type: 'hidden',
        bias: 0,
        activation: 0,
        geneId: nodeIndex,
        layer: 1,
      },
      xPx,
      yPx,
    };
  });
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------

/** Paints the LOD canvas base color for the resolved frame. */
function paintLodCanvasBase(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationLODResolvedFrame,
): void {
  // Step 1: Clear and paint the dense-network canvas base color.
  context.clearRect(
    0,
    0,
    resolvedFrame.canvasWidthPx,
    resolvedFrame.canvasHeightPx,
  );
  context.save();
  context.fillStyle =
    resolvedFrame.canvasBackground ?? NETWORK_LOD_CANVAS_BACKGROUND;
  context.fillRect(
    0,
    0,
    resolvedFrame.canvasWidthPx,
    resolvedFrame.canvasHeightPx,
  );
  context.restore();
}

/** Paints abstract shelf nodes, cluster markers, and density bundles. */
function drawAbstractLodNodesAndConnections(
  context: CanvasRenderingContext2D,
  scene: NetworkVisualizationPositionedScene,
  connectionLayerStyle?: Partial<WeightedConnectionLayerStyle>,
): void {
  // Step 1: Split the abstract scene into layer buckets for bundle drawing.
  const { positionedNodes, nodeDimensions } = scene;
  const clusterNodes = positionedNodes.filter(
    (positionedNode) => positionedNode.node.type === 'hidden',
  );
  const inputNodes = positionedNodes.filter(
    (positionedNode) => positionedNode.node.type === 'input',
  );
  const outputNodes = positionedNodes.filter(
    (positionedNode) => positionedNode.node.type === 'output',
  );

  // Step 2: Draw one faint connection bundle per cluster to suggest density.
  context.save();
  context.strokeStyle = `rgba(${NETWORK_LOD_CLUSTER_CONNECTION_RGB}, ${
    connectionLayerStyle?.defaultConnectionOpacity ??
    NETWORK_LOD_CLUSTER_CONNECTION_DEFAULT_OPACITY
  })`;
  context.lineWidth =
    connectionLayerStyle?.lineWidthPx ?? NETWORK_LOD_CONNECTION_LINE_WIDTH_PX;
  context.beginPath();

  for (const clusterNode of clusterNodes) {
    for (const inputNode of inputNodes) {
      context.moveTo(inputNode.xPx, inputNode.yPx);
      context.lineTo(clusterNode.xPx, clusterNode.yPx);
    }
    for (const outputNode of outputNodes) {
      context.moveTo(clusterNode.xPx, clusterNode.yPx);
      context.lineTo(outputNode.xPx, outputNode.yPx);
    }
  }

  context.stroke();
  context.restore();

  // Step 3: Draw the shelf and cluster node rectangles.
  for (const positionedNode of positionedNodes) {
    const { node, xPx, yPx } = positionedNode;
    const isCluster = node.type === 'hidden';
    const nodeWidthPx = isCluster
      ? LOD_CLUSTER_DIMENSIONS.widthPx
      : nodeDimensions.widthPx;
    const nodeHeightPx = isCluster
      ? LOD_CLUSTER_DIMENSIONS.heightPx
      : nodeDimensions.heightPx;

    context.save();
    context.beginPath();
    context.rect(
      xPx - nodeWidthPx * 0.5,
      yPx - nodeHeightPx * 0.5,
      nodeWidthPx,
      nodeHeightPx,
    );

    if (node.type === 'input') {
      context.fillStyle = NETWORK_LOD_INPUT_NODE_FILL;
    } else if (node.type === 'output') {
      context.fillStyle = NETWORK_LOD_OUTPUT_NODE_FILL;
    } else {
      context.fillStyle = NETWORK_LOD_HIDDEN_CLUSTER_FILL;
    }

    context.fill();
    context.strokeStyle = NETWORK_LOD_NODE_STROKE_COLOR;
    context.lineWidth = NETWORK_LOD_NODE_STROKE_WIDTH_PX;
    context.stroke();
    context.restore();
  }
}

/** Paints the hovered ego-neighborhood overlay above the abstract scene. */
function drawEgoDetailOverlay(
  context: CanvasRenderingContext2D,
  detail: EgoDetailScene,
  connectionLayerStyle?: Partial<WeightedConnectionLayerStyle>,
): void {
  // Step 1: Index the detail node positions for edge endpoint lookup.
  const positionByIndex = new Map<number, PositionedNetworkNodeLike>();
  for (const positionedNode of detail.positionedNodes) {
    positionByIndex.set(positionedNode.node.index, positionedNode);
  }

  // Step 2: Draw detail connections between co-located nodes.
  context.save();
  context.strokeStyle = `rgba(${NETWORK_LOD_EGO_CONNECTION_RGB}, ${
    connectionLayerStyle?.highlightConnectionOpacity ??
    NETWORK_LOD_EGO_CONNECTION_DEFAULT_OPACITY
  })`;
  context.lineWidth =
    connectionLayerStyle?.lineWidthPx ?? NETWORK_LOD_CONNECTION_LINE_WIDTH_PX;
  context.beginPath();

  for (const { fromIndex, toIndex } of detail.edges) {
    const fromPosition = positionByIndex.get(fromIndex);
    const toPosition = positionByIndex.get(toIndex);
    if (!fromPosition || !toPosition) {
      continue;
    }
    context.moveTo(fromPosition.xPx, fromPosition.yPx);
    context.lineTo(toPosition.xPx, toPosition.yPx);
  }

  context.stroke();
  context.restore();

  // Step 3: Draw the hidden detail nodes in the ego accent color.
  for (const positionedNode of detail.positionedNodes) {
    const { xPx, yPx } = positionedNode;

    context.save();
    context.beginPath();
    context.rect(
      xPx - LOD_NODE_DIMENSIONS.widthPx * 0.5,
      yPx - LOD_NODE_DIMENSIONS.heightPx * 0.5,
      LOD_NODE_DIMENSIONS.widthPx,
      LOD_NODE_DIMENSIONS.heightPx,
    );
    context.fillStyle = NETWORK_LOD_EGO_DETAIL_FILL;
    context.fill();
    context.strokeStyle = NETWORK_LOD_NODE_STROKE_COLOR;
    context.lineWidth = NETWORK_LOD_NODE_STROKE_WIDTH_PX;
    context.stroke();
    context.restore();
  }
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

/** Builds a node-index lookup map for resolved positioned nodes. */
function buildPositionByNodeIndex(
  positionedNodes: readonly PositionedNetworkNodeLike[],
): Map<number, PositionedNetworkNodeLike> {
  // Step 1: Materialize a lookup map so draw passes resolve endpoints quickly.
  return new Map<number, PositionedNetworkNodeLike>(
    positionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );
}
