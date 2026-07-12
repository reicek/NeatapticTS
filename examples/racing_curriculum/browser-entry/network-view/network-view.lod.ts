/**
 * Level-of-detail (LOD) renderer for dense racing-curriculum network diagrams.
 *
 * The shared Flappy Bird visualizer draws every node and edge, which collapses
 * FPS once networks grow past a few hundred nodes. This module replaces that
 * full-detail path for the racing demo with a cheap abstraction:
 *
 * - input and output shelves are always rendered in full so tooltips and
 *   output labels keep working;
 * - hidden nodes above a threshold are collapsed into a handful of density
 *   clusters;
 * - hovering a hidden node expands a deterministic 2-hop ego neighborhood so
 *   the user can still inspect local topology without redrawing the whole graph.
 */

import type Network from '../../../../src/architecture/network';
import type {
  NetworkHiddenColumnLabelScene,
  NetworkInputDescriptionScene,
  NetworkInputGroupLabelBandScene,
  NetworkVisualizationHoverState,
  NetworkVisualizationPositionedScene,
  PositionedNetworkNodeLike,
} from '../../../flappy_bird/browser-entry/browser-entry.visualization.types';
import {
  resolveNetworkArchitectureLabel as resolveSharedNetworkArchitectureLabel,
  type NetworkVisualizationResolvedFrame,
} from '../../../flappy_bird/browser-entry/network-view/network-view';
import { resolveNetworkVisualizationColorScales } from '../../../flappy_bird/browser-entry/visualization/visualization.colors.utils';
import type { WeightedConnectionLayerStyle } from '../../../flappy_bird/browser-entry/visualization/visualization.draw.service';
import {
  RACING_GROUP_COLORS,
  RACING_INPUT_GROUP_DEFS,
  RACING_INPUT_SIZE,
  RACING_LABEL_BAND_WIDTH_PX,
  RACING_LABEL_LEFT_PADDING_PX,
  RACING_NETWORK_LOD_HIDDEN_CLUSTER_COUNT,
  RACING_NETWORK_LOD_HIDDEN_NODE_THRESHOLD,
  RACING_NETWORK_LOD_HOVER_MAX_LOCAL_NODES,
  RACING_OUTPUT_SIZE,
} from './network-view.constants';

/** Frame brand used to route racing network-view drawing through the LOD path. */
type RacingLODResolvedFrame = NetworkVisualizationResolvedFrame & {
  __racingLod: true;
  sourceNetwork: Network;
};

export type { RacingLODResolvedFrame };

/** Decide whether the racing network view should use the LOD abstraction. */
export function shouldUseRacingNetworkLOD(
  network: Network | undefined,
): boolean {
  if (!network) {
    return false;
  }

  let hiddenNodeCount = 0;
  for (const node of network.nodes) {
    if (node.type === 'hidden') {
      hiddenNodeCount += 1;
    }
  }

  return hiddenNodeCount > RACING_NETWORK_LOD_HIDDEN_NODE_THRESHOLD;
}

/** Resolve a reusable LOD frame for a dense racing network. */
export function resolveRacingNetworkLODFrame(
  context: CanvasRenderingContext2D,
  network: Network,
): RacingLODResolvedFrame {
  const { widthPx: canvasWidthPx, heightPx: canvasHeightPx } =
    resolveLodCanvasDimensions(context);
  const architectureLabel = resolveSharedNetworkArchitectureLabel(
    network,
    RACING_INPUT_SIZE,
    RACING_OUTPUT_SIZE,
  );
  const colorScales = resolveNetworkVisualizationColorScales(network);
  const abstractScene = resolveAbstractPositionedScene(
    network,
    canvasWidthPx,
    canvasHeightPx,
  );

  return {
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
    __racingLod: true,
    sourceNetwork: network,
  };
}

/** Draw a previously resolved LOD frame, optionally expanding a hovered ego graph. */
export function drawRacingNetworkLODFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: RacingLODResolvedFrame,
  hoveredNodeIndices: readonly number[] | undefined,
  connectionLayerStyle?: Partial<WeightedConnectionLayerStyle>,
): NetworkVisualizationPositionedScene {
  const abstractScene = resolvedFrame.positionedScene;

  paintLodCanvasBase(context, resolvedFrame);
  drawAbstractNodesAndConnections(context, abstractScene, connectionLayerStyle);

  const hoverState = resolveRacingLodHoverState(hoveredNodeIndices);
  const geometry = resolveLodGeometry(
    resolvedFrame.canvasWidthPx,
    resolvedFrame.canvasHeightPx,
  );
  const detail = hoverState
    ? resolveHoveredEgoDetail(
        resolvedFrame.sourceNetwork,
        abstractScene,
        hoverState.hoveredNodeIndices ?? [],
        geometry,
      )
    : undefined;

  if (detail) {
    drawEgoDetailOverlay(context, detail, connectionLayerStyle);

    const mergedScene: NetworkVisualizationPositionedScene = {
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

    return mergedScene;
  }

  return abstractScene;
}

// ---------------------------------------------------------------------------
// Layout helpers
// ---------------------------------------------------------------------------

const LOD_NODE_DIMENSIONS = {
  widthPx: 12,
  heightPx: 12,
};

const LOD_CLUSTER_DIMENSIONS = {
  widthPx: 20,
  heightPx: 20,
};

const LOD_GRAPH_TOP_PADDING_PX = 20;
const LOD_GRAPH_BOTTOM_PADDING_PX = 20;
const LOD_OUTPUT_RIGHT_RESERVE_PX = 60;
const LOD_INPUT_LEFT_RESERVE_PX = 6;

interface LodGeometry {
  graphLeftPx: number;
  graphRightPx: number;
  graphTopPx: number;
  graphBottomPx: number;
  graphWidthPx: number;
  graphHeightPx: number;
  centerXPx: number;
}

function resolveLodCanvasDimensions(context: CanvasRenderingContext2D): {
  widthPx: number;
  heightPx: number;
} {
  return {
    widthPx: Math.max(1, Math.floor(context.canvas.width)),
    heightPx: Math.max(1, Math.floor(context.canvas.height)),
  };
}

function resolveLodGeometry(
  canvasWidthPx: number,
  canvasHeightPx: number,
): LodGeometry {
  const graphTopPx = LOD_GRAPH_TOP_PADDING_PX;
  const graphBottomPx = Math.max(
    graphTopPx + 1,
    canvasHeightPx - LOD_GRAPH_BOTTOM_PADDING_PX,
  );
  const graphHeightPx = graphBottomPx - graphTopPx;
  const graphLeftPx = RACING_LABEL_LEFT_PADDING_PX + LOD_INPUT_LEFT_RESERVE_PX;
  const graphRightPx = Math.max(
    graphLeftPx + 1,
    canvasWidthPx - LOD_OUTPUT_RIGHT_RESERVE_PX,
  );
  const graphWidthPx = graphRightPx - graphLeftPx;
  const centerXPx = graphLeftPx + graphWidthPx * 0.5;

  return {
    graphLeftPx,
    graphRightPx,
    graphTopPx,
    graphBottomPx,
    graphWidthPx,
    graphHeightPx,
    centerXPx,
  };
}

function resolveAbstractPositionedScene(
  network: Network,
  canvasWidthPx: number,
  canvasHeightPx: number,
): NetworkVisualizationPositionedScene {
  const geometry = resolveLodGeometry(canvasWidthPx, canvasHeightPx);
  const inputNodes = resolveInputPositionedNodes(network, geometry);
  const outputNodes = resolveOutputPositionedNodes(network, geometry);
  const hiddenClusterNodes = resolveHiddenClusterNodes(network, geometry);

  const positionedNodes = inputNodes.concat(outputNodes, hiddenClusterNodes);
  const inputDescriptionScenes = resolveInputDescriptionScenes(inputNodes);
  const inputGroupLabelBandScenes =
    resolveInputGroupLabelBandScenes(inputNodes);
  const hiddenColumnLabelScenes =
    resolveHiddenClusterLabelScenes(hiddenClusterNodes);

  return {
    positionedNodes,
    nodeDimensions: LOD_NODE_DIMENSIONS,
    inputDescriptionScenes,
    inputGroupLabelBandScenes,
    hiddenColumnLabelScenes,
  };
}

function resolveInputPositionedNodes(
  network: Network,
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  const inputNodes = collectNodesByType(network, 'input');
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

function resolveOutputPositionedNodes(
  network: Network,
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  const outputNodes = collectNodesByType(network, 'output');
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

function resolveHiddenClusterNodes(
  network: Network,
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  const hiddenIndices = collectNodesByType(network, 'hidden').map(
    (hiddenNode) => hiddenNode.index,
  );
  const hiddenCount = hiddenIndices.length;

  if (hiddenCount === 0) {
    return [];
  }

  const clusterCount = Math.min(
    RACING_NETWORK_LOD_HIDDEN_CLUSTER_COUNT,
    Math.max(1, hiddenCount),
  );
  const clusters: PositionedNetworkNodeLike[] = [];

  for (let clusterIndex = 0; clusterIndex < clusterCount; clusterIndex += 1) {
    const startRatio = clusterIndex / clusterCount;
    const endRatio = (clusterIndex + 1) / clusterCount;
    const startOffset = Math.floor(startRatio * hiddenCount);
    const endOffset = Math.floor(endRatio * hiddenCount);
    const clusterNodeIndices = hiddenIndices.slice(startOffset, endOffset);

    if (clusterNodeIndices.length === 0) {
      continue;
    }

    const representativeIndex = clusterNodeIndices[0] ?? clusterIndex;
    const xPx = geometry.centerXPx;
    const yPx =
      geometry.graphTopPx +
      ((clusterIndex + 0.5) / clusterCount) * geometry.graphHeightPx;

    clusters.push({
      node: {
        index: representativeIndex,
        type: 'hidden',
        bias: 0,
        activation: 0,
        geneId: clusterIndex,
        layer: 1,
      },
      xPx,
      yPx,
    });
  }

  return clusters;
}

function resolveInputDescriptionScenes(
  inputNodes: readonly PositionedNetworkNodeLike[],
): NetworkInputDescriptionScene[] {
  const scenes: NetworkInputDescriptionScene[] = [];
  let inputIndex = 0;

  for (const inputGroup of RACING_INPUT_GROUP_DEFS) {
    for (
      let groupNodeIndex = 0;
      groupNodeIndex < inputGroup.nodeCount;
      groupNodeIndex += 1
    ) {
      const positionedNode = inputNodes[inputIndex];
      const nodeDescription = inputGroup.nodeDescriptions[groupNodeIndex];
      if (positionedNode && nodeDescription) {
        scenes.push({
          labelLines: nodeDescription.labelLines,
          tooltipHeading: nodeDescription.tooltipHeading,
          tooltipBodyParagraphs: nodeDescription.tooltipBodyParagraphs,
          leftPx: 0,
          topPx: positionedNode.yPx - LOD_NODE_DIMENSIONS.heightPx * 0.5,
          widthPx: RACING_LABEL_LEFT_PADDING_PX - 6,
          heightPx: LOD_NODE_DIMENSIONS.heightPx,
          nodeIndex: positionedNode.node.index,
        });
      }
      inputIndex += 1;
    }
  }

  return scenes;
}

function resolveInputGroupLabelBandScenes(
  inputNodes: readonly PositionedNetworkNodeLike[],
): NetworkInputGroupLabelBandScene[] {
  let inputIndex = 0;
  const scenes: NetworkInputGroupLabelBandScene[] = [];

  for (const [groupIndex, inputGroup] of RACING_INPUT_GROUP_DEFS.entries()) {
    const groupNodes = inputNodes.slice(
      inputIndex,
      inputIndex + inputGroup.nodeCount,
    );
    const firstNode = groupNodes[0];
    const lastNode = groupNodes[groupNodes.length - 1];
    inputIndex += inputGroup.nodeCount;

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
      widthPx: RACING_LABEL_BAND_WIDTH_PX,
      heightPx: bottomPx - topPx,
      backgroundColor: RACING_GROUP_COLORS[groupIndex]?.bandFill ?? '#2bd9ff',
      orientation: 'vertical',
      nodeIndices: groupNodes.map(
        (positionedNode) => positionedNode.node.index,
      ),
    });
  }

  return scenes;
}

function resolveHiddenClusterLabelScenes(
  hiddenClusterNodes: readonly PositionedNetworkNodeLike[],
): NetworkHiddenColumnLabelScene[] {
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
    backgroundColor: '#9b59b6',
    nodeIndices: [clusterNode.node.index],
  }));
}

// ---------------------------------------------------------------------------
// Ego-detail helpers
// ---------------------------------------------------------------------------

interface EgoDetailScene {
  positionedNodes: PositionedNetworkNodeLike[];
  hiddenColumnLabelScenes: NetworkHiddenColumnLabelScene[];
  edges: Array<{ fromIndex: number; toIndex: number }>;
}

function resolveHoveredEgoDetail(
  network: Network,
  abstractScene: NetworkVisualizationPositionedScene,
  hoveredNodeIndices: readonly number[],
  geometry: LodGeometry,
): EgoDetailScene | undefined {
  const hoveredIndex = hoveredNodeIndices[0];
  const hoveredNode = network.nodes.find((node) => node.index === hoveredIndex);

  if (!hoveredNode || hoveredNode.type !== 'hidden') {
    return undefined;
  }

  const nodeByIndex = new Map<number, Network['nodes'][number]>();
  for (const node of network.nodes) {
    if (typeof node.index === 'number') {
      nodeByIndex.set(node.index, node);
    }
  }

  const localIndices = resolveTwoHopEgoIndices(
    network,
    hoveredIndex,
    RACING_NETWORK_LOD_HOVER_MAX_LOCAL_NODES,
  );

  if (localIndices.length === 0) {
    return undefined;
  }

  // Keep only hidden nodes in the detail overlay; IO nodes remain from the
  // abstract scene so hit-testing and hover counts stay unambiguous.
  const hiddenIndices = localIndices.filter((localIndex) => {
    const node = nodeByIndex.get(localIndex);
    return node?.type === 'hidden';
  });

  if (hiddenIndices.length === 0) {
    return undefined;
  }

  const hiddenPositions = positionEgoHiddenNodes(hiddenIndices, geometry);

  const detailNodeIndices = new Set<number>(hiddenIndices);
  const adjacency = buildAdjacencyMap(network);
  const detailEdges: Array<{ fromIndex: number; toIndex: number }> = [];
  for (const fromIndex of hiddenIndices) {
    for (const toIndex of adjacency.get(fromIndex) ?? []) {
      if (fromIndex < toIndex && detailNodeIndices.has(toIndex)) {
        detailEdges.push({ fromIndex, toIndex });
      }
    }
  }

  return {
    positionedNodes: hiddenPositions,
    hiddenColumnLabelScenes: [
      {
        labelLines: ['HOVER DETAIL'],
        tooltipHeading: 'Hovered hidden-node neighborhood',
        tooltipBodyParagraphs: [
          `Local 2-hop ego graph centered on hidden node ${hoveredIndex}.`,
          `Showing up to ${RACING_NETWORK_LOD_HOVER_MAX_LOCAL_NODES} connected nodes.`,
        ],
        leftPx: geometry.centerXPx - 40,
        topPx: geometry.graphTopPx,
        widthPx: 80,
        heightPx: geometry.graphHeightPx,
        backgroundColor: '#ff6b6b',
        nodeIndices: hiddenIndices,
      },
    ],
    edges: detailEdges,
  };
}

function resolveTwoHopEgoIndices(
  network: Network,
  startIndex: number,
  maxNodes: number,
): number[] {
  const adjacency = buildAdjacencyMap(network);
  const visitedOrder: number[] = [];
  const visited = new Set<number>();
  const queue: Array<{ index: number; distance: number }> = [
    { index: startIndex, distance: 0 },
  ];
  visited.add(startIndex);
  visitedOrder.push(startIndex);

  while (queue.length > 0 && visitedOrder.length < maxNodes) {
    const { index, distance } = queue.shift()!;
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

  return visitedOrder;
}

function buildAdjacencyMap(network: Network): Map<number, number[]> {
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

function positionEgoHiddenNodes(
  hiddenIndices: readonly number[],
  geometry: LodGeometry,
): PositionedNetworkNodeLike[] {
  if (hiddenIndices.length === 0) {
    return [];
  }

  const count = hiddenIndices.length;
  const columns = Math.max(1, Math.ceil(Math.sqrt(count)));
  const rows = Math.max(1, Math.ceil(count / columns));
  const cellWidthPx = geometry.graphWidthPx / columns;
  const cellHeightPx = geometry.graphHeightPx / rows;
  const sortedIndices = hiddenIndices.toSorted((a, b) => a - b);

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

function paintLodCanvasBase(
  context: CanvasRenderingContext2D,
  resolvedFrame: RacingLODResolvedFrame,
): void {
  context.clearRect(
    0,
    0,
    resolvedFrame.canvasWidthPx,
    resolvedFrame.canvasHeightPx,
  );
  context.save();
  context.fillStyle = '#080b14';
  context.fillRect(
    0,
    0,
    resolvedFrame.canvasWidthPx,
    resolvedFrame.canvasHeightPx,
  );
  context.restore();
}

function drawAbstractNodesAndConnections(
  context: CanvasRenderingContext2D,
  scene: NetworkVisualizationPositionedScene,
  connectionLayerStyle?: Partial<WeightedConnectionLayerStyle>,
): void {
  const { positionedNodes, nodeDimensions } = scene;

  // Draw one faint connection bundle per cluster to suggest density.
  const clusterNodes = positionedNodes.filter(
    (positionedNode) => positionedNode.node.type === 'hidden',
  );
  const inputNodes = positionedNodes.filter(
    (positionedNode) => positionedNode.node.type === 'input',
  );
  const outputNodes = positionedNodes.filter(
    (positionedNode) => positionedNode.node.type === 'output',
  );

  context.save();
  context.strokeStyle = `rgba(154, 89, 182, ${connectionLayerStyle?.defaultConnectionOpacity ?? 0.45})`;
  context.lineWidth = connectionLayerStyle?.lineWidthPx ?? 1.5;
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

  // Draw the nodes themselves.
  for (const positionedNode of positionedNodes) {
    const { node, xPx, yPx } = positionedNode;
    const isCluster = node.type === 'hidden';
    const nodeWidth = isCluster
      ? LOD_CLUSTER_DIMENSIONS.widthPx
      : nodeDimensions.widthPx;
    const nodeHeight = isCluster
      ? LOD_CLUSTER_DIMENSIONS.heightPx
      : nodeDimensions.heightPx;

    context.save();
    context.beginPath();
    context.rect(
      xPx - nodeWidth * 0.5,
      yPx - nodeHeight * 0.5,
      nodeWidth,
      nodeHeight,
    );

    if (node.type === 'input') {
      context.fillStyle = '#7dffd2';
    } else if (node.type === 'output') {
      context.fillStyle = '#8ad8ff';
    } else {
      context.fillStyle = '#9b59b6';
    }

    context.fill();
    context.strokeStyle = '#ffffff';
    context.lineWidth = 0.5;
    context.stroke();
    context.restore();
  }
}

function drawEgoDetailOverlay(
  context: CanvasRenderingContext2D,
  detail: EgoDetailScene,
  connectionLayerStyle?: Partial<WeightedConnectionLayerStyle>,
): void {
  const positionByIndex = new Map<number, PositionedNetworkNodeLike>();
  for (const positionedNode of detail.positionedNodes) {
    positionByIndex.set(positionedNode.node.index, positionedNode);
  }

  // Draw detail connections between co-located nodes.
  context.save();
  context.strokeStyle = `rgba(255, 107, 107, ${connectionLayerStyle?.highlightConnectionOpacity ?? 1})`;
  context.lineWidth = connectionLayerStyle?.lineWidthPx ?? 1.5;
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

  // Draw detail nodes.
  for (const positionedNode of detail.positionedNodes) {
    const { node, xPx, yPx } = positionedNode;
    if (node.type === 'input' || node.type === 'output') {
      continue;
    }

    context.save();
    context.beginPath();
    context.rect(
      xPx - LOD_NODE_DIMENSIONS.widthPx * 0.5,
      yPx - LOD_NODE_DIMENSIONS.heightPx * 0.5,
      LOD_NODE_DIMENSIONS.widthPx,
      LOD_NODE_DIMENSIONS.heightPx,
    );
    context.fillStyle = '#ff6b6b';
    context.fill();
    context.strokeStyle = '#ffffff';
    context.lineWidth = 0.5;
    context.stroke();
    context.restore();
  }
}

// ---------------------------------------------------------------------------
// Shared utilities
// ---------------------------------------------------------------------------

function buildPositionByNodeIndex(
  positionedNodes: readonly PositionedNetworkNodeLike[],
): Map<number, PositionedNetworkNodeLike> {
  return new Map<number, PositionedNetworkNodeLike>(
    positionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );
}

function collectNodesByType(
  network: Network,
  nodeType: string,
): Array<Network['nodes'][number]> {
  return network.nodes
    .filter((node) => node.type === nodeType)
    .toSorted((nodeA, nodeB) => (nodeA.index ?? -1) - (nodeB.index ?? -1));
}

function resolveRacingLodHoverState(
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationHoverState | undefined {
  if (!hoveredNodeIndices || hoveredNodeIndices.length === 0) {
    return undefined;
  }

  return {
    hoveredNodeIndices,
    animatedHoveredNodes: hoveredNodeIndices.map((nodeIndex) => ({
      nodeIndex,
      intensity: 1,
    })),
  };
}
