import Network from '../../../src/architecture/network';
import { clamp } from './browser-entry.math.utils';
import {
  drawBiasNodesLayer as drawBiasNodes,
  drawNetworkColorLegend,
  drawWeightedConnectionsLayer as drawWeightedConnections,
  resolveDefaultNetworkLegendLayout,
  resolveNetworkVisualizationLayers,
} from './browser-entry.visualization.utils';
import {
  FLAPPY_NETWORK_BASELINE_HEIGHT_PX,
  FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_LAYER_TARGET_GAP_PX,
  FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX,
  FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
  FLAPPY_NETWORK_MAX_HEIGHT_PX,
  FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX,
  FLAPPY_NETWORK_MAX_NODE_WIDTH_PX,
  FLAPPY_NETWORK_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX,
  FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
  FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX,
  FLAPPY_NETWORK_MIN_NODE_WIDTH_PX,
  FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX,
  FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
} from './browser-entry.constants';
import type {
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  PositionedNetworkNodeLike as PositionedNetworkNode,
  VisualNetworkConnectionLike,
  VisualNetworkNodeLike,
} from './browser-entry.types';

/**
 * Draws a complete, layer-based visualization of the active network.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Nothing.
 */
export function drawNetworkVisualization(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): void {
  // Step 1: Clear the canvas and paint visualization background.
  const canvasWidthPx = context.canvas.width;
  const canvasHeightPx = context.canvas.height;
  context.clearRect(0, 0, canvasWidthPx, canvasHeightPx);

  context.fillStyle = FLAPPY_UI_NETWORK_CANVAS_BACKGROUND;
  context.fillRect(0, 0, canvasWidthPx, canvasHeightPx);

  // Step 2: Resolve architecture label and base graph paddings.
  const architectureLabel = resolveNetworkArchitectureLabel(
    network,
    inputSize,
    outputSize,
  );

  const graphLeftPaddingPx = FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX;
  const graphTopPaddingPx = FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX;
  const graphRightPaddingPx = FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX;
  const graphBottomPaddingPx = FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX;
  const legendGraphGapPx = FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX;
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ?? canvasWidthPx;
  const hideNetworkOverlays =
    viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX;
  const nodeLayoutPaddingPx = FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX;
  const minimumLabelHeightPx = FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX;
  const minimumNodeInnerPaddingPx = FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX;
  const minimumNodeHeightPx =
    minimumLabelHeightPx + minimumNodeInnerPaddingPx * 2;
  const minimumNodeWidthPx = FLAPPY_NETWORK_MIN_NODE_WIDTH_PX;

  // Step 3: Resolve legend layout and reserve horizontal graph space around it.
  let adjustedGraphLeftPaddingPx = graphLeftPaddingPx;
  let adjustedGraphRightPaddingPx = graphRightPaddingPx;
  if (!hideNetworkOverlays) {
    const legendLayout = resolveDefaultNetworkLegendLayout(context);
    const legendMidpointPx =
      legendLayout.legendLeftPx + legendLayout.legendWidthPx * 0.5;
    if (legendMidpointPx >= canvasWidthPx * 0.5) {
      adjustedGraphRightPaddingPx = Math.max(
        graphRightPaddingPx,
        canvasWidthPx - legendLayout.legendLeftPx + legendGraphGapPx,
      );
    } else {
      adjustedGraphLeftPaddingPx = Math.max(
        graphLeftPaddingPx,
        legendLayout.legendLeftPx +
          legendLayout.legendWidthPx +
          legendGraphGapPx,
      );
    }
  }

  // Step 4: Resolve drawable graph bounds.
  const drawableWidthPx = Math.max(
    1,
    canvasWidthPx - adjustedGraphLeftPaddingPx - adjustedGraphRightPaddingPx,
  );
  const drawableHeightPx = Math.max(
    1,
    canvasHeightPx - graphTopPaddingPx - graphBottomPaddingPx,
  );

  // Step 5: Resolve network layers and dynamic node dimensions.
  const networkLayers = resolveNetworkVisualizationLayers(
    network,
    inputSize,
    outputSize,
  );
  const maxLayerNodeCount = Math.max(
    1,
    ...networkLayers.map((layerNodes) => layerNodes.length),
  );
  const layerCount = Math.max(1, networkLayers.length);
  const availableNodeStackHeightPx = Math.max(
    1,
    drawableHeightPx - nodeLayoutPaddingPx * 2,
  );
  const strictMaximumNodeHeightByFitPx = Math.max(
    4,
    availableNodeStackHeightPx / maxLayerNodeCount,
  );
  const effectiveMinimumNodeHeightPx = Math.min(
    minimumNodeHeightPx,
    strictMaximumNodeHeightByFitPx,
  );
  const nodeHeightPx = Math.max(
    effectiveMinimumNodeHeightPx,
    Math.min(
      FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX,
      (drawableHeightPx - nodeLayoutPaddingPx * 2) /
        Math.max(4, maxLayerNodeCount * 1.6),
      drawableWidthPx / Math.max(8, layerCount * 3.6),
      strictMaximumNodeHeightByFitPx,
    ),
  );
  const nodeWidthPx = Math.max(
    minimumNodeWidthPx,
    Math.min(
      FLAPPY_NETWORK_MAX_NODE_WIDTH_PX,
      nodeHeightPx * 2.15,
      drawableWidthPx / Math.max(4, layerCount * 1.45),
    ),
  );
  const nodeDimensions: NetworkNodeDimensions = {
    widthPx: nodeWidthPx,
    heightPx: nodeHeightPx,
  };

  // Step 6: Position nodes and build node-index lookup map.
  const positionedNodes = positionNetworkNodes(
    networkLayers,
    adjustedGraphLeftPaddingPx,
    graphTopPaddingPx,
    drawableWidthPx,
    drawableHeightPx,
    nodeLayoutPaddingPx,
    nodeDimensions,
  );
  const positionByNodeIndex = new Map<number, PositionedNetworkNode>(
    positionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );

  const runtimeConnections =
    ((network?.connections ?? []) as VisualNetworkConnectionLike[]) ?? [];

  // Step 7: Draw connections, nodes, and legend (with architecture text above it).
  drawWeightedConnections(context, runtimeConnections, positionByNodeIndex);
  drawBiasNodes(context, positionedNodes, nodeDimensions);
  drawNetworkColorLegend(context, architectureLabel);
}

/**
 * Resolves the recommended network visualization host height.
 *
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Recommended height in pixels.
 */
export function resolveNetworkVisualizationHeightPx(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): number {
  // Step 1: Resolve readability baselines and graph paddings.
  const minimumLabelHeightPx = FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX;
  const minimumNodeInnerPaddingPx = FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX;
  const minimumInterNodeGapPx = FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX;
  const minimumNodeSidePx =
    minimumLabelHeightPx + minimumNodeInnerPaddingPx * 2;
  const graphTopPaddingPx = FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX;
  const graphBottomPaddingPx = FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX;
  const graphInnerPaddingPx = FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX;
  const layers = resolveNetworkVisualizationLayers(
    network,
    inputSize,
    outputSize,
  );
  const layerCount = Math.max(1, layers.length);
  const maxLayerNodeCount = Math.max(
    1,
    ...layers.map((layerNodes) => layerNodes.length),
  );

  // Step 2: Compute topology-driven minimum readable height.
  const minimumReadableStackHeightPx =
    maxLayerNodeCount <= 1
      ? minimumNodeSidePx
      : maxLayerNodeCount * minimumNodeSidePx +
        (maxLayerNodeCount - 1) * minimumInterNodeGapPx;
  const topologyDrivenHeightPx =
    graphTopPaddingPx +
    graphBottomPaddingPx +
    graphInnerPaddingPx +
    minimumReadableStackHeightPx;

  // Step 3: Add complexity-based height adjustments and choose the max.
  const baselineHeightPx = FLAPPY_NETWORK_BASELINE_HEIGHT_PX;
  const nodeDensityHeightPx =
    Math.max(0, maxLayerNodeCount - 6) *
    FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX;
  const layerComplexityHeightPx =
    Math.max(0, layerCount - 3) *
    FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX;
  const recommendedHeightPx = Math.max(
    Math.ceil(topologyDrivenHeightPx * 1.45),
    baselineHeightPx + nodeDensityHeightPx + layerComplexityHeightPx,
  );

  // Step 4: Clamp final height to configured min/max bounds.
  return Math.floor(
    clamp(
      recommendedHeightPx,
      FLAPPY_NETWORK_MIN_HEIGHT_PX,
      FLAPPY_NETWORK_MAX_HEIGHT_PX,
    ),
  );
}

/**
 * Produces a concise neural-network architecture label.
 *
 * @param network - Network to describe.
 * @param inputSize - Configured input size.
 * @param outputSize - Configured output size.
 * @returns Readable architecture label.
 */
export function resolveNetworkArchitectureLabel(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): string {
  if (!network) {
    return formatArchitectureLabel(
      inputSize,
      '-',
      outputSize,
      inputSize + outputSize,
      0,
    );
  }

  const architectureDescriptor = network.describeArchitecture();
  const hiddenLayersLabel = resolveHiddenLayersLabel(
    architectureDescriptor.hiddenLayerSizes,
    architectureDescriptor.source,
  );

  return formatArchitectureLabel(
    inputSize,
    hiddenLayersLabel,
    outputSize,
    architectureDescriptor.totalNodes,
    architectureDescriptor.totalConnections,
  );
}

/**
 * Positions network nodes into drawable canvas coordinates.
 *
 * @param networkLayers - Resolved network layers.
 * @param leftPaddingPx - Left graph padding.
 * @param topPaddingPx - Top graph padding.
 * @param drawableWidthPx - Drawable graph width.
 * @param drawableHeightPx - Drawable graph height.
 * @param nodeLayoutPaddingPx - Inner graph padding.
 * @param nodeDimensions - Node dimensions.
 * @returns Positioned nodes.
 */
function positionNetworkNodes(
  networkLayers: VisualNetworkNodeLike[][],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensions,
): PositionedNetworkNode[] {
  // Step 1: Initialize positioning accumulators and reusable geometry values.
  const positionedNodes: PositionedNetworkNode[] = [];
  const lastLayerIndex = Math.max(0, networkLayers.length - 1);
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = nodeDimensions.heightPx * 0.5;
  const minimumNodeCenterYPx =
    topPaddingPx + nodeLayoutPaddingPx + halfNodeHeightPx;
  const maximumNodeCenterYPx =
    topPaddingPx + drawableHeightPx - nodeLayoutPaddingPx - halfNodeHeightPx;

  // Step 2: Distribute layers horizontally and nodes vertically per layer.
  networkLayers.forEach((layerNodes, layerIndex) => {
    const horizontalProgress =
      lastLayerIndex === 0 ? 0.5 : layerIndex / lastLayerIndex;
    const layerXPx =
      leftPaddingPx +
      nodeLayoutPaddingPx +
      halfNodeWidthPx +
      horizontalProgress *
        Math.max(
          1,
          drawableWidthPx - nodeLayoutPaddingPx * 2 - nodeDimensions.widthPx,
        );

    const layerNodeCount = Math.max(1, layerNodes.length);
    const availableLayerStackHeightPx = Math.max(
      1,
      drawableHeightPx - nodeLayoutPaddingPx * 2,
    );
    const baselineInterNodeGapPx =
      layerNodeCount <= 1
        ? 0
        : Math.max(
            0,
            (availableLayerStackHeightPx -
              nodeDimensions.heightPx * layerNodeCount) /
              (layerNodeCount - 1),
          );
    const preferredLayerInterNodeGapPx =
      layerIndex === 0
        ? FLAPPY_NETWORK_INPUT_LAYER_TARGET_GAP_PX
        : baselineInterNodeGapPx * 0.5;
    const maximumLayerInterNodeGapToFitPx =
      layerNodeCount <= 1
        ? 0
        : Math.max(
            0,
            (availableLayerStackHeightPx -
              nodeDimensions.heightPx * layerNodeCount) /
              (layerNodeCount - 1),
          );
    const resolvedLayerInterNodeGapPx = Math.min(
      preferredLayerInterNodeGapPx,
      maximumLayerInterNodeGapToFitPx,
    );
    const layerStackHeightPx =
      nodeDimensions.heightPx * layerNodeCount +
      resolvedLayerInterNodeGapPx * Math.max(0, layerNodeCount - 1);
    const centeredStackTopPx =
      topPaddingPx +
      nodeLayoutPaddingPx +
      Math.max(0, (availableLayerStackHeightPx - layerStackHeightPx) * 0.5);

    layerNodes.forEach((node, nodeInLayerIndex) => {
      const unclampedLayerYPx =
        centeredStackTopPx +
        halfNodeHeightPx +
        nodeInLayerIndex *
          (nodeDimensions.heightPx + resolvedLayerInterNodeGapPx);
      const layerYPx = clamp(
        unclampedLayerYPx,
        minimumNodeCenterYPx,
        maximumNodeCenterYPx,
      );
      positionedNodes.push({
        node,
        xPx: layerXPx,
        yPx: layerYPx,
      });
    });
  });

  // Step 3: Return flattened positioned node list.
  return positionedNodes;
}

/**
 * Formats architecture label text.
 *
 * @param architectureInputSize - Input-layer size.
 * @param hiddenLayersLabel - Hidden-layer section.
 * @param architectureOutputSize - Output-layer size.
 * @param totalNodeCount - Total node count.
 * @param totalConnectionCount - Total connection count.
 * @returns Formatted label.
 */
function formatArchitectureLabel(
  architectureInputSize: number,
  hiddenLayersLabel: string,
  architectureOutputSize: number,
  totalNodeCount: number,
  totalConnectionCount: number,
): string {
  return `${architectureInputSize} | ${hiddenLayersLabel} | ${architectureOutputSize}\n(${totalNodeCount} nodes, ${totalConnectionCount} connections)`;
}

/**
 * Formats hidden-layer section by descriptor source.
 *
 * @param hiddenLayerSizes - Hidden-layer widths.
 * @param architectureSource - Descriptor provenance.
 * @returns Hidden-layer label section.
 */
function resolveHiddenLayersLabel(
  hiddenLayerSizes: number[],
  architectureSource: 'layer-metadata' | 'graph-topology' | 'inferred',
): string {
  if (hiddenLayerSizes.length === 0) {
    return '-';
  }

  if (architectureSource === 'inferred') {
    return hiddenLayerSizes
      .map((hiddenLayerSize) => `~${hiddenLayerSize}`)
      .join(' - ');
  }

  return hiddenLayerSizes.join(' - ');
}
