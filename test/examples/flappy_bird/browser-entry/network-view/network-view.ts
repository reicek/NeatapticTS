import Network from '../../../../../src/architecture/network';
import {
  FLAPPY_NETWORK_BASELINE_HEIGHT_PX,
  FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT,
  FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX,
  FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
  FLAPPY_NETWORK_MAX_HEIGHT_PX,
  FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX,
  FLAPPY_NETWORK_MAX_NODE_WIDTH_PX,
  FLAPPY_NETWORK_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX,
  FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
  FLAPPY_NETWORK_MIN_NODE_HEIGHT_LABEL_EXTRA_PX,
  FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX,
  FLAPPY_NETWORK_MIN_NODE_WIDTH_PX,
  FLAPPY_NETWORK_NODE_DENSITY_BASELINE_COUNT,
  FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX,
  FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_DIVISOR,
  FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_MIN_DENOMINATOR,
  FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_DIVISOR,
  FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_MIN_DENOMINATOR,
  FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX,
  FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_DIVISOR,
  FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_MIN_DENOMINATOR,
  FLAPPY_NETWORK_NODE_WIDTH_TO_HEIGHT_RATIO,
  FLAPPY_NETWORK_TOPOLOGY_HEIGHT_MULTIPLIER,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
} from '../../constants/constants';
import { clamp } from '../browser-entry.math.utils';
import type {
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  PositionedNetworkNodeLike as PositionedNetworkNode,
  VisualNetworkConnectionLike,
} from '../browser-entry.types';
import {
  drawBiasNodesLayer,
  drawNetworkColorLegend,
  drawWeightedConnectionsLayer,
} from '../visualization/visualization.draw.service';
import { resolveDefaultNetworkLegendLayout } from '../visualization/visualization.legend.utils';
import { resolveNetworkVisualizationColorScales } from '../visualization/visualization.colors.utils';
import { drawInputGroupLabelBands } from './network-view.draw.service';
import {
  centerPositionedNodesInDrawableArea,
  positionNetworkNodes,
} from './network-view.layout.utils';
import { resolveNetworkVisualizationLayers } from './network-view.topology.utils';

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

  // Step 2: Resolve architecture label, graph paddings, and overlay state.
  const architectureLabel = resolveNetworkArchitectureLabel(
    network,
    inputSize,
    outputSize,
  );
  const groupLabelBandReserveWidthPx =
    FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX +
    FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX +
    FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX;
  const graphLeftPaddingPx =
    FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX + groupLabelBandReserveWidthPx;
  const graphTopPaddingPx = FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX;
  const graphRightPaddingPx = FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX;
  const graphBottomPaddingPx = FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX;
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ?? canvasWidthPx;
  const hideNetworkOverlays =
    viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX;
  const colorScales = resolveNetworkVisualizationColorScales(network);

  // Step 3: Reserve graph space around the legend when overlays are visible.
  let adjustedGraphLeftPaddingPx = graphLeftPaddingPx;
  let adjustedGraphRightPaddingPx = graphRightPaddingPx;
  if (!hideNetworkOverlays) {
    const legendLayout = resolveDefaultNetworkLegendLayout(context, network);
    const legendMidpointPx =
      legendLayout.legendLeftPx + legendLayout.legendWidthPx * 0.5;
    if (legendMidpointPx >= canvasWidthPx * 0.5) {
      adjustedGraphRightPaddingPx = Math.max(
        graphRightPaddingPx,
        canvasWidthPx -
          legendLayout.legendLeftPx +
          FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
      );
    } else {
      adjustedGraphLeftPaddingPx = Math.max(
        graphLeftPaddingPx,
        legendLayout.legendLeftPx +
          legendLayout.legendWidthPx +
          FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
      );
    }
  }

  // Step 4: Resolve drawable graph bounds and node dimensions.
  const drawableWidthPx = Math.max(
    1,
    canvasWidthPx - adjustedGraphLeftPaddingPx - adjustedGraphRightPaddingPx,
  );
  const drawableHeightPx = Math.max(
    1,
    canvasHeightPx - graphTopPaddingPx - graphBottomPaddingPx,
  );
  const nodeDimensions = resolveNetworkNodeDimensions(
    network,
    inputSize,
    outputSize,
    drawableWidthPx,
    drawableHeightPx,
  );

  // Step 5: Position nodes and build the node-index lookup map.
  const networkLayers = resolveNetworkVisualizationLayers(
    network,
    inputSize,
    outputSize,
  );
  const positionedNodes = positionNetworkNodes(
    networkLayers,
    adjustedGraphLeftPaddingPx,
    graphTopPaddingPx,
    drawableWidthPx,
    drawableHeightPx,
    FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX,
    nodeDimensions,
  );
  const centeredPositionedNodes = centerPositionedNodesInDrawableArea(
    positionedNodes,
    adjustedGraphLeftPaddingPx,
    graphTopPaddingPx,
    drawableWidthPx,
    drawableHeightPx,
    FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX,
    nodeDimensions,
  );
  const positionByNodeIndex = new Map<number, PositionedNetworkNode>(
    centeredPositionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );
  const runtimeConnections =
    ((network?.connections ?? []) as VisualNetworkConnectionLike[]) ?? [];

  // Step 6: Draw graph layers and optional overlay helpers.
  drawWeightedConnectionsLayer(
    context,
    runtimeConnections,
    positionByNodeIndex,
    colorScales.connectionScale,
  );
  if (!hideNetworkOverlays) {
    drawInputGroupLabelBands(context, centeredPositionedNodes, nodeDimensions);
  }
  drawBiasNodesLayer(
    context,
    centeredPositionedNodes,
    nodeDimensions,
    colorScales.biasScale,
  );
  drawNetworkColorLegend(context, architectureLabel, colorScales);
}

/**
 * Resolves responsive visualization canvas height from network shape.
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
  const minimumNodeSidePx =
    FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX +
    FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX * 2;
  const layers = resolveNetworkVisualizationLayers(
    network,
    inputSize,
    outputSize,
  );
  const layerCount = Math.max(1, layers.length);
  const maximumLayerNodeCount = Math.max(
    1,
    ...layers.map((layerNodes) => layerNodes.length),
  );

  // Step 2: Compute topology-driven minimum readable height.
  const minimumReadableStackHeightPx =
    maximumLayerNodeCount <= 1
      ? minimumNodeSidePx
      : maximumLayerNodeCount * minimumNodeSidePx +
        (maximumLayerNodeCount - 1) * FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX;
  const topologyDrivenHeightPx =
    FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX +
    FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX +
    FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX +
    minimumReadableStackHeightPx;

  // Step 3: Add complexity-based adjustments and choose the larger height.
  const nodeDensityHeightPx =
    Math.max(
      0,
      maximumLayerNodeCount - FLAPPY_NETWORK_NODE_DENSITY_BASELINE_COUNT,
    ) * FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX;
  const layerComplexityHeightPx =
    Math.max(0, layerCount - FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT) *
    FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX;
  const recommendedHeightPx = Math.max(
    Math.ceil(
      topologyDrivenHeightPx * FLAPPY_NETWORK_TOPOLOGY_HEIGHT_MULTIPLIER,
    ),
    FLAPPY_NETWORK_BASELINE_HEIGHT_PX +
      nodeDensityHeightPx +
      layerComplexityHeightPx,
  );

  // Step 4: Clamp to configured min and max bounds.
  return Math.floor(
    clamp(
      recommendedHeightPx,
      FLAPPY_NETWORK_MIN_HEIGHT_PX,
      FLAPPY_NETWORK_MAX_HEIGHT_PX,
    ),
  );
}

/**
 * Resolves compact architecture label text for headers and HUD rows.
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

function resolveNetworkNodeDimensions(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
): NetworkNodeDimensions {
  const networkLayers = resolveNetworkVisualizationLayers(
    network,
    inputSize,
    outputSize,
  );
  const maximumLayerNodeCount = Math.max(
    1,
    ...networkLayers.map((layerNodes) => layerNodes.length),
  );
  const layerCount = Math.max(1, networkLayers.length);
  const availableNodeStackHeightPx = Math.max(
    1,
    drawableHeightPx - FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX * 2,
  );
  const strictMaximumNodeHeightByFitPx = Math.max(
    4,
    availableNodeStackHeightPx / maximumLayerNodeCount,
  );
  const minimumNodeHeightPx =
    FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX +
    FLAPPY_NETWORK_MIN_NODE_HEIGHT_LABEL_EXTRA_PX;
  const effectiveMinimumNodeHeightPx = Math.min(
    minimumNodeHeightPx,
    strictMaximumNodeHeightByFitPx,
  );
  const nodeHeightPx = Math.max(
    effectiveMinimumNodeHeightPx,
    Math.min(
      FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX,
      (drawableHeightPx - FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX * 2) /
        Math.max(
          FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_MIN_DENOMINATOR,
          maximumLayerNodeCount * FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_DIVISOR,
        ),
      drawableWidthPx /
        Math.max(
          FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_MIN_DENOMINATOR,
          layerCount * FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_DIVISOR,
        ),
      strictMaximumNodeHeightByFitPx,
    ),
  );
  const nodeWidthPx = Math.max(
    FLAPPY_NETWORK_MIN_NODE_WIDTH_PX,
    Math.min(
      FLAPPY_NETWORK_MAX_NODE_WIDTH_PX,
      nodeHeightPx * FLAPPY_NETWORK_NODE_WIDTH_TO_HEIGHT_RATIO,
      drawableWidthPx /
        Math.max(
          FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_MIN_DENOMINATOR,
          layerCount * FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_DIVISOR,
        ),
    ),
  );

  return {
    widthPx: nodeWidthPx,
    heightPx: nodeHeightPx,
  };
}

function formatArchitectureLabel(
  architectureInputSize: number,
  hiddenLayersLabel: string,
  architectureOutputSize: number,
  totalNodeCount: number,
  totalConnectionCount: number,
): string {
  return `${architectureInputSize} | ${hiddenLayersLabel} | ${architectureOutputSize}\n(${totalNodeCount} nodes, ${totalConnectionCount} connections)`;
}

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
