import Network from '../../../../src/architecture/network';
import { clamp } from './browser-entry.math.utils';
import { resolveNetworkVisualizationColorScales } from './browser-entry.visualization.utils';
import {
  drawBiasNodesLayer as drawBiasNodes,
  drawNetworkColorLegend,
  drawWeightedConnectionsLayer as drawWeightedConnections,
} from './visualization/visualization.draw.service';
import { resolveDefaultNetworkLegendLayout } from './visualization/visualization.legend.utils';
import { resolveNetworkVisualizationLayers } from './visualization/visualization';
import {
  FLAPPY_NETWORK_BASELINE_HEIGHT_PX,
  FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
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
  FLAPPY_NETWORK_MIN_NODE_HEIGHT_LABEL_EXTRA_PX,
  FLAPPY_NETWORK_MIN_NODE_WIDTH_PX,
  FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX,
  FLAPPY_NETWORK_NODE_DENSITY_BASELINE_COUNT,
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
  FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT,
} from '../constants/constants';
import type {
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  PositionedNetworkNodeLike as PositionedNetworkNode,
  VisualNetworkConnectionLike,
  VisualNetworkNodeLike,
} from './browser-entry.types';
import {
  centerPositionedNodesInDrawableArea,
  positionNetworkNodes,
} from './network-view/network-view.layout.utils';
import { drawInputGroupLabelBands } from './network-view/network-view.draw.service';

/**
 * Draws a complete, layer-based visualization of the active network.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Nothing.
 */
export function drawNetworkVisualizationInternal(
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
  const architectureLabel = resolveNetworkArchitectureLabelInternal(
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
  const legendGraphGapPx = FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX;
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ?? canvasWidthPx;
  const hideNetworkOverlays =
    viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX;
  const nodeLayoutPaddingPx = FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX;
  const minimumLabelHeightPx = FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX;
  const minimumNodeHeightPx =
    minimumLabelHeightPx + FLAPPY_NETWORK_MIN_NODE_HEIGHT_LABEL_EXTRA_PX;
  const minimumNodeWidthPx = FLAPPY_NETWORK_MIN_NODE_WIDTH_PX;
  const colorScales = resolveNetworkVisualizationColorScales(network);

  // Step 3: Resolve legend layout and reserve horizontal graph space around it.
  let adjustedGraphLeftPaddingPx = graphLeftPaddingPx;
  let adjustedGraphRightPaddingPx = graphRightPaddingPx;
  if (!hideNetworkOverlays) {
    const legendLayout = resolveDefaultNetworkLegendLayout(context, network);
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
        Math.max(
          FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_MIN_DENOMINATOR,
          maxLayerNodeCount * FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_DIVISOR,
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
    minimumNodeWidthPx,
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
  const centeredPositionedNodes = centerPositionedNodesInDrawableArea(
    positionedNodes,
    adjustedGraphLeftPaddingPx,
    graphTopPaddingPx,
    drawableWidthPx,
    drawableHeightPx,
    nodeLayoutPaddingPx,
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

  // Step 7: Draw connections, nodes, and legend (with architecture text above it).
  drawWeightedConnections(
    context,
    runtimeConnections,
    positionByNodeIndex,
    colorScales.connectionScale,
  );
  if (!hideNetworkOverlays) {
    drawInputGroupLabelBands(context, centeredPositionedNodes, nodeDimensions);
  }
  drawBiasNodes(
    context,
    centeredPositionedNodes,
    nodeDimensions,
    colorScales.biasScale,
  );
  drawNetworkColorLegend(context, architectureLabel, colorScales);
}

/**
 * Resolves the recommended network visualization host height.
 *
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Recommended height in pixels.
 */
export function resolveNetworkVisualizationHeightPxInternal(
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
    Math.max(
      0,
      maxLayerNodeCount - FLAPPY_NETWORK_NODE_DENSITY_BASELINE_COUNT,
    ) * FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX;
  const layerComplexityHeightPx =
    Math.max(0, layerCount - FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT) *
    FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX;
  const recommendedHeightPx = Math.max(
    Math.ceil(
      topologyDrivenHeightPx * FLAPPY_NETWORK_TOPOLOGY_HEIGHT_MULTIPLIER,
    ),
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
export function resolveNetworkArchitectureLabelInternal(
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
