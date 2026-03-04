import Network from '../../../../src/architecture/network';
import { clamp } from './browser-entry.math.utils';
import {
  drawBiasNodesLayer as drawBiasNodes,
  drawNetworkColorLegend,
  drawWeightedConnectionsLayer as drawWeightedConnections,
  resolveDefaultNetworkLegendLayout,
  resolveNetworkVisualizationColorScales,
  resolveNetworkVisualizationLayers,
} from './browser-entry.visualization.utils';
import {
  FLAPPY_LIGHT_NEON_RAMP,
  FLAPPY_MEMORY_CORE_FEATURE_COUNT,
  FLAPPY_MEMORY_STACKED_FRAME_COUNT,
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NETWORK_BASELINE_HEIGHT_PX,
  FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR,
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

interface InputGroupLabelBand {
  label: string;
  startNodeIndex: number;
  endNodeIndex: number;
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
}

const FLAPPY_INPUT_GROUP_LABELS: readonly string[] = [
  'CURRENT FRAME',
  'PREVIOUS FRAME',
  'TWO FRAMES AGO',
  'ACT',
  'RATE',
] as const;

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
 * Draws vertical neon bands that label semantic groups in the input layer.
 *
 * @param context - Canvas 2D rendering context.
 * @param positionedNodes - Positioned nodes in graph coordinates.
 * @param nodeDimensions - Resolved node dimensions.
 * @returns Nothing.
 */
function drawInputGroupLabelBands(
  context: CanvasRenderingContext2D,
  positionedNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): void {
  // Step 1: Collect input/constant nodes in top-to-bottom order.
  const inputNodes = positionedNodes
    .filter(
      (positionedNode) =>
        positionedNode.node.type === 'input' ||
        positionedNode.node.type === 'constant',
    )
    .toSorted((leftNode, rightNode) => leftNode.yPx - rightNode.yPx);

  if (inputNodes.length === 0) {
    return;
  }

  // Step 2: Resolve semantic range bands based on Flappy observation layout.
  const labelBands = resolveInputGroupLabelBands(inputNodes.length);
  if (labelBands.length === 0) {
    return;
  }

  // Step 3: Anchor label bands to the left side of the input node column.
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const leftmostInputCenterXPx = Math.min(
    ...inputNodes.map((inputNode) => inputNode.xPx),
  );
  const inputLeftEdgeXPx = leftmostInputCenterXPx - halfNodeWidthPx;
  const labelBandRightXPx =
    inputLeftEdgeXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX;
  const labelBandLeftXPx =
    labelBandRightXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX;

  // Step 4: Render each grouped band with rotated black text on neon background.
  labelBands.forEach((labelBand) => {
    const startNode = inputNodes[labelBand.startNodeIndex];
    const endNode = inputNodes[labelBand.endNodeIndex];
    if (!startNode || !endNode) {
      return;
    }

    const groupTopYPx = startNode.yPx - nodeDimensions.heightPx * 0.5;
    const groupBottomYPx = endNode.yPx + nodeDimensions.heightPx * 0.5;
    const desiredBandHeightPx =
      labelBand.orientation === 'vertical'
        ? Math.max(
            FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
            groupBottomYPx - groupTopYPx,
          )
        : Math.max(nodeDimensions.heightPx + 2, groupBottomYPx - groupTopYPx);
    const labelBandCenterYPx = (groupTopYPx + groupBottomYPx) * 0.5;
    const labelBandTopYPx = labelBandCenterYPx - desiredBandHeightPx * 0.5;

    drawRoundedRect(
      context,
      labelBandLeftXPx,
      labelBandTopYPx,
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
      desiredBandHeightPx,
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX,
      labelBand.backgroundColor,
    );

    context.save();
    context.translate(
      labelBandLeftXPx + FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX * 0.5,
      labelBandCenterYPx,
    );
    if (labelBand.orientation === 'vertical') {
      context.rotate(-Math.PI / 2);
    }
    context.fillStyle = FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR;
    context.font = `${FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT} ${FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText(labelBand.label, 0, 0);
    context.restore();
  });
}

/**
 * Resolves input-layer semantic label bands for Flappy temporal observation channels.
 *
 * @param inputNodeCount - Input-layer node count.
 * @returns Group label ranges with band colors.
 */
function resolveInputGroupLabelBands(
  inputNodeCount: number,
): InputGroupLabelBand[] {
  // Step 1: Resolve expected grouped layout (12 + 12 + 12 + 1 + 1).
  const perFrameFeatureCount = FLAPPY_MEMORY_CORE_FEATURE_COUNT;
  const stackedFrameCount = FLAPPY_MEMORY_STACKED_FRAME_COUNT;
  const temporalFeatureCount = perFrameFeatureCount * stackedFrameCount;
  const actionChannelsCount = 2;
  const expectedInputNodeCount = temporalFeatureCount + actionChannelsCount;
  if (inputNodeCount !== expectedInputNodeCount) {
    return [];
  }

  // Step 2: Build range map with distinct light-neon palette entries.
  const groupedCounts = [
    perFrameFeatureCount,
    perFrameFeatureCount,
    perFrameFeatureCount,
    1,
    1,
  ];
  const groupedColors = [
    FLAPPY_LIGHT_NEON_RAMP[0],
    FLAPPY_LIGHT_NEON_RAMP[2],
    FLAPPY_LIGHT_NEON_RAMP[4],
    FLAPPY_LIGHT_NEON_RAMP[6],
    FLAPPY_LIGHT_NEON_RAMP[8],
  ];
  const groupedOrientations: Array<'vertical' | 'horizontal'> = [
    'vertical',
    'vertical',
    'vertical',
    'horizontal',
    'horizontal',
  ];

  const groupBands: InputGroupLabelBand[] = [];
  let runningNodeIndex = 0;
  groupedCounts.forEach((groupCount, groupIndex) => {
    const startNodeIndex = runningNodeIndex;
    const endNodeIndex = runningNodeIndex + groupCount - 1;
    groupBands.push({
      label: FLAPPY_INPUT_GROUP_LABELS[groupIndex] ?? `GROUP ${groupIndex + 1}`,
      startNodeIndex,
      endNodeIndex,
      backgroundColor:
        groupedColors[groupIndex] ??
        FLAPPY_LIGHT_NEON_RAMP.at(-1) ??
        FLAPPY_LIGHT_NEON_RAMP[0],
      orientation: groupedOrientations[groupIndex] ?? 'vertical',
    });
    runningNodeIndex += groupCount;
  });

  // Step 3: Return fully resolved semantic bands.
  return groupBands;
}

/**
 * Draws a filled rounded rectangle path.
 *
 * @param context - Canvas 2D rendering context.
 * @param leftXPx - Left x coordinate.
 * @param topYPx - Top y coordinate.
 * @param widthPx - Rectangle width.
 * @param heightPx - Rectangle height.
 * @param radiusPx - Corner radius.
 * @param fillColor - Fill color.
 * @returns Nothing.
 */
function drawRoundedRect(
  context: CanvasRenderingContext2D,
  leftXPx: number,
  topYPx: number,
  widthPx: number,
  heightPx: number,
  radiusPx: number,
  fillColor: string,
): void {
  // Step 1: Clamp radius so corners remain valid for thin rectangles.
  const resolvedRadiusPx = Math.max(
    0,
    Math.min(radiusPx, widthPx * 0.5, heightPx * 0.5),
  );

  // Step 2: Trace rounded rectangle segments and fill.
  context.beginPath();
  context.moveTo(leftXPx + resolvedRadiusPx, topYPx);
  context.lineTo(leftXPx + widthPx - resolvedRadiusPx, topYPx);
  context.quadraticCurveTo(
    leftXPx + widthPx,
    topYPx,
    leftXPx + widthPx,
    topYPx + resolvedRadiusPx,
  );
  context.lineTo(leftXPx + widthPx, topYPx + heightPx - resolvedRadiusPx);
  context.quadraticCurveTo(
    leftXPx + widthPx,
    topYPx + heightPx,
    leftXPx + widthPx - resolvedRadiusPx,
    topYPx + heightPx,
  );
  context.lineTo(leftXPx + resolvedRadiusPx, topYPx + heightPx);
  context.quadraticCurveTo(
    leftXPx,
    topYPx + heightPx,
    leftXPx,
    topYPx + heightPx - resolvedRadiusPx,
  );
  context.lineTo(leftXPx, topYPx + resolvedRadiusPx);
  context.quadraticCurveTo(leftXPx, topYPx, leftXPx + resolvedRadiusPx, topYPx);
  context.closePath();
  context.fillStyle = fillColor;
  context.fill();
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
    const minimumStackTopPx = topPaddingPx + nodeLayoutPaddingPx;
    const maximumStackTopPx =
      topPaddingPx +
      drawableHeightPx -
      nodeLayoutPaddingPx -
      layerStackHeightPx;
    const resolvedStackTopPx = clamp(
      centeredStackTopPx,
      minimumStackTopPx,
      Math.max(minimumStackTopPx, maximumStackTopPx),
    );

    layerNodes.forEach((node, nodeInLayerIndex) => {
      const unclampedLayerYPx =
        resolvedStackTopPx +
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
 * Centers positioned nodes within the drawable graph area.
 *
 * @param positionedNodes - Positioned nodes before centering.
 * @param leftPaddingPx - Left graph padding.
 * @param topPaddingPx - Top graph padding.
 * @param drawableWidthPx - Drawable graph width.
 * @param drawableHeightPx - Drawable graph height.
 * @param nodeLayoutPaddingPx - Inner graph padding.
 * @param nodeDimensions - Node dimensions.
 * @returns Center-aligned positioned nodes.
 */
function centerPositionedNodesInDrawableArea(
  positionedNodes: PositionedNetworkNode[],
  leftPaddingPx: number,
  topPaddingPx: number,
  drawableWidthPx: number,
  drawableHeightPx: number,
  nodeLayoutPaddingPx: number,
  nodeDimensions: NetworkNodeDimensions,
): PositionedNetworkNode[] {
  if (positionedNodes.length === 0) {
    return positionedNodes;
  }

  // Step 1: Resolve current positioned-node bounds.
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = nodeDimensions.heightPx * 0.5;
  const currentLeftPx = Math.min(
    ...positionedNodes.map((positionedNode) => positionedNode.xPx - halfNodeWidthPx),
  );
  const currentRightPx = Math.max(
    ...positionedNodes.map((positionedNode) => positionedNode.xPx + halfNodeWidthPx),
  );
  const currentTopPx = Math.min(
    ...positionedNodes.map((positionedNode) => positionedNode.yPx - halfNodeHeightPx),
  );
  const currentBottomPx = Math.max(
    ...positionedNodes.map((positionedNode) => positionedNode.yPx + halfNodeHeightPx),
  );

  // Step 2: Resolve target centered bounds inside drawable area.
  const minimumLeftPx = leftPaddingPx + nodeLayoutPaddingPx;
  const maximumRightPx =
    leftPaddingPx + drawableWidthPx - nodeLayoutPaddingPx;
  const minimumTopPx = topPaddingPx + nodeLayoutPaddingPx;
  const maximumBottomPx = topPaddingPx + drawableHeightPx - nodeLayoutPaddingPx;

  const currentCenterXPx = (currentLeftPx + currentRightPx) * 0.5;
  const targetCenterXPx = (minimumLeftPx + maximumRightPx) * 0.5;
  const currentCenterYPx = (currentTopPx + currentBottomPx) * 0.5;
  const targetCenterYPx = (minimumTopPx + maximumBottomPx) * 0.5;

  const desiredShiftXPx = targetCenterXPx - currentCenterXPx;
  const minimumShiftXPx = minimumLeftPx - currentLeftPx;
  const maximumShiftXPx = maximumRightPx - currentRightPx;
  const resolvedShiftXPx = clamp(
    desiredShiftXPx,
    minimumShiftXPx,
    maximumShiftXPx,
  );

  const desiredShiftYPx = targetCenterYPx - currentCenterYPx;
  const minimumShiftYPx = minimumTopPx - currentTopPx;
  const maximumShiftYPx = maximumBottomPx - currentBottomPx;
  const resolvedShiftYPx = clamp(
    desiredShiftYPx,
    minimumShiftYPx,
    maximumShiftYPx,
  );

  // Step 3: Return translated positioned nodes.
  return positionedNodes.map((positionedNode) => ({
    ...positionedNode,
    xPx: positionedNode.xPx + resolvedShiftXPx,
    yPx: positionedNode.yPx + resolvedShiftYPx,
  }));
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
