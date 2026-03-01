import Network from '../../../src/architecture/network';
import {
  FLAPPY_BIAS_TIER_CENTER_THRESHOLD,
  FLAPPY_BIAS_TIER_EDGE_START_ABS_VALUE,
  FLAPPY_BIAS_TIER_MAX_ABS_VALUE,
  FLAPPY_CENTER_BLUE_RAMP,
  FLAPPY_CONNECTION_TIER_CENTER_THRESHOLD,
  FLAPPY_CONNECTION_TIER_EDGE_START_ABS_VALUE,
  FLAPPY_CONNECTION_TIER_MAX_ABS_VALUE,
  FLAPPY_LIGHT_NEON_RAMP,
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_NETWORK_HEADER_FONT_SIZE_PX,
  FLAPPY_NETWORK_HEADER_TEXT_COLOR,
  FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR,
  FLAPPY_NETWORK_LEGEND_BACKGROUND,
  FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR,
  FLAPPY_NETWORK_LEGEND_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_HEIGHT_THRESHOLD_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_ROW_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_GAP_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_TITLE_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR,
  FLAPPY_NETWORK_LEGEND_HEADER_COLOR,
  FLAPPY_NETWORK_LEGEND_HEADER_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_MARGIN_PX,
  FLAPPY_NETWORK_LEGEND_MIN_TOP_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_ROW_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_GAP_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_TITLE_HEIGHT_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR,
  FLAPPY_NETWORK_LEGEND_TARGET_TOP_PX,
  FLAPPY_NETWORK_LEGEND_TOP_LEFT_THRESHOLD_PX,
  FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
  FLAPPY_NETWORK_NODE_LABEL_FONT_WEIGHT,
  FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR,
  FLAPPY_NETWORK_NODE_LABEL_SIZE_RATIO,
  FLAPPY_NETWORK_OUTPUT_NODE_GLOW_COLOR,
  FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR,
  FLAPPY_REGULAR_NEON_RAMP,
  FLAPPY_TIER_ABOVE_COLOR,
  FLAPPY_TIER_EDGE_COUNT,
  FLAPPY_TIER_LOGARITHMIC_STEEPNESS,
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
} from './constants';
import { applyAlphaToHexColor, clamp } from './browser-entry.math.utils';
import type {
  ColorLegendRow,
  ColorTier,
  NetworkLegendLayout,
  NetworkNodeDimensionsLike,
  PositionedNetworkNodeLike,
  VisualNetworkConnectionLike,
  VisualNetworkNodeLike,
} from './browser-entry.types';

/**
 * Builds logarithmic diverging color tiers with a center band and edge extension.
 *
 * @param input - Tier creation options.
 * @returns Ordered tier list.
 */
export function createLogDivergingColorTiers(input: {
  maxAbsValue: number;
  centerBlueThreshold: number;
  negativePalette: readonly string[];
  centerBluePalette: readonly string[];
  positivePalette: readonly string[];
  logarithmicSteepness: number;
  edgeStartAbsValue?: number;
  edgeTierCount?: number;
}): ColorTier[] {
  // Step 1: Resolve mirrored magnitude tracks for negative and positive ranges.
  const negativeMagnitudes = resolveTwoZoneMagnitudes({
    minimumMagnitude: input.centerBlueThreshold,
    maximumMagnitude: input.maxAbsValue,
    tierCount: input.negativePalette.length,
    logarithmicSteepness: input.logarithmicSteepness,
    edgeStartAbsValue: input.edgeStartAbsValue,
    edgeTierCount: input.edgeTierCount,
  });

  const positiveMagnitudes = resolveTwoZoneMagnitudes({
    minimumMagnitude: input.centerBlueThreshold,
    maximumMagnitude: input.maxAbsValue,
    tierCount: input.positivePalette.length,
    logarithmicSteepness: input.logarithmicSteepness,
    edgeStartAbsValue: input.edgeStartAbsValue,
    edgeTierCount: input.edgeTierCount,
  });

  // Step 2: Build negative, center, and positive tier groups.
  const negativeTiers = input.negativePalette.map((color, colorIndex) => {
    const magnitude = negativeMagnitudes[colorIndex];

    return {
      upperBound: -magnitude,
      color,
    };
  });

  const centerTiers = input.centerBluePalette.map((color, colorIndex) => {
    const linearProgress = (colorIndex + 1) / input.centerBluePalette.length;
    return {
      upperBound:
        -input.centerBlueThreshold +
        linearProgress * input.centerBlueThreshold * 2,
      color,
    };
  });

  const positiveTiers = input.positivePalette.map((color, colorIndex) => {
    const magnitude = positiveMagnitudes[colorIndex];

    return {
      upperBound: magnitude,
      color,
    };
  });

  // Step 3: Merge and sort tiers by ascending upper bounds.
  return [...negativeTiers, ...centerTiers, ...positiveTiers].toSorted(
    (leftTier, rightTier) => leftTier.upperBound - rightTier.upperBound,
  );
}

/**
 * Resolves a color from ordered tier definitions.
 *
 * @param value - Numeric value to classify.
 * @param tiers - Ordered tier list.
 * @param aboveTierColor - Fallback color for values above the last tier.
 * @returns Resolved color string.
 */
export function resolveTierColor(
  value: number,
  tiers: ColorTier[],
  aboveTierColor: string,
): string {
  const resolvedTier = tiers.find((tier) => value <= tier.upperBound);
  return resolvedTier?.color ?? aboveTierColor;
}

/**
 * Creates legend rows from ordered tiers.
 *
 * @param tiers - Ordered color tiers.
 * @param aboveTierColor - Color for values above top tier.
 * @param symbol - Label symbol.
 * @returns Legend rows.
 */
export function createColorLegendRows(
  tiers: ColorTier[],
  aboveTierColor: string,
  symbol: 'w' | 'b',
): ColorLegendRow[] {
  const allRows = tiers.map((tier, tierIndex) => {
    if (tierIndex === 0) {
      return {
        label: `${symbol} <= ${formatLegendBound(tier.upperBound)}`,
        color: tier.color,
      };
    }

    const previousTier = tiers[tierIndex - 1];
    return {
      label: `${formatLegendBound(previousTier.upperBound)} < ${symbol} <= ${formatLegendBound(tier.upperBound)}`,
      color: tier.color,
    };
  });

  allRows.push({
    label: `${symbol} > ${formatLegendBound(tiers.at(-1)?.upperBound ?? 0)}`,
    color: aboveTierColor,
  });

  return compressLegendRowsAroundZero(allRows);
}

/**
 * Resolves network legend layout from canvas constraints.
 *
 * @param context - Render context.
 * @param connectionLegendRows - Connection legend rows.
 * @param biasLegendRows - Bias legend rows.
 * @returns Computed legend layout.
 */
export function resolveNetworkLegendLayout(
  context: CanvasRenderingContext2D,
  connectionLegendRows: ColorLegendRow[],
  biasLegendRows: ColorLegendRow[],
): NetworkLegendLayout {
  // Step 1: Resolve compact/regular legend mode from canvas constraints.
  const compactLegend =
    context.canvas.width < FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX ||
    context.canvas.height < FLAPPY_NETWORK_LEGEND_COMPACT_HEIGHT_THRESHOLD_PX;
  const legendWidthPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX;
  const legendHeaderHeightPx = FLAPPY_NETWORK_LEGEND_HEADER_HEIGHT_PX;
  const legendSectionTitleHeightPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_TITLE_HEIGHT_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_TITLE_HEIGHT_PX;
  const legendRowHeightPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_ROW_HEIGHT_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_ROW_HEIGHT_PX;
  const legendSectionGapPx = compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_GAP_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_GAP_PX;

  // Step 2: Compute full legend panel height from section and row geometry.
  const legendBottomPaddingPx = FLAPPY_NETWORK_LEGEND_BOTTOM_PADDING_PX;
  const legendHeightPx =
    legendHeaderHeightPx +
    legendSectionTitleHeightPx +
    connectionLegendRows.length * legendRowHeightPx +
    legendSectionGapPx +
    legendSectionTitleHeightPx +
    biasLegendRows.length * legendRowHeightPx +
    legendBottomPaddingPx;
  const legendMarginPx = FLAPPY_NETWORK_LEGEND_MARGIN_PX;
  const preferTopLeft =
    context.canvas.width < FLAPPY_NETWORK_LEGEND_TOP_LEFT_THRESHOLD_PX;

  // Step 3: Clamp left/top placement to visible canvas bounds.
  const maximumLeftPx = Math.max(
    legendMarginPx,
    context.canvas.width - legendWidthPx - legendMarginPx,
  );
  const legendLeftPx = preferTopLeft ? legendMarginPx : maximumLeftPx;

  const maximumTopPx = Math.max(
    legendMarginPx,
    context.canvas.height - legendHeightPx - legendMarginPx,
  );
  const minimumTopPx = Math.min(FLAPPY_NETWORK_LEGEND_MIN_TOP_PX, maximumTopPx);
  const targetTopPx = FLAPPY_NETWORK_LEGEND_TARGET_TOP_PX;
  const legendTopPx = clamp(targetTopPx, minimumTopPx, maximumTopPx);

  return {
    compactLegend,
    legendLeftPx,
    legendTopPx,
    legendWidthPx,
    legendHeightPx,
    legendHeaderHeightPx,
    legendSectionTitleHeightPx,
    legendRowHeightPx,
    legendSectionGapPx,
  };
}

const CONNECTION_COLOR_TIERS = createLogDivergingColorTiers({
  maxAbsValue: FLAPPY_CONNECTION_TIER_MAX_ABS_VALUE,
  centerBlueThreshold: FLAPPY_CONNECTION_TIER_CENTER_THRESHOLD,
  negativePalette: FLAPPY_LIGHT_NEON_RAMP,
  centerBluePalette: FLAPPY_CENTER_BLUE_RAMP,
  positivePalette: FLAPPY_REGULAR_NEON_RAMP,
  logarithmicSteepness: FLAPPY_TIER_LOGARITHMIC_STEEPNESS,
  edgeStartAbsValue: FLAPPY_CONNECTION_TIER_EDGE_START_ABS_VALUE,
  edgeTierCount: FLAPPY_TIER_EDGE_COUNT,
});

const BIAS_COLOR_TIERS = createLogDivergingColorTiers({
  maxAbsValue: FLAPPY_BIAS_TIER_MAX_ABS_VALUE,
  centerBlueThreshold: FLAPPY_BIAS_TIER_CENTER_THRESHOLD,
  negativePalette: FLAPPY_LIGHT_NEON_RAMP,
  centerBluePalette: FLAPPY_CENTER_BLUE_RAMP,
  positivePalette: FLAPPY_REGULAR_NEON_RAMP,
  logarithmicSteepness: FLAPPY_TIER_LOGARITHMIC_STEEPNESS,
  edgeStartAbsValue: FLAPPY_BIAS_TIER_EDGE_START_ABS_VALUE,
  edgeTierCount: FLAPPY_TIER_EDGE_COUNT,
});

const CONNECTION_COLOR_ABOVE_TIER = FLAPPY_TIER_ABOVE_COLOR;
const BIAS_COLOR_ABOVE_TIER = FLAPPY_TIER_ABOVE_COLOR;

/**
 * Draws weighted connection lines.
 *
 * @param context - Render context.
 * @param runtimeConnections - Runtime connection list.
 * @param positionByNodeIndex - Node layout map.
 * @returns Nothing.
 */
export function drawWeightedConnectionsLayer(
  context: CanvasRenderingContext2D,
  runtimeConnections: VisualNetworkConnectionLike[],
  positionByNodeIndex: Map<number, PositionedNetworkNodeLike>,
): void {
  runtimeConnections.forEach((runtimeConnection) => {
    const fromNodeIndex = runtimeConnection.from?.index;
    const toNodeIndex = runtimeConnection.to?.index;
    if (typeof fromNodeIndex !== 'number' || typeof toNodeIndex !== 'number') {
      return;
    }

    const fromPosition = positionByNodeIndex.get(fromNodeIndex);
    const toPosition = positionByNodeIndex.get(toNodeIndex);
    if (!fromPosition || !toPosition) {
      return;
    }

    const connectionWeight = runtimeConnection.weight ?? 0;
    const connectionEnabled = runtimeConnection.enabled !== false;
    const absoluteWeight = Math.abs(connectionWeight);
    const baseColor = resolveTierColor(
      connectionWeight,
      CONNECTION_COLOR_TIERS,
      CONNECTION_COLOR_ABOVE_TIER,
    );
    const alphaValue = connectionEnabled ? 0.95 : 0.2;

    context.strokeStyle = applyAlphaToHexColor(baseColor, alphaValue);
    context.lineWidth = 0.75 + Math.min(2.75, absoluteWeight * 1.3);
    context.setLineDash(connectionEnabled ? [] : [5, 4]);
    context.beginPath();
    context.moveTo(fromPosition.xPx, fromPosition.yPx);
    context.lineTo(toPosition.xPx, toPosition.yPx);
    context.stroke();
  });

  context.setLineDash([]);
}

/**
 * Draws all network nodes with bias labels.
 *
 * @param context - Render context.
 * @param positionedNodes - Positioned nodes.
 * @param nodeDimensions - Node dimensions.
 * @returns Nothing.
 */
export function drawBiasNodesLayer(
  context: CanvasRenderingContext2D,
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
): void {
  const minimumLabelHeightPx = FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX;
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const hiddenNodeVerticalPaddingPx = 2;

  positionedNodes.forEach((positionedNode) => {
    const nodeBias = positionedNode.node.bias;
    const nodeLabel = formatNodeBiasLabel(nodeBias);
    const isOutputNode = positionedNode.node.type === 'output';
    const nodeFillColor = isOutputNode
      ? FLAPPY_NEON_PALETTE.currentRunText
      : resolveTierColor(nodeBias, BIAS_COLOR_TIERS, BIAS_COLOR_ABOVE_TIER);
    const nodeStrokeColor = isOutputNode
      ? FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR
      : FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR;
    const nodeStrokeWidthPx = isOutputNode ? 2.1 : 1.3;

    const maximumReadableLabelHeightPx = Math.max(
      minimumLabelHeightPx,
      Math.floor(nodeDimensions.heightPx),
    );
    const computedLabelHeightPx = Math.floor(
      nodeDimensions.heightPx * FLAPPY_NETWORK_NODE_LABEL_SIZE_RATIO,
    );
    const labelHeightPx = clamp(
      computedLabelHeightPx,
      minimumLabelHeightPx,
      maximumReadableLabelHeightPx,
    );
    context.font = `${FLAPPY_NETWORK_NODE_LABEL_FONT_WEIGHT} ${labelHeightPx}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
    // Set baseline BEFORE measuring so actualBoundingBoxAscent/Descent are
    // relative to the alphabetic baseline — not a stale 'top' from prior draws.
    context.textBaseline = 'alphabetic';
    const labelMetrics = context.measureText(nodeLabel);
    const labelAscentPx =
      labelMetrics.actualBoundingBoxAscent || labelHeightPx * 0.72;
    const labelDescentPx =
      labelMetrics.actualBoundingBoxDescent || labelHeightPx * 0.28;
    const measuredLabelHeightPx = Math.max(
      minimumLabelHeightPx,
      Math.ceil(labelAscentPx + labelDescentPx),
    );
    const resolvedNodeHeightPx = isOutputNode
      ? Math.max(4, nodeDimensions.heightPx - 2)
      : Math.max(
          4,
          Math.min(
            nodeDimensions.heightPx,
            measuredLabelHeightPx + hiddenNodeVerticalPaddingPx,
          ),
        );
    const halfNodeHeightPx = resolvedNodeHeightPx * 0.5;

    context.fillStyle = nodeFillColor;
    context.strokeStyle = nodeStrokeColor;
    context.lineWidth = nodeStrokeWidthPx;
    context.shadowBlur = isOutputNode ? 7 : 0;
    context.shadowColor = isOutputNode
      ? FLAPPY_NETWORK_OUTPUT_NODE_GLOW_COLOR
      : 'transparent';
    context.fillRect(
      positionedNode.xPx - halfNodeWidthPx,
      positionedNode.yPx - halfNodeHeightPx,
      nodeDimensions.widthPx,
      resolvedNodeHeightPx,
    );
    context.strokeRect(
      positionedNode.xPx - halfNodeWidthPx,
      positionedNode.yPx - halfNodeHeightPx,
      nodeDimensions.widthPx,
      resolvedNodeHeightPx,
    );
    context.shadowBlur = 0;
    context.shadowColor = 'transparent';

    if (isOutputNode) {
      return;
    }

    context.fillStyle = FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR;
    context.textAlign = 'center';
    const labelBaselineYPx =
      positionedNode.yPx + (labelAscentPx - labelDescentPx) * 0.5;
    context.fillText(nodeLabel, positionedNode.xPx, labelBaselineYPx);
  });
}

/**
 * Draws network architecture header text.
 *
 * @param context - Render context.
 * @param architectureLabel - Header label.
 * @returns Nothing.
 */
export function drawNetworkVisualizationHeader(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
): void {
  context.fillStyle = FLAPPY_NETWORK_HEADER_TEXT_COLOR;
  context.font = `${FLAPPY_NETWORK_HEADER_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'left';
  context.textBaseline = 'top';
  const headerLines = architectureLabel.split('\n');
  headerLines.forEach((headerLine, lineIndex) => {
    context.fillText(headerLine, 8, 8 + lineIndex * 12);
  });
}

/**
 * Draws the color legend for connections and node bias values.
 *
 * @param context - Render context.
 * @returns Nothing.
 */
export function drawNetworkColorLegend(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
): void {
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ??
    context.canvas.width;
  if (viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX) {
    return;
  }

  // Step 1: Build legend row models for connection and bias tiers.
  const connectionLegendRows = createColorLegendRows(
    CONNECTION_COLOR_TIERS,
    CONNECTION_COLOR_ABOVE_TIER,
    'w',
  );
  const biasLegendRows = createColorLegendRows(
    BIAS_COLOR_TIERS,
    BIAS_COLOR_ABOVE_TIER,
    'b',
  );
  const legendLayout = resolveNetworkLegendLayout(
    context,
    connectionLegendRows,
    biasLegendRows,
  );

  // Step 2: Draw legend panel frame and header.
  const {
    compactLegend,
    legendLeftPx,
    legendTopPx,
    legendWidthPx,
    legendHeightPx,
    legendHeaderHeightPx,
    legendSectionTitleHeightPx,
    legendRowHeightPx,
    legendSectionGapPx,
  } = legendLayout;

  // Step 2: Draw architecture description directly above legend panel.
  const architectureLines = architectureLabel.split('\n');
  const architectureLineHeightPx = 12;
  const architectureTextTopPx = Math.max(
    4,
    legendTopPx - architectureLines.length * architectureLineHeightPx - 4,
  );

  context.fillStyle = FLAPPY_NETWORK_HEADER_TEXT_COLOR;
  context.font = `${FLAPPY_NETWORK_HEADER_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'left';
  context.textBaseline = 'top';
  architectureLines.forEach((architectureLine, lineIndex) => {
    context.fillText(
      architectureLine,
      legendLeftPx + 8,
      architectureTextTopPx + lineIndex * architectureLineHeightPx,
    );
  });

  // Step 3: Draw legend panel frame and header.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_BACKGROUND;
  context.strokeStyle = FLAPPY_NEON_PALETTE.statusText;
  context.lineWidth = 1;
  context.fillRect(legendLeftPx, legendTopPx, legendWidthPx, legendHeightPx);
  context.strokeRect(legendLeftPx, legendTopPx, legendWidthPx, legendHeightPx);

  context.fillStyle = FLAPPY_NETWORK_LEGEND_HEADER_COLOR;
  context.font = `${compactLegend ? FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX : FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'left';
  context.textBaseline = 'top';
  context.fillText('Legend', legendLeftPx + 8, legendTopPx + 6);

  // Step 4: Draw connection-weight section and row entries.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR;
  const connectionSectionTopPx = legendTopPx + legendHeaderHeightPx;
  context.fillText(
    'Connection weight',
    legendLeftPx + 8,
    connectionSectionTopPx,
  );

  connectionLegendRows.forEach((legendRow, legendRowIndex) => {
    const rowTopPx =
      connectionSectionTopPx +
      legendSectionTitleHeightPx +
      legendRowIndex * legendRowHeightPx;
    context.strokeStyle = legendRow.color;
    context.lineWidth = FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX;
    context.beginPath();
    context.moveTo(legendLeftPx + 8, rowTopPx + 4);
    context.lineTo(legendLeftPx + 28, rowTopPx + 4);
    context.stroke();

    context.fillStyle = FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR;
    context.fillText(legendRow.label, legendLeftPx + 32, rowTopPx - 1);
  });

  // Step 5: Draw node-bias section and row entries.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR;
  const biasSectionTopPx =
    connectionSectionTopPx +
    legendSectionTitleHeightPx +
    connectionLegendRows.length * legendRowHeightPx +
    legendSectionGapPx;
  context.fillText('Node bias', legendLeftPx + 8, biasSectionTopPx);

  biasLegendRows.forEach((legendRow, legendRowIndex) => {
    const rowTopPx =
      biasSectionTopPx +
      legendSectionTitleHeightPx +
      legendRowIndex * legendRowHeightPx;
    context.fillStyle = legendRow.color;
    context.beginPath();
    context.rect(legendLeftPx + 10, rowTopPx + 1, 6, 6);
    context.fill();

    context.fillStyle = FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR;
    context.fillText(legendRow.label, legendLeftPx + 22, rowTopPx - 1);
  });
}

/**
 * Resolves default legend layout from internal tier definitions.
 *
 * @param context - Render context.
 * @returns Legend layout.
 */
export function resolveDefaultNetworkLegendLayout(
  context: CanvasRenderingContext2D,
): NetworkLegendLayout {
  const connectionLegendRows = createColorLegendRows(
    CONNECTION_COLOR_TIERS,
    CONNECTION_COLOR_ABOVE_TIER,
    'w',
  );
  const biasLegendRows = createColorLegendRows(
    BIAS_COLOR_TIERS,
    BIAS_COLOR_ABOVE_TIER,
    'b',
  );
  return resolveNetworkLegendLayout(
    context,
    connectionLegendRows,
    biasLegendRows,
  );
}

/**
 * Resolves connection color for a raw weight.
 *
 * @param connectionWeight - Connection weight.
 * @returns Tier color.
 */
export function resolveConnectionRangeColor(connectionWeight: number): string {
  return resolveTierColor(
    connectionWeight,
    CONNECTION_COLOR_TIERS,
    CONNECTION_COLOR_ABOVE_TIER,
  );
}

/**
 * Resolves bias color for a raw node bias.
 *
 * @param nodeBias - Node bias.
 * @returns Tier color.
 */
export function resolveBiasRangeColor(nodeBias: number): string {
  return resolveTierColor(nodeBias, BIAS_COLOR_TIERS, BIAS_COLOR_ABOVE_TIER);
}

/**
 * Formats node bias labels with fixed sign and precision.
 *
 * @param nodeBias - Node bias value.
 * @returns Label text.
 */
export function formatNodeBiasLabel(nodeBias: number): string {
  const roundedBias = Number.isFinite(nodeBias) ? nodeBias : 0;
  return `${roundedBias >= 0 ? '+' : ''}${roundedBias.toFixed(2)}`;
}

/**
 * Resolves layered node groups for visualization.
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
  // Step 1: Build fallback two-layer structure when no network is available.
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

  // Step 2: Normalize runtime node shape into visualization node model.
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

  // Step 3: Resolve hidden-layer grouping (metadata first, topology fallback).
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

  // Step 4: Return layered representation with final fallback.
  return layeredNodes.length > 0 ? layeredNodes : [runtimeNodes];
}

/**
 * Groups hidden nodes by layer metadata.
 *
 * @param hiddenNodes - Hidden nodes.
 * @returns Layer groups.
 */
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

/**
 * Groups hidden nodes by topology-derived depth.
 *
 * @param network - Runtime network.
 * @param runtimeNodes - Runtime node list.
 * @param hiddenNodes - Hidden node list.
 * @returns Layer groups.
 */
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

/**
 * Resolves hidden node depth map from topology.
 *
 * @param runtimeNodes - Runtime nodes.
 * @param runtimeConnections - Runtime connections.
 * @returns Depth map.
 */
function resolveHiddenNodeDepthByTopology(
  runtimeNodes: VisualNetworkNodeLike[],
  runtimeConnections: VisualNetworkConnectionLike[],
): Map<number, number> {
  // Step 1: Initialize node map and adjacency bookkeeping containers.
  const nodeByIndex = new Map<number, VisualNetworkNodeLike>(
    runtimeNodes.map((runtimeNode) => [runtimeNode.index, runtimeNode]),
  );
  const outgoingTargetsByNode = new Map<number, number[]>();
  const incomingEdgeCountByNode = new Map<number, number>();

  nodeByIndex.forEach((_runtimeNode, runtimeNodeIndex) => {
    outgoingTargetsByNode.set(runtimeNodeIndex, []);
    incomingEdgeCountByNode.set(runtimeNodeIndex, 0);
  });

  // Step 2: Populate adjacency and incoming-degree counts from enabled edges.
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

  // Step 3: Run Kahn topological traversal to detect cycles and ordering.
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

  // Step 4: Fallback to flat hidden depth when cycles exist.
  const hasCycles = topologicalOrder.length !== nodeByIndex.size;
  if (hasCycles) {
    return new Map<number, number>(
      runtimeNodes
        .filter((runtimeNode) => runtimeNode.type === 'hidden')
        .map((runtimeNode) => [runtimeNode.index, 1]),
    );
  }

  // Step 5: Propagate depth values through topological order.
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

  // Step 6: Return hidden-node depth map only.
  return new Map<number, number>(
    runtimeNodes
      .filter((runtimeNode) => runtimeNode.type === 'hidden')
      .map((runtimeNode) => [
        runtimeNode.index,
        depthByNodeIndex.get(runtimeNode.index) ?? 1,
      ]),
  );
}

/**
 * Resolves magnitudes in near and edge zones.
 *
 * @param input - Magnitude options.
 * @returns Ordered magnitudes.
 */
function resolveTwoZoneMagnitudes(input: {
  minimumMagnitude: number;
  maximumMagnitude: number;
  tierCount: number;
  logarithmicSteepness: number;
  edgeStartAbsValue?: number;
  edgeTierCount?: number;
}): number[] {
  // Step 1: Resolve safe tier counts and edge-zone split configuration.
  const safeTierCount = Math.max(1, input.tierCount);
  const targetEdgeStart = clamp(
    input.edgeStartAbsValue ?? input.maximumMagnitude,
    input.minimumMagnitude,
    input.maximumMagnitude,
  );
  const requestedEdgeTierCount = clamp(
    input.edgeTierCount ?? 0,
    0,
    safeTierCount,
  );
  const edgeTierCount =
    targetEdgeStart >= input.maximumMagnitude ? 0 : requestedEdgeTierCount;
  const nearTierCount = Math.max(1, safeTierCount - edgeTierCount);

  // Step 2: Build logarithmic near-zone magnitudes.
  const nearMagnitudes = Array.from(
    { length: nearTierCount },
    (_unusedValue, tierIndex) => {
      const logarithmicProgress = mapLogarithmicProgress(
        tierIndex + 1,
        nearTierCount,
        input.logarithmicSteepness,
      );
      return (
        input.minimumMagnitude +
        (targetEdgeStart - input.minimumMagnitude) * logarithmicProgress
      );
    },
  );

  // Step 3: Return early when no edge-zone tiers are requested.
  if (edgeTierCount === 0) {
    return nearMagnitudes;
  }

  // Step 4: Build linear edge-zone magnitudes and concatenate.
  const edgeMagnitudes = Array.from(
    { length: edgeTierCount },
    (_unusedValue, edgeIndex) => {
      const linearProgress = (edgeIndex + 1) / edgeTierCount;
      return (
        targetEdgeStart +
        (input.maximumMagnitude - targetEdgeStart) * linearProgress
      );
    },
  );

  return [...nearMagnitudes, ...edgeMagnitudes];
}

/**
 * Maps linear position to logarithmic progress.
 *
 * @param position - 1-based position.
 * @param totalPositions - Total position count.
 * @param logarithmicSteepness - Log curve steepness.
 * @returns Logarithmic progress.
 */
function mapLogarithmicProgress(
  position: number,
  totalPositions: number,
  logarithmicSteepness: number,
): number {
  const normalizedPosition = clamp(
    position / Math.max(1, totalPositions),
    0,
    1,
  );
  return (
    Math.log1p(logarithmicSteepness * normalizedPosition) /
    Math.log1p(logarithmicSteepness)
  );
}

/**
 * Compresses legend rows while preserving center and edge zones.
 *
 * @param rows - Full legend rows.
 * @returns Compressed row set.
 */
function compressLegendRowsAroundZero(
  rows: ColorLegendRow[],
): ColorLegendRow[] {
  // Step 1: Return early when row count is already compact enough.
  const maximumLegendRows = 14;
  if (rows.length <= maximumLegendRows) {
    return rows;
  }

  // Step 2: Select boundary rows and centered window around zero.
  const centerIndex = Math.floor(rows.length / 2);
  const selectedIndexes = new Set<number>([
    0,
    1,
    rows.length - 2,
    rows.length - 1,
  ]);

  const centerWindowRadius = 4;
  for (
    let rowIndex = Math.max(0, centerIndex - centerWindowRadius);
    rowIndex <= Math.min(rows.length - 1, centerIndex + centerWindowRadius);
    rowIndex++
  ) {
    selectedIndexes.add(rowIndex);
  }

  // Step 3: Add two extra evenly-spaced representatives from both sides.
  const evenlySpacedExtraIndexes = [
    Math.floor(rows.length * 0.2),
    Math.floor(rows.length * 0.8),
  ];
  evenlySpacedExtraIndexes.forEach((extraIndex) => {
    selectedIndexes.add(clamp(extraIndex, 0, rows.length - 1));
  });

  // Step 4: Return selected rows in ascending original order.
  return [...selectedIndexes]
    .toSorted((leftIndex, rightIndex) => leftIndex - rightIndex)
    .map((rowIndex) => rows[rowIndex]);
}

/**
 * Formats a legend bound with fixed precision.
 *
 * @param value - Numeric bound.
 * @returns Formatted bound string.
 */
function formatLegendBound(value: number): string {
  return value.toFixed(2);
}
