import Network from '../../../../src/architecture/network';
import {
  FLAPPY_CENTER_BLUE_RAMP,
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
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
} from '../constants/constants';
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

interface DynamicColorScale {
  minimumValue: number;
  maximumValue: number;
  tiers: ColorTier[];
  aboveTierColor: string;
}

interface NetworkVisualizationColorScales {
  connectionScale: DynamicColorScale;
  biasScale: DynamicColorScale;
}

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
  scale: DynamicColorScale,
  symbol: 'w' | 'b',
): ColorLegendRow[] {
  return scale.tiers.map((tier, tierIndex) => {
    const lowerBound =
      tierIndex === 0
        ? scale.minimumValue
        : scale.tiers[tierIndex - 1].upperBound;
    return {
      label: `${formatLegendBound(lowerBound)} <= ${symbol} <= ${formatLegendBound(tier.upperBound)}`,
      color: tier.color,
      minimumValue: lowerBound,
      maximumValue: tier.upperBound,
    };
  });
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
  connectionScale: DynamicColorScale,
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
    const isNegativeConnection = connectionWeight < 0;
    const baseColor = resolveTierColor(
      connectionWeight,
      connectionScale.tiers,
      connectionScale.aboveTierColor,
    );
    const alphaValue = connectionEnabled ? 0.95 : 0.2;
    const connectionColor = applyAlphaToHexColor(baseColor, alphaValue);
    const resolvedLineWidthPx = FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX;

    if (isNegativeConnection) {
      drawSquareDottedConnection(context, {
        fromXPx: fromPosition.xPx,
        fromYPx: fromPosition.yPx,
        toXPx: toPosition.xPx,
        toYPx: toPosition.yPx,
        color: connectionColor,
        lineWidthPx: resolvedLineWidthPx,
      });
      return;
    }

    context.strokeStyle = connectionColor;
    context.lineWidth = resolvedLineWidthPx;
    context.setLineDash(connectionEnabled ? [] : [5, 4]);
    context.beginPath();
    context.moveTo(fromPosition.xPx, fromPosition.yPx);
    context.lineTo(toPosition.xPx, toPosition.yPx);
    context.stroke();
  });

  context.setLineDash([]);
}

/**
 * Draws one connection as square dots along the segment.
 *
 * @param context - Render context.
 * @param input - Segment and style input.
 * @returns Nothing.
 */
function drawSquareDottedConnection(
  context: CanvasRenderingContext2D,
  input: {
    fromXPx: number;
    fromYPx: number;
    toXPx: number;
    toYPx: number;
    color: string;
    lineWidthPx: number;
  },
): void {
  const deltaXPx = input.toXPx - input.fromXPx;
  const deltaYPx = input.toYPx - input.fromYPx;
  const segmentLengthPx = Math.hypot(deltaXPx, deltaYPx);
  if (segmentLengthPx <= 0) {
    return;
  }

  const squareSidePx = 2;
  const stepDistancePx = Math.max(
    (squareSidePx + 2) * 0.8,
    (input.lineWidthPx * 2.6) * 0.8,
  );
  const directionXPx = deltaXPx / segmentLengthPx;
  const directionYPx = deltaYPx / segmentLengthPx;
  const halfSquareSidePx = squareSidePx * 0.5;

  context.fillStyle = input.color;

  // Step 1: Stamp dots at fixed step distances so spacing is consistent
  // regardless of segment length.
  for (
    let traveledDistancePx = 0;
    traveledDistancePx <= segmentLengthPx;
    traveledDistancePx += stepDistancePx
  ) {
    const centerXPx = input.fromXPx + directionXPx * traveledDistancePx;
    const centerYPx = input.fromYPx + directionYPx * traveledDistancePx;

    const axisAlignedCenterXPx =
      Math.round(centerXPx + directionXPx * 0.01) -
      Math.round(directionXPx * 0.01);
    const axisAlignedCenterYPx =
      Math.round(centerYPx + directionYPx * 0.01) -
      Math.round(directionYPx * 0.01);

    context.fillRect(
      axisAlignedCenterXPx - halfSquareSidePx,
      axisAlignedCenterYPx - halfSquareSidePx,
      squareSidePx,
      squareSidePx,
    );
  }

  // Step 2: Ensure the segment endpoint is always represented.
  if (segmentLengthPx > 0) {
    context.fillRect(
      Math.round(input.toXPx - halfSquareSidePx),
      Math.round(input.toYPx - halfSquareSidePx),
      squareSidePx,
      squareSidePx,
    );
  }
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
  biasScale: DynamicColorScale,
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
      : resolveTierColor(
          nodeBias,
          biasScale.tiers,
          biasScale.aboveTierColor,
        );
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
  colorScales: NetworkVisualizationColorScales,
): void {
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ??
    context.canvas.width;
  if (viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX) {
    return;
  }

  // Step 1: Build legend row models for connection and bias tiers.
  const connectionLegendRows = createColorLegendRows(
    colorScales.connectionScale,
    'w',
  );
  const biasLegendRows = createColorLegendRows(
    colorScales.biasScale,
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
    const sampleStartXPx = legendLeftPx + 8;
    const sampleEndXPx = legendLeftPx + 28;
    const sampleYPx = rowTopPx + 4;

    if (legendRow.maximumValue <= 0) {
      drawSquareDottedConnection(context, {
        fromXPx: sampleStartXPx,
        fromYPx: sampleYPx,
        toXPx: sampleEndXPx,
        toYPx: sampleYPx,
        color: legendRow.color,
        lineWidthPx: FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX,
      });
    } else {
      context.strokeStyle = legendRow.color;
      context.lineWidth = FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX;
      context.beginPath();
      context.moveTo(sampleStartXPx, sampleYPx);
      context.lineTo(sampleEndXPx, sampleYPx);
      context.stroke();
    }

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
  network: Network | undefined,
): NetworkLegendLayout {
  const colorScales = resolveNetworkVisualizationColorScales(network);
  const connectionLegendRows = createColorLegendRows(
    colorScales.connectionScale,
    'w',
  );
  const biasLegendRows = createColorLegendRows(
    colorScales.biasScale,
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
  const connectionScale = createDynamicColorScale([connectionWeight], {
    minimumValue: -1,
    maximumValue: 1,
  });
  return resolveTierColor(
    connectionWeight,
    connectionScale.tiers,
    connectionScale.aboveTierColor,
  );
}

/**
 * Resolves bias color for a raw node bias.
 *
 * @param nodeBias - Node bias.
 * @returns Tier color.
 */
export function resolveBiasRangeColor(nodeBias: number): string {
  const biasScale = createDynamicColorScale([nodeBias], {
    minimumValue: -1,
    maximumValue: 1,
  });
  return resolveTierColor(nodeBias, biasScale.tiers, biasScale.aboveTierColor);
}

/**
 * Resolves dynamic connection/bias color scales from the active network range.
 *
 * @param network - Active network.
 * @returns Dynamic scales used by graph drawing and legend rows.
 */
export function resolveNetworkVisualizationColorScales(
  network: Network | undefined,
): NetworkVisualizationColorScales {
  const connectionValues = ((network?.connections ?? []) as Array<{
    weight?: number;
  }>)
    .map((connection) => Number(connection.weight ?? 0))
    .filter((weight) => Number.isFinite(weight));

  const biasValues = ((network?.nodes ?? []) as Array<{
    type?: string;
    bias?: number;
  }>)
    .filter((node) => node.type !== 'output')
    .map((node) => Number(node.bias ?? 0))
    .filter((bias) => Number.isFinite(bias));

  return {
    connectionScale: createDynamicColorScale(connectionValues, {
      minimumValue: -1,
      maximumValue: 1,
    }),
    biasScale: createDynamicColorScale(biasValues, {
      minimumValue: -1,
      maximumValue: 1,
    }),
  };
}

/**
 * Creates a linear color scale that spans the observed value range.
 *
 * @param values - Runtime values to map.
 * @param fallbackRange - Fallback range when values are empty/non-finite.
 * @returns Dynamic color scale.
 */
function createDynamicColorScale(
  values: number[],
  fallbackRange: { minimumValue: number; maximumValue: number },
): DynamicColorScale {
  const finiteValues = values.filter((value) => Number.isFinite(value));
  const observedMinimumValue =
    finiteValues.length > 0
      ? Math.min(...finiteValues)
      : fallbackRange.minimumValue;
  const observedMaximumValue =
    finiteValues.length > 0
      ? Math.max(...finiteValues)
      : fallbackRange.maximumValue;
  const hasRange = observedMaximumValue > observedMinimumValue;

  const minimumValue = hasRange
    ? observedMinimumValue
    : observedMinimumValue - Math.max(1e-6, Math.abs(observedMinimumValue) * 0.01);
  const maximumValue = hasRange
    ? observedMaximumValue
    : observedMaximumValue + Math.max(1e-6, Math.abs(observedMaximumValue) * 0.01);

  const dynamicTiers = resolveSignedDynamicTiers(minimumValue, maximumValue);
  const aboveTierColor =
    dynamicTiers.at(-1)?.color ?? FLAPPY_NEON_PALETTE.currentRunText;

  return {
    minimumValue,
    maximumValue,
    tiers: dynamicTiers,
    aboveTierColor,
  };
}

/**
 * Resolves tier colors by sign so negative and positive sides use distinct ramps.
 *
 * @param minimumValue - Observed minimum value.
 * @param maximumValue - Observed maximum value.
 * @returns Ordered dynamic tiers.
 */
function resolveSignedDynamicTiers(
  minimumValue: number,
  maximumValue: number,
): ColorTier[] {
  const negativePaletteLowToHigh = [
    ...FLAPPY_LIGHT_NEON_RAMP.toReversed(),
    ...FLAPPY_CENTER_BLUE_RAMP,
  ] as const;
  const positivePaletteLowToHigh = [
    ...FLAPPY_CENTER_BLUE_RAMP,
    ...FLAPPY_REGULAR_NEON_RAMP,
  ] as const;

  if (minimumValue >= 0) {
    return createLinearColorTiers({
      minimumValue,
      maximumValue,
      palette: positivePaletteLowToHigh,
    });
  }

  if (maximumValue <= 0) {
    return createLinearColorTiers({
      minimumValue,
      maximumValue,
      palette: negativePaletteLowToHigh,
    });
  }

  const negativeTiers = createLinearColorTiers({
    minimumValue,
    maximumValue: 0,
    palette: negativePaletteLowToHigh,
  });
  const positiveTiers = createLinearColorTiers({
    minimumValue: 0,
    maximumValue,
    palette: positivePaletteLowToHigh,
  });
  return [...negativeTiers, ...positiveTiers];
}

/**
 * Creates linearly spaced color tiers between observed min and max.
 *
 * @param input - Tier creation options.
 * @returns Ordered tier list.
 */
function createLinearColorTiers(input: {
  minimumValue: number;
  maximumValue: number;
  palette: readonly string[];
}): ColorTier[] {
  const paletteSize = Math.max(1, input.palette.length);
  const range = Math.max(1e-12, input.maximumValue - input.minimumValue);
  const step = range / paletteSize;

  return input.palette.map((color, colorIndex) => ({
    upperBound:
      colorIndex === paletteSize - 1
        ? input.maximumValue
        : input.minimumValue + step * (colorIndex + 1),
    color,
  }));
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
 * Formats a legend bound with fixed precision.
 *
 * @param value - Numeric bound.
 * @returns Formatted bound string.
 */
function formatLegendBound(value: number): string {
  return value.toFixed(2);
}
