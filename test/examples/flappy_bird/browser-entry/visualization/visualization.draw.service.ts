import {
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_NETWORK_HEADER_FONT_SIZE_PX,
  FLAPPY_NETWORK_HEADER_TEXT_COLOR,
  FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR,
  FLAPPY_NETWORK_LEGEND_BACKGROUND,
  FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR,
  FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR,
  FLAPPY_NETWORK_LEGEND_HEADER_COLOR,
  FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX,
  FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR,
  FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
  FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR,
  FLAPPY_NETWORK_NODE_LABEL_FONT_WEIGHT,
  FLAPPY_NETWORK_NODE_LABEL_SIZE_RATIO,
  FLAPPY_NETWORK_OUTPUT_NODE_GLOW_COLOR,
  FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR,
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
} from '../../constants/constants';
import { applyAlphaToHexColor, clamp } from '../browser-entry.math.utils';
import type {
  NetworkNodeDimensionsLike,
  PositionedNetworkNodeLike,
  VisualNetworkConnectionLike,
} from '../browser-entry.types';
import {
  FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON,
  FLAPPY_NETWORK_DOTTED_CONNECTION_SQUARE_SIDE_PX,
  FLAPPY_NETWORK_DOTTED_CONNECTION_STEP_COMPACT_RATIO,
  FLAPPY_NETWORK_DOTTED_CONNECTION_WIDTH_SPACING_RATIO,
} from './visualization.constants';
import {
  createColorLegendRows,
  resolveNetworkLegendLayout,
} from './visualization.legend.utils';
import { formatNodeBiasLabel } from './visualization.topology.utils';
import type {
  DynamicColorScale,
  NetworkVisualizationColorScales,
} from './visualization.types';
import { resolveTierColor } from './visualization.colors.utils';

/**
 * Draws weighted connection lines.
 *
 * @param context - Render context.
 * @param runtimeConnections - Runtime connection list.
 * @param positionByNodeIndex - Node layout map.
 * @param connectionScale - Dynamic connection color scale.
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
 * Draws all network nodes with bias labels.
 *
 * @param context - Render context.
 * @param positionedNodes - Positioned nodes.
 * @param nodeDimensions - Node dimensions.
 * @param biasScale - Dynamic bias color scale.
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
      : resolveTierColor(nodeBias, biasScale.tiers, biasScale.aboveTierColor);
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
 * @param architectureLabel - Compact architecture description.
 * @param colorScales - Connection and bias color scales.
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

  const connectionLegendRows = createColorLegendRows(
    colorScales.connectionScale,
    'w',
  );
  const biasLegendRows = createColorLegendRows(colorScales.biasScale, 'b');
  const legendLayout = resolveNetworkLegendLayout(
    context,
    connectionLegendRows,
    biasLegendRows,
  );
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

  context.fillStyle = FLAPPY_NETWORK_LEGEND_BACKGROUND;
  context.strokeStyle = FLAPPY_NEON_PALETTE.statusText;
  context.lineWidth = 1;
  context.fillRect(legendLeftPx, legendTopPx, legendWidthPx, legendHeightPx);
  context.strokeRect(legendLeftPx, legendTopPx, legendWidthPx, legendHeightPx);

  context.fillStyle = FLAPPY_NETWORK_LEGEND_HEADER_COLOR;
  context.font = `${compactLegend ? FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX : FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.fillText('Legend', legendLeftPx + 8, legendTopPx + 6);

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

  const squareSidePx = FLAPPY_NETWORK_DOTTED_CONNECTION_SQUARE_SIDE_PX;
  const stepDistancePx = Math.max(
    (squareSidePx + 2) * FLAPPY_NETWORK_DOTTED_CONNECTION_STEP_COMPACT_RATIO,
    input.lineWidthPx *
      FLAPPY_NETWORK_DOTTED_CONNECTION_WIDTH_SPACING_RATIO *
      FLAPPY_NETWORK_DOTTED_CONNECTION_STEP_COMPACT_RATIO,
  );
  const directionXPx = deltaXPx / segmentLengthPx;
  const directionYPx = deltaYPx / segmentLengthPx;
  const halfSquareSidePx = squareSidePx * 0.5;

  context.fillStyle = input.color;

  for (
    let traveledDistancePx = 0;
    traveledDistancePx <= segmentLengthPx;
    traveledDistancePx += stepDistancePx
  ) {
    const centerXPx = input.fromXPx + directionXPx * traveledDistancePx;
    const centerYPx = input.fromYPx + directionYPx * traveledDistancePx;

    const axisAlignedCenterXPx =
      Math.round(
        centerXPx +
          directionXPx * FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON,
      ) -
      Math.round(
        directionXPx * FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON,
      );
    const axisAlignedCenterYPx =
      Math.round(
        centerYPx +
          directionYPx * FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON,
      ) -
      Math.round(
        directionYPx * FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON,
      );

    context.fillRect(
      axisAlignedCenterXPx - halfSquareSidePx,
      axisAlignedCenterYPx - halfSquareSidePx,
      squareSidePx,
      squareSidePx,
    );
  }

  context.fillRect(
    Math.round(input.toXPx - halfSquareSidePx),
    Math.round(input.toYPx - halfSquareSidePx),
    squareSidePx,
    squareSidePx,
  );
}
