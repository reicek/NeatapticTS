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
  FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR,
  FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX,
} from '../../constants/constants';
import { clamp } from '../browser-entry.math.utils';
import type {
  ColorLegendRow,
  NetworkVisualizationAnimatedHoveredNode,
  NetworkNodeDimensionsLike,
  NetworkLegendLayout,
  PositionedNetworkNodeLike,
  VisualNetworkConnectionLike,
} from '../browser-entry.types';
import {
  FLAPPY_NETWORK_DEFAULT_CONNECTION_ALPHA,
  FLAPPY_NETWORK_DISABLED_CONNECTION_DASH_PATTERN,
  FLAPPY_NETWORK_DIMMED_CONNECTION_ALPHA,
  FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX,
  FLAPPY_NETWORK_HEADER_PADDING_PX,
  FLAPPY_NETWORK_HIGHLIGHT_CONNECTION_ALPHA,
  FLAPPY_NETWORK_HIDDEN_NODE_STROKE_WIDTH_PX,
  FLAPPY_NETWORK_HIDDEN_NODE_VERTICAL_PADDING_PX,
  FLAPPY_NETWORK_HOVERED_NODE_STROKE_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_ARCHITECTURE_GAP_PX,
  FLAPPY_NETWORK_LEGEND_BIAS_LABEL_X_PX,
  FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_SIZE_PX,
  FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_X_PX,
  FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_Y_PX,
  FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_LABEL_X_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_END_X_PX,
  FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_Y_OFFSET_PX,
  FLAPPY_NETWORK_LEGEND_HEADER_TOP_PADDING_PX,
  FLAPPY_NETWORK_LEGEND_MIN_ARCHITECTURE_TOP_PX,
  FLAPPY_NETWORK_MIN_RENDER_NODE_HEIGHT_PX,
  FLAPPY_NETWORK_NEGATIVE_CONNECTION_DASH_PATTERN,
  FLAPPY_NETWORK_OUTPUT_NODE_HEIGHT_REDUCTION_PX,
  FLAPPY_NETWORK_OUTPUT_NODE_STROKE_WIDTH_PX,
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
 * Canvas drawing helpers for the network visualization panel.
 *
 * This module turns prepared topology, color scales, and legend layout into the
 * actual rendered network view. The emphasis is educational readability: clear
 * topology, readable bias labels, and a legend that explains the neon encoding.
 */

const FLAPPY_MULTILINE_LABEL_SEPARATOR = '\n';
const FLAPPY_CANVAS_TEXT_ALIGN_LEFT: CanvasTextAlign = 'left';
const FLAPPY_CANVAS_TEXT_ALIGN_CENTER: CanvasTextAlign = 'center';
const FLAPPY_CANVAS_TEXT_BASELINE_TOP: CanvasTextBaseline = 'top';
const FLAPPY_CANVAS_TEXT_BASELINE_ALPHABETIC: CanvasTextBaseline = 'alphabetic';
const FLAPPY_CANVAS_LINE_CAP_BUTT: CanvasLineCap = 'butt';
const FLAPPY_CANVAS_LINE_CAP_SQUARE: CanvasLineCap = 'square';
const FLAPPY_TRANSPARENT_CANVAS_COLOR = 'transparent';
const FLAPPY_NETWORK_OUTPUT_NODE_TYPE = 'output';
const FLAPPY_NETWORK_LEGEND_TITLE = 'Legend';
const FLAPPY_NETWORK_CONNECTION_SECTION_TITLE = 'Connection weight';
const FLAPPY_NETWORK_BIAS_SECTION_TITLE = 'Node bias';
const FLAPPY_CONNECTION_LEGEND_SYMBOL = 'w';
const FLAPPY_BIAS_LEGEND_SYMBOL = 'b';
const EMPTY_CONNECTION_DASH_PATTERN: number[] = [];

type HoveredNodeAnimationState = {
  intensityByNodeIndex: ReadonlyMap<number, number>;
  maximumIntensity: number;
};

const EMPTY_HOVERED_NODE_INTENSITY_BY_NODE_INDEX = new Map<number, number>();
const EMPTY_HOVERED_NODE_ANIMATION_STATE: HoveredNodeAnimationState = {
  intensityByNodeIndex: EMPTY_HOVERED_NODE_INTENSITY_BY_NODE_INDEX,
  maximumIntensity: 0,
};

type WeightedConnectionScene = {
  fromPosition: PositionedNetworkNodeLike;
  toPosition: PositionedNetworkNodeLike;
  connectionColor: string;
  connectionOpacity: number;
  dashPattern: number[];
  lineCap: CanvasLineCap;
};

type ConnectionStrokeRequest = {
  fromXPx: number;
  fromYPx: number;
  toXPx: number;
  toYPx: number;
  color: string;
  lineWidthPx: number;
  opacity: number;
  dashPattern: number[];
  lineCap: CanvasLineCap;
};

type BiasNodeLabelMetrics = {
  labelAscentPx: number;
  labelDescentPx: number;
  measuredLabelHeightPx: number;
  labelFont: string;
};

type BiasNodePaintStyle = {
  isOutputNode: boolean;
  nodeFillColor: string;
  nodeStrokeColor: string;
  nodeStrokeWidthPx: number;
};

type BiasNodeScene = BiasNodePaintStyle & {
  hoverIntensity: number;
  positionedNode: PositionedNetworkNodeLike;
  nodeLabel: string;
  nodeRectLeftPx: number;
  nodeRectTopPx: number;
  resolvedNodeHeightPx: number;
  labelBaselineYPx: number;
  labelFont: string;
};

type LegendSceneContext = NetworkLegendLayout & {
  architectureLines: string[];
  architectureTextTopPx: number;
  connectionLegendRows: ColorLegendRow[];
  biasLegendRows: ColorLegendRow[];
};

/**
 * Draws weighted connection lines.
 *
 * Connection styling carries semantic meaning: color encodes magnitude and sign,
 * while dash patterns help distinguish disabled or negative
 * edges in a way that still reads quickly on a dense graph.
 *
 * @param context - Render context.
 * @param runtimeConnections - Runtime connection list.
 * @param positionByNodeIndex - Node layout map.
 * @param connectionScale - Dynamic connection color scale.
 * @param animatedHoveredNodes - Optional animated hovered-node samples owned by the host canvas.
 * @returns Nothing.
 */
export function drawWeightedConnectionsLayer(
  context: CanvasRenderingContext2D,
  runtimeConnections: VisualNetworkConnectionLike[],
  positionByNodeIndex: Map<number, PositionedNetworkNodeLike>,
  connectionScale: DynamicColorScale,
  animatedHoveredNodes?: readonly NetworkVisualizationAnimatedHoveredNode[],
): void {
  context.shadowBlur = 0;
  context.shadowColor = FLAPPY_TRANSPARENT_CANVAS_COLOR;

  const hoveredNodeAnimationState =
    resolveHoveredNodeAnimationState(animatedHoveredNodes);

  // Step 1: Resolve and paint each drawable connection without building an intermediate list.
  runtimeConnections.forEach((runtimeConnection) => {
    const weightedConnectionScene = resolveWeightedConnectionScene(
      runtimeConnection,
      positionByNodeIndex,
      connectionScale,
      hoveredNodeAnimationState,
    );
    if (!weightedConnectionScene) {
      return;
    }

    drawWeightedConnectionScene(context, weightedConnectionScene);
  });

  // Step 2: Reset the dash pattern so later layers inherit a clean canvas state.
  context.setLineDash([]);
  context.globalAlpha = 1;
  context.lineCap = FLAPPY_CANVAS_LINE_CAP_BUTT;
}

/**
 * Draws all network nodes with bias labels.
 *
 * The node layer pairs each rectangle with a compact bias label so the panel can
 * show both topology and a lightweight hint of parameter state.
 *
 * @param context - Render context.
 * @param positionedNodes - Positioned nodes.
 * @param nodeDimensions - Node dimensions.
 * @param biasScale - Dynamic bias color scale.
 * @param animatedHoveredNodes - Optional animated hovered-node samples owned by the host canvas.
 * @returns Nothing.
 */
export function drawBiasNodesLayer(
  context: CanvasRenderingContext2D,
  positionedNodes: PositionedNetworkNodeLike[],
  nodeDimensions: NetworkNodeDimensionsLike,
  biasScale: DynamicColorScale,
  animatedHoveredNodes?: readonly NetworkVisualizationAnimatedHoveredNode[],
): void {
  context.shadowBlur = 0;
  context.shadowColor = FLAPPY_TRANSPARENT_CANVAS_COLOR;

  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const hoveredNodeAnimationState =
    resolveHoveredNodeAnimationState(animatedHoveredNodes);

  // Step 1: Resolve and paint each node without building an intermediate scene array.
  positionedNodes.forEach((positionedNode) => {
    const biasNodeScene = resolveBiasNodeScene(
      context,
      positionedNode,
      nodeDimensions,
      halfNodeWidthPx,
      biasScale,
      hoveredNodeAnimationState.intensityByNodeIndex.get(
        positionedNode.node.index,
      ) ?? 0,
    );

    drawBiasNodeScene(context, biasNodeScene, nodeDimensions.widthPx);
  });
}

/**
 * Draws network architecture header text.
 *
 * The header gives viewers a compact architecture summary before they inspect
 * individual nodes and edges.
 *
 * @param context - Render context.
 * @param architectureLabel - Header label.
 * @returns Nothing.
 */
export function drawNetworkVisualizationHeader(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
): void {
  // Step 1: Normalize the potentially multiline architecture label into rows.
  const headerLines = architectureLabel.split(FLAPPY_MULTILINE_LABEL_SEPARATOR);

  // Step 2: Paint the header block using the shared left-aligned text helper.
  drawLeftAlignedTextRows(context, {
    lines: headerLines,
    leftPx: FLAPPY_NETWORK_HEADER_PADDING_PX,
    topPx: FLAPPY_NETWORK_HEADER_PADDING_PX,
    lineHeightPx: FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX,
    font: `${FLAPPY_NETWORK_HEADER_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`,
    fillStyle: FLAPPY_NETWORK_HEADER_TEXT_COLOR,
  });
}

/**
 * Draws the color legend for connections and node bias values.
 *
 * This legend is what turns the panel from "colorful art" into an interpretable
 * instrument: it tells the viewer what each weight and bias color actually
 * means numerically.
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
  // Step 1: Skip legend work entirely when the viewport intentionally hides overlays.
  if (shouldHideNetworkColorLegend(context)) {
    return;
  }

  // Step 2: Resolve legend rows, layout, and derived text bounds once up front.
  const legendSceneContext = resolveLegendSceneContext(
    context,
    architectureLabel,
    colorScales,
  );

  // Step 3: Paint the architecture label that sits above the legend box.
  drawLegendArchitectureLabel(context, legendSceneContext);

  // Step 4: Paint the legend container and section header.
  drawLegendFrame(context, legendSceneContext);
  drawLegendHeader(context, legendSceneContext);

  // Step 5: Paint the connection and bias sections using the resolved rows.
  drawLegendConnectionSection(context, legendSceneContext);
  drawLegendBiasSection(context, legendSceneContext);
}

/**
 * Resolves a renderable connection scene from runtime data and node positions.
 *
 * @param runtimeConnection - Candidate runtime connection.
 * @param positionByNodeIndex - Node layout map.
 * @param connectionScale - Connection color scale.
 * @returns Renderable connection scene, when both endpoint nodes exist.
 */
function resolveWeightedConnectionScene(
  runtimeConnection: VisualNetworkConnectionLike,
  positionByNodeIndex: Map<number, PositionedNetworkNodeLike>,
  connectionScale: DynamicColorScale,
  hoveredNodeAnimationState: HoveredNodeAnimationState,
): WeightedConnectionScene | undefined {
  // Step 1: Resolve both endpoint indices and exit early when either side is missing.
  const fromNodeIndex = runtimeConnection.from?.index;
  const toNodeIndex = runtimeConnection.to?.index;
  if (typeof fromNodeIndex !== 'number' || typeof toNodeIndex !== 'number') {
    return undefined;
  }

  // Step 2: Resolve endpoint coordinates from the current node layout map.
  const fromPosition = positionByNodeIndex.get(fromNodeIndex);
  const toPosition = positionByNodeIndex.get(toNodeIndex);
  if (!fromPosition || !toPosition) {
    return undefined;
  }

  // Step 3: Convert weight semantics into color, alpha, and stroke styling.
  const connectionWeight = runtimeConnection.weight ?? 0;
  const connectionEnabled = runtimeConnection.enabled !== false;
  const baseColor = resolveTierColor(
    connectionWeight,
    connectionScale.tiers,
    connectionScale.aboveTierColor,
  );
  const adjacentHoverIntensity = resolveAdjacentConnectionHoverIntensity(
    runtimeConnection,
    hoveredNodeAnimationState.intensityByNodeIndex,
  );

  return {
    fromPosition,
    toPosition,
    connectionColor: baseColor,
    connectionOpacity: resolveConnectionOpacity(
      hoveredNodeAnimationState.maximumIntensity,
      adjacentHoverIntensity,
    ),
    dashPattern: resolveConnectionDashPattern(
      connectionWeight < 0,
      connectionEnabled,
    ),
    lineCap:
      connectionWeight < 0
        ? FLAPPY_CANVAS_LINE_CAP_SQUARE
        : FLAPPY_CANVAS_LINE_CAP_BUTT,
  };
}

/**
 * Draws a previously resolved weighted connection scene.
 *
 * @param context - Render context.
 * @param weightedConnectionScene - Render-ready connection scene.
 * @returns Nothing.
 */
function drawWeightedConnectionScene(
  context: CanvasRenderingContext2D,
  weightedConnectionScene: WeightedConnectionScene,
): void {
  // Step 1: Paint the resolved connection stroke with the requested alpha and dash style.
  drawConnectionStroke(context, {
    fromXPx: weightedConnectionScene.fromPosition.xPx,
    fromYPx: weightedConnectionScene.fromPosition.yPx,
    toXPx: weightedConnectionScene.toPosition.xPx,
    toYPx: weightedConnectionScene.toPosition.yPx,
    color: weightedConnectionScene.connectionColor,
    lineWidthPx: FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX,
    opacity: weightedConnectionScene.connectionOpacity,
    dashPattern: weightedConnectionScene.dashPattern,
    lineCap: weightedConnectionScene.lineCap,
  });
}

function resolveConnectionDashPattern(
  isNegativeConnection: boolean,
  connectionEnabled: boolean,
): number[] {
  if (isNegativeConnection) {
    return FLAPPY_NETWORK_NEGATIVE_CONNECTION_DASH_PATTERN;
  }

  if (!connectionEnabled) {
    return FLAPPY_NETWORK_DISABLED_CONNECTION_DASH_PATTERN;
  }

  return EMPTY_CONNECTION_DASH_PATTERN;
}

function drawConnectionStroke(
  context: CanvasRenderingContext2D,
  connectionStrokeRequest: ConnectionStrokeRequest,
): void {
  context.globalAlpha = connectionStrokeRequest.opacity;
  context.strokeStyle = connectionStrokeRequest.color;
  context.lineWidth = connectionStrokeRequest.lineWidthPx;
  context.setLineDash(connectionStrokeRequest.dashPattern);
  context.lineCap = connectionStrokeRequest.lineCap;
  context.beginPath();
  context.moveTo(
    connectionStrokeRequest.fromXPx,
    connectionStrokeRequest.fromYPx,
  );
  context.lineTo(connectionStrokeRequest.toXPx, connectionStrokeRequest.toYPx);
  context.stroke();
}

function resolveAdjacentConnectionHoverIntensity(
  runtimeConnection: VisualNetworkConnectionLike,
  hoveredNodeIntensityByNodeIndex: ReadonlyMap<number, number>,
): number {
  const fromNodeIndex = runtimeConnection.from?.index;
  const toNodeIndex = runtimeConnection.to?.index;
  return Math.max(
    typeof fromNodeIndex === 'number'
      ? (hoveredNodeIntensityByNodeIndex.get(fromNodeIndex) ?? 0)
      : 0,
    typeof toNodeIndex === 'number'
      ? (hoveredNodeIntensityByNodeIndex.get(toNodeIndex) ?? 0)
      : 0,
  );
}

function resolveConnectionOpacity(
  maximumHoverIntensity: number,
  adjacentHoverIntensity: number,
): number {
  // Step 1: Keep the shared default opacity when no animated hover is active.
  if (maximumHoverIntensity <= 0) {
    return FLAPPY_NETWORK_DEFAULT_CONNECTION_ALPHA;
  }

  // Step 2: Dim the whole graph while any hover remains active.
  const dimmedConnectionOpacity = interpolateNumber(
    FLAPPY_NETWORK_DEFAULT_CONNECTION_ALPHA,
    FLAPPY_NETWORK_DIMMED_CONNECTION_ALPHA,
    maximumHoverIntensity,
  );

  // Step 3: Lift directly adjacent connections back toward the highlighted opacity.
  if (adjacentHoverIntensity <= 0) {
    return dimmedConnectionOpacity;
  }

  return interpolateNumber(
    dimmedConnectionOpacity,
    FLAPPY_NETWORK_HIGHLIGHT_CONNECTION_ALPHA,
    adjacentHoverIntensity,
  );
}

function resolveHoveredNodeAnimationState(
  animatedHoveredNodes:
    | readonly NetworkVisualizationAnimatedHoveredNode[]
    | undefined,
): HoveredNodeAnimationState {
  if (!animatedHoveredNodes || animatedHoveredNodes.length === 0) {
    return EMPTY_HOVERED_NODE_ANIMATION_STATE;
  }

  const intensityByNodeIndex = new Map<number, number>();
  let maximumIntensity = 0;

  animatedHoveredNodes?.forEach((animatedHoveredNode) => {
    const resolvedIntensity = clamp(animatedHoveredNode.intensity, 0, 1);
    if (resolvedIntensity <= 0) {
      return;
    }

    intensityByNodeIndex.set(animatedHoveredNode.nodeIndex, resolvedIntensity);
    maximumIntensity = Math.max(maximumIntensity, resolvedIntensity);
  });

  return {
    intensityByNodeIndex,
    maximumIntensity,
  };
}

/**
 * Resolves all paint attributes needed to render a single node.
 *
 * @param context - Render context.
 * @param positionedNode - Positioned node payload.
 * @param nodeDimensions - Shared node dimensions.
 * @param halfNodeWidthPx - Cached half node width.
 * @param biasScale - Bias color scale.
 * @returns Paint-ready node scene.
 */
function resolveBiasNodeScene(
  context: CanvasRenderingContext2D,
  positionedNode: PositionedNetworkNodeLike,
  nodeDimensions: NetworkNodeDimensionsLike,
  halfNodeWidthPx: number,
  biasScale: DynamicColorScale,
  hoverIntensity: number,
): BiasNodeScene {
  // Step 1: Resolve color and emphasis rules from node type and bias value.
  const nodeLabel = formatNodeBiasLabel(positionedNode.node.bias);
  const biasNodePaintStyle = resolveBiasNodePaintStyle(
    positionedNode,
    biasScale,
  );

  // Step 2: Measure the label so the node rectangle can fit readable text.
  const biasNodeLabelMetrics = resolveBiasNodeLabelMetrics(
    context,
    nodeLabel,
    nodeDimensions,
  );
  const resolvedNodeHeightPx = resolveBiasNodeHeightPx(
    nodeDimensions,
    biasNodeLabelMetrics,
    biasNodePaintStyle.isOutputNode,
  );
  const halfNodeHeightPx = resolvedNodeHeightPx * 0.5;

  // Step 3: Convert center-based node positions into top-left render coordinates.
  return {
    ...biasNodePaintStyle,
    hoverIntensity,
    positionedNode,
    nodeLabel,
    nodeRectLeftPx: positionedNode.xPx - halfNodeWidthPx,
    nodeRectTopPx: positionedNode.yPx - halfNodeHeightPx,
    resolvedNodeHeightPx,
    labelBaselineYPx:
      positionedNode.yPx +
      (biasNodeLabelMetrics.labelAscentPx -
        biasNodeLabelMetrics.labelDescentPx) *
        0.5,
    labelFont: biasNodeLabelMetrics.labelFont,
  };
}

/**
 * Resolves node fill and stroke styling.
 *
 * @param positionedNode - Positioned node payload.
 * @param biasScale - Bias color scale.
 * @returns Node paint style.
 */
function resolveBiasNodePaintStyle(
  positionedNode: PositionedNetworkNodeLike,
  biasScale: DynamicColorScale,
): BiasNodePaintStyle {
  // Step 1: Determine whether the node is an output node, which changes both color and emphasis.
  const isOutputNode =
    positionedNode.node.type === FLAPPY_NETWORK_OUTPUT_NODE_TYPE;

  // Step 2: Resolve the fill and stroke styles for the active node category.
  return {
    isOutputNode,
    nodeFillColor: isOutputNode
      ? FLAPPY_NEON_PALETTE.currentRunText
      : resolveTierColor(
          positionedNode.node.bias,
          biasScale.tiers,
          biasScale.aboveTierColor,
        ),
    nodeStrokeColor: isOutputNode
      ? FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR
      : FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR,
    nodeStrokeWidthPx: isOutputNode
      ? FLAPPY_NETWORK_OUTPUT_NODE_STROKE_WIDTH_PX
      : FLAPPY_NETWORK_HIDDEN_NODE_STROKE_WIDTH_PX,
  };
}

/**
 * Measures a bias label and resolves its font declaration.
 *
 * @param context - Render context.
 * @param nodeLabel - Bias label string.
 * @param nodeDimensions - Shared node dimensions.
 * @returns Measured label metrics.
 */
function resolveBiasNodeLabelMetrics(
  context: CanvasRenderingContext2D,
  nodeLabel: string,
  nodeDimensions: NetworkNodeDimensionsLike,
): BiasNodeLabelMetrics {
  // Step 1: Clamp label height into the readable range for the current node size.
  const maximumReadableLabelHeightPx = Math.max(
    FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
    Math.floor(nodeDimensions.heightPx),
  );
  const computedLabelHeightPx = Math.floor(
    nodeDimensions.heightPx * FLAPPY_NETWORK_NODE_LABEL_SIZE_RATIO,
  );
  const labelHeightPx = clamp(
    computedLabelHeightPx,
    FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
    maximumReadableLabelHeightPx,
  );
  const labelFont = `${FLAPPY_NETWORK_NODE_LABEL_FONT_WEIGHT} ${labelHeightPx}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;

  // Step 2: Measure the label using the resolved font so baseline math stays stable.
  context.font = labelFont;
  context.textBaseline = FLAPPY_CANVAS_TEXT_BASELINE_ALPHABETIC;
  const labelMetrics = context.measureText(nodeLabel);
  const labelAscentPx =
    labelMetrics.actualBoundingBoxAscent || labelHeightPx * 0.72;
  const labelDescentPx =
    labelMetrics.actualBoundingBoxDescent || labelHeightPx * 0.28;

  return {
    labelAscentPx,
    labelDescentPx,
    measuredLabelHeightPx: Math.max(
      FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
      Math.ceil(labelAscentPx + labelDescentPx),
    ),
    labelFont,
  };
}

/**
 * Resolves node rectangle height from label metrics and node role.
 *
 * @param nodeDimensions - Shared node dimensions.
 * @param biasNodeLabelMetrics - Measured label metrics.
 * @param isOutputNode - Whether the node is an output node.
 * @returns Render height for the node rectangle.
 */
function resolveBiasNodeHeightPx(
  nodeDimensions: NetworkNodeDimensionsLike,
  biasNodeLabelMetrics: BiasNodeLabelMetrics,
  isOutputNode: boolean,
): number {
  // Step 1: Use a slightly tighter output-node box while enforcing a hard minimum height.
  if (isOutputNode) {
    return Math.max(
      FLAPPY_NETWORK_MIN_RENDER_NODE_HEIGHT_PX,
      nodeDimensions.heightPx - FLAPPY_NETWORK_OUTPUT_NODE_HEIGHT_REDUCTION_PX,
    );
  }

  // Step 2: Fit hidden and input nodes around the measured label height plus inner padding.
  return Math.max(
    FLAPPY_NETWORK_MIN_RENDER_NODE_HEIGHT_PX,
    Math.min(
      nodeDimensions.heightPx,
      biasNodeLabelMetrics.measuredLabelHeightPx +
        FLAPPY_NETWORK_HIDDEN_NODE_VERTICAL_PADDING_PX,
    ),
  );
}

/**
 * Draws a resolved node rectangle and optional bias label.
 *
 * @param context - Render context.
 * @param biasNodeScene - Paint-ready node scene.
 * @param nodeWidthPx - Shared node width.
 * @returns Nothing.
 */
function drawBiasNodeScene(
  context: CanvasRenderingContext2D,
  biasNodeScene: BiasNodeScene,
  nodeWidthPx: number,
): void {
  const hoveredStrokeWidthPx = resolveHoveredNodeStrokeWidth(biasNodeScene);

  // Step 1: Paint the node rectangle with its resolved fill, stroke, and hover emphasis.
  context.fillStyle = biasNodeScene.nodeFillColor;
  context.strokeStyle = biasNodeScene.nodeStrokeColor;
  context.lineWidth = hoveredStrokeWidthPx;
  context.fillRect(
    biasNodeScene.nodeRectLeftPx,
    biasNodeScene.nodeRectTopPx,
    nodeWidthPx,
    biasNodeScene.resolvedNodeHeightPx,
  );
  context.strokeRect(
    biasNodeScene.nodeRectLeftPx,
    biasNodeScene.nodeRectTopPx,
    nodeWidthPx,
    biasNodeScene.resolvedNodeHeightPx,
  );

  // Step 2: Skip label painting for output nodes, which render as emphasized solid blocks.
  if (biasNodeScene.isOutputNode) {
    return;
  }

  // Step 3: Paint the centered bias label inside the node rectangle.
  context.fillStyle = FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR;
  context.font = biasNodeScene.labelFont;
  context.textAlign = FLAPPY_CANVAS_TEXT_ALIGN_CENTER;
  context.textBaseline = FLAPPY_CANVAS_TEXT_BASELINE_ALPHABETIC;
  context.fillText(
    biasNodeScene.nodeLabel,
    biasNodeScene.positionedNode.xPx,
    biasNodeScene.labelBaselineYPx,
  );
}

function resolveHoveredNodeStrokeWidth(biasNodeScene: BiasNodeScene): number {
  if (biasNodeScene.hoverIntensity <= 0) {
    return biasNodeScene.nodeStrokeWidthPx;
  }

  return interpolateNumber(
    biasNodeScene.nodeStrokeWidthPx,
    Math.max(
      biasNodeScene.nodeStrokeWidthPx,
      FLAPPY_NETWORK_HOVERED_NODE_STROKE_WIDTH_PX,
    ),
    biasNodeScene.hoverIntensity,
  );
}

function interpolateNumber(
  startValue: number,
  endValue: number,
  interpolationRatio: number,
): number {
  const clampedInterpolationRatio = clamp(interpolationRatio, 0, 1);
  return startValue + (endValue - startValue) * clampedInterpolationRatio;
}

/**
 * Determines whether the responsive viewport intentionally hides the overlay legend.
 *
 * @param context - Render context.
 * @returns True when the legend should be omitted.
 */
function shouldHideNetworkColorLegend(
  context: CanvasRenderingContext2D,
): boolean {
  // Step 1: Prefer the live window width when available, otherwise fall back to canvas width.
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ??
    context.canvas.width;
  return viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX;
}

/**
 * Resolves the legend rows, layout, and architecture label bounds.
 *
 * @param context - Render context.
 * @param architectureLabel - Multiline architecture label.
 * @param colorScales - Connection and bias color scales.
 * @returns Legend scene context.
 */
function resolveLegendSceneContext(
  context: CanvasRenderingContext2D,
  architectureLabel: string,
  colorScales: NetworkVisualizationColorScales,
): LegendSceneContext {
  // Step 1: Build the connection and bias legend rows from the active scales.
  const connectionLegendRows = createColorLegendRows(
    colorScales.connectionScale,
    FLAPPY_CONNECTION_LEGEND_SYMBOL,
  );
  const biasLegendRows = createColorLegendRows(
    colorScales.biasScale,
    FLAPPY_BIAS_LEGEND_SYMBOL,
  );
  const networkLegendLayout = resolveNetworkLegendLayout(
    context,
    connectionLegendRows,
    biasLegendRows,
  );

  // Step 2: Resolve the multiline architecture label bounds above the legend box.
  const architectureLines = architectureLabel.split(
    FLAPPY_MULTILINE_LABEL_SEPARATOR,
  );
  const architectureTextTopPx = Math.max(
    FLAPPY_NETWORK_LEGEND_MIN_ARCHITECTURE_TOP_PX,
    networkLegendLayout.legendTopPx -
      architectureLines.length * FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX -
      FLAPPY_NETWORK_LEGEND_ARCHITECTURE_GAP_PX,
  );

  return {
    ...networkLegendLayout,
    architectureLines,
    architectureTextTopPx,
    connectionLegendRows,
    biasLegendRows,
  };
}

/**
 * Draws the architecture label block above the legend frame.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @returns Nothing.
 */
function drawLegendArchitectureLabel(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void {
  // Step 1: Paint the architecture lines aligned to the legend frame left edge.
  drawLeftAlignedTextRows(context, {
    lines: legendSceneContext.architectureLines,
    leftPx:
      legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX,
    topPx: legendSceneContext.architectureTextTopPx,
    lineHeightPx: FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX,
    font: `${FLAPPY_NETWORK_HEADER_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`,
    fillStyle: FLAPPY_NETWORK_HEADER_TEXT_COLOR,
  });
}

/**
 * Draws the legend container frame.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @returns Nothing.
 */
function drawLegendFrame(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void {
  // Step 1: Paint the legend background block.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_BACKGROUND;
  context.strokeStyle = FLAPPY_NEON_PALETTE.statusText;
  context.lineWidth = 1;
  context.fillRect(
    legendSceneContext.legendLeftPx,
    legendSceneContext.legendTopPx,
    legendSceneContext.legendWidthPx,
    legendSceneContext.legendHeightPx,
  );
  context.strokeRect(
    legendSceneContext.legendLeftPx,
    legendSceneContext.legendTopPx,
    legendSceneContext.legendWidthPx,
    legendSceneContext.legendHeightPx,
  );
}

/**
 * Draws the legend title row.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @returns Nothing.
 */
function drawLegendHeader(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void {
  // Step 1: Resolve the legend title font based on the compact layout mode.
  const legendFontSizePx = legendSceneContext.compactLegend
    ? FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX
    : FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX;

  // Step 2: Paint the title inside the legend frame.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_HEADER_COLOR;
  context.font = `${legendFontSizePx}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = FLAPPY_CANVAS_TEXT_ALIGN_LEFT;
  context.textBaseline = FLAPPY_CANVAS_TEXT_BASELINE_TOP;
  context.fillText(
    FLAPPY_NETWORK_LEGEND_TITLE,
    legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX,
    legendSceneContext.legendTopPx +
      FLAPPY_NETWORK_LEGEND_HEADER_TOP_PADDING_PX,
  );
}

/**
 * Draws the connection-weight legend section.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @returns Nothing.
 */
function drawLegendConnectionSection(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void {
  // Step 1: Paint the section title just below the legend header.
  const connectionSectionTopPx =
    legendSceneContext.legendTopPx + legendSceneContext.legendHeaderHeightPx;
  context.fillStyle = FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR;
  context.font = `${legendSceneContext.compactLegend ? FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX : FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = FLAPPY_CANVAS_TEXT_ALIGN_LEFT;
  context.textBaseline = FLAPPY_CANVAS_TEXT_BASELINE_TOP;
  context.fillText(
    FLAPPY_NETWORK_CONNECTION_SECTION_TITLE,
    legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX,
    connectionSectionTopPx,
  );

  // Step 2: Paint each legend row sample and label.
  legendSceneContext.connectionLegendRows.forEach(
    (connectionLegendRow, connectionLegendRowIndex) => {
      const connectionRowTopPx =
        connectionSectionTopPx +
        legendSceneContext.legendSectionTitleHeightPx +
        connectionLegendRowIndex * legendSceneContext.legendRowHeightPx;
      drawLegendConnectionRow(
        context,
        legendSceneContext,
        connectionLegendRow,
        connectionRowTopPx,
      );
    },
  );
}

/**
 * Draws a single connection legend row.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @param connectionLegendRow - Legend row.
 * @param connectionRowTopPx - Row top coordinate.
 * @returns Nothing.
 */
function drawLegendConnectionRow(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
  connectionLegendRow: ColorLegendRow,
  connectionRowTopPx: number,
): void {
  // Step 1: Resolve the sample line coordinates for the row.
  const sampleStartXPx =
    legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX;
  const sampleEndXPx =
    legendSceneContext.legendLeftPx +
    FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_END_X_PX;
  const sampleYPx =
    connectionRowTopPx + FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_Y_OFFSET_PX;

  // Step 2: Paint the sample stroke using the same polarity styling as live edges.
  drawConnectionStroke(context, {
    fromXPx: sampleStartXPx,
    fromYPx: sampleYPx,
    toXPx: sampleEndXPx,
    toYPx: sampleYPx,
    color: connectionLegendRow.color,
    lineWidthPx: FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX,
    opacity: 1,
    dashPattern: resolveConnectionDashPattern(
      connectionLegendRow.maximumValue <= 0,
      true,
    ),
    lineCap:
      connectionLegendRow.maximumValue <= 0
        ? FLAPPY_CANVAS_LINE_CAP_SQUARE
        : FLAPPY_CANVAS_LINE_CAP_BUTT,
  });

  // Step 3: Paint the row label beside the sample.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR;
  context.fillText(
    connectionLegendRow.label,
    legendSceneContext.legendLeftPx +
      FLAPPY_NETWORK_LEGEND_CONNECTION_LABEL_X_PX,
    connectionRowTopPx - 1,
  );
}

/**
 * Draws the bias legend section.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @returns Nothing.
 */
function drawLegendBiasSection(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
): void {
  // Step 1: Resolve the bias section top from the connection section height.
  const biasSectionTopPx =
    legendSceneContext.legendTopPx +
    legendSceneContext.legendHeaderHeightPx +
    legendSceneContext.legendSectionTitleHeightPx +
    legendSceneContext.connectionLegendRows.length *
      legendSceneContext.legendRowHeightPx +
    legendSceneContext.legendSectionGapPx;
  context.fillStyle = FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR;
  context.font = `${legendSceneContext.compactLegend ? FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX : FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = FLAPPY_CANVAS_TEXT_ALIGN_LEFT;
  context.textBaseline = FLAPPY_CANVAS_TEXT_BASELINE_TOP;
  context.fillText(
    FLAPPY_NETWORK_BIAS_SECTION_TITLE,
    legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX,
    biasSectionTopPx,
  );

  // Step 2: Paint each bias swatch row beneath the section title.
  legendSceneContext.biasLegendRows.forEach(
    (biasLegendRow, biasLegendRowIndex) => {
      const biasRowTopPx =
        biasSectionTopPx +
        legendSceneContext.legendSectionTitleHeightPx +
        biasLegendRowIndex * legendSceneContext.legendRowHeightPx;
      drawLegendBiasRow(
        context,
        legendSceneContext,
        biasLegendRow,
        biasRowTopPx,
      );
    },
  );
}

/**
 * Draws a single bias legend row.
 *
 * @param context - Render context.
 * @param legendSceneContext - Legend scene context.
 * @param biasLegendRow - Legend row.
 * @param biasRowTopPx - Row top coordinate.
 * @returns Nothing.
 */
function drawLegendBiasRow(
  context: CanvasRenderingContext2D,
  legendSceneContext: LegendSceneContext,
  biasLegendRow: ColorLegendRow,
  biasRowTopPx: number,
): void {
  // Step 1: Paint the color swatch that represents the bias bucket.
  context.fillStyle = biasLegendRow.color;
  context.beginPath();
  context.rect(
    legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_X_PX,
    biasRowTopPx + FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_Y_PX,
    FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_SIZE_PX,
    FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_SIZE_PX,
  );
  context.fill();

  // Step 2: Paint the descriptive label beside the swatch.
  context.fillStyle = FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR;
  context.fillText(
    biasLegendRow.label,
    legendSceneContext.legendLeftPx + FLAPPY_NETWORK_LEGEND_BIAS_LABEL_X_PX,
    biasRowTopPx - 1,
  );
}

/**
 * Draws multiline text rows aligned to a fixed left edge.
 *
 * @param context - Render context.
 * @param request - Multiline text draw request.
 * @returns Nothing.
 */
function drawLeftAlignedTextRows(
  context: CanvasRenderingContext2D,
  request: {
    lines: string[];
    leftPx: number;
    topPx: number;
    lineHeightPx: number;
    font: string;
    fillStyle: string;
  },
): void {
  // Step 1: Configure the shared text state used by every row.
  context.fillStyle = request.fillStyle;
  context.font = request.font;
  context.textAlign = FLAPPY_CANVAS_TEXT_ALIGN_LEFT;
  context.textBaseline = FLAPPY_CANVAS_TEXT_BASELINE_TOP;

  // Step 2: Paint the rows with fixed vertical spacing.
  request.lines.forEach((lineText, lineIndex) => {
    context.fillText(
      lineText,
      request.leftPx,
      request.topPx + lineIndex * request.lineHeightPx,
    );
  });
}
