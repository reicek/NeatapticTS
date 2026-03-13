import Network from '../../../../../src/architecture/network';
import {
  FLAPPY_NETWORK_ARCHITECTURE_COLUMN_SEPARATOR,
  FLAPPY_NETWORK_ARCHITECTURE_LINE_SEPARATOR,
  FLAPPY_NETWORK_BASELINE_HEIGHT_PX,
  FLAPPY_NETWORK_EMPTY_HIDDEN_LAYER_LABEL,
  FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX,
  FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX,
  FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR,
  FLAPPY_NETWORK_INFERRED_HIDDEN_LAYER_PREFIX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT,
  FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX,
  FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
  FLAPPY_NETWORK_LEGEND_RIGHT_SIDE_THRESHOLD_RATIO,
  FLAPPY_NETWORK_MAX_HEIGHT_PX,
  FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX,
  FLAPPY_NETWORK_MAX_NODE_WIDTH_PX,
  FLAPPY_NETWORK_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX,
  FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX,
  FLAPPY_NETWORK_MIN_DRAWABLE_SIZE_PX,
  FLAPPY_NETWORK_MIN_NODE_FIT_HEIGHT_PX,
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
import type { NetworkVisualizationColorScales } from '../visualization/visualization.types';
import { drawInputGroupLabelBands } from './network-view.draw.service';
import {
  centerPositionedNodesInDrawableArea,
  positionNetworkNodes,
} from './network-view.layout.utils';
import { resolveNetworkVisualizationLayers } from './network-view.topology.utils';

/**
 * Network-view orchestration for the browser-side architecture panel.
 *
 * This subsystem sits between raw network data and the lower-level visualization
 * drawing helpers. It resolves topology summaries, chooses panel size, lays out
 * nodes inside the drawable area, and coordinates overlays such as legends and
 * input-group bands.
 */

type NetworkTopologySummary = {
  networkLayers: ReturnType<typeof resolveNetworkVisualizationLayers>;
  layerCount: number;
  maximumLayerNodeCount: number;
};

type NetworkGraphPaddingContext = {
  graphLeftPaddingPx: number;
  graphTopPaddingPx: number;
  graphRightPaddingPx: number;
  graphBottomPaddingPx: number;
};

type NetworkVisualizationScene = NetworkGraphPaddingContext & {
  canvasWidthPx: number;
  canvasHeightPx: number;
  architectureLabel: string;
  colorScales: NetworkVisualizationColorScales;
  hideNetworkOverlays: boolean;
  adjustedGraphLeftPaddingPx: number;
  adjustedGraphRightPaddingPx: number;
};

type NetworkDrawableArea = {
  drawableWidthPx: number;
  drawableHeightPx: number;
};

type PositionedNetworkGraphScene = {
  centeredPositionedNodes: PositionedNetworkNode[];
  positionByNodeIndex: Map<number, PositionedNetworkNode>;
  runtimeConnections: VisualNetworkConnectionLike[];
  nodeDimensions: NetworkNodeDimensions;
};

/**
 * Draws a complete, layer-based visualization of the active network.
 *
 * Conceptually, this is the main fold from network object to finished panel:
 * resolve scene state, compute layout, paint the graph, then paint overlays.
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
  // Step 1: Resolve canvas, label, overlay, and padding state for this frame.
  const networkVisualizationScene = resolveNetworkVisualizationScene(
    context,
    network,
    inputSize,
    outputSize,
  );

  // Step 2: Paint the base canvas background before graph layers are drawn.
  paintNetworkVisualizationCanvasBase(context, networkVisualizationScene);

  // Step 3: Resolve node layout, node dimensions, and connection lookup state.
  const positionedNetworkGraphScene = resolvePositionedNetworkGraphScene(
    networkVisualizationScene,
    network,
    inputSize,
    outputSize,
  );

  // Step 4: Draw the graph layers and optional helper overlays.
  drawPositionedNetworkGraph(
    context,
    networkVisualizationScene,
    positionedNetworkGraphScene,
  );

  // Step 5: Draw the legend after the graph so the overlay stays visually on top.
  drawNetworkColorLegend(
    context,
    networkVisualizationScene.architectureLabel,
    networkVisualizationScene.colorScales,
  );
}

/**
 * Resolves responsive visualization canvas height from network shape.
 *
 * Dense or deeper networks need more vertical room to stay readable, so panel
 * height is driven by topology rather than fixed to a single constant.
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
  // Step 1: Summarize layer count and density from the resolved topology.
  const networkTopologySummary = resolveNetworkTopologySummary(
    network,
    inputSize,
    outputSize,
  );

  // Step 2: Compute the minimum readable height required by the topology.
  const topologyDrivenHeightPx = resolveTopologyDrivenHeightPx(
    networkTopologySummary,
  );

  // Step 3: Blend topology minimums with density and complexity adjustments.
  const recommendedHeightPx = resolveRecommendedNetworkHeightPx(
    networkTopologySummary,
    topologyDrivenHeightPx,
  );

  // Step 4: Clamp the result into the configured host panel range.
  return clampRecommendedNetworkHeightPx(recommendedHeightPx);
}

/**
 * Resolves compact architecture label text for headers and HUD rows.
 *
 * The label compresses the active network into a short human-readable summary:
 * input size, hidden-layer structure, output size, and graph size metadata.
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
  // Step 1: Fall back to a placeholder hidden-layer label when no network is active.
  if (!network) {
    return formatArchitectureLabel(
      inputSize,
      FLAPPY_NETWORK_EMPTY_HIDDEN_LAYER_LABEL,
      outputSize,
      inputSize + outputSize,
      0,
    );
  }

  // Step 2: Describe the active network and resolve the hidden-layer fragment.
  const architectureDescriptor = network.describeArchitecture();
  const hiddenLayersLabel = resolveHiddenLayersLabel(
    architectureDescriptor.hiddenLayerSizes,
    architectureDescriptor.source,
  );

  // Step 3: Compose the two-line architecture label from the descriptor values.
  return formatArchitectureLabel(
    inputSize,
    hiddenLayersLabel,
    outputSize,
    architectureDescriptor.totalNodes,
    architectureDescriptor.totalConnections,
  );
}

/**
 * Resolves all non-topology canvas state needed to draw the network view.
 *
 * This separates frame-scene concerns such as canvas size, overlays, and color
 * scales from the later graph-topology layout step.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Scene context for the current frame.
 */
function resolveNetworkVisualizationScene(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): NetworkVisualizationScene {
  // Step 1: Resolve the shared canvas dimensions and overlay label state.
  const canvasWidthPx = context.canvas.width;
  const canvasHeightPx = context.canvas.height;
  const architectureLabel = resolveNetworkArchitectureLabel(
    network,
    inputSize,
    outputSize,
  );
  const colorScales = resolveNetworkVisualizationColorScales(network);
  const graphPaddingContext = resolveBaseGraphPaddingContext();
  const hideNetworkOverlays = shouldHideNetworkOverlays(context, canvasWidthPx);

  // Step 2: Adjust graph-side padding when the legend is visible.
  const adjustedGraphPaddingContext = resolveAdjustedGraphPaddingContext(
    context,
    network,
    canvasWidthPx,
    hideNetworkOverlays,
    graphPaddingContext,
  );

  // Step 3: Fold the resolved scene state for downstream drawing helpers.
  return {
    canvasWidthPx,
    canvasHeightPx,
    architectureLabel,
    colorScales,
    hideNetworkOverlays,
    graphLeftPaddingPx: graphPaddingContext.graphLeftPaddingPx,
    graphTopPaddingPx: graphPaddingContext.graphTopPaddingPx,
    graphRightPaddingPx: graphPaddingContext.graphRightPaddingPx,
    graphBottomPaddingPx: graphPaddingContext.graphBottomPaddingPx,
    adjustedGraphLeftPaddingPx: adjustedGraphPaddingContext.graphLeftPaddingPx,
    adjustedGraphRightPaddingPx:
      adjustedGraphPaddingContext.graphRightPaddingPx,
  };
}

/**
 * Paints the static background fill for the network visualization canvas.
 *
 * @param context - Canvas 2D drawing context.
 * @param networkVisualizationScene - Frame scene context.
 * @returns Nothing.
 */
function paintNetworkVisualizationCanvasBase(
  context: CanvasRenderingContext2D,
  networkVisualizationScene: NetworkVisualizationScene,
): void {
  context.save();
  context.fillStyle = FLAPPY_UI_NETWORK_CANVAS_BACKGROUND;
  context.fillRect(
    0,
    0,
    networkVisualizationScene.canvasWidthPx,
    networkVisualizationScene.canvasHeightPx,
  );
  context.restore();
}

/**
 * Resolves positioned nodes, connection lookup state, and shared node dimensions.
 *
 * @param networkVisualizationScene - Frame scene context.
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Positioned graph scene.
 */
function resolvePositionedNetworkGraphScene(
  networkVisualizationScene: NetworkVisualizationScene,
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): PositionedNetworkGraphScene {
  // Step 1: Resolve topology and drawable bounds for the current graph region.
  const networkTopologySummary = resolveNetworkTopologySummary(
    network,
    inputSize,
    outputSize,
  );
  const networkDrawableArea = resolveNetworkDrawableArea(
    networkVisualizationScene,
  );
  const nodeDimensions = resolveNetworkNodeDimensionsFromTopologySummary(
    networkTopologySummary,
    networkDrawableArea.drawableWidthPx,
    networkDrawableArea.drawableHeightPx,
  );

  // Step 2: Position nodes inside the padded drawable region and center them.
  const positionedNodes = positionNetworkNodes(
    networkTopologySummary.networkLayers,
    networkVisualizationScene.adjustedGraphLeftPaddingPx,
    networkVisualizationScene.graphTopPaddingPx,
    networkDrawableArea.drawableWidthPx,
    networkDrawableArea.drawableHeightPx,
    FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX,
    nodeDimensions,
  );
  const centeredPositionedNodes = centerPositionedNodesInDrawableArea(
    positionedNodes,
    networkVisualizationScene.adjustedGraphLeftPaddingPx,
    networkVisualizationScene.graphTopPaddingPx,
    networkDrawableArea.drawableWidthPx,
    networkDrawableArea.drawableHeightPx,
    FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX,
    nodeDimensions,
  );

  // Step 3: Build the connection lookup state used by the drawing layers.
  return {
    centeredPositionedNodes,
    positionByNodeIndex: createPositionByNodeIndex(centeredPositionedNodes),
    runtimeConnections: resolveRuntimeConnections(network),
    nodeDimensions,
  };
}

/**
 * Draws the positioned graph layers and optional guide overlays.
 *
 * @param context - Canvas 2D drawing context.
 * @param networkVisualizationScene - Frame scene context.
 * @param positionedNetworkGraphScene - Positioned graph scene.
 * @returns Nothing.
 */
function drawPositionedNetworkGraph(
  context: CanvasRenderingContext2D,
  networkVisualizationScene: NetworkVisualizationScene,
  positionedNetworkGraphScene: PositionedNetworkGraphScene,
): void {
  // Step 1: Draw weighted connections beneath the node rectangles.
  drawWeightedConnectionsLayer(
    context,
    positionedNetworkGraphScene.runtimeConnections,
    positionedNetworkGraphScene.positionByNodeIndex,
    networkVisualizationScene.colorScales.connectionScale,
  );

  // Step 2: Draw input-group label bands only when overlays are visible.
  if (!networkVisualizationScene.hideNetworkOverlays) {
    drawInputGroupLabelBands(
      context,
      positionedNetworkGraphScene.centeredPositionedNodes,
      positionedNetworkGraphScene.nodeDimensions,
    );
  }

  // Step 3: Draw node rectangles and bias labels over the connection layer.
  drawBiasNodesLayer(
    context,
    positionedNetworkGraphScene.centeredPositionedNodes,
    positionedNetworkGraphScene.nodeDimensions,
    networkVisualizationScene.colorScales.biasScale,
  );
}

/**
 * Resolves the base graph padding before legend-aware adjustments are applied.
 *
 * @returns Base graph padding context.
 */
function resolveBaseGraphPaddingContext(): NetworkGraphPaddingContext {
  // Step 1: Reserve the input-group label band to the left of the graph body.
  const groupLabelBandReserveWidthPx =
    FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX +
    FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX +
    FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX;

  return {
    graphLeftPaddingPx:
      FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX + groupLabelBandReserveWidthPx,
    graphTopPaddingPx: FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX,
    graphRightPaddingPx: FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX,
    graphBottomPaddingPx: FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX,
  };
}

/**
 * Determines whether responsive rules hide auxiliary network overlays.
 *
 * @param context - Canvas 2D drawing context.
 * @param fallbackViewportWidthPx - Fallback viewport width.
 * @returns True when overlays should be hidden.
 */
function shouldHideNetworkOverlays(
  context: CanvasRenderingContext2D,
  fallbackViewportWidthPx: number,
): boolean {
  // Step 1: Prefer the browser viewport width and fall back to canvas width in tests.
  const viewportWidthPx =
    context.canvas.ownerDocument?.defaultView?.innerWidth ??
    fallbackViewportWidthPx;
  return viewportWidthPx < FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX;
}

/**
 * Adjusts graph-side padding to keep the floating legend from overlapping nodes.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @param canvasWidthPx - Canvas width.
 * @param hideNetworkOverlays - Whether overlays are hidden.
 * @param graphPaddingContext - Base graph padding context.
 * @returns Adjusted graph padding context.
 */
function resolveAdjustedGraphPaddingContext(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
  canvasWidthPx: number,
  hideNetworkOverlays: boolean,
  graphPaddingContext: NetworkGraphPaddingContext,
): Pick<
  NetworkGraphPaddingContext,
  'graphLeftPaddingPx' | 'graphRightPaddingPx'
> {
  // Step 1: Keep the base paddings unchanged when overlays are hidden.
  if (hideNetworkOverlays) {
    return {
      graphLeftPaddingPx: graphPaddingContext.graphLeftPaddingPx,
      graphRightPaddingPx: graphPaddingContext.graphRightPaddingPx,
    };
  }

  // Step 2: Measure the legend and determine which half of the canvas it occupies.
  const legendLayout = resolveDefaultNetworkLegendLayout(context, network);
  const legendMidpointPx =
    legendLayout.legendLeftPx +
    legendLayout.legendWidthPx *
      FLAPPY_NETWORK_LEGEND_RIGHT_SIDE_THRESHOLD_RATIO;
  if (
    legendMidpointPx >=
    canvasWidthPx * FLAPPY_NETWORK_LEGEND_RIGHT_SIDE_THRESHOLD_RATIO
  ) {
    return {
      graphLeftPaddingPx: graphPaddingContext.graphLeftPaddingPx,
      graphRightPaddingPx: Math.max(
        graphPaddingContext.graphRightPaddingPx,
        canvasWidthPx -
          legendLayout.legendLeftPx +
          FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
      ),
    };
  }

  // Step 3: Otherwise, shift the graph body right so it clears a left-side legend.
  return {
    graphLeftPaddingPx: Math.max(
      graphPaddingContext.graphLeftPaddingPx,
      legendLayout.legendLeftPx +
        legendLayout.legendWidthPx +
        FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX,
    ),
    graphRightPaddingPx: graphPaddingContext.graphRightPaddingPx,
  };
}

/**
 * Resolves the drawable graph area after scene padding is applied.
 *
 * @param networkVisualizationScene - Frame scene context.
 * @returns Drawable area dimensions.
 */
function resolveNetworkDrawableArea(
  networkVisualizationScene: NetworkVisualizationScene,
): NetworkDrawableArea {
  // Step 1: Subtract side paddings while enforcing a positive drawable width.
  const drawableWidthPx = Math.max(
    FLAPPY_NETWORK_MIN_DRAWABLE_SIZE_PX,
    networkVisualizationScene.canvasWidthPx -
      networkVisualizationScene.adjustedGraphLeftPaddingPx -
      networkVisualizationScene.adjustedGraphRightPaddingPx,
  );

  // Step 2: Subtract vertical paddings while enforcing a positive drawable height.
  const drawableHeightPx = Math.max(
    FLAPPY_NETWORK_MIN_DRAWABLE_SIZE_PX,
    networkVisualizationScene.canvasHeightPx -
      networkVisualizationScene.graphTopPaddingPx -
      networkVisualizationScene.graphBottomPaddingPx,
  );

  return {
    drawableWidthPx,
    drawableHeightPx,
  };
}

/**
 * Resolves a reusable topology summary for layout and sizing helpers.
 *
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Topology summary.
 */
function resolveNetworkTopologySummary(
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): NetworkTopologySummary {
  // Step 1: Resolve the layer groups once so downstream helpers share the same topology view.
  const networkLayers = resolveNetworkVisualizationLayers(
    network,
    inputSize,
    outputSize,
  );

  // Step 2: Summarize the topology density values used by sizing logic.
  return {
    networkLayers,
    layerCount: Math.max(1, networkLayers.length),
    maximumLayerNodeCount: Math.max(
      1,
      ...networkLayers.map((layerNodes) => layerNodes.length),
    ),
  };
}

/**
 * Resolves node rectangle dimensions from topology density and drawable bounds.
 *
 * @param networkTopologySummary - Topology summary.
 * @param drawableWidthPx - Drawable graph width.
 * @param drawableHeightPx - Drawable graph height.
 * @returns Node dimensions.
 */
function resolveNetworkNodeDimensionsFromTopologySummary(
  networkTopologySummary: NetworkTopologySummary,
  drawableWidthPx: number,
  drawableHeightPx: number,
): NetworkNodeDimensions {
  // Step 1: Resolve fit constraints from the available node stack height.
  const availableNodeStackHeightPx = Math.max(
    FLAPPY_NETWORK_MIN_DRAWABLE_SIZE_PX,
    drawableHeightPx - FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX * 2,
  );
  const strictMaximumNodeHeightByFitPx = Math.max(
    FLAPPY_NETWORK_MIN_NODE_FIT_HEIGHT_PX,
    availableNodeStackHeightPx / networkTopologySummary.maximumLayerNodeCount,
  );
  const minimumNodeHeightPx =
    FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX +
    FLAPPY_NETWORK_MIN_NODE_HEIGHT_LABEL_EXTRA_PX;
  const effectiveMinimumNodeHeightPx = Math.min(
    minimumNodeHeightPx,
    strictMaximumNodeHeightByFitPx,
  );

  // Step 2: Resolve the final node height from density, width, and fit constraints.
  const nodeHeightPx = Math.max(
    effectiveMinimumNodeHeightPx,
    Math.min(
      FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX,
      availableNodeStackHeightPx /
        Math.max(
          FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_MIN_DENOMINATOR,
          networkTopologySummary.maximumLayerNodeCount *
            FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_DIVISOR,
        ),
      drawableWidthPx /
        Math.max(
          FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_MIN_DENOMINATOR,
          networkTopologySummary.layerCount *
            FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_DIVISOR,
        ),
      strictMaximumNodeHeightByFitPx,
    ),
  );

  // Step 3: Resolve node width from the final height and layer-width cap.
  return {
    widthPx: Math.max(
      FLAPPY_NETWORK_MIN_NODE_WIDTH_PX,
      Math.min(
        FLAPPY_NETWORK_MAX_NODE_WIDTH_PX,
        nodeHeightPx * FLAPPY_NETWORK_NODE_WIDTH_TO_HEIGHT_RATIO,
        drawableWidthPx /
          Math.max(
            FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_MIN_DENOMINATOR,
            networkTopologySummary.layerCount *
              FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_DIVISOR,
          ),
      ),
    ),
    heightPx: nodeHeightPx,
  };
}

/**
 * Resolves the topology-driven minimum readable height.
 *
 * @param networkTopologySummary - Topology summary.
 * @returns Minimum readable height in pixels.
 */
function resolveTopologyDrivenHeightPx(
  networkTopologySummary: NetworkTopologySummary,
): number {
  // Step 1: Resolve the minimum readable node side from label and padding requirements.
  const minimumNodeSidePx =
    FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX +
    FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX * 2;

  // Step 2: Stack the densest layer using the minimum node size and inter-node gaps.
  const minimumReadableStackHeightPx =
    networkTopologySummary.maximumLayerNodeCount <= 1
      ? minimumNodeSidePx
      : networkTopologySummary.maximumLayerNodeCount * minimumNodeSidePx +
        (networkTopologySummary.maximumLayerNodeCount - 1) *
          FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX;

  // Step 3: Add graph padding so the stack fits within the full visualization panel.
  return (
    FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX +
    FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX +
    FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX +
    minimumReadableStackHeightPx
  );
}

/**
 * Resolves the recommended panel height from topology and density adjustments.
 *
 * @param networkTopologySummary - Topology summary.
 * @param topologyDrivenHeightPx - Minimum readable topology height.
 * @returns Recommended panel height.
 */
function resolveRecommendedNetworkHeightPx(
  networkTopologySummary: NetworkTopologySummary,
  topologyDrivenHeightPx: number,
): number {
  // Step 1: Resolve density and layer-complexity adjustments beyond the baseline topology.
  const nodeDensityHeightPx =
    Math.max(
      0,
      networkTopologySummary.maximumLayerNodeCount -
        FLAPPY_NETWORK_NODE_DENSITY_BASELINE_COUNT,
    ) * FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX;
  const layerComplexityHeightPx =
    Math.max(
      0,
      networkTopologySummary.layerCount -
        FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT,
    ) * FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX;

  // Step 2: Choose the larger of the topology minimum and baseline complexity model.
  return Math.max(
    Math.ceil(
      topologyDrivenHeightPx * FLAPPY_NETWORK_TOPOLOGY_HEIGHT_MULTIPLIER,
    ),
    FLAPPY_NETWORK_BASELINE_HEIGHT_PX +
      nodeDensityHeightPx +
      layerComplexityHeightPx,
  );
}

/**
 * Clamps a recommended network height into the configured panel range.
 *
 * @param recommendedHeightPx - Recommended panel height.
 * @returns Clamped panel height.
 */
function clampRecommendedNetworkHeightPx(recommendedHeightPx: number): number {
  // Step 1: Clamp the height into the configured min/max bounds and round down for canvas sizing.
  return Math.floor(
    clamp(
      recommendedHeightPx,
      FLAPPY_NETWORK_MIN_HEIGHT_PX,
      FLAPPY_NETWORK_MAX_HEIGHT_PX,
    ),
  );
}

/**
 * Builds a node-index lookup map for resolved positioned nodes.
 *
 * @param centeredPositionedNodes - Positioned nodes after centering.
 * @returns Map keyed by node index.
 */
function createPositionByNodeIndex(
  centeredPositionedNodes: PositionedNetworkNode[],
): Map<number, PositionedNetworkNode> {
  // Step 1: Materialize a lookup map so connection rendering can resolve endpoints quickly.
  return new Map<number, PositionedNetworkNode>(
    centeredPositionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );
}

/**
 * Resolves the runtime connection array from the active network.
 *
 * @param network - Network to visualize.
 * @returns Runtime connection list.
 */
function resolveRuntimeConnections(
  network: Network | undefined,
): VisualNetworkConnectionLike[] {
  // Step 1: Fall back to an empty list when no network is active.
  return ((network?.connections ?? []) as VisualNetworkConnectionLike[]) ?? [];
}

/**
 * Formats the two-line architecture label used by the header and legend.
 *
 * @param architectureInputSize - Input layer size.
 * @param hiddenLayersLabel - Hidden-layer description.
 * @param architectureOutputSize - Output layer size.
 * @param totalNodeCount - Total node count.
 * @param totalConnectionCount - Total connection count.
 * @returns Formatted architecture label.
 */
function formatArchitectureLabel(
  architectureInputSize: number,
  hiddenLayersLabel: string,
  architectureOutputSize: number,
  totalNodeCount: number,
  totalConnectionCount: number,
): string {
  // Step 1: Build the compact architecture row from input, hidden, and output sizes.
  const architectureColumnsLabel = [
    architectureInputSize,
    hiddenLayersLabel,
    architectureOutputSize,
  ].join(FLAPPY_NETWORK_ARCHITECTURE_COLUMN_SEPARATOR);

  // Step 2: Build the totals row and join both lines into the final label block.
  const architectureTotalsLabel = `(${totalNodeCount} nodes, ${totalConnectionCount} connections)`;
  return [architectureColumnsLabel, architectureTotalsLabel].join(
    FLAPPY_NETWORK_ARCHITECTURE_LINE_SEPARATOR,
  );
}

/**
 * Resolves the hidden-layer portion of the compact architecture label.
 *
 * @param hiddenLayerSizes - Hidden-layer sizes.
 * @param architectureSource - Architecture source metadata.
 * @returns Hidden-layer label.
 */
function resolveHiddenLayersLabel(
  hiddenLayerSizes: number[],
  architectureSource: 'layer-metadata' | 'graph-topology' | 'inferred',
): string {
  // Step 1: Use the placeholder label when the network has no hidden layers.
  if (hiddenLayerSizes.length === 0) {
    return FLAPPY_NETWORK_EMPTY_HIDDEN_LAYER_LABEL;
  }

  // Step 2: Prefix inferred hidden layers so the label communicates uncertainty.
  if (architectureSource === 'inferred') {
    return hiddenLayerSizes
      .map(
        (hiddenLayerSize) =>
          `${FLAPPY_NETWORK_INFERRED_HIDDEN_LAYER_PREFIX}${hiddenLayerSize}`,
      )
      .join(FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR);
  }

  // Step 3: Join explicit hidden-layer sizes without inference prefixes.
  return hiddenLayerSizes.join(FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR);
}
