/**
 * Network-view orchestration for the browser-side architecture panel.
 *
 * This module is the browser-facing fold from a live evolved controller to a
 * readable inspection panel. It does not own the low-level drawing primitives,
 * and it does not invent topology semantics from scratch. Instead it composes
 * both into one higher-level question: how should this network be laid out so a
 * human can actually learn from it?
 *
 * The boundary exists because "draw the network" hides several distinct jobs:
 * summarize topology, size the panel, place nodes, choose overlay policy, and
 * then delegate the final painting work. Keeping those steps together here makes
 * the generated README read like an inspection chapter instead of a pile of
 * canvas helpers.
 */
import Network from '../../../../src/architecture/network';
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
  FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TOP_RESERVE_PX,
  FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR,
  FLAPPY_NETWORK_INFERRED_HIDDEN_LAYER_PREFIX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX,
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
  NetworkInputDescriptionScene,
  NetworkInputGroupLabelBandScene,
  NetworkHiddenColumnLabelScene,
  NetworkNodeDimensionsLike as NetworkNodeDimensions,
  NetworkVisualizationHoverState,
  NetworkVisualizationPositionedScene,
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
import {
  alignInputNodesToDescriptionScenes,
  drawHiddenColumnLabelScenes,
  drawInputNodeDescriptions,
  drawInputGroupLabelBands,
  resolveHiddenColumnLabelScenes,
  resolveInputDescriptionScenes,
  resolveInputGroupLabelBandScenes,
} from './network-view.draw.service';
import {
  centerPositionedNodesInDrawableArea,
  positionNetworkNodes,
} from './network-view.layout.utils';
import { resolveInputDescriptionColumnWidthPx } from './network-view.labels.utils';
import {
  resolveNetworkVisualizationTopologyPlan,
  type NetworkHiddenColumnAnnotation,
} from './network-view.topology.utils';

type NetworkTopologySummary = {
  networkLayers: ReturnType<
    typeof resolveNetworkVisualizationTopologyPlan
  >['networkLayers'];
  hiddenColumnAnnotations: NetworkHiddenColumnAnnotation[];
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
  inputDescriptionScenes: NetworkInputDescriptionScene[];
  inputGroupLabelBandScenes: NetworkInputGroupLabelBandScene[];
  hiddenColumnLabelScenes: NetworkHiddenColumnLabelScene[];
  positionByNodeIndex: Map<number, PositionedNetworkNode>;
  runtimeConnections: VisualNetworkConnectionLike[];
  nodeDimensions: NetworkNodeDimensions;
};

/**
 * Reusable resolved frame cache for hover-only network redraws.
 *
 * The host can keep this resolved frame between pointer-driven redraws so it
 * does not recompute topology, layout, legend inputs, or connection lookups
 * until the network payload or canvas size changes.
 */
export interface NetworkVisualizationResolvedFrame {
  canvasWidthPx: number;
  canvasHeightPx: number;
  architectureLabel: string;
  colorScales: NetworkVisualizationColorScales;
  hideNetworkOverlays: boolean;
  positionedScene: NetworkVisualizationPositionedScene;
  positionByNodeIndex: Map<number, PositionedNetworkNode>;
  runtimeConnections: VisualNetworkConnectionLike[];
}

/**
 * Draws a complete, layer-based visualization of the active network.
 *
 * Conceptually, this is the main fold from network object to finished panel:
 * resolve scene state, compute layout, paint the graph, then paint overlays.
 *
 * @example
 * ```ts
 * drawNetworkVisualization(networkContext, bestNetwork, 12, 2);
 * ```
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @param hoverState - Optional host-owned hover state for interactive emphasis.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 */
export function drawNetworkVisualization(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
  hoverState?: NetworkVisualizationHoverState,
): NetworkVisualizationPositionedScene {
  // Step 1: Resolve the reusable frame cache for the current network payload.
  const resolvedNetworkVisualizationFrame = resolveNetworkVisualizationFrame(
    context,
    network,
    inputSize,
    outputSize,
  );

  // Step 2: Paint the resolved frame using the current interactive hover state.
  drawResolvedNetworkVisualization(
    context,
    resolvedNetworkVisualizationFrame,
    hoverState,
  );

  // Step 3: Return the positioned node scene for host-owned hover hit testing.
  return resolvedNetworkVisualizationFrame.positionedScene;
}

/**
 * Resolves a reusable network-visualization frame from the active payload.
 *
 * This fold captures the expensive static work for the panel in one object so
 * hover-only redraws can repaint from cached layout and legend data.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @param inputSize - Input-layer size.
 * @param outputSize - Output-layer size.
 * @returns Reusable resolved frame for subsequent draw passes.
 */
export function resolveNetworkVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
  inputSize: number,
  outputSize: number,
): NetworkVisualizationResolvedFrame {
  // Step 1: Resolve canvas, label, overlay, and padding state for this frame.
  const networkVisualizationScene = resolveNetworkVisualizationScene(
    context,
    network,
    inputSize,
    outputSize,
  );

  // Step 2: Resolve node layout, node dimensions, and connection lookup state.
  const positionedNetworkGraphScene = resolvePositionedNetworkGraphScene(
    networkVisualizationScene,
    network,
    inputSize,
    outputSize,
  );

  // Step 3: Fold the reusable frame fields that hover-only redraws need later.
  return {
    canvasWidthPx: networkVisualizationScene.canvasWidthPx,
    canvasHeightPx: networkVisualizationScene.canvasHeightPx,
    architectureLabel: networkVisualizationScene.architectureLabel,
    colorScales: networkVisualizationScene.colorScales,
    hideNetworkOverlays: networkVisualizationScene.hideNetworkOverlays,
    positionedScene: {
      positionedNodes: positionedNetworkGraphScene.centeredPositionedNodes,
      nodeDimensions: positionedNetworkGraphScene.nodeDimensions,
      inputDescriptionScenes:
        positionedNetworkGraphScene.inputDescriptionScenes,
      inputGroupLabelBandScenes:
        positionedNetworkGraphScene.inputGroupLabelBandScenes,
      hiddenColumnLabelScenes:
        positionedNetworkGraphScene.hiddenColumnLabelScenes,
    },
    positionByNodeIndex: positionedNetworkGraphScene.positionByNodeIndex,
    runtimeConnections: positionedNetworkGraphScene.runtimeConnections,
  };
}

/**
 * Draws a previously resolved network-visualization frame.
 *
 * The host uses this path for hover-only repaint work because it can reuse the
 * cached static frame and only vary interactive emphasis.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedNetworkVisualizationFrame - Reusable frame cache.
 * @param hoverState - Optional host-owned hover state for interactive emphasis.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 */
export function drawResolvedNetworkVisualization(
  context: CanvasRenderingContext2D,
  resolvedNetworkVisualizationFrame: NetworkVisualizationResolvedFrame,
  hoverState?: NetworkVisualizationHoverState,
): NetworkVisualizationPositionedScene {
  // Step 1: Paint the cached base canvas background before graph layers are drawn.
  paintNetworkVisualizationCanvasBase(
    context,
    resolvedNetworkVisualizationFrame,
  );

  // Step 2: Draw the graph layers and optional helper overlays from the cached frame.
  drawPositionedNetworkGraph(
    context,
    resolvedNetworkVisualizationFrame,
    hoverState,
  );

  // Step 3: Draw the legend after the graph so the overlay stays visually on top.
  drawNetworkColorLegend(
    context,
    resolvedNetworkVisualizationFrame.architectureLabel,
    resolvedNetworkVisualizationFrame.colorScales,
  );

  return resolvedNetworkVisualizationFrame.positionedScene;
}

/**
 * Resolves responsive visualization canvas height from network shape.
 *
 * Dense or deeper networks need more vertical room to stay readable, so panel
 * height is driven by topology rather than fixed to a single constant.
 *
 * @example
 * ```ts
 * const recommendedHeightPx = resolveNetworkVisualizationHeightPx(network, 12, 2);
 * ```
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
 * When a runtime network is present, explicit input/output role metadata is
 * treated as the authoritative boundary size instead of the caller's fallback
 * hints so the browser panel reflects the network's current public contract.
 * The label can also append a compact scheduling line when the runtime exposes
 * a non-standard activation contract such as recurrent execution or cycle
 * fallback behavior.
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
  const temporalStructure = network.describeTemporalStructure();
  const hiddenLayersLabel =
    temporalStructure.recurrentModules.length > 0
      ? resolveTemporalHiddenLayersLabel(network, temporalStructure)
      : resolveHiddenLayersLabel(
          architectureDescriptor.hiddenLayerSizes,
          architectureDescriptor.source,
        );
  const architectureInputSize =
    network.inputNodeIds.length > 0 ? network.inputNodeIds.length : inputSize;
  const architectureOutputSize =
    network.outputNodeIds.length > 0
      ? network.outputNodeIds.length
      : outputSize;
  const schedulingStatusLine = resolveSchedulingStatusLine(network);

  // Step 3: Compose the two-line architecture label from the descriptor values.
  return formatArchitectureLabel(
    architectureInputSize,
    hiddenLayersLabel,
    architectureOutputSize,
    architectureDescriptor.totalNodes,
    architectureDescriptor.totalConnections,
    schedulingStatusLine,
  );
}

/**
 * Resolve a compact scheduling status line for the architecture label.
 *
 * The browser panel should stay quiet for the standard feed-forward contract,
 * but it should surface a small extra line when a network is recurrent or when
 * acyclic scheduling fell back because of a detected cycle.
 *
 * @param network - Network being visualized.
 * @returns Scheduling status line or null for the normal feed-forward path.
 */
function resolveSchedulingStatusLine(network: Network): string | null {
  const schedulingDiagnostics = network.getActivationSchedulingDiagnostics();

  if (schedulingDiagnostics.issue === 'cycle-detected') {
    return 'warning: acyclic via cycle fallback';
  }

  if (schedulingDiagnostics.issue === 'schedule-missing') {
    return null;
  }

  if (schedulingDiagnostics.requestedMode !== 'recurrent') {
    return null;
  }

  return `schedule: recurrent via ${resolveSchedulingExecutionLabel(
    schedulingDiagnostics.executionPath,
  )}`;
}

/**
 * Resolve a short human-readable execution label for browser architecture text.
 *
 * @param executionPath - Scheduling execution path reported by the runtime.
 * @returns Compact browser-facing label.
 */
function resolveSchedulingExecutionLabel(
  executionPath: ReturnType<
    Network['getActivationSchedulingDiagnostics']
  >['executionPath'],
): string {
  if (executionPath === 'compiled-schedule') {
    return 'compiled schedule';
  }

  if (executionPath === 'cycle-fallback-order') {
    return 'cycle fallback';
  }

  return 'raw node order';
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
  const hideNetworkOverlays = shouldHideNetworkOverlays(context, canvasWidthPx);
  const graphPaddingContext = resolveBaseGraphPaddingContext(
    hideNetworkOverlays,
    inputSize,
    network,
  );

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
  networkVisualizationScene: Pick<
    NetworkVisualizationResolvedFrame,
    'canvasWidthPx' | 'canvasHeightPx'
  >,
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
  const initialInputDescriptionScenes =
    networkVisualizationScene.hideNetworkOverlays
      ? []
      : resolveInputDescriptionScenes(centeredPositionedNodes, nodeDimensions);
  const overlayAlignedPositionedNodes =
    networkVisualizationScene.hideNetworkOverlays
      ? centeredPositionedNodes
      : alignInputNodesToDescriptionScenes(
          centeredPositionedNodes,
          initialInputDescriptionScenes,
        );
  const inputGroupLabelBandScenes =
    networkVisualizationScene.hideNetworkOverlays
      ? []
      : resolveInputGroupLabelBandScenes(
          overlayAlignedPositionedNodes,
          nodeDimensions,
        );
  const inputDescriptionScenes =
    networkVisualizationScene.hideNetworkOverlays
      ? []
      : resolveInputDescriptionScenes(
          overlayAlignedPositionedNodes,
          nodeDimensions,
        );
  const hiddenColumnLabelScenes =
    networkVisualizationScene.hideNetworkOverlays
      ? []
      : resolveHiddenColumnLabelScenes(
          overlayAlignedPositionedNodes,
          nodeDimensions,
          networkTopologySummary.hiddenColumnAnnotations,
        );

  // Step 3: Build the connection lookup state used by the drawing layers.
  return {
    centeredPositionedNodes: overlayAlignedPositionedNodes,
    inputDescriptionScenes,
    inputGroupLabelBandScenes,
    hiddenColumnLabelScenes,
    positionByNodeIndex: createPositionByNodeIndex(overlayAlignedPositionedNodes),
    runtimeConnections: resolveRuntimeConnections(network),
    nodeDimensions,
  };
}

/**
 * Draws the positioned graph layers and optional guide overlays.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedNetworkVisualizationFrame - Resolved network visualization frame containing positioned scene, connections, and color scales.
 * @param hoverState - Optional host-owned hover state for interactive emphasis.
 * @returns Nothing.
 */
function drawPositionedNetworkGraph(
  context: CanvasRenderingContext2D,
  resolvedNetworkVisualizationFrame: NetworkVisualizationResolvedFrame,
  hoverState?: NetworkVisualizationHoverState,
): void {
  // Step 1: Draw weighted connections beneath the node rectangles.
  drawWeightedConnectionsLayer(
    context,
    resolvedNetworkVisualizationFrame.runtimeConnections,
    resolvedNetworkVisualizationFrame.positionByNodeIndex,
    resolvedNetworkVisualizationFrame.colorScales.connectionScale,
    hoverState?.animatedHoveredNodes,
  );

  // Step 2: Draw input-group label bands only when overlays are visible.
  if (!resolvedNetworkVisualizationFrame.hideNetworkOverlays) {
    drawInputGroupLabelBands(
      context,
      resolvedNetworkVisualizationFrame.positionedScene
        .inputGroupLabelBandScenes,
      hoverState?.hoveredNodeIndices,
    );
    drawInputNodeDescriptions(
      context,
      resolvedNetworkVisualizationFrame.positionedScene.inputDescriptionScenes,
      hoverState?.hoveredNodeIndices,
    );
    drawHiddenColumnLabelScenes(
      context,
      resolvedNetworkVisualizationFrame.positionedScene.hiddenColumnLabelScenes ?? [],
      hoverState?.hoveredNodeIndices,
    );
  }

  // Step 3: Draw node rectangles and bias labels over the connection layer.
  drawBiasNodesLayer(
    context,
    resolvedNetworkVisualizationFrame.positionedScene.positionedNodes,
    resolvedNetworkVisualizationFrame.positionedScene.nodeDimensions,
    resolvedNetworkVisualizationFrame.colorScales.biasScale,
    hoverState?.animatedHoveredNodes,
  );
}

/**
 * Resolves the base graph padding before legend-aware adjustments are applied.
 *
 * @returns Base graph padding context.
 */
function resolveBaseGraphPaddingContext(
  hideNetworkOverlays: boolean,
  inputNodeCount: number,
  network: Network | undefined,
): NetworkGraphPaddingContext {
  // Step 1: Reserve the full input-overlay shelf only when overlays are visible.
  const descriptionColumnReserveWidthPx =
    hideNetworkOverlays
      ? 0
      : resolveInputDescriptionColumnWidthPx(inputNodeCount);
  const groupLabelBandReserveWidthPx =
    hideNetworkOverlays || descriptionColumnReserveWidthPx === 0
      ? 0
      : FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX +
        FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX +
        descriptionColumnReserveWidthPx +
        FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX +
        FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX;

  return {
    graphLeftPaddingPx:
      FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX + groupLabelBandReserveWidthPx,
    graphTopPaddingPx:
      FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX +
      resolveHiddenColumnLabelReserveHeightPx(network, hideNetworkOverlays),
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
  const topologyPlan = resolveNetworkVisualizationTopologyPlan(
    network,
    inputSize,
    outputSize,
  );
  const networkLayers = topologyPlan.networkLayers;

  // Step 2: Summarize the topology density values used by sizing logic.
  return {
    networkLayers,
    hiddenColumnAnnotations: topologyPlan.hiddenColumnAnnotations,
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
  schedulingStatusLine?: string | null,
): string {
  // Step 1: Build the compact architecture row from input, hidden, and output sizes.
  const architectureColumnsLabel = [
    architectureInputSize,
    hiddenLayersLabel,
    architectureOutputSize,
  ].join(FLAPPY_NETWORK_ARCHITECTURE_COLUMN_SEPARATOR);

  // Step 2: Build the totals row and join both lines into the final label block.
  const architectureTotalsLabel = `(${totalNodeCount} nodes, ${totalConnectionCount} connections)`;
  return [
    architectureColumnsLabel,
    schedulingStatusLine,
    architectureTotalsLabel,
  ]
    .filter(
      (architectureLabelLine): architectureLabelLine is string =>
        typeof architectureLabelLine === 'string' && architectureLabelLine.length > 0,
    )
    .join(
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

function resolveTemporalHiddenLayersLabel(
  network: Network,
  temporalStructure: ReturnType<Network['describeTemporalStructure']>,
): string {
  const recurrentKinds = [...new Set(
    temporalStructure.recurrentModules.map((recurrentModule) => recurrentModule.kind),
  )];
  const sortedRecurrentModules = [...temporalStructure.recurrentModules].toSorted(
    (leftModule, rightModule) =>
      resolveTemporalModuleOrderValue(leftModule) -
      resolveTemporalModuleOrderValue(rightModule),
  );

  if (recurrentKinds.length === 1 && recurrentKinds[0] === 'lstm') {
    return resolveUniformTemporalFamilyLabel(
      'LSTM',
      sortedRecurrentModules,
      ['outputBlock', 'memoryCell'],
      network,
    );
  }

  if (recurrentKinds.length === 1 && recurrentKinds[0] === 'gru') {
    return resolveUniformTemporalFamilyLabel(
      'GRU',
      sortedRecurrentModules,
      ['output', 'memoryCell'],
      network,
    );
  }

  if (recurrentKinds.length === 1 && recurrentKinds[0] === 'narx-memory') {
    return resolveNarxTemporalFamilyLabel(sortedRecurrentModules, network);
  }

  const extraHiddenCount = resolveTemporalExtraHiddenCount(
    sortedRecurrentModules,
    network,
  );
  return `TEMP[${recurrentKinds.join('+')}]${extraHiddenCount > 0 ? `+${extraHiddenCount}` : ''}`;
}

function resolveUniformTemporalFamilyLabel(
  familyLabel: string,
  recurrentModules: ReturnType<Network['describeTemporalStructure']>['recurrentModules'],
  preferredRoleNames: readonly string[],
  network: Network,
): string {
  const blockSizes = recurrentModules.map((recurrentModule) =>
    resolvePreferredRoleSize(recurrentModule, preferredRoleNames),
  );
  const extraHiddenCount = resolveTemporalExtraHiddenCount(
    recurrentModules,
    network,
  );
  return `${familyLabel}[${blockSizes.join(',')}]${extraHiddenCount > 0 ? `+${extraHiddenCount}` : ''}`;
}

function resolveNarxTemporalFamilyLabel(
  recurrentModules: ReturnType<Network['describeTemporalStructure']>['recurrentModules'],
  network: Network,
): string {
  const inputDelayModule = recurrentModules.find(
    (recurrentModule) => recurrentModule.moduleLabel === 'input',
  );
  const outputDelayModule = recurrentModules.find(
    (recurrentModule) => recurrentModule.moduleLabel === 'output',
  );
  const inputDelayCount = inputDelayModule
    ? Object.keys(inputDelayModule.nodeGeneIdsByRole).length
    : 0;
  const outputDelayCount = outputDelayModule
    ? Object.keys(outputDelayModule.nodeGeneIdsByRole).length
    : 0;
  const extraHiddenCount = resolveTemporalExtraHiddenCount(
    recurrentModules,
    network,
  );
  return `NARX[i${inputDelayCount},o${outputDelayCount}${extraHiddenCount > 0 ? `,+${extraHiddenCount}` : ''}]`;
}

function resolvePreferredRoleSize(
  recurrentModule: ReturnType<Network['describeTemporalStructure']>['recurrentModules'][number],
  preferredRoleNames: readonly string[],
): number {
  for (const preferredRoleName of preferredRoleNames) {
    const preferredRoleNodeGeneIds =
      recurrentModule.nodeGeneIdsByRole[preferredRoleName];
    if (Array.isArray(preferredRoleNodeGeneIds)) {
      return preferredRoleNodeGeneIds.length;
    }
  }

  return Math.max(
    0,
    ...Object.values(recurrentModule.nodeGeneIdsByRole).map(
      (roleNodeGeneIds) => roleNodeGeneIds.length,
    ),
  );
}

function resolveTemporalExtraHiddenCount(
  recurrentModules: ReturnType<Network['describeTemporalStructure']>['recurrentModules'],
  network: Network,
): number {
  const moduleOwnedGeneIds = new Set(
    recurrentModules.flatMap((recurrentModule) =>
      Object.values(recurrentModule.nodeGeneIdsByRole).flat(),
    ),
  );

  return network.nodes.filter(
    (runtimeNode) =>
      runtimeNode.type !== 'input' &&
      runtimeNode.type !== 'output' &&
      runtimeNode.type !== 'constant' &&
      !moduleOwnedGeneIds.has(runtimeNode.geneId),
  ).length;
}

function resolveTemporalModuleOrderValue(
  recurrentModule: ReturnType<Network['describeTemporalStructure']>['recurrentModules'][number],
): number {
  const orderedGeneIds = Object.values(recurrentModule.nodeGeneIdsByRole)
    .flat()
    .filter((geneId): geneId is number => Number.isFinite(geneId));
  return orderedGeneIds.length > 0
    ? Math.min(...orderedGeneIds)
    : Number.MAX_SAFE_INTEGER;
}

function resolveHiddenColumnLabelReserveHeightPx(
  network: Network | undefined,
  hideNetworkOverlays: boolean,
): number {
  if (hideNetworkOverlays || !network) {
    return 0;
  }

  return network.describeTemporalStructure().recurrentModules.length > 0
    ? FLAPPY_NETWORK_HIDDEN_COLUMN_LABEL_TOP_RESERVE_PX
    : 0;
}
