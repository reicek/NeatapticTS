/**
 * Rich network-view renderer for the ASCII Maze browser demo.
 *
 * This module wraps the shared `renderNetworkView` canvas renderer and layers
 * on maze-specific educational overlays: colored input group bands, per-node
 * chip labels, output direction labels, and a weight/bias legend.
 *
 * The result includes hit areas that the host can use for hover tooltip testing
 * without needing to re-render the canvas on every pointer event.
 *
 * ```ts
 * const result = drawMazeNetworkVisualization(canvas, network, graph);
 * // result.hitAreas — hover hit test geometry + tooltip content
 * // result.frame    — positioned nodes (reusable for custom redraws)
 * ```
 *
 * @see {@link https://en.wikipedia.org/wiki/Breadth-first_search BFS distance map (Wikipedia)}
 */

import type Network from '../../../../src/architecture/network';
import { renderNetworkView } from '../../../../src/visualization/visualization';
import type { VisualizationGraphV1 } from '../../../../src/neataptic';
import type {
  NetworkVisualizationResolvedFrame,
  PositionedNetworkNode,
  NetworkNodeDimensions,
} from '../../../../src/visualization/network-view/network-view.types';
import {
  MAZE_GROUP_COLORS,
  MAZE_INPUT_GROUP_DEFS,
  MAZE_LABEL_LEFT_PADDING_PX,
  MAZE_OUTPUT_LABELS,
  type MazeInputGroupDef,
} from './network-view.constants';
import {
  FLAPPY_NETWORK_ARCHITECTURE_COLUMN_SEPARATOR,
  FLAPPY_NETWORK_ARCHITECTURE_LINE_SEPARATOR,
  FLAPPY_NETWORK_EMPTY_HIDDEN_LAYER_LABEL,
  FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_CHIP_VERTICAL_GAP_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_LINE_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_VERTICAL_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
  FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
  FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
  FLAPPY_NETWORK_INPUT_GROUP_VERTICAL_GAP_PX,
  FLAPPY_NETWORK_INFERRED_HIDDEN_LAYER_PREFIX,
  FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX,
  FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX,
  FLAPPY_NETWORK_LEGEND_MARGIN_PX,
  FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX,
  FLAPPY_UI_NETWORK_CANVAS_BACKGROUND,
} from '../../../flappy_bird/constants/constants';
import {
  drawInputGroupLabelBands,
  drawInputNodeDescriptions,
} from '../../../flappy_bird/browser-entry/network-view/network-view.draw.service';
import { resolveInputDescriptionChipWidthPx } from '../../../flappy_bird/browser-entry/network-view/network-view.labels.utils';
import {
  drawBiasNodesLayer,
  drawNetworkColorLegend,
  drawWeightedConnectionsLayer,
} from '../../../flappy_bird/browser-entry/visualization/visualization.draw.service';
import { resolveNetworkVisualizationColorScales } from '../../../flappy_bird/browser-entry/visualization/visualization.colors.utils';
import type {
  NetworkInputDescriptionScene,
  NetworkInputGroupLabelBandScene,
} from '../../../flappy_bird/browser-entry/browser-entry.visualization.types';

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

/**
 * A canvas-space rectangular hit area with associated tooltip content.
 *
 * Returned in bulk by `drawMazeNetworkVisualization` so the host can test
 * pointer positions against them during mousemove without re-rendering.
 */
export interface MazeHitArea {
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  heading: string;
  bodyParagraphs: readonly string[];
  hoveredNodeIndices: readonly number[];
}

/**
 * Full return value from `drawMazeNetworkVisualization`.
 *
 * The frame gives access to positioned node geometry (useful for custom
 * overlay logic) and the hit areas are ready for pointer hit testing.
 */
export interface MazeNetworkRenderResult {
  /** Resolved canvas state including positioned nodes and connections. */
  frame: NetworkVisualizationResolvedFrame;
  /** Hover hit areas for input groups, per-node chips, and output labels. */
  hitAreas: MazeHitArea[];
}

// ---------------------------------------------------------------------------
// Private layout constants
// ---------------------------------------------------------------------------

/** Canvas height expressed as a fraction of the canvas width. */
const CANVAS_ASPECT_RATIO = 0.64;

/** Minimum canvas backing-store width. */
const MIN_CANVAS_WIDTH_PX = 320;

/** Minimum canvas backing-store height. */
const MIN_CANVAS_HEIGHT_PX = 240;

/** Flat fallback colors used by the shared renderer only for its layout pass. */
const MAZE_LAYOUT_PASS_COLOR_SCALES = {
  weightPositive: '#00ff88',
  weightNegative: '#ff3366',
  activationHot: '#ffcc00',
  activationCold: '#0088ff',
  bias: '#aa44ff',
};

/** Extra right-side reserve so the Flappy-style legend can sit beside the graph. */
const MAZE_RIGHT_LEGEND_RESERVE_PX =
  FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX + FLAPPY_NETWORK_LEGEND_MARGIN_PX * 2;

/** Minimum left reserve kept for readable maze input-label overlays. */
const MAZE_MIN_LEFT_LABEL_PANEL_WIDTH_PX = 132;

/** Minimum drawable graph width needed to keep hidden layers visually separate. */
const MAZE_MIN_DRAWABLE_GRAPH_WIDTH_PX = 420;

type MazeRuntimePositionedNode = {
  xPx: number;
  yPx: number;
  node: {
    index: number;
    type: 'input' | 'hidden' | 'output';
    bias: number;
  };
};

type MazeRuntimeConnection = {
  from: { index: number };
  to: { index: number };
  weight: number;
  enabled: boolean;
};

type ResolvedMazeInputGroupLayout = {
  groupDef: MazeInputGroupDef;
  groupColor: (typeof MAZE_GROUP_COLORS)[number];
  nodeIndices: number[];
  labelBandLeftPx: number;
  topPx: number;
  heightPx: number;
  descriptionScenes: NetworkInputDescriptionScene[];
};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/**
 * Draws a full educational network visualization for the ASCII Maze demo.
 *
 * Orchestration:
 * 1. Sync canvas backing-store dimensions to the panel.
 * 2. Render the base graph (nodes + connections + background) via the shared renderer.
 * 3. Draw the input label panel in the reserved left padding area.
 * 4. Draw output direction labels to the right of output nodes.
 * 5. Draw the connection weight + bias legend.
 *
 * @param canvas - Canvas element to render onto.
 * @param network - Runtime network used for architecture and dynamic color scales.
 * @param graph  - Exported visualization graph from `exportVisualizationGraph`.
 * @returns Resolved frame plus hover hit areas for the host tooltip system.
 */
export function drawMazeNetworkVisualization(
  canvas: HTMLCanvasElement,
  network: Network,
  graph: VisualizationGraphV1,
  hoveredNodeIndices: readonly number[] = [],
): MazeNetworkRenderResult {
  // Step 1: Sync canvas dimensions to the current panel width.
  syncCanvasToPanel(canvas);

  const panelPadding = resolveMazeNetworkPanelPadding(canvas.width);

  const dynamicColorScales = resolveNetworkVisualizationColorScales(network);
  const architectureLabel = resolveMazeArchitectureLabel(network, graph);

  // Step 2: Render base graph with generous left padding for the label panel.
  const frame = renderNetworkView(canvas, graph, {
    nodeDimensions: { widthPx: 28, heightPx: 10 },
    panelPaddingPx: panelPadding,
    colorScales: MAZE_LAYOUT_PASS_COLOR_SCALES,
  });

  const context = canvas.getContext('2d');
  if (!context) {
    return { frame, hitAreas: [] };
  }

  // Step 3: Sort input and output nodes top-to-bottom for consistent ordering.
  const inputNodes = frame.positionedNodes
    .filter((node) => node.type === 'input')
    .toSorted((nodeA, nodeB) => nodeA.centerYPx - nodeB.centerYPx);

  const outputNodes = frame.positionedNodes
    .filter((node) => node.type === 'output')
    .toSorted((nodeA, nodeB) => nodeA.centerYPx - nodeB.centerYPx);

  const runtimePositionedNodes = resolveRuntimePositionedNodes(
    frame.positionedNodes,
  );
  const runtimeConnections = resolveRuntimeConnections(graph);
  const positionByNodeIndex = new Map(
    runtimePositionedNodes.map((positionedNode) => [
      positionedNode.node.index,
      positionedNode,
    ]),
  );

  // Step 4: Repaint the graph using the Flappy visual language so the legend remains truthful.
  paintMazeNetworkBase(context, canvas);
  drawWeightedConnectionsLayer(
    context,
    runtimeConnections,
    positionByNodeIndex,
    dynamicColorScales.connectionScale,
  );
  drawBiasNodesLayer(
    context,
    runtimePositionedNodes,
    frame.nodeDimensions,
    dynamicColorScales.biasScale,
  );

  // Step 5: Draw Flappy-style semantic input overlays and collect hover hit areas.
  const { inputDescriptionScenes, inputGroupLabelBandScenes, hitAreas } =
    resolveAndDrawInputLabelPanel(context, inputNodes, frame.nodeDimensions);

  drawInputGroupLabelBands(
    context,
    inputGroupLabelBandScenes,
    hoveredNodeIndices,
  );
  drawInputNodeDescriptions(
    context,
    inputDescriptionScenes,
    hoveredNodeIndices,
  );

  // Step 6: Draw output direction labels.
  drawOutputNodeLabels(context, outputNodes, frame.nodeDimensions);

  // Step 7: Draw Flappy-style architecture summary and full legend table.
  drawNetworkColorLegend(context, architectureLabel, dynamicColorScales);

  return { frame, hitAreas };
}

// ---------------------------------------------------------------------------
// Canvas sync helper
// ---------------------------------------------------------------------------

/**
 * Aligns canvas backing-store dimensions to the responsive panel width.
 *
 * @param canvas - Target canvas.
 */
function syncCanvasToPanel(canvas: HTMLCanvasElement): void {
  const measuredWidthPx = Math.floor(canvas.clientWidth);
  const resolvedWidthPx = Math.max(MIN_CANVAS_WIDTH_PX, measuredWidthPx);
  const resolvedHeightPx = Math.max(
    MIN_CANVAS_HEIGHT_PX,
    Math.floor(resolvedWidthPx * CANVAS_ASPECT_RATIO),
  );

  canvas.style.width = '100%';
  canvas.style.height = `${resolvedHeightPx}px`;

  if (canvas.width !== resolvedWidthPx || canvas.height !== resolvedHeightPx) {
    canvas.width = resolvedWidthPx;
    canvas.height = resolvedHeightPx;
  }
}

// ---------------------------------------------------------------------------
// Input label panel
// ---------------------------------------------------------------------------

/**
 * Draws colored group bands and per-node chip labels in the left padding area.
 *
 * @param context      - Canvas 2D drawing context.
 * @param inputNodes   - Input nodes sorted top-to-bottom.
 * @param nodeDimensions - Node width/height from the resolved frame.
 * @returns Hit areas for all drawn groups and chips.
 */
function resolveAndDrawInputLabelPanel(
  _context: CanvasRenderingContext2D,
  inputNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): {
  inputDescriptionScenes: NetworkInputDescriptionScene[];
  inputGroupLabelBandScenes: NetworkInputGroupLabelBandScene[];
  hitAreas: MazeHitArea[];
} {
  const descriptionColumnWidthPx = resolveMazeDescriptionColumnWidthPx();
  const inputOverlayLayouts = resolveMazeInputOverlayLayouts(
    inputNodes,
    nodeDimensions,
    descriptionColumnWidthPx,
  );

  const inputDescriptionScenes = inputOverlayLayouts.flatMap(
    (inputOverlayLayout) => inputOverlayLayout.descriptionScenes,
  );
  const inputGroupLabelBandScenes = inputOverlayLayouts.map(
    (inputOverlayLayout) => ({
      label: inputOverlayLayout.groupDef.label,
      labelLines: inputOverlayLayout.groupDef.labelLines,
      tooltipHeading: inputOverlayLayout.groupDef.tooltipHeading,
      tooltipBodyParagraphs: inputOverlayLayout.groupDef.tooltipBodyParagraphs,
      leftPx: inputOverlayLayout.labelBandLeftPx,
      topPx: inputOverlayLayout.topPx,
      widthPx: FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX,
      heightPx: inputOverlayLayout.heightPx,
      backgroundColor: inputOverlayLayout.groupColor.bandFill,
      orientation: 'vertical' as const,
      nodeIndices: inputOverlayLayout.nodeIndices,
    }),
  );

  return {
    inputDescriptionScenes,
    inputGroupLabelBandScenes,
    hitAreas: [
      ...inputGroupLabelBandScenes.map((bandScene) => ({
        leftPx: bandScene.leftPx,
        topPx: bandScene.topPx,
        widthPx: bandScene.widthPx,
        heightPx: bandScene.heightPx,
        heading: bandScene.tooltipHeading,
        bodyParagraphs: bandScene.tooltipBodyParagraphs,
        hoveredNodeIndices: bandScene.nodeIndices,
      })),
      ...inputDescriptionScenes.map((descriptionScene) => ({
        leftPx: descriptionScene.leftPx,
        topPx: descriptionScene.topPx,
        widthPx: descriptionScene.widthPx,
        heightPx: descriptionScene.heightPx,
        heading: descriptionScene.tooltipHeading,
        bodyParagraphs: descriptionScene.tooltipBodyParagraphs,
        hoveredNodeIndices: [descriptionScene.nodeIndex],
      })),
    ],
  };
}

// ---------------------------------------------------------------------------
// Output label helpers
// ---------------------------------------------------------------------------

/**
 * Draws short direction labels (N / E / S / W) to the right of each output node.
 *
 * @param context        - Canvas 2D context.
 * @param outputNodes    - Output nodes sorted top-to-bottom.
 * @param nodeDimensions - Node dimensions for offset calculation.
 */
function drawOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
): void {
  for (const [outputIndex, node] of outputNodes.entries()) {
    const label = MAZE_OUTPUT_LABELS[outputIndex] ?? `OUT${outputIndex}`;
    context.fillStyle = '#8ad8ff';
    context.font = '700 10px Consolas, Menlo, Monaco, monospace';
    context.textAlign = 'left';
    context.textBaseline = 'middle';
    context.fillText(
      label,
      node.centerXPx + nodeDimensions.widthPx * 0.5 + 4,
      node.centerYPx,
    );
  }
}

function resolveMazeLabelPanelWidthPx(): number {
  return Math.max(
    MAZE_LABEL_LEFT_PADDING_PX,
    resolveMazeDescriptionColumnWidthPx() +
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX +
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX +
      FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX +
      12,
  );
}

function resolveMazeNetworkPanelPadding(canvasWidthPx: number): {
  topPx: number;
  rightPx: number;
  bottomPx: number;
  leftPx: number;
} {
  const preferredLeftPanelWidthPx = resolveMazeLabelPanelWidthPx();
  const preferredRightLegendReservePx = MAZE_RIGHT_LEGEND_RESERVE_PX;
  const minimumRightLegendReservePx =
    resolveMazeMinimumLegendReservePx(canvasWidthPx);

  let leftPanelWidthPx = preferredLeftPanelWidthPx;
  let rightLegendReservePx = preferredRightLegendReservePx;

  const preferredDrawableWidthPx =
    canvasWidthPx - leftPanelWidthPx - rightLegendReservePx;

  if (preferredDrawableWidthPx < MAZE_MIN_DRAWABLE_GRAPH_WIDTH_PX) {
    const requiredHorizontalSpacePx =
      MAZE_MIN_DRAWABLE_GRAPH_WIDTH_PX - preferredDrawableWidthPx;
    const rightReserveReductionPx = Math.min(
      requiredHorizontalSpacePx,
      rightLegendReservePx - minimumRightLegendReservePx,
    );

    rightLegendReservePx -= rightReserveReductionPx;

    const remainingRequiredHorizontalSpacePx =
      requiredHorizontalSpacePx - rightReserveReductionPx;
    if (remainingRequiredHorizontalSpacePx > 0) {
      leftPanelWidthPx -= Math.min(
        remainingRequiredHorizontalSpacePx,
        leftPanelWidthPx - MAZE_MIN_LEFT_LABEL_PANEL_WIDTH_PX,
      );
    }
  }

  return {
    topPx: 18,
    rightPx: rightLegendReservePx,
    bottomPx: 24,
    leftPx: leftPanelWidthPx,
  };
}

function resolveMazeMinimumLegendReservePx(canvasWidthPx: number): number {
  const legendWidthPx =
    canvasWidthPx < FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX
      ? FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX
      : FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX;

  return legendWidthPx + FLAPPY_NETWORK_LEGEND_MARGIN_PX * 2;
}

function resolveMazeDescriptionColumnWidthPx(): number {
  return Math.max(
    ...MAZE_INPUT_GROUP_DEFS.flatMap((groupDef) =>
      groupDef.nodeDescriptions.map((nodeDescription) =>
        resolveInputDescriptionChipWidthPx(nodeDescription.labelLines),
      ),
    ),
  );
}

function resolveMazeInputOverlayLayouts(
  inputNodes: PositionedNetworkNode[],
  nodeDimensions: NetworkNodeDimensions,
  descriptionColumnWidthPx: number,
): ResolvedMazeInputGroupLayout[] {
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const leftmostInputCenterXPx = Math.min(
    ...inputNodes.map((inputNode) => inputNode.centerXPx),
  );
  const inputLeftEdgeXPx = leftmostInputCenterXPx - halfNodeWidthPx;
  const descriptionRightXPx =
    inputLeftEdgeXPx - FLAPPY_NETWORK_INPUT_DESCRIPTION_GAP_PX;
  const descriptionColumnLeftXPx =
    descriptionRightXPx - descriptionColumnWidthPx;
  const labelBandRightXPx =
    descriptionColumnLeftXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX;
  const labelBandLeftXPx =
    labelBandRightXPx - FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX;
  const inputStackCenterYPx = resolveInputStackCenterYPx(inputNodes);

  let runningTopPx = 0;

  const relativeLayouts = MAZE_INPUT_GROUP_DEFS.map((groupDef, groupIndex) => {
    const groupColor = MAZE_GROUP_COLORS[groupIndex];
    const groupNodeStartIndex = MAZE_INPUT_GROUP_DEFS.slice(
      0,
      groupIndex,
    ).reduce(
      (runningNodeCount, currentGroupDef) =>
        runningNodeCount + currentGroupDef.nodeCount,
      0,
    );
    const resolvedGroupNodes = inputNodes.slice(
      groupNodeStartIndex,
      groupNodeStartIndex + groupDef.nodeCount,
    );

    const descriptionScenes = groupDef.nodeDescriptions.map(
      (nodeDescription, nodeOffset) => {
        const descriptionWidthPx = resolveInputDescriptionChipWidthPx(
          nodeDescription.labelLines,
        );
        const descriptionHeightPx = Math.max(
          FLAPPY_NETWORK_INPUT_DESCRIPTION_MIN_HEIGHT_PX,
          nodeDescription.labelLines.length *
            FLAPPY_NETWORK_INPUT_DESCRIPTION_LINE_HEIGHT_PX +
            FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_VERTICAL_PADDING_PX * 2,
          nodeDimensions.heightPx +
            FLAPPY_NETWORK_INPUT_DESCRIPTION_TEXT_VERTICAL_PADDING_PX * 2,
        );

        return {
          labelLines: nodeDescription.labelLines,
          tooltipHeading: nodeDescription.tooltipHeading,
          tooltipBodyParagraphs: nodeDescription.tooltipBodyParagraphs,
          leftPx: descriptionRightXPx - descriptionWidthPx,
          topPx: 0,
          widthPx: descriptionWidthPx,
          heightPx: descriptionHeightPx,
          nodeIndex:
            resolvedGroupNodes[nodeOffset]?.index ??
            groupNodeStartIndex + nodeOffset,
        };
      },
    );

    const groupedDescriptionContentHeightPx =
      resolveGroupedDescriptionContentHeightPx(descriptionScenes);
    const groupHeightPx = Math.max(
      FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX,
      groupedDescriptionContentHeightPx +
        FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX * 2,
    );
    const descriptionContentTopPx =
      runningTopPx +
      Math.max(
        FLAPPY_NETWORK_INPUT_GROUP_PADDING_PX,
        (groupHeightPx - groupedDescriptionContentHeightPx) * 0.5,
      );

    let runningDescriptionTopPx = descriptionContentTopPx;
    const placedDescriptionScenes = descriptionScenes.map(
      (descriptionScene, descriptionIndex) => {
        const resolvedTopPx = runningDescriptionTopPx;
        runningDescriptionTopPx += descriptionScene.heightPx;
        if (descriptionIndex < descriptionScenes.length - 1) {
          runningDescriptionTopPx +=
            FLAPPY_NETWORK_INPUT_DESCRIPTION_CHIP_VERTICAL_GAP_PX;
        }

        return {
          ...descriptionScene,
          topPx: resolvedTopPx,
        };
      },
    );

    const resolvedLayout: ResolvedMazeInputGroupLayout = {
      groupDef,
      groupColor,
      nodeIndices: resolvedGroupNodes.map((node) => node.index),
      labelBandLeftPx: labelBandLeftXPx,
      topPx: runningTopPx,
      heightPx: groupHeightPx,
      descriptionScenes: placedDescriptionScenes,
    };

    runningTopPx += groupHeightPx;
    if (groupIndex < MAZE_INPUT_GROUP_DEFS.length - 1) {
      runningTopPx += FLAPPY_NETWORK_INPUT_GROUP_VERTICAL_GAP_PX;
    }

    return resolvedLayout;
  }).filter(
    (layout): layout is ResolvedMazeInputGroupLayout =>
      layout.groupColor != null,
  );

  const totalOverlayHeightPx = relativeLayouts.at(-1)
    ? relativeLayouts.at(-1)!.topPx + relativeLayouts.at(-1)!.heightPx
    : 0;
  const overlayTopOffsetPx = inputStackCenterYPx - totalOverlayHeightPx * 0.5;

  return relativeLayouts.map((relativeLayout) => ({
    ...relativeLayout,
    topPx: relativeLayout.topPx + overlayTopOffsetPx,
    descriptionScenes: relativeLayout.descriptionScenes.map(
      (descriptionScene) => ({
        ...descriptionScene,
        topPx: descriptionScene.topPx + overlayTopOffsetPx,
      }),
    ),
  }));
}

function resolveGroupedDescriptionContentHeightPx(
  descriptionScenes: readonly NetworkInputDescriptionScene[],
): number {
  if (descriptionScenes.length === 0) {
    return 0;
  }

  return descriptionScenes.reduce(
    (runningHeightPx, descriptionScene, descriptionIndex) =>
      runningHeightPx +
      descriptionScene.heightPx +
      (descriptionIndex < descriptionScenes.length - 1
        ? FLAPPY_NETWORK_INPUT_DESCRIPTION_CHIP_VERTICAL_GAP_PX
        : 0),
    0,
  );
}

function resolveInputStackCenterYPx(
  inputNodes: readonly PositionedNetworkNode[],
): number {
  const firstInputNode = inputNodes[0];
  const lastInputNode = inputNodes.at(-1);

  if (!firstInputNode || !lastInputNode) {
    return 0;
  }

  return (firstInputNode.centerYPx + lastInputNode.centerYPx) * 0.5;
}

function resolveRuntimePositionedNodes(
  positionedNodes: PositionedNetworkNode[],
): MazeRuntimePositionedNode[] {
  return positionedNodes.map((positionedNode) => ({
    xPx: positionedNode.centerXPx,
    yPx: positionedNode.centerYPx,
    node: {
      index: positionedNode.index,
      type: positionedNode.type,
      bias: positionedNode.bias,
    },
  }));
}

function resolveRuntimeConnections(
  graph: VisualizationGraphV1,
): MazeRuntimeConnection[] {
  return graph.edges.map((edge) => ({
    from: { index: edge.from },
    to: { index: edge.to },
    weight: edge.weight,
    enabled: edge.enabled !== false,
  }));
}

function paintMazeNetworkBase(
  context: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
): void {
  context.fillStyle = FLAPPY_UI_NETWORK_CANVAS_BACKGROUND;
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function resolveMazeArchitectureLabel(
  network: Network,
  graph: VisualizationGraphV1,
): string {
  const architectureDescriptor = network.describeArchitecture();
  const hiddenLayersLabel = resolveMazeHiddenLayersLabel(
    architectureDescriptor.hiddenLayerSizes,
    architectureDescriptor.source,
  );
  const architectureColumnsLabel = [
    graph.io.inputNodeIds.length,
    hiddenLayersLabel,
    graph.io.outputNodeIds.length,
  ].join(FLAPPY_NETWORK_ARCHITECTURE_COLUMN_SEPARATOR);
  const architectureTotalsLabel = `(${architectureDescriptor.totalNodes} nodes, ${architectureDescriptor.totalConnections} connections)`;

  return [architectureColumnsLabel, architectureTotalsLabel].join(
    FLAPPY_NETWORK_ARCHITECTURE_LINE_SEPARATOR,
  );
}

function resolveMazeHiddenLayersLabel(
  hiddenLayerSizes: number[],
  architectureSource: 'layer-metadata' | 'graph-topology' | 'inferred',
): string {
  if (hiddenLayerSizes.length === 0) {
    return FLAPPY_NETWORK_EMPTY_HIDDEN_LAYER_LABEL;
  }

  if (architectureSource === 'inferred') {
    return hiddenLayerSizes
      .map(
        (hiddenLayerSize) =>
          `${FLAPPY_NETWORK_INFERRED_HIDDEN_LAYER_PREFIX}${hiddenLayerSize}`,
      )
      .join(FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR);
  }

  return hiddenLayerSizes.join(FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR);
}
