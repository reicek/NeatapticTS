/**
 * ASCII Maze adapter over the shared rich browser network visualizer.
 *
 * The maze demo should not own its own network-frame math. Instead it reuses
 * the same resolved frame, padding, node sizing, and connection drawing path
 * as Flappy Bird, while only swapping the semantic input labels and the short
 * output tags.
 */

import type Network from '../../../../src/architecture/network';
import type { VisualizationGraphV1 } from '../../../../src/architecture/network/visualization/network.visualization.types';
import {
  drawResolvedNetworkVisualization,
  resolveNetworkArchitectureLabel as resolveSharedNetworkArchitectureLabel,
  resolveNetworkVisualizationFrame as resolveSharedNetworkVisualizationFrame,
  type NetworkVisualizationResolvedFrame,
} from '../../../flappy_bird/browser-entry/network-view/network-view';
import type { InputLabelGroupDefinition } from '../../../flappy_bird/browser-entry/network-view/network-view.types';
import {
  resolveNetworkVisualizationTopologyPlan as resolveSharedNetworkVisualizationTopologyPlan,
  type NetworkVisualizationTopologyPlan,
} from '../../../flappy_bird/browser-entry/network-view/network-view.topology.utils';
import { resolveNetworkVisualizationColorScales } from '../../../flappy_bird/browser-entry/visualization/visualization.colors.utils';
import type {
  NetworkVisualizationAnimatedHoveredNode,
  NetworkVisualizationHoverState,
  PositionedNetworkNodeLike,
} from '../../../flappy_bird/browser-entry/browser-entry.visualization.types';
import {
  MAZE_GROUP_COLORS,
  MAZE_INPUT_GROUP_DEFS,
  MAZE_OUTPUT_LABELS,
} from './network-view.constants';

/** Minimum canvas backing-store width. */
const MIN_CANVAS_WIDTH_PX = 320;

/** Minimum canvas backing-store height. */
const MIN_CANVAS_HEIGHT_PX = 240;

/** Canvas height expressed as a fraction of the canvas width. */
const CANVAS_ASPECT_RATIO = 0.64;

/**
 * A canvas-space rectangular hit area with associated tooltip content.
 */
export interface MazeHitArea {
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  heading: string;
  bodyParagraphs: readonly string[];
  hoveredNodeIndices: readonly number[];
  suppressTooltip?: boolean;
}

/**
 * Full return value from `drawMazeNetworkVisualization`.
 */
export interface MazeNetworkRenderResult {
  frame: NetworkVisualizationResolvedFrame;
  hitAreas: MazeHitArea[];
}

/**
 * Draw the ASCII Maze network panel using the shared Flappy visualizer owner.
 *
 * @param canvas - Canvas element to render onto.
 * @param network - Runtime network used for architecture metadata and weights.
 * @param graph - Exported graph carrying authoritative input/output counts.
 * @param hoveredNodeIndices - Host-owned hovered node ids.
 * @returns Shared resolved frame plus maze hover hit areas.
 */
export function drawMazeNetworkVisualization(
  canvas: HTMLCanvasElement,
  network: Network,
  graph: VisualizationGraphV1,
  hoveredNodeIndices: readonly number[] = [],
): MazeNetworkRenderResult {
  syncCanvasToPanel(canvas);

  const context = canvas.getContext('2d');
  if (!context) {
    return {
      frame: createEmptyMazeResolvedFrame(canvas, network, graph),
      hitAreas: [],
    };
  }

  const resolvedFrame = resolveSharedNetworkVisualizationFrame(
    context,
    network,
    graph.io.inputNodeIds.length,
    graph.io.outputNodeIds.length,
    resolveMazeInputLabelGroupDefinitions(),
  );
  const hoverState = resolveMazeHoverState(hoveredNodeIndices);

  drawResolvedNetworkVisualization(context, resolvedFrame, hoverState);
  drawOutputNodeLabels(
    context,
    resolveSortedOutputNodes(resolvedFrame.positionedScene.positionedNodes),
    MAZE_OUTPUT_LABELS,
  );

  return {
    frame: resolvedFrame,
    hitAreas: resolveMazeHitAreas(resolvedFrame),
  };
}

/**
 * Resolve the shared visualizer input-label definitions for the maze demo.
 *
 * @returns Maze semantic label groups expressed in the shared visualizer format.
 */
export function resolveMazeInputLabelGroupDefinitions(): readonly InputLabelGroupDefinition[] {
  return MAZE_INPUT_GROUP_DEFS.map((mazeInputGroupDefinition, groupIndex) => ({
    label: mazeInputGroupDefinition.label,
    labelLines: mazeInputGroupDefinition.labelLines,
    tooltipHeading: mazeInputGroupDefinition.tooltipHeading,
    tooltipBodyParagraphs: mazeInputGroupDefinition.tooltipBodyParagraphs,
    nodeDescriptionDefinitions: mazeInputGroupDefinition.nodeDescriptions.map(
      (nodeDescriptionDefinition) => ({
        labelLines: nodeDescriptionDefinition.labelLines,
        tooltipHeading: nodeDescriptionDefinition.tooltipHeading,
        tooltipBodyParagraphs: nodeDescriptionDefinition.tooltipBodyParagraphs,
      }),
    ),
    backgroundColor: MAZE_GROUP_COLORS[groupIndex]?.bandFill ?? '#2bd9ff',
    orientation: 'vertical',
  }));
}

/**
 * Resolve responsive network-canvas dimensions from the host panel shelf.
 *
 * @param measuredWidthPx - Current measured canvas width from layout.
 * @param measuredHeightPx - Current measured host-panel height from layout.
 * @returns Width and height for the canvas backing store.
 */
export function resolveMazeNetworkCanvasDimensions(
  measuredWidthPx: number,
  measuredHeightPx: number,
): {
  widthPx: number;
  heightPx: number;
} {
  const resolvedWidthPx = Math.max(
    MIN_CANVAS_WIDTH_PX,
    Math.floor(measuredWidthPx),
  );
  const widthDrivenHeightPx = Math.max(
    MIN_CANVAS_HEIGHT_PX,
    Math.floor(resolvedWidthPx * CANVAS_ASPECT_RATIO),
  );
  const measuredPanelHeightPx = Math.floor(measuredHeightPx);
  const resolvedHeightPx =
    Number.isFinite(measuredPanelHeightPx) && measuredPanelHeightPx > 0
      ? measuredPanelHeightPx
      : widthDrivenHeightPx;

  return {
    widthPx: resolvedWidthPx,
    heightPx: resolvedHeightPx,
  };
}

/**
 * Resolve the compact architecture summary for the maze network legend.
 *
 * @param network - Runtime network being visualized.
 * @param graph - Exported graph carrying authoritative input/output counts.
 * @returns Shared architecture label with maze IO counts.
 */
export function resolveMazeArchitectureLabel(
  network: Network,
  graph: VisualizationGraphV1,
): string {
  return resolveSharedNetworkArchitectureLabel(
    network,
    graph.io.inputNodeIds.length,
    graph.io.outputNodeIds.length,
  );
}

/**
 * Resolve the shared topology plan for the maze network.
 *
 * @param network - Runtime network being visualized.
 * @param graph - Exported graph carrying authoritative input/output counts.
 * @returns Shared topology plan with recurrent annotations when present.
 */
export function resolveMazeVisualizationTopologyPlan(
  network: Network,
  graph: VisualizationGraphV1,
): NetworkVisualizationTopologyPlan {
  return resolveSharedNetworkVisualizationTopologyPlan(
    network,
    graph.io.inputNodeIds.length,
    graph.io.outputNodeIds.length,
  );
}

function syncCanvasToPanel(canvas: HTMLCanvasElement): void {
  const hostElement = canvas.parentElement;
  const measuredWidthPx = resolveHostContentBoxDimensionPx(
    hostElement,
    hostElement?.clientWidth ?? canvas.clientWidth,
    'paddingLeft',
    'paddingRight',
  );
  const measuredHeightPx = resolveHostContentBoxDimensionPx(
    hostElement,
    hostElement?.clientHeight ?? canvas.clientHeight,
    'paddingTop',
    'paddingBottom',
  );
  const { widthPx: resolvedWidthPx, heightPx: resolvedHeightPx } =
    resolveMazeNetworkCanvasDimensions(measuredWidthPx, measuredHeightPx);

  canvas.style.width = `${resolvedWidthPx}px`;
  canvas.style.height = `${resolvedHeightPx}px`;

  if (canvas.width !== resolvedWidthPx || canvas.height !== resolvedHeightPx) {
    canvas.width = resolvedWidthPx;
    canvas.height = resolvedHeightPx;
  }
}

function resolveHostContentBoxDimensionPx(
  hostElement: HTMLElement | null,
  hostClientDimensionPx: number,
  startPaddingProperty:
    'paddingBottom' | 'paddingLeft' | 'paddingRight' | 'paddingTop',
  endPaddingProperty:
    'paddingBottom' | 'paddingLeft' | 'paddingRight' | 'paddingTop',
): number {
  if (!hostElement) {
    return hostClientDimensionPx;
  }

  const defaultView = hostElement.ownerDocument?.defaultView;
  const computedStyle = defaultView?.getComputedStyle(hostElement);
  const startPaddingPx = Number.parseFloat(
    computedStyle?.[startPaddingProperty] ?? '0',
  );
  const endPaddingPx = Number.parseFloat(
    computedStyle?.[endPaddingProperty] ?? '0',
  );
  const resolvedStartPaddingPx = Number.isFinite(startPaddingPx)
    ? startPaddingPx
    : 0;
  const resolvedEndPaddingPx = Number.isFinite(endPaddingPx) ? endPaddingPx : 0;

  return Math.max(
    0,
    hostClientDimensionPx - resolvedStartPaddingPx - resolvedEndPaddingPx,
  );
}

function resolveMazeHoverState(
  hoveredNodeIndices: readonly number[],
): NetworkVisualizationHoverState | undefined {
  if (hoveredNodeIndices.length === 0) {
    return undefined;
  }

  const animatedHoveredNodes: readonly NetworkVisualizationAnimatedHoveredNode[] =
    hoveredNodeIndices.map((nodeIndex) => ({
      nodeIndex,
      intensity: 1,
    }));

  return {
    hoveredNodeIndices,
    animatedHoveredNodes,
  };
}

function resolveSortedOutputNodes(
  positionedNodes: readonly PositionedNetworkNodeLike[],
): PositionedNetworkNodeLike[] {
  return positionedNodes
    .filter((positionedNode) => positionedNode.node.type === 'output')
    .toSorted(
      (positionedNodeA, positionedNodeB) =>
        positionedNodeA.yPx - positionedNodeB.yPx,
    );
}

function drawOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: readonly PositionedNetworkNodeLike[],
  outputLabels: readonly string[],
): void {
  context.save();
  context.fillStyle = '#8ad8ff';
  context.font = '700 10px Consolas, Menlo, Monaco, monospace';
  context.textAlign = 'left';
  context.textBaseline = 'middle';

  outputNodes.forEach((outputNode, outputIndex) => {
    const outputLabel = outputLabels[outputIndex] ?? `OUT${outputIndex}`;
    context.fillText(outputLabel, outputNode.xPx + 18, outputNode.yPx);
  });

  context.restore();
}

function resolveMazeHitAreas(
  resolvedFrame: NetworkVisualizationResolvedFrame,
): MazeHitArea[] {
  const nodeDimensions = resolvedFrame.positionedScene.nodeDimensions;
  const halfNodeWidthPx = nodeDimensions.widthPx * 0.5;
  const halfNodeHeightPx = nodeDimensions.heightPx * 0.5;

  const inputGroupHitAreas =
    resolvedFrame.positionedScene.inputGroupLabelBandScenes.map(
      (inputGroupLabelBandScene) => ({
        leftPx: inputGroupLabelBandScene.leftPx,
        topPx: inputGroupLabelBandScene.topPx,
        widthPx: inputGroupLabelBandScene.widthPx,
        heightPx: inputGroupLabelBandScene.heightPx,
        heading: inputGroupLabelBandScene.tooltipHeading,
        bodyParagraphs: inputGroupLabelBandScene.tooltipBodyParagraphs,
        hoveredNodeIndices: inputGroupLabelBandScene.nodeIndices,
      }),
    );
  const inputDescriptionHitAreas =
    resolvedFrame.positionedScene.inputDescriptionScenes.map(
      (inputDescriptionScene) => ({
        leftPx: inputDescriptionScene.leftPx,
        topPx: inputDescriptionScene.topPx,
        widthPx: inputDescriptionScene.widthPx,
        heightPx: inputDescriptionScene.heightPx,
        heading: inputDescriptionScene.tooltipHeading,
        bodyParagraphs: inputDescriptionScene.tooltipBodyParagraphs,
        hoveredNodeIndices: [inputDescriptionScene.nodeIndex],
      }),
    );
  const hiddenColumnHitAreas = (
    resolvedFrame.positionedScene.hiddenColumnLabelScenes ?? []
  ).map((hiddenColumnLabelScene) => ({
    leftPx: hiddenColumnLabelScene.leftPx,
    topPx: hiddenColumnLabelScene.topPx,
    widthPx: hiddenColumnLabelScene.widthPx,
    heightPx: hiddenColumnLabelScene.heightPx,
    heading: hiddenColumnLabelScene.tooltipHeading,
    bodyParagraphs: hiddenColumnLabelScene.tooltipBodyParagraphs,
    hoveredNodeIndices: hiddenColumnLabelScene.nodeIndices,
  }));
  const nodeHitAreas = resolvedFrame.positionedScene.positionedNodes.map(
    (positionedNode) => ({
      leftPx: positionedNode.xPx - halfNodeWidthPx,
      topPx: positionedNode.yPx - halfNodeHeightPx,
      widthPx: nodeDimensions.widthPx,
      heightPx: nodeDimensions.heightPx,
      heading: '',
      bodyParagraphs: [],
      hoveredNodeIndices: [positionedNode.node.index],
      suppressTooltip: true,
    }),
  );

  return [
    ...inputGroupHitAreas,
    ...inputDescriptionHitAreas,
    ...hiddenColumnHitAreas,
    ...nodeHitAreas,
  ];
}

function createEmptyMazeResolvedFrame(
  canvas: HTMLCanvasElement,
  network: Network,
  graph: VisualizationGraphV1,
): NetworkVisualizationResolvedFrame {
  return {
    canvasWidthPx: canvas.width,
    canvasHeightPx: canvas.height,
    architectureLabel: resolveMazeArchitectureLabel(network, graph),
    colorScales: resolveNetworkVisualizationColorScales(network),
    hideNetworkOverlays: false,
    positionedScene: {
      positionedNodes: [],
      nodeDimensions: { widthPx: 0, heightPx: 0 },
      inputGroupLabelBandScenes: [],
      inputDescriptionScenes: [],
      hiddenColumnLabelScenes: [],
    },
    positionByNodeIndex: new Map(),
    runtimeConnections: [],
  };
}
