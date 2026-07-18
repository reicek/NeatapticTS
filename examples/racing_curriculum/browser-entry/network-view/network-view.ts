/**
 * Racing curriculum adapter over the shared rich browser network visualizer.
 *
 * The racing browser demo should not own its own network-frame math. Instead it
 * reuses the same resolved frame, padding, node sizing, and connection drawing
 * path as Flappy Bird and ASCII Maze, while only swapping the semantic input
 * labels and the short output tags to match the Tier 1 racing controller.
 */

import type Network from '../../../../src/architecture/network';
import {
  drawResolvedNetworkVisualization,
  resolveNetworkArchitectureLabel as resolveSharedNetworkArchitectureLabel,
  resolveNetworkVisualizationFrame as resolveSharedNetworkVisualizationFrame,
  type NetworkVisualizationResolvedFrame,
} from '../../../flappy_bird/browser-entry/network-view/network-view';

export type { NetworkVisualizationResolvedFrame };
import type { InputLabelGroupDefinition } from '../../../flappy_bird/browser-entry/network-view/network-view.types';
import type {
  NetworkVisualizationHoverState,
  PositionedNetworkNodeLike,
} from '../../../flappy_bird/browser-entry/browser-entry.visualization.types';
import type { NetworkVisualizationPositionedScene } from '../host/host.network-tooltip.service';
import {
  RACING_GROUP_COLORS,
  RACING_INPUT_SIZE,
  RACING_NETWORK_CONNECTION_LAYER_STYLE,
  RACING_OUTPUT_LABELS,
  RACING_OUTPUT_SIZE,
  resolveRacingInputGroupDefinitions,
} from './network-view.constants';
import {
  drawRacingNetworkLODFromFrame,
  resolveRacingNetworkLODFrame,
  shouldUseRacingNetworkLOD,
  type RacingLODResolvedFrame,
} from './network-view.lod';

/** Minimum canvas backing-store width. */
const MIN_CANVAS_WIDTH_PX = 320;

/** Minimum canvas backing-store height. */
const MIN_CANVAS_HEIGHT_PX = 240;

/** Canvas height expressed as a fraction of the canvas width. */
const CANVAS_ASPECT_RATIO = 0.75;

/**
 * Full return value from `drawRacingNetworkVisualization`.
 */
export interface RacingNetworkRenderResult {
  /** Reusable resolved frame cache for hover-only redraws. */
  frame: NetworkVisualizationResolvedFrame;
  /** Positioned scene used by the racing tooltip resolver. */
  positionedScene: NetworkVisualizationPositionedScene;
}

/**
 * Resolves the shared visualizer input-label definitions for the racing demo.
 *
 * @param inputSize - Optional network input size; when 124, the full Tier 6
 *   label set is returned, otherwise the Tier 1 set is returned.
 * @returns Racing semantic label groups expressed in the shared visualizer format.
 */
export function resolveRacingInputLabelGroupDefinitions(
  inputSize?: number,
): readonly InputLabelGroupDefinition[] {
  return resolveRacingInputGroupDefinitions(inputSize).map(
    (racingInputGroupDefinition, groupIndex) => ({
      label: racingInputGroupDefinition.label,
      labelLines: racingInputGroupDefinition.labelLines,
      tooltipHeading: racingInputGroupDefinition.tooltipHeading,
      tooltipBodyParagraphs: racingInputGroupDefinition.tooltipBodyParagraphs,
      nodeDescriptionDefinitions:
        racingInputGroupDefinition.nodeDescriptions.map(
          (nodeDescriptionDefinition) => ({
            labelLines: nodeDescriptionDefinition.labelLines,
            tooltipHeading: nodeDescriptionDefinition.tooltipHeading,
            tooltipBodyParagraphs:
              nodeDescriptionDefinition.tooltipBodyParagraphs,
          }),
        ),
      backgroundColor: RACING_GROUP_COLORS[groupIndex]?.bandFill ?? '#2bd9ff',
      orientation: 'vertical',
    }),
  );
}

/**
 * Resolve responsive network-canvas dimensions from the host panel shelf.
 *
 * @param measuredWidthPx - Current measured canvas width from layout.
 * @param measuredHeightPx - Current measured host-panel height from layout.
 * @returns Width and height for the canvas backing store.
 */
export function resolveRacingNetworkCanvasDimensions(
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
 * Resolve the compact architecture summary for the racing network legend.
 *
 * @param network - Runtime network being visualized.
 * @returns Shared architecture label with racing IO counts.
 */
export function resolveRacingArchitectureLabel(network: Network): string {
  return resolveSharedNetworkArchitectureLabel(
    network,
    network.input ?? RACING_INPUT_SIZE,
    RACING_OUTPUT_SIZE,
  );
}

/**
 * Resolves a reusable network-visualization frame for the racing controller.
 *
 * Host-owned caching can reuse this frame between pointer-driven redraws so the
 * expensive topology/layout work is not repeated for every hover event.
 *
 * @param context - Canvas 2D drawing context.
 * @param network - Network to visualize.
 * @returns Reusable resolved frame for subsequent draw passes.
 */
export function resolveRacingNetworkVisualizationFrame(
  context: CanvasRenderingContext2D,
  network: Network | undefined,
): NetworkVisualizationResolvedFrame {
  const inputSize = network?.input ?? RACING_INPUT_SIZE;

  if (network && shouldUseRacingNetworkLOD(network)) {
    return resolveRacingNetworkLODFrame(context, network);
  }

  return resolveSharedNetworkVisualizationFrame(
    context,
    network,
    inputSize,
    RACING_OUTPUT_SIZE,
    resolveRacingInputLabelGroupDefinitions(inputSize),
  );
}

/**
 * Draws a previously resolved racing network-visualization frame.
 *
 * The host uses this path for hover-only repaint work because it can reuse the
 * cached static frame and only vary interactive emphasis.
 *
 * @param context - Canvas 2D drawing context.
 * @param resolvedFrame - Reusable frame cache.
 * @param hoveredNodeIndices - Optional host-owned hovered node ids.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 */
export function drawRacingNetworkVisualizationFromFrame(
  context: CanvasRenderingContext2D,
  resolvedFrame: NetworkVisualizationResolvedFrame,
  hoveredNodeIndices?: readonly number[],
): NetworkVisualizationPositionedScene {
  if (isRacingLODResolvedFrame(resolvedFrame)) {
    const positionedScene = drawRacingNetworkLODFromFrame(
      context,
      resolvedFrame,
      hoveredNodeIndices,
      RACING_NETWORK_CONNECTION_LAYER_STYLE,
    );
    drawRacingOutputNodeLabels(
      context,
      resolveSortedOutputNodes(positionedScene.positionedNodes),
    );

    return positionedScene as unknown as NetworkVisualizationPositionedScene;
  }

  const hoverState = resolveRacingHoverState(hoveredNodeIndices);

  const positionedScene = drawResolvedNetworkVisualization(
    context,
    resolvedFrame,
    hoverState,
    RACING_NETWORK_CONNECTION_LAYER_STYLE,
  );
  drawRacingOutputNodeLabels(
    context,
    resolveSortedOutputNodes(positionedScene.positionedNodes),
  );

  return positionedScene as unknown as NetworkVisualizationPositionedScene;
}

function isRacingLODResolvedFrame(
  resolvedFrame: NetworkVisualizationResolvedFrame,
): resolvedFrame is RacingLODResolvedFrame {
  return '__racingLod' in resolvedFrame && resolvedFrame.__racingLod === true;
}

/**
 * Draw the racing curriculum network panel using the shared Flappy visualizer.
 *
 * @param canvas - Canvas element to render onto.
 * @param network - Runtime network used for architecture metadata and weights.
 * @param hoveredNodeIndices - Optional host-owned hovered node ids.
 * @returns Positioned node snapshot reused by host-side hover hit testing.
 */
export function drawRacingNetworkVisualization(
  canvas: HTMLCanvasElement,
  network: Network | undefined,
  hoveredNodeIndices?: readonly number[],
): NetworkVisualizationPositionedScene {
  syncRacingCanvasToPanel(canvas);

  const context = canvas.getContext('2d', { desynchronized: true });
  if (!context) {
    return {
      positionedNodes: [],
      nodeDimensions: { widthPx: 0, heightPx: 0 },
      inputDescriptionScenes: [],
      inputGroupLabelBandScenes: [],
      hiddenColumnLabelScenes: [],
    } as NetworkVisualizationPositionedScene;
  }

  const resolvedFrame = resolveRacingNetworkVisualizationFrame(
    context,
    network,
  );
  return drawRacingNetworkVisualizationFromFrame(
    context,
    resolvedFrame,
    hoveredNodeIndices,
  );
}

function syncRacingCanvasToPanel(canvas: HTMLCanvasElement): void {
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
    resolveRacingNetworkCanvasDimensions(measuredWidthPx, measuredHeightPx);

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

function resolveRacingHoverState(
  hoveredNodeIndices: readonly number[] | undefined,
): NetworkVisualizationHoverState | undefined {
  if (!hoveredNodeIndices || hoveredNodeIndices.length === 0) {
    return undefined;
  }

  return {
    hoveredNodeIndices,
    animatedHoveredNodes: hoveredNodeIndices.map((nodeIndex) => ({
      nodeIndex,
      intensity: 1,
    })),
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

function drawRacingOutputNodeLabels(
  context: CanvasRenderingContext2D,
  outputNodes: readonly PositionedNetworkNodeLike[],
): void {
  context.save();
  context.fillStyle = '#8ad8ff';
  context.font = '700 10px Consolas, Menlo, Monaco, monospace';
  context.textAlign = 'left';
  context.textBaseline = 'middle';

  outputNodes.forEach((outputNode, outputIndex) => {
    const outputLabel =
      RACING_OUTPUT_LABELS[outputIndex] ?? `OUT${outputIndex}`;
    context.fillText(outputLabel, outputNode.xPx + 18, outputNode.yPx);
  });

  context.restore();
}
