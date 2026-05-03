/**
 * Shared canvas renderer for browser network visualization.
 *
 * This module provides the main public API: `renderNetworkView(canvas, graph, options)`.
 * It orchestrates the layout, topology inference, and drawing steps, allowing demos
 * to inject custom overlays via optional hooks while keeping the core renderer
 * demo-agnostic.
 */

import type { VisualizationGraphV1 } from '../../architecture/network';
import type {
  NetworkNodeDimensions,
  EdgePadding,
  NetworkVisualizationColorScales,
  RenderNetworkViewOptions,
  NetworkVisualizationResolvedFrame,
} from './network-view.types';
import {
  positionNetworkNodes,
  centerPositionedNodesInDrawableArea,
  type VisualNetworkNode,
} from './network-view.layout.utils';
/**
 * Default node dimensions (px).
 */
const DEFAULT_NODE_DIMENSIONS: NetworkNodeDimensions = {
  widthPx: 24,
  heightPx: 24,
};

/**
 * Default panel padding (px).
 */
const DEFAULT_PANEL_PADDING: EdgePadding = {
  topPx: 32,
  rightPx: 32,
  bottomPx: 32,
  leftPx: 32,
};

/**
 * Default color scales for visualization.
 */
const DEFAULT_COLOR_SCALES: NetworkVisualizationColorScales = {
  weightPositive: '#00ff88',
  weightNegative: '#ff0088',
  activationHot: '#ffcc00',
  activationCold: '#0088ff',
  bias: '#8800ff',
};

/**
 * Inner graph padding (space around nodes within the drawable area).
 */
const INNER_GRAPH_PADDING_PX = 16;

/**
 * Target gap between input-layer nodes (can be overridden via options).
 */
const DEFAULT_INPUT_LAYER_TARGET_GAP_PX = 8;

/**
 * Renders a network visualization onto a canvas.
 *
 * This is the main public entry point. It accepts a `VisualizationGraphV1` (from
 * `exportVisualizationGraph`), lays out the nodes, and draws them with optional
 * demo-specific overlays.
 *
 * **Typical usage:**
 * ```ts
 * const graph = exportVisualizationGraph(network);
 * const canvas = document.getElementById('network-canvas') as HTMLCanvasElement;
 * const frame = renderNetworkView(canvas, graph, {
 *   nodeDimensions: { widthPx: 32, heightPx: 32 },
 *   overlayFactory: { createDemoOverlayScenes: myCustomOverlays },
 * });
 * // frame contains positioned nodes for hover hit testing
 * ```
 *
 * @param canvas - Canvas element to render onto.
 * @param graph - Visualization graph (from `exportVisualizationGraph`).
 * @param options - Optional render settings (dimensions, padding, colors, overlays).
 * @returns Resolved frame with positioned nodes and scene state (reusable for hover).
 */
export function renderNetworkView(
  canvas: HTMLCanvasElement,
  graph: VisualizationGraphV1,
  options?: RenderNetworkViewOptions,
): NetworkVisualizationResolvedFrame {
  // Step 1: Resolve options with defaults.
  const nodeDimensions = options?.nodeDimensions ?? DEFAULT_NODE_DIMENSIONS;
  const panelPadding = options?.panelPaddingPx ?? DEFAULT_PANEL_PADDING;
  const colorScales = options?.colorScales ?? DEFAULT_COLOR_SCALES;

  // Step 2: Infer topology and layers from the graph.
  const networkLayers = mapGraphToNetworkLayers(graph);
  const topologyMode = (graph.metadata?.mode ?? 'recurrent') === 'acyclic' ? ('acyclic' as const) : ('recurrent' as const);

  // Step 3: Calculate drawable area.
  const canvasWidthPx = canvas.width;
  const canvasHeightPx = canvas.height;
  const drawableWidthPx = canvasWidthPx - panelPadding.leftPx - panelPadding.rightPx;
  const drawableHeightPx = canvasHeightPx - panelPadding.topPx - panelPadding.bottomPx;

  // Step 4: Position nodes in canvas coordinates.
  const positionedNodes = positionNetworkNodes(
    networkLayers,
    panelPadding.leftPx,
    panelPadding.topPx,
    drawableWidthPx,
    drawableHeightPx,
    INNER_GRAPH_PADDING_PX,
    nodeDimensions,
    DEFAULT_INPUT_LAYER_TARGET_GAP_PX,
  );

  // Step 5: Center nodes horizontally.
  const centeredPositionedNodes = centerPositionedNodesInDrawableArea(
    positionedNodes,
    drawableWidthPx,
  );

  // Step 6: Convert graph edges to visual connections.
  const connections = graph.edges.map((edge) => ({
    fromIndex: edge.from,
    toIndex: edge.to,
    weight: edge.weight,
    enabled: true,
  }));

  // Step 7: Create resolved frame (for reuse in hover/incremental redraws).
  const frame: NetworkVisualizationResolvedFrame = {
    canvasWidthPx,
    canvasHeightPx,
    positionedNodes: centeredPositionedNodes,
    connections,
    nodeDimensions,
    colorScales,
    topologyMode,
  };

  // Step 8: Draw the frame.
  const context = canvas.getContext('2d');
  if (context) {
    drawNetworkVisualization(context, frame);
  }

  return frame;
}

/**
 * Converts a VisualizationGraphV1 into network layers for layout.
 *
 * @param graph - Visualization graph.
 * @returns Layered nodes (input, hidden, output).
 */
function mapGraphToNetworkLayers(graph: VisualizationGraphV1): VisualNetworkNode[][] {
  const layers: VisualNetworkNode[][] = [];

  // Input nodes.
  const inputNodeIds = new Set(graph.io.inputNodeIds);
  const inputNodes = graph.nodes
    .filter((n) => inputNodeIds.has(n.id))
    .map((n) => ({
      index: n.id,
      type: 'input' as const,
      bias: n.bias ?? 0,
    }));
  if (inputNodes.length > 0) {
    layers.push(inputNodes);
  }

  // Hidden nodes.
  const outputNodeIds = new Set(graph.io.outputNodeIds);
  const hiddenNodes = graph.nodes
    .filter((n) => !inputNodeIds.has(n.id) && !outputNodeIds.has(n.id))
    .map((n) => ({
      index: n.id,
      type: 'hidden' as const,
      bias: n.bias ?? 0,
    }));
  if (hiddenNodes.length > 0) {
    layers.push(hiddenNodes);
  }

  // Output nodes.
  const outputNodes = graph.nodes
    .filter((n) => outputNodeIds.has(n.id))
    .map((n) => ({
      index: n.id,
      type: 'output' as const,
      bias: n.bias ?? 0,
    }));
  if (outputNodes.length > 0) {
    layers.push(outputNodes);
  }

  return layers;
}

/**
 * Draws the network visualization on a canvas context.
 *
 * @param context - Canvas 2D context.
 * @param frame - Resolved visualization frame.
 */
function drawNetworkVisualization(
  context: CanvasRenderingContext2D,
  frame: NetworkVisualizationResolvedFrame,
): void {
  // Clear canvas.
  context.fillStyle = '#1a1a1a';
  context.fillRect(0, 0, frame.canvasWidthPx, frame.canvasHeightPx);

  // Draw connections.
  drawConnections(context, frame);

  // Draw nodes.
  drawNodes(context, frame);
}

/**
 * Draws weighted connections between nodes.
 *
 * @param context - Canvas 2D context.
 * @param frame - Resolved frame.
 */
function drawConnections(
  context: CanvasRenderingContext2D,
  frame: NetworkVisualizationResolvedFrame,
): void {
  const positionByIndex = new Map(
    frame.positionedNodes.map((n) => [n.index, n]),
  );

  frame.connections.forEach((conn) => {
    const from = positionByIndex.get(conn.fromIndex);
    const to = positionByIndex.get(conn.toIndex);

    if (!from || !to) return;

    // Determine color based on weight sign.
    const color =
      conn.weight >= 0
        ? frame.colorScales.weightPositive
        : frame.colorScales.weightNegative;

    // Draw line.
    context.strokeStyle = color;
    context.lineWidth = Math.max(0.5, Math.abs(conn.weight) * 2);
    context.globalAlpha = Math.min(1, Math.abs(conn.weight));
    context.beginPath();
    context.moveTo(from.centerXPx, from.centerYPx);
    context.lineTo(to.centerXPx, to.centerYPx);
    context.stroke();
    context.globalAlpha = 1;
  });
}

/**
 * Draws nodes with type-specific shapes and styling.
 *
 * @param context - Canvas 2D context.
 * @param frame - Resolved frame.
 */
function drawNodes(
  context: CanvasRenderingContext2D,
  frame: NetworkVisualizationResolvedFrame,
): void {
  frame.positionedNodes.forEach((node) => {
    const radius = Math.min(node.widthPx, node.heightPx) * 0.5;

    // Draw node circle.
    context.fillStyle = '#333333';
    context.strokeStyle = '#00ccff';
    context.lineWidth = 2;
    context.beginPath();
    context.arc(node.centerXPx, node.centerYPx, radius, 0, Math.PI * 2);
    context.fill();
    context.stroke();

    // Draw bias indicator if bias is non-zero.
    if (Math.abs(node.bias) > 0.01) {
      context.fillStyle = frame.colorScales.bias;
      context.globalAlpha = Math.min(1, Math.abs(node.bias));
      context.beginPath();
      context.arc(
        node.centerXPx,
        node.centerYPx,
        radius * 0.4,
        0,
        Math.PI * 2,
      );
      context.fill();
      context.globalAlpha = 1;
    }
  });
}
