import type Network from '../../../src/architecture/network';

/**
 * Shared network-visualization contracts.
 *
 * These lightweight types let any example render evolved network topology
 * without depending on a specific demo. They describe the public shapes used
 * by the architecture panel so rendering logic can stay decoupled from the
 * full internal network implementation.
 *
 * The visualization subsystem is split into three responsibilities:
 * - Types (this file): the public shapes a consumer must provide.
 * - Network-view: topology resolution and node-positioning.
 * - Visualization: color scales, drawing, and legend helpers.
 *
 * ```mermaid
 * flowchart LR
 *   Host["Consumer (flappy / racing / asciiMaze)"] -->|"NetworkVisualizationHandle"| Types["network-visualization.types.ts"]
 *   Types -->|"VisualNetwork*Like"| View["network-view"]
 *   Types -->|"ColorScale*"| Viz["visualization"]
 *   View -->|"positioned scene"| Host
 *   Viz -->|"drawn legend + weights"| Host
 * ```
 */

/**
 * Browser-facing handle that lets a demo render the champion network panel.
 *
 * @example
 * ```ts
 * const visualizationHandle: NetworkVisualizationHandle = {
 *   renderNetworkArchitecture: (network, inputSize, outputSize) => {
 *     renderer.draw(network, inputSize, outputSize);
 *   },
 *   applyNetworkActivationOverlay: (network, activations) => {
 *     network.nodes.forEach((node, i) => {
 *       node.activation = activations[i] ?? node.activation;
 *     });
 *   },
 * };
 * visualizationHandle.renderNetworkArchitecture(championNetwork, 5, 2);
 * visualizationHandle.applyNetworkActivationOverlay(
 *   championNetwork,
 *   new Float32Array([0.2, -0.4, 0.9]),
 * );
 * ```
 */
export interface NetworkVisualizationHandle {
  /**
   * Renders the network architecture panel for a given network.
   *
   * The host calls this whenever the champion network changes so the panel
   * reflects the current topology. Passing `undefined` clears the panel.
   *
   * @param network - Network to visualize, or `undefined` to clear.
   * @param inputSize - Number of input nodes expected by the layout.
   * @param outputSize - Number of output nodes expected by the layout.
   * @returns Nothing.
   *
   * @example
   * ```ts
   * const handle: NetworkVisualizationHandle = {
   *   renderNetworkArchitecture: (network, inputSize, outputSize) => {
   *     renderer.draw(network, inputSize, outputSize);
   *   },
   *   applyNetworkActivationOverlay: (_network, _activations) => {},
   * };
   * handle.renderNetworkArchitecture(championNetwork, 5, 2);
   * ```
   */
  renderNetworkArchitecture: (
    network: Network | undefined,
    inputSize: number,
    outputSize: number,
  ) => void;
  /**
   * Paints streamed winner activations onto the currently visualized network.
   *
   * The overlay copies per-frame winner activations into the visualized
   * network's nodes and requests one coalesced repaint so activation labels
   * refresh. Stale payloads are rejected: a network that is not the visualized
   * instance, or a stream whose length no longer matches the node count,
   * leaves the panel untouched without scheduling a redraw.
   *
   * @param network - Network instance currently shown in the panel.
   * @param winnerNodeActivations - Streamed post-step activations per node.
   * @returns Nothing.
   *
   * @example
   * ```ts
   * const handle: NetworkVisualizationHandle = {
   *   renderNetworkArchitecture: (_network, _inputSize, _outputSize) => {},
   *   applyNetworkActivationOverlay: (network, activations) => {
   *     network.nodes.forEach((node, i) => {
   *       node.activation = activations[i] ?? node.activation;
   *     });
   *   },
   * };
   * handle.applyNetworkActivationOverlay(
   *   championNetwork,
   *   new Float32Array([0.2, -0.4, 0.9]),
   * );
   * ```
   */
  applyNetworkActivationOverlay: (
    network: Network,
    winnerNodeActivations: Float32Array,
  ) => void;
}

/**
 * Connection or bias tier used for color mapping ramps.
 *
 * Visualization buckets continuous weights into legible color bands so humans
 * can scan sign and magnitude at a glance.
 */
export interface ColorTier {
  upperBound: number;
  color: string;
}

/**
 * Legend row model for network visualization color legends.
 *
 * Each row labels a numeric interval and the color used to render it.
 */
export interface ColorLegendRow {
  label: string;
  color: string;
  minimumValue: number;
  maximumValue: number;
}

/**
 * Precomputed legend panel layout used by visualization renderer.
 *
 * Layout is resolved up front so the draw path can stay focused on painting,
 * not recomputing geometry every frame.
 */
export interface NetworkLegendLayout {
  compactLegend: boolean;
  legendLeftPx: number;
  legendTopPx: number;
  legendWidthPx: number;
  legendHeightPx: number;
  legendHeaderHeightPx: number;
  legendSectionTitleHeightPx: number;
  legendRowHeightPx: number;
  legendSectionGapPx: number;
}

/**
 * Lightweight connection shape used by network visualization drawing.
 *
 * The renderer only needs connectivity, weight, and enabled state, not the full
 * training-time behavior of a connection object.
 */
export interface VisualNetworkConnectionLike {
  from?: { index?: number };
  to?: { index?: number };
  weight?: number;
  enabled?: boolean;
}

/**
 * Lightweight node shape used by network visualization drawing.
 *
 * This shape keeps the renderer independent from the concrete Network class
 * while still exposing the semantic fields that matter visually.
 */
export interface VisualNetworkNodeLike {
  index: number;
  type: string;
  bias: number;
  activation?: number;
  geneId?: number;
  layer?: number;
}

/**
 * Positioned node instance used by network visualization drawing.
 *
 * Layout and rendering are split: first a node is assigned screen coordinates,
 * then the renderer paints it.
 */
export interface PositionedNetworkNodeLike {
  node: VisualNetworkNodeLike;
  xPx: number;
  yPx: number;
}

/**
 * Animated hovered-node intensity sample used during fade transitions.
 *
 * The host can keep several recent hover targets partially active at once so
 * quick pointer motion produces overlapping line-emphasis fades
 * instead of abrupt binary flicker.
 */
export interface NetworkVisualizationAnimatedHoveredNode {
  nodeIndex: number;
  intensity: number;
}

/**
 * Browser-owned hover state used by interactive network visualization passes.
 *
 * Hover is resolved from the canvas pointer and then passed into drawing as a
 * tiny UI-only contract. Supporting multiple node indices keeps direct node
 * hover and category combo-hover on the same rendering path, while animated
 * hover samples let the renderer fade highlights in and out.
 */
export interface NetworkVisualizationHoverState {
  hoveredNodeIndices?: readonly number[];
  animatedHoveredNodes?: readonly NetworkVisualizationAnimatedHoveredNode[];
}

/**
 * Tooltip scene model resolved from the hovered network overlay target.
 *
 * The host uses this shape to position and populate the educational tooltip
 * that appears when the pointer rests on an input node, hidden column, or
 * semantic input group band.
 */
export interface NetworkVisualizationTooltipScene {
  /** Tooltip category, matching the overlay target that produced it. */
  kind: 'group' | 'input' | 'column';
  /** Short heading shown at the top of the tooltip. */
  heading: string;
  /** Body paragraphs rendered inside the tooltip. */
  bodyParagraphs: readonly string[];
  /** Left edge of the tooltip anchor region in canvas coordinates. */
  anchorLeftPx: number;
  /** Top edge of the tooltip anchor region in canvas coordinates. */
  anchorTopPx: number;
  /** Width of the tooltip anchor region in canvas coordinates. */
  anchorWidthPx: number;
  /** Horizontal center of the tooltip anchor region in canvas coordinates. */
  anchorCenterXPx: number;
}

/**
 * Positioned input-group label band scene reused by drawing and hit testing.
 *
 * The host relies on this exact geometry when category hovers need to
 * highlight every node in a semantic input group.
 */
export interface NetworkInputGroupLabelBandScene {
  label: string;
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  backgroundColor: string;
  orientation: 'vertical' | 'horizontal';
  nodeIndices: number[];
}

/**
 * Positioned input-description row scene reused by drawing and hover hit testing.
 *
 * Each row maps one human-readable description to one input node so hovering
 * the text can emphasize the same node the description explains.
 */
export interface NetworkInputDescriptionScene {
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  nodeIndex: number;
}

/**
 * Positioned hidden-column label scene reused by drawing and future hit testing.
 *
 * Recurrent-aware layouts use these scenes to explain what one hidden column
 * means, for example an LSTM gate or a NARX delay shelf, without replacing the
 * underlying node bias encoding.
 */
export interface NetworkHiddenColumnLabelScene {
  labelLines: readonly string[];
  tooltipHeading: string;
  tooltipBodyParagraphs: readonly string[];
  leftPx: number;
  topPx: number;
  widthPx: number;
  heightPx: number;
  backgroundColor: string;
  nodeIndices: number[];
}

/**
 * Reusable positioned-node snapshot returned by the network-view draw path.
 *
 * The host reuses this exact layout snapshot for pointer hit testing so hover
 * logic can stay aligned with the scene that was actually rendered.
 */
export interface NetworkVisualizationPositionedScene {
  positionedNodes: PositionedNetworkNodeLike[];
  nodeDimensions: NetworkNodeDimensionsLike;
  inputGroupLabelBandScenes: NetworkInputGroupLabelBandScene[];
  inputDescriptionScenes: NetworkInputDescriptionScene[];
  hiddenColumnLabelScenes?: NetworkHiddenColumnLabelScene[];
}

/**
 * Pixel dimensions used for network-node rectangle rendering.
 *
 * Keeping node box dimensions explicit makes legend and topology layout easier
 * to tune without hidden drawing constants.
 */
export interface NetworkNodeDimensionsLike {
  widthPx: number;
  heightPx: number;
}

/**
 * Level-of-detail (LOD) settings for dense-network visualization.
 *
 * When a network's hidden-node count exceeds `hiddenNodeThreshold`, the shared
 * visualizer can replace the full-detail graph with a cheap abstraction: input
 * and output shelves stay fully rendered while hidden nodes collapse into a
 * few density clusters. Hovering a hidden node expands a deterministic 2-hop
 * ego neighborhood capped at `hoverMaxLocalNodes` local nodes.
 *
 * @example
 * ```ts
 * const settings: NetworkVisualizationSettings = {
 *   lod: {
 *     enabled: true,
 *     hiddenNodeThreshold: 2048,
 *     clusterCount: 4,
 *     hoverMaxLocalNodes: 64,
 *   },
 * };
 * ```
 */
export interface NetworkVisualizationLodSettings {
  /** Master switch; LOD stays enabled by default when omitted. */
  enabled?: boolean;
  /** Hidden-node count above which the LOD abstraction activates. */
  hiddenNodeThreshold?: number;
  /** Number of abstract hidden clusters rendered in the abstract scene. */
  clusterCount?: number;
  /** Maximum local nodes rendered when hovering a hidden node. */
  hoverMaxLocalNodes?: number;
}
