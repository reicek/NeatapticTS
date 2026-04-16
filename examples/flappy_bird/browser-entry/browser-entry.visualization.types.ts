import type Network from '../../../src/architecture/network';

/**
 * Network-visualization contracts for the Flappy Bird browser demo.
 *
 * One of the educational goals of the example is to let people watch evolved
 * controllers as structures, not just as scores. These types describe the
 * lightweight shapes used by the architecture panel so rendering logic can stay
 * decoupled from the full internal network implementation.
 */

/** Draw callback contract for network architecture panel updates. */
export interface NetworkVisualizationHandle {
  renderNetworkArchitecture: (
    network: Network | undefined,
    inputSize: number,
    outputSize: number,
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
