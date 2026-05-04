/**
 * Shared type contracts for the browser-based network canvas renderer.
 *
 * These types define the generic layout, positioning, and scene contracts
 * that any demo or external user can implement. Demo-specific overlays
 * (Flappy input bands, ASCII Maze labels, etc.) are injected as optional
 * callback hooks rather than baked into this core layer.
 */

/**
 * A node with its position and dimensions resolved in canvas coordinates.
 */
export interface PositionedNetworkNode {
  index: number;
  type: 'input' | 'hidden' | 'output';
  centerXPx: number;
  centerYPx: number;
  widthPx: number;
  heightPx: number;
  bias: number;
}

/**
 * A visual connection between two positioned nodes.
 */
export interface VisualNetworkConnection {
  fromIndex: number;
  toIndex: number;
  weight: number;
  enabled: boolean;
}

/**
 * Dimensions shared across all nodes in a rendering pass.
 */
export interface NetworkNodeDimensions {
  widthPx: number;
  heightPx: number;
}

/**
 * Padding on all four edges.
 */
export interface EdgePadding {
  topPx: number;
  rightPx: number;
  bottomPx: number;
  leftPx: number;
}

/**
 * Color scale for weight and activation visualization.
 */
export interface NetworkVisualizationColorScales {
  weightPositive: string;
  weightNegative: string;
  activationHot: string;
  activationCold: string;
  bias: string;
}

/**
 * Complete resolved frame for hover-driven incremental redraws.
 *
 * The host can cache this between pointer events so it only recomputes
 * topology, layout, and legend when the network payload changes.
 */
export interface NetworkVisualizationResolvedFrame {
  canvasWidthPx: number;
  canvasHeightPx: number;
  positionedNodes: PositionedNetworkNode[];
  connections: VisualNetworkConnection[];
  nodeDimensions: NetworkNodeDimensions;
  colorScales: NetworkVisualizationColorScales;
  topologyMode: 'acyclic' | 'recurrent';
}

/**
 * Optional hook functions that demos can use to inject custom overlays.
 *
 * Flappy Bird injects input-group label bands and per-input descriptions.
 * ASCII Maze could inject custom layer labels, or leave hooks undefined.
 */
export interface OverlayFactoryHooks {
  /**
   * Optional factory to create demo-specific overlay scenes.
   * For example, Flappy Bird creates input-group label bands here.
   */
  createDemoOverlayScenes?: (
    positionedNodes: PositionedNetworkNode[],
    nodeDimensions: NetworkNodeDimensions,
  ) => unknown[];
}

/**
 * Options passed to the shared renderer.
 */
export interface RenderNetworkViewOptions {
  /** Node dimensions (width × height px). */
  nodeDimensions?: NetworkNodeDimensions;
  /** Padding around the graph. */
  panelPaddingPx?: EdgePadding;
  /** Color scales for visualization. */
  colorScales?: NetworkVisualizationColorScales;
  /** Optional overlay factory hooks. */
  overlayFactory?: OverlayFactoryHooks;
}
