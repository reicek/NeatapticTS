/**
 * Shared type contracts for the browser-based network canvas renderer.
 *
 * These types define the generic layout, positioning, and scene contracts
 * that any demo or external user can implement. Demo-specific overlays
 * (Flappy input bands, ASCII Maze labels, etc.) are injected as optional
 * callback hooks rather than baked into this core layer.
 */

/**
 * A network node with its center position and pixel dimensions resolved in canvas space, ready for hit-testing and rendering passes.
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
 * A visual edge between two positioned network nodes, carrying the synapse weight and enabled state for color-coded rendering.
 */
export interface VisualNetworkConnection {
  fromIndex: number;
  toIndex: number;
  weight: number;
  enabled: boolean;
}

/**
 * Pixel dimensions shared by every node in a single rendering pass, controlling both the visual node size and the hit-test bounding box for hover interactions.
 */
export interface NetworkNodeDimensions {
  widthPx: number;
  heightPx: number;
}

/**
 * Per-edge pixel padding that defines the inset drawable area inside the canvas, providing margins for graph layout and label overflow.
 */
export interface EdgePadding {
  topPx: number;
  rightPx: number;
  bottomPx: number;
  leftPx: number;
}

/**
 * Color palette strings used to encode positive and negative weights, hot and cold activations, and bias magnitudes during canvas rendering passes.
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
   * Optional factory that creates demo-specific overlay scenes drawn on top of the base network graph
   * after each rendering pass; for example, Flappy Bird uses this hook to add input-group label bands.
   */
  createDemoOverlayScenes?: (
    positionedNodes: PositionedNetworkNode[],
    nodeDimensions: NetworkNodeDimensions,
  ) => unknown[];
}

/**
 * Configuration bag passed to the shared canvas renderer to override default node dimensions, padding, color scales, and optional demo overlay hooks.
 */
export interface RenderNetworkViewOptions {
  /** Pixel dimensions applied uniformly to every node in this render pass, controlling visual node size and hover hit-test bounding box area. */
  nodeDimensions?: NetworkNodeDimensions;
  /** Inset pixel margins defining the drawable area boundary inside the canvas, providing layout spacing and preventing label overflow at edges. */
  panelPaddingPx?: EdgePadding;
  /** Optional color palette overrides for weight polarity, activation temperature, and bias magnitude applied during canvas rendering passes. */
  colorScales?: NetworkVisualizationColorScales;
  /** Optional hook set injected by demos to add custom overlay scenes rendered on top of the base network graph after each pass. */
  overlayFactory?: OverlayFactoryHooks;
}
