/**
 * DOM elements that participate in responsive host sizing.
 */
export type ResponsiveViewportSizingElements = {
  canvas: HTMLCanvasElement;
  containerElement: HTMLElement;
  mainSplitContainer: HTMLElement;
  statsContainer: HTMLElement;
  statsSplitContainer: HTMLElement;
  statsTableHost: HTMLElement;
  networkCanvas: HTMLCanvasElement;
  networkCanvasHost: HTMLElement;
};

/**
 * Deferred redraw controller used after layout-affecting changes.
 */
export type DeferredNetworkRedrawController = {
  requestNetworkRedrawAfterLayout: () => void;
};

/**
 * Measured viewport and panel budgets for responsive layout.
 */
export type ResponsiveViewportMeasurements = {
  headerHeightPx: number;
  nonNetworkStatsHeightPx: number;
  hardMinimumSimulationHeightPx: number;
  totalCanvasBudgetPx: number;
  viewportWidthPx: number;
  viewportHeightPx: number;
};

/**
 * Layout-mode flags resolved from the current viewport.
 */
export type ResponsiveViewportLayoutFlags = {
  useMinimalMobileLayout: boolean;
  useLandscapeSplitLayout: boolean;
  useNetworkOnlyPanel: boolean;
};

/**
 * Full responsive layout context shared across sizing helpers.
 */
export type ResponsiveViewportLayoutContext = ResponsiveViewportMeasurements &
  ResponsiveViewportLayoutFlags;

/**
 * Resolved height and width budgets for the stats panel.
 */
export type StatsPanelDimensions = {
  adjustedStatsPanelHeightPx: number;
  resolvedStatsPanelWidthPx: number;
};

/**
 * Width and height bounds for the simulation canvas.
 */
export type SimulationCanvasBounds = {
  availableWidthPx: number;
  availableHeightPx: number;
};
