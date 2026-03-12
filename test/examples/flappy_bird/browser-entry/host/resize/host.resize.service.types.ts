/**
 * Shared contracts for responsive browser-host sizing.
 *
 * The resize subsystem coordinates several DOM elements and several layout modes
 * at once, so these types capture the measured viewport state, the active mode
 * flags, and the grouped DOM handles needed by the appliers.
 */

/**
 * DOM elements that participate in responsive host sizing.
 *
 * Keeping the participating elements together makes the responsive helpers read
 * as layout orchestration rather than long DOM parameter lists.
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
 *
 * The network panel redraw is deferred because flexbox and canvas sizing can
 * settle over multiple animation frames.
 */
export type DeferredNetworkRedrawController = {
  requestNetworkRedrawAfterLayout: () => void;
};

/**
 * Measured viewport and panel budgets for responsive layout.
 *
 * These values are the raw numeric inputs to the resize policy before any mode
 * branching happens.
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
 *
 * The host currently distinguishes between minimal mobile, landscape split, and
 * compact network-only variants.
 */
export type ResponsiveViewportLayoutFlags = {
  useMinimalMobileLayout: boolean;
  useLandscapeSplitLayout: boolean;
  useNetworkOnlyPanel: boolean;
};

/**
 * Full responsive layout context shared across sizing helpers.
 *
 * This merges measurements and mode flags into the one context object used by
 * the resize appliers.
 */
export type ResponsiveViewportLayoutContext = ResponsiveViewportMeasurements &
  ResponsiveViewportLayoutFlags;

/**
 * Resolved height and width budgets for the stats panel.
 *
 * The stats panel dimensions are derived from the current viewport budget after
 * minimum readable canvas space is reserved.
 */
export type StatsPanelDimensions = {
  adjustedStatsPanelHeightPx: number;
  resolvedStatsPanelWidthPx: number;
};

/**
 * Width and height bounds for the simulation canvas.
 *
 * These are the final pixel budgets the simulation canvas may consume for the
 * current responsive layout.
 */
export type SimulationCanvasBounds = {
  availableWidthPx: number;
  availableHeightPx: number;
};
