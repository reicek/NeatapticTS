/**
 * Browser UI layout and responsive-viewport constants.
 *
 * This module contains spacing, panel sizing, and breakpoint values used to
 * keep the demo readable from compact to wide viewports.
 */

/** Screen-edge padding between viewport border and demo frame. */
export const FLAPPY_SCREEN_PADDING_PX = 24;

/**
 * Horizontal viewport anchor for the bird.
 *
 * A value of `0.33` places the bird roughly one-third from the left, leaving
 * more lookahead space for incoming pipes.
 */
export const FLAPPY_BIRD_VIEWPORT_X_RATIO = 0.33;

/** Outer frame background color for the browser demo host. */
export const FLAPPY_UI_OUTER_FRAME_BACKGROUND = '#050a12';

/** Host panel background color behind the network visualization canvas. */
export const FLAPPY_UI_NETWORK_HOST_BACKGROUND = '#030812';

/** Inner canvas clear color for the network visualization area. */
export const FLAPPY_UI_NETWORK_CANVAS_BACKGROUND = '#02050c';

/** Shared inset glow shadow used by neon HUD panels. */
export const FLAPPY_UI_UNIFIED_INSET_SHADOW =
  'inset 0 0 0 1px rgba(15,181,255,0.16), 0 0 12px rgba(15,181,255,0.12)';

/** Shared double-line border style matching title-box aesthetics. */
export const FLAPPY_UI_DOUBLE_PANEL_BORDER = '3px double rgba(15,181,255,0.92)';

/** Thin inset shadow used on canvases for subtle neon depth. */
export const FLAPPY_UI_CANVAS_INSET_SHADOW =
  'inset 0 0 0 1px rgba(15,181,255,0.12)';

/** Vertical split guide color between stats and network panes. */
export const FLAPPY_UI_SPLIT_BRIDGE_COLOR = 'rgba(15,181,255,0.55)';

/** Side padding fallback for the outer frame when viewport is very narrow. */
export const FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX = 8;

/** Side padding offset applied to shared screen padding for frame layout. */
export const FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX = 10;

/** Top padding for the vertical content column hosting canvases and stats. */
export const FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX = 0;

/** Initial network panel height before topology-driven resizing runs. */
export const FLAPPY_UI_NETWORK_HOST_INITIAL_HEIGHT_PX = 120;

/** Inner host padding subtracted from measured network canvas dimensions. */
export const FLAPPY_UI_NETWORK_HOST_INSET_PX = 8;

/** Minimum stats panel height used by responsive viewport sizing. */
export const FLAPPY_VIEWPORT_MINIMUM_STATS_HEIGHT_PX = 500;

/** Minimum simulation canvas height floor in responsive viewport sizing. */
export const FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_PX = 60;

/** Minimum simulation height ratio relative to window height. */
export const FLAPPY_VIEWPORT_MINIMUM_SIMULATION_HEIGHT_RATIO = 0.08;

/** Vertical layout gutter between stats panel and simulation canvas. */
export const FLAPPY_VIEWPORT_VERTICAL_LAYOUT_GUTTER_PX = 8;

/** Additional bottom margin reserved during simulation canvas sizing. */
export const FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX = 0;

/** Minimum network panel height budget for responsive viewport sizing. */
export const FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX = 96;

/** Width breakpoint below which stats pane is hidden and network gets full width. */
export const FLAPPY_VIEWPORT_NETWORK_ONLY_BREAKPOINT_PX = 1_000;

/** Width breakpoint below which network legend and architecture text are hidden. */
export const FLAPPY_VIEWPORT_NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX = 800;

/**
 * Width breakpoint below which only title + main simulation canvas are shown.
 *
 * In this minimal mobile layout, stats and network-visualization panels are
 * hidden to maximize readable gameplay area.
 */
export const FLAPPY_VIEWPORT_MOBILE_MINIMAL_UI_BREAKPOINT_PX = 860;
