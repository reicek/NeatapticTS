import type {
  FlappyStatsKey,
  FlappyStatsRowDescriptor,
} from './browser-entry.types';
import {
  FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX as SHARED_FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX as SHARED_FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX,
  FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX as SHARED_FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX,
} from './constants';

/** Default host container id for the browser demo mount point. */
export const DEFAULT_CONTAINER_ID = 'flappy-bird-output';

/** Emulation speed multiplier for browser playback (1.5 => 50% faster). */
export const FLAPPY_EMULATION_SPEED_MULTIPLIER = 1.5;

/** Update HUD counters every N simulation frames to reduce DOM churn. */
export const FLAPPY_HUD_UPDATE_INTERVAL_FRAMES = 10;

/** Screen-edge padding between viewport border and demo frame. */
export const FLAPPY_SCREEN_PADDING_PX = 24;

/** TRON-like neon palette matching asciiMaze style. */
export const FLAPPY_NEON_PALETTE = {
  background: '#060b14',
  pipeFill: '#00ff66',
  pipeEdgeOuter: '#2dff78',
  pipeEdgeInner: '#bfffd4',
  // Champion bird red matching ANSI xterm color 196 (rgb(255, 0, 0)).
  championBird: '#ff0000',
  nonChampionBird: '#ffe94d',
  // Champion outline ring matching ANSI xterm color 231 (rgb(255, 255, 255)).
  leaderRing: '#ffffff',
  // Indigo trail matching ANSI xterm color 99 (rgb(135, 95, 255)).
  trail: '#875fff',
  currentRunText: '#00ff66',
  bestRunText: '#ff9a2e',
  statusText: '#ff5cff',
  hudText: '#9fdcff',
  hudAccent: '#ff9a2e',
  hudPanelBackground: '#000000',
  hudPanelBorder: '#0fb5ff',
} as const;

/** Ordered keys for the runtime stats table. */
export const FLAPPY_STATS_KEYS = [
  'currentHeader',
  'currentFrames',
  'currentPipes',
  'currentMaxFrames',
  'currentMaxPipes',
  'currentArchitecture',
  'telemetryHeader',
  'telemetryActivationsPerFrame',
  'telemetrySimulationStepsPerRaf',
  'telemetryHudUpdatesPerSecond',
  'telemetryMinorGcPerMinute',
  'bestHeader',
  'bestFrames',
  'bestPipes',
  'bestMaxFrames',
  'bestMaxPipes',
  'bestArchitecture',
  'status',
] as const;

/** Default population size for browser playback worker initialization. */
export const FLAPPY_BROWSER_POPULATION_SIZE = 50;

/**
 * Horizontal viewport anchor for the bird.
 *
 * This is used to keep the bird at a stable on-screen x-position while pipes
 * enter from the right and drift left.
 *
 * A value of `0.33` places the bird roughly one-third of the way from the left
 * edge of the visible world, leaving extra screen space to preview incoming
 * pipes while still showing a long trail behind.
 */
export const FLAPPY_BIRD_VIEWPORT_X_RATIO = 0.33;

/** Default elitism count for browser playback worker initialization. */
export const FLAPPY_BROWSER_ELITISM_COUNT = 20;

/** Reusable monospaced HUD font for glyph-based frame rendering. */
export const FLAPPY_FRAME_MONOSPACE_FONT =
  '16px Consolas, Menlo, Monaco, monospace';

/** Title rendered inside the standalone header box. */
export const FLAPPY_HEADER_TITLE_TEXT = ' Astro Bird (NeatapticTS) ';

/** Header canvas fixed height (pixels). */
export const FLAPPY_HEADER_CANVAS_HEIGHT_PX = 64;

/** Glyph-row height used for box-drawing rows (pixels). */
export const FLAPPY_FRAME_GLYPH_ROW_HEIGHT_PX = 16;

/** Minimum measured glyph width fallback (pixels). */
export const FLAPPY_FRAME_MIN_GLYPH_WIDTH_PX = 6;

/** Minimum glyph columns to keep frame readable on narrow widths. */
export const FLAPPY_FRAME_MIN_COLUMNS = 34;

/** Reserved columns to keep right edge in bounds during metrics fit. */
export const FLAPPY_FRAME_RESERVED_COLUMNS = 1;

/** Minimum box width when building standalone title frame. */
export const FLAPPY_TITLE_BOX_MIN_WIDTH = 16;

/** Total side margin columns reserved around centered title box. */
export const FLAPPY_TITLE_BOX_MARGIN_COLUMNS = 4;

/** Preferred title width relative to available frame width. */
export const FLAPPY_TITLE_BOX_WIDTH_RATIO = 0.68;

/** Minimum half-span when clamping centered title extents. */
export const FLAPPY_TITLE_BOX_MIN_HALF_SPAN = 2;

/** Minimum columns between title box and outer rails. */
export const FLAPPY_TITLE_BOX_MIN_OUTER_GAP_COLUMNS = 2;

/** Minimum safe columns for box-drawing helper output. */
export const FLAPPY_BOX_MIN_COLUMNS = 3;

/** Minimum safe rows for box-drawing helper output. */
export const FLAPPY_BOX_MIN_ROWS = 2;

/** Glyph character for top-left box corner. */
export const FLAPPY_GLYPH_TOP_LEFT = '╔';

/** Glyph character for top-right box corner. */
export const FLAPPY_GLYPH_TOP_RIGHT = '╗';

/** Glyph character for bottom-left box corner. */
export const FLAPPY_GLYPH_BOTTOM_LEFT = '╚';

/** Glyph character for bottom-right box corner. */
export const FLAPPY_GLYPH_BOTTOM_RIGHT = '╝';

/** Glyph character for horizontal box segment. */
export const FLAPPY_GLYPH_HORIZONTAL = '═';

/** Glyph character for vertical box segment. */
export const FLAPPY_GLYPH_VERTICAL = '║';

/** Glyph character for interior spacing. */
export const FLAPPY_GLYPH_SPACE = ' ';

/** Normalized decision threshold used for scalar output flap policies. */
export const FLAPPY_FLAP_THRESHOLD = 0.5;

/** Small epsilon divisor guard for world/physics normalization. */
export const FLAPPY_NORMALIZATION_EPSILON = 0.001;

/** Canonical half multiplier for centering and gap math. */
export const FLAPPY_HALF = 0.5;

/** Neon bird palette for per-agent render color assignment. */
export const FLAPPY_NEON_BIRD_PALETTE = [
  '#00e5ff',
  '#00ff66',
  '#ff9a2e',
  '#00b7ff',
  '#ff5cff',
  '#9fffff',
  '#a6ff00',
  '#ff4a8d',
] as const;

/** Shared HUD value for integer zero fields. */
export const FLAPPY_HUD_ZERO_TEXT = '0';

/** Shared HUD value for decimal zero fields. */
export const FLAPPY_HUD_ZERO_DECIMAL_TEXT = '0.00';

/** Shared HUD value when a metric is intentionally disabled. */
export const FLAPPY_HUD_OFF_TEXT = 'off';

/** Initial status text displayed before evolution starts. */
export const FLAPPY_HUD_INITIALIZING_TEXT = 'initializing';

/** Shared monospace font stack used by all HUD and visualization text. */
export const FLAPPY_MONOSPACE_FONT_FAMILY =
  'Consolas, Menlo, Monaco, monospace';

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

/** Border style for regular stats rows. */
export const FLAPPY_UI_STATS_ROW_BORDER = '1px double rgba(15,181,255,0.3)';

/** Border style for stats section headers. */
export const FLAPPY_UI_STATS_SECTION_BORDER = '2px double rgba(15,181,255,0.4)';

/** Regular neon ramp used for strong positive/baseline scales. */
export const FLAPPY_REGULAR_NEON_RAMP = [
  '#00ff9d',
  '#00ff66',
  '#7dff33',
  '#ccff00',
  '#ffe100',
  '#ffb400',
  '#ff8600',
  '#ff5a00',
  '#ff3300',
  '#ff1a1a',
] as const;

/** Light neon ramp used for high-contrast negative scales. */
export const FLAPPY_LIGHT_NEON_RAMP = [
  '#7dffd2',
  '#8dffb7',
  '#b8ff8a',
  '#ddff8a',
  '#fff38a',
  '#ffd98a',
  '#ffc18a',
  '#ffa98a',
  '#ff9696',
  '#ff8383',
] as const;

/** Neutral-center blue ramp for near-zero diverging tiers. */
export const FLAPPY_CENTER_BLUE_RAMP = ['#0091ff', '#5ad1ff'] as const;

/** Connection-tier max absolute magnitude used by diverging color tiers. */
export const FLAPPY_CONNECTION_TIER_MAX_ABS_VALUE = 1.5;

/** Connection-tier center threshold around zero. */
export const FLAPPY_CONNECTION_TIER_CENTER_THRESHOLD = 0.1;

/** Connection-tier edge region start magnitude. */
export const FLAPPY_CONNECTION_TIER_EDGE_START_ABS_VALUE = 1.0;

/** Bias-tier max absolute magnitude used by diverging color tiers. */
export const FLAPPY_BIAS_TIER_MAX_ABS_VALUE = 0.75;

/** Bias-tier center threshold around zero. */
export const FLAPPY_BIAS_TIER_CENTER_THRESHOLD = 0.06;

/** Bias-tier edge region start magnitude. */
export const FLAPPY_BIAS_TIER_EDGE_START_ABS_VALUE = 0.5;

/** Shared logarithmic steepness for diverging color tier mapping. */
export const FLAPPY_TIER_LOGARITHMIC_STEEPNESS = 72;

/** Number of linearly distributed edge tiers near maximum magnitudes. */
export const FLAPPY_TIER_EDGE_COUNT = 3;

/** Fallback color when a value exceeds all configured tiers. */
export const FLAPPY_TIER_ABOVE_COLOR = '#ff4a4a';

/** Graph-left padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX = 20;

/** Graph-top padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX = 34;

/** Graph-right padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX = 20;

/** Graph-bottom padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX = 18;

/** Gap between graph body and floating legend panel. */
export const FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX = 10;

/** Inner node-layout padding inside drawable network region. */
export const FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX = 4;

/** Minimum node label height in pixels. */
export const FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX = 10;

/** Minimum inner padding for node labels. */
export const FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX = 4;

/** Relative label-height ratio used when rendering node bias values. */
export const FLAPPY_NETWORK_NODE_LABEL_SIZE_RATIO = 0.72;

/** Font-weight used when rendering node bias values. */
export const FLAPPY_NETWORK_NODE_LABEL_FONT_WEIGHT = 700;

/** Minimum node width in network visualization. */
export const FLAPPY_NETWORK_MIN_NODE_WIDTH_PX = 28;

/** Maximum node height in network visualization. */
export const FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX = 14;

/** Maximum node width in network visualization. */
export const FLAPPY_NETWORK_MAX_NODE_WIDTH_PX = 36;

/** Minimum inter-node vertical gap for topology-driven sizing. */
export const FLAPPY_NETWORK_MIN_INTER_NODE_GAP_PX = 5;

/** Extra inner graph padding for topology-driven sizing. */
export const FLAPPY_NETWORK_GRAPH_INNER_PADDING_PX = 8;

/** Baseline network panel height before complexity adjustments. */
export const FLAPPY_NETWORK_BASELINE_HEIGHT_PX = 132;

/** Additional height increment per node above baseline density. */
export const FLAPPY_NETWORK_NODE_DENSITY_HEIGHT_STEP_PX = 9;

/** Additional height increment per extra layer above baseline. */
export const FLAPPY_NETWORK_LAYER_COMPLEXITY_HEIGHT_STEP_PX = 12;

/** Minimum clamped network panel height. */
export const FLAPPY_NETWORK_MIN_HEIGHT_PX = 180;

/** Maximum clamped network panel height. */
export const FLAPPY_NETWORK_MAX_HEIGHT_PX = 900;

/** Compact legend width threshold for network visualization. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_THRESHOLD_PX = 500;

/** Compact legend height threshold for network visualization. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_HEIGHT_THRESHOLD_PX = 250;

/** Compact legend panel width. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_WIDTH_PX = 212;

/** Regular legend panel width. */
export const FLAPPY_NETWORK_LEGEND_REGULAR_WIDTH_PX = 248;

/** Legend header row height. */
export const FLAPPY_NETWORK_LEGEND_HEADER_HEIGHT_PX = 18;

/** Legend section title height in compact mode. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_TITLE_HEIGHT_PX = 11;

/** Legend section title height in regular mode. */
export const FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_TITLE_HEIGHT_PX = 13;

/** Legend row height in compact mode. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_ROW_HEIGHT_PX = 9;

/** Legend row height in regular mode. */
export const FLAPPY_NETWORK_LEGEND_REGULAR_ROW_HEIGHT_PX = 10;

/** Legend section gap in compact mode. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_SECTION_GAP_PX = 6;

/** Legend section gap in regular mode. */
export const FLAPPY_NETWORK_LEGEND_REGULAR_SECTION_GAP_PX = 8;

/** Legend bottom padding in compact/regular layouts. */
export const FLAPPY_NETWORK_LEGEND_BOTTOM_PADDING_PX = 8;

/** Legend panel outer margin. */
export const FLAPPY_NETWORK_LEGEND_MARGIN_PX = 8;

/** Canvas width threshold to prefer top-left legend placement. */
export const FLAPPY_NETWORK_LEGEND_TOP_LEFT_THRESHOLD_PX = 420;

/** Target top offset used for legend placement. */
export const FLAPPY_NETWORK_LEGEND_TARGET_TOP_PX = 40;

/** Minimum top clamp for legend placement. */
export const FLAPPY_NETWORK_LEGEND_MIN_TOP_PX = 34;

/** Legend panel background fill color. */
export const FLAPPY_NETWORK_LEGEND_BACKGROUND = 'rgba(0, 0, 0, 0.72)';

/** Legend header color. */
export const FLAPPY_NETWORK_LEGEND_HEADER_COLOR =
  FLAPPY_NEON_PALETTE.statusText;

/** Legend section title color for connection weight rows. */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR = '#00e5ff';

/** Legend section title color for node bias rows. */
export const FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR = '#00ff66';

/** Legend row text color. */
export const FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR = '#9fdcff';

/** Connection row stroke line width in legend. */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX = 1.8;

/** Legend font size for compact mode. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX = 9;

/** Legend font size for regular mode. */
export const FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX = 10;

/** Header text color for architecture label in visualization. */
export const FLAPPY_NETWORK_HEADER_TEXT_COLOR = '#9fdcff';

/** Header font size for architecture label lines. */
export const FLAPPY_NETWORK_HEADER_FONT_SIZE_PX = 11;

/** Dark fill color used for node bias labels. */
export const FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR = '#001522';

/** Output-node stroke color. */
export const FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR = '#d8ffe9';

/** Hidden-node stroke color. */
export const FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR = '#9fdcff';

/** Output-node glow color. */
export const FLAPPY_NETWORK_OUTPUT_NODE_GLOW_COLOR = 'rgba(0, 255, 102, 0.65)';

/** Ordered row descriptors rendered into the runtime stats table. */
export const FLAPPY_STATS_ROWS: readonly FlappyStatsRowDescriptor[] = [
  { key: 'currentHeader', label: 'Current run' },
  { key: 'currentFrames', label: 'Frames' },
  { key: 'currentPipes', label: 'Pipes' },
  { key: 'currentMaxFrames', label: 'Max frames' },
  { key: 'currentMaxPipes', label: 'Max pipes' },
  { key: 'currentArchitecture', label: 'NN architecture' },
  { key: 'telemetryHeader', label: 'Instrumentation' },
  { key: 'telemetryActivationsPerFrame', label: 'Act/frame' },
  { key: 'telemetrySimulationStepsPerRaf', label: 'Steps/RAF' },
  { key: 'telemetryHudUpdatesPerSecond', label: 'HUD upd/s' },
  { key: 'telemetryMinorGcPerMinute', label: 'Minor GC/min' },
  { key: 'bestHeader', label: 'Best run' },
  { key: 'bestFrames', label: 'Frames' },
  { key: 'bestPipes', label: 'Pipes' },
  { key: 'bestMaxFrames', label: 'Max frames' },
  { key: 'bestMaxPipes', label: 'Max pipes' },
  { key: 'bestArchitecture', label: 'NN architecture' },
  { key: 'status', label: 'Status' },
] as const;

/** Stats keys that are hidden when runtime instrumentation is disabled. */
export const FLAPPY_INSTRUMENTATION_STATS_KEYS: readonly FlappyStatsKey[] = [
  'telemetryHeader',
  'telemetryActivationsPerFrame',
  'telemetrySimulationStepsPerRaf',
  'telemetryHudUpdatesPerSecond',
  'telemetryMinorGcPerMinute',
] as const;

/** Stats keys rendered as section headers rather than key/value rows. */
export const FLAPPY_STATS_SECTION_KEYS: readonly FlappyStatsKey[] = [
  'currentHeader',
  'telemetryHeader',
  'bestHeader',
] as const;

/** Stats keys whose values should render multi-line architecture content. */
export const FLAPPY_STATS_ARCHITECTURE_KEYS: readonly FlappyStatsKey[] = [
  'currentArchitecture',
  'bestArchitecture',
] as const;

/** Side padding fallback for the outer frame when viewport is very narrow. */
export const FLAPPY_UI_OUTER_FRAME_MIN_SIDE_PADDING_PX = 8;

/** Side padding offset applied to shared screen padding for frame layout. */
export const FLAPPY_UI_OUTER_FRAME_SIDE_PADDING_OFFSET_PX = 10;

/** Top padding for the vertical content column hosting canvases and stats. */
export const FLAPPY_UI_CONTENT_COLUMN_TOP_PADDING_PX = 16;

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
export const FLAPPY_VIEWPORT_SIMULATION_BOTTOM_MARGIN_PX = 16;

/** Minimum network panel height budget for responsive viewport sizing. */
export const FLAPPY_VIEWPORT_MIN_NETWORK_HEIGHT_BUDGET_PX = 96;

/** HUD sliding-window size used when computing updates-per-second metric. */
export const FLAPPY_HUD_UPDATES_WINDOW_MS = 10_000;

/** HUD updates window duration in seconds for per-second conversion. */
export const FLAPPY_HUD_UPDATES_WINDOW_SECONDS = 10;

/** Sliding-window size used when computing minor GC events per minute. */
export const FLAPPY_MINOR_GC_WINDOW_MS = 60_000;

/** Runtime status text shown while a generation playback is running. */
export const FLAPPY_STATUS_PLAYING_TEXT = 'playing';

/** Runtime status text shown between playback episodes. */
export const FLAPPY_STATUS_EVOLVING_TEXT = 'evolving';

/** Maximum number of trail points retained per bird trail polyline. */
export const FLAPPY_TRAIL_MAX_POINTS = 40;

/** Stroke width used for per-bird trail line rendering. */
export const FLAPPY_TRAIL_LINE_WIDTH_PX = 1.5;

/** Distance from world edge over which trails fade to transparent (pixels). */
export const FLAPPY_TRAIL_EDGE_FADE_DISTANCE_PX = 48;

/** Opacity used for non-champion bird fills and trails. */
export const FLAPPY_NON_CHAMPION_OPACITY = 0.1;

/** Ratio of bird side length used for top-left shine square. */
export const FLAPPY_BIRD_SHINE_SIZE_RATIO = 0.42;

/** Ratio of bird side length used to inset shine from top-left corner. */
export const FLAPPY_BIRD_SHINE_INSET_RATIO = 0.2;

/** Fill style used for the bird shine highlight. */
export const FLAPPY_BIRD_SHINE_FILL_STYLE = 'rgba(255, 255, 255, 0.7)';

/** Fill style used for the champion bird shine highlight (xterm 196 red). */
export const FLAPPY_BIRD_CHAMPION_SHINE_FILL_STYLE = 'rgba(255, 0, 0, 0.7)';

/** Neon glow color used for the condensed white highlight shine. */
export const FLAPPY_BIRD_WHITE_SHINE_GLOW_COLOR = '#ffffff';

/** Neon glow color used for the champion shine highlight (xterm 196 red). */
export const FLAPPY_BIRD_CHAMPION_SHINE_GLOW_COLOR = '#ff0000';

/** Neon blur radius used for the condensed white highlight shine. */
export const FLAPPY_BIRD_WHITE_SHINE_GLOW_BLUR_PX = 2;

/** Base neon blur radius used for bird body glow. */
export const FLAPPY_BIRD_BODY_GLOW_BLUR_PX = 10;

/**
 * Opacity used for the extra Radiant-style aura around each bird.
 *
 * This is intentionally subtle: it should read as a soft bloom that lifts the
 * bird off the background, without turning the bird into a big glowing blob.
 */
export const FLAPPY_BIRD_AURA_ALPHA = 0.16;

/** Pixel expansion used for the Radiant-style bird aura plate. */
export const FLAPPY_BIRD_AURA_EXPAND_PX = 3;

/** Blur multiplier used for the Radiant-style bird aura plate. */
export const FLAPPY_BIRD_AURA_BLUR_MULTIPLIER = 2.6;

/** Additional blur radius applied to the champion red body glow. */
export const FLAPPY_BIRD_CHAMPION_EXTRA_GLOW_BLUR_PX = 7;

/** Opacity used for the expanded champion red glow plate. */
export const FLAPPY_BIRD_CHAMPION_RED_GLOW_ALPHA = 0.62;

/** Pixel expansion used for the champion red glow plate. */
export const FLAPPY_BIRD_CHAMPION_RED_GLOW_EXPAND_PX = 4;

/** Minimum horizontal segment length used by stepped trail rendering. */
export const FLAPPY_TRAIL_MIN_HORIZONTAL_SEGMENT_PX = 2;

/** Minimum vertical segment length used by stepped trail rendering. */
export const FLAPPY_TRAIL_MIN_VERTICAL_SEGMENT_PX = 2;

/** Additional radius applied when drawing champion leader ring. */
export const FLAPPY_LEADER_RING_RADIUS_OFFSET_PX = 2;

/** Stroke width used when drawing champion leader ring. */
export const FLAPPY_LEADER_RING_LINE_WIDTH_PX = 2;

/** Neon blur radius used when drawing the champion outline stroke. */
export const FLAPPY_LEADER_RING_GLOW_BLUR_PX = 10;

/** Visual gap between the pipe body and its outline on the sides (pixels). */
export const FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX =
  SHARED_FLAPPY_PIPE_OUTLINE_SIDE_GAP_PX;

/** Visual gap between the pipe body and its outline at the pipe entrance rim (pixels). */
export const FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX =
  SHARED_FLAPPY_PIPE_OUTLINE_ENTRANCE_GAP_PX;

/** Stroke width used for the pipe outline (pixels). */
export const FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX =
  SHARED_FLAPPY_PIPE_OUTLINE_STROKE_WIDTH_PX;

/** Opacity used for the soft pipe glow stroke pass. */
export const FLAPPY_PIPE_OUTLINE_GLOW_ALPHA = 0.45;

/** Stroke width used for the soft pipe glow stroke pass (pixels). */
export const FLAPPY_PIPE_OUTLINE_GLOW_STROKE_WIDTH_PX = 8;

/**
 * Cyan neon glow used for pipe outline shadow.
 *
 * This intentionally matches the asciiMaze `neonCyan` ANSI color (`\x1b[38;5;87m`)
 * which maps to xterm color 87 ~= rgb(95, 255, 255) / hex `#5fffff`.
 */
export const FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR = 'rgba(95, 255, 255, 0.95)';

/** Blur radius used for the cyan pipe outline glow (pixels). */
export const FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX = Math.max(
  24,
  Math.round(FLAPPY_BIRD_BODY_GLOW_BLUR_PX * 3.2),
);
