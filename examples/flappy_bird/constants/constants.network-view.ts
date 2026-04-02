/**
 * Network visualization layout and tier-mapping constants.
 *
 * These values define graph padding, node sizing, legend geometry, and tier
 * thresholds used to map connection and bias magnitudes to color ramps.
 */

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

/** Graph-left padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_LEFT_PADDING_PX = 3;

/** Graph-top padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_TOP_PADDING_PX = 0;

/** Graph-right padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_RIGHT_PADDING_PX = 20;

/** Graph-bottom padding for network visualization content. */
export const FLAPPY_NETWORK_GRAPH_BOTTOM_PADDING_PX = 18;

/** Minimum drawable graph dimension after padding is removed. */
export const FLAPPY_NETWORK_MIN_DRAWABLE_SIZE_PX = 1;

/** Gap between graph body and floating legend panel. */
export const FLAPPY_NETWORK_LEGEND_GRAPH_GAP_PX = 10;

/** Ratio used to decide whether the legend occupies the right half of the canvas. */
export const FLAPPY_NETWORK_LEGEND_RIGHT_SIDE_THRESHOLD_RATIO = 0.5;

/** Inner node-layout padding inside drawable network region. */
export const FLAPPY_NETWORK_NODE_LAYOUT_PADDING_PX = 2;

/** Minimum node label height in pixels. */
export const FLAPPY_NETWORK_MIN_LABEL_HEIGHT_PX = 6;

/** Minimum inner padding for node labels. */
export const FLAPPY_NETWORK_MIN_NODE_INNER_PADDING_PX = 1;

/** Preferred vertical spacing between input-layer nodes. */
export const FLAPPY_NETWORK_INPUT_LAYER_TARGET_GAP_PX = 5;

/** Fixed top margin for node stacks inside the network drawable area. */
export const FLAPPY_NETWORK_NODE_TOP_MARGIN_PX = 10;

/** Relative label-height ratio used when rendering node bias values. */
export const FLAPPY_NETWORK_NODE_LABEL_SIZE_RATIO = 1;

/** Font-weight used when rendering node bias values. */
export const FLAPPY_NETWORK_NODE_LABEL_FONT_WEIGHT = 700;

/** Minimum node width in network visualization. */
export const FLAPPY_NETWORK_MIN_NODE_WIDTH_PX = 28;

/** Maximum node height in network visualization. */
export const FLAPPY_NETWORK_MAX_NODE_HEIGHT_PX = 10;

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

/** Connection row stroke line width in legend. */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_LINE_WIDTH_PX = 1.8;

/** Legend font size for compact mode. */
export const FLAPPY_NETWORK_LEGEND_COMPACT_FONT_SIZE_PX = 9;

/** Legend font size for regular mode. */
export const FLAPPY_NETWORK_LEGEND_REGULAR_FONT_SIZE_PX = 10;

/** Header font size for architecture label lines. */
export const FLAPPY_NETWORK_HEADER_FONT_SIZE_PX = 11;

/** Extra pixel allowance above label baseline for minimum node-height readability. */
export const FLAPPY_NETWORK_MIN_NODE_HEIGHT_LABEL_EXTRA_PX = 4;

/** Minimum fit-based node height before width and label constraints are applied. */
export const FLAPPY_NETWORK_MIN_NODE_FIT_HEIGHT_PX = 4;

/**
 * Layer-fit divisor used when deriving max node height from dense layer stacks.
 *
 * Larger values keep nodes shorter in dense topologies so labels remain legible.
 */
export const FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_DIVISOR = 1.6;

/** Minimum denominator clamp for dense-layer node-height derivation. */
export const FLAPPY_NETWORK_NODE_HEIGHT_DENSITY_MIN_DENOMINATOR = 4;

/** Width-fit divisor used when deriving node height from layer count. */
export const FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_DIVISOR = 3.6;

/** Minimum denominator clamp for layer-width node-height derivation. */
export const FLAPPY_NETWORK_NODE_HEIGHT_LAYER_WIDTH_MIN_DENOMINATOR = 8;

/** Width-to-height ratio used for node rectangle proportions. */
export const FLAPPY_NETWORK_NODE_WIDTH_TO_HEIGHT_RATIO = 2.15;

/** Width-fit divisor used when deriving node width from layer count. */
export const FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_DIVISOR = 1.45;

/** Minimum denominator clamp for layer-width node-width derivation. */
export const FLAPPY_NETWORK_NODE_WIDTH_LAYER_WIDTH_MIN_DENOMINATOR = 4;

/** Baseline node-count threshold before node-density height increments apply. */
export const FLAPPY_NETWORK_NODE_DENSITY_BASELINE_COUNT = 6;

/** Baseline layer-count threshold before layer-complexity height increments apply. */
export const FLAPPY_NETWORK_LAYER_COMPLEXITY_BASELINE_COUNT = 3;

/** Additional multiplier used for topology-driven minimum host-height recommendation. */
export const FLAPPY_NETWORK_TOPOLOGY_HEIGHT_MULTIPLIER = 1.45;

/** Horizontal gap between input-node column and vertical group label band. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_GAP_PX = 6;

/** Width of the vertical input-group label band. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_BAND_WIDTH_PX = 24;

/** Minimum visual height for any input-group label band. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_MIN_HEIGHT_PX = 22;

/** Corner radius used by input-group label band backgrounds. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_RADIUS_PX = 5;

/** Font size used for vertical input-group label text. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_SIZE_PX = 9;

/** Font weight used for vertical input-group label text. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_FONT_WEIGHT = 700;

/** Text color for vertical input-group labels on neon backgrounds. */
export const FLAPPY_NETWORK_INPUT_GROUP_LABEL_TEXT_COLOR = '#000000';

/** Placeholder label used when the network has no hidden layers. */
export const FLAPPY_NETWORK_EMPTY_HIDDEN_LAYER_LABEL = '-';

/** Prefix used when hidden-layer counts are inferred rather than declared. */
export const FLAPPY_NETWORK_INFERRED_HIDDEN_LAYER_PREFIX = '~';

/** Separator used between hidden-layer sizes inside architecture labels. */
export const FLAPPY_NETWORK_HIDDEN_LAYER_SEPARATOR = ' - ';

/** Separator used between architecture columns in the compact header label. */
export const FLAPPY_NETWORK_ARCHITECTURE_COLUMN_SEPARATOR = ' | ';

/** Line separator used by the two-line architecture label block. */
export const FLAPPY_NETWORK_ARCHITECTURE_LINE_SEPARATOR = '\n';
