/**
 * Stroke alpha used for connection lines when no node is hovered.
 */
export const FLAPPY_NETWORK_DEFAULT_CONNECTION_ALPHA = 0.3;

/**
 * Stroke alpha used for non-adjacent connection lines while a node is hovered.
 */
export const FLAPPY_NETWORK_DIMMED_CONNECTION_ALPHA = 0.1;

/**
 * Stroke alpha used for directly adjacent connection lines while a node is hovered.
 */
export const FLAPPY_NETWORK_HIGHLIGHT_CONNECTION_ALPHA = 0.9;

/**
 * Duration used for network hover highlight fade-in and fade-out animation.
 */
export const FLAPPY_NETWORK_HOVER_TRANSITION_DURATION_MS = 50;

/**
 * Maximum stroke width used by the hovered node outline emphasis pass.
 */
export const FLAPPY_NETWORK_HOVERED_NODE_STROKE_WIDTH_PX = 2.6;

/**
 * Dash pattern used for disabled positive connection lines.
 */
export const FLAPPY_NETWORK_DISABLED_CONNECTION_DASH_PATTERN = [5, 4];

/**
 * Compact dash pattern used for negative connection lines.
 */
export const FLAPPY_NETWORK_NEGATIVE_CONNECTION_DASH_PATTERN = [2, 4];

/**
 * Extra vertical padding reserved for hidden-node labels inside rectangles.
 */
export const FLAPPY_NETWORK_HIDDEN_NODE_VERTICAL_PADDING_PX = 2;

/**
 * Stroke width used for emphasized output nodes.
 */
export const FLAPPY_NETWORK_OUTPUT_NODE_STROKE_WIDTH_PX = 2.1;

/**
 * Stroke width used for hidden and input nodes.
 */
export const FLAPPY_NETWORK_HIDDEN_NODE_STROKE_WIDTH_PX = 1.3;

/**
 * Minimum render height for any node rectangle.
 */
export const FLAPPY_NETWORK_MIN_RENDER_NODE_HEIGHT_PX = 4;

/**
 * Height reduction applied to output-node rectangles for tighter framing.
 */
export const FLAPPY_NETWORK_OUTPUT_NODE_HEIGHT_REDUCTION_PX = 2;

/**
 * Left and top padding used by the network header block.
 */
export const FLAPPY_NETWORK_HEADER_PADDING_PX = 8;

/**
 * Vertical spacing between multiline header rows.
 */
export const FLAPPY_NETWORK_HEADER_LINE_HEIGHT_PX = 12;

/**
 * Minimum top bound for architecture text above the legend box.
 */
export const FLAPPY_NETWORK_LEGEND_MIN_ARCHITECTURE_TOP_PX = 4;

/**
 * Gap between architecture text and the legend container.
 */
export const FLAPPY_NETWORK_LEGEND_ARCHITECTURE_GAP_PX = 8;

/**
 * Shared inner padding used by legend labels and swatches.
 */
export const FLAPPY_NETWORK_LEGEND_BOX_PADDING_PX = 8;

/**
 * Top offset for the legend title inside the legend box.
 */
export const FLAPPY_NETWORK_LEGEND_HEADER_TOP_PADDING_PX = 6;

/**
 * End x-position for connection sample lines in legend rows.
 */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_END_X_PX = 28;

/**
 * Vertical offset for connection sample lines inside legend rows.
 */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_SAMPLE_Y_OFFSET_PX = 4;

/**
 * Label x-position for connection legend row text.
 */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_LABEL_X_PX = 32;

/**
 * X-position for bias legend swatches.
 */
export const FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_X_PX = 10;

/**
 * Y-position offset for bias legend swatches.
 */
export const FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_Y_PX = 1;

/**
 * Side length for bias legend color swatches.
 */
export const FLAPPY_NETWORK_LEGEND_BIAS_SWATCH_SIZE_PX = 6;

/**
 * Label x-position for bias legend row text.
 */
export const FLAPPY_NETWORK_LEGEND_BIAS_LABEL_X_PX = 22;
