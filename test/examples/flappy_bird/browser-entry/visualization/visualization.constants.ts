/**
 * Pixel side length for dotted negative-connection square markers.
 */
export const FLAPPY_NETWORK_DOTTED_CONNECTION_SQUARE_SIDE_PX = 2;

/**
 * Stroke alpha used for enabled connection lines.
 */
export const FLAPPY_NETWORK_ENABLED_CONNECTION_ALPHA = 0.95;

/**
 * Stroke alpha used for disabled connection lines.
 */
export const FLAPPY_NETWORK_DISABLED_CONNECTION_ALPHA = 0.2;

/**
 * Dash pattern used for disabled positive connection lines.
 */
export const FLAPPY_NETWORK_DISABLED_CONNECTION_DASH_PATTERN = [5, 4] as const;

/**
 * Spacing multiplier used to tighten square-dot trail cadence.
 */
export const FLAPPY_NETWORK_DOTTED_CONNECTION_STEP_COMPACT_RATIO = 0.8;

/**
 * Line-width multiplier used to scale dotted-step spacing.
 */
export const FLAPPY_NETWORK_DOTTED_CONNECTION_WIDTH_SPACING_RATIO = 2.6;

/**
 * Tiny alignment epsilon to stabilize axis-aligned dot centering.
 */
export const FLAPPY_NETWORK_DOTTED_CONNECTION_ALIGNMENT_EPSILON = 0.01;

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
 * Glow blur radius used for output-node emphasis.
 */
export const FLAPPY_NETWORK_OUTPUT_NODE_SHADOW_BLUR_PX = 7;

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
export const FLAPPY_NETWORK_LEGEND_ARCHITECTURE_GAP_PX = 4;

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
