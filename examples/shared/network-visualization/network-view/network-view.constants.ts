/**
 * Value-identical NETWORK_* defaults for the shared network visualizer.
 *
 * These constants back the optional fields of the
 * {@link NetworkVisualizationSettings} settings bag so hosts that omit
 * settings get the exact rendering that shipped before settings existed.
 * Values are copied as literals rather than re-exported from the
 * demo-branded constants module so the shared visualizer owns its defaults.
 */

/**
 * Default canvas background fill for the network visualization panel.
 *
 * @example
 * ```ts
 * const background = NETWORK_UI_CANVAS_BACKGROUND; // '#02050c'
 * ```
 */
export const NETWORK_UI_CANVAS_BACKGROUND = '#02050c';

/**
 * Default font family used for all network overlay text.
 *
 * @example
 * ```ts
 * const font = NETWORK_FONT_FAMILY; // 'Consolas, Menlo, Monaco, monospace'
 * ```
 */
export const NETWORK_FONT_FAMILY = 'Consolas, Menlo, Monaco, monospace';

/**
 * Default light neon ramp used for topology heatmaps.
 *
 * @example
 * ```ts
 * const heatmapColor = NETWORK_LIGHT_NEON_RAMP[0]; // '#7dffd2'
 * ```
 */
export const NETWORK_LIGHT_NEON_RAMP = [
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

/**
 * Default regular neon ramp used for strong positive/baseline scales.
 *
 * @example
 * ```ts
 * const strongPositiveColor = NETWORK_REGULAR_NEON_RAMP[2]; // '#7dff33'
 * ```
 */
export const NETWORK_REGULAR_NEON_RAMP = [
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

/**
 * Default neutral-center blue ramp used for near-zero diverging tiers.
 *
 * @example
 * ```ts
 * const nearZeroColor = NETWORK_CENTER_BLUE_RAMP[1]; // '#5ad1ff'
 * ```
 */
export const NETWORK_CENTER_BLUE_RAMP = ['#0091ff', '#5ad1ff'] as const;

/**
 * Default viewport width (px) below which the overlay is hidden.
 *
 * @example
 * ```ts
 * const hideBelow = NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX; // 800
 * ```
 */
export const NETWORK_OVERLAY_HIDDEN_BREAKPOINT_PX = 800;

/**
 * Default CSS transition duration (ms) for hover highlights.
 *
 * @example
 * ```ts
 * const fadeMs = NETWORK_HOVER_TRANSITION_DURATION_MS; // 50
 * ```
 */
export const NETWORK_HOVER_TRANSITION_DURATION_MS = 50;

/**
 * Default neon theme palette for the network visualization panel.
 *
 * Value-identical copy of the demo neon palette; only the fields declared on
 * {@link NetworkVisualizationPalette} are consumed today, and the full copy
 * keeps future palette threading value-identical.
 *
 * @example
 * ```ts
 * const currentRunColor = NETWORK_NEON_PALETTE.currentRunText;
 * ```
 */
export const NETWORK_NEON_PALETTE = {
  background: '#060b14',
  pipeFill: '#00ff66',
  pipeEdgeOuter: '#2dff78',
  pipeEdgeInner: '#bfffd4',
  championBird: '#ff4a8d',
  nonChampionBird: '#00e5ff',
  leaderRing: '#ffffff',
  trail: '#875fff',
  currentRunText: '#00ff66',
  bestRunText: '#ff9a2e',
  statusText: '#ff5cff',
  hudText: '#9fdcff',
  hudAccent: '#ff9a2e',
  hudPanelBackground: '#000000',
  hudPanelBorder: '#0fb5ff',
  horizonLine: '#0a8ea0',
  horizonGlow: 'rgba(10, 142, 160, 0.95)',
  groundGridLine: '#0a8ea0',
  groundGridGlow: 'rgba(10, 142, 160, 0.9)',
  groundGridFog: 'rgba(10, 142, 160, 0.55)',
  groundGridPulseFill: '#fff14a',
  groundGridPulseGlow: 'rgba(255, 241, 74, 0.92)',
} as const;

/** Default header/architecture label text color. */
export const NETWORK_HEADER_TEXT_COLOR = '#9fdcff';

/** Default fill color for node activation labels. */
export const NETWORK_NODE_LABEL_FILL_COLOR = '#001522';

/** Default stroke color for hidden and input nodes. */
export const NETWORK_HIDDEN_NODE_STROKE_COLOR = '#9fdcff';

/** Default stroke color for output nodes. */
export const NETWORK_OUTPUT_NODE_STROKE_COLOR = '#d8ffe9';

/** Default fill color for output nodes. */
export const NETWORK_OUTPUT_NODE_FILL_COLOR = NETWORK_NEON_PALETTE.currentRunText;

/** Default background fill for the color legend panel. */
export const NETWORK_LEGEND_BACKGROUND = 'rgba(0, 0, 0, 0.72)';

/** Default title color for the legend panel. */
export const NETWORK_LEGEND_HEADER_COLOR = NETWORK_NEON_PALETTE.statusText;

/** Default legend panel frame stroke color. */
export const NETWORK_LEGEND_STROKE_COLOR = NETWORK_NEON_PALETTE.statusText;

/** Default section title color for connection-weight legend rows. */
export const NETWORK_LEGEND_CONNECTION_TITLE_COLOR = '#00e5ff';

/** Default section title color for node-bias legend rows. */
export const NETWORK_LEGEND_BIAS_TITLE_COLOR = '#00ff66';

/** Default text color for legend row labels and swatch descriptions. */
export const NETWORK_LEGEND_ROW_TEXT_COLOR = '#9fdcff';
