/**
 * Browser visual palette constants for the Flappy demo.
 *
 * This module centralizes theme colors and diverging ramps so visual tuning is
 * easy to find and consistent across HUD, birds, pipes, and network overlays.
 */

/** TRON-like neon palette matching asciiMaze style. */
export const FLAPPY_NEON_PALETTE = {
  background: '#060b14',
  pipeFill: '#00ff66',
  pipeEdgeOuter: '#2dff78',
  pipeEdgeInner: '#bfffd4',
  championBird: '#ff0000',
  nonChampionBird: '#ffe94d',
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
  groundGridPulseFill: '#8ef3ff',
  groundGridPulseGlow: 'rgba(142, 243, 255, 0.92)',
} as const;

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

/** Fallback color when a value exceeds all configured tiers. */
export const FLAPPY_TIER_ABOVE_COLOR = '#ff4a4a';

/** Legend section title color for connection weight rows. */
export const FLAPPY_NETWORK_LEGEND_CONNECTION_TITLE_COLOR = '#00e5ff';

/** Legend section title color for node bias rows. */
export const FLAPPY_NETWORK_LEGEND_BIAS_TITLE_COLOR = '#00ff66';

/** Legend row text color. */
export const FLAPPY_NETWORK_LEGEND_ROW_TEXT_COLOR = '#9fdcff';

/** Header text color for architecture label in visualization. */
export const FLAPPY_NETWORK_HEADER_TEXT_COLOR = '#9fdcff';

/** Dark fill color used for node bias labels. */
export const FLAPPY_NETWORK_NODE_LABEL_FILL_COLOR = '#001522';

/** Output-node stroke color. */
export const FLAPPY_NETWORK_OUTPUT_NODE_STROKE_COLOR = '#d8ffe9';

/** Hidden-node stroke color. */
export const FLAPPY_NETWORK_HIDDEN_NODE_STROKE_COLOR = '#9fdcff';

/** Output-node glow color. */
export const FLAPPY_NETWORK_OUTPUT_NODE_GLOW_COLOR = 'rgba(0, 255, 102, 0.65)';
