import type { ColorTier } from '../browser-entry.types';

/**
 * Visualization-specific color-scale contracts for network rendering.
 *
 * The Flappy Bird demo renders connection weights and node biases with tiered
 * neon ramps so humans can quickly read sign and magnitude without parsing raw
 * numbers on every edge and node.
 */

/**
 * Dynamic tiered color scale used by visualization render layers.
 *
 * The scale records the observed numeric range plus the ordered threshold tiers
 * used to map values into colors.
 */
export interface DynamicColorScale {
  minimumValue: number;
  maximumValue: number;
  tiers: ColorTier[];
  aboveTierColor: string;
}

/**
 * Grouped color scales for connection and bias channels.
 *
 * Keeping the two scales together ensures the legend and drawing code read from
 * one consistent view of the active network range.
 */
export interface NetworkVisualizationColorScales {
  connectionScale: DynamicColorScale;
  biasScale: DynamicColorScale;
}
