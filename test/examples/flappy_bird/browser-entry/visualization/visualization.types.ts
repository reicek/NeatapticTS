import type { ColorTier } from '../browser-entry.types';

/**
 * Dynamic tiered color scale used by visualization render layers.
 */
export interface DynamicColorScale {
  minimumValue: number;
  maximumValue: number;
  tiers: ColorTier[];
  aboveTierColor: string;
}

/**
 * Grouped color scales for connection and bias channels.
 */
export interface NetworkVisualizationColorScales {
  connectionScale: DynamicColorScale;
  biasScale: DynamicColorScale;
}
