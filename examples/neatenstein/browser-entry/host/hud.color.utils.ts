/**
 * Threshold-dependent color resolvers for the Neatenstein HUD overlay.
 *
 * Pure leaf functions that map a density or health fraction to a CSS `rgb()`
 * color string based on thresholds imported from the shared constants module.
 *
 * @module
 */

import {
  NEATENSTEIN_HIVE_DENSITY_COLOR_CALM,
  NEATENSTEIN_HIVE_DENSITY_COLOR_HIGH,
  NEATENSTEIN_HIVE_DENSITY_COLOR_LOW,
  NEATENSTEIN_HIVE_DENSITY_COLOR_MID,
  NEATENSTEIN_HIVE_DENSITY_THRESHOLD_HIGH,
  NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW,
  NEATENSTEIN_HIVE_DENSITY_THRESHOLD_MID,
  NEATENSTEIN_HEALTH_COLOR_AMBER,
  NEATENSTEIN_HEALTH_COLOR_CYAN,
  NEATENSTEIN_HEALTH_COLOR_MAGENTA,
  NEATENSTEIN_HEALTH_THRESHOLD_AMBER,
  NEATENSTEIN_HEALTH_THRESHOLD_CYAN,
} from '../constants';

/**
 * Resolve a threshold-dependent CSS color for a hive density value.
 *
 * - Below {@link NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW} → calm cyan.
 * - From low (inclusive) to mid (exclusive) → low green.
 * - From mid (inclusive) to high (exclusive) → mid amber.
 * - At or above high → high magenta.
 *
 * @param density - Current density, expected in [0, 1].
 * @returns CSS `rgb()` color string.
 */
export function resolveHiveDensityColor(density: number): string {
  if (density < NEATENSTEIN_HIVE_DENSITY_THRESHOLD_LOW) {
    return NEATENSTEIN_HIVE_DENSITY_COLOR_CALM;
  }
  if (density < NEATENSTEIN_HIVE_DENSITY_THRESHOLD_MID) {
    return NEATENSTEIN_HIVE_DENSITY_COLOR_LOW;
  }
  if (density < NEATENSTEIN_HIVE_DENSITY_THRESHOLD_HIGH) {
    return NEATENSTEIN_HIVE_DENSITY_COLOR_MID;
  }
  return NEATENSTEIN_HIVE_DENSITY_COLOR_HIGH;
}

/**
 * Resolve a threshold-dependent CSS color for a health fraction.
 *
 * - At or above {@link NEATENSTEIN_HEALTH_THRESHOLD_CYAN} → cyan.
 * - From amber (inclusive) to cyan (exclusive) → amber.
 * - Below amber → magenta.
 *
 * @param fraction - Current health fraction (health / maxHealth), expected in
 *   [0, 1].
 * @returns CSS `rgb()` color string.
 */
export function resolveHealthColor(fraction: number): string {
  if (fraction >= NEATENSTEIN_HEALTH_THRESHOLD_CYAN) {
    return NEATENSTEIN_HEALTH_COLOR_CYAN;
  }
  if (fraction >= NEATENSTEIN_HEALTH_THRESHOLD_AMBER) {
    return NEATENSTEIN_HEALTH_COLOR_AMBER;
  }
  return NEATENSTEIN_HEALTH_COLOR_MAGENTA;
}
