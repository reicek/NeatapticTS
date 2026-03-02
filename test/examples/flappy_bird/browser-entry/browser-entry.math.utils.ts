import {
  clamp01 as sharedClamp01,
  clampValue as sharedClampValue,
  interpolateValue as sharedInterpolateValue,
} from '../flappy.simulation.shared.utils';

/**
 * Clamps a numeric value to the inclusive `[min, max]` interval.
 *
 * @param value - Candidate value.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Clamped value.
 */
export function clamp(value: number, min: number, max: number): number {
  return sharedClampValue(value, min, max);
}

/**
 * Clamps a numeric value to the inclusive `[0, 1]` interval.
 *
 * @param value - Candidate value.
 * @returns Value clamped between 0 and 1.
 */
export function clamp01(value: number): number {
  return sharedClamp01(value);
}

/**
 * Linear interpolation helper.
 *
 * @param startValue - Start value at progress `0`.
 * @param endValue - End value at progress `1`.
 * @param progress - Normalized interpolation progress.
 * @returns Interpolated value.
 */
export function interpolateValue(
  startValue: number,
  endValue: number,
  progress: number,
): number {
  return sharedInterpolateValue(startValue, endValue, progress);
}

/**
 * Converts a six-digit hex color to rgba with the requested alpha.
 *
 * @param hexColor - Color in `#RRGGBB` form.
 * @param alphaValue - Alpha value to apply.
 * @returns rgba color string, or original value when not 6-digit hex.
 */
export function applyAlphaToHexColor(
  hexColor: string,
  alphaValue: number,
): string {
  const normalizedColor = hexColor.replace('#', '');
  if (normalizedColor.length !== 6) {
    return hexColor;
  }

  const redChannel = parseInt(normalizedColor.slice(0, 2), 16);
  const greenChannel = parseInt(normalizedColor.slice(2, 4), 16);
  const blueChannel = parseInt(normalizedColor.slice(4, 6), 16);

  return `rgba(${redChannel}, ${greenChannel}, ${blueChannel}, ${clamp(alphaValue, 0, 1)})`;
}
