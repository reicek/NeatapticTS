/**
 * Clamps a numeric value to the inclusive `[min, max]` interval.
 *
 * @param value - Candidate value.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Clamped value.
 */
export function clampValue(value: number, min: number, max: number): number {
  return clamp(value, min, max);
}

/**
 * Clamps a numeric value to the inclusive `[0, 1]` interval.
 *
 * @param value - Candidate value.
 * @returns Value clamped between 0 and 1.
 */
export function clamp01(value: number): number {
  return clamp(value, 0, 1);
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
  return startValue + (endValue - startValue) * progress;
}

/**
 * Internal clamp primitive.
 *
 * @param value - Candidate value.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
