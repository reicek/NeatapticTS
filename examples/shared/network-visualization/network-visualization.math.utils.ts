/**
 * Shared math utilities for the network-visualization domain.
 *
 * Keep these helpers self-contained so the visualizer stays dependency-free
 * and can be imported by any example without pulling in core library internals.
 */

/**
 * Clamps a numeric value to the inclusive `[minimum, maximum]` range.
 *
 * @param value - Value to clamp.
 * @param minimum - Lower bound.
 * @param maximum - Upper bound.
 * @returns Clamped value.
 */
export const clamp = (
  value: number,
  minimum: number,
  maximum: number,
): number => Math.min(maximum, Math.max(minimum, value));
