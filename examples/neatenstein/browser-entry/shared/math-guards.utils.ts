/**
 * Consolidated math guard helpers for the Neatenstein demo.
 *
 * This module is the single source of truth for value-clamping and
 * finite-number predicate helpers used across the renderer, worker,
 * harness, and shared modules.  All other modules should import from
 * here rather than defining their own local copies.
 *
 * @module
 */

/**
 * Clamp a numeric value to the inclusive range `[min, max]`.
 *
 * @param value - Value to clamp.
 * @param min - Inclusive minimum.
 * @param max - Inclusive maximum.
 * @returns The clamped value.
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Clamp a numeric value to the inclusive `[0, 1]` range.
 *
 * `NaN` inputs return `0` so that division-by-zero edge cases produce a
 * safe lower bound instead of `NaN`.
 *
 * @param value - Raw numeric value.
 * @returns The clamped value in `[0, 1]`.
 */
export function clamp01(value: number): number {
  if (Number.isNaN(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

/**
 * Clamp a value to an integer range, truncating the value and normalising
 * reversed bounds.
 *
 * @param value - Value to clamp (will be truncated toward zero).
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Integer clamped into the normalised range.
 */
export function clampInt(value: number, min: number, max: number): number {
  const rawMin = Number.isFinite(min) ? Math.trunc(min) : 0;
  const rawMax = Number.isFinite(max) ? Math.trunc(max) : rawMin;

  const lower = Math.min(rawMin, rawMax);
  const upper = Math.max(rawMin, rawMax);

  if (Number.isNaN(value) || value === Number.NEGATIVE_INFINITY) {
    return lower;
  }

  if (value === Number.POSITIVE_INFINITY) {
    return upper;
  }

  const truncated = Math.trunc(value);

  if (truncated < lower) {
    return lower;
  }

  if (truncated > upper) {
    return upper;
  }

  return truncated;
}

/**
 * Clamp a floating-point colour value to an 8-bit unsigned byte.
 *
 * @param value - Value to clamp.
 * @returns Integer in `[0, 255]`.
 */
export function clampByte(value: number): number {
  return Math.max(0, Math.min(255, Math.round(value)));
}

/**
 * Return whether a value is a finite number.
 *
 * @param value - Candidate value.
 * @returns Whether the value is finite.
 */
export function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Return whether a number is finite and greater than zero.
 *
 * @param value - Candidate scalar.
 * @returns Whether the value is positive and finite.
 */
export function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Return whether a value is usable as a positive render dimension.
 *
 * @param value - Candidate width or height.
 * @returns Whether the value is a positive finite number.
 */
export function isPositiveFiniteDimension(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}