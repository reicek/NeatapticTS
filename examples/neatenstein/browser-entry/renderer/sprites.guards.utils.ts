/**
 * Pure guard predicates extracted from the sprite renderer.
 *
 * These helpers are framework-agnostic value checks with no module-level
 * state, making them safe to unit-test in isolation and reuse across
 * renderer modules.
 *
 * @module
 */

/**
 * Return whether a number is a positive finite integer dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is usable as a framebuffer/canvas dimension.
 */
export function isPositiveIntegerDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0;
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
