/**
 * Map generation constants extracted from {@link module:./map}.
 *
 * Centralises the wall density, arena clearance, and cell-value constants so
 * they can be adjusted in one place.
 *
 * @module
 */

/**
 * Fraction of interior cells converted to walls by the seeded scatter pass.
 *
 * This value is intentionally local to map generation so gameplay systems only
 * depend on the resulting collision data, not generation policy.
 */
export const INTERIOR_WALL_DENSITY = 0.12;

/**
 * Half-size, in cells, of the open central spawn arena.
 *
 * The generated map clears a square around the center so the player and other
 * entities always have a safe starting area.
 */
export const CENTRAL_ARENA_CLEARANCE_CELLS = 4;

/** Open floor cell value in the flat map. */
export const FLOOR_CELL = 0;

/** Solid wall cell value in the flat map. */
export const WALL_CELL = 1;

/** Minimum non-zero PRNG state so a seed of `0` does not freeze generation. */
export const LCG_MIN_NONZERO_STATE = 1;