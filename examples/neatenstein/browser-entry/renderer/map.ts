/**
 * Map generation for the Neatenstein neon raycasting demo.
 *
 * The arena is a fixed-size square grid, defined by
 * {@link NEATENSTEIN_MAP_SIZE}, with deterministic wall placement. Given the
 * same seed, every runtime tier receives the same geometry.
 *
 * The canonical map representation is a flat row-major `Uint8Array`:
 *
 * ```ts
 * const index = y * side + x;
 * ```
 *
 * A cell value of `0` means open floor. Any non-zero value is treated as solid.
 *
 * @module
 */

import { NEATENSTEIN_MAP_SIZE } from '../constants';
import {
  PARK_MILLER_MODULUS,
  PARK_MILLER_MULTIPLIER,
} from './renderer.rng.constants';
import {
  CENTRAL_ARENA_CLEARANCE_CELLS,
  FLOOR_CELL,
  INTERIOR_WALL_DENSITY,
  LCG_MIN_NONZERO_STATE,
  WALL_CELL,
} from './renderer.map.constants';
import type { CollisionMap } from './renderer.map.types';

// Re-export constants and types for external consumers.
export type { CollisionMap } from './renderer.map.types';

/**
 * Normalize a caller-provided seed into a positive PRNG state in the valid LCG
 * range.
 *
 * The generator cannot use `0` as state because it would remain `0` forever.
 * Non-finite seeds are also coerced into the minimum valid state.
 *
 * @param seed - Caller-provided seed.
 * @returns A positive non-zero LCG state.
 */
function normalizeLcgSeed(seed: number): number {
  let state =
    Number.isFinite(seed) && seed !== 0 ? seed : LCG_MIN_NONZERO_STATE;

  // Keep negative and oversized seeds deterministic by wrapping them into the
  // valid modulus range.
  state = ((state % PARK_MILLER_MODULUS) + PARK_MILLER_MODULUS) % PARK_MILLER_MODULUS;

  return state === 0 ? LCG_MIN_NONZERO_STATE : state;
}

/**
 * Advance the deterministic PRNG one step.
 *
 * @param state - Current positive LCG state.
 * @returns The next positive LCG state.
 */
function nextLcgState(state: number): number {
  return (state * PARK_MILLER_MULTIPLIER) % PARK_MILLER_MODULUS;
}

/**
 * Convert a PRNG state to a floating-point value in the half-open range
 * `[0, 1)`.
 *
 * @param state - Current positive LCG state.
 * @returns Pseudo-random unit value.
 */
function lcgStateToUnit(state: number): number {
  return state / PARK_MILLER_MODULUS;
}

/**
 * Flatten a 2D grid coordinate into a row-major array offset.
 *
 * @param x - Grid X coordinate.
 * @param y - Grid Y coordinate.
 * @param side - Width and height of the square grid.
 * @returns Flat row-major array index.
 */
function cellIndex(x: number, y: number, side: number): number {
  return y * side + x;
}

/**
 * Return whether a coordinate is outside the square map bounds.
 *
 * @param x - Grid X coordinate.
 * @param y - Grid Y coordinate.
 * @param side - Width and height of the square grid.
 * @returns Whether the coordinate is out of bounds.
 */
function isOutOfBounds(x: number, y: number, side: number): boolean {
  return x < 0 || x >= side || y < 0 || y >= side;
}

/**
 * Fill the outer map perimeter with wall cells.
 *
 * The perimeter guarantees that rays and entities remain inside the playable
 * arena under normal map generation.
 *
 * @param map - Mutable flat row-major map.
 * @param side - Width and height of the square grid.
 */
function writePerimeterWalls(map: Uint8Array, side: number): void {
  // Top and bottom rows.
  for (let x = 0; x < side; x++) {
    map[cellIndex(x, 0, side)] = WALL_CELL;
    map[cellIndex(x, side - 1, side)] = WALL_CELL;
  }

  // Left and right columns.
  for (let y = 0; y < side; y++) {
    map[cellIndex(0, y, side)] = WALL_CELL;
    map[cellIndex(side - 1, y, side)] = WALL_CELL;
  }
}

/**
 * Scatter deterministic interior walls using the seeded PRNG.
 *
 * The iteration order intentionally matches the original implementation:
 * X outer loop, Y inner loop. Keeping this order preserves existing generated
 * maps for a given seed.
 *
 * @param map - Mutable flat row-major map.
 * @param side - Width and height of the square grid.
 * @param seed - Caller-provided deterministic seed.
 */
function scatterInteriorWalls(
  map: Uint8Array,
  side: number,
  seed: number,
): void {
  let state = normalizeLcgSeed(seed);

  // Skip the perimeter because it is always solid.
  for (let x = 1; x < side - 1; x++) {
    for (let y = 1; y < side - 1; y++) {
      state = nextLcgState(state);

      if (lcgStateToUnit(state) < INTERIOR_WALL_DENSITY) {
        map[cellIndex(x, y, side)] = WALL_CELL;
      }
    }
  }
}

/**
 * Clear the central spawn arena.
 *
 * The central arena prevents the seeded scatter pass from trapping the player
 * or other entities in walls at startup.
 *
 * @param map - Mutable flat row-major map.
 * @param side - Width and height of the square grid.
 */
function carveCentralArena(map: Uint8Array, side: number): void {
  const center = Math.floor(side / 2);
  const min = center - CENTRAL_ARENA_CLEARANCE_CELLS;
  const max = center + CENTRAL_ARENA_CLEARANCE_CELLS;

  for (let x = min; x <= max; x++) {
    for (let y = min; y <= max; y++) {
      map[cellIndex(x, y, side)] = FLOOR_CELL;
    }
  }
}

/**
 * Build a deterministic wall grid for the given seed.
 *
 * The returned map is a `Uint8Array` of length
 * `NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE`.
 *
 * Generation steps:
 *
 * 1. Allocate an empty floor map.
 * 2. Write solid perimeter walls.
 * 3. Scatter deterministic interior walls using the seeded PRNG.
 * 4. Clear the central spawn arena.
 *
 * @param seed - Seed used to initialize deterministic wall scattering.
 * @returns A freshly allocated flat row-major wall grid.
 *
 * @example
 * ```ts
 * const grid = buildNeatensteinMap(12345);
 * const collision = createCollisionMap(grid, NEATENSTEIN_MAP_SIZE);
 *
 * console.log(collision.isSolid(0, 0));
 * ```
 */
export function buildNeatensteinMap(seed: number): Uint8Array {
  const side = NEATENSTEIN_MAP_SIZE;
  const map = new Uint8Array(side * side);

  // Step 1: Seal the arena so rays and entities remain within map bounds.
  writePerimeterWalls(map, side);

  // Step 2: Add deterministic interior structure.
  scatterInteriorWalls(map, side, seed);

  // Step 3: Ensure a guaranteed open start region.
  carveCentralArena(map, side);

  return map;
}

/**
 * Build a {@link CollisionMap} from a flat `Uint8Array` wall grid.
 *
 * Any non-zero cell value is treated as solid. Coordinates outside the square
 * grid boundary are also reported as solid so the player cannot step out of
 * the arena.
 *
 * This function does not copy the supplied `flatMap`; it intentionally keeps a
 * read-only view over the provided typed array. If the caller mutates the array
 * later, collision queries will reflect those mutations.
 *
 * @param flatMap - Row-major wall grid built by {@link buildNeatensteinMap}.
 * @param side - Width and height of the square grid.
 * @returns A collision map ready for movement/collision systems.
 *
 * @example
 * ```ts
 * const map = buildNeatensteinMap(123);
 * const collision = createCollisionMap(map, NEATENSTEIN_MAP_SIZE);
 *
 * console.log(collision.isSolid(0, 0)); // perimeter wall
 * ```
 */
export function createCollisionMap(
  flatMap: Uint8Array,
  side: number,
): CollisionMap {
  if (!Number.isInteger(side) || side <= 0) {
    throw new Error('Invalid map dimensions: expected a square Uint8Array.');
  }

  return {
    isSolid(x: number, y: number): boolean {
      // Out-of-bounds cells are solid by design.
      if (isOutOfBounds(x, y, side)) {
        return true;
      }

      return flatMap[cellIndex(x, y, side)] !== FLOOR_CELL;
    },
  };
}
