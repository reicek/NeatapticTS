/**
 * Map generation for the Neatenstein neon raycasting demo.
 *
 * The arena is a fixed-size square grid (see {@link NEATENSTEIN_MAP_SIZE})
 * with seeded, deterministic wall placement so every mode plays the same
 * geometry for a given seed.
 *
 * @module
 */

import { NEATENSTEIN_MAP_SIZE } from '../constants';

/** Linear PRNG modulus: the largest Mersenne prime fitting in a signed 32-bit int. */
const LCG_MODULUS = 2_147_483_647;

/** Linear PRNG multiplier from the Park-Miller LCG. */
const LCG_MULTIPLIER = 16_807;

/** Minimum non-zero PRNG state so a seed of 0 does not freeze the generator. */
const LCG_MIN_NONZERO_STATE = 1;

/** Fraction of interior cells converted to walls by the seeded scatter pass. */
const INTERIOR_WALL_DENSITY = 0.22;

/** Clearance around the map center kept open so entities always have a playable start. */
const CENTRAL_ARENA_CLEARANCE_CELLS = 2;

/**
 * Normalize a caller-provided seed into a positive PRNG state in the
 * valid LCG range.
 */
function normalizeLcgSeed(seed: number): number {
  let state =
    Number.isFinite(seed) && seed !== 0 ? seed : LCG_MIN_NONZERO_STATE;
  state = ((state % LCG_MODULUS) + LCG_MODULUS) % LCG_MODULUS;
  return state === 0 ? LCG_MIN_NONZERO_STATE : state;
}

/** Advance the deterministic PRNG one step. */
function nextLcgState(state: number): number {
  return (state * LCG_MULTIPLIER) % LCG_MODULUS;
}

/** Convert a uniform PRNG state to a floating-point value in [0, 1). */
function lcgStateToUnit(state: number): number {
  return state / LCG_MODULUS;
}

/** Flatten a 2D grid coordinate into a 1D array offset. */
function cellIndex(x: number, y: number, side: number): number {
  return y * side + x;
}

/**
 * Build a deterministic 24×24 wall grid for the given seed.
 *
 * The map is a `Uint8Array` of length `NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE`
 * where `0` denotes open floor and `1` denotes a wall. Perimeter cells are
 * always walls, interior cells are scattered using the seeded PRNG, and the
 * central arena is kept open to guarantee a playable start region.
 *
 * @param seed - Integer seed used to initialize the deterministic wall scatter.
 * @returns A freshly allocated wall grid.
 *
 * @example
 * ```ts
 * const grid = buildNeatensteinMap(12345);
 * console.log(grid.length); // 576
 * ```
 */
export function buildNeatensteinMap(seed: number): Uint8Array {
  const side = NEATENSTEIN_MAP_SIZE;
  const cells = side * side;
  const map = new Uint8Array(cells);

  // Step 1: Close the outer perimeter so the player cannot leave the arena.
  for (let x = 0; x < side; x++) {
    map[cellIndex(x, 0, side)] = 1;
    map[cellIndex(x, side - 1, side)] = 1;
  }
  for (let y = 0; y < side; y++) {
    map[cellIndex(0, y, side)] = 1;
    map[cellIndex(side - 1, y, side)] = 1;
  }

  // Step 2: Scatter interior walls using the seeded PRNG.
  let state = normalizeLcgSeed(seed);
  for (let x = 1; x < side - 1; x++) {
    for (let y = 1; y < side - 1; y++) {
      state = nextLcgState(state);
      if (lcgStateToUnit(state) < INTERIOR_WALL_DENSITY) {
        map[cellIndex(x, y, side)] = 1;
      }
    }
  }

  // Step 3: Carve the central arena so the player and enemies can spawn.
  const center = Math.floor(side / 2);
  const min = center - CENTRAL_ARENA_CLEARANCE_CELLS;
  const max = center + CENTRAL_ARENA_CLEARANCE_CELLS;
  for (let x = min; x <= max; x++) {
    for (let y = min; y <= max; y++) {
      map[cellIndex(x, y, side)] = 0;
    }
  }

  return map;
}
