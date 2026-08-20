/**
 * Map generation constants extracted from {@link module:./map}.
 *
 * Centralises the wall density, arena clearance, and cell-value constants so
 * they can be adjusted in one place.
 *
 * @module
 */

/**
 * Width, in fine cells, of every carved corridor in the backtracker maze.
 *
 * Each coarse cell maps to a `(W+1)×(W+1)` fine block where `W` is this width.
 * The extra `+1` row/column per block becomes the structural wall between
 * corridors, so corridors are uniformly wide — including at corners.
 */
export const MAZE_CORRIDOR_WIDTH = 3;

/**
 * Fine-grid divisor equal to `MAZE_CORRIDOR_WIDTH + 1`.
 *
 * Each coarse cell occupies a `D×D` fine block. The last row/column of each
 * block is the structural wall; the first `W×W` cells are the corridor.
 */
export const MAZE_COARSE_GRID_DIVISOR = 4;

/**
 * Fraction of single-cell wall stubs removed to add loops to the maze.
 *
 * With 3-wide corridors, each removed wall opens a 3-cell breach. A low rate
 * preserves the corridor structure while preventing a purely tree-like maze.
 */
export const MAZE_LOOP_REMOVAL_RATE = 0.03;

/**
 * Half-size, in cells, of the open central spawn arena.
 *
 * The generated map clears a square around the center so the player and other
 * entities always have a safe starting area.
 */
export const CENTRAL_ARENA_CLEARANCE_CELLS = 4;

/**
 * Open floor cell value in the flat map representation.
 *
 * Consumers such as the DDA raycaster and collision queries treat any value
 * equal to `FLOOR_CELL` as passable space.
 */
export const FLOOR_CELL = 0;

/**
 * Solid wall cell value in the flat map representation.
 *
 * Any non-zero value is treated as solid, so `WALL_CELL` is the canonical
 * opaque cell used by the maze generator and collision checks.
 */
export const WALL_CELL = 1;

/** Minimum non-zero PRNG state so a seed of `0` does not freeze generation. */
export const LCG_MIN_NONZERO_STATE = 1;
