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
  LCG_MIN_NONZERO_STATE,
  MAZE_COARSE_GRID_DIVISOR,
  MAZE_CORRIDOR_WIDTH,
  MAZE_LOOP_REMOVAL_RATE,
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
  state =
    ((state % PARK_MILLER_MODULUS) + PARK_MILLER_MODULUS) % PARK_MILLER_MODULUS;

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
 * Mutable Park-Miller LCG pseudo-random unit generator for maze carving.
 *
 * Each call to {@link MazeRng.unit} advances the LCG one step and returns the
 * next value in `[0, 1)`. Draw order across maze phases is documented in
 * {@link buildBacktrackerMaze}.
 */
interface MazeRng {
  /** Advance the LCG and return the next pseudo-random value in `[0, 1)`. */
  unit(): number;
}

/** Four cardinal directions for coarse-grid traversal and flood-fill. */
const CARDINAL_DIRECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, -1],
  [0, 1],
  [-1, 0],
  [1, 0],
];

/**
 * Create a mutable Park-Miller LCG generator from a caller seed.
 *
 * @param seed - Caller-provided deterministic seed.
 * @returns A mutable generator whose `unit()` method advances the LCG.
 */
function createMazeRng(seed: number): MazeRng {
  let state = normalizeLcgSeed(seed);
  return {
    unit(): number {
      state = nextLcgState(state);
      return lcgStateToUnit(state);
    },
  };
}

/**
 * Carve the `W×W` corridor area of a coarse cell to floor.
 *
 * The last row/column of each coarse block remains as a structural wall;
 * only the corridor area is opened. Perimeter cells are never carved.
 *
 * @param map - Mutable flat row-major map.
 * @param cx - Coarse-grid X coordinate.
 * @param cy - Coarse-grid Y coordinate.
 * @param side - Width and height of the square fine grid.
 */
function carveCoarseCell(
  map: Uint8Array,
  cx: number,
  cy: number,
  side: number,
): void {
  const baseX = cx * MAZE_COARSE_GRID_DIVISOR;
  const baseY = cy * MAZE_COARSE_GRID_DIVISOR;

  for (let dx = 0; dx < MAZE_CORRIDOR_WIDTH; dx++) {
    for (let dy = 0; dy < MAZE_CORRIDOR_WIDTH; dy++) {
      const fx = baseX + dx;
      const fy = baseY + dy;
      // Skip perimeter cells — they are sealed after generation.
      if (fx > 0 && fx < side - 1 && fy > 0 && fy < side - 1) {
        map[cellIndex(fx, fy, side)] = FLOOR_CELL;
      }
    }
  }
}

/**
 * Carve a vertical passage column at a specific fine X coordinate.
 *
 * @param map - Mutable flat row-major map.
 * @param wallX - Fine X coordinate of the wall column to carve.
 * @param baseY - Fine Y coordinate of the corridor start.
 * @param side - Width and height of the square fine grid.
 */
function carveWallColumn(
  map: Uint8Array,
  wallX: number,
  baseY: number,
  side: number,
): void {
  for (let dy = 0; dy < MAZE_CORRIDOR_WIDTH; dy++) {
    const fy = baseY + dy;
    if (wallX > 0 && wallX < side - 1 && fy > 0 && fy < side - 1) {
      map[cellIndex(wallX, fy, side)] = FLOOR_CELL;
    }
  }
}

/**
 * Carve a horizontal passage row at a specific fine Y coordinate.
 *
 * @param map - Mutable flat row-major map.
 * @param wallY - Fine Y coordinate of the wall row to carve.
 * @param baseX - Fine X coordinate of the corridor start.
 * @param side - Width and height of the square fine grid.
 */
function carveWallRow(
  map: Uint8Array,
  wallY: number,
  baseX: number,
  side: number,
): void {
  for (let dx = 0; dx < MAZE_CORRIDOR_WIDTH; dx++) {
    const fx = baseX + dx;
    if (wallY > 0 && wallY < side - 1 && fx > 0 && fx < side - 1) {
      map[cellIndex(fx, wallY, side)] = FLOOR_CELL;
    }
  }
}

/**
 * Carve the structural wall between two adjacent coarse cells.
 *
 * @param map - Mutable flat row-major map.
 * @param cx - Current coarse X.
 * @param cy - Current coarse Y.
 * @param nx - Neighbor coarse X.
 * @param ny - Neighbor coarse Y.
 * @param side - Width and height of the square fine grid.
 */
function carvePassageBetween(
  map: Uint8Array,
  cx: number,
  cy: number,
  nx: number,
  ny: number,
  side: number,
): void {
  const d = MAZE_COARSE_GRID_DIVISOR;
  const w = MAZE_CORRIDOR_WIDTH;

  if (nx > cx) {
    carveWallColumn(map, cx * d + w, cy * d, side);
  } else if (nx < cx) {
    carveWallColumn(map, nx * d + w, cy * d, side);
  } else if (ny > cy) {
    carveWallRow(map, cy * d + w, cx * d, side);
  } else if (ny < cy) {
    carveWallRow(map, ny * d + w, cx * d, side);
  }
}

/**
 * Collect unvisited cardinal neighbors of a coarse cell.
 *
 * @param cx - Coarse X coordinate.
 * @param cy - Coarse Y coordinate.
 * @param coarseSide - Width and height of the coarse grid.
 * @param visited - Visited marker array for the coarse grid.
 * @returns Array of `[nx, ny]` neighbor coordinates.
 */
function collectUnvisitedNeighbors(
  cx: number,
  cy: number,
  coarseSide: number,
  visited: Uint8Array,
): Array<[number, number]> {
  const neighbors: Array<[number, number]> = [];

  for (const [dx, dy] of CARDINAL_DIRECTIONS) {
    const nx = cx + dx;
    const ny = cy + dy;
    if (nx >= 0 && nx < coarseSide && ny >= 0 && ny < coarseSide) {
      if (visited[ny * coarseSide + nx] === 0) {
        neighbors.push([nx, ny]);
      }
    }
  }

  return neighbors;
}

/**
 * Fisher-Yates shuffle using LCG draws.
 *
 * @param arr - Array to shuffle in place.
 * @param rng - Mutable LCG generator (draw order: phase 1 — backtracker).
 */
function shuffleInPlace(arr: Array<[number, number]>, rng: MazeRng): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng.unit() * (i + 1));
    const temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
  }
}

/**
 * Run the coarse-grid recursive backtracker to carve a perfect maze.
 *
 * The backtracker roots at the center coarse cell and produces a spanning tree
 * where every coarse cell has exactly one path back to the center. Corridors
 * are {@link MAZE_CORRIDOR_WIDTH}-wide by construction; corners are uniformly
 * wide because each carved coarse cell opens a full `W×W` block.
 *
 * LCG draw order (phase 1): one draw per Fisher-Yates step during neighbor
 * shuffle.
 *
 * @param map - Mutable flat row-major map (all walls before entry).
 * @param side - Width and height of the square fine grid.
 * @param rng - Mutable LCG generator.
 */
function carveBacktrackerCorridors(
  map: Uint8Array,
  side: number,
  rng: MazeRng,
): void {
  const coarseSide = Math.floor(side / MAZE_COARSE_GRID_DIVISOR);
  const visited = new Uint8Array(coarseSide * coarseSide);
  const stack: number[] = [];

  // Root at the center coarse cell.
  const centerCoarse = Math.floor(coarseSide / 2);
  const rootIndex = centerCoarse * coarseSide + centerCoarse;
  visited[rootIndex] = 1;
  stack.push(rootIndex);
  carveCoarseCell(map, centerCoarse, centerCoarse, side);

  while (stack.length > 0) {
    const current = stack[stack.length - 1];
    const cx = current % coarseSide;
    const cy = Math.floor(current / coarseSide);

    const neighbors = collectUnvisitedNeighbors(cx, cy, coarseSide, visited);

    if (neighbors.length === 0) {
      stack.pop();
      continue;
    }

    shuffleInPlace(neighbors, rng);

    const [nx, ny] = neighbors[0];
    carvePassageBetween(map, cx, cy, nx, ny, side);

    const nextIndex = ny * coarseSide + nx;
    visited[nextIndex] = 1;
    carveCoarseCell(map, nx, ny, side);
    stack.push(nextIndex);
  }
}

/**
 * Remove a fraction of single-cell wall stubs to add loops to the maze.
 *
 * A wall stub is a wall cell with floor on both sides in at least one
 * cardinal direction. Only stubs strictly inside the perimeter are
 * considered. Each candidate draws the LCG once (draw order: phase 2 —
 * loop-removal candidate selection).
 *
 * @param map - Mutable flat row-major map.
 * @param side - Width and height of the square fine grid.
 * @param rng - Mutable LCG generator.
 */
function addMazeLoops(map: Uint8Array, side: number, rng: MazeRng): void {
  for (let y = 1; y < side - 1; y++) {
    for (let x = 1; x < side - 1; x++) {
      const idx = cellIndex(x, y, side);
      if (map[idx] !== WALL_CELL) {
        continue;
      }

      const leftFloor = map[cellIndex(x - 1, y, side)] === FLOOR_CELL;
      const rightFloor = map[cellIndex(x + 1, y, side)] === FLOOR_CELL;
      const upFloor = map[cellIndex(x, y - 1, side)] === FLOOR_CELL;
      const downFloor = map[cellIndex(x, y + 1, side)] === FLOOR_CELL;

      const isHorizontalStub = leftFloor && rightFloor;
      const isVerticalStub = upFloor && downFloor;

      if (isHorizontalStub || isVerticalStub) {
        if (rng.unit() < MAZE_LOOP_REMOVAL_RATE) {
          map[idx] = FLOOR_CELL;
        }
      }
    }
  }
}

/**
 * BFS flood-fill from the center, marking all reachable floor cells.
 *
 * @param map - Flat row-major map.
 * @param side - Width and height of the square grid.
 * @param reachable - Mutable marker array; reachable cells are set to `1`.
 */
function floodFillFromCenter(
  map: Uint8Array,
  side: number,
  reachable: Uint8Array,
): void {
  const center = Math.floor(side / 2);
  const startIdx = cellIndex(center, center, side);

  if (map[startIdx] !== FLOOR_CELL) {
    return;
  }

  const queue: number[] = [startIdx];
  reachable[startIdx] = 1;

  let head = 0;
  while (head < queue.length) {
    const idx = queue[head++];
    const x = idx % side;
    const y = Math.floor(idx / side);

    for (const [dx, dy] of CARDINAL_DIRECTIONS) {
      const nx = x + dx;
      const ny = y + dy;

      if (nx < 1 || nx >= side - 1 || ny < 1 || ny >= side - 1) {
        continue;
      }

      const nidx = cellIndex(nx, ny, side);
      if (reachable[nidx] === 0 && map[nidx] === FLOOR_CELL) {
        reachable[nidx] = 1;
        queue.push(nidx);
      }
    }
  }
}

/**
 * Carve a passage from an unreachable floor cell toward the center.
 *
 * Moves step-by-step toward the map center, carving through walls until a
 * reachable floor cell is encountered. Each ambiguous step (both X and Y
 * differ from center) draws the LCG once (draw order: phase 3 — repair).
 *
 * @param map - Mutable flat row-major map.
 * @param side - Width and height of the square fine grid.
 * @param startX - Fine X of the unreachable floor cell.
 * @param startY - Fine Y of the unreachable floor cell.
 * @param reachable - Mutable reachable marker array.
 * @param rng - Mutable LCG generator.
 */
function carveRepairPassage(
  map: Uint8Array,
  side: number,
  startX: number,
  startY: number,
  reachable: Uint8Array,
  rng: MazeRng,
): void {
  const center = Math.floor(side / 2);
  let cx = startX;
  let cy = startY;

  while (reachable[cellIndex(cx, cy, side)] === 0) {
    map[cellIndex(cx, cy, side)] = FLOOR_CELL;
    reachable[cellIndex(cx, cy, side)] = 1;

    const dx = cx < center ? 1 : cx > center ? -1 : 0;
    const dy = cy < center ? 1 : cy > center ? -1 : 0;

    if (dx !== 0 && dy !== 0) {
      if (rng.unit() < 0.5) {
        cx += dx;
      } else {
        cy += dy;
      }
    } else if (dx !== 0) {
      cx += dx;
    } else {
      cy += dy;
    }

    // Clamp to interior bounds.
    cx = Math.max(1, Math.min(side - 2, cx));
    cy = Math.max(1, Math.min(side - 2, cy));
  }
}

/**
 * Verify all floor cells are reachable from the center and repair isolated
 * clusters.
 *
 * With a spanning-tree maze, unreachable clusters should not occur. This
 * function is a safety net: it flood-fills from the center, then carves
 * connecting passages for any floor cell the flood-fill missed.
 *
 * @param map - Mutable flat row-major map.
 * @param side - Width and height of the square fine grid.
 * @param rng - Mutable LCG generator.
 */
function verifyAndRepairConnectivity(
  map: Uint8Array,
  side: number,
  rng: MazeRng,
): void {
  const reachable = new Uint8Array(side * side);
  floodFillFromCenter(map, side, reachable);

  for (let y = 1; y < side - 1; y++) {
    for (let x = 1; x < side - 1; x++) {
      const idx = cellIndex(x, y, side);
      if (map[idx] === FLOOR_CELL && reachable[idx] === 0) {
        carveRepairPassage(map, side, x, y, reachable, rng);
      }
    }
  }
}

/**
 * Build the coarse-grid recursive backtracker maze.
 *
 * This replaces the former scatter-and-arena approach with a structured maze
 * that guarantees every floor cell is reachable from the center.
 *
 * Generation steps:
 *
 * 1. Carve corridors via the coarse-grid recursive backtracker rooted at
 *    the center (LCG draw order phase 1).
 * 2. Expand the central arena to the full clearance zone.
 * 3. Add loops by removing single-cell wall stubs (LCG phase 2).
 * 4. Verify connectivity and repair any isolated clusters (LCG phase 3).
 *
 * @param map - Mutable flat row-major map (all walls before entry).
 * @param side - Width and height of the square fine grid.
 * @param seed - Caller-provided deterministic seed.
 * @returns The same `map` reference, now carved with floor corridors.
 */
function buildBacktrackerMaze(
  map: Uint8Array,
  side: number,
  seed: number,
): void {
  const rng = createMazeRng(seed);

  // Step 1: Carve the spanning-tree maze corridors.
  carveBacktrackerCorridors(map, side, rng);

  // Step 2: Expand the central arena hub.
  carveCentralArena(map, side);

  // Step 3: Add loops to break the purely tree-like structure.
  addMazeLoops(map, side, rng);

  // Step 4: Verify all floor cells are reachable; repair if needed.
  verifyAndRepairConnectivity(map, side, rng);
}

/**
 * Clear the central spawn arena.
 *
 * The central arena is the hub of the recursive backtracker maze. It is kept
 * clear so the player and other entities cannot spawn inside walls at startup.
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
 * 1. Allocate a fully walled map.
 * 2. Build the coarse-grid recursive backtracker maze rooted at the center,
 *    producing 3-wide corridors that all converge on the central arena hub.
 * 3. Seal the outer perimeter.
 *
 * @param seed - Seed used to initialize deterministic maze generation.
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

  // Step 1: Initialize every cell as a wall.
  map.fill(WALL_CELL);

  // Step 2: Carve the coarse-grid backtracker maze (corridors, arena, loops, repair).
  buildBacktrackerMaze(map, side, seed);

  // Step 3: Seal the perimeter so rays and entities stay within the arena.
  writePerimeterWalls(map, side);

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
 * @throws Error when `side` is not a positive integer.
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
