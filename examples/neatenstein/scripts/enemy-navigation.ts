/**
 * BFS distance map navigation for Neatenstein enemies.
 *
 * Recycles the BFS distance map pattern from `examples/asciiMaze/mazeUtils.ts`
 * (`buildDistanceMapFlat`) and the cardinal-neighbor gradient descent from
 * `examples/asciiMaze/mazeMovement/`. The distance map is built from the
 * player's grid cell each tick; enemies follow the decreasing distance
 * gradient to navigate around walls toward the player.
 *
 * @module
 */

import type { CollisionMap } from '../browser-entry/renderer/map';

/** Sentinel value stored in the distance buffer for wall cells. */
const WALL_VALUE = -2;

/** Sentinel value stored in the distance buffer for unreachable cells. */
const UNREACHABLE_VALUE = -1;

/**
 * Four cardinal directions in N, E, S, W order (matches asciiMaze).
 *
 * The order matters for tie-breaking: when two neighbors have the same
 * distance, the first one in this array wins (strict `<` comparison), so
 * east is preferred over south on equal distances.
 */
const DIRECTIONS: readonly (readonly [number, number])[] = [
  [0, -1], // N
  [1, 0], // E
  [0, 1], // S
  [-1, 0], // W
];

/** Shared, reusable queue buffer to avoid per-call allocations. */
let queueBuffer: Int32Array = new Int32Array(0);

/**
 * Ensure the shared queue buffer has at least `minLength` capacity.
 *
 * Grows to the next power-of-two for exponential allocation behaviour,
 * mirroring the pooling strategy in `mazeUtils.ts`.
 *
 * @param minLength - Minimum number of elements the buffer must hold.
 * @returns The shared queue buffer (may be newly allocated).
 */
function getQueueBuffer(minLength: number): Int32Array {
  if (queueBuffer.length < minLength) {
    let capacity = queueBuffer.length || 1;
    while (capacity < minLength) capacity <<= 1;
    queueBuffer = new Int32Array(capacity);
  }
  return queueBuffer;
}

/**
 * BFS distance map from a goal cell to every reachable cell.
 *
 * The `distances` buffer is a flat `Int32Array` of length `size * size`,
 * indexed as `y * size + x`. Wall cells are `wallValue`, unreachable cells
 * are `unreachableValue`, and reachable cells hold a non-negative distance
 * (0 at the goal, increasing by 1 per BFS step).
 */
export interface DistanceMap {
  /** Grid width and height (square grid). */
  readonly size: number;
  /** Flat distance buffer indexed as `y * size + x`. */
  readonly distances: Int32Array;
  /** Sentinel for wall cells. */
  readonly wallValue: number;
  /** Sentinel for unreachable cells. */
  readonly unreachableValue: number;
}

/**
 * Build a BFS distance map from the goal cell to every reachable cell.
 *
 * Recycles the `buildDistanceMapFlat` pattern from `mazeUtils.ts`: typed
 * arrays, sentinel values, shared queue buffer pooling, and four-cardinal
 * neighbour expansion.
 *
 * @param collisionMap - Map queried for solid cells.
 * @param goalX - Goal cell X (typically the player's grid X).
 * @param goalY - Goal cell Y (typically the player's grid Y).
 * @param size - Grid width and height.
 * @returns Distance map with distances, sentinels, and metadata.
 */
export function buildEnemyDistanceMap(
  collisionMap: CollisionMap,
  goalX: number,
  goalY: number,
  size: number,
): DistanceMap {
  const cellCount = size * size;
  const distances = new Int32Array(cellCount);

  // Initialize all cells to UNREACHABLE.
  for (let i = 0; i < cellCount; i++) {
    distances[i] = UNREACHABLE_VALUE;
  }

  // Mark walls.
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (collisionMap.isSolid(x, y)) {
        distances[y * size + x] = WALL_VALUE;
      }
    }
  }

  // If the goal is out of bounds or on a wall, return early (no paths).
  // Uses `!(x >= 0)` instead of `x < 0` so NaN coordinates are rejected.
  const goalIndex = goalY * size + goalX;
  if (
    !(goalX >= 0) ||
    !(goalX < size) ||
    !(goalY >= 0) ||
    !(goalY < size) ||
    distances[goalIndex] === WALL_VALUE
  ) {
    return {
      size,
      distances,
      wallValue: WALL_VALUE,
      unreachableValue: UNREACHABLE_VALUE,
    };
  }

  distances[goalIndex] = 0;

  // BFS from the goal outward.
  const queue = getQueueBuffer(cellCount);
  let queueHead = 0;
  let queueTail = 0;
  queue[queueTail++] = goalIndex;

  while (queueHead < queueTail) {
    const currentIndex = queue[queueHead++];
    const currentDistance = distances[currentIndex];

    const currentY = (currentIndex / size) | 0;
    const currentX = currentIndex - currentY * size;

    for (const [dx, dy] of DIRECTIONS) {
      const nx = currentX + dx;
      const ny = currentY + dy;
      if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;

      const ni = ny * size + nx;
      if (distances[ni] === UNREACHABLE_VALUE) {
        distances[ni] = currentDistance + 1;
        queue[queueTail++] = ni;
      }
    }
  }

  return {
    size,
    distances,
    wallValue: WALL_VALUE,
    unreachableValue: UNREACHABLE_VALUE,
  };
}

/**
 * Look up the distance value for a grid cell.
 *
 * @param map - Distance map built by {@link buildEnemyDistanceMap}.
 * @param x - Grid X coordinate.
 * @param y - Grid Y coordinate.
 * @returns Distance value (≥ 0 for reachable, `wallValue` for walls,
 *   `unreachableValue` for unreachable or out-of-bounds).
 */
export function getDistance(map: DistanceMap, x: number, y: number): number {
  const { size, distances, unreachableValue } = map;
  // `!(x >= 0)` rejects NaN as well as negatives.
  if (!(x >= 0) || !(x < size) || !(y >= 0) || !(y < size))
    return unreachableValue;
  return distances[y * size + x];
}

/** Direction toward the best navigation neighbour. */
export interface NavigationStep {
  /** X delta (−1, 0, or 1). */
  readonly dx: number;
  /** Y delta (−1, 0, or 1). */
  readonly dy: number;
  /** Distance value of the target cell. */
  readonly distance: number;
}

/**
 * Find the cardinal neighbour with the lowest distance to the goal.
 *
 * Neighbours are checked in N, E, S, W order. Ties are broken by
 * first-found-wins (strict `<` comparison), so east is preferred over
 * south on equal distances.
 *
 * @param map - Distance map built by {@link buildEnemyDistanceMap}.
 * @param cellX - Current grid X.
 * @param cellY - Current grid Y.
 * @returns Best navigation step, or `null` when the current cell is a wall,
 *   unreachable, or no neighbour has a lower distance (dead-end).
 */
export function findBestNavigationStep(
  map: DistanceMap,
  cellX: number,
  cellY: number,
): NavigationStep | null {
  const currentDist = getDistance(map, cellX, cellY);
  if (currentDist < 0) return null;

  let best: NavigationStep | null = null;

  for (const [dx, dy] of DIRECTIONS) {
    const dist = getDistance(map, cellX + dx, cellY + dy);
    if (dist >= 0 && dist < currentDist) {
      if (best === null || dist < best.distance) {
        best = { dx, dy, distance: dist };
      }
    }
  }

  return best;
}

/** Scalar step per compass direction: 0 = N, 0.25 = E, 0.5 = S, 0.75 = W. */
const COMPASS_STEP = 0.25;

/** Absolute clip applied to step-delta when computing the progress signal. */
const PROGRESS_CLIP = 2;

/** Scale applied to clipped progress delta before adding to neutral baseline. */
const PROGRESS_SCALE = 4;

/** Neutral progress value returned when progress cannot be computed. */
const PROGRESS_NEUTRAL = 0.5;

/**
 * Build a 6-element vision vector from BFS distance data for neural network
 * input.
 *
 * The vector layout is:
 * `[compassScalar, openN, openE, openS, openW, progressDelta]`
 *
 * - **compassScalar**: `bestDirection * 0.25` (range `[0, 0.75]`) where
 *   `bestDirection` is the cardinal direction index (0 = N, 1 = E, 2 = S,
 *   3 = W) with the lowest neighbour distance.
 * - **openN/E/S/W**: Normalised path quality. `1.0` for the best (lowest
 *   distance) reachable neighbour, `0` for walls and unreachable cells.
 *   Other reachable neighbours are scaled by `bestDist / neighbourDist`.
 * - **progressDelta**: `0.5 + clip(prevDist - curDist, -2, 2) / 4`
 *   (range `[0, 1]`). Neutral `0.5` when `previousDistance` is unavailable
 *   or the current cell is unreachable.
 *
 * @param distanceMap - BFS distance map from the player position.
 * @param cellX - Enemy grid X coordinate.
 * @param cellY - Enemy grid Y coordinate.
 * @param previousDistance - Distance at the enemy's cell from the previous
 *   tick, or `undefined` when no previous data is available.
 * @returns `Float32Array` of length 6.
 */
export function buildVisionVector(
  distanceMap: DistanceMap,
  cellX: number,
  cellY: number,
  previousDistance: number | undefined,
): Float32Array {
  const vision = new Float32Array(6);

  // Gather neighbour distances for each cardinal direction.
  const neighborDists = [Infinity, Infinity, Infinity, Infinity];
  let bestDir = 0;
  let bestDist = Infinity;

  for (let d = 0; d < DIRECTIONS.length; d++) {
    const [dx, dy] = DIRECTIONS[d];
    const dist = getDistance(distanceMap, cellX + dx, cellY + dy);
    neighborDists[d] = dist;
    if (dist >= 0 && dist < bestDist) {
      bestDist = dist;
      bestDir = d;
    }
  }

  // Compass scalar: bestDirection * COMPASS_STEP (range [0, 0.75]).
  vision[0] = bestDir * COMPASS_STEP;

  // Openness: 1.0 for best, scaled for others, 0 for walls/unreachable.
  if (bestDist < Infinity) {
    for (let d = 0; d < DIRECTIONS.length; d++) {
      const dist = neighborDists[d];
      if (dist >= 0) {
        vision[1 + d] = dist === bestDist ? 1.0 : bestDist / dist;
      }
      // else: walls/unreachable → 0 (already initialized to 0)
    }
  }

  // Progress delta: 0.5 + clip(prevDist - curDist, -2, 2) / 4.
  const curDist = getDistance(distanceMap, cellX, cellY);
  if (
    previousDistance !== undefined &&
    Number.isFinite(previousDistance) &&
    previousDistance >= 0 &&
    Number.isFinite(curDist) &&
    curDist >= 0
  ) {
    const delta = previousDistance - curDist;
    const clipped = Math.max(-PROGRESS_CLIP, Math.min(PROGRESS_CLIP, delta));
    vision[5] = PROGRESS_NEUTRAL + clipped / PROGRESS_SCALE;
  } else {
    vision[5] = PROGRESS_NEUTRAL;
  }

  return vision;
}
