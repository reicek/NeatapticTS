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
import { castRayDDAFromFlatMap } from '../browser-entry/renderer/raycast';
import type { GameState } from '../browser-entry/host/game/types';

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
 * When `distances` is provided and has the correct length (`size * size`),
 * the function fills it in-place instead of allocating a new buffer. This
 * enables callers to pool a reusable `Int32Array` — worker-scoped for the
 * live path or per-episode for the headless evaluation path — avoiding
 * ~57 KB of allocation per tick. When `distances` is `undefined` or the
 * wrong length, a fresh `Int32Array` is allocated (backward-compatible).
 *
 * @param collisionMap - Map queried for solid cells.
 * @param goalX - Goal cell X (typically the player's grid X).
 * @param goalY - Goal cell Y (typically the player's grid Y).
 * @param size - Grid width and height.
 * @param distances - Optional pre-allocated `Int32Array` of length
 *   `size * size` to fill in-place. When omitted or incorrectly sized, a
 *   fresh buffer is allocated.
 * @returns Distance map with distances, sentinels, and metadata. The
 *   `distances` field references the same buffer passed in when one was
 *   provided and correctly sized; otherwise it references a newly allocated
 *   buffer.
 */
export function buildEnemyDistanceMap(
  collisionMap: CollisionMap,
  goalX: number,
  goalY: number,
  size: number,
  distances?: Int32Array,
): DistanceMap {
  const cellCount = size * size;
  // Reuse the caller-provided buffer when it has the correct length;
  // otherwise allocate a fresh one (backward-compatible).
  const distancesBuffer =
    distances !== undefined && distances.length === cellCount
      ? distances
      : new Int32Array(cellCount);

  // Initialize all cells to UNREACHABLE.
  for (let i = 0; i < cellCount; i++) {
    distancesBuffer[i] = UNREACHABLE_VALUE;
  }

  // Mark walls.
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (collisionMap.isSolid(x, y)) {
        distancesBuffer[y * size + x] = WALL_VALUE;
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
    distancesBuffer[goalIndex] === WALL_VALUE
  ) {
    return {
      size,
      distances: distancesBuffer,
      wallValue: WALL_VALUE,
      unreachableValue: UNREACHABLE_VALUE,
    };
  }

  distancesBuffer[goalIndex] = 0;

  // BFS from the goal outward.
  const queue = getQueueBuffer(cellCount);
  let queueHead = 0;
  let queueTail = 0;
  queue[queueTail++] = goalIndex;

  while (queueHead < queueTail) {
    const currentIndex = queue[queueHead++];
    const currentDistance = distancesBuffer[currentIndex];

    const currentY = (currentIndex / size) | 0;
    const currentX = currentIndex - currentY * size;

    for (const [dx, dy] of DIRECTIONS) {
      const nx = currentX + dx;
      const ny = currentY + dy;
      if (nx < 0 || nx >= size || ny < 0 || ny >= size) continue;

      const ni = ny * size + nx;
      if (distancesBuffer[ni] === UNREACHABLE_VALUE) {
        distancesBuffer[ni] = currentDistance + 1;
        queue[queueTail++] = ni;
      }
    }
  }

  return {
    size,
    distances: distancesBuffer,
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

/**
 * Extract a 12-element sensor observation vector from the current game state
 * for NEAT network activation.
 *
 * Sensor layout (indices 0–11):
 * - [0] player health ratio (`health / maxHealth`, clamped to [0, 1])
 * - [1] player ammo
 * - [2] player look angle in radians
 * - [3] player position X (world units)
 * - [4] player position Y (world units)
 * - [5] nearest enemy relative bearing in radians, normalized to [-π, π]
 * - [6] nearest enemy Euclidean distance (world units)
 * - [7] nearest enemy health
 * - [8] wall raycast distance — North
 * - [9] wall raycast distance — East
 * - [10] wall raycast distance — South
 * - [11] wall raycast distance — West
 *
 * When no active enemies exist, sensors [5]–[7] are 0. Wall raycast distances
 * use the existing {@link castRayDDAFromFlatMap} DDA primitive and may be
 * `Infinity` when no wall is found within the render distance cap.
 *
 * Placing this helper in `scripts/enemy-navigation.ts` (not the worker) allows
 * the evolution harness to reuse it for episode fitness evaluation in Phase 5
 * without importing worker code.
 *
 * @param gameState - Current deterministic game-state snapshot.
 * @param flatMap - Row-major wall map (`Uint8Array`, non-zero = wall).
 * @param mapSize - Width and height of the square grid.
 * @returns A 12-element observation vector for `Network.activate(sensors)`.
 *
 * @example
 * ```ts
 * const sensors = extractSensors(gameState, wallMap, NEATENSTEIN_MAP_SIZE);
 * const outputs = network.activate(sensors);
 * ```
 *
 * @see AC-065
 */
export function extractSensors(
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
): number[] {
  const sensors = new Array<number>(12).fill(0);
  const p = gameState.player;

  // Player sensors (5)
  sensors[0] = p.maxHealth > 0 ? p.health / p.maxHealth : 0;
  sensors[1] = p.ammo;
  sensors[2] = p.angleRad;
  sensors[3] = p.position.x;
  sensors[4] = p.position.y;

  // Nearest enemy sensors (3)
  const enemies = gameState.enemies.filter((e) => e.active !== false);
  if (enemies.length > 0) {
    let nearestEnemy = enemies[0];
    let nearestDist = Infinity;
    for (const e of enemies) {
      const dx = e.position.x - p.position.x;
      const dy = e.position.y - p.position.y;
      const dist = Math.hypot(dx, dy);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearestEnemy = e;
      }
    }
    const ndx = nearestEnemy.position.x - p.position.x;
    const ndy = nearestEnemy.position.y - p.position.y;
    const bearing = Math.atan2(ndy, ndx) - p.angleRad;
    // Normalize to [-π, π].
    sensors[5] = Math.atan2(Math.sin(bearing), Math.cos(bearing));
    sensors[6] = nearestDist;
    sensors[7] = nearestEnemy.health;
  }

  // Wall raycasts (4) — N, E, S, W cardinal directions.
  for (let i = 0; i < DIRECTIONS.length; i++) {
    const [dirX, dirY] = DIRECTIONS[i];
    const hit = castRayDDAFromFlatMap(
      flatMap,
      mapSize,
      p.position.x,
      p.position.y,
      dirX,
      dirY,
    );
    sensors[8 + i] = hit.perpWallDist;
  }

  return sensors;
}
