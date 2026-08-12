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
import {
  castRayDDAFromFlatMap,
  hasLineOfSight,
} from '../browser-entry/renderer/raycast';
import { NEATENSTEIN_RENDER_DISTANCE_CAP } from '../browser-entry/renderer/framebuffer';
import type { GameState, EnemyState } from '../browser-entry/host/game/types';

/** Sentinel value stored in the distance buffer for wall cells. */
const WALL_VALUE = -2;

/** Sentinel value stored in the distance buffer for unreachable cells. */
const UNREACHABLE_VALUE = -1;

/**
 * Maximum vision range in grid cells for enemy detection.
 *
 * Enemies beyond this Euclidean distance from the player are invisible —
 * their sensor values are zeroed. Set to 15 (half the bolt max range of 30)
 * to force active exploration without making the hero omniscient.
 *
 * @see AC-P3S1b-003
 */
const VISION_RANGE_CELLS = 15;

/**
 * Half-angle of the firing arc in radians (30°).
 *
 * An enemy is considered "in the firing arc" when the absolute relative
 * bearing is within this angle of the player's facing direction. Matches
 * the fallback AI's `NEATENSTEIN_FALLBACK_FIRE_ARC`.
 */
const FIRING_ARC_HALF_ANGLE = Math.PI / 6;

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
 * Find the nearest active enemy that is within vision range and has a clear
 * line of sight from the player's position.
 *
 * Filters active enemies by:
 * 1. Euclidean distance ≤ {@link VISION_RANGE_CELLS}
 * 2. Line-of-sight via {@link hasLineOfSight} (no wall between player and enemy)
 *
 * Returns the nearest enemy that satisfies both conditions, or `null` when
 * no enemy is visible.
 *
 * @param gameState - Current deterministic game-state snapshot.
 * @param flatMap - Row-major wall map (`Uint8Array`, non-zero = wall).
 * @param mapSize - Width and height of the square grid.
 * @returns The nearest visible enemy, or `null`.
 *
 * @see AC-P3S1b-005
 */
export function findNearestVisibleEnemy(
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
): EnemyState | null {
  const p = gameState.player;

  let nearest: EnemyState | null = null;
  let nearestDist = Infinity;

  for (const e of gameState.enemies) {
    if (e.active === false) continue;
    const dx = e.position.x - p.position.x;
    const dy = e.position.y - p.position.y;
    const dist = Math.hypot(dx, dy);

    if (dist > VISION_RANGE_CELLS) continue;
    if (dist >= nearestDist) continue;
    if (!hasLineOfSight(flatMap, mapSize, p.position, e.position)) continue;

    nearest = e;
    nearestDist = dist;
  }

  return nearest;
}

/**
 * Extract a 15-element sensor observation vector from the current game state
 * for NEAT network activation.
 *
 * Sensor layout (indices 0–14), all values normalized to [0, 1]:
 * - [0] player health ratio (`health / maxHealth`, clamped to [0, 1])
 * - [1] player ammo ratio (`ammo / maxAmmo`, clamped to [0, 1])
 * - [2] player look angle (`angleRad / (2π)` mapped to [0, 1])
 * - [3] player position X (`position.x / mapSize`, clamped to [0, 1])
 * - [4] player position Y (`position.y / mapSize`, clamped to [0, 1])
 * - [5] nearest visible enemy bearing (`(bearing + π) / (2π)`, [0, 1])
 * - [6] nearest visible enemy distance (`dist / VISION_RANGE_CELLS`, [0, 1])
 * - [7] nearest visible enemy health ratio (`health / maxHealth`, [0, 1])
 * - [8] wall raycast distance North (`dist / renderCap`, [0, 1])
 * - [9] wall raycast distance East (`dist / renderCap`, [0, 1])
 * - [10] wall raycast distance South (`dist / renderCap`, [0, 1])
 * - [11] wall raycast distance West (`dist / renderCap`, [0, 1])
 * - [12] enemyVisible — binary (1 if a visible enemy exists, 0 otherwise)
 * - [13] enemyInFiringArc — binary (1 if visible enemy bearing ≤ 30°, 0 otherwise)
 * - [14] lastShotHit — binary (1 if the previous shot hit an enemy, 0 otherwise)
 *
 * Enemy sensors [5]–[7] are zeroed when no enemy is within
 * {@link VISION_RANGE_CELLS} or when a wall blocks line-of-sight. Wall
 * raycast distances use {@link castRayDDAFromFlatMap} and are normalized
 * by {@link NEATENSTEIN_RENDER_DISTANCE_CAP}.
 *
 * @param gameState - Current deterministic game-state snapshot.
 * @param flatMap - Row-major wall map (`Uint8Array`, non-zero = wall).
 * @param mapSize - Width and height of the square grid.
 * @returns A 15-element observation vector for `Network.activate(sensors)`.
 *
 * @example
 * ```ts
 * const sensors = extractSensors(gameState, wallMap, NEATENSTEIN_MAP_SIZE);
 * const outputs = network.activate(sensors);
 * ```
 *
 * @see AC-P3S1b-001, AC-P3S1b-002, AC-P3S1b-003
 */
export function extractSensors(
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
): number[] {
  const sensors = new Array<number>(15).fill(0);
  const p = gameState.player;
  const twoPi = 2 * Math.PI;

  // Player sensors (5) — normalized to [0, 1].
  sensors[0] =
    p.maxHealth > 0 ? Math.min(1, Math.max(0, p.health / p.maxHealth)) : 0;
  sensors[1] = p.maxAmmo > 0 ? Math.min(1, Math.max(0, p.ammo / p.maxAmmo)) : 0;
  sensors[2] = (((p.angleRad % twoPi) + twoPi) % twoPi) / twoPi;
  sensors[3] = Math.min(1, Math.max(0, p.position.x / mapSize));
  sensors[4] = Math.min(1, Math.max(0, p.position.y / mapSize));

  // Nearest visible enemy sensors (3) — zeroed if no visible enemy.
  const visibleEnemy = findNearestVisibleEnemy(gameState, flatMap, mapSize);
  if (visibleEnemy !== null) {
    const ndx = visibleEnemy.position.x - p.position.x;
    const ndy = visibleEnemy.position.y - p.position.y;
    const dist = Math.hypot(ndx, ndy);
    const bearing = Math.atan2(ndy, ndx) - p.angleRad;
    const normalizedBearing = Math.atan2(Math.sin(bearing), Math.cos(bearing));

    sensors[5] = (normalizedBearing + Math.PI) / twoPi;
    sensors[6] = Math.min(1, dist / VISION_RANGE_CELLS);
    const enemyMax = visibleEnemy.maxHealth ?? 100;
    sensors[7] =
      enemyMax > 0
        ? Math.min(1, Math.max(0, visibleEnemy.health / enemyMax))
        : 0;

    sensors[12] = 1;
    sensors[13] = Math.abs(normalizedBearing) <= FIRING_ARC_HALF_ANGLE ? 1 : 0;
  }

  // Wall raycasts (4) — N, E, S, W cardinal directions, normalized to [0, 1].
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
    sensors[8 + i] = Number.isFinite(hit.perpWallDist)
      ? Math.min(
          1,
          Math.max(0, hit.perpWallDist / NEATENSTEIN_RENDER_DISTANCE_CAP),
        )
      : 1;
  }

  // lastShotHit — read from gameState (field added in P3S1-clear-champions).
  const lastShotHit = (gameState as GameState & { lastShotHit?: boolean })
    .lastShotHit;
  sensors[14] = lastShotHit === true ? 1 : 0;

  return sensors;
}
