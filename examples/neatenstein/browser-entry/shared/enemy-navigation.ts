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

import { type CollisionMap } from '../renderer/map';
import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { DIRECTIONS } from './enemy-controller.constants';
import { WALL_VALUE, UNREACHABLE_VALUE } from './enemy-navigation.constants';
import type { DistanceMap } from './enemy-navigation.types';

// Re-export DIRECTIONS from the single source of truth in
// enemy-controller.constants.ts to preserve the public API.
export { DIRECTIONS };

// Re-export extracted types and constants so existing imports stay valid.
export type { DistanceMap } from './enemy-navigation.types';
export {
  WALL_VALUE,
  UNREACHABLE_VALUE,
  VISION_RANGE_CELLS,
  FIRING_ARC_HALF_ANGLE,
  COMPASS_STEP,
  PROGRESS_CLIP,
  PROGRESS_SCALE,
  PROGRESS_NEUTRAL,
  SENSOR_INDEX_PLAYER_HEALTH,
  SENSOR_INDEX_PLAYER_AMMO,
  SENSOR_INDEX_PLAYER_LOOK_ANGLE,
  SENSOR_INDEX_PLAYER_POS_X,
  SENSOR_INDEX_PLAYER_POS_Y,
  SENSOR_INDEX_ENEMY_BEARING,
  SENSOR_INDEX_ENEMY_DISTANCE,
  SENSOR_INDEX_ENEMY_HEALTH,
  SENSOR_INDEX_WALL_NORTH,
  SENSOR_INDEX_WALL_EAST,
  SENSOR_INDEX_WALL_SOUTH,
  SENSOR_INDEX_WALL_WEST,
  SENSOR_INDEX_ENEMY_VISIBLE,
  SENSOR_INDEX_ENEMY_IN_FIRING_ARC,
  SENSOR_INDEX_LAST_SHOT_HIT,
  SENSOR_INDEX_LOW_AMMO_GATE,
} from './enemy-navigation.constants';

// Re-export public symbols that moved to the sibling utils file.
export {
  buildVisionVector,
  findNearestAmmoPickups,
  findNearestVisibleEnemy,
  extractSensors,
} from './enemy-navigation.utils';

/** Shared, reusable queue buffer to avoid per-call allocations. */
let queueBuffer: Int32Array = new Int32Array(0);

// --- BFS distance map cache (A2 Fix 3) ---

/** Module-level reusable Int32Array for the distance map buffer. */
let cachedDistanceMapBuffer: Int32Array = new Int32Array(
  NEATENSTEIN_MAP_SIZE * NEATENSTEIN_MAP_SIZE,
);

/** Cached map seed — invalidation key for wall-mask precomputation. */
let cachedMapSeed: number = Number.NaN;

/** Cached player cell X — invalidation key for distance map rebuild. */
let cachedPlayerCellX: number = Number.NaN;

/** Cached player cell Y — invalidation key for distance map rebuild. */
let cachedPlayerCellY: number = Number.NaN;

/** Cached collision-map reference — invalidation key for distance map rebuild. */
let cachedCollisionMap: CollisionMap | null = null;

/** Cached full distance-map object — returned to callers on cache hits. */
let cachedDistanceMap: DistanceMap | null = null;

/**
 * Return a cached BFS distance map, rebuilding only when the player crosses a
 * cell boundary, the map seed changes, or the collision map object changes.
 *
 * Allocates one module-level reusable {@link Int32Array} and passes it into
 * every {@link buildEnemyDistanceMap} call, avoiding ~57 KB of allocation per
 * tick. The cache is keyed by `(mapSeed, Math.floor(playerX),
 * Math.floor(playerY), collisionMap)` — the distance map only changes when
 * the player moves to a new grid cell or a different collision map is used.
 *
 * @param collisionMap - Map queried for solid cells.
 * @param playerX - Player world X (floored to get the grid cell).
 * @param playerY - Player world Y (floored to get the grid cell).
 * @param mapSeed - Map seed used for cache invalidation.
 * @returns The cached (or freshly rebuilt) {@link DistanceMap}.
 */
export function getOrBuildCachedDistanceMap(
  collisionMap: CollisionMap,
  playerX: number,
  playerY: number,
  mapSeed: number,
): DistanceMap {
  const playerCellX = Math.floor(playerX);
  const playerCellY = Math.floor(playerY);

  if (
    mapSeed === cachedMapSeed &&
    playerCellX === cachedPlayerCellX &&
    playerCellY === cachedPlayerCellY &&
    cachedCollisionMap === collisionMap &&
    cachedDistanceMap !== null
  ) {
    return cachedDistanceMap;
  }

  // Rebuild the distance map into the reusable buffer.
  const distanceMap = buildEnemyDistanceMap(
    collisionMap,
    playerCellX,
    playerCellY,
    NEATENSTEIN_MAP_SIZE,
    cachedDistanceMapBuffer,
  );

  cachedDistanceMapBuffer = distanceMap.distances;
  cachedDistanceMap = distanceMap;
  cachedCollisionMap = collisionMap;
  cachedMapSeed = mapSeed;
  cachedPlayerCellX = playerCellX;
  cachedPlayerCellY = playerCellY;

  return cachedDistanceMap;
}

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
