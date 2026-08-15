/**
 * @module enemy-navigation.utils
 *
 * Vision and sensor executors for Neatenstein enemy navigation.
 *
 * Extracts the vision-vector builder, ammo-pickup ranking, visible-enemy
 * detection, and full sensor extraction pipeline from the main navigation
 * module. BFS distance-map construction and queue-buffer pooling stay in
 * the main module to preserve the `queueBuffer` coupling (AC-060).
 */

import {
  createCollisionMap,
  type CollisionMap,
} from '../browser-entry/renderer/map';
import {
  castRayDDAFromFlatMap,
  hasLineOfSight,
} from '../browser-entry/renderer/raycast';
import { NEATENSTEIN_RENDER_DISTANCE_CAP } from '../browser-entry/renderer/framebuffer';
import type {
  AmmoPickupState,
  GameState,
  EnemyState,
} from '../browser-entry/host/game/types';
import {
  NEATENSTEIN_MAIN_NEAT_INPUTS,
  NEATENSTEIN_LOW_AMMO_RATIO,
  NEATENSTEIN_AMMO_PICKUP_START_INDEX,
  NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT,
} from '../browser-entry/harness/neat-io-config';
import {
  DIRECTIONS,
  buildEnemyDistanceMap,
  getDistance,
  type DistanceMap,
} from './enemy-navigation';
import {
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
  SENSOR_INDEX_ENEMY_VISIBLE,
  SENSOR_INDEX_ENEMY_IN_FIRING_ARC,
  SENSOR_INDEX_LAST_SHOT_HIT,
  SENSOR_INDEX_LOW_AMMO_GATE,
} from './enemy-navigation.constants';

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
 * Rank active ammo pickups by a pre-built BFS distance map.
 *
 * Internal helper shared by {@link findNearestAmmoPickups} and
 * {@link extractSensors} so the BFS distance map is built only once per
 * sensor extraction. Pickups inside an unreachable cell fall back to
 * Euclidean distance from the player.
 *
 * @param gameState - Current deterministic game-state snapshot.
 * @param distanceMap - BFS distance map built from the player's cell.
 * @param maxPickups - Maximum number of pickups to return (default 3).
 * @returns A readonly array of the nearest active ammo pickups in ascending
 *   distance order.
 */
export function findNearestAmmoPickupsWithMap(
  gameState: GameState,
  distanceMap: DistanceMap,
  maxPickups = 3,
): readonly AmmoPickupState[] {
  const p = gameState.player;
  const pickups = gameState.ammoPickups ?? [];
  const ranked = pickups
    .filter((pickup) => pickup.active === true)
    .map((pickup) => {
      const cellX = Math.floor(pickup.position.x);
      const cellY = Math.floor(pickup.position.y);
      const pathDistance = getDistance(distanceMap, cellX, cellY);
      const dx = pickup.position.x - p.position.x;
      const dy = pickup.position.y - p.position.y;
      const distance = pathDistance >= 0 ? pathDistance : Math.hypot(dx, dy);
      return { pickup, distance };
    })
    .toSorted((a, b) => a.distance - b.distance)
    .slice(0, maxPickups)
    .map(({ pickup }) => pickup);

  return ranked;
}

/**
 * Find the nearest active ammo pickups by BFS distance from the player.
 *
 * Builds a BFS distance map from the player's current grid cell and looks
 * up the path distance to every active pickup. Pickups with a negative
 * distance (unreachable or inside a wall cell) fall back to Euclidean
 * distance so the NEAT sensors still receive a useful ranking signal even
 * when walls temporarily block the computed path.
 *
 * The result is capped at `maxPickups` (default 3) and sorted by distance
 * ascending. Missing active pickups simply produce a shorter array; callers
 * zero-fill the corresponding sensor slots.
 *
 * Tie-break rule: when two pickups have the same computed distance, the
 * stable `Array.prototype.toSorted` preserves the order of
 * `gameState.ammoPickups` (insertion / kill order), which is deterministic
 * for a given replay seed.
 *
 * @param gameState - Current deterministic game-state snapshot.
 * @param flatMap - Row-major wall map (`Uint8Array`, non-zero = wall).
 * @param mapSize - Width and height of the square grid.
 * @param collisionMap - Map queried for solid cells by the BFS builder.
 *   Defaults to a collision map derived from `flatMap` when omitted, so callers
 *   that only have the wall map can still obtain path-ranked pickups.
 * @param maxPickups - Maximum number of pickups to return (default 3).
 * @returns A readonly array of the nearest active ammo pickups in ascending
 *   distance order.
 *
 * @see AC-P2S1-001
 */
export function findNearestAmmoPickups(
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
  collisionMap: CollisionMap = createCollisionMap(flatMap, mapSize),
  maxPickups = 3,
): readonly AmmoPickupState[] {
  const p = gameState.player;
  const playerCellX = Math.floor(p.position.x);
  const playerCellY = Math.floor(p.position.y);

  const distanceMap = buildEnemyDistanceMap(
    collisionMap,
    playerCellX,
    playerCellY,
    mapSize,
  );

  return findNearestAmmoPickupsWithMap(gameState, distanceMap, maxPickups);
}

/**
 * Find the nearest active enemy that is within vision range and has a clear
 * line of sight from the player's position.
 *
 * Filters active enemies by:
 * 1. Euclidean distance ≤ `VISION_RANGE_CELLS` (15 cells)
 * 2. Line-of-sight via `hasLineOfSight` (no wall between player and enemy)
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
 * Extract a 22-element sensor observation vector from the current game state
 * for NEAT network activation.
 *
 * Sensor layout (indices 0–21), all values normalized to [0, 1]:
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
 * - [15] ammo pickup 1 bearing (`(bearing + π) / (2π)`, [0, 1])
 * - [16] ammo pickup 1 distance (`dist / mapSize`, capped at 1)
 * - [17] ammo pickup 2 bearing (`(bearing + π) / (2π)`, [0, 1])
 * - [18] ammo pickup 2 distance (`dist / mapSize`, capped at 1)
 * - [19] ammo pickup 3 bearing (`(bearing + π) / (2π)`, [0, 1])
 * - [20] ammo pickup 3 distance (`dist / mapSize`, capped at 1)
 * - [21] lowAmmoGate — binary (1 if `ammo / maxAmmo < NEATENSTEIN_LOW_AMMO_RATIO`)
 *
 * Enemy sensors [5]–[7] are zeroed when no enemy is within vision range or
 * when a wall blocks line-of-sight. Ammo-pickup sensors [15]–[20] are zeroed
 * when fewer than three active pickups exist; distances use path distance when
 * available and Euclidean fallback otherwise. Wall raycast distances use
 * `castRayDDAFromFlatMap` and are normalized by
 * `NEATENSTEIN_RENDER_DISTANCE_CAP`.
 *
 * @param gameState - Current deterministic game-state snapshot.
 * @param flatMap - Row-major wall map (`Uint8Array`, non-zero = wall).
 * @param mapSize - Width and height of the square grid.
 * @param collisionMap - Map queried for solid cells by the BFS builder; used
 *   to compute path distances to ammo pickups. Defaults to a collision map
 *   derived from `flatMap` when omitted.
 * @returns A 22-element observation vector for `Network.activate(sensors)`.
 *
 * @see AC-P3S1b-001, AC-P3S1b-002, AC-P3S1b-003, AC-P2S1-001
 */
export function extractSensors(
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
  collisionMap: CollisionMap = createCollisionMap(flatMap, mapSize),
): number[] {
  const sensors = new Array<number>(NEATENSTEIN_MAIN_NEAT_INPUTS).fill(0);
  const p = gameState.player;
  const twoPi = 2 * Math.PI;

  // Player sensors (5) — normalized to [0, 1].
  sensors[SENSOR_INDEX_PLAYER_HEALTH] =
    p.maxHealth > 0 ? Math.min(1, Math.max(0, p.health / p.maxHealth)) : 0;
  sensors[SENSOR_INDEX_PLAYER_AMMO] = p.maxAmmo > 0 ? Math.min(1, Math.max(0, p.ammo / p.maxAmmo)) : 0;
  sensors[SENSOR_INDEX_PLAYER_LOOK_ANGLE] = (((p.angleRad % twoPi) + twoPi) % twoPi) / twoPi;
  sensors[SENSOR_INDEX_PLAYER_POS_X] = Math.min(1, Math.max(0, p.position.x / mapSize));
  sensors[SENSOR_INDEX_PLAYER_POS_Y] = Math.min(1, Math.max(0, p.position.y / mapSize));

  // Nearest visible enemy sensors (3) — zeroed if no visible enemy.
  const visibleEnemy = findNearestVisibleEnemy(gameState, flatMap, mapSize);
  if (visibleEnemy !== null) {
    const ndx = visibleEnemy.position.x - p.position.x;
    const ndy = visibleEnemy.position.y - p.position.y;
    const dist = Math.hypot(ndx, ndy);
    const bearing = Math.atan2(ndy, ndx) - p.angleRad;
    const normalizedBearing = Math.atan2(Math.sin(bearing), Math.cos(bearing));

    sensors[SENSOR_INDEX_ENEMY_BEARING] = (normalizedBearing + Math.PI) / twoPi;
    sensors[SENSOR_INDEX_ENEMY_DISTANCE] = Math.min(1, dist / VISION_RANGE_CELLS);
    const enemyMax = visibleEnemy.maxHealth ?? 100;
    sensors[SENSOR_INDEX_ENEMY_HEALTH] =
      enemyMax > 0
        ? Math.min(1, Math.max(0, visibleEnemy.health / enemyMax))
        : 0;

    sensors[SENSOR_INDEX_ENEMY_VISIBLE] = 1;
    sensors[SENSOR_INDEX_ENEMY_IN_FIRING_ARC] = Math.abs(normalizedBearing) <= FIRING_ARC_HALF_ANGLE ? 1 : 0;
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
    sensors[SENSOR_INDEX_WALL_NORTH + i] = Number.isFinite(hit.perpWallDist)
      ? Math.min(
          1,
          Math.max(0, hit.perpWallDist / NEATENSTEIN_RENDER_DISTANCE_CAP),
        )
      : 1;
  }

  // lastShotHit — read from gameState (field added in P3S1-clear-champions).
  const lastShotHit = (gameState as GameState & { lastShotHit?: boolean })
    .lastShotHit;
  sensors[SENSOR_INDEX_LAST_SHOT_HIT] = lastShotHit === true ? 1 : 0;

  // P2S1 — Ammo-pickup awareness sensors (7), zeroed when pickups are absent.
  // Build one BFS distance map from the player and use it for both ranking
  // the three nearest active pickups and computing their path-distance sensors.
  const playerCellX = Math.floor(p.position.x);
  const playerCellY = Math.floor(p.position.y);
  const pickupDistanceMap = buildEnemyDistanceMap(
    collisionMap,
    playerCellX,
    playerCellY,
    mapSize,
  );
  const maxPickupPairs = Math.floor(
    (NEATENSTEIN_AMMO_PICKUP_SENSOR_COUNT - 1) / 2,
  );
  const nearestPickups = findNearestAmmoPickupsWithMap(
    gameState,
    pickupDistanceMap,
    maxPickupPairs,
  );
  for (let i = 0; i < nearestPickups.length; i++) {
    const pickup = nearestPickups[i];
    const dx = pickup.position.x - p.position.x;
    const dy = pickup.position.y - p.position.y;
    const euclidean = Math.hypot(dx, dy);
    const pickupCellX = Math.floor(pickup.position.x);
    const pickupCellY = Math.floor(pickup.position.y);
    const pathDistance = getDistance(
      pickupDistanceMap,
      pickupCellX,
      pickupCellY,
    );
    const dist = pathDistance >= 0 ? pathDistance : euclidean;
    const bearing = Math.atan2(dy, dx) - p.angleRad;
    const normalizedBearing = Math.atan2(Math.sin(bearing), Math.cos(bearing));

    sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX + i * 2] =
      (normalizedBearing + Math.PI) / twoPi;
    sensors[NEATENSTEIN_AMMO_PICKUP_START_INDEX + 1 + i * 2] = Math.min(
      1,
      dist / mapSize,
    );
  }

  // Low-ammo gate — binary urgency signal.
  sensors[SENSOR_INDEX_LOW_AMMO_GATE] =
    p.maxAmmo > 0 && p.ammo / p.maxAmmo < NEATENSTEIN_LOW_AMMO_RATIO ? 1 : 0;

  return sensors;
}
