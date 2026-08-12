/**
 * Deterministic enemy wave spawning for the Neatenstein host-side simulation.
 *
 * Enemies spawn as a continuous trickle: at most one new enemy per simulation
 * tick, and never more than {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT} active at
 * once. All transitions return immutable state snapshots so replay stays
 * deterministic.
 *
 * @module
 */

import {
  NEATENSTEIN_ENEMY_MAX_CONCURRENT,
  NEATENSTEIN_ENEMY_MAX_HEALTH,
  NEATENSTEIN_MAP_SIZE,
  NEATENSTEIN_SPAWN_CENTER_X,
  NEATENSTEIN_SPAWN_CENTER_Y,
} from './constants';
import type { CollisionMap } from '../../renderer/map';
import { createGameRng } from './state';
import type { EnemyState, GameState } from './types';

/** Result of a single spawn tick. */
export interface SpawnWaveTickResult {
  /** Number of enemies added this tick (always 0 or 1). */
  spawnedThisTick: number;
  /** New state snapshot with any spawned enemy included. */
  state: GameState;
}

/**
 * Edge directions used for deterministic enemy wave placement.
 *
 * Each wave spawns one enemy at each of these eight map edges, cycling
 * clockwise from north through the intercardinal corners.
 */
const SPAWN_EDGE_ORDER: Array<{ direction: string; dx: number; dy: number }> = [
  { direction: 'N', dx: 0, dy: 1 },
  { direction: 'NW', dx: 1, dy: 1 },
  { direction: 'W', dx: 1, dy: 0 },
  { direction: 'SW', dx: 1, dy: -1 },
  { direction: 'S', dx: 0, dy: -1 },
  { direction: 'SE', dx: -1, dy: -1 },
  { direction: 'E', dx: -1, dy: 0 },
  { direction: 'NE', dx: -1, dy: 1 },
];

/** Return a safe edge or center spawn position for the given direction. */
function resolveEdgeSpawn(
  directionIndex: number,
  collisionMap?: CollisionMap,
): { x: number; y: number } {
  const edge = SPAWN_EDGE_ORDER[directionIndex % SPAWN_EDGE_ORDER.length];
  const edgeOffset = 0.5;
  const centerX = NEATENSTEIN_SPAWN_CENTER_X;
  const centerY = NEATENSTEIN_SPAWN_CENTER_Y;
  const max = NEATENSTEIN_MAP_SIZE - edgeOffset;

  let x: number;
  let y: number;

  switch (edge.direction) {
    case 'N':
      x = centerX;
      y = edgeOffset;
      break;
    case 'S':
      x = centerX;
      y = max;
      break;
    case 'W':
      x = edgeOffset;
      y = centerY;
      break;
    case 'E':
      x = max;
      y = centerY;
      break;
    case 'NW':
      x = edgeOffset;
      y = edgeOffset;
      break;
    case 'NE':
      x = max;
      y = edgeOffset;
      break;
    case 'SW':
      x = edgeOffset;
      y = max;
      break;
    case 'SE':
    default:
      x = max;
      y = max;
      break;
  }

  if (!collisionMap) {
    return { x, y };
  }

  // Scan inward along the edge normal until an open cell is found. If the
  // whole edge is solid, fall back to the guaranteed-open spawn center.
  // The cell under test must match the position that will be returned, so we
  // check the current (x, y) before incrementing — not a step-offset copy.
  const limit = Math.floor(NEATENSTEIN_MAP_SIZE / 2);
  for (let step = 0; step <= limit; step += 1) {
    const cx = Math.floor(x);
    const cy = Math.floor(y);

    if (
      cx >= 0 &&
      cy >= 0 &&
      cx < NEATENSTEIN_MAP_SIZE &&
      cy < NEATENSTEIN_MAP_SIZE &&
      !collisionMap.isSolid(cx, cy)
    ) {
      return { x, y };
    }

    x += edge.dx;
    y += edge.dy;
  }

  return { x: centerX, y: centerY };
}

/** Return whether every enemy in the roster has been killed or deactivated. */
export function allEnemiesCleared(enemies: EnemyState[]): boolean {
  return (
    enemies.length === 0 ||
    enemies.every((enemy) => (enemy.health ?? 0) <= 0 || enemy.active === false)
  );
}

/**
 * Advance enemy spawning by one fixed tick.
 *
 * Enemies spawn as a continuous trickle: at most one new enemy per tick, and
 * never more than {@link NEATENSTEIN_ENEMY_MAX_CONCURRENT} alive at once. The
 * spawner cycles through the eight cardinal/intercardinal map edges so each
 * new enemy arrives from a different direction. Dead or inactive enemies stay
 * in the returned array — the display worker syncs enemy positions by array
 * index, so removing them would break index alignment and cause new enemies
 * to inherit the positions of dead ones. The worker's de-rez system removes
 * dead enemies from the display array after the de-rez animation completes
 * (700ms).
 *
 * @param state - Snapshot before the spawn tick.
 * @param _dtMs - Elapsed simulation time in milliseconds (reserved for future
 *   spawn-rate modulation).
 * @param collisionMap - Optional collision map used to avoid spawning inside
 *   solid edge cells.
 * @returns Object reporting how many enemies spawned and the new state.
 *
 * @example
 * ```ts
 * import { NEATENSTEIN_FIXED_TIMESTEP_MS } from './constants';
 * const before = createGameState({ seed: 1 });
 * const result = spawnWaveTick(before, NEATENSTEIN_FIXED_TIMESTEP_MS);
 * expect(result.spawnedThisTick).toBeLessThanOrEqual(1);
 * ```
 */
export function spawnWaveTick(
  state: GameState,
  _dtMs: number,
  collisionMap?: CollisionMap,
): SpawnWaveTickResult {
  // Reserved for future spawn-rate modulation; the public contract accepts
  // the elapsed tick time even though the current policy is one enemy per
  // tick while the arena is below the alive-enemy cap.
  void _dtMs;

  // Count alive enemies for the concurrent limit check. Do NOT filter dead
  // enemies from the returned array — the display worker syncs enemy
  // positions by array index, so removing dead entries would break index
  // alignment and cause new enemies to inherit the positions of dead ones.
  // The worker's de-rez system removes dead enemies from the display array
  // after the de-rez animation completes (700ms).
  const aliveCount = state.enemies.filter(
    (enemy) => (enemy.health ?? 0) > 0 && enemy.active !== false,
  ).length;

  // Do not exceed the concurrent alive enemy limit.
  if (aliveCount >= NEATENSTEIN_ENEMY_MAX_CONCURRENT) {
    return {
      spawnedThisTick: 0,
      state: { ...state },
    };
  }

  // Use a dedicated monotonic spawn counter for the RNG seed so different
  // game histories (different kill counts / active enemy rosters) can never
  // produce the same deterministic spawn position.
  const rng = createGameRng(state.seed + state.spawnCount);
  // Consume one RNG value so the deterministic sequence stays stable with
  // future randomization tweaks.
  void rng();

  const directionIndex = state.spawnCount % SPAWN_EDGE_ORDER.length;
  const position = resolveEdgeSpawn(directionIndex, collisionMap);
  const enemy: EnemyState = {
    position: { ...position },
    health: NEATENSTEIN_ENEMY_MAX_HEALTH,
    maxHealth: NEATENSTEIN_ENEMY_MAX_HEALTH,
    active: true,
    controllerPosition: { ...position },
    stunTimerMs: 0,
  };

  return {
    spawnedThisTick: 1,
    state: {
      ...state,
      enemies: [...state.enemies, enemy],
      spawnCount: state.spawnCount + 1,
    },
  };
}
