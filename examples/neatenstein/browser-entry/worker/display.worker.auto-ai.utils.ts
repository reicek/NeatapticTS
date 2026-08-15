/**
 * Auto-AI executors for the Neatenstein display worker.
 *
 * State-in/state-out executors for the champion NEAT network path and the
 * fallback hunter AI. Each executor takes an {@link AutoAiState} slice
 * (smoothing + fire-gate + fallback counters) and returns the updated slice
 * alongside the produced {@link GameTickInputSnapshot}.
 *
 * @module
 */

import {
  ENEMY_VISIBLE_SENSOR_INDEX,
  NEATENSTEIN_FALLBACK_TURN_RATE,
  NEATENSTEIN_MOVE_ACCEL_PER_TICK,
  NEATENSTEIN_MOVE_DECEL_PER_TICK,
  applyFireGate,
  networkOutputToTickInput,
  smoothCommand,
} from '../harness/neat-io-config';
import {
  NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD,
  NEATENSTEIN_EXPLORATION_WALL_BOUNCE_LOOKAHEAD_CELLS,
  NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS,
  NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS,
} from '../host/game/constants';
import { NEATENSTEIN_MAP_SIZE } from '../constants';
import { castRayDDAFromFlatMap } from '../renderer/raycast';
import {
  extractSensors,
  findNearestVisibleEnemy,
} from '../../scripts/enemy-navigation';
import {
  FALLBACK_FIRE_RANGE,
  FALLBACK_FIRE_ANGLE,
} from './display.worker.constants';
import type { AutoAiState } from './display.worker.types';
import type { CollisionMap } from '../renderer/map';
import type { EnemyState, GameState } from '../host/game/types';
import type { GameTickInputSnapshot } from '../host/game/tick';
import type { Network } from 'neataptic';

/**
 * Check whether any enemies in the roster are alive (health > 0 and active).
 *
 * Dead/inactive enemies are kept in the array for index alignment and are
 * intentionally ignored.
 *
 * @param enemies - Enemy roster (may be undefined or null).
 * @returns `true` when at least one enemy has health > 0 and is active.
 */
export function hasAliveEnemies(
  enemies: EnemyState[] | undefined | null,
): boolean {
  const list =
    enemies ??
    /* istanbul ignore next -- defensive fallback for malformed gameState.enemies */ [];
  return list.some((e) => e.health > 0 && e.active !== false);
}

/**
 * Build a {@link GameTickInputSnapshot} from the champion NEAT network.
 *
 * Extracts real sensors from the current game state, activates the champion
 * network, and maps the output vector to a tick input via
 * {@link networkOutputToTickInput}. Applies per-tick accel/decel smoothing
 * to move/look and passes the fire output through the soft fire gate.
 *
 * @param network - The champion main-agent network from the arms-race evaluation.
 * @param gameState - Current deterministic game state (non-null).
 * @param flatMap - Row-major wall map for raycast sensors.
 * @param mapSize - Width and height of the square grid.
 * @param collisionMap - Map queried for solid cells; used for ammo-pickup
 *   path-distance sensors.
 * @param ai - Mutable auto-AI state slice (smoothing + fire-gate).
 * @returns The produced tick input and the updated auto-AI state.
 * @see AC-065, AC-066, AC-P2S1-001
 */
export function buildAutoTickInput(
  network: Network,
  gameState: GameState,
  flatMap: Uint8Array,
  mapSize: number,
  collisionMap: CollisionMap,
  ai: AutoAiState,
): { tickInput: GameTickInputSnapshot; ai: AutoAiState } {
  const sensors = extractSensors(gameState, flatMap, mapSize, collisionMap);
  const raw = network.activate(sensors);

  // P5S1: Apply soft fire gate with hysteresis on enemyVisible sensor.
  // When no enemy is visible (sensor below 0.15 floor), fire is suppressed
  // regardless of network output.  Hysteresis (0.15/0.18) prevents rapid
  // on/off oscillation at the vision boundary.
  // @see AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003
  // istanbul ignore next -- defensive fallback; ENEMY_VISIBLE_SENSOR_INDEX
  // is always populated by the extractSensors vector above.
  const enemyVisible = sensors[ENEMY_VISIBLE_SENSOR_INDEX] ?? 0;

  const tick = networkOutputToTickInput(raw, {
    state: ai.fireGateState,
    enemyVisible,
  });

  // P1S1: Apply per-tick accel/decel smoothing to move/look so the AI
  // does not snap from 0 to max in a single tick.
  const smoothedMoveX = smoothCommand(ai.smoothedMoveX, tick.move.x);
  const smoothedMoveY = smoothCommand(ai.smoothedMoveY, tick.move.y);
  const smoothedLookDelta = smoothCommand(
    ai.smoothedLookDelta,
    tick.lookDelta,
    NEATENSTEIN_MOVE_ACCEL_PER_TICK,
    NEATENSTEIN_MOVE_DECEL_PER_TICK,
  );

  return {
    tickInput: {
      move: { x: smoothedMoveX, y: smoothedMoveY },
      lookDelta: smoothedLookDelta,
      fire: tick.fire,
      dash: tick.dash,
    },
    ai: {
      ...ai,
      smoothedMoveX,
      smoothedMoveY,
      smoothedLookDelta,
    },
  };
}

/**
 * Build a fallback auto-mode tick input when no champion network is available.
 *
 * **P6S2 — Chicken-and-egg deadlock resolution.**
 *
 * `championMainNetwork` starts as `null` and is only set after the first
 * wave-clear via the eval worker delegation.  But clearing the first
 * wave requires an active player, and the auto-mode player is paralyzed
 * (zero input) without a champion network — a circular dependency.
 *
 * This fallback breaks the deadlock by providing a deterministic hunter AI:
 *
 * - **Wall-bounce exploration** — when no enemy is visible, move forward in
 *   the current direction and only turn when a wall is detected within
 *   {@link NEATENSTEIN_EXPLORATION_WALL_BOUNCE_LOOKAHEAD_CELLS}.  The hunter
 *   casts angled rays at ±{@link NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD},
 *   picks the most open direction, and turns toward it at the capped rate.
 * - **Distance-aware kiting** — when an enemy is visible, turn toward its
 *   bearing and choose movement based on the 15–20 cell band defined by
 *   {@link NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS} and
 *   {@link NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS}.
 * - **Fire through the shared soft fire gate** — computes a raw fire intent
 *   when the visible enemy is within ±{@link FALLBACK_FIRE_ANGLE} and
 *   the cooldown interval has elapsed, then passes it through
 *   {@link applyFireGate} with the auto-AI state's `fireGateState` so the
 *   fallback respects the same hysteresis gate as the champion-network path.
 *
 * Once the first wave is cleared and the arms-race evaluation runs, the
 * evolved champion network replaces this fallback.
 *
 * @param gameState - Current deterministic game state (player + enemies).
 * @param wallMap - Flat wall map for raycast sensors (may be null before init).
 * @param ai - Mutable auto-AI state slice (smoothing + fire-gate + counters).
 * @returns The produced tick input and the updated auto-AI state.
 * @see AC-060a, AC-P5S1a-001, AC-P5S1a-002, AC-P5S1a-003, AC-P9S2-001
 */
export function buildFallbackAutoTickInput(
  gameState: GameState,
  wallMap: Uint8Array | null,
  ai: AutoAiState,
): { tickInput: GameTickInputSnapshot; ai: AutoAiState } {
  const fallbackTickCounter = ai.fallbackTickCounter + 1;

  let lookDelta = 0;
  let rawFire = 0;
  let moveY = 1;

  // Use the same vision query as the network path: nearest active enemy within
  // VISION_RANGE_CELLS and with a clear line of sight.
  const visibleEnemy =
    wallMap !== null
      ? findNearestVisibleEnemy(gameState, wallMap, NEATENSTEIN_MAP_SIZE)
      : null;
  const enemyVisible = visibleEnemy !== null ? 1 : 0;

  const px = gameState.player.position.x;
  const py = gameState.player.position.y;
  const pa = gameState.player.angleRad;

  if (visibleEnemy !== null) {
    const dx = visibleEnemy.position.x - px;
    const dy = visibleEnemy.position.y - py;
    const bearing = Math.atan2(dy, dx);
    const distanceCells = Math.hypot(dx, dy);

    // Normalize angle difference to [-π, π].
    let angleDiff = bearing - pa;
    while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
    while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

    // Turn toward the enemy, capped by the fallback turn rate.
    lookDelta = Math.max(
      -NEATENSTEIN_FALLBACK_TURN_RATE,
      Math.min(NEATENSTEIN_FALLBACK_TURN_RATE, angleDiff),
    );

    // Kiting: backpedal if too close, hold in the 15–20 cell band, approach
    // if the enemy is beyond the preferred band.
    if (distanceCells < NEATENSTEIN_KITING_BACKPEDAL_DISTANCE_CELLS) {
      moveY = -1;
    } else if (distanceCells < NEATENSTEIN_KITING_APPROACH_DISTANCE_CELLS) {
      moveY = 0;
    } else {
      moveY = 1;
    }

    // Raw fire intent: only when the visible enemy is within the forward arc
    // AND the cooldown interval has elapsed.
    if (
      Math.abs(angleDiff) <= FALLBACK_FIRE_ANGLE &&
      fallbackTickCounter % FALLBACK_FIRE_RANGE === 0
    ) {
      rawFire = 1;
    }
  } else if (wallMap !== null) {
    // Wall-bounce exploration: move forward, but if a wall is imminent, turn
    // toward the more open of the two angled lookahead directions.
    const forwardHit = castRayDDAFromFlatMap(
      wallMap,
      NEATENSTEIN_MAP_SIZE,
      px,
      py,
      Math.cos(pa),
      Math.sin(pa),
    );

    const lookahead = NEATENSTEIN_EXPLORATION_WALL_BOUNCE_LOOKAHEAD_CELLS;
    if (
      Number.isFinite(forwardHit.perpWallDist) &&
      forwardHit.perpWallDist <= lookahead
    ) {
      const bounce = NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD;
      const candidates = [
        {
          offset: bounce,
          hit: castRayDDAFromFlatMap(
            wallMap,
            NEATENSTEIN_MAP_SIZE,
            px,
            py,
            Math.cos(pa + bounce),
            Math.sin(pa + bounce),
          ),
        },
        {
          offset: -bounce,
          hit: castRayDDAFromFlatMap(
            wallMap,
            NEATENSTEIN_MAP_SIZE,
            px,
            py,
            Math.cos(pa - bounce),
            Math.sin(pa - bounce),
          ),
        },
      ];

      const best = candidates.reduce((chosen, current) =>
        current.hit.perpWallDist > chosen.hit.perpWallDist ? current : chosen,
      );

      let angleDiff = best.offset;
      /* istanbul ignore next -- defensive angle-normalisation loop; bounce
       * offsets are bounded by NEATENSTEIN_EXPLORATION_BOUNCE_ANGLE_RAD, so
       * this wrap is unreachable with current constants but kept for robustness
       * if offset sources change in the future. */
      while (angleDiff > Math.PI) angleDiff -= 2 * Math.PI;
      /* istanbul ignore next -- defensive angle-normalisation loop (see above). */
      while (angleDiff < -Math.PI) angleDiff += 2 * Math.PI;

      lookDelta = Math.max(
        -NEATENSTEIN_FALLBACK_TURN_RATE,
        Math.min(NEATENSTEIN_FALLBACK_TURN_RATE, angleDiff),
      );
    }
  }

  // P1S1: Apply per-tick accel/decel smoothing to move/look so the AI
  // does not snap from 0 to max in a single tick.
  const smoothedMoveX = smoothCommand(ai.smoothedMoveX, 0);
  const smoothedMoveY = smoothCommand(ai.smoothedMoveY, moveY);
  const smoothedLookDelta = smoothCommand(
    ai.smoothedLookDelta,
    lookDelta,
    NEATENSTEIN_MOVE_ACCEL_PER_TICK,
    NEATENSTEIN_MOVE_DECEL_PER_TICK,
  );

  // Apply the same hysteresis fire gate used by buildAutoTickInput.
  const fire = applyFireGate(ai.fireGateState, enemyVisible, rawFire);

  const tickInput: GameTickInputSnapshot = {
    move: { x: smoothedMoveX, y: smoothedMoveY },
    lookDelta: smoothedLookDelta,
    fire,
    dash: false,
  };

  return {
    tickInput,
    ai: {
      ...ai,
      fallbackTickCounter,
      smoothedMoveX,
      smoothedMoveY,
      smoothedLookDelta,
      lastFallbackInputForTest: tickInput,
    },
  };
}
