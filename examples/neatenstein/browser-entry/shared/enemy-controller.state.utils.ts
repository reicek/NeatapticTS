/**
 * State and timestep helper functions for the enemy controller.
 *
 * Extracted from {@link enemy-controller.ts} as pure leaf executors. These
 * functions own no module-level mutable state and are safe to call from any
 * context.
 *
 * @module
 */

import { NEATENSTEIN_FIXED_TIMESTEP_MS } from '../host/game/constants';
import type { GameState } from '../host/game/types';
import {
  ENEMY_CONTROLLER_STARTING_AMMO,
  PREVIOUS_STEP_DISTANCE_SENTINEL,
} from './enemy-controller.constants';
import type { EnemyControllerState } from './enemy-controller.types';
import { isFiniteNumber } from './math-guards.utils';

/**
 * Clamp a timestep to a positive finite value.
 *
 * Invalid or non-positive timesteps fall back to the canonical fixed timestep
 * so malformed caller input cannot poison enemy behavior.
 *
 * @param dtMs - Candidate timestep in milliseconds.
 * @returns Positive finite timestep.
 */
export function resolveTimestepMs(dtMs: number): number {
  if (dtMs === 0) return 0;
  return isFiniteNumber(dtMs) && dtMs > 0
    ? dtMs
    : NEATENSTEIN_FIXED_TIMESTEP_MS;
}

/**
 * Build a fresh controller state from a {@link GameState} snapshot.
 *
 * Each source enemy is initialized with default ammunition, no fire cooldown,
 * and the idle animation state. Callers must hold the returned state across
 * ticks so fire cooldowns, ammunition, and de-rez timing advance correctly.
 *
 * @param state - Source game snapshot.
 * @returns Fresh controller state.
 */
export function createEnemyControllerState(
  state: GameState,
): EnemyControllerState {
  return {
    enemies: state.enemies.map((enemy, index) => ({
      index,
      position: { ...enemy.position },
      health: enemy.health,
      yawRad: 0,
      animationState: enemy.health <= 0 ? 'death' : 'idle',
      ammo: ENEMY_CONTROLLER_STARTING_AMMO,
      fireCooldownMs: 0,
      deRezElapsedMs: 0,
      active: true,
      walkTick: 0,
      shootBlinkTicks: 0,
      flankStallTicks: 0,
      bfsStallTicks: 0,
      weights: undefined,
      variantId: 0,
      previousStepDistance: PREVIOUS_STEP_DISTANCE_SENTINEL,
      stunTimerMs: 0,
    })),
    hitscanEvents: [],
  };
}