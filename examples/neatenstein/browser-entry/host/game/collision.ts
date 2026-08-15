/**
 * Enemy/player contact damage for the Neatenstein host-side simulation.
 *
 * This module owns the melee half of AC-202 and AC-207: any living enemy that
 * overlaps the player's contact radius deals damage once per cooldown window,
 * and the player cannot be damaged again until the i-frame timer expires or
 * while a dash grants invulnerability.
 *
 * @module
 */

import {
  NEATENSTEIN_CONTACT_DAMAGE,
  NEATENSTEIN_CONTACT_IFRAME_MS,
  NEATENSTEIN_CONTACT_RANGE_CELLS,
} from './constants';
import { isInvulnerable } from './state';
import type { GameState } from './types';
import type { ContactPosition } from '../types';

// Re-export consolidated type so existing imports from this module remain valid.
export type { ContactPosition } from '../types';

/**
 * Minimum health value for the player.
 *
 * Health is clamped to this value so contact damage cannot overkill below zero.
 */
const MIN_PLAYER_HEALTH = 0;

/**
 * Minimum contact i-frame duration.
 *
 * The timer is clamped to this value after each tick.
 */
const MIN_CONTACT_IFRAME_MS = 0;

/**
 * Return whether a number is finite.
 *
 * @param value - Candidate numeric value.
 * @returns Whether the value is finite.
 */
function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

/**
 * Return whether a world position has finite coordinates.
 *
 * Invalid positions are treated as non-contacting so malformed enemy data
 * cannot accidentally damage the player from an undefined location.
 *
 * @param position - Position to validate.
 * @returns Whether both coordinates are finite.
 */
function isFinitePosition(position: ContactPosition): boolean {
  return isFiniteNumber(position.x) && isFiniteNumber(position.y);
}

/**
 * Resolve a safe elapsed timestep for contact-damage timers.
 *
 * Invalid or negative elapsed times are treated as zero. This keeps the contact
 * i-frame timer stable instead of allowing `NaN`, infinities, or negative time
 * to contaminate player state.
 *
 * @param dtMs - Candidate elapsed milliseconds.
 * @returns Non-negative finite elapsed milliseconds.
 */
function resolveElapsedMs(dtMs: number): number {
  return isFiniteNumber(dtMs) && dtMs > 0 ? dtMs : 0;
}

/**
 * Clamp a timer value to a non-negative finite number.
 *
 * Missing, invalid, or negative timers are treated as expired.
 *
 * @param value - Candidate timer value in milliseconds.
 * @returns Non-negative finite timer value.
 */
function resolveNonNegativeTimerMs(value: number | undefined): number {
  return typeof value === 'number' && isFiniteNumber(value) && value > 0
    ? value
    : MIN_CONTACT_IFRAME_MS;
}

/**
 * Advance the contact i-frame timer by one tick.
 *
 * @param currentIFrameMs - Current contact i-frame timer.
 * @param dtMs - Elapsed milliseconds this tick.
 * @returns Updated non-negative i-frame timer.
 */
function tickContactIFrame(
  currentIFrameMs: number | undefined,
  dtMs: number,
): number {
  const current = resolveNonNegativeTimerMs(currentIFrameMs);
  const elapsed = resolveElapsedMs(dtMs);

  return Math.max(MIN_CONTACT_IFRAME_MS, current - elapsed);
}

/**
 * Return the squared distance between two finite positions.
 *
 * Squared distance avoids a square-root operation during repeated contact
 * checks while preserving the same radius comparison semantics.
 *
 * @param a - First position.
 * @param b - Second position.
 * @returns Squared Euclidean distance.
 */
function squaredDistance(a: ContactPosition, b: ContactPosition): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;

  return dx * dx + dy * dy;
}

/**
 * Return whether an enemy is alive and eligible to deal contact damage.
 *
 * @param enemy - Enemy entry from the game state.
 * @returns Whether the enemy can damage the player.
 */
function isLivingContactEnemy(enemy: GameState['enemies'][number]): boolean {
  return (
    enemy.health > 0 &&
    enemy.active !== false &&
    isFinitePosition(enemy.position)
  );
}

/**
 * Return whether the player is close enough to any living enemy to take contact
 * damage this tick.
 *
 * @param state - State after i-frame timer advancement.
 * @returns Whether contact damage should be applied.
 */
function isPlayerTouchingLivingEnemy(state: GameState): boolean {
  const playerPosition = state.player.position;

  if (!isFinitePosition(playerPosition)) {
    return false;
  }

  const contactRange =
    isFiniteNumber(NEATENSTEIN_CONTACT_RANGE_CELLS) &&
    NEATENSTEIN_CONTACT_RANGE_CELLS > 0
      ? NEATENSTEIN_CONTACT_RANGE_CELLS
      : 0;

  const contactRangeSquared = contactRange * contactRange;

  return state.enemies.some((enemy) => {
    if (!isLivingContactEnemy(enemy)) {
      return false;
    }

    return (
      squaredDistance(enemy.position, playerPosition) <= contactRangeSquared
    );
  });
}

/**
 * Apply contact damage to the player and restart the contact i-frame timer.
 *
 * Health is clamped at zero so one contact event cannot overkill the player.
 *
 * @param state - State after i-frame timer advancement.
 * @returns New state with reduced player health and restarted i-frames.
 */
function applyContactDamage(state: GameState): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      health: Math.max(
        MIN_PLAYER_HEALTH,
        state.player.health - NEATENSTEIN_CONTACT_DAMAGE,
      ),
      contactIFrameMs: NEATENSTEIN_CONTACT_IFRAME_MS,
    },
  };
}

/**
 * Return a state with an advanced contact i-frame timer.
 *
 * @param state - Source state.
 * @param nextIFrameMs - Updated contact i-frame timer.
 * @returns State with the updated timer.
 */
function withContactIFrame(state: GameState, nextIFrameMs: number): GameState {
  return {
    ...state,
    player: {
      ...state.player,
      contactIFrameMs: nextIFrameMs,
    },
  };
}

/**
 * Resolve contact damage between the player and any overlapping enemy.
 *
 * The function performs these steps in order each tick:
 *
 * 1. Decrement the contact i-frame timer by the elapsed timestep.
 * 2. Skip damage while the player is dashing, invulnerable, dead, or still in
 *    i-frames.
 * 3. Check whether any living enemy overlaps the player's contact radius.
 * 4. Apply contact damage once and restart contact i-frames.
 * 5. Clamp player health at zero.
 *
 * @param state - Snapshot before contact damage resolution.
 * @param dtMs - Elapsed milliseconds since the last tick.
 * @returns New snapshot with updated `contactIFrameMs` and possibly reduced
 *   player health.
 *
 * @example
 * ```ts
 * const after = resolveContactDamage(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
 * if (after.player.health < state.player.health) {
 *   playDamageSound();
 * }
 * ```
 */
export function resolveContactDamage(
  state: GameState,
  dtMs: number,
): GameState {
  // Step 1: Always tick down the contact i-frame timer first so invulnerability
  // naturally expires even when no enemy is currently touching the player.
  const nextIFrameMs = tickContactIFrame(state.player.contactIFrameMs, dtMs);
  const next = withContactIFrame(state, nextIFrameMs);

  // Step 2: Dead players and invulnerable players cannot take contact damage.
  if (next.player.health <= MIN_PLAYER_HEALTH || isInvulnerable(next)) {
    return next;
  }

  // Step 3: Apply at most one contact-damage event per tick, even if multiple
  // enemies overlap the player.
  if (!isPlayerTouchingLivingEnemy(next)) {
    return next;
  }

  // Step 4: Damage the player and restart the contact i-frame cooldown.
  return applyContactDamage(next);
}
