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
import type { GameState } from './types';

/**
 * Resolve contact damage between the player and any overlapping enemy.
 *
 * The function performs these steps in order each tick:
 *   1. Decrement the contact i-frame timer.
 *   2. Skip damage while the player is dashing or still in i-frames.
 *   3. If any living enemy is within {@link NEATENSTEIN_CONTACT_RANGE_CELLS},
 *      apply {@link NEATENSTEIN_CONTACT_DAMAGE} and restart i-frames.
 *   4. Clamp health at zero so a single tick can never overkill the player.
 *
 * @param state - Snapshot before contact damage resolution.
 * @param dtMs - Elapsed milliseconds since the last tick.
 * @returns New snapshot with updated `contactIFrameMs` and possibly reduced
 *   player health.
 *
 * @example
 * ```ts
 * const after = resolveContactDamage(state, NEATENSTEIN_FIXED_TIMESTEP_MS);
 * if (after.player.health < before.player.health) {
 *   playDamageSound();
 * }
 * ```
 */
export function resolveContactDamage(
  state: GameState,
  dtMs: number,
): GameState {
  const nextIFrame = Math.max(0, (state.player.contactIFrameMs ?? 0) - dtMs);
  const next: GameState = {
    ...state,
    player: {
      ...state.player,
      contactIFrameMs: nextIFrame,
    },
  };

  if (
    next.player.dashTimeRemainingMs > 0 ||
    (next.player.contactIFrameMs ?? 0) > 0
  ) {
    return next;
  }

  const touching = next.enemies.some(
    (enemy) =>
      enemy.health > 0 &&
      Math.hypot(
        enemy.position.x - next.player.position.x,
        enemy.position.y - next.player.position.y,
      ) <= NEATENSTEIN_CONTACT_RANGE_CELLS,
  );

  if (!touching) {
    return next;
  }

  return {
    ...next,
    player: {
      ...next.player,
      health: Math.max(0, next.player.health - NEATENSTEIN_CONTACT_DAMAGE),
      contactIFrameMs: NEATENSTEIN_CONTACT_IFRAME_MS,
    },
  };
}
