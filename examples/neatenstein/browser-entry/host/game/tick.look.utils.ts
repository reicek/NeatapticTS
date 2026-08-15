/**
 * Look/aim executor for the Neatenstein tick pipeline.
 *
 * @module
 */

import { MAX_TURN_RATE } from '../../harness/neat-io-config';
import type { GameState } from './types';

/**
 * Rotate the player by a yaw delta for one tick.
 *
 * Non-finite or zero deltas leave the state unchanged. The `lookDelta` is
 * clamped to {@link MAX_TURN_RATE} so no caller can exceed the canonical
 * turn-rate cap, regardless of whether the input originated from the NEAT
 * network mapping or a direct call.
 *
 * @param state - Game state before look input.
 * @param lookDelta - Horizontal yaw delta in radians.
 * @returns Updated state, or the original state if no rotation is needed.
 */
export function applyLook(state: GameState, lookDelta: number): GameState {
  if (!Number.isFinite(lookDelta) || lookDelta === 0) {
    return state;
  }

  const clampedDelta =
    lookDelta > MAX_TURN_RATE
      ? MAX_TURN_RATE
      : lookDelta < -MAX_TURN_RATE
        ? -MAX_TURN_RATE
        : lookDelta;

  return {
    ...state,
    player: {
      ...state.player,
      angleRad: state.player.angleRad + clampedDelta,
    },
  };
}
