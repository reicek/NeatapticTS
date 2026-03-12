import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX,
  FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import type { FlappyGameState } from './environment.types';

/**
 * Apply out-of-bounds, pipe-collision, and pass-credit rules for one substep.
 *
 * Educational note:
 * Collision resolution and progress credit live together because both depend on
 * the same bird-vs-pipe geometry for the current substep. Keeping them in one
 * place helps the environment avoid inconsistent "passed but also collided"
 * edge cases.
 *
 * @param state - Mutable simulation state to update in-place.
 * @returns Nothing.
 */
export function updateCollisionAndProgressState(state: FlappyGameState): void {
  // Step 1: Resolve world-bound collisions.
  if (state.bird.yPx - FLAPPY_BIRD_RADIUS_PX <= 0) {
    state.done = true;
    state.doneReason = 'out_of_bounds';
    return;
  }
  if (state.bird.yPx + FLAPPY_BIRD_RADIUS_PX >= FLAPPY_WORLD_HEIGHT_PX) {
    state.done = true;
    state.doneReason = 'out_of_bounds';
    return;
  }

  // Step 2: Resolve pipe overlap collisions and pass credit.
  const birdLeft = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
  const birdRight = FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX;

  for (const pipe of state.pipes) {
    const pipeLeft = pipe.xPx - FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
    const pipeRight =
      pipe.xPx + FLAPPY_PIPE_WIDTH_PX + FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;

    const overlapsHorizontally = birdRight >= pipeLeft && birdLeft <= pipeRight;
    if (overlapsHorizontally) {
      const gapHalfHeight = pipe.gapSizePx * 0.5;
      const gapTopY =
        pipe.gapCenterYPx -
        gapHalfHeight +
        FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;
      const gapBottomY =
        pipe.gapCenterYPx +
        gapHalfHeight -
        FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;

      const birdTopY = state.bird.yPx - FLAPPY_BIRD_RADIUS_PX;
      const birdBottomY = state.bird.yPx + FLAPPY_BIRD_RADIUS_PX;
      const birdInsideGap = birdTopY >= gapTopY && birdBottomY <= gapBottomY;

      if (!birdInsideGap) {
        state.done = true;
        state.doneReason = 'collision';
        return;
      }
    }

    if (!pipe.passed && pipeRight < FLAPPY_BIRD_X_PX) {
      pipe.passed = true;
      state.pipesPassed++;
    }
  }
}
