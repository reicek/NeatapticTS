import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_NORMALIZATION_EPSILON,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_WORLD_WIDTH_PX,
} from '../constants/constants';
import {
  resolveAdaptiveDifficultyProfile,
  resolveObservationFeatures,
  resolveObservationVectorFromFeatures,
} from '../flappy.simulation.shared.utils';
import { FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE } from './environment.constants';
import type {
  FlappyDifficultyScale,
  FlappyGameState,
  FlappyObservationFeatures,
} from './environment.types';

/**
 * Generate the network observation vector for the current state.
 *
 * Observation (12 numbers):
 *  1) bird y position normalized to [0, 1]
 *  2) bird vertical velocity normalized to [-1, 1]
 *  3) distance to next pipe normalized to [0, 1]
 *  4) delta (bird y - gap center y) normalized to [-1, 1]
 *  5) next pipe gap top normalized to [0, 1]
 *  6) next pipe gap bottom normalized to [0, 1]
 *  7) distance to second pipe normalized to [0, 1]
 *  8) delta to second gap center normalized to [-1, 1]
 *  9) time-to-next-pipe closeness normalized to [0, 1]
 * 10) signed clearance relative to next gap normalized to [-1, 1]
 * 11) required vertical velocity toward next gap center normalized to [-1, 1]
 * 12) gap-center transition (next to second) normalized to [-1, 1]
 *
 * @param state - Current state.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @returns Input vector for the neural network.
 */
export function getFlappyObservation(
  state: FlappyGameState,
  difficultyScale: FlappyDifficultyScale = FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE,
): number[] {
  return resolveObservationVectorFromFeatures(
    getFlappyObservationFeatures(state, difficultyScale),
  );
}

/**
 * Resolve structured observation features for policy input and reward shaping.
 *
 * @param state - Current state.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @returns Named feature object.
 */
export function getFlappyObservationFeatures(
  state: FlappyGameState,
  difficultyScale: FlappyDifficultyScale = FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE,
): FlappyObservationFeatures {
  const difficultyProfile = resolveAdaptiveDifficultyProfile(
    state.pipesPassed,
    difficultyScale,
  );
  return resolveObservationFeatures({
    birdYPx: state.bird.yPx,
    velocityYPxPerFrame: state.bird.velocityYPxPerFrame,
    pipes: state.pipes,
    visibleWorldWidthPx: FLAPPY_WORLD_WIDTH_PX,
    difficultyProfile,
    activeSpawnIntervalFrames: Math.max(
      1,
      state.lastSpawnedPipeSpawnIntervalFrames,
    ),
    defaultGapSizePx: FLAPPY_PIPE_GAP_PX,
    birdCenterXPx: FLAPPY_BIRD_X_PX,
    birdRadiusPx: FLAPPY_BIRD_RADIUS_PX,
    pipeWidthPx: FLAPPY_PIPE_WIDTH_PX,
    worldHeightPx: FLAPPY_WORLD_HEIGHT_PX,
    maxFallSpeedPxPerFrame: FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    normalizationEpsilon: FLAPPY_NORMALIZATION_EPSILON,
  });
}
