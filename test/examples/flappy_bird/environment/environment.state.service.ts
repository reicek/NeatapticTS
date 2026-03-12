import {
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_WORLD_WIDTH_PX,
} from '../constants/constants';
import type { FlappyRng } from '../rng';
import {
  resolveAdaptiveDifficultyProfile,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
  sampleGapCenterY,
} from '../flappy.simulation.shared.utils';
import { FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE } from './environment.constants';
import type { FlappyGameState } from './environment.types';

/**
 * Create a fresh Flappy Bird episode state.
 *
 * Educational note:
 * A new episode starts with one initial pipe already materialized so the first
 * observation is meaningful immediately. That avoids a cold-start phase where a
 * policy would receive mostly empty-space inputs.
 *
 * @example
 * ```ts
 * const state = createInitialFlappyState(rng);
 * ```
 *
 * @param rng - Random source used to generate initial pipe configuration.
 * @returns Initial state for one deterministic rollout.
 */
export function createInitialFlappyState(rng: FlappyRng): FlappyGameState {
  const initialGapCenterYPx = sampleGapCenterY(rng);
  const initialDifficultyProfile = resolveAdaptiveDifficultyProfile(
    0,
    FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE,
  );
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );

  return {
    frameIndex: 0,
    bird: {
      yPx: FLAPPY_WORLD_HEIGHT_PX * 0.5,
      velocityYPxPerFrame: 0,
    },
    pipes: [
      {
        xPx: FLAPPY_WORLD_WIDTH_PX + FLAPPY_PIPE_WIDTH_PX,
        gapCenterYPx: initialGapCenterYPx,
        gapSizePx: initialGapSizePx,
        passed: false,
      },
    ],
    lastSpawnedPipeGapPx: initialGapSizePx,
    lastSpawnedPipeGapCenterYPx: initialGapCenterYPx,
    lastSpawnedPipeSpawnIntervalFrames: initialSpawnIntervalFrames,
    framesUntilNextPipeSpawn: initialSpawnIntervalFrames,
    pipesPassed: 0,
    done: false,
  };
}
