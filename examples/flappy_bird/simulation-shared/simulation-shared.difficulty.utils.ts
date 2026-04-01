import {
  FLAPPY_DIFFICULTY_RAMP_PIPES,
  FLAPPY_PIPE_GAP_MIN_PX,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
} from '../constants/constants';
import { clampValue, interpolateValue } from './simulation-shared.math.utils';
import { FLAPPY_SHARED_DEFAULT_DIFFICULTY_SCALE } from './simulation-shared.constants';
import type { SharedDifficultyProfile } from './simulation-shared.types';

/**
 * Resolves adaptive difficulty profile from passed-pipe progress.
 *
 * Educational note:
 * Difficulty is ramped as a smooth profile rather than as a sequence of hard
 * level jumps. That keeps the task readable for humans and less noisy for
 * evolution.
 *
 * The idea is closely related to curriculum learning: easier versions of the
 * task dominate early, then the example interpolates toward the harder target
 * settings as progress increases.
 *
 * @param pipesPassed - Number of passed pipes.
 * @param difficultyScale - Curriculum scale in `[0, 1]`.
 * @returns Active difficulty profile.
 */
export function resolveAdaptiveDifficultyProfile(
  pipesPassed: number,
  difficultyScale = FLAPPY_SHARED_DEFAULT_DIFFICULTY_SCALE,
): SharedDifficultyProfile {
  const normalizedDifficultyScale = clampValue(difficultyScale, 0, 1);
  const normalizedDifficultyProgress = clampValue(
    (pipesPassed / Math.max(1, FLAPPY_DIFFICULTY_RAMP_PIPES)) *
      normalizedDifficultyScale,
    0,
    1,
  );

  return {
    pipeGapPx: Math.round(
      interpolateValue(
        FLAPPY_PIPE_GAP_PX,
        FLAPPY_PIPE_GAP_MIN_PX,
        normalizedDifficultyProgress,
      ),
    ),
    pipeSpeedPxPerFrame: interpolateValue(
      FLAPPY_PIPE_SPEED_PX_PER_FRAME,
      FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
      normalizedDifficultyProgress,
    ),
    pipeSpawnIntervalFrames: Math.round(
      interpolateValue(
        FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
        FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
        normalizedDifficultyProgress,
      ),
    ),
  };
}
