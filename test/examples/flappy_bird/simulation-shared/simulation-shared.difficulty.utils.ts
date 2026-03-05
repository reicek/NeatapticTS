import {
  FLAPPY_DIFFICULTY_RAMP_PIPES,
  FLAPPY_PIPE_GAP_MIN_PX,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
} from '../constants/constants';
import { FLAPPY_SHARED_DEFAULT_DIFFICULTY_SCALE } from './simulation-shared.constants';
import type { SharedDifficultyProfile } from './simulation-shared.types';

/**
 * Resolves adaptive difficulty profile from passed-pipe progress.
 *
 * @param pipesPassed - Number of passed pipes.
 * @param difficultyScale - Curriculum scale in `[0, 1]`.
 * @returns Active difficulty profile.
 */
export function resolveAdaptiveDifficultyProfile(
  pipesPassed: number,
  difficultyScale = FLAPPY_SHARED_DEFAULT_DIFFICULTY_SCALE,
): SharedDifficultyProfile {
  const normalizedDifficultyScale = clamp(difficultyScale, 0, 1);
  const normalizedDifficultyProgress = clamp(
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

/**
 * Clamps a number between bounds.
 *
 * @param value - Input value.
 * @param minimum - Lower bound.
 * @param maximum - Upper bound.
 * @returns Bounded value.
 */
function clamp(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}

/**
 * Linear interpolation helper.
 *
 * @param startValue - Start value.
 * @param endValue - End value.
 * @param interpolationFactor - Blend factor in `[0, 1]`.
 * @returns Interpolated value.
 */
function interpolateValue(
  startValue: number,
  endValue: number,
  interpolationFactor: number,
): number {
  return startValue + (endValue - startValue) * interpolationFactor;
}
