import {
  FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  FLAPPY_PIPE_GAP_RANDOM_JITTER_PX,
  FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX,
  FLAPPY_PIPE_GAP_START_MULTIPLIER,
  FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER,
} from '../constants/constants';
import type {
  SharedDifficultyProfile,
  SharedRngLike,
} from './simulation-shared.types';

/**
 * Samples a random gap center y-position.
 *
 * @param rng - Deterministic RNG.
 * @returns Sampled y-position.
 */
export function sampleGapCenterY(rng: SharedRngLike): number {
  return rng.nextInt(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  );
}

/**
 * Resolves next gap center with bounded per-pipe delta.
 *
 * @param previousGapCenterYPx - Previous spawn gap center.
 * @param rng - Deterministic RNG.
 * @returns Next gap center y-position.
 */
export function resolveNextSpawnGapCenterY(
  previousGapCenterYPx: number,
  rng: SharedRngLike,
): number {
  const sampledGapCenterYPx = sampleGapCenterY(rng);
  const minimumGapCenterYPx = Math.max(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    previousGapCenterYPx - FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  const maximumGapCenterYPx = Math.min(
    FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
    previousGapCenterYPx + FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  return clamp(sampledGapCenterYPx, minimumGapCenterYPx, maximumGapCenterYPx);
}

/**
 * Resolves next spawn gap size using progressive shrink and jitter.
 *
 * @param previousSpawnGapPx - Previous spawn gap size.
 * @param difficultyProfile - Active difficulty profile.
 * @param rng - Deterministic RNG.
 * @returns Next spawn gap size.
 */
export function resolveNextSpawnGapSize(
  previousSpawnGapPx: number | undefined,
  difficultyProfile: SharedDifficultyProfile,
  rng: SharedRngLike,
): number {
  const hardestGapPx = difficultyProfile.pipeGapPx;
  const initialWideGapPx = Math.round(
    hardestGapPx * FLAPPY_PIPE_GAP_START_MULTIPLIER,
  );

  const progressiveGapPx =
    previousSpawnGapPx == null
      ? initialWideGapPx
      : Math.max(
          hardestGapPx,
          previousSpawnGapPx - FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX,
        );

  const randomJitterPx = rng.nextInt(
    -FLAPPY_PIPE_GAP_RANDOM_JITTER_PX,
    FLAPPY_PIPE_GAP_RANDOM_JITTER_PX + 1,
  );
  const randomizedGapPx = progressiveGapPx + randomJitterPx;

  return Math.round(clamp(randomizedGapPx, hardestGapPx, initialWideGapPx));
}

/**
 * Resolves next spawn interval using progressive shrink.
 *
 * @param previousSpawnIntervalFrames - Previous spawn interval.
 * @param difficultyProfile - Active difficulty profile.
 * @returns Next spawn interval in frames.
 */
export function resolveNextSpawnIntervalFrames(
  previousSpawnIntervalFrames: number | undefined,
  difficultyProfile: SharedDifficultyProfile,
): number {
  const hardestIntervalFrames = difficultyProfile.pipeSpawnIntervalFrames;
  const initialWideIntervalFrames = Math.round(
    hardestIntervalFrames * FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER,
  );

  const progressiveIntervalFrames =
    previousSpawnIntervalFrames == null
      ? initialWideIntervalFrames
      : Math.max(
          hardestIntervalFrames,
          previousSpawnIntervalFrames -
            FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES,
        );

  return Math.round(
    clamp(
      progressiveIntervalFrames,
      hardestIntervalFrames,
      initialWideIntervalFrames,
    ),
  );
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
