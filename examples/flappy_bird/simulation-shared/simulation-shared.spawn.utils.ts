import {
  FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  FLAPPY_PIPE_GAP_EDGE_MARGIN_RATIO,
  FLAPPY_PIPE_GAP_RANDOM_JITTER_PX,
  FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX,
  FLAPPY_PIPE_GAP_START_MULTIPLIER,
  FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import { clampValue } from './simulation-shared.math.utils';
import type {
  SharedDifficultyProfile,
  SharedRngLike,
} from './simulation-shared.types';

/**
 * Resolves gap-size-aware minimum and maximum gap center y-positions.
 *
 * The minimum edge margin ensures that at least `FLAPPY_PIPE_GAP_EDGE_MARGIN_RATIO`
 * of the world height appears as solid pipe above the opening and below the
 * opening. This prevents the gap from clipping the canvas boundary even when the
 * initial wide gap is active.
 *
 * @param currentGapSizePx - Actual gap size for the pipe being placed.
 * @param maximumGapCenterYPx - Viewport-derived or default upper center bound.
 * @returns Effective [minY, maxY) range for gap center sampling.
 */
function resolveGapCenterBounds(
  currentGapSizePx: number,
  maximumGapCenterYPx: number,
): { minY: number; maxY: number } {
  const edgeMarginPx = Math.ceil(
    FLAPPY_WORLD_HEIGHT_PX * FLAPPY_PIPE_GAP_EDGE_MARGIN_RATIO,
  );
  const halfGapPx = currentGapSizePx / 2;
  const gapAwareMinY = Math.ceil(halfGapPx + edgeMarginPx);
  const gapAwareMaxY = Math.floor(
    FLAPPY_WORLD_HEIGHT_PX - halfGapPx - edgeMarginPx,
  );
  const minY = Math.max(FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX, gapAwareMinY);
  const maxY = Math.min(maximumGapCenterYPx, gapAwareMaxY);
  return { minY, maxY: Math.max(minY + 1, maxY) };
}

/**
 * Samples a random gap center y-position.
 *
 * The sampled center is bounded so the gap opening always stays inside the
 * visible play area and at least `FLAPPY_PIPE_GAP_EDGE_MARGIN_RATIO` of the
 * world height remains as solid pipe on each side.
 *
 * @param rng - Deterministic RNG.
 * @param currentGapSizePx - Actual gap size for the pipe being placed.
 * @param maximumGapCenterYPx - Optional inclusive upper bound for smaller viewports.
 * @returns Sampled y-position.
 */
export function sampleGapCenterY(
  rng: SharedRngLike,
  currentGapSizePx: number,
  maximumGapCenterYPx: number = FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
): number {
  const { minY, maxY } = resolveGapCenterBounds(
    currentGapSizePx,
    maximumGapCenterYPx,
  );
  return rng.nextInt(minY, maxY);
}

/**
 * Resolves next gap center with bounded per-pipe delta.
 *
 * Consecutive gaps are deliberately constrained to avoid unfair zig-zag jumps.
 * The center is additionally bounded so the gap opening always keeps at least
 * `FLAPPY_PIPE_GAP_EDGE_MARGIN_RATIO` of world height as solid pipe on each side.
 *
 * @param previousGapCenterYPx - Previous spawn gap center.
 * @param rng - Deterministic RNG.
 * @param currentGapSizePx - Actual gap size for the pipe being placed.
 * @param maximumGapCenterYPx - Optional inclusive upper bound for smaller viewports.
 * @returns Next gap center y-position.
 */
export function resolveNextSpawnGapCenterY(
  previousGapCenterYPx: number,
  rng: SharedRngLike,
  currentGapSizePx: number,
  maximumGapCenterYPx: number = FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
): number {
  const { minY, maxY } = resolveGapCenterBounds(
    currentGapSizePx,
    maximumGapCenterYPx,
  );
  const sampledGapCenterYPx = rng.nextInt(minY, maxY);
  const minimumGapCenterYPx = Math.max(
    minY,
    previousGapCenterYPx - FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  const clampedMaximumGapCenterYPx = Math.min(
    maxY,
    previousGapCenterYPx + FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  return clampValue(
    sampledGapCenterYPx,
    minimumGapCenterYPx,
    Math.max(minimumGapCenterYPx, clampedMaximumGapCenterYPx),
  );
}

/**
 * Resolves next spawn gap size using progressive shrink and jitter.
 *
 * The gap starts wider than the current hardest target, then shrinks toward the
 * active difficulty profile with a small amount of deterministic jitter so runs
 * do not feel mechanically repetitive.
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

  return Math.round(
    clampValue(randomizedGapPx, hardestGapPx, initialWideGapPx),
  );
}

/**
 * Resolves next spawn interval using progressive shrink.
 *
 * This mirrors the gap-size logic: early pipes are spaced more generously, then
 * spacing contracts toward the current difficulty target as the episode settles
 * into its harder rhythm.
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
    clampValue(
      progressiveIntervalFrames,
      hardestIntervalFrames,
      initialWideIntervalFrames,
    ),
  );
}
