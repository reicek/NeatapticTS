import { FLAPPY_NEON_BIRD_PALETTE } from '../constants/constants';
import {
  FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import {
  type SharedDifficultyProfile,
  resolveAdaptiveDifficultyProfile,
  resolveNextSpawnGapSize as resolveSharedNextSpawnGapSize,
  resolveNextSpawnIntervalFrames as resolveSharedNextSpawnIntervalFrames,
} from '../flappy.simulation.shared.utils';
import type { BrowserDifficultyProfile, RngLike } from './browser-entry.types';

/**
 * Samples a random gap center y-position.
 *
 * @param rng - Deterministic RNG.
 * @param worldHeightPx - World height used to derive valid gap-center bounds.
 * @returns Sampled y-position.
 */
export function sampleGapCenterY(
  rng: RngLike,
  worldHeightPx: number = FLAPPY_WORLD_HEIGHT_PX,
): number {
  const lowerBoundGapCenterYPx = FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX;
  const upperBoundGapCenterYPx = resolveGapCenterUpperBoundYPx(worldHeightPx);
  return rng.nextInt(lowerBoundGapCenterYPx, upperBoundGapCenterYPx);
}

/**
 * Resolves next gap center with bounded per-pipe delta.
 *
 * @param previousGapCenterYPx - Previous spawn gap center.
 * @param rng - Deterministic RNG.
 * @param worldHeightPx - World height used to clamp candidate gap centers.
 * @returns Next gap center y-position.
 */
export function resolveNextSpawnGapCenterY(
  previousGapCenterYPx: number,
  rng: RngLike,
  worldHeightPx: number = FLAPPY_WORLD_HEIGHT_PX,
): number {
  const sampledGapCenterYPx = sampleGapCenterY(rng, worldHeightPx);
  const minimumGapCenterYPx = Math.max(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    previousGapCenterYPx - FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  const maximumGapCenterYPx = Math.min(
    resolveGapCenterUpperBoundYPx(worldHeightPx),
    previousGapCenterYPx + FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  return Math.max(
    minimumGapCenterYPx,
    Math.min(sampledGapCenterYPx, maximumGapCenterYPx),
  );
}

/**
 * Resolves the exclusive upper bound used for gap-center sampling.
 *
 * Educational note:
 * We cap dynamic viewport-derived bounds at the shared simulation maximum to
 * keep browser playback distribution aligned with trainer/evaluation defaults,
 * while still supporting smaller world heights.
 *
 * @param worldHeightPx - Current world height.
 * @returns Exclusive upper bound for `nextInt(minInclusive, maxExclusive)`.
 */
function resolveGapCenterUpperBoundYPx(worldHeightPx: number): number {
  const lowerBoundGapCenterYPx = FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX;
  const viewportDerivedUpperBound = Math.max(
    lowerBoundGapCenterYPx,
    worldHeightPx - FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  );
  return Math.max(
    lowerBoundGapCenterYPx,
    Math.min(FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX, viewportDerivedUpperBound),
  );
}

/**
 * Resolves deterministic bird color from palette index.
 *
 * @param birdIndex - Bird index in current population.
 * @param totalBirds - Population size.
 * @returns Hex color string.
 */
export function createBirdColor(birdIndex: number, totalBirds: number): string {
  if (totalBirds <= 0) return FLAPPY_NEON_BIRD_PALETTE[0];
  return FLAPPY_NEON_BIRD_PALETTE[birdIndex % FLAPPY_NEON_BIRD_PALETTE.length];
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
  difficultyProfile: BrowserDifficultyProfile,
  rng: RngLike,
): number {
  return resolveSharedNextSpawnGapSize(
    previousSpawnGapPx,
    difficultyProfile as SharedDifficultyProfile,
    rng,
  );
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
  difficultyProfile: BrowserDifficultyProfile,
): number {
  return resolveSharedNextSpawnIntervalFrames(
    previousSpawnIntervalFrames,
    difficultyProfile as SharedDifficultyProfile,
  );
}

/**
 * Resolves current difficulty profile from pipes-passed progress.
 *
 * @param pipesPassed - Current pipes passed.
 * @returns Difficulty profile.
 */
export function resolveDifficultyProfile(
  pipesPassed: number,
): BrowserDifficultyProfile {
  return resolveAdaptiveDifficultyProfile(pipesPassed, 1);
}
