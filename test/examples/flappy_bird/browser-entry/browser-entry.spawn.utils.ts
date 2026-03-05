import { FLAPPY_NEON_BIRD_PALETTE } from '../constants/constants';
import {
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
 * @returns Sampled y-position.
 */
export function sampleGapCenterY(
  rng: RngLike,
  worldHeightPx: number = FLAPPY_WORLD_HEIGHT_PX,
): number {
  const lowerBoundGapCenterYPx = FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX;
  const upperBoundGapCenterYPx = Math.max(
    lowerBoundGapCenterYPx,
    worldHeightPx - FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  );
  return rng.nextInt(lowerBoundGapCenterYPx, upperBoundGapCenterYPx + 1);
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
  rng: RngLike,
  worldHeightPx: number = FLAPPY_WORLD_HEIGHT_PX,
): number {
  const sampledGapCenterYPx = sampleGapCenterY(rng, worldHeightPx);
  const minimumGapCenterYPx = Math.max(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    previousGapCenterYPx - FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  const maximumGapCenterYPx = Math.min(
    Math.max(
      FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
      worldHeightPx - FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    ),
    previousGapCenterYPx + FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  return Math.max(
    minimumGapCenterYPx,
    Math.min(sampledGapCenterYPx, maximumGapCenterYPx),
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
