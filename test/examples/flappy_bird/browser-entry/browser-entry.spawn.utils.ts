import { FLAPPY_NEON_BIRD_PALETTE } from '../constants/constants';
import {
  type SharedDifficultyProfile,
  resolveAdaptiveDifficultyProfile,
  resolveNextSpawnGapCenterY as resolveSharedNextSpawnGapCenterY,
  resolveNextSpawnGapSize as resolveSharedNextSpawnGapSize,
  resolveNextSpawnIntervalFrames as resolveSharedNextSpawnIntervalFrames,
  sampleGapCenterY as sampleSharedGapCenterY,
} from '../flappy.simulation.shared.utils';
import type { BrowserDifficultyProfile, RngLike } from './browser-entry.types';

/**
 * Samples a random gap center y-position.
 *
 * @param rng - Deterministic RNG.
 * @returns Sampled y-position.
 */
export function sampleGapCenterY(rng: RngLike): number {
  return sampleSharedGapCenterY(rng);
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
): number {
  return resolveSharedNextSpawnGapCenterY(previousGapCenterYPx, rng);
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
