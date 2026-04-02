import {
  FLAPPY_NEON_PALETTE,
  FLAPPY_NON_CHAMPION_OPACITY,
} from '../../constants/constants';
import type { PopulationRenderState } from '../browser-entry.types';
import { resolveFramePrimaryWinnerIndex } from '../browser-entry.observation.utils';

/**
 * Pure render-style result for one bird body draw pass.
 */
export interface PlaybackBirdRenderStyle {
  birdOpacity: number;
  birdRenderColor: string;
  isChampionBird: boolean;
}

/**
 * Resolves the champion bird index for the current render frame.
 *
 * Champion selection first prefers the primary winner resolver and then
 * falls back to the first alive bird when no winner index is available.
 *
 * @param renderState - Current frame render snapshot.
 * @returns Champion index or `-1` when no bird is alive.
 */
export function resolveChampionBirdIndex(
  renderState: PopulationRenderState,
): number {
  const leaderBirdIndex = resolveFramePrimaryWinnerIndex(
    renderState.birds,
    true,
  );
  if (leaderBirdIndex >= 0) {
    return leaderBirdIndex;
  }

  return renderState.birds.findIndex((bird) => !bird.done);
}

/**
 * Resolves opacity, body color, and champion marker for one bird.
 *
 * @param birdIndex - Index of the bird currently being rendered.
 * @param championBirdIndex - Resolved champion index for the frame.
 * @returns Pure style payload used by the render service.
 */
export function resolveBirdRenderStyle(
  birdIndex: number,
  championBirdIndex: number,
): PlaybackBirdRenderStyle {
  const isChampionBird = birdIndex === championBirdIndex;

  return {
    birdOpacity: isChampionBird ? 1 : FLAPPY_NON_CHAMPION_OPACITY,
    birdRenderColor: isChampionBird
      ? FLAPPY_NEON_PALETTE.championBird
      : FLAPPY_NEON_PALETTE.nonChampionBird,
    isChampionBird,
  };
}
