import type { PopulationRenderState } from '../../browser-entry.types';

/**
 * Resolves the maximum survived-frame count in the current render state.
 *
 * @param renderState - Current render state.
 * @returns Maximum frames survived by any bird.
 */
export function resolveLeaderFramesSurvived(
  renderState: PopulationRenderState,
): number {
  return renderState.birds.reduce(
    (maximumFramesSurvived, bird) =>
      Math.max(maximumFramesSurvived, bird.framesSurvived),
    0,
  );
}