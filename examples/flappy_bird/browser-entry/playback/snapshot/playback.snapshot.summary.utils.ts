import type { PopulationRenderState } from '../../browser-entry.types';

/**
 * Small summary helpers derived from hydrated playback render state.
 *
 * Once a packed snapshot has been synchronized into the browser render state,
 * these helpers compute simple aggregate values needed by the HUD and playback
 * reporting flow.
 */

/**
 * Resolves the maximum survived-frame count in the current render state.
 *
 * This is the "leader frames survived" view of the current frame: the best raw
 * frame count among all birds currently represented in the render state.
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
