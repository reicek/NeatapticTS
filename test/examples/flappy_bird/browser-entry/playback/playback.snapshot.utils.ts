import type {
  EvolutionPlaybackStepSnapshot,
  PopulationRenderState,
} from '../browser-entry.types';

/**
 * Applies worker snapshot data to the mutable playback render state.
 *
 * @param renderState - Mutable render state mirror used by the browser.
 * @param snapshot - Worker playback snapshot for the current render tick.
 * @returns Nothing.
 */
export function applyPlaybackSnapshot(
  renderState: PopulationRenderState,
  snapshot: EvolutionPlaybackStepSnapshot,
): void {
  renderState.frameIndex = snapshot.frameIndex;
  renderState.visibleWorldWidthPx = snapshot.visibleWorldWidthPx;
  renderState.visibleWorldHeightPx = snapshot.visibleWorldHeightPx;
  renderState.pipes = snapshot.pipes;
  renderState.birds = snapshot.birds.map((birdSnapshot) => ({
    color: birdSnapshot.color,
    yPx: birdSnapshot.yPx,
    pipesPassed: birdSnapshot.pipesPassed,
    framesSurvived: birdSnapshot.framesSurvived,
    done: birdSnapshot.done,
  }));
}

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
