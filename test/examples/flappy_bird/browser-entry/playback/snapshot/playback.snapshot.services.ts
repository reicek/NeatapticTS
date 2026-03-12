import type {
  EvolutionPlaybackStepSnapshot,
  PopulationRenderState,
} from '../../browser-entry.types';

/**
 * Snapshot synchronization helpers for playback.
 *
 * The worker streams packed typed-array snapshots, while the browser renderer
 * wants stable mutable arrays of pipes and birds. This module performs that
 * translation in place so playback can stay fast and allocation-light.
 */

/**
 * Applies worker snapshot data to the mutable playback render state.
 *
 * This is the top-level hydration step for one frame: copy scalar frame fields,
 * then synchronize packed pipe and bird arrays into reusable browser-side
 * objects.
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
  renderState.cumulativePipeTravelPx = snapshot.cumulativePipeTravelPx;
  renderState.visibleWorldWidthPx = snapshot.visibleWorldWidthPx;
  renderState.visibleWorldHeightPx = snapshot.visibleWorldHeightPx;
  syncPlaybackSnapshotPipes(renderState, snapshot);
  syncPlaybackSnapshotBirds(renderState, snapshot);
}

/**
 * Synchronizes packed pipe snapshot fields into the reusable render-state pipe array.
 *
 * Instead of recreating pipe objects every frame, the browser grows the array as
 * needed and then mutates the existing records in place.
 *
 * @param renderState - Mutable render state mirror used by the browser.
 * @param snapshot - Packed worker playback snapshot for the current render tick.
 * @returns Nothing.
 */
function syncPlaybackSnapshotPipes(
  renderState: PopulationRenderState,
  snapshot: EvolutionPlaybackStepSnapshot,
): void {
  const targetPipeCount = snapshot.pipeCount;
  while (renderState.pipes.length < targetPipeCount) {
    renderState.pipes.push({
      id: 0,
      xPx: 0,
      gapCenterYPx: 0,
      gapSizePx: 0,
    });
  }

  renderState.pipes.length = targetPipeCount;
  for (let pipeIndex = 0; pipeIndex < targetPipeCount; pipeIndex += 1) {
    const pipe = renderState.pipes[pipeIndex];
    pipe.id = pipeIndex;
    pipe.xPx = snapshot.pipes.xPositionsPx[pipeIndex] ?? 0;
    pipe.gapCenterYPx = snapshot.pipes.gapCenterYPositionsPx[pipeIndex] ?? 0;
    pipe.gapSizePx = snapshot.pipes.gapSizesPx[pipeIndex] ?? 0;
  }
}

/**
 * Synchronizes packed bird snapshot fields into the reusable render-state bird array.
 *
 * This mirrors the pipe strategy: keep a stable array shape when possible and
 * update fields in place from the packed worker buffers.
 *
 * @param renderState - Mutable render state mirror used by the browser.
 * @param snapshot - Packed worker playback snapshot for the current render tick.
 * @returns Nothing.
 */
function syncPlaybackSnapshotBirds(
  renderState: PopulationRenderState,
  snapshot: EvolutionPlaybackStepSnapshot,
): void {
  const targetBirdCount = snapshot.birdCount;
  while (renderState.birds.length < targetBirdCount) {
    renderState.birds.push({
      yPx: 0,
      pipesPassed: 0,
      framesSurvived: 0,
      done: false,
    });
  }

  renderState.birds.length = targetBirdCount;
  for (let birdIndex = 0; birdIndex < targetBirdCount; birdIndex += 1) {
    const bird = renderState.birds[birdIndex];
    bird.yPx = snapshot.birds.yPositionsPx[birdIndex] ?? 0;
    bird.pipesPassed = snapshot.birds.pipesPassed[birdIndex] ?? 0;
    bird.framesSurvived = snapshot.birds.framesSurvived[birdIndex] ?? 0;
    bird.done = (snapshot.birds.doneFlags[birdIndex] ?? 0) === 1;
  }
}
