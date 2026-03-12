import type {
  WorkerPlaybackFrameSnapshot,
  WorkerPlaybackState,
} from './flappy-evolution-worker.types';

/**
 * Creates a serializable snapshot of current playback state.
 *
 * Workers should send only structured-clone-safe payloads. This helper strips
 * runtime-only references (e.g., network instances, sets) and keeps only
 * renderer-relevant fields.
 *
 * Educational note:
 * The snapshot is intentionally column-oriented. By packing values into typed
 * arrays, the worker can transfer large bird populations to the host with much
 * lower overhead than a per-frame array of nested objects.
 *
 * This is a small example of a structure-of-arrays transport layout. If that
 * pattern is unfamiliar, the Wikipedia article on "AoS and SoA" is a good short
 * reference for why packed columns are often friendlier to hot-path data
 * movement than arrays of rich objects.
 *
 * @param playbackState - Current mutable playback state.
 * @returns Immutable frame snapshot for the host.
 */
export function createWorkerPlaybackSnapshot(
  playbackState: WorkerPlaybackState,
): WorkerPlaybackFrameSnapshot {
  const pipeCount = playbackState.pipes.length;
  const birdCount = playbackState.birds.length;
  const pipeXPositionsPx = new Float32Array(pipeCount);
  const pipeGapCenterYPositionsPx = new Float32Array(pipeCount);
  const pipeGapSizesPx = new Float32Array(pipeCount);
  const birdYPositionsPx = new Float32Array(birdCount);
  const birdPipesPassed = new Uint32Array(birdCount);
  const birdFramesSurvived = new Uint32Array(birdCount);
  const birdDoneFlags = new Uint8Array(birdCount);

  for (let pipeIndex = 0; pipeIndex < pipeCount; pipeIndex += 1) {
    const pipe = playbackState.pipes[pipeIndex];
    pipeXPositionsPx[pipeIndex] = pipe.xPx;
    pipeGapCenterYPositionsPx[pipeIndex] = pipe.gapCenterYPx;
    pipeGapSizesPx[pipeIndex] = pipe.gapSizePx;
  }

  for (let birdIndex = 0; birdIndex < birdCount; birdIndex += 1) {
    const bird = playbackState.birds[birdIndex];
    birdYPositionsPx[birdIndex] = bird.yPx;
    birdPipesPassed[birdIndex] = bird.pipesPassed;
    birdFramesSurvived[birdIndex] = bird.framesSurvived;
    birdDoneFlags[birdIndex] = bird.done ? 1 : 0;
  }

  return {
    format: 'packed-v1',
    frameIndex: playbackState.frameIndex,
    cumulativePipeTravelPx: playbackState.cumulativePipeTravelPx,
    visibleWorldWidthPx: playbackState.visibleWorldWidthPx,
    visibleWorldHeightPx: playbackState.visibleWorldHeightPx,
    pipeCount,
    birdCount,
    pipes: {
      xPositionsPx: pipeXPositionsPx,
      gapCenterYPositionsPx: pipeGapCenterYPositionsPx,
      gapSizesPx: pipeGapSizesPx,
    },
    birds: {
      yPositionsPx: birdYPositionsPx,
      pipesPassed: birdPipesPassed,
      framesSurvived: birdFramesSurvived,
      doneFlags: birdDoneFlags,
    },
  };
}

/**
 * Resolves transferable buffers for one packed playback snapshot.
 *
 * The returned buffers should be passed as the second argument to
 * `postMessage(...)` so ownership moves to the host thread instead of copying
 * the typed-array contents.
 *
 * That ownership transfer is a large part of why the worker can stream full
 * population snapshots without forcing the main thread to pay unnecessary copy
 * costs every frame.
 *
 * @param snapshot - Packed playback snapshot posted back to the browser host.
 * @returns Transfer list used to move typed-array buffers without copying.
 */
export function resolveWorkerPlaybackSnapshotTransferList(
  snapshot: WorkerPlaybackFrameSnapshot,
): Transferable[] {
  return [
    snapshot.pipes.xPositionsPx.buffer,
    snapshot.pipes.gapCenterYPositionsPx.buffer,
    snapshot.pipes.gapSizesPx.buffer,
    snapshot.birds.yPositionsPx.buffer,
    snapshot.birds.pipesPassed.buffer,
    snapshot.birds.framesSurvived.buffer,
    snapshot.birds.doneFlags.buffer,
  ];
}
