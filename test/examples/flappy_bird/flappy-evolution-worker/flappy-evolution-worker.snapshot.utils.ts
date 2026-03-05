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
 * @param playbackState - Current mutable playback state.
 * @returns Immutable frame snapshot for the host.
 */
export function createWorkerPlaybackSnapshot(
  playbackState: WorkerPlaybackState,
): WorkerPlaybackFrameSnapshot {
  return {
    frameIndex: playbackState.frameIndex,
    visibleWorldWidthPx: playbackState.visibleWorldWidthPx,
    visibleWorldHeightPx: playbackState.visibleWorldHeightPx,
    pipes: playbackState.pipes.map((pipe) => ({
      id: pipe.id,
      xPx: pipe.xPx,
      gapCenterYPx: pipe.gapCenterYPx,
      gapSizePx: pipe.gapSizePx,
    })),
    birds: playbackState.birds.map((bird) => ({
      color: bird.color,
      yPx: bird.yPx,
      pipesPassed: bird.pipesPassed,
      framesSurvived: bird.framesSurvived,
      done: bird.done,
    })),
  };
}
