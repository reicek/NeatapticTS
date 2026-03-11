import type Network from '../../../../src/architecture/network';
import { createSharedObservationMemoryState } from '../flappy.simulation.shared.utils';
import {
  createWorkerPlaybackSnapshot,
  resolveWorkerPlaybackSnapshotTransferList,
} from './flappy-evolution-worker.snapshot.utils';
import type { WorkerPlaybackState } from './flappy-evolution-worker.types';

describe('createWorkerPlaybackSnapshot', () => {
  it('packs playback state into typed arrays for worker transport', () => {
    const playbackState = createPlaybackState();

    expect(createWorkerPlaybackSnapshot(playbackState)).toEqual({
      format: 'packed-v1',
      frameIndex: 12,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
      pipeCount: 1,
      birdCount: 1,
      pipes: {
        xPositionsPx: new Float32Array([320]),
        gapCenterYPositionsPx: new Float32Array([220]),
        gapSizesPx: new Float32Array([120]),
      },
      birds: {
        yPositionsPx: new Float32Array([144]),
        pipesPassed: new Uint32Array([3]),
        framesSurvived: new Uint32Array([18]),
        doneFlags: new Uint8Array([0]),
      },
    });
  });

  it('returns all typed-array buffers in the transfer list', () => {
    const snapshot = createWorkerPlaybackSnapshot(createPlaybackState());

    expect(resolveWorkerPlaybackSnapshotTransferList(snapshot)).toEqual([
      snapshot.pipes.xPositionsPx.buffer,
      snapshot.pipes.gapCenterYPositionsPx.buffer,
      snapshot.pipes.gapSizesPx.buffer,
      snapshot.birds.yPositionsPx.buffer,
      snapshot.birds.pipesPassed.buffer,
      snapshot.birds.framesSurvived.buffer,
      snapshot.birds.doneFlags.buffer,
    ]);
  });
});

function createPlaybackState(): WorkerPlaybackState {
  return {
    frameIndex: 12,
    visibleWorldWidthPx: 640,
    visibleWorldHeightPx: 480,
    nextPipeId: 5,
    lastSpawnedPipeGapPx: 120,
    lastSpawnedPipeGapCenterYPx: 220,
    lastSpawnedPipeSpawnIntervalFrames: 40,
    framesUntilNextPipeSpawn: 12,
    pipes: [
      {
        id: 4,
        xPx: 320,
        gapCenterYPx: 220,
        gapSizePx: 120,
      },
    ],
    birds: [
      {
        network: {} as Network,
        observationMemoryState: createSharedObservationMemoryState(),
        yPx: 144,
        velocityYPxPerFrame: 2,
        pipesPassed: 3,
        framesSurvived: 18,
        passedPipeIds: new Set<number>([1, 2, 3]),
        done: false,
      },
    ],
  };
}
