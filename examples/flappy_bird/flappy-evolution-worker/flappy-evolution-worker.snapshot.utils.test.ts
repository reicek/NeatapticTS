import type Network from '../../../src/architecture/network';
import { createSharedObservationMemoryState } from '../flappy.simulation.shared.utils';
import {
  createWorkerPlaybackSnapshot,
  resolveWorkerPlaybackSnapshotTransferList,
} from './flappy-evolution-worker.snapshot.utils';
import type { WorkerPlaybackState } from './flappy-evolution-worker.types';

/**
 * Snapshot transport tests for the worker playback packing helpers.
 *
 * These tests are intentionally narrow: they lock in the packed transport shape
 * and transfer-list ownership contract without depending on the full simulation
 * loop.
 */
describe('createWorkerPlaybackSnapshot', () => {
  it('packs playback state into typed arrays for worker transport', () => {
    const playbackState = createPlaybackState();

    expect(createWorkerPlaybackSnapshot(playbackState)).toEqual({
      format: 'packed-v1',
      frameIndex: 12,
      cumulativePipeTravelPx: 96,
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
    const transferList = resolveWorkerPlaybackSnapshotTransferList(snapshot);

    expect({
      birdDoneFlagsBuffer: transferList[6] === snapshot.birds.doneFlags.buffer,
      birdFramesSurvivedBuffer:
        transferList[5] === snapshot.birds.framesSurvived.buffer,
      birdPipesPassedBuffer:
        transferList[4] === snapshot.birds.pipesPassed.buffer,
      birdYPositionsBuffer:
        transferList[3] === snapshot.birds.yPositionsPx.buffer,
      pipeGapCenterBuffer:
        transferList[1] === snapshot.pipes.gapCenterYPositionsPx.buffer,
      pipeGapSizesBuffer: transferList[2] === snapshot.pipes.gapSizesPx.buffer,
      pipeXPositionsBuffer:
        transferList[0] === snapshot.pipes.xPositionsPx.buffer,
      transferListLength: transferList.length,
    }).toEqual({
      birdDoneFlagsBuffer: true,
      birdFramesSurvivedBuffer: true,
      birdPipesPassedBuffer: true,
      birdYPositionsBuffer: true,
      pipeGapCenterBuffer: true,
      pipeGapSizesBuffer: true,
      pipeXPositionsBuffer: true,
      transferListLength: 7,
    });
  });

  it('reuses shared snapshot buffers when cross-origin isolation allows shared memory', () => {
    const originalCrossOriginIsolated = Object.getOwnPropertyDescriptor(
      globalThis,
      'crossOriginIsolated',
    );
    const originalImportScripts = Object.getOwnPropertyDescriptor(
      globalThis,
      'importScripts',
    );
    const originalWorker = Object.getOwnPropertyDescriptor(
      globalThis,
      'Worker',
    );
    const playbackState = createPlaybackState();

    Object.defineProperty(globalThis, 'crossOriginIsolated', {
      configurable: true,
      value: true,
    });
    Object.defineProperty(globalThis, 'Worker', {
      configurable: true,
      value: class Worker {},
    });
    Object.defineProperty(globalThis, 'importScripts', {
      configurable: true,
      value: jest.fn(),
    });

    try {
      const firstSnapshot = createWorkerPlaybackSnapshot(playbackState);
      playbackState.birds[0].yPx = 188;
      playbackState.frameIndex = 13;
      const nextSnapshot = createWorkerPlaybackSnapshot(playbackState);

      expect({
        nextBirdYPx: nextSnapshot.birds.yPositionsPx[0],
        reusesBirdBuffer:
          firstSnapshot.birds.yPositionsPx.buffer ===
          nextSnapshot.birds.yPositionsPx.buffer,
        transferListLength:
          resolveWorkerPlaybackSnapshotTransferList(nextSnapshot).length,
      }).toEqual({
        nextBirdYPx: 188,
        reusesBirdBuffer: true,
        transferListLength: 0,
      });
    } finally {
      restoreGlobalProperty('crossOriginIsolated', originalCrossOriginIsolated);
      restoreGlobalProperty('importScripts', originalImportScripts);
      restoreGlobalProperty('Worker', originalWorker);
    }
  });
});

/**
 * Builds a minimal but realistic playback state fixture for snapshot tests.
 *
 * @returns Worker playback state containing one pipe and one living bird.
 */
function createPlaybackState(): WorkerPlaybackState {
  return {
    frameIndex: 12,
    cumulativePipeTravelPx: 96,
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

/**
 * Restores a temporary global property override used by shared-memory tests.
 *
 * @param propertyName - Global property name that was overridden.
 * @param descriptor - Original descriptor captured before the override.
 * @returns Nothing.
 */
function restoreGlobalProperty(
  propertyName: 'crossOriginIsolated' | 'importScripts' | 'Worker',
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) {
    Object.defineProperty(globalThis, propertyName, descriptor);
    return;
  }

  Reflect.deleteProperty(globalThis, propertyName);
}
