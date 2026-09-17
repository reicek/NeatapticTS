import type Network from '../../../src/architecture/network';
import type { InferenceChannel } from '../../../src/neataptic';
import { createSharedObservationMemoryState } from '../flappy.simulation.shared.utils';
import {
  createWorkerPlaybackSnapshot,
  resolveWorkerPlaybackSnapshotTransferList,
} from './flappy-evolution-worker.snapshot.utils';
import type {
  WorkerPlaybackFrameSnapshot,
  WorkerPlaybackState,
} from './flappy-evolution-worker.types';

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
      winnerNodeActivations: new Float32Array([0.25, -0.5]),
    });
  });

  it('returns all typed-array buffers in the transfer list', () => {
    const snapshot = createWorkerPlaybackSnapshot(createPlaybackState());
    const transferList = resolveWorkerPlaybackSnapshotTransferList(snapshot);
    const winnerNodeActivations = readWinnerNodeActivations(snapshot);

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
      winnerActivationsBuffer:
        winnerNodeActivations !== undefined &&
        transferList[7] === winnerNodeActivations.buffer,
      transferListLength: transferList.length,
    }).toEqual({
      birdDoneFlagsBuffer: true,
      birdFramesSurvivedBuffer: true,
      birdPipesPassedBuffer: true,
      birdYPositionsBuffer: true,
      pipeGapCenterBuffer: true,
      pipeGapSizesBuffer: true,
      pipeXPositionsBuffer: true,
      winnerActivationsBuffer: true,
      transferListLength: 8,
    });
  });

  it('packs the frame winner post-step node activations for winner streaming', () => {
    const playbackState = createWinnerActivationPlaybackState();

    // Bird index 1 owns the highest alive pipesPassed score, so the packed
    // activation stream must describe the winner bird only (not bird index 0).
    expect(
      readWinnerNodeActivations(createWorkerPlaybackSnapshot(playbackState)),
    ).toEqual(new Float32Array([0.75, 1.5]));
  });

  it('repopulates node activations when the frame winner used an inference channel', () => {
    const playbackState = createInferenceChannelPlaybackState();
    const firstSnapshot = createWorkerPlaybackSnapshot(playbackState);
    const firstActivations = readWinnerNodeActivations(firstSnapshot);

    // Change the bird and pipe geometry so the observation vector on the
    // next snapshot differs, proving the activation pass is recomputed each
    // frame rather than copying stale values.
    playbackState.birds[0].yPx = 300;
    playbackState.pipes[0].xPx = 100;
    playbackState.pipes[0].gapCenterYPx = 350;

    const secondSnapshot = createWorkerPlaybackSnapshot(playbackState);
    const secondActivations = readWinnerNodeActivations(secondSnapshot);

    expect({
      firstDefined: firstActivations !== undefined,
      secondDefined: secondActivations !== undefined,
      firstHasNonZero: firstActivations?.some(
        (activation) => activation !== 0,
      ),
      secondHasNonZero: secondActivations?.some(
        (activation) => activation !== 0,
      ),
      activationsDiffer:
        firstActivations !== undefined &&
        secondActivations !== undefined &&
        !firstActivations.every(
          (value, index) => value === secondActivations![index],
        ),
    }).toEqual({
      firstDefined: true,
      secondDefined: true,
      firstHasNonZero: true,
      secondHasNonZero: true,
      activationsDiffer: true,
    });
  });

  it('reuses shared snapshot buffers when cross-origin isolation allows shared memory', () => {
    const originalCrossOriginIsolated = Object.getOwnPropertyDescriptor(
      globalThis,
      'crossOriginIsolated',
    );
    const playbackState = createPlaybackState();

    Object.defineProperty(globalThis, 'crossOriginIsolated', {
      configurable: true,
      value: true,
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
        network: createWinnerActivationNetwork([0.25, -0.5]),
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
 * Builds a two-bird playback state whose frame winner (bird index 1, highest
 * alive pipesPassed score) carries a distinct node activation stream so the
 * winner-selection contract is observable.
 *
 * @returns Worker playback state containing one pipe and two living birds.
 */
function createWinnerActivationPlaybackState(): WorkerPlaybackState {
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
        network: createWinnerActivationNetwork([0.25, -0.5]),
        observationMemoryState: createSharedObservationMemoryState(),
        yPx: 144,
        velocityYPxPerFrame: 2,
        pipesPassed: 3,
        framesSurvived: 18,
        passedPipeIds: new Set<number>([1, 2, 3]),
        done: false,
      },
      {
        network: createWinnerActivationNetwork([0.75, 1.5]),
        observationMemoryState: createSharedObservationMemoryState(),
        yPx: 200,
        velocityYPxPerFrame: 2,
        pipesPassed: 5,
        framesSurvived: 30,
        passedPipeIds: new Set<number>([1, 2, 3, 4, 5]),
        done: false,
      },
    ],
  };
}

/**
 * Builds a mock network exposing post-step node activation values so the
 * winner-activation packing has a deterministic surface to read.
 *
 * @param activations - Activation value per node, in node order.
 * @returns Mock network carrying the supplied node activations.
 */
function createWinnerActivationNetwork(activations: number[]): Network {
  return {
    nodes: activations.map((activation) => ({ activation })),
  } as unknown as Network;
}

/**
 * Builds a playback state whose only bird is steered by a transferable
 * inference channel. The underlying network starts with zero activations and
 * exposes an `activate` implementation that writes the observation vector into
 * node activations. This reproduces the real failure mode where the channel
 * fast path leaves `network.nodes` untouched.
 *
 * @returns Worker playback state exercising the inference-channel path.
 */
function createInferenceChannelPlaybackState(): WorkerPlaybackState {
  const playbackState = createPlaybackState();
  const bird = playbackState.birds[0];
  bird.network = createInferenceChannelNetwork();
  bird.inferenceChannel = {
    close: jest.fn().mockResolvedValue(undefined),
    isOpen: true,
    predict: jest.fn().mockResolvedValue(new Float64Array([0.5])),
    reset: jest.fn().mockResolvedValue(undefined),
    strategy: 'channel',
  } as unknown as InferenceChannel;
  return playbackState;
}

/**
 * Builds a mock network whose node activations are populated by the input
 * observation vector. This mirrors the real `Network.activate` side effect that
 * the snapshot packer relies on when the winner used a transferable inference
 * channel.
 *
 * @returns Mock network with an observation-driven `activate` side effect.
 */
function createInferenceChannelNetwork(): Network {
  const network = {
    nodes: [{ activation: 0 }, { activation: 0 }, { activation: 0 }],
    activate(input: number[]): number[] {
      network.nodes.forEach((node, nodeIndex) => {
        node.activation = (input[nodeIndex] ?? 0) + 0.1 * (nodeIndex + 1);
      });
      return network.nodes.map((node) => node.activation);
    },
  } as unknown as Network;
  return network;
}

/**
 * Reads the packed winner activation stream from a snapshot.
 *
 * @param snapshot - Packed worker playback snapshot.
 * @returns Winner node activation stream, or undefined when absent.
 */
function readWinnerNodeActivations(
  snapshot: WorkerPlaybackFrameSnapshot,
): Float32Array | undefined {
  return (snapshot as { winnerNodeActivations?: Float32Array })
    .winnerNodeActivations;
}

/**
 * Restores a temporary global property override used by shared-memory tests.
 *
 * @param propertyName - Global property name that was overridden.
 * @param descriptor - Original descriptor captured before the override.
 * @returns Nothing.
 */
function restoreGlobalProperty(
  propertyName: string,
  descriptor: PropertyDescriptor | undefined,
): void {
  if (descriptor) {
    Object.defineProperty(globalThis, propertyName, descriptor);
    return;
  }

  Reflect.deleteProperty(globalThis, propertyName);
}
