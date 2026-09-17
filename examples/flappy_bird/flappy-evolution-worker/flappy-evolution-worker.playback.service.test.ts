import { createXorshift32 } from '../rng';
import { createWorkerPlaybackSnapshot } from './flappy-evolution-worker.snapshot.utils';
import { processWorkerPlaybackStep } from './flappy-evolution-worker.playback.service';

type MockNetwork = {
  clone: jest.Mock<MockNetwork, []>;
  toJSON: jest.Mock<{ connections: unknown[] }, []>;
};

type WinnerActivationNetwork = MockNetwork & {
  nodes: { activation: number }[];
};

describe('processWorkerPlaybackStep', () => {
  describe('when the completed playback winner clears the browser success target', () => {
    it('downshifts the future browser population budget before the next generation', async () => {
      // Arrange
      const postWorkerMessage = jest.fn();
      const runtimeNetwork = createMockNetwork();
      const currentPopulation = [runtimeNetwork, createMockNetwork()];
      const neatRuntime = {
        options: {
          popsize: 30,
          elitism: 6,
        },
        population: [...currentPopulation],
      };

      // Act
      await processWorkerPlaybackStep({
        playbackStepPayload: {
          requestId: 1,
          simulationSteps: 1,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
        },
        currentPlaybackState: {
          frameIndex: 12,
          cumulativePipeTravelPx: 0,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
          nextPipeId: 0,
          lastSpawnedPipeGapPx: 0,
          lastSpawnedPipeGapCenterYPx: 0,
          lastSpawnedPipeSpawnIntervalFrames: 0,
          framesUntilNextPipeSpawn: 0,
          pipes: [],
          birds: [
            createPlaybackBird({
              network: runtimeNetwork,
              pipesPassed: 10,
              framesSurvived: 200,
              done: true,
            }),
            createPlaybackBird({
              network: createMockNetwork(),
              pipesPassed: 3,
              framesSurvived: 140,
              done: true,
            }),
          ],
        } as never,
        currentPlaybackRng: createXorshift32(12345),
        currentPopulation: currentPopulation as never,
        neatRuntime: neatRuntime as never,
        stepPopulationFrame: async () => 0,
        createPlaybackSnapshot: () => ({
          format: 'packed-v1',
          frameIndex: 12,
          cumulativePipeTravelPx: 0,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
          pipeCount: 0,
          birdCount: 2,
          pipes: {
            xPositionsPx: new Float32Array(),
            gapCenterYPositionsPx: new Float32Array(),
            gapSizesPx: new Float32Array(),
          },
          birds: {
            yPositionsPx: new Float32Array(),
            pipesPassed: new Uint32Array(),
            framesSurvived: new Uint32Array(),
            doneFlags: new Uint8Array(),
          },
        }),
        resolvePlaybackSnapshotTransferList: () => [],
        postWorkerMessage,
      });

      // Assert
      expect({
        elitism: neatRuntime.options.elitism,
        populationSize: neatRuntime.options.popsize,
        done: postWorkerMessage.mock.calls.at(-1)?.[0]?.payload?.done,
        winnerPipesPassed:
          postWorkerMessage.mock.calls.at(-1)?.[0]?.payload?.winnerPipesPassed,
        winnerNetworkJson:
          postWorkerMessage.mock.calls.at(-1)?.[0]?.payload?.winnerNetworkJson,
      }).toEqual({
        elitism: 2,
        populationSize: 8,
        done: true,
        winnerPipesPassed: 10,
        winnerNetworkJson: { connections: [] },
      });
    });
  });

  describe('when the completed playback winner stays below the browser success target', () => {
    it('keeps the existing browser population budget for the next generation', async () => {
      // Arrange
      const postWorkerMessage = jest.fn();
      const runtimeNetwork = createMockNetwork();
      const neatRuntime = {
        options: {
          popsize: 30,
          elitism: 6,
        },
        population: [runtimeNetwork],
      };

      // Act
      await processWorkerPlaybackStep({
        playbackStepPayload: {
          requestId: 1,
          simulationSteps: 1,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
        },
        currentPlaybackState: {
          frameIndex: 12,
          cumulativePipeTravelPx: 0,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
          nextPipeId: 0,
          lastSpawnedPipeGapPx: 0,
          lastSpawnedPipeGapCenterYPx: 0,
          lastSpawnedPipeSpawnIntervalFrames: 0,
          framesUntilNextPipeSpawn: 0,
          pipes: [],
          birds: [
            createPlaybackBird({
              network: runtimeNetwork,
              pipesPassed: 9,
              framesSurvived: 180,
              done: true,
            }),
          ],
        } as never,
        currentPlaybackRng: createXorshift32(12345),
        currentPopulation: [runtimeNetwork] as never,
        neatRuntime: neatRuntime as never,
        stepPopulationFrame: async () => 0,
        createPlaybackSnapshot: () => ({
          format: 'packed-v1',
          frameIndex: 12,
          cumulativePipeTravelPx: 0,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
          pipeCount: 0,
          birdCount: 1,
          pipes: {
            xPositionsPx: new Float32Array(),
            gapCenterYPositionsPx: new Float32Array(),
            gapSizesPx: new Float32Array(),
          },
          birds: {
            yPositionsPx: new Float32Array(),
            pipesPassed: new Uint32Array(),
            framesSurvived: new Uint32Array(),
            doneFlags: new Uint8Array(),
          },
        }),
        resolvePlaybackSnapshotTransferList: () => [],
        postWorkerMessage,
      });

      // Assert
      expect({
        elitism: neatRuntime.options.elitism,
        populationSize: neatRuntime.options.popsize,
        winnerPipesPassed:
          postWorkerMessage.mock.calls.at(-1)?.[0]?.payload?.winnerPipesPassed,
      }).toEqual({
        elitism: 6,
        populationSize: 30,
        winnerPipesPassed: 9,
      });
    });
  });

  describe('when playback completes with persistent inference channels', () => {
    it('closes each bird channel before retiring the playback state', async () => {
      const firstClose = jest.fn(async () => undefined);
      const secondClose = jest.fn(async () => undefined);

      await processWorkerPlaybackStep({
        playbackStepPayload: {
          requestId: 1,
          simulationSteps: 1,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
        },
        currentPlaybackState: {
          frameIndex: 12,
          cumulativePipeTravelPx: 0,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
          nextPipeId: 0,
          lastSpawnedPipeGapPx: 0,
          lastSpawnedPipeGapCenterYPx: 0,
          lastSpawnedPipeSpawnIntervalFrames: 0,
          framesUntilNextPipeSpawn: 0,
          pipes: [],
          birds: [
            createPlaybackBird({
              network: createMockNetwork(),
              pipesPassed: 2,
              framesSurvived: 40,
              done: true,
              inferenceChannel: { close: firstClose },
            }),
            createPlaybackBird({
              network: createMockNetwork(),
              pipesPassed: 1,
              framesSurvived: 30,
              done: true,
              inferenceChannel: { close: secondClose },
            }),
          ],
        } as never,
        currentPlaybackRng: createXorshift32(12345),
        currentPopulation: [createMockNetwork()] as never,
        neatRuntime: undefined,
        stepPopulationFrame: async () => 0,
        createPlaybackSnapshot: () => ({
          format: 'packed-v1',
          frameIndex: 12,
          cumulativePipeTravelPx: 0,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
          pipeCount: 0,
          birdCount: 2,
          pipes: {
            xPositionsPx: new Float32Array(),
            gapCenterYPositionsPx: new Float32Array(),
            gapSizesPx: new Float32Array(),
          },
          birds: {
            yPositionsPx: new Float32Array(),
            pipesPassed: new Uint32Array(),
            framesSurvived: new Uint32Array(),
            doneFlags: new Uint8Array(),
          },
        }),
        resolvePlaybackSnapshotTransferList: () => [],
        postWorkerMessage: jest.fn(),
      });

      expect({
        firstCloseCalls: firstClose.mock.calls.length,
        secondCloseCalls: secondClose.mock.calls.length,
      }).toEqual({
        firstCloseCalls: 1,
        secondCloseCalls: 1,
      });
    });
  });

  describe('when live playback frames stream the winner activation snapshot', () => {
    it('carries the frame winner activations and winner bird index on each live frame', async () => {
      // Arrange
      const postWorkerMessage = jest.fn();
      const winnerBirdNetwork = createWinnerActivationNetwork([0.75, 1.5]);
      const runnerUpBirdNetwork = createWinnerActivationNetwork([0.25, -0.5]);
      const runnerUpBird = createPlaybackBird({
        network: runnerUpBirdNetwork,
        pipesPassed: 3,
        framesSurvived: 18,
        done: false,
      }) as {
        network: WinnerActivationNetwork;
        pipesPassed: number;
        framesSurvived: number;
      };
      const currentPlaybackState = {
        frameIndex: 12,
        cumulativePipeTravelPx: 0,
        visibleWorldWidthPx: 1280,
        visibleWorldHeightPx: 720,
        nextPipeId: 0,
        lastSpawnedPipeGapPx: 0,
        lastSpawnedPipeGapCenterYPx: 0,
        lastSpawnedPipeSpawnIntervalFrames: 0,
        framesUntilNextPipeSpawn: 0,
        pipes: [],
        birds: [
          runnerUpBird,
          createPlaybackBird({
            network: winnerBirdNetwork,
            pipesPassed: 5,
            framesSurvived: 30,
            done: false,
          }),
        ],
      } as never;

      // Act — two live frames with the simulation advancing between them, so
      // the streamed winner fields must change with each frame.
      await processWorkerPlaybackStep({
        playbackStepPayload: {
          requestId: 1,
          simulationSteps: 1,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
        },
        currentPlaybackState,
        currentPlaybackRng: createXorshift32(12345),
        currentPopulation: [createMockNetwork()] as never,
        neatRuntime: undefined,
        stepPopulationFrame: async () => 0,
        createPlaybackSnapshot: createWorkerPlaybackSnapshot,
        resolvePlaybackSnapshotTransferList: () => [],
        postWorkerMessage,
      });

      runnerUpBird.pipesPassed = 9;
      runnerUpBird.framesSurvived = 60;
      runnerUpBird.network.nodes[0].activation = 2.5;

      await processWorkerPlaybackStep({
        playbackStepPayload: {
          requestId: 2,
          simulationSteps: 1,
          visibleWorldWidthPx: 1280,
          visibleWorldHeightPx: 720,
        },
        currentPlaybackState,
        currentPlaybackRng: createXorshift32(12345),
        currentPopulation: [createMockNetwork()] as never,
        neatRuntime: undefined,
        stepPopulationFrame: async () => 0,
        createPlaybackSnapshot: createWorkerPlaybackSnapshot,
        resolvePlaybackSnapshotTransferList: () => [],
        postWorkerMessage,
      });

      // Assert
      const firstPayload = postWorkerMessage.mock.calls.at(0)?.[0]?.payload as
        | {
            done?: boolean;
            winnerBirdIndex?: number;
            winnerNodeActivations?: Float32Array;
          }
        | undefined;
      const secondPayload = postWorkerMessage.mock.calls.at(1)?.[0]?.payload as
        | {
            done?: boolean;
            winnerBirdIndex?: number;
            winnerNodeActivations?: Float32Array;
          }
        | undefined;

      expect({
        firstFrameDone: firstPayload?.done,
        firstFrameWinnerBirdIndex: firstPayload?.winnerBirdIndex,
        firstFrameWinnerNodeActivations: firstPayload?.winnerNodeActivations,
      }).toEqual({
        firstFrameDone: false,
        firstFrameWinnerBirdIndex: 1,
        firstFrameWinnerNodeActivations: new Float32Array([0.75, 1.5]),
      });

      expect({
        secondFrameDone: secondPayload?.done,
        secondFrameWinnerBirdIndex: secondPayload?.winnerBirdIndex,
        secondFrameWinnerNodeActivations: secondPayload?.winnerNodeActivations,
      }).toEqual({
        secondFrameDone: false,
        secondFrameWinnerBirdIndex: 0,
        secondFrameWinnerNodeActivations: new Float32Array([2.5, -0.5]),
      });
    });
  });
});

function createMockNetwork(): MockNetwork {
  return {
    clone: jest.fn(() => createMockNetwork()),
    toJSON: jest.fn(() => ({ connections: [] })),
  };
}

/**
 * Builds a mock network exposing post-step node activation values so the
 * per-frame winner activation streaming has a deterministic surface to read.
 *
 * @param activations - Activation value per node, in node order.
 * @returns Mock network carrying the supplied node activations.
 */
function createWinnerActivationNetwork(
  activations: number[],
): WinnerActivationNetwork {
  return {
    clone: jest.fn(() => createMockNetwork()),
    toJSON: jest.fn(() => ({ connections: [] })),
    nodes: activations.map((activation) => ({ activation })),
  };
}

function createPlaybackBird(options: {
  network: MockNetwork;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
  inferenceChannel?: {
    close: jest.Mock<Promise<void>, []>;
  };
}) {
  return {
    inferenceChannel: options.inferenceChannel,
    network: options.network,
    observationMemoryState: {
      previousCoreObservationFrames: [],
      recentFlapActions: [],
    },
    yPx: 0,
    velocityYPxPerFrame: 0,
    pipesPassed: options.pipesPassed,
    framesSurvived: options.framesSurvived,
    passedPipeIds: new Set<number>(),
    done: options.done,
  } as never;
}
