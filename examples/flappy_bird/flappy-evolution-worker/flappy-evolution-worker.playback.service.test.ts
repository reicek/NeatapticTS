import { createXorshift32 } from '../rng';
import { processWorkerPlaybackStep } from './flappy-evolution-worker.playback.service';

type MockNetwork = {
  clone: jest.Mock<MockNetwork, []>;
  toJSON: jest.Mock<{ connections: unknown[] }, []>;
};

describe('processWorkerPlaybackStep', () => {
  describe('when the completed playback winner clears the browser success target', () => {
    it('downshifts the future browser population budget before the next generation', () => {
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
      processWorkerPlaybackStep({
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
        stepPopulationFrame: () => 0,
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
      }).toEqual({
        elitism: 2,
        populationSize: 8,
        done: true,
        winnerPipesPassed: 10,
      });
    });
  });

  describe('when the completed playback winner stays below the browser success target', () => {
    it('keeps the existing browser population budget for the next generation', () => {
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
      processWorkerPlaybackStep({
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
        stepPopulationFrame: () => 0,
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
});

function createMockNetwork(): MockNetwork {
  return {
    clone: jest.fn(() => createMockNetwork()),
    toJSON: jest.fn(() => ({ connections: [] })),
  };
}

function createPlaybackBird(options: {
  network: MockNetwork;
  pipesPassed: number;
  framesSurvived: number;
  done: boolean;
}) {
  return {
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