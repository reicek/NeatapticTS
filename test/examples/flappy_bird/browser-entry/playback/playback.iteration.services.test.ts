import { emitChampionChangedEvent } from './playback.iteration.services';
import type { PlaybackIterationContext } from './playback.orchestration.types';

describe('emitChampionChangedEvent', () => {
  it('publishes an event when the first resolved red bird becomes available', () => {
    const championChangedEvents: Array<{
      championBirdIndex: number;
    }> = [];
    const iterationContext = createPlaybackIterationContext({
      birds: [
        createBird({ pipesPassed: 2, framesSurvived: 20 }),
        createBird({ pipesPassed: 1, framesSurvived: 10 }),
      ],
      currentChampionBirdIndex: -1,
      onChampionChanged: (event) => {
        championChangedEvents.push(event);
      },
    });

    emitChampionChangedEvent(iterationContext);

    expect(championChangedEvents).toEqual([
      {
        championBirdIndex: 0,
      },
    ]);
  });

  it('publishes an event when the red bird changes after the old champion dies', () => {
    const championChangedEvents: Array<{
      championBirdIndex: number;
    }> = [];
    const iterationContext = createPlaybackIterationContext({
      birds: [
        createBird({ pipesPassed: 2, framesSurvived: 12, done: true }),
        createBird({ pipesPassed: 1, framesSurvived: 20 }),
      ],
      currentChampionBirdIndex: 0,
      onChampionChanged: (event) => {
        championChangedEvents.push(event);
      },
    });

    emitChampionChangedEvent(iterationContext);

    expect(championChangedEvents).toEqual([
      {
        championBirdIndex: 1,
      },
    ]);
  });

  it('skips publication when the red bird stays the same', () => {
    const championChangedEvents: Array<{
      championBirdIndex: number;
    }> = [];
    const iterationContext = createPlaybackIterationContext({
      birds: [
        createBird({ pipesPassed: 2, framesSurvived: 12 }),
        createBird({ pipesPassed: 1, framesSurvived: 20 }),
      ],
      currentChampionBirdIndex: 0,
      onChampionChanged: (event) => {
        championChangedEvents.push(event);
      },
    });

    emitChampionChangedEvent(iterationContext);

    expect(championChangedEvents).toEqual([]);
  });
});

function createPlaybackIterationContext(options: {
  birds: Array<{
    yPx: number;
    pipesPassed: number;
    framesSurvived: number;
    done: boolean;
  }>;
  currentChampionBirdIndex: number;
  onChampionChanged?: (event: { championBirdIndex: number }) => void;
}): PlaybackIterationContext {
  return {
    canvas: {} as HTMLCanvasElement,
    context: {} as CanvasRenderingContext2D,
    evolutionWorker: {} as Worker,
    onFrameStats: () => undefined,
    onChampionChanged: options.onChampionChanged,
    sessionContext: {
      renderState: {
        frameIndex: 0,
        cumulativePipeTravelPx: 0,
        visibleWorldWidthPx: 640,
        visibleWorldHeightPx: 480,
        nextPipeId: 0,
        lastSpawnedPipeGapPx: 0,
        lastSpawnedPipeGapCenterYPx: 0,
        lastSpawnedPipeSpawnIntervalFrames: 0,
        framesUntilNextPipeSpawn: 0,
        pipes: [],
        birds: options.birds,
      },
      trailState: {
        birdTrailsY: [],
      },
      loopState: {
        simulationFrameBudget: 0,
        finished: false,
        currentChampionBirdIndex: options.currentChampionBirdIndex,
        summary: {
          averagePipesPassed: 0,
          p90FramesSurvived: 0,
          winnerPipesPassed: 0,
          winnerFramesSurvived: 0,
          latestLeaderPipesPassed: 0,
          latestLeaderFramesSurvived: 0,
        },
      },
    },
  };
}

function createBird(overrides?: {
  yPx?: number;
  pipesPassed?: number;
  framesSurvived?: number;
  done?: boolean;
}) {
  return {
    yPx: overrides?.yPx ?? 32,
    pipesPassed: overrides?.pipesPassed ?? 0,
    framesSurvived: overrides?.framesSurvived ?? 0,
    done: overrides?.done ?? false,
  };
}
