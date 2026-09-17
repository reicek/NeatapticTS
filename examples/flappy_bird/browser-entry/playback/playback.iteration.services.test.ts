import { emitChampionChangedEvent, emitPlaybackFrameStats } from './playback.iteration.services';
import type { PlaybackFrameStats } from '../browser-entry.types';
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

describe('emitPlaybackFrameStats', () => {
  it('forwards the winner activation stream and winner bird index to the frame-stats callback', () => {
    // Arrange
    let capturedStats: PlaybackFrameStats | undefined;
    const iterationContext = createPlaybackIterationContext({
      birds: [
        createBird({ pipesPassed: 2, framesSurvived: 20 }),
        createBird({ pipesPassed: 1, framesSurvived: 10 }),
      ],
      currentChampionBirdIndex: 0,
      onFrameStats: (stats) => {
        capturedStats = stats;
      },
    });

    // Act
    emitPlaybackFrameStats(
      iterationContext,
      createPlaybackStepPayloadWithWinnerStream(),
    );

    // Assert
    expect(
      readWinnerActivationStreamStats(capturedStats as PlaybackFrameStats),
    ).toEqual({
      winnerBirdIndex: 1,
      winnerNodeActivations: new Float32Array([0.25, -0.75]),
    });
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
  onFrameStats?: (stats: PlaybackFrameStats) => void;
}): PlaybackIterationContext {
  return {
    canvas: {} as HTMLCanvasElement,
    context: {} as CanvasRenderingContext2D,
    evolutionWorker: {} as Worker,
    onFrameStats: options.onFrameStats ?? (() => undefined),
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

/**
 * Builds a worker playback step payload carrying the winner activation stream
 * so the frame-stats forwarding contract has a deterministic surface to read.
 *
 * @returns Playback step payload with winner activation stream fields attached.
 */
function createPlaybackStepPayloadWithWinnerStream(): Parameters<
  typeof emitPlaybackFrameStats
>[1] {
  return {
    requestId: 7,
    snapshot: {
      format: 'packed-v1',
      frameIndex: 21,
      cumulativePipeTravelPx: 84,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
      pipeCount: 0,
      birdCount: 2,
      pipes: {
        xPositionsPx: new Float32Array(),
        gapCenterYPositionsPx: new Float32Array(),
        gapSizesPx: new Float32Array(),
      },
      birds: {
        yPositionsPx: new Float32Array([32, 40]),
        pipesPassed: new Uint32Array([2, 1]),
        framesSurvived: new Uint32Array([20, 10]),
        doneFlags: new Uint8Array([0, 0]),
      },
    },
    instrumentation: {
      activationCallsPerFrame: 3,
      simulationStepsPerRaf: 1,
    },
    done: false,
    winnerBirdIndex: 1,
    winnerNodeActivations: new Float32Array([0.25, -0.75]),
  } as unknown as Parameters<typeof emitPlaybackFrameStats>[1];
}

/**
 * Reads the winner activation stream fields from frame stats.
 *
 * Returns a fresh projection containing only the winner stream fields so the
 * assertion can compare against a narrow expected object (toEqual treats
 * extra defined keys as mismatches, so returning the full stats object would
 * make the contract impossible to satisfy).
 *
 * @param stats - Frame stats emitted by the playback iteration.
 * @returns Winner activation stream fields, or undefined when absent.
 */
function readWinnerActivationStreamStats(stats: PlaybackFrameStats): {
  winnerBirdIndex?: number;
  winnerNodeActivations?: Float32Array;
} {
  const winnerActivationStreamStats = stats as unknown as {
    winnerBirdIndex?: number;
    winnerNodeActivations?: Float32Array;
  };
  return {
    winnerBirdIndex: winnerActivationStreamStats.winnerBirdIndex,
    winnerNodeActivations: winnerActivationStreamStats.winnerNodeActivations,
  };
}
