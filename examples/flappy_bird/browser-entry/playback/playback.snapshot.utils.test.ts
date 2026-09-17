import type {
  EvolutionPlaybackStepSnapshot,
  PopulationRenderState,
} from '../browser-entry.types';
import { applyPlaybackSnapshot } from './playback.snapshot.utils';

describe('applyPlaybackSnapshot', () => {
  it('reuses the existing bird objects while decoding the packed snapshot', () => {
    const snapshot = createSnapshot();
    const renderState = createRenderStateWithOneBirdAndOnePipe();
    const originalBirdReference = renderState.birds[0];

    applyPlaybackSnapshot(renderState, snapshot);

    expect(renderState.birds[0]).toBe(originalBirdReference);
  });

  it('decodes packed bird and pipe scalars into the reusable render state', () => {
    const snapshot = createSnapshot();
    const renderState = createRenderState();

    applyPlaybackSnapshot(renderState, snapshot);

    expect(renderState).toEqual({
      frameIndex: 21,
      cumulativePipeTravelPx: 84,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
      nextPipeId: 0,
      lastSpawnedPipeGapPx: 0,
      lastSpawnedPipeGapCenterYPx: 0,
      lastSpawnedPipeSpawnIntervalFrames: 0,
      framesUntilNextPipeSpawn: 0,
      pipes: [
        {
          id: 0,
          xPx: 250,
          gapCenterYPx: 200,
          gapSizePx: 110,
        },
      ],
      birds: [
        {
          yPx: 150,
          pipesPassed: 2,
          framesSurvived: 17,
          done: false,
        },
      ],
    });
  });

  it('surfaces the streamed winner activations and winner bird index on the render state', () => {
    const snapshot = createSnapshotWithWinnerActivationStream();
    const renderState = createRenderState();

    applyPlaybackSnapshot(renderState, snapshot);

    expect(readWinnerActivationStreamState(renderState)).toEqual({
      winnerBirdIndex: 0,
      winnerNodeActivations: new Float32Array([0.5, -1.5, 2]),
    });
  });

  it('clears stale winner activation state when the snapshot omits the winner stream', () => {
    const snapshot = createSnapshot();
    const renderState = createRenderStateWithStaleWinnerActivationStream();

    applyPlaybackSnapshot(renderState, snapshot);

    expect(readWinnerActivationStreamState(renderState)).toEqual({
      winnerBirdIndex: undefined,
      winnerNodeActivations: undefined,
    });
  });
});

function createSnapshot(): EvolutionPlaybackStepSnapshot {
  return {
    format: 'packed-v1',
    frameIndex: 21,
    cumulativePipeTravelPx: 84,
    visibleWorldWidthPx: 640,
    visibleWorldHeightPx: 480,
    pipeCount: 1,
    birdCount: 1,
    pipes: {
      xPositionsPx: new Float32Array([250]),
      gapCenterYPositionsPx: new Float32Array([200]),
      gapSizesPx: new Float32Array([110]),
    },
    birds: {
      yPositionsPx: new Float32Array([150]),
      pipesPassed: new Uint32Array([2]),
      framesSurvived: new Uint32Array([17]),
      doneFlags: new Uint8Array([0]),
    },
  };
}

function createRenderStateWithOneBirdAndOnePipe(): PopulationRenderState {
  return {
    frameIndex: 0,
    cumulativePipeTravelPx: 0,
    visibleWorldWidthPx: 1,
    visibleWorldHeightPx: 1,
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [
      {
        id: 0,
        xPx: 1,
        gapCenterYPx: 2,
        gapSizePx: 3,
      },
    ],
    birds: [
      {
        yPx: 1,
        pipesPassed: 2,
        framesSurvived: 3,
        done: true,
      },
    ],
  };
}

function createRenderState(): PopulationRenderState {
  return {
    frameIndex: 0,
    cumulativePipeTravelPx: 0,
    visibleWorldWidthPx: 1,
    visibleWorldHeightPx: 1,
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [],
    birds: [],
  };
}

/**
 * Builds a packed snapshot carrying the winner activation stream so the
 * unpack contract has a deterministic surface to read.
 *
 * @returns Packed snapshot with winner activation stream fields attached.
 */
function createSnapshotWithWinnerActivationStream(): EvolutionPlaybackStepSnapshot {
  return {
    ...createSnapshot(),
    winnerBirdIndex: 0,
    winnerNodeActivations: new Float32Array([0.5, -1.5, 2]),
  } as unknown as EvolutionPlaybackStepSnapshot;
}

/**
 * Builds a render state pre-seeded with stale winner activation values so the
 * unpack contract can prove legacy snapshots clear the stale stream.
 *
 * @returns Render state carrying stale winner activation state.
 */
function createRenderStateWithStaleWinnerActivationStream(): PopulationRenderState {
  const renderState = createRenderState();
  const staleWinnerStream = renderState as {
    winnerBirdIndex?: number;
    winnerNodeActivations?: Float32Array;
  };

  staleWinnerStream.winnerBirdIndex = 1;
  staleWinnerStream.winnerNodeActivations = new Float32Array([9, 9, 9]);
  return renderState;
}

/**
 * Reads the winner activation stream fields from a render state.
 *
 * Returns a fresh projection containing only the winner stream fields so the
 * assertion can compare against a narrow expected object (toEqual treats
 * extra defined keys as mismatches, so returning the full render state would
 * make the contract impossible to satisfy).
 *
 * @param renderState - Population render state under test.
 * @returns Winner activation stream fields, or undefined when absent.
 */
function readWinnerActivationStreamState(renderState: PopulationRenderState): {
  winnerBirdIndex?: number;
  winnerNodeActivations?: Float32Array;
} {
  const winnerActivationStreamState = renderState as {
    winnerBirdIndex?: number;
    winnerNodeActivations?: Float32Array;
  };
  return {
    winnerBirdIndex: winnerActivationStreamState.winnerBirdIndex,
    winnerNodeActivations: winnerActivationStreamState.winnerNodeActivations,
  };
}
