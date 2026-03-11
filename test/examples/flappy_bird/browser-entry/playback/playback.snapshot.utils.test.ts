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
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
      nextPipeId: 0,
      lastSpawnedPipeGapPx: 0,
      lastSpawnedPipeGapCenterYPx: 0,
      lastSpawnedPipeSpawnIntervalFrames: 0,
      framesUntilNextPipeSpawn: 0,
      pipes: [
        {
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
});

function createSnapshot(): EvolutionPlaybackStepSnapshot {
  return {
    format: 'packed-v1',
    frameIndex: 21,
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
    visibleWorldWidthPx: 1,
    visibleWorldHeightPx: 1,
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [
      {
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
