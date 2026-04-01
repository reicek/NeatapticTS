import type { PopulationRenderState } from '../../browser-entry.types';
import { resolveLeaderFramesSurvived } from './playback.snapshot.summary.utils';

describe('resolveLeaderFramesSurvived', () => {
  it('returns the maximum survived-frame count across all birds', () => {
    const renderState = createRenderState();

    expect(resolveLeaderFramesSurvived(renderState)).toBe(17);
  });
});

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
    birds: [
      {
        yPx: 10,
        pipesPassed: 1,
        framesSurvived: 4,
        done: false,
      },
      {
        yPx: 20,
        pipesPassed: 2,
        framesSurvived: 17,
        done: true,
      },
      {
        yPx: 30,
        pipesPassed: 0,
        framesSurvived: 9,
        done: false,
      },
    ],
  };
}
