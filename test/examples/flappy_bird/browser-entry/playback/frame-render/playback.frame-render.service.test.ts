import type {
  PopulationRenderState,
  TrailState,
} from '../../../../../../test/examples/flappy_bird/browser-entry/browser-entry.types';
import { updateTrailState } from '../../../../../../test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.service';
import { renderPlaybackFrameBirds } from '../../../../../../test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.services';
import { renderPlaybackBird } from '../../../../../../test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.utils';
import { FLAPPY_CHAMPION_TRAIL_MAX_POINTS } from '../../../../../../test/examples/flappy_bird/constants/constants.birds';

describe('updateTrailState', () => {
  it('retains a trail for the current champion only', () => {
    const renderState = createRenderState({
      birds: [
        createBird({ pipesPassed: 1, framesSurvived: 10 }),
        createBird({ pipesPassed: 3, framesSurvived: 20 }),
      ],
    });
    const trailState: TrailState = {
      birdTrailsY: [[{ frameIndex: 4, yPx: 10 }], []],
    };

    updateTrailState(trailState, renderState);

    expect(trailState.birdTrailsY.map((birdTrail) => birdTrail.length)).toEqual([
      0,
      1,
    ]);
  });

  it('clears trail history for eliminated birds', () => {
    const renderState = createRenderState({
      birds: [createBird({ done: true, pipesPassed: 5, framesSurvived: 30 })],
    });
    const trailState: TrailState = {
      birdTrailsY: [[{ frameIndex: 7, yPx: 24 }]],
    };

    updateTrailState(trailState, renderState);

    expect(trailState.birdTrailsY[0].length).toBe(0);
  });

  it('caps champion trail history to the short champion limit', () => {
    const renderState = createRenderState({
      frameIndex: FLAPPY_CHAMPION_TRAIL_MAX_POINTS + 5,
      birds: [createBird({ pipesPassed: 2, framesSurvived: 12 })],
    });
    const trailState: TrailState = {
      birdTrailsY: [Array.from({ length: FLAPPY_CHAMPION_TRAIL_MAX_POINTS }, (_, frameIndex) => ({
        frameIndex,
        yPx: frameIndex,
      }))],
    };

    updateTrailState(trailState, renderState);

    expect(trailState.birdTrailsY[0].length).toBe(
      FLAPPY_CHAMPION_TRAIL_MAX_POINTS,
    );
  });

  it('keeps rendering all active birds while trails stay champion-only', () => {
    const renderBird = jest.fn();
    const renderState = createRenderState({
      birds: [
        createBird({ pipesPassed: 1, framesSurvived: 10 }),
        createBird({ pipesPassed: 3, framesSurvived: 20 }),
        createBird({ done: true, pipesPassed: 0, framesSurvived: 5 }),
      ],
    });

    renderPlaybackFrameBirds(
      {} as CanvasRenderingContext2D,
      renderState,
      createSceneContext(),
      renderBird,
    );

    expect(renderBird.mock.calls.map(([, , birdIndex]) => birdIndex)).toEqual([
      0,
      1,
    ]);
  });

  it('renders non-champion birds without champion highlight passes', () => {
    const drawingContext = createMockDrawingContext();

    renderPlaybackBird(drawingContext, 40, 0, 1);

    expect(drawingContext.strokeRect).not.toHaveBeenCalled();
  });

  it('keeps champion highlight passes for the current leader', () => {
    const drawingContext = createMockDrawingContext();

    renderPlaybackBird(drawingContext, 40, 1, 1);

    expect(drawingContext.strokeRect).toHaveBeenCalledTimes(1);
  });
});

function createRenderState(overrides?: {
  frameIndex?: number;
  birds?: PopulationRenderState['birds'];
}): PopulationRenderState {
  return {
    frameIndex: overrides?.frameIndex ?? 8,
    visibleWorldWidthPx: 640,
    visibleWorldHeightPx: 480,
    nextPipeId: 0,
    lastSpawnedPipeGapPx: 0,
    lastSpawnedPipeGapCenterYPx: 0,
    lastSpawnedPipeSpawnIntervalFrames: 0,
    framesUntilNextPipeSpawn: 0,
    pipes: [],
    birds: overrides?.birds ?? [],
  };
}

function createBird(overrides?: {
  yPx?: number;
  pipesPassed?: number;
  framesSurvived?: number;
  done?: boolean;
}): PopulationRenderState['birds'][number] {
  return {
    yPx: overrides?.yPx ?? 32,
    pipesPassed: overrides?.pipesPassed ?? 0,
    framesSurvived: overrides?.framesSurvived ?? 0,
    done: overrides?.done ?? false,
  };
}

function createSceneContext() {
  return {
    viewport: {
      offsetXPx: 0,
      offsetYPx: 0,
      scale: 1,
    },
    visibleWorldWidthPx: 640,
    visibleWorldHeightPx: 480,
    cameraLeftPx: 0,
    championBirdIndex: 1,
    edgeBounds: {
      leftXPx: 0,
      rightXPx: 640,
      topYPx: 0,
      bottomYPx: 480,
    },
  };
}

function createMockDrawingContext(): CanvasRenderingContext2D {
  return {
    fillRect: jest.fn(),
    strokeRect: jest.fn(),
    beginPath: jest.fn(),
    moveTo: jest.fn(),
    lineTo: jest.fn(),
    stroke: jest.fn(),
    globalAlpha: 1,
    globalCompositeOperation: 'source-over',
    fillStyle: '#000000',
    strokeStyle: '#000000',
    shadowColor: 'transparent',
    shadowBlur: 0,
    lineWidth: 1,
  } as unknown as CanvasRenderingContext2D;
}