import { createXorshift32 } from '../rng';
import { resolveObservationVector } from '../browser-entry/browser-entry.observation.utils';
import { resolvePipeSpawnXPx } from '../browser-entry/browser-entry.viewport.utils';
import {
  exportTransferableInferencePayload,
  openInferenceChannel,
} from '../../../src/neataptic';
import {
  FLAPPY_BIRD_VIEWPORT_X_RATIO,
  FLAPPY_BIRD_X_PX,
  FLAPPY_PIPE_WIDTH_PX,
} from '../constants/constants';
import { resolveAdaptiveDifficultyProfile } from '../flappy.simulation.shared.utils';
import { createWorkerPopulationRenderState } from './flappy-evolution-worker.simulation.utils';

jest.mock('../../../src/neataptic', () => ({
  exportTransferableInferencePayload: jest.fn(() => ({
    strategy: 'transferable',
  })),
  openInferenceChannel: jest.fn(() => ({
    close: jest.fn(async () => undefined),
    predict: jest.fn(async () => new Float64Array([0.25, 0.75])),
    reset: jest.fn(async () => undefined),
  })),
}));

describe('createWorkerPopulationRenderState', () => {
  it('clears carried network state before a fresh playback session starts', () => {
    const firstClear = jest.fn();
    const secondClear = jest.fn();

    createWorkerPopulationRenderState(
      [{ clear: firstClear }, { clear: secondClear }] as never[],
      createXorshift32(12345),
      1280,
      720,
    );

    expect({
      first: firstClear.mock.calls.length,
      second: secondClear.mock.calls.length,
    }).toEqual({
      first: 1,
      second: 1,
    });
  });

  it('keeps the initial pipe inside the immediate visible horizon on wide viewports', () => {
    // Arrange
    const visibleWorldWidthPx = 2560;
    const visibleWorldHeightPx = 720;
    const renderState = createWorkerPopulationRenderState(
      [{ clear: jest.fn() }] as never[],
      createXorshift32(12345),
      visibleWorldWidthPx,
      visibleWorldHeightPx,
    );
    const initialPipe = renderState.pipes[0];
    const primaryBird = renderState.birds[0];
    const visibleRightXPx =
      FLAPPY_BIRD_X_PX +
      visibleWorldWidthPx * (1 - FLAPPY_BIRD_VIEWPORT_X_RATIO);
    const respawnXPx = resolvePipeSpawnXPx(visibleWorldWidthPx);
    const observation = resolveObservationVector(
      primaryBird!.yPx,
      primaryBird!.velocityYPxPerFrame,
      renderState.pipes,
      visibleWorldWidthPx,
      visibleWorldHeightPx,
      resolveAdaptiveDifficultyProfile(0, 1),
      renderState.lastSpawnedPipeSpawnIntervalFrames,
      primaryBird!.observationMemoryState,
    );

    // Assert
    expect({
      initialPipeXPx: initialPipe?.xPx,
      initialPipeOffsetFromVisibleRightPx:
        (initialPipe?.xPx ?? 0) - visibleRightXPx,
      respawnPipeXPx: respawnXPx,
      nextPipeStaysWithinObservationHorizon:
        observation.observationFeatures.normalizedDistanceToNextPipe < 1,
    }).toEqual({
      initialPipeXPx: visibleRightXPx - FLAPPY_PIPE_WIDTH_PX,
      initialPipeOffsetFromVisibleRightPx: -FLAPPY_PIPE_WIDTH_PX,
      respawnPipeXPx: visibleRightXPx + FLAPPY_PIPE_WIDTH_PX,
      nextPipeStaysWithinObservationHorizon: true,
    });
  });

  it('opens one persistent inference channel per bird when a worker bundle URL is provided', () => {
    const firstChannel = { close: jest.fn(async () => undefined) };
    const secondChannel = { close: jest.fn(async () => undefined) };

    (openInferenceChannel as jest.Mock)
      .mockReturnValueOnce(firstChannel)
      .mockReturnValueOnce(secondChannel);

    const renderState = createWorkerPopulationRenderState(
      [{ clear: jest.fn() }, { clear: jest.fn() }] as never[],
      createXorshift32(12345),
      1280,
      720,
      'flappy-inference-channel.worker.bundle.js',
    );

    expect({
      exportCalls: (exportTransferableInferencePayload as jest.Mock).mock.calls
        .length,
      firstBirdHasChannel: renderState.birds[0]?.inferenceChannel != null,
      openCalls: (openInferenceChannel as jest.Mock).mock.calls.length,
      secondBirdHasChannel: renderState.birds[1]?.inferenceChannel != null,
    }).toEqual({
      exportCalls: 2,
      firstBirdHasChannel: true,
      openCalls: 2,
      secondBirdHasChannel: true,
    });
  });
});
