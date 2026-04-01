import {
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS,
} from '../../constants/constants';
import { resolveRuntimeStartupPreviewVisualState } from './runtime.startup-preview.utils';

describe('resolveRuntimeStartupPreviewVisualState', () => {
  it('returns half opacity midway through the fade-in window', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: 150,
      previewStartTimeMs: 0,
      canvasWidthPx: 640,
      canvasHeightPx: 480,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.opacity).toBeCloseTo(0.5, 6);
  });

  it('holds full opacity until the scheduled fade-out window begins', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: 450,
      previewStartTimeMs: 0,
      previewExitStartTimeMs: 600,
      canvasWidthPx: 640,
      canvasHeightPx: 480,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.opacity).toBeCloseTo(1, 6);
  });

  it('marks the exit animation complete after the fade-out duration elapses', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: 650,
      previewStartTimeMs: 0,
      previewExitStartTimeMs: 350,
      canvasWidthPx: 640,
      canvasHeightPx: 480,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.exitAnimationCompleted).toBe(true);
  });

  it('advances preview scroll using the same pipe-speed frame clock as playback', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: 1_000,
      previewStartTimeMs: 0,
      canvasWidthPx: 640,
      canvasHeightPx: 480,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.scrollBasePx).toBeCloseTo(
      (1_000 / FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS) *
        FLAPPY_PIPE_SPEED_PX_PER_FRAME,
      6,
    );
  });
});
