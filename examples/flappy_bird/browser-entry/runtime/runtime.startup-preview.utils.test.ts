import {
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_LEGEND_MAX_FONT_SIZE_PX,
  FLAPPY_STARTUP_PREVIEW_LEGEND_MIN_FONT_SIZE_PX,
} from '../../constants/constants';
import { resolveRuntimeStartupPreviewVisualState } from './runtime.startup-preview.utils';

describe('resolveRuntimeStartupPreviewVisualState', () => {
  it('returns half opacity midway through the fade-in window', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS / 2,
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

  it('clamps legend font size to the minimum on a very small canvas', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: 0,
      previewStartTimeMs: 0,
      canvasWidthPx: 50,
      canvasHeightPx: 50,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.legendFontSizePx).toBe(FLAPPY_STARTUP_PREVIEW_LEGEND_MIN_FONT_SIZE_PX);
  });

  it('clamps legend font size to the maximum on a very large canvas', () => {
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs: 0,
      previewStartTimeMs: 0,
      canvasWidthPx: 2_000,
      canvasHeightPx: 2_000,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.legendFontSizePx).toBe(FLAPPY_STARTUP_PREVIEW_LEGEND_MAX_FONT_SIZE_PX);
  });

  it('returns half opacity halfway through the fade-out window', () => {
    const nowMs = 1_000;
    const previewExitStartTimeMs = nowMs - FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS / 2;
    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs,
      previewStartTimeMs: 0,
      previewExitStartTimeMs,
      canvasWidthPx: 640,
      canvasHeightPx: 480,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    expect(visualState.opacity).toBeCloseTo(0.5, 6);
  });
});
