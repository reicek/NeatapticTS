import {
  FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_SIZE_RATIO,
  FLAPPY_STARTUP_PREVIEW_LEGEND_MAX_FONT_SIZE_PX,
  FLAPPY_STARTUP_PREVIEW_LEGEND_MIN_FONT_SIZE_PX,
} from '../../constants/constants';
import { clamp, clamp01 } from '../browser-entry.math.utils';

/**
 * Resolved visual state for one startup-preview render frame.
 */
export interface RuntimeStartupPreviewVisualState {
  opacity: number;
  frameIndex: number;
  scrollBasePx: number;
  legendFontSizePx: number;
  exitAnimationCompleted: boolean;
}

/**
 * Resolves the visual state needed to draw one startup-preview frame.
 *
 * The preview needs only a few derived values: the current opacity for the
 * fade-in/fade-out lifecycle, a deterministic frame index and scroll offset for
 * the animated background, and a responsive font size that stays readable on
 * both compact and wide canvases.
 *
 * @param input - Timing, canvas size, and scroll-speed inputs for the preview frame.
 * @returns Resolved visual state for the current startup-preview frame.
 */
export function resolveRuntimeStartupPreviewVisualState(input: {
  nowMs: number;
  previewStartTimeMs: number;
  previewExitStartTimeMs?: number;
  canvasWidthPx: number;
  canvasHeightPx: number;
  pipeScrollSpeedPxPerFrame: number;
}): RuntimeStartupPreviewVisualState {
  // Step 1: Resolve elapsed startup time and the deterministic background clock.
  const elapsedPreviewTimeMs = Math.max(
    0,
    input.nowMs - input.previewStartTimeMs,
  );
  const continuousFrameIndex =
    elapsedPreviewTimeMs / FLAPPY_STARTUP_PREVIEW_FRAME_DURATION_MS;

  // Step 2: Combine fade-in and optional fade-out into one stable opacity.
  const fadeInOpacity = clamp01(
    elapsedPreviewTimeMs / FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
  );
  const fadeOutOpacity =
    input.previewExitStartTimeMs === undefined
      ? 1
      : 1 -
        clamp01(
          (input.nowMs - input.previewExitStartTimeMs) /
            FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
        );
  const opacity = fadeInOpacity * fadeOutOpacity;

  // Step 3: Resolve background scroll and responsive legend sizing.
  return {
    opacity,
    frameIndex: Math.max(0, Math.floor(continuousFrameIndex)),
    scrollBasePx: continuousFrameIndex * input.pipeScrollSpeedPxPerFrame,
    legendFontSizePx: resolveRuntimeStartupPreviewLegendFontSizePx(
      input.canvasWidthPx,
      input.canvasHeightPx,
    ),
    exitAnimationCompleted:
      input.previewExitStartTimeMs !== undefined &&
      input.nowMs - input.previewExitStartTimeMs >=
        FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
  };
}

/**
 * Resolves a responsive legend font size for the startup preview.
 *
 * @param canvasWidthPx - Current preview canvas width.
 * @param canvasHeightPx - Current preview canvas height.
 * @returns Responsive legend font size in pixels.
 */
function resolveRuntimeStartupPreviewLegendFontSizePx(
  canvasWidthPx: number,
  canvasHeightPx: number,
): number {
  // Step 1: Base the responsive text size on the smaller current canvas dimension.
  const smallerCanvasDimensionPx = Math.max(
    1,
    Math.min(canvasWidthPx, canvasHeightPx),
  );

  // Step 2: Clamp the result into the configured readable range.
  return Math.round(
    clamp(
      smallerCanvasDimensionPx * FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_SIZE_RATIO,
      FLAPPY_STARTUP_PREVIEW_LEGEND_MIN_FONT_SIZE_PX,
      FLAPPY_STARTUP_PREVIEW_LEGEND_MAX_FONT_SIZE_PX,
    ),
  );
}
