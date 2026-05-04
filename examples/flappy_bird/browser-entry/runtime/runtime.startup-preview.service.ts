import {
  FLAPPY_GENERATION_PREVIEW_HOLD_DURATION_MS,
  FLAPPY_MONOSPACE_FONT_FAMILY,
  FLAPPY_NEON_PALETTE,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX,
  FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR,
  FLAPPY_PIPE_OUTLINE_GLOW_ALPHA,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS,
  FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_WEIGHT,
  FLAPPY_STARTUP_PREVIEW_LEGEND_TEXT,
} from '../../constants/constants';
import { renderPlaybackBackground } from '../playback/background/playback.background';
import { PlaybackAnimationFrameUnavailableError } from '../playback/playback.errors';
import type { RuntimeStartupPreviewHandle } from './runtime.types';
import { resolveRuntimeStartupPreviewVisualState } from './runtime.startup-preview.utils';

type RuntimeStartupPreviewOptions = {
  canvas: HTMLCanvasElement;
  context: CanvasRenderingContext2D;
  isStopped: () => boolean;
};

type RuntimeGenerationPreviewOptions = RuntimeStartupPreviewOptions & {
  legendText: string;
};

type RuntimeLegendPreviewOptions = RuntimeStartupPreviewOptions & {
  legendText: string;
  previewStartTimeMs?: number;
  scheduledExitStartTimeMs?: number;
};

/**
 * Starts the animated browser startup preview shown before the first birds appear.
 *
 * This preview owns only the initial loading window. It reuses the existing
 * background renderer so the canvas stays alive during worker initialization,
 * then fades a centered loading legend out before the first playback session is
 * allowed to begin.
 *
 * @param options - Canvas, context, and stop-state inputs for the startup preview.
 * @returns Handle used to begin the exit transition or stop the preview outright.
 */
export function startRuntimeStartupPreview(
  options: RuntimeStartupPreviewOptions,
): RuntimeStartupPreviewHandle {
  return startRuntimeLegendPreview({
    ...options,
    legendText: FLAPPY_STARTUP_PREVIEW_LEGEND_TEXT,
  });
}

/**
 * Starts the centered evolving-preview shown between completed generations.
 *
 * This preview keeps the simulation canvas visibly alive while the worker is
 * evolving the next generation and there are intentionally no birds to render.
 * Unlike the first-load startup preview, this one is driven by the request
 * lifecycle of the next generation rather than by initial boot.
 *
 * @param options - Canvas, context, stop-state, and evolving legend inputs.
 * @returns Handle used to fade the overlay out once the next generation is ready.
 */
export function startRuntimeEvolvingPreview(
  options: RuntimeGenerationPreviewOptions,
): RuntimeStartupPreviewHandle {
  return startRuntimeLegendPreview(options);
}

/**
 * Presents a short generation title card before playback begins.
 *
 * Every generation-ready event can reuse the same background-plus-neon style as
 * the startup loading screen, but with a bounded fade-in, hold, and fade-out
 * lifecycle that automatically completes before playback starts.
 *
 * @param options - Canvas, context, stop-state, and generation legend inputs.
 * @returns Nothing.
 */
export async function presentRuntimeGenerationPreview(
  options: RuntimeGenerationPreviewOptions,
): Promise<void> {
  // Step 1: Resolve one shared start timestamp for the timed presentation.
  const generationPreviewStartTimeMs = resolveRuntimeStartupPreviewNowMs();
  const generationPreviewHandle = startRuntimeLegendPreview({
    ...options,
    previewStartTimeMs: generationPreviewStartTimeMs,
    scheduledExitStartTimeMs:
      generationPreviewStartTimeMs +
      FLAPPY_STARTUP_PREVIEW_FADE_DURATION_MS +
      FLAPPY_GENERATION_PREVIEW_HOLD_DURATION_MS,
  });

  // Step 2: Wait for the timed presentation lifecycle to finish, then stop the loop.
  await finalizeRuntimeLegendPreview(generationPreviewHandle);
}

/**
 * Starts a legend-driven runtime preview loop on the main playback canvas.
 *
 * This shared helper powers both the startup loading screen and the bounded
 * generation title cards. The difference is whether the fade-out is scheduled
 * up front or started later by calling `complete()`.
 *
 * @param options - Canvas, legend text, and preview lifecycle inputs.
 * @returns Handle used to complete or stop the preview.
 */
function startRuntimeLegendPreview(
  options: RuntimeLegendPreviewOptions,
): RuntimeStartupPreviewHandle {
  // Step 1: Fail fast when the environment cannot drive animation frames.
  if (
    typeof requestAnimationFrame !== 'function' ||
    typeof cancelAnimationFrame !== 'function'
  ) {
    throw new PlaybackAnimationFrameUnavailableError();
  }

  const previewStartTimeMs =
    options.previewStartTimeMs ?? resolveRuntimeStartupPreviewNowMs();
  let previewExitStartTimeMs = options.scheduledExitStartTimeMs;
  let pendingAnimationFrameId: number | undefined;
  let stopped = false;
  let resolveDone: (() => void) | undefined;
  const done = new Promise<void>((resolve) => {
    resolveDone = resolve;
  });

  // Step 2: Paint the first frame immediately so the canvas never appears blank.
  renderRuntimeLegendPreviewFrame(previewStartTimeMs);

  return {
    complete: () => {
      // Step 3: Start the fade-out only once, then return the shared completion promise.
      if (!stopped && previewExitStartTimeMs === undefined) {
        previewExitStartTimeMs = resolveRuntimeStartupPreviewNowMs();
      }
      return done;
    },
    stop: stopLegendPreview,
  };

  /**
   * Paints the current startup-preview frame and schedules the next one.
   *
   * @param nowMs - Current animation-frame timestamp in milliseconds.
   * @returns Nothing.
   */
  function renderRuntimeLegendPreviewFrame(nowMs: number): void {
    // Step 5: Stop the preview as soon as the runtime itself has been stopped.
    if (stopped || options.isStopped()) {
      stopLegendPreview();
      return;
    }

    const visualState = resolveRuntimeStartupPreviewVisualState({
      nowMs,
      previewStartTimeMs,
      previewExitStartTimeMs,
      canvasWidthPx: options.canvas.width,
      canvasHeightPx: options.canvas.height,
      pipeScrollSpeedPxPerFrame: FLAPPY_PIPE_SPEED_PX_PER_FRAME,
    });

    paintRuntimeStartupPreviewFrame(options.context, {
      canvasWidthPx: options.canvas.width,
      canvasHeightPx: options.canvas.height,
      legendText: options.legendText,
      opacity: visualState.opacity,
      frameIndex: visualState.frameIndex,
      scrollBasePx: visualState.scrollBasePx,
      legendFontSizePx: visualState.legendFontSizePx,
    });

    // Step 6: Resolve completion after the exit animation has fully finished.
    if (
      previewExitStartTimeMs !== undefined &&
      visualState.exitAnimationCompleted
    ) {
      stopLegendPreview();
      return;
    }

    // Step 7: Queue the next animation frame while the preview remains active.
    pendingAnimationFrameId = requestAnimationFrame(
      renderRuntimeLegendPreviewFrame,
    );
  }

  /**
   * Stops the active preview loop and resolves its completion promise.
   *
   * @returns Nothing.
   */
  function stopLegendPreview(): void {
    // Step 8: Keep teardown idempotent and cancel any queued animation frame.
    if (stopped) {
      return;
    }

    stopped = true;
    if (pendingAnimationFrameId !== undefined) {
      cancelAnimationFrame(pendingAnimationFrameId);
      pendingAnimationFrameId = undefined;
    }
    resolveDone?.();
  }
}

/**
 * Waits for a runtime legend preview to finish and then stops its animation loop.
 *
 * @param legendPreviewHandle - Active preview handle.
 * @returns Nothing.
 */
async function finalizeRuntimeLegendPreview(
  legendPreviewHandle: RuntimeStartupPreviewHandle,
): Promise<void> {
  try {
    await legendPreviewHandle.complete();
  } finally {
    legendPreviewHandle.stop();
  }
}

/**
 * Paints one full startup-preview frame.
 *
 * @param context - Target canvas 2D context.
 * @param input - Render-ready startup-preview frame values.
 * @returns Nothing.
 */
function paintRuntimeStartupPreviewFrame(
  context: CanvasRenderingContext2D,
  input: {
    canvasWidthPx: number;
    canvasHeightPx: number;
    legendText: string;
    opacity: number;
    frameIndex: number;
    scrollBasePx: number;
    legendFontSizePx: number;
  },
): void {
  // Step 1: Reset the canvas into a neutral identity-transform paint state.
  context.save();
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.globalAlpha = 1;
  context.globalCompositeOperation = 'source-over';
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.clearRect(0, 0, input.canvasWidthPx, input.canvasHeightPx);

  // Step 2: Reuse the normal playback background so the loading scene stays familiar.
  renderPlaybackBackground(context, {
    viewportLeftXPx: 0,
    visibleWorldWidthPx: input.canvasWidthPx,
    visibleWorldHeightPx: input.canvasHeightPx,
    frameIndex: input.frameIndex,
    scrollBasePx: input.scrollBasePx,
  });

  // Step 3: Paint the centered legend above the animated background.
  renderRuntimeStartupPreviewLegend(context, {
    canvasWidthPx: input.canvasWidthPx,
    canvasHeightPx: input.canvasHeightPx,
    legendText: input.legendText,
    opacity: input.opacity,
    legendFontSizePx: input.legendFontSizePx,
  });
  context.restore();
}

/**
 * Draws the centered neon loading legend over the startup preview canvas.
 *
 * @param context - Target canvas 2D context.
 * @param input - Canvas geometry and resolved opacity for the legend.
 * @returns Nothing.
 */
function renderRuntimeStartupPreviewLegend(
  context: CanvasRenderingContext2D,
  input: {
    canvasWidthPx: number;
    canvasHeightPx: number;
    legendText: string;
    opacity: number;
    legendFontSizePx: number;
  },
): void {
  // Step 1: Skip text work entirely once the legend is fully transparent.
  if (input.opacity <= 0) {
    return;
  }

  const legendLines = input.legendText.split('\n');
  const legendCenterXPx = input.canvasWidthPx * 0.5;
  const legendCenterYPx = input.canvasHeightPx * 0.5;
  const legendLineHeightPx = input.legendFontSizePx * 1.15;
  const legendLineBlockHeightPx = (legendLines.length - 1) * legendLineHeightPx;
  context.font = `${FLAPPY_STARTUP_PREVIEW_LEGEND_FONT_WEIGHT} ${input.legendFontSizePx}px ${FLAPPY_MONOSPACE_FONT_FAMILY}`;
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillStyle = FLAPPY_NEON_PALETTE.pipeFill;

  // Step 2: Paint the soft additive glow pass using the same pipe glow treatment.
  context.globalCompositeOperation = 'lighter';
  context.globalAlpha = input.opacity * FLAPPY_PIPE_OUTLINE_GLOW_ALPHA;
  context.shadowColor = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_COLOR;
  context.shadowBlur = FLAPPY_PIPE_OUTLINE_CYAN_GLOW_BLUR_PX;
  legendLines.forEach((legendLine, legendLineIndex) => {
    context.fillText(
      legendLine,
      legendCenterXPx,
      legendCenterYPx -
        legendLineBlockHeightPx * 0.5 +
        legendLineIndex * legendLineHeightPx,
    );
  });

  // Step 3: Restore a crisp core text pass above the glow bloom.
  context.globalCompositeOperation = 'source-over';
  context.globalAlpha = input.opacity;
  context.shadowBlur = 0;
  context.shadowColor = 'transparent';
  legendLines.forEach((legendLine, legendLineIndex) => {
    context.fillText(
      legendLine,
      legendCenterXPx,
      legendCenterYPx -
        legendLineBlockHeightPx * 0.5 +
        legendLineIndex * legendLineHeightPx,
    );
  });
}

/**
 * Resolves the best available startup-preview clock source.
 *
 * @returns Current timestamp in milliseconds.
 */
function resolveRuntimeStartupPreviewNowMs(): number {
  if (
    typeof performance === 'object' &&
    typeof performance.now === 'function'
  ) {
    return performance.now();
  }

  return Date.now();
}
