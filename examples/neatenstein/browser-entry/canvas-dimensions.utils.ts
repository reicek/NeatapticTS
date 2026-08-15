/**
 * Canvas dimensions utilities for the Neatenstein browser entrypoint.
 *
 * Pure leaf functions for computing CSS-derived render dimensions, applying
 * them to the canvas backing store, and reacting to resize events. These
 * executors are stateless and deterministic.
 *
 * @module canvas-dimensions.utils
 */

import { NEATENSTEIN_FALLBACK_CANVAS_WIDTH } from './constants';
import type { NeatensteinRendererBridge } from './host/renderer-bridge';

/**
 * Fixed render height in pixels used for the canvas backing store and for
 * the worker-tier render dimensions forwarded in simState.
 */
export const NEATENSTEIN_FIXED_RENDER_HEIGHT = 480;

// ---------------------------------------------------------------------------
// Dimension computation
// ---------------------------------------------------------------------------

/**
 * Compute the CSS-derived render dimensions for the visible canvas.
 *
 * The width is proportional to the canvas CSS box aspect ratio so the
 * rendered scene is never stretched, while the height stays fixed at 480px
 * to keep the raycaster projection math stable. Falls back to viewport
 * dimensions when client dimensions are zero, or to the fallback canvas
 * width when both are unavailable.
 *
 * @param element - Visible host canvas.
 * @returns Render width and height in pixels.
 */
export function resolveCanvasRenderDimensions(element: HTMLCanvasElement): {
  width: number;
  height: number;
} {
  const clientWidth = element.clientWidth;
  const clientHeight = element.clientHeight;

  if (clientWidth && clientHeight) {
    return {
      width: Math.round(
        NEATENSTEIN_FIXED_RENDER_HEIGHT * (clientWidth / clientHeight),
      ),
      height: NEATENSTEIN_FIXED_RENDER_HEIGHT,
    };
  }

  // Fall back to viewport dimensions when the canvas has not been laid out yet.
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const fallbackWidth =
    viewportWidth && viewportHeight
      ? Math.round(
          NEATENSTEIN_FIXED_RENDER_HEIGHT * (viewportWidth / viewportHeight),
        )
      : NEATENSTEIN_FALLBACK_CANVAS_WIDTH;

  return {
    width: fallbackWidth,
    height: NEATENSTEIN_FIXED_RENDER_HEIGHT,
  };
}

// ---------------------------------------------------------------------------
// Backing store application
// ---------------------------------------------------------------------------

/**
 * Apply computed render dimensions to the visible canvas backing store.
 *
 * Safe to call only when the canvas is still owned by the host (i.e., the
 * CPU fallback tier). The worker tier transfers the canvas to the worker,
 * after which direct width/height assignment throws.
 *
 * @param element - Visible host canvas.
 * @param dimensions - Render width and height in pixels.
 */
export function applyCanvasBackingStore(
  element: HTMLCanvasElement,
  dimensions: { width: number; height: number },
): void {
  element.width = dimensions.width;
  element.height = dimensions.height;
}

// ---------------------------------------------------------------------------
// Resize handling
// ---------------------------------------------------------------------------

/**
 * React to a change in the visible canvas CSS box.
 *
 * For the worker tier, the host canvas is transferred to the worker, so the
 * host cannot mutate its backing store. Instead the new dimensions are routed
 * to the bridge, which posts them to the worker. For the CPU fallback tier,
 * the host still owns the canvas and updates the backing store directly.
 *
 * @param params - Resize parameters.
 * @param params.canvas - Visible host canvas.
 * @param params.useWorkerTier - Whether the worker tier is active.
 * @param params.bridge - Renderer bridge (may be `null` before bridge creation).
 * @returns The new CSS-derived render dimensions.
 */
export function updateRendererSize(params: {
  canvas: HTMLCanvasElement;
  useWorkerTier: boolean;
  bridge: NeatensteinRendererBridge | null;
}): { width: number; height: number } {
  const dimensions = resolveCanvasRenderDimensions(params.canvas);

  if (params.useWorkerTier && params.bridge !== null) {
    params.bridge.resize(dimensions.width, dimensions.height);
    return dimensions;
  }

  applyCanvasBackingStore(params.canvas, dimensions);
  return dimensions;
}
