/**
 * Bootstrap utilities for the Neatenstein browser entrypoint.
 *
 * Pure leaf functions for worker URL resolution, capability detection, and
 * CPU-fallback canvas status rendering. These executors are stateless and
 * deterministic.
 *
 * @module bootstrap.utils
 */

import {
  NEATENSTEIN_CANVAS_2D_CONTEXT,
  NEATENSTEIN_FALLBACK_STATUS_TEXT_RGB,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
} from './constants';
import { NEATENSTEIN_BACKGROUND_RGB } from './renderer/framebuffer';

// ---------------------------------------------------------------------------
// Worker URL resolution
// ---------------------------------------------------------------------------

/**
 * Resolve the published worker URL relative to the host bundle script.
 *
 * The worker asset lives next to the host bundle under `docs/assets/`.
 * Resolving against the host script's `src` URL works correctly under any
 * server root.
 *
 * @param hostScript - The `<script>` element that loaded the IIFE bundle,
 *   captured at evaluation time via `document.currentScript`.
 * @returns Absolute worker URL string.
 * @throws {Error} When `hostScript` is not a valid `HTMLScriptElement`.
 */
export function resolveWorkerUrl(hostScript: unknown): string {
  if (!hostScript || !(hostScript instanceof HTMLScriptElement)) {
    throw new Error(
      'Neatenstein bundle must be loaded through a <script> tag so the worker URL can be resolved relative to the host bundle.',
    );
  }
  return new URL(NEATENSTEIN_WORKER_BUNDLE_FILENAME, hostScript.src).href;
}

// ---------------------------------------------------------------------------
// Capability detection
// ---------------------------------------------------------------------------

/**
 * Runtime capability check for the OffscreenCanvas worker transfer path.
 *
 * Returns `true` only when both the visible canvas can produce an
 * {@link OffscreenCanvas} and the worker-side type is present.
 *
 * @returns `true` when OffscreenCanvas worker transfer is supported.
 */
export function supportsWorkerOffscreenCanvas(): boolean {
  return (
    typeof HTMLCanvasElement !== 'undefined' &&
    typeof OffscreenCanvas !== 'undefined' &&
    typeof HTMLCanvasElement.prototype.transferControlToOffscreen === 'function'
  );
}

// ---------------------------------------------------------------------------
// Canvas status rendering (CPU fallback)
// ---------------------------------------------------------------------------

/**
 * Format an RGB triple as a CSS `rgb(...)` string.
 *
 * Small local helper so the fallback status path can share the same canonical
 * background color as the worker/CPU/GPU renderers.
 *
 * @param color - RGB color triple.
 * @returns CSS `rgb(r, g, b)` string.
 */
export function formatRgb(color: { r: number; g: number; b: number }): string {
  return `rgb(${Math.round(color.r)}, ${Math.round(color.g)}, ${Math.round(color.b)})`;
}

/**
 * Draw a centered status message on the visible canvas.
 *
 * Used by the CPU fallback path when OffscreenCanvas is unavailable so the
 * page shows an explanation instead of a blank screen.
 *
 * @param canvas - Visible host canvas to draw on.
 * @param message - Status message to center on the canvas.
 */
export function drawCanvasStatus(
  canvas: HTMLCanvasElement,
  message: string,
): void {
  const context = canvas.getContext(NEATENSTEIN_CANVAS_2D_CONTEXT);
  if (!context) {
    return;
  }

  context.fillStyle = formatRgb(NEATENSTEIN_BACKGROUND_RGB);
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = formatRgb(NEATENSTEIN_FALLBACK_STATUS_TEXT_RGB);
  context.font = '0.875rem system-ui, Segoe UI, Arial, sans-serif';
  context.textAlign = 'center';
  context.textBaseline = 'middle';
  context.fillText(message, canvas.width / 2, canvas.height / 2);
}
