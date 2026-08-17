/**
 * Pure canvas utility executors extracted from the display worker.
 *
 * These helpers validate dimensions, synchronize the OffscreenCanvas backing
 * store, resolve column counts, and build the no-op sprite render context.
 * They are pure functions with no dependency on module-level mutable state.
 *
 * @module
 */

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  RENDER_TIER_GPU,
} from '../constants';
import type { NeatensteinSpriteRenderContext } from '../renderer/sprites';
import type { DisplayTier } from './display.worker.types';

/**
 * Return whether a value is usable as a positive render dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the dimension is positive and finite.
 */
export function isPositiveFiniteDimension(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Synchronize the transferred OffscreenCanvas with the host-provided render
 * size.
 *
 * This is critical for direct worker rendering. Projection math, floor/ceiling
 * drawing, wall stripes, z-buffer columns, and the actual canvas backing store
 * must agree on the same dimensions.
 *
 * @param canvas - Transferred worker-owned canvas.
 * @param width - Host-provided backing-store width.
 * @param height - Host-provided backing-store height.
 */
export function syncWorkerCanvasSize(
  canvas: OffscreenCanvas,
  width: number,
  height: number,
): void {
  if (canvas.width !== width) {
    canvas.width = width;
  }

  if (canvas.height !== height) {
    canvas.height = height;
  }
}

/**
 * Resolve the direct worker-tier raycast column count.
 *
 * The worker tier renders directly into the OffscreenCanvas, so its horizontal
 * ray density should match the host-provided backing-store width.
 *
 * @param canvasWidth - Canvas backing-store width in pixels.
 * @returns Number of direct worker raycast columns.
 */
export function resolveWorkerCanvasColumnCount(canvasWidth: number): number {
  return Math.max(1, Math.floor(canvasWidth));
}

/**
 * Map a packed-frame tier to the column count used in typed frame payloads.
 *
 * Direct worker rendering does not use this helper; it raycasts at the
 * host-provided canvas width instead.
 *
 * @param tier - Active renderer tier.
 * @returns Packed-frame column count.
 */
export function resolvePackedColumnCount(tier: DisplayTier): number {
  if (tier === RENDER_TIER_GPU) {
    return NEATENSTEIN_GPU_COLUMN_COUNT;
  }
  return NEATENSTEIN_CPU_COLUMN_COUNT;
}

/**
 * Build a canvas-like context that does not flush per sprite.
 *
 * The worker tier renders all sprites into a single canvas snapshot and flushes
 * it back once after the sprite pass. The renderer's `putImageData` contract
 * still expects a context, so this no-op adapter satisfies the type without
 * redundant per-sprite copies.
 *
 * @returns Canvas-like context with a no-op `putImageData`.
 */
export function buildNoOpSpriteRenderContext(): NeatensteinSpriteRenderContext {
  return {
    putImageData: () => {
      // Intentionally empty: the worker flushes the snapshot once after all
      // sprites are drawn.
    },
  };
}
