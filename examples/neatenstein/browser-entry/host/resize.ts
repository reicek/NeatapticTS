/**
 * Host-side canvas resize handling for the Neatenstein demo.
 *
 * When the host canvas changes size or the renderer tier changes, this module
 * re-derives the column stride and allocates a fresh Structure-of-Arrays (SoA)
 * render frame sized to the tier's column count. The stride tells the host how
 * many CSS pixels each renderer column covers, which is needed to place
 * overlays and scale input coordinates correctly.
 *
 * @module
 */

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  NEATENSTEIN_WORKER_COLUMN_COUNT,
  type NeatensteinTier,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  type NeatensteinRenderFrame,
  type NeatensteinRenderState,
} from '../renderer/frame';

/** Fixed mapping from renderer tier to column count. */
const TIER_COLUMN_COUNTS: Record<NeatensteinTier, number> = {
  gpu: NEATENSTEIN_GPU_COLUMN_COUNT,
  worker: NEATENSTEIN_WORKER_COLUMN_COUNT,
  cpu: NEATENSTEIN_CPU_COLUMN_COUNT,
};

/**
 * Result of a host resize operation.
 *
 * @property columnStride - Number of CSS pixels per renderer column.
 * @property frame - Fresh render frame sized to the tier's column count.
 */
export interface NeatensteinResizeResult {
  /** Number of CSS pixels per renderer column for the chosen tier. */
  columnStride: number;
  /** Fresh, tier-sized render frame. */
  frame: NeatensteinRenderFrame;
}

/**
 * Handle a host canvas resize by choosing the tier column count and allocating
 * a matching frame.
 *
 * @param state - Current canvas size and simulation tick.
 * @param tier - Renderer tier to size the frame for.
 * @returns The derived column stride and a freshly allocated frame.
 *
 * @example
 * ```ts
 * const result = handleNeatensteinResize(
 *   {
 *     canvasWidth: 640,
 *     canvasHeight: 360,
 *     simTick: 1,
 *     cameraX: 12.5,
 *     cameraY: 12.5,
 *     cameraYaw: 0.25,
 *     mapSeed: 42,
 *   },
 *   'cpu',
 * );
 * // result.columnStride === 4 for the CPU tier at 640px width.
 * ```
 */
export function handleNeatensteinResize(
  state: NeatensteinRenderState,
  tier: NeatensteinTier,
): NeatensteinResizeResult {
  if (
    !Number.isFinite(state.canvasWidth) ||
    !Number.isFinite(state.canvasHeight)
  ) {
    throw new Error(
      `Canvas dimensions must be finite numbers, got width=${String(state.canvasWidth)}, height=${String(state.canvasHeight)}`,
    );
  }
  if (state.canvasWidth <= 0 || state.canvasHeight <= 0) {
    throw new Error(
      `Canvas dimensions must be positive, got width=${state.canvasWidth}, height=${state.canvasHeight}`,
    );
  }

  const columnCount = resolveColumnCount(tier);
  const columnStride = state.canvasWidth / columnCount;
  const frame = buildNeatensteinRenderFrame(state, columnCount);

  return { columnStride, frame };
}

/**
 * Map a renderer tier to its fixed column count.
 *
 * @param tier - Renderer tier to resolve.
 * @returns The column count for the tier.
 * @throws {Error} When the tier is not one of the supported values.
 */
function resolveColumnCount(tier: NeatensteinTier): number {
  const count = TIER_COLUMN_COUNTS[tier];
  if (count === undefined) {
    throw new Error(`Unknown render tier: ${String(tier)}`);
  }
  return count;
}
