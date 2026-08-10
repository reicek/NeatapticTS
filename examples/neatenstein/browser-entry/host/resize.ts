/**
 * Host-side canvas resize handling for the Neatenstein demo.
 *
 * When the host canvas changes size or the renderer tier changes, this module
 * re-derives the renderer column count, column stride, and a fresh
 * Structure-of-Arrays (SoA) render frame.
 *
 * Packed CPU/GPU tiers can use fixed tier column counts because they produce
 * frame data that is later blitted/scaled by the host. The direct worker tier,
 * however, renders into an {@link OffscreenCanvas}; for that path, the render
 * column count should match the active canvas backing-store width so wall
 * columns are not stretched across multiple pixels.
 *
 * @module
 */

import {
  NEATENSTEIN_CPU_COLUMN_COUNT,
  NEATENSTEIN_GPU_COLUMN_COUNT,
  type NeatensteinTier,
} from '../constants';
import {
  buildNeatensteinRenderFrame,
  type NeatensteinRenderFrame,
  type NeatensteinRenderState,
} from '../renderer/frame';

/**
 * Fixed mapping from packed-frame renderer tier to column count.
 *
 * The worker tier is intentionally excluded because direct worker rendering
 * should derive its ray column count from the actual canvas backing-store
 * width.
 */
const PACKED_TIER_COLUMN_COUNTS: Partial<Record<NeatensteinTier, number>> = {
  gpu: NEATENSTEIN_GPU_COLUMN_COUNT,
  cpu: NEATENSTEIN_CPU_COLUMN_COUNT,
};

/**
 * Result of a host resize operation.
 */
export interface NeatensteinResizeResult {
  /**
   * Number of backing-store pixels per renderer column.
   *
   * For the direct worker tier this should normally be `1`, because the worker
   * renders one ray column per backing-store pixel.
   */
  columnStride: number;

  /** Fresh render frame sized to the resolved column count. */
  frame: NeatensteinRenderFrame;
}

/**
 * Validate the canvas dimensions carried by a render state.
 *
 * @param state - Render state to validate.
 * @throws {Error} When dimensions are not finite positive numbers.
 */
function assertValidCanvasDimensions(state: NeatensteinRenderState): void {
  if (
    !Number.isFinite(state.canvasWidth) ||
    !Number.isFinite(state.canvasHeight)
  ) {
    throw new Error(
      `Canvas dimensions must be finite numbers, got width=${String(
        state.canvasWidth,
      )}, height=${String(state.canvasHeight)}`,
    );
  }

  if (state.canvasWidth <= 0 || state.canvasHeight <= 0) {
    throw new Error(
      `Canvas dimensions must be positive, got width=${state.canvasWidth}, height=${state.canvasHeight}`,
    );
  }
}

/**
 * Resolve the render column count for a tier and canvas size.
 *
 * CPU/GPU packed-frame tiers use fixed tier-specific column counts. The direct
 * worker tier uses the backing-store width so it can render one raycast column
 * per horizontal pixel.
 *
 * @param tier - Renderer tier to resolve.
 * @param canvasWidth - Active canvas backing-store width in pixels.
 * @returns Column count for the render path.
 * @throws {Error} When the tier is unknown.
 */
function resolveColumnCount(
  tier: NeatensteinTier,
  canvasWidth: number,
): number {
  if (tier === 'worker') {
    // Match the direct worker ray density to the backing-store width.
    return Math.max(1, Math.floor(canvasWidth));
  }

  const count = PACKED_TIER_COLUMN_COUNTS[tier];

  if (count === undefined) {
    throw new Error(`Unknown render tier: ${String(tier)}`);
  }

  return count;
}

/**
 * Handle a host canvas resize by choosing the renderer column count and
 * allocating a matching frame.
 *
 * @param state - Current canvas size and simulation tick.
 * @param tier - Renderer tier to size the frame for.
 * @returns The derived column stride and a freshly allocated frame.
 *
 * @example
 * ```ts
 * const result = handleNeatensteinResize(
 *   {
 *     canvasWidth: 1280,
 *     canvasHeight: 960,
 *     simTick: 1,
 *     cameraX: 12.5,
 *     cameraY: 12.5,
 *     cameraYaw: 0.25,
 *     mapSeed: 42,
 *   },
 *   'worker',
 * );
 *
 * console.log(result.columnStride);
 * // 1
 * ```
 */
export function handleNeatensteinResize(
  state: NeatensteinRenderState,
  tier: NeatensteinTier,
): NeatensteinResizeResult {
  assertValidCanvasDimensions(state);

  const columnCount = resolveColumnCount(tier, state.canvasWidth);
  const columnStride = state.canvasWidth / columnCount;
  const frame = buildNeatensteinRenderFrame(state, columnCount);

  return { columnStride, frame };
}
