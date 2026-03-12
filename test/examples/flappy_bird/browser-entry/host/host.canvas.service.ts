/**
 * Canvas sizing helpers for the browser host boundary.
 *
 * These utilities keep host layout and backing-store sizing aligned so the
 * simulation and network canvases render crisply without stretching artifacts.
 */

/**
 * Applies a canvas backing store size and CSS width/height.
 *
 * Browser canvases have both backing-store dimensions and CSS box dimensions;
 * this helper updates both together.
 *
 * @param canvas - Target canvas element.
 * @param widthPx - Desired backing-store width in pixels.
 * @param heightPx - Desired backing-store height in pixels.
 * @returns True when canvas dimensions changed.
 */
export function applyCanvasBackingSize(
  canvas: HTMLCanvasElement,
  widthPx: number,
  heightPx: number,
): boolean {
  const nextWidthPx = Math.max(1, Math.floor(widthPx));
  const nextHeightPx = Math.max(1, Math.floor(heightPx));

  if (canvas.width === nextWidthPx && canvas.height === nextHeightPx) {
    return false;
  }

  canvas.width = nextWidthPx;
  canvas.height = nextHeightPx;
  canvas.style.width = `${nextWidthPx}px`;
  canvas.style.height = `${nextHeightPx}px`;
  return true;
}

/**
 * Applies fixed simulation-canvas bounds so layout does not stretch unexpectedly.
 *
 * The main simulation canvas uses fixed bounds because the world renderer is
 * tuned for a controlled viewport rather than fluid DOM stretching.
 *
 * @param canvas - Simulation canvas element.
 * @param widthPx - Desired width in pixels.
 * @param heightPx - Desired height in pixels.
 * @returns True when backing-store dimensions changed.
 */
export function applySimulationCanvasBounds(
  canvas: HTMLCanvasElement,
  widthPx: number,
  heightPx: number,
): boolean {
  const didResize = applyCanvasBackingSize(canvas, widthPx, heightPx);
  const resolvedWidthPx = Math.max(1, Math.floor(widthPx));
  const resolvedHeightPx = Math.max(1, Math.floor(heightPx));

  canvas.style.minWidth = `${resolvedWidthPx}px`;
  canvas.style.maxWidth = `${resolvedWidthPx}px`;
  canvas.style.minHeight = `${resolvedHeightPx}px`;
  canvas.style.maxHeight = `${resolvedHeightPx}px`;
  return didResize;
}

/**
 * Computes the drawable network canvas size from host element dimensions.
 *
 * The side-panel network view needs the drawable size after panel insets are
 * accounted for, not just the raw host client box.
 *
 * @param networkCanvasHost - Host element wrapping the network canvas.
 * @param hostInsetPx - Total inset to subtract from both dimensions.
 * @returns Width/height pair in pixels.
 */
export function resolveNetworkCanvasSizePx(
  networkCanvasHost: HTMLElement,
  hostInsetPx: number,
): { widthPx: number; heightPx: number } {
  return {
    widthPx: Math.max(
      1,
      Math.floor(networkCanvasHost.clientWidth - hostInsetPx),
    ),
    heightPx: Math.max(
      1,
      Math.floor(networkCanvasHost.clientHeight - hostInsetPx),
    ),
  };
}
