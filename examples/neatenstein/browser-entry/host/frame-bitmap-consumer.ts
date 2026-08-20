/**
 * Host-side frame bitmap consumer for the Neatenstein renderer (C1.5).
 *
 * When the worker presents frames via `transferToImageBitmap`
 * ({@link presentNeatensteinFrameBitmap}), the host thread receives an
 * `ImageBitmap` and must composite it onto the visible canvas. This module
 * provides the host-side `consumeNeatensteinFrameBitmap` entry point that
 * draws the transferred bitmap using `createImageBitmap` and `drawImage`.
 *
 * @module
 */

/**
 * Minimal canvas 2D context shape needed for bitmap compositing.
 */
interface NeatensteinHostRenderContext {
  drawImage(
    image: CanvasImageSource,
    dx: number,
    dy: number,
    dWidth: number,
    dHeight: number,
  ): void;
}

/**
 * Consume a transferred frame `ImageBitmap` on the host thread (C1.5).
 *
 * The bitmap is drawn onto the host canvas context at the specified dimensions.
 * The bitmap is closed after drawing to release GPU memory immediately.
 *
 * @param bitmap - `ImageBitmap` transferred from the worker's
 *   `OffscreenCanvas.transferToImageBitmap()`.
 * @param ctx - Host canvas 2D rendering context.
 * @param width - Destination width in pixels.
 * @param height - Destination height in pixels.
 *
 * @example
 * ```ts
 * worker.onmessage = (e) => {
 *   if (e.data.type === 'frame') {
 *     consumeNeatensteinFrameBitmap(e.data.bitmap, ctx, canvas.width, canvas.height);
 *   }
 * };
 * ```
 */
export function consumeNeatensteinFrameBitmap(
  bitmap: ImageBitmap,
  ctx: NeatensteinHostRenderContext,
  width: number,
  height: number,
): void {
  ctx.drawImage(bitmap, 0, 0, width, height);

  // Release the bitmap immediately after drawing to avoid GPU memory leaks.
  if (typeof bitmap.close === 'function') {
    bitmap.close();
  }
}
