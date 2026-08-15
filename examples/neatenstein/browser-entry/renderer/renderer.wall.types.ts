/**
 * Type definitions for the wall renderer extracted from {@link module:./walls}.
 *
 * @module
 */

/**
 * Minimal canvas-like context consumed by the CPU wall renderer.
 *
 * Only `putImageData` is required. The interface is intentionally narrow so
 * the renderer can be unit-tested with a lightweight mock and still accept a
 * real `CanvasRenderingContext2D` at runtime through structural typing.
 */
export interface NeatensteinWallRenderContext {
  /**
   * Flush an ImageData-like payload to the canvas.
   *
   * @param imageData - Object with `data`, `width`, and `height`.
   * @param dx - Destination X coordinate.
   * @param dy - Destination Y coordinate.
   */
  putImageData(
    imageData: { data: Uint8ClampedArray; width: number; height: number },
    dx: number,
    dy: number,
  ): void;
}

/**
 * Parsed RGB triplet from a `#rrggbb` hex color string.
 */
export interface ParsedRgb {
  /** Red channel in `[0, 255]`. */
  r: number;
  /** Green channel in `[0, 255]`. */
  g: number;
  /** Blue channel in `[0, 255]`. */
  b: number;
}
