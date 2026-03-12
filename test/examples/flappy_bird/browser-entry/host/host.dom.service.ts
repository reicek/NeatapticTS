/**
 * Low-level DOM safety helpers for the browser host boundary.
 *
 * Host assembly depends on several canvases. This helper turns the browser's
 * nullable `getContext` API into a strict contract before higher-level host
 * assembly begins.
 */

/**
 * Resolves a required 2D context from a canvas element.
 *
 * Failing early here keeps later rendering code free from repeated null checks.
 *
 * @param canvas - Target canvas element.
 * @param errorMessage - Error message when 2D context is unavailable.
 * @returns Canvas 2D rendering context.
 */
export function resolveRequiredCanvas2dContext(
  canvas: HTMLCanvasElement,
  errorMessage: string,
): CanvasRenderingContext2D {
  const context = canvas.getContext('2d');
  if (!context) {
    throw new Error(errorMessage);
  }

  return context;
}
