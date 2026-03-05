/**
 * Resolves a required 2D context from a canvas element.
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
