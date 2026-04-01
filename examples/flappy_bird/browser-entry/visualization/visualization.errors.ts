/**
 * Error text for non-finite legend bounds.
 */
export const FLAPPY_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE =
  'Visualization legend bound must be a finite number.';

/**
 * Thrown when a legend bound cannot be safely formatted.
 */
export class VisualizationNonFiniteBoundError extends Error {
  public constructor() {
    super(FLAPPY_VISUALIZATION_NON_FINITE_BOUND_ERROR_MESSAGE);
    this.name = 'VisualizationNonFiniteBoundError';
  }
}

/**
 * Guards legend-bound formatting against non-finite values.
 *
 * @param value - Legend bound candidate.
 * @returns Nothing.
 */
export function assertFiniteLegendBound(value: number): void {
  if (!Number.isFinite(value)) {
    throw new VisualizationNonFiniteBoundError();
  }
}
