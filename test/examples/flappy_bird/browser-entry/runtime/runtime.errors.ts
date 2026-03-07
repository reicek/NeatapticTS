import type { RuntimeContainerTarget } from './runtime.types';

/**
 * Error raised when the browser runtime host container cannot be resolved.
 */
export class RuntimeContainerNotFoundError extends Error {
  /**
   * @param container - Requested container selector or element reference.
   */
  public constructor(container: RuntimeContainerTarget) {
    super(`Flappy demo container not found: ${String(container)}`);
    this.name = 'RuntimeContainerNotFoundError';
  }
}

/**
 * Resolves and validates the browser runtime host element.
 *
 * @param container - Element id or HTMLElement provided to runtime start.
 * @returns Resolved host element.
 */
export function resolveRequiredRuntimeHostElement(
  container: RuntimeContainerTarget,
): HTMLElement {
  const hostElement =
    typeof container === 'string'
      ? document.getElementById(container)
      : container;
  if (!hostElement) {
    throw new RuntimeContainerNotFoundError(container);
  }
  return hostElement;
}

/**
 * Formats unknown runtime failures into a stable HUD status string.
 *
 * @param error - Unknown runtime exception value.
 * @returns Normalized status string for HUD output.
 */
export function resolveRuntimeHudErrorStatus(error: unknown): string {
  return `error: ${String((error as Error)?.message ?? error)}`;
}
