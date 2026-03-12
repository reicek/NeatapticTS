import type { RuntimeContainerTarget } from './runtime.types';

/**
 * Runtime-specific error helpers for the browser entrypoint.
 *
 * These errors normalize two user-facing failure modes: the browser cannot find
 * the requested host container, or the runtime needs to report an unexpected
 * failure back into the HUD.
 */

/**
 * Error raised when the browser runtime host container cannot be resolved.
 *
 * This usually means the caller passed the wrong element id or attempted to
 * start the demo before the target container existed in the DOM.
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
 * The runtime accepts either a string id or a concrete element so this helper
 * folds that loose input into one validated host node.
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
 * The HUD should not need to understand arbitrary thrown values, so this helper
 * normalizes anything throwable into one readable status line.
 *
 * @param error - Unknown runtime exception value.
 * @returns Normalized status string for HUD output.
 */
export function resolveRuntimeHudErrorStatus(error: unknown): string {
  return `error: ${String((error as Error)?.message ?? error)}`;
}
