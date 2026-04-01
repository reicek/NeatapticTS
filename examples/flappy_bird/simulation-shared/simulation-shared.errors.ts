/**
 * Prefix used when formatting unexpected shared-simulation errors.
 *
 * A stable prefix makes logs easier to scan when multiple Flappy subsystems are
 * emitting diagnostics.
 */
export const FLAPPY_SHARED_SIMULATION_ERROR_PREFIX = 'Shared simulation error:';

/**
 * Formats unknown shared-simulation errors for stable logs.
 *
 * Shared utilities are used from several runtime contexts, so this helper keeps
 * the error surface human-readable even when the thrown value is not an
 * `Error` instance.
 *
 * @param error - Unknown error value.
 * @returns Readable error message.
 */
export function formatSharedSimulationErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return `${FLAPPY_SHARED_SIMULATION_ERROR_PREFIX} ${error.message}`;
  }
  return `${FLAPPY_SHARED_SIMULATION_ERROR_PREFIX} ${String(error)}`;
}
