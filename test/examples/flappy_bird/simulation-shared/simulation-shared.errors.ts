/** Prefix used when formatting unexpected shared-simulation errors. */
export const FLAPPY_SHARED_SIMULATION_ERROR_PREFIX = 'Shared simulation error:';

/**
 * Formats unknown shared-simulation errors for stable logs.
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
