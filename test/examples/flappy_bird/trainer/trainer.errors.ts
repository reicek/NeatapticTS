/** Prefix used when rendering unexpected trainer failures to stderr. */
export const FLAPPY_TRAINER_UNEXPECTED_ERROR_PREFIX = 'Flappy trainer failed:';

/**
 * Formats unknown trainer failures into a stable human-readable message.
 *
 * @param error - Unknown rejection reason from trainer execution.
 * @returns Formatted error string for CLI logging.
 */
export function formatTrainerErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return `${FLAPPY_TRAINER_UNEXPECTED_ERROR_PREFIX} ${error.message}`;
  }
  return `${FLAPPY_TRAINER_UNEXPECTED_ERROR_PREFIX} ${String(error)}`;
}
