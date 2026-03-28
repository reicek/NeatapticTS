/**
 * Small CLI-facing error-rendering boundary for the trainer.
 *
 * Keeping fatal-error formatting in one file prevents setup, evaluation, and
 * shutdown paths from inventing slightly different terminal messages. That is a
 * small detail, but it makes long-running scripts and quick debugging sessions
 * easier to scan.
 */
/** Prefix used when rendering unexpected trainer failures to stderr. */
export const FLAPPY_TRAINER_UNEXPECTED_ERROR_PREFIX = 'Flappy trainer failed:';

/**
 * Formats unknown trainer failures into a stable human-readable message.
 *
 * Errors can arrive here as real `Error` objects or as arbitrary rejected
 * values. Normalizing both cases into one predictable string keeps the CLI
 * surface boring in the good way.
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
