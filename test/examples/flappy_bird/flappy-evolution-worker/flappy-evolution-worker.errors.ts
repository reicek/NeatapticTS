import type { WorkerErrorMessage } from './flappy-evolution-worker.types';

/** Worker error emitted when runtime initialization fails unexpectedly. */
export const FLAPPY_WORKER_INIT_FAILED_ERROR_MESSAGE =
  'Failed to initialize Flappy evolution worker runtime.';

/** Worker error emitted when playback start is requested before evolution output exists. */
export const FLAPPY_WORKER_PLAYBACK_START_REQUIRES_GENERATION_ERROR_MESSAGE =
  'Cannot start playback before a generation is available. Request a generation first.';

/** Worker error emitted when playback stepping is requested before playback start. */
export const FLAPPY_WORKER_PLAYBACK_STEP_REQUIRES_START_ERROR_MESSAGE =
  'Playback step requested before playback was initialized. Send start-playback first.';

/**
 * Resolves unknown error-like values into display-safe worker error messages.
 *
 * @param error - Unknown error value thrown by worker logic.
 * @returns Normalized error message string.
 */
export function resolveWorkerUnknownErrorMessage(error: unknown): string {
  return String((error as Error)?.message ?? error);
}

/**
 * Creates a typed worker error response payload from a message string.
 *
 * @param message - Error message text.
 * @returns Worker error response message.
 */
export function createWorkerErrorMessage(message: string): WorkerErrorMessage {
  return {
    type: 'error',
    payload: { message },
  };
}

/**
 * Creates a typed worker error response payload from an unknown thrown value.
 *
 * @param error - Unknown thrown value.
 * @returns Worker error response message.
 */
export function createWorkerErrorMessageFromUnknown(
  error: unknown,
): WorkerErrorMessage {
  return createWorkerErrorMessage(resolveWorkerUnknownErrorMessage(error));
}
