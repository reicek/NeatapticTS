import type { WorkerErrorMessage } from './flappy-evolution-worker.types';

/**
 * Worker error emitted when runtime initialization fails unexpectedly.
 *
 * This message is intentionally stable so the browser host can show a readable
 * error without leaking internal exception shapes into the UI contract.
 */
export const FLAPPY_WORKER_INIT_FAILED_ERROR_MESSAGE =
  'Failed to initialize Flappy evolution worker runtime.';

/**
 * Worker error emitted when playback start is requested before evolution output exists.
 *
 * Playback is defined over an already-evolved population snapshot. The host must
 * request at least one generation before asking the worker to start playback.
 */
export const FLAPPY_WORKER_PLAYBACK_START_REQUIRES_GENERATION_ERROR_MESSAGE =
  'Cannot start playback before a generation is available. Request a generation first.';

/**
 * Worker error emitted when playback stepping is requested before playback start.
 *
 * The protocol is stateful: `start-playback` materializes the mutable playback
 * state that later `request-playback-step` messages advance.
 */
export const FLAPPY_WORKER_PLAYBACK_STEP_REQUIRES_START_ERROR_MESSAGE =
  'Playback step requested before playback was initialized. Send start-playback first.';

/**
 * Resolves unknown error-like values into display-safe worker error messages.
 *
 * Educational note:
 * Browser workers can throw anything, including strings or arbitrary objects.
 * Normalizing that value here gives the rest of the protocol a simple
 * `string`-only error surface.
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
 * @example
 * ```ts
 * postWorkerMessage(
 *   createWorkerErrorMessage('Playback step requested before playback start.'),
 * );
 * ```
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
 * This helper keeps the protocol boundary narrow: worker internals can use
 * regular exceptions, while the browser host still receives one predictable
 * `WorkerErrorMessage` shape.
 *
 * @param error - Unknown thrown value.
 * @returns Worker error response message.
 */
export function createWorkerErrorMessageFromUnknown(
  error: unknown,
): WorkerErrorMessage {
  return createWorkerErrorMessage(resolveWorkerUnknownErrorMessage(error));
}
