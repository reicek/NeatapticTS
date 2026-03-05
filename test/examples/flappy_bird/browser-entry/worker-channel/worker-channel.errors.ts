/**
 * Error raised when the evolution worker responds with an explicit protocol error payload.
 */
export class WorkerChannelResponseError extends Error {
  /**
   * @param message - Worker-supplied protocol error message.
   */
  public constructor(message: string) {
    super(message);
    this.name = 'WorkerChannelResponseError';
  }
}

/**
 * Converts worker protocol error payloads into typed worker-channel errors.
 *
 * @param message - Message supplied by the worker error payload.
 * @returns Typed worker-channel protocol error.
 */
export function createWorkerChannelResponseError(message: string): Error {
  return new WorkerChannelResponseError(message);
}

/**
 * Resolves a worker `ErrorEvent` into a normalized `Error` instance.
 *
 * @param errorLike - Optional `event.error` payload.
 * @param fallbackMessage - Fallback message from `event.message`.
 * @returns Normalized runtime error.
 */
export function resolveWorkerChannelRuntimeError(
  errorLike: unknown,
  fallbackMessage: string,
): Error {
  if (errorLike instanceof Error) {
    return errorLike;
  }
  return new Error(fallbackMessage);
}
