/**
 * Error raised when the evolution worker responds with an explicit protocol error payload.
 *
 * Protocol errors are different from runtime worker crashes: the worker is
 * alive, but it is explicitly telling the browser that the requested operation
 * could not be completed.
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
 * Using a dedicated error class makes it easier for browser code to distinguish
 * "worker rejected my request" from "the worker crashed".
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
 * Browser worker errors are not always surfaced as proper `Error` objects, so
 * this helper converts the event payload into a predictable error shape before
 * it escapes the channel layer.
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
