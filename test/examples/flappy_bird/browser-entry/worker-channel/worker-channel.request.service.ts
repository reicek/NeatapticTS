import {
  createWorkerChannelResponseError,
  resolveWorkerChannelRuntimeError,
} from './worker-channel.errors';
import type { WorkerChannelMessage } from './worker-channel.types';

/**
 * Generic request/response helper for worker-channel interactions.
 *
 * This module implements a small RPC-like pattern on top of browser worker
 * events. The browser sends one message, listens for the first matching reply,
 * and normalizes protocol failures into ordinary `Error` instances.
 */

/**
 * Message shape sent to the worker request channel.
 *
 * The worker protocol stays stringly-typed at the transport edge on purpose so
 * the message stream is easy to inspect during debugging.
 */
export interface WorkerChannelRequestMessage {
  type: string;
  payload?: unknown;
}

/**
 * Configuration used for one worker request/response lifecycle.
 *
 * Callers provide the worker, the outbound message, and the predicate that says
 * which inbound worker message should satisfy the request.
 */
export interface WorkerChannelRequestOptions<ResponsePayload> {
  evolutionWorker: Worker;
  requestMessage: WorkerChannelRequestMessage;
  resolveResponsePayload: (
    workerMessage: WorkerChannelMessage,
  ) => ResponsePayload | undefined;
}

/**
 * Sends one request to the worker and resolves with the first matching response payload.
 *
 * This keeps generation requests simple: the caller describes the response it is
 * waiting for, and this helper handles transient listeners, protocol errors,
 * and runtime worker failures.
 *
 * @param options - Worker request options and response resolver callback.
 * @returns Promise resolving with the matched worker response payload.
 * @example
 * ```ts
 * const generation = await requestWorkerResponse({
 *   evolutionWorker,
 *   requestMessage: { type: 'request-generation' },
 *   resolveResponsePayload: (message) =>
 *     message.type === 'generation-ready' ? message.payload : undefined,
 * });
 * ```
 */
export function requestWorkerResponse<ResponsePayload>(
  options: WorkerChannelRequestOptions<ResponsePayload>,
): Promise<ResponsePayload> {
  const { evolutionWorker, requestMessage, resolveResponsePayload } = options;

  return new Promise((resolve, reject) => {
    // Step 1: Route worker messages into success, protocol-error, or ignore branches.
    const handleMessage = (event: MessageEvent<WorkerChannelMessage>): void => {
      const workerMessage = event.data;
      const resolvedPayload = resolveResponsePayload(workerMessage);
      if (resolvedPayload !== undefined) {
        cleanup();
        resolve(resolvedPayload);
        return;
      }

      if (workerMessage.type === 'error') {
        cleanup();
        reject(createWorkerChannelResponseError(workerMessage.payload.message));
      }
    };

    // Step 2: Convert runtime worker errors into normalized Error instances.
    const handleError = (event: ErrorEvent): void => {
      cleanup();
      reject(resolveWorkerChannelRuntimeError(event.error, event.message));
    };

    // Step 3: Remove transient listeners once request settles.
    const cleanup = (): void => {
      evolutionWorker.removeEventListener(
        'message',
        handleMessage as EventListener,
      );
      evolutionWorker.removeEventListener(
        'error',
        handleError as EventListener,
      );
    };

    // Step 4: Register listeners and send request.
    evolutionWorker.addEventListener('message', handleMessage as EventListener);
    evolutionWorker.addEventListener('error', handleError as EventListener);
    evolutionWorker.postMessage(requestMessage);
  });
}
