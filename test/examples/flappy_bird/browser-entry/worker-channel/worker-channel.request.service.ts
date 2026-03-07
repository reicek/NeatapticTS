import {
  createWorkerChannelResponseError,
  resolveWorkerChannelRuntimeError,
} from './worker-channel.errors';
import type { WorkerChannelMessage } from './worker-channel.types';

/**
 * Message shape sent to the worker request channel.
 */
export interface WorkerChannelRequestMessage {
  type: string;
  payload?: unknown;
}

/**
 * Configuration used for one worker request/response lifecycle.
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
 * @param options - Worker request options and response resolver callback.
 * @returns Promise resolving with the matched worker response payload.
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
