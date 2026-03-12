import type {
  WorkerChannelGenerationPayload,
  WorkerChannelMessage,
} from './worker-channel.types';
import { requestWorkerResponse } from './worker-channel.request.service';

/**
 * Generation request helper for the browser-entry worker channel.
 *
 * This module asks the worker to evolve until the next generation boundary and
 * then returns the summary payload the browser needs for HUD updates and best
 * network visualization.
 */

/**
 * Requests the next evolved generation payload from the worker channel.
 *
 * Unlike playback streaming, generation evolution is a simple single-response
 * exchange: ask for the next generation and wait for the next
 * `generation-ready` message.
 *
 * @param evolutionWorker - Worker emitting generation-ready messages.
 * @returns Next generation payload.
 */
export function requestWorkerGeneration(
  evolutionWorker: Worker,
): Promise<WorkerChannelGenerationPayload> {
  return requestWorkerResponse({
    evolutionWorker,
    requestMessage: { type: 'request-generation' },
    resolveResponsePayload: (
      workerMessage: WorkerChannelMessage,
    ): WorkerChannelGenerationPayload | undefined => {
      if (workerMessage.type === 'generation-ready') {
        return workerMessage.payload;
      }
      return undefined;
    },
  });
}
