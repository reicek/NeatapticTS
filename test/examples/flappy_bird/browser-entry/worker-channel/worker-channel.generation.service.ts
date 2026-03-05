import type {
  WorkerChannelGenerationPayload,
  WorkerChannelMessage,
} from './worker-channel.types';
import { requestWorkerResponse } from './worker-channel.request.service';

/**
 * Requests the next evolved generation payload from the worker channel.
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
