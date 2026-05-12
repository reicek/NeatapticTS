import type {
  WorkerChannelGenerationPayload,
  WorkerChannelMessage,
} from './worker-channel.types';
import { requestWorkerResponse } from './worker-channel.request.service';

/**
 * Generation request helper for the browser-entry worker channel.
 *
 * This module asks the worker for the next playable population boundary. The
 * first response may be the warmed generation-zero population so playback can
 * begin promptly; subsequent responses are normally evolved generations.
 */

/**
 * Requests the next playable generation payload from the worker channel.
 *
 * Unlike playback streaming, generation readiness is a simple single-response
 * exchange: ask for the next playable population and wait for the next
 * `generation-ready` message. Startup may publish generation zero before the
 * first full recurrent `evolve()` batch.
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
