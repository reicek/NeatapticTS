import type {
  WorkerChannelMessage,
  WorkerChannelPlaybackStepPayload,
  WorkerChannelPlaybackStepRequest,
} from './worker-channel.types';
import { requestWorkerResponse } from './worker-channel.request.service';

/**
 * Requests one playback batch step from the worker channel.
 *
 * @param evolutionWorker - Worker that owns playback simulation state.
 * @param playbackStepRequest - Requested simulation budget and viewport size.
 * @returns Playback-step payload including snapshot and completion marker.
 */
export function requestWorkerPlaybackStep(
  evolutionWorker: Worker,
  playbackStepRequest: WorkerChannelPlaybackStepRequest,
): Promise<WorkerChannelPlaybackStepPayload> {
  return requestWorkerResponse({
    evolutionWorker,
    requestMessage: {
      type: 'request-playback-step',
      payload: playbackStepRequest,
    },
    resolveResponsePayload: (
      workerMessage: WorkerChannelMessage,
    ): WorkerChannelPlaybackStepPayload | undefined => {
      if (workerMessage.type === 'playback-step') {
        return workerMessage.payload;
      }
      return undefined;
    },
  });
}
