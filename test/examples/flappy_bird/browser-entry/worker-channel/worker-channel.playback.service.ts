import type {
  WorkerChannelMessage,
  WorkerChannelPlaybackStepPayload,
  WorkerChannelPlaybackStepRequest,
} from './worker-channel.types';
import {
  createWorkerChannelResponseError,
  resolveWorkerChannelRuntimeError,
} from './worker-channel.errors';

type PendingPlaybackRequest = {
  requestId: number;
  reject: (error: Error) => void;
  resolve: (payload: WorkerChannelPlaybackStepPayload) => void;
};

type PlaybackWorkerChannelState = {
  nextRequestId: number;
  pendingPlaybackRequest: PendingPlaybackRequest | null;
};

const playbackWorkerChannelStateByWorker = new WeakMap<
  Worker,
  PlaybackWorkerChannelState
>();

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
  const playbackWorkerChannelState =
    resolvePlaybackWorkerChannelState(evolutionWorker);

  return new Promise((resolve, reject) => {
    if (playbackWorkerChannelState.pendingPlaybackRequest) {
      reject(
        new Error(
          'Concurrent playback-step requests are not supported by the playback worker channel.',
        ),
      );
      return;
    }

    const requestId = playbackWorkerChannelState.nextRequestId;
    playbackWorkerChannelState.nextRequestId += 1;

    playbackWorkerChannelState.pendingPlaybackRequest = {
      requestId,
      reject,
      resolve,
    };

    evolutionWorker.postMessage({
      type: 'request-playback-step',
      payload: {
        ...playbackStepRequest,
        requestId,
      },
    });
  });
}

/**
 * Resolves persistent playback worker-channel state for one worker instance.
 *
 * @param evolutionWorker - Worker that owns playback simulation state.
 * @returns Persistent playback worker-channel state for the worker.
 */
function resolvePlaybackWorkerChannelState(
  evolutionWorker: Worker,
): PlaybackWorkerChannelState {
  const cachedState = playbackWorkerChannelStateByWorker.get(evolutionWorker);
  if (cachedState) {
    return cachedState;
  }

  const playbackWorkerChannelState: PlaybackWorkerChannelState = {
    nextRequestId: 1,
    pendingPlaybackRequest: null,
  };

  evolutionWorker.addEventListener(
    'message',
    ((event: MessageEvent<WorkerChannelMessage>): void => {
      const pendingPlaybackRequest =
        playbackWorkerChannelState.pendingPlaybackRequest;
      if (!pendingPlaybackRequest) {
        return;
      }

      if (event.data.type === 'playback-step') {
        if (event.data.payload.requestId !== pendingPlaybackRequest.requestId) {
          return;
        }

        playbackWorkerChannelState.pendingPlaybackRequest = null;
        pendingPlaybackRequest.resolve(event.data.payload);
        return;
      }

      if (event.data.type === 'error') {
        playbackWorkerChannelState.pendingPlaybackRequest = null;
        pendingPlaybackRequest.reject(
          createWorkerChannelResponseError(event.data.payload.message),
        );
      }
    }) as EventListener,
  );

  evolutionWorker.addEventListener(
    'error',
    ((event: ErrorEvent): void => {
      const pendingPlaybackRequest =
        playbackWorkerChannelState.pendingPlaybackRequest;
      if (!pendingPlaybackRequest) {
        return;
      }

      playbackWorkerChannelState.pendingPlaybackRequest = null;
      pendingPlaybackRequest.reject(
        resolveWorkerChannelRuntimeError(event.error, event.message),
      );
    }) as EventListener,
  );

  playbackWorkerChannelStateByWorker.set(
    evolutionWorker,
    playbackWorkerChannelState,
  );
  return playbackWorkerChannelState;
}
