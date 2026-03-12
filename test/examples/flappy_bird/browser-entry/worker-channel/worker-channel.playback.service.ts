import type {
  WorkerChannelMessage,
  WorkerChannelPlaybackStepPayload,
  WorkerChannelPlaybackStepRequest,
} from './worker-channel.types';
import {
  createWorkerChannelResponseError,
  resolveWorkerChannelRuntimeError,
} from './worker-channel.errors';

/**
 * Stateful playback request channel for one evolution worker.
 *
 * Playback is intentionally handled differently from generation requests. The
 * browser asks for a sequence of incremental frames, and the channel keeps a
 * small amount of per-worker state so each request can be matched to the
 * correct reply.
 */

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
 * A playback-step request is effectively "advance the simulation by this many
 * internal steps, then send me a packed frame I can render". The request is
 * tagged with a monotonically increasing request id so stale or out-of-order
 * replies can be ignored safely.
 *
 * @param evolutionWorker - Worker that owns playback simulation state.
 * @param playbackStepRequest - Requested simulation budget and viewport size.
 * @returns Playback-step payload including snapshot and completion marker.
 * @example
 * ```ts
 * const playbackPayload = await requestWorkerPlaybackStep(evolutionWorker, {
 *   simulationSteps: 2,
 *   visibleWorldWidthPx: 640,
 *   visibleWorldHeightPx: 480,
 * });
 * ```
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
 * The state tracks request ids and the one allowed in-flight playback request.
 * That single-flight rule keeps the protocol simple and avoids ambiguous frame
 * ordering on the browser side.
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

  evolutionWorker.addEventListener('message', ((
    event: MessageEvent<WorkerChannelMessage>,
  ): void => {
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
  }) as EventListener);

  evolutionWorker.addEventListener('error', ((event: ErrorEvent): void => {
    const pendingPlaybackRequest =
      playbackWorkerChannelState.pendingPlaybackRequest;
    if (!pendingPlaybackRequest) {
      return;
    }

    playbackWorkerChannelState.pendingPlaybackRequest = null;
    pendingPlaybackRequest.reject(
      resolveWorkerChannelRuntimeError(event.error, event.message),
    );
  }) as EventListener);

  playbackWorkerChannelStateByWorker.set(
    evolutionWorker,
    playbackWorkerChannelState,
  );
  return playbackWorkerChannelState;
}
