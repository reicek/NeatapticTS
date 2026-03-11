import { requestWorkerPlaybackStep } from './worker-channel.playback.service';
import type { WorkerChannelPlaybackStepPayload } from './worker-channel.types';

describe('requestWorkerPlaybackStep', () => {
  it('registers playback listeners only once per worker across sequential requests', async () => {
    const evolutionWorker = createMockWorker();
    const firstRequestPromise = requestWorkerPlaybackStep(evolutionWorker, {
      simulationSteps: 1,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
    });
    evolutionWorker.emitMessage(createPlaybackStepMessage(1, 1));
    await firstRequestPromise;

    const secondRequestPromise = requestWorkerPlaybackStep(evolutionWorker, {
      simulationSteps: 2,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
    });
    evolutionWorker.emitMessage(createPlaybackStepMessage(2, 2));
    await secondRequestPromise;

    expect(evolutionWorker.addEventListener).toHaveBeenCalledTimes(2);
  });

  it('rejects concurrent playback-step requests on the same worker', async () => {
    const evolutionWorker = createMockWorker();
    const firstRequestPromise = requestWorkerPlaybackStep(evolutionWorker, {
      simulationSteps: 1,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
    });

    const concurrentRequestResult = requestWorkerPlaybackStep(evolutionWorker, {
      simulationSteps: 1,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
    }).catch((error: Error) => error.message);

    evolutionWorker.emitMessage(createPlaybackStepMessage(1, 1));

    expect(
      await Promise.all([firstRequestPromise, concurrentRequestResult]),
    ).toEqual([
      createPlaybackStepPayload(1, 1),
      'Concurrent playback-step requests are not supported by the playback worker channel.',
    ]);
  });

  it('ignores stale playback-step responses with older request ids', async () => {
    const evolutionWorker = createMockWorker();
    const playbackRequestPromise = requestWorkerPlaybackStep(evolutionWorker, {
      simulationSteps: 1,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
    });

    evolutionWorker.emitMessage(createPlaybackStepMessage(99, 0));
    evolutionWorker.emitMessage(createPlaybackStepMessage(1, 1));

    expect(await playbackRequestPromise).toEqual(
      createPlaybackStepPayload(1, 1),
    );
  });
});

function createMockWorker(): Worker & {
  emitMessage: (message: unknown) => void;
} {
  const messageListeners: Array<(event: MessageEvent<unknown>) => void> = [];
  const errorListeners: Array<(event: ErrorEvent) => void> = [];

  return {
    addEventListener: jest.fn(
      (
        eventType: string,
        listener: EventListenerOrEventListenerObject,
      ): void => {
        if (typeof listener !== 'function') {
          return;
        }

        if (eventType === 'message') {
          messageListeners.push(
            listener as (event: MessageEvent<unknown>) => void,
          );
          return;
        }

        if (eventType === 'error') {
          errorListeners.push(listener as (event: ErrorEvent) => void);
        }
      },
    ),
    postMessage: jest.fn(),
    emitMessage: (message: unknown): void => {
      for (const listener of messageListeners) {
        listener({ data: message } as MessageEvent<unknown>);
      }

      for (const listener of errorListeners) {
        void listener;
      }
    },
  } as unknown as Worker & { emitMessage: (message: unknown) => void };
}

function createPlaybackStepMessage(frameIndex: number, requestId: number) {
  return {
    type: 'playback-step' as const,
    payload: createPlaybackStepPayload(frameIndex, requestId),
  };
}

function createPlaybackStepPayload(
  frameIndex: number,
  requestId: number,
): WorkerChannelPlaybackStepPayload {
  return {
    requestId,
    snapshot: {
      format: 'packed-v1',
      frameIndex,
      visibleWorldWidthPx: 640,
      visibleWorldHeightPx: 480,
      pipeCount: 0,
      birdCount: 0,
      pipes: {
        xPositionsPx: new Float32Array([]),
        gapCenterYPositionsPx: new Float32Array([]),
        gapSizesPx: new Float32Array([]),
      },
      birds: {
        yPositionsPx: new Float32Array([]),
        pipesPassed: new Uint32Array([]),
        framesSurvived: new Uint32Array([]),
        doneFlags: new Uint8Array([]),
      },
    },
    done: false,
  };
}
