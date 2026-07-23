import { beforeEach, describe, expect, it } from '@jest/globals';
import {
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
} from '../constants';

const loadModule = (path: string): Promise<unknown> => import(path);

interface MockWorkerGlobal {
  postMessage: jest.Mock;
  requestAnimationFrame: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

type WorkerPost = {
  type?: string;
  frame?: Record<string, unknown>;
  [key: string]: unknown;
};

function installMockWorkerGlobal(): MockWorkerGlobal {
  const self: MockWorkerGlobal = {
    postMessage: jest.fn(),
    requestAnimationFrame: jest.fn(() => 0),
    onmessage: null,
  };
  (globalThis as unknown as Record<string, unknown>).self = self;
  return self;
}

const workerSelf = installMockWorkerGlobal();

function sendInitMessage(tier: 'worker' | 'cpu' | 'gpu') {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'init',
        tier,
        version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
        mapSeed: 42,
      },
    } as unknown as MessageEvent);
  }
}

function sendSimStateMessage(cameraYaw = 0.25) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'simState',
        state: {
          canvasWidth: 640,
          canvasHeight: 360,
          simTick: 1,
          cameraX: 12.5,
          cameraY: 12.5,
          cameraYaw,
          mapSeed: 42,
        },
      },
    } as unknown as MessageEvent);
  }
}

function sendInputMessage(yawDelta: number) {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input: { yawDelta },
      },
    } as unknown as MessageEvent);
  }
}

function findPostByType<T extends WorkerPost>(
  postMessage: jest.Mock,
  type: string,
): T | undefined {
  const calls = postMessage.mock.calls as unknown[][];
  const match = calls.find((call) => {
    const first = call[0];
    return (
      first !== null &&
      typeof first === 'object' &&
      (first as WorkerPost).type === type
    );
  });
  return match ? (match[0] as T) : undefined;
}

describe('Neatenstein display worker', () => {
  beforeEach(() => {
    workerSelf.postMessage.mockClear();
    workerSelf.requestAnimationFrame.mockClear();
  });

  it('acknowledges an init message with the received tier and version', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    const initializedCall = findPostByType<{
      type: string;
      tier: string;
      version: string;
    }>(workerSelf.postMessage, 'initialized');
    expect(initializedCall).toEqual({
      type: 'initialized',
      tier: 'cpu',
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
    });
  });

  it('posts a frame-shaped payload after receiving a simState message', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');
    workerSelf.postMessage.mockClear();
    sendSimStateMessage();
    const frameCall = findPostByType<{
      frame: { format: string; version: string; requestId: number };
    }>(workerSelf.postMessage, 'frame');
    expect({
      hasFrameMessage: frameCall !== undefined,
      format: frameCall?.frame?.format,
      version: frameCall?.frame?.version,
      requestIdType: typeof frameCall?.frame?.requestId,
    }).toEqual({
      hasFrameMessage: true,
      format: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      version: NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
      requestIdType: 'number',
    });
  });

  it('does not schedule requestAnimationFrame for the worker tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('worker');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });

  it('does not schedule requestAnimationFrame for the cpu tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('cpu');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });

  it('does not schedule requestAnimationFrame for the gpu tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('gpu');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });

  it('changes the rendered frame after receiving an input message', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    sendInitMessage('cpu');

    sendSimStateMessage(0);
    const firstMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    workerSelf.postMessage.mockClear();
    sendInputMessage(Math.PI / 2);
    sendSimStateMessage(0);
    const secondMessage = findPostByType<{
      frame: { wallDistances: number[] };
    }>(workerSelf.postMessage, 'frame');

    const firstWallDistances = firstMessage?.frame?.wallDistances as
      number[] | undefined;
    const secondWallDistances = secondMessage?.frame?.wallDistances as
      number[] | undefined;
    expect(
      secondWallDistances?.some(
        (distance, index) => distance !== firstWallDistances?.[index],
      ),
    ).toBe(true);
  });
});
