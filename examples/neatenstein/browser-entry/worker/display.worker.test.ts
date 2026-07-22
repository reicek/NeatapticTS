import { beforeEach, describe, expect, it } from '@jest/globals';
import { NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION } from '../constants';

const loadModule = (path: string): Promise<any> => import(path);

interface MockWorkerGlobal {
  postMessage: jest.Mock;
  requestAnimationFrame: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

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
      },
    } as unknown as MessageEvent);
  }
}

function sendSimStateMessage() {
  if (typeof workerSelf.onmessage === 'function') {
    workerSelf.onmessage({
      data: {
        type: 'simState',
        state: { canvasWidth: 640, canvasHeight: 360, simTick: 1 },
      },
    } as unknown as MessageEvent);
  }
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
    const initializedCall = workerSelf.postMessage.mock.calls.find(
      (call: any[]) => call[0]?.type === 'initialized',
    )?.[0];
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
    const frameCall = workerSelf.postMessage.mock.calls.find(
      (call: any[]) => call[0]?.type === 'frame',
    )?.[0];
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

  it('schedules a requestAnimationFrame tick for the cpu tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('cpu');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBeGreaterThan(
      before,
    );
  });

  it('does not schedule requestAnimationFrame for the worker tier', async () => {
    jest.resetModules();
    await loadModule('./display.worker.ts');
    const before = workerSelf.requestAnimationFrame.mock.calls.length;
    sendInitMessage('worker');
    expect(workerSelf.requestAnimationFrame.mock.calls.length).toBe(before);
  });
});
