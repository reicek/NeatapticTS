import { describe, expect, it, jest } from '@jest/globals';
import {
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
} from '../constants';

const loadModule = <T>(path: string): Promise<T> => import(path) as Promise<T>;

interface MockWorker {
  postMessage: jest.Mock;
  terminate: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

function installMockWorker(): { instances: MockWorker[] } {
  const instances: MockWorker[] = [];
  const Worker = jest.fn(() => {
    const worker: MockWorker = {
      postMessage: jest.fn(),
      terminate: jest.fn(),
      onmessage: null,
    };
    instances.push(worker);
    return worker;
  });
  (globalThis as unknown as Record<string, unknown>).Worker = Worker;
  return { instances };
}

function createMockCanvas(): {
  canvas: HTMLCanvasElement;
  offscreen: OffscreenCanvas;
  transferControlToOffscreen: jest.Mock;
} {
  const offscreen = {} as OffscreenCanvas;
  const transferControlToOffscreen = jest.fn(() => offscreen);
  const canvas = {
    transferControlToOffscreen,
    width: 640,
    height: 360,
  } as unknown as HTMLCanvasElement;
  return { canvas, offscreen, transferControlToOffscreen };
}

describe('Neatenstein host renderer bridge', () => {
  describe('createNeatensteinRendererBridge', () => {
    it('returns an object with postSimState, destroy, and a worker property', async () => {
      const { instances } = installMockWorker();
      const { canvas } = createMockCanvas();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      const bridge = createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'cpu',
        mapSeed: 42,
      });
      expect({
        hasPostSimState: typeof bridge.postSimState === 'function',
        hasDestroy: typeof bridge.destroy === 'function',
        ownsWorker: instances.includes(bridge.worker as unknown as MockWorker),
      }).toEqual({
        hasPostSimState: true,
        hasDestroy: true,
        ownsWorker: true,
      });
    });

    it('worker tier transfers OffscreenCanvas to the worker on init', async () => {
      const { instances } = installMockWorker();
      const { canvas, offscreen, transferControlToOffscreen } =
        createMockCanvas();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'worker',
        mapSeed: 42,
      });
      const worker = instances[0];
      const initCall = worker.postMessage.mock.calls[0];
      expect({
        transferControlCalled:
          transferControlToOffscreen.mock.calls.length === 1,
        offscreenInTransferList:
          initCall &&
          Array.isArray(initCall[1]) &&
          initCall[1].includes(offscreen),
      }).toEqual({
        transferControlCalled: true,
        offscreenInTransferList: true,
      });
    });

    it('cpu tier registers onmessage and exposes the latest frame requestId', async () => {
      const { instances } = installMockWorker();
      const { canvas } = createMockCanvas();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      const bridge = createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'cpu',
        mapSeed: 42,
      });
      const worker = instances[0];
      const onmessage = worker.onmessage;
      if (typeof onmessage === 'function') {
        onmessage({
          data: { type: 'frame', frame: { requestId: 42 } },
        } as unknown as MessageEvent);
      }
      expect({
        onmessageRegistered: typeof onmessage === 'function',
        latestRequestId: bridge.requestId,
      }).toEqual({
        onmessageRegistered: true,
        latestRequestId: 42,
      });
    });

    it('gpu tier registers onmessage and exposes the latest frame requestId', async () => {
      const { instances } = installMockWorker();
      const { canvas } = createMockCanvas();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      const bridge = createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'gpu',
        mapSeed: 42,
      });
      const worker = instances[0];
      const onmessage = worker.onmessage;
      if (typeof onmessage === 'function') {
        onmessage({
          data: { type: 'frame', frame: { requestId: 99 } },
        } as unknown as MessageEvent);
      }
      expect({
        onmessageRegistered: typeof onmessage === 'function',
        latestRequestId: bridge.requestId,
      }).toEqual({
        onmessageRegistered: true,
        latestRequestId: 99,
      });
    });

    it('init message includes the versioned frame format and tier name', async () => {
      const { instances } = installMockWorker();
      const { canvas } = createMockCanvas();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'gpu',
        mapSeed: 42,
      });
      const initMessage = instances[0].postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect({
        hasVersion:
          initMessage?.version === NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
        hasTier: initMessage?.tier === 'gpu',
        hasMapSeed: initMessage?.mapSeed === 42,
      }).toEqual({
        hasVersion: true,
        hasTier: true,
        hasMapSeed: true,
      });
    });

    it('destroy terminates the worker', async () => {
      const { instances } = installMockWorker();
      const { canvas } = createMockCanvas();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      const bridge = createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'cpu',
        mapSeed: 42,
      });
      bridge.destroy();
      expect(instances[0].terminate).toHaveBeenCalled();
    });
  });
});
