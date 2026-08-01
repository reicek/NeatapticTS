import { describe, expect, it, jest } from '@jest/globals';
import {
  NEATENSTEIN_INPUT_MESSAGE_TYPE,
  NEATENSTEIN_RENDER_FRAME_FORMAT_VERSION,
  NEATENSTEIN_WORKER_BUNDLE_FILENAME,
} from '../constants';

const loadModule = <T>(path: string): Promise<T> => import(path) as Promise<T>;

interface MockWorker {
  postMessage: jest.Mock;
  terminate: jest.Mock;
  onmessage: ((event: MessageEvent) => void) | null;
}

function installMockWorker(): { Worker: jest.Mock; instances: MockWorker[] } {
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
  return { Worker, instances };
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

    it('uses the default worker URL when none is provided', async () => {
      const { Worker, instances } = installMockWorker();
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      createNeatensteinRendererBridge({
        canvas: createMockCanvas().canvas,
        tier: 'cpu',
        mapSeed: 1,
      });
      expect(Worker).toHaveBeenCalledWith(
        `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        { type: 'module' },
      );
      const initMessage = instances[0].postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect(initMessage?.type).toBe('init');
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

    it('exposes a setFrameConsumer method for the cpu tier', async () => {
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
      expect(
        typeof (bridge as unknown as Record<string, unknown>).setFrameConsumer,
      ).toBe('function');
    });

    it('delivers the frame payload to the registered consumer on cpu tier', async () => {
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
      const consumer = jest.fn();
      (
        bridge as unknown as Record<
          string,
          (callback: (frame: Record<string, unknown>) => void) => void
        >
      ).setFrameConsumer(consumer);
      const worker = instances[0];
      const frame = { requestId: 42, format: 'cpu-frame' };
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame },
        } as unknown as MessageEvent);
      }
      expect(consumer).toHaveBeenCalledWith(frame);
    });

    it('delivers the frame payload to the registered consumer on gpu tier', async () => {
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
      const consumer = jest.fn();
      (
        bridge as unknown as Record<
          string,
          (callback: (frame: Record<string, unknown>) => void) => void
        >
      ).setFrameConsumer(consumer);
      const worker = instances[0];
      const frame = { requestId: 99, format: 'gpu-frame' };
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame },
        } as unknown as MessageEvent);
      }
      expect(consumer).toHaveBeenCalledWith(frame);
    });

    it('throws when worker tier is selected but OffscreenCanvas transfer is unavailable', async () => {
      installMockWorker();
      const canvas = {
        width: 640,
        height: 360,
      } as unknown as HTMLCanvasElement;
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      expect(() =>
        createNeatensteinRendererBridge({
          canvas,
          workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
          tier: 'worker',
          mapSeed: 42,
        }),
      ).toThrow(
        'Neatenstein worker tier requires HTMLCanvasElement.transferControlToOffscreen().',
      );
    });

    it('queues simulation state before the worker initializes', async () => {
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
      const state: import('../renderer/frame').NeatensteinRenderState = {
        canvasWidth: 1280,
        canvasHeight: 720,
        simTick: 7,
        cameraX: 1,
        cameraY: 2,
        cameraYaw: 0.5,
        mapSeed: 42,
      };
      bridge.postSimState(state);
      expect(worker.postMessage).toHaveBeenCalledTimes(1);
      const initCall = worker.postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect(initCall?.type).toBe('init');
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      expect(worker.postMessage).toHaveBeenCalledTimes(2);
      expect(worker.postMessage.mock.calls[1]?.[0]).toEqual({
        type: 'simState',
        state,
      });
    });

    it('queues input snapshots before the worker initializes', async () => {
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
      const input: import('./input').InputSnapshot = {
        timestamp: 1234,
        movement: { forward: true, backward: false, left: false, right: false },
        look: { yawDelta: 0.1, pitchDelta: 0.2 },
        touch: { active: false, yawDelta: 0, pitchDelta: 0 },
        pointerLocked: false,
        fire: false,
        dash: false,
        lightToggle: false,
      };
      bridge.forwardWorkerInput(input);
      expect(worker.postMessage).toHaveBeenCalledTimes(1);
      const initCall = worker.postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect(initCall?.type).toBe('init');
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      expect(worker.postMessage).toHaveBeenCalledTimes(2);
      expect(worker.postMessage.mock.calls[1]?.[0]).toEqual({
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input,
      });
    });

    it('sends state and input immediately when already initialized', async () => {
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
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      const state: import('../renderer/frame').NeatensteinRenderState = {
        canvasWidth: 1280,
        canvasHeight: 720,
        simTick: 3,
        cameraX: 4,
        cameraY: 5,
        cameraYaw: 0,
        mapSeed: 42,
      };
      const input: import('./input').InputSnapshot = {
        timestamp: 5678,
        movement: { forward: false, backward: false, left: true, right: false },
        look: { yawDelta: 0, pitchDelta: 0 },
        touch: { active: false, yawDelta: 0, pitchDelta: 0 },
        pointerLocked: false,
        fire: false,
        dash: false,
        lightToggle: false,
      };
      bridge.postSimState(state);
      expect(worker.postMessage.mock.calls[1]?.[0]).toEqual({
        type: 'simState',
        state,
      });
      bridge.forwardWorkerInput(input);
      expect(worker.postMessage.mock.calls[2]?.[0]).toEqual({
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input,
      });
    });

    it('flushes pending state and input when the worker sends initialized', async () => {
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
      const state: import('../renderer/frame').NeatensteinRenderState = {
        canvasWidth: 1280,
        canvasHeight: 720,
        simTick: 3,
        cameraX: 4,
        cameraY: 5,
        cameraYaw: 0,
        mapSeed: 42,
      };
      const input: import('./input').InputSnapshot = {
        timestamp: 5678,
        movement: { forward: false, backward: false, left: true, right: false },
        look: { yawDelta: 0, pitchDelta: 0 },
        touch: { active: false, yawDelta: 0, pitchDelta: 0 },
        pointerLocked: false,
        fire: false,
        dash: false,
        lightToggle: false,
      };
      bridge.postSimState(state);
      bridge.forwardWorkerInput(input);
      expect(worker.postMessage).toHaveBeenCalledTimes(1);
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      expect(worker.postMessage).toHaveBeenCalledTimes(3);
      expect(worker.postMessage.mock.calls[1]?.[0]).toEqual({
        type: NEATENSTEIN_INPUT_MESSAGE_TYPE,
        input,
      });
      expect(worker.postMessage.mock.calls[2]?.[0]).toEqual({
        type: 'simState',
        state,
      });
    });

    it('is idempotent when destroy is called more than once', async () => {
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
      bridge.destroy();
      expect(instances[0].terminate).toHaveBeenCalledTimes(1);
    });

    it('ignores worker messages after destroy', async () => {
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
      bridge.destroy();
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 77 } },
        } as unknown as MessageEvent);
      }
      expect(bridge.requestId).toBe(0);
    });

    it('drops simulation state and input calls after destroy', async () => {
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
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      bridge.destroy();
      const state: import('../renderer/frame').NeatensteinRenderState = {
        canvasWidth: 1280,
        canvasHeight: 720,
        simTick: 1,
        cameraX: 1,
        cameraY: 1,
        cameraYaw: 0,
        mapSeed: 42,
      };
      const input: import('./input').InputSnapshot = {
        timestamp: 1,
        movement: {
          forward: false,
          backward: false,
          left: false,
          right: false,
        },
        look: { yawDelta: 0, pitchDelta: 0 },
        touch: { active: false, yawDelta: 0, pitchDelta: 0 },
        pointerLocked: false,
        fire: false,
        dash: false,
        lightToggle: false,
      };
      bridge.postSimState(state);
      bridge.forwardWorkerInput(input);
      expect(worker.postMessage).toHaveBeenCalledTimes(1);
    });

    it('falls back to client dimensions when the backing store is invalid', async () => {
      const { instances } = installMockWorker();
      const canvas = {
        width: 0,
        height: 0,
        clientWidth: 640,
        clientHeight: 360,
      } as unknown as HTMLCanvasElement;
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'cpu',
        mapSeed: 42,
      });
      const initCall = instances[0].postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect(initCall?.canvasWidth).toBe(640);
      expect(initCall?.canvasHeight).toBe(360);
    });

    it('forwards the existing canvas backing-store dimensions unchanged', async () => {
      const { instances } = installMockWorker();
      const canvas = {
        width: 1280,
        height: 720,
      } as unknown as HTMLCanvasElement;
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'cpu',
        mapSeed: 42,
      });
      const initCall = instances[0].postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect(initCall?.canvasWidth).toBe(1280);
      expect(initCall?.canvasHeight).toBe(720);
    });

    it('skips canvas sizing when both backing and client dimensions are invalid', async () => {
      const { instances } = installMockWorker();
      const canvas = {
        width: 0,
        height: 0,
        clientWidth: 0,
        clientHeight: 0,
      } as unknown as HTMLCanvasElement;
      const { createNeatensteinRendererBridge } = await loadModule<
        typeof import('./renderer-bridge.ts')
      >('./renderer-bridge.ts');
      createNeatensteinRendererBridge({
        canvas,
        workerUrl: `/assets/${NEATENSTEIN_WORKER_BUNDLE_FILENAME}`,
        tier: 'cpu',
        mapSeed: 42,
      });
      const initCall = instances[0].postMessage.mock.calls[0]?.[0] as
        Record<string, unknown> | undefined;
      expect(initCall?.canvasWidth).toBeUndefined();
      expect(initCall?.canvasHeight).toBeUndefined();
    });

    it('does not mutate a state with invalid canvas dimensions', async () => {
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
      const state: import('../renderer/frame').NeatensteinRenderState = {
        canvasWidth: 0,
        canvasHeight: 0,
        simTick: 1,
        cameraX: 1,
        cameraY: 2,
        cameraYaw: 0,
        mapSeed: 42,
      };
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'initialized' },
        } as unknown as MessageEvent);
      }
      bridge.postSimState(state);
      expect(worker.postMessage.mock.calls.at(-1)?.[0]).toEqual({
        type: 'simState',
        state,
      });
    });

    it('replaces an existing frame consumer when setFrameConsumer is called twice', async () => {
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
      const firstConsumer = jest.fn();
      const secondConsumer = jest.fn();
      (
        bridge as unknown as Record<
          string,
          (callback: (frame: Record<string, unknown>) => void) => void
        >
      ).setFrameConsumer(firstConsumer);
      (
        bridge as unknown as Record<
          string,
          (callback: (frame: Record<string, unknown>) => void) => void
        >
      ).setFrameConsumer(secondConsumer);
      const worker = instances[0];
      const frame = { requestId: 11, format: 'replacement' };
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame },
        } as unknown as MessageEvent);
      }
      expect(firstConsumer).not.toHaveBeenCalled();
      expect(secondConsumer).toHaveBeenCalledWith(frame);
    });

    it('ignores frame messages whose payload lacks a numeric requestId', async () => {
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
      if (typeof worker.onmessage === 'function') {
        worker.onmessage({
          data: { type: 'frame', frame: { requestId: 'not-a-number' } },
        } as unknown as MessageEvent);
      }
      expect(bridge.requestId).toBe(0);
    });
  });
});
