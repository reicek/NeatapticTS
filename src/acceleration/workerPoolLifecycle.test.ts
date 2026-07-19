/**
 * Red tests for the centralized worker pool lifecycle manager.
 *
 * `WorkerPoolLifecycle` owns creation, reuse, and teardown of worker pools.
 * It is instantiated through `createWorkerPoolLifecycle()` and is never a
 * module-level singleton. These tests import from `./workerPoolLifecycle`,
 * which does not exist yet, so the focused run fails with TS2307
 * "Cannot find module" until the implementation slice (P5S3-02) lands.
 *
 * External dependencies (the global `Worker` constructor and browser runtime
 * facts) are mocked so the suite stays deterministic and does not spawn real
 * worker threads.
 */

import { createWorkerPoolLifecycle } from './workerPoolLifecycle';
import type { AccelerationConfig } from './acceleration.types';
import type { AccelerationObserver } from './acceleration.observer';

/**
 * Minimal mock worker used to verify lifecycle behavior without real threads.
 *
 * The mock records `postMessage` and `terminate` calls and exposes the event
 * handlers that the lifecycle manager is expected to attach for health
 * monitoring.
 */
type MockWorker = Worker & {
  postMessage: jest.Mock<void, [unknown, Transferable[] | undefined]>;
  terminate: jest.Mock<void, []>;
};

function createMockWorker(): MockWorker {
  return {
    postMessage: jest.fn(),
    terminate: jest.fn(),
    onmessage: null,
    onerror: null,
    onmessageerror: null,
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  } as unknown as MockWorker;
}

const globalRecord = globalThis as unknown as Record<string, unknown>;
const originalWorker = globalRecord.Worker as typeof Worker | undefined;

function installMockWorkerFactory(): () => void {
  globalRecord.Worker = jest.fn(createMockWorker) as unknown as typeof Worker;
  return () => {
    if (originalWorker !== undefined) {
      globalRecord.Worker = originalWorker;
    } else {
      delete globalRecord.Worker;
    }
  };
}

afterEach(() => {
  jest.restoreAllMocks();
  if (originalWorker !== undefined) {
    globalRecord.Worker = originalWorker;
  } else {
    delete globalRecord.Worker;
  }
});

describe('workerPoolLifecycle', () => {
  describe('createWorkerPoolLifecycle', () => {
    it('returns a scoped lifecycle instance', () => {
      const lifecycle = createWorkerPoolLifecycle();

      expect(lifecycle).toMatchObject({
        create: expect.any(Function),
        reuse: expect.any(Function),
        dispose: expect.any(Function),
        activeCount: expect.any(Function),
      });
    });

    it('returns a distinct instance on every call', () => {
      const firstLifecycle = createWorkerPoolLifecycle();
      const secondLifecycle = createWorkerPoolLifecycle();

      expect(firstLifecycle).not.toBe(secondLifecycle);
    });
  });

  describe('WorkerPoolLifecycle activeCount', () => {
    it('reports zero active workers before create is called', () => {
      const lifecycle = createWorkerPoolLifecycle();

      expect(lifecycle.activeCount()).toBe(0);
    });
  });

  describe('WorkerPoolLifecycle.create', () => {
    it('returns a WorkerPoolHandle with broadcast and terminate methods', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      const handle = await lifecycle.create();

      expect(handle).toMatchObject({
        broadcast: expect.any(Function),
        terminate: expect.any(Function),
      });
    });

    it('increases activeCount when workers are created', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      await lifecycle.create();

      expect(lifecycle.activeCount()).toBeGreaterThan(0);
    });
  });

  describe('WorkerPoolLifecycle.reuse', () => {
    it('returns undefined before a pool has been created', () => {
      const lifecycle = createWorkerPoolLifecycle();

      expect(lifecycle.reuse()).toBeUndefined();
    });

    it('returns the same handle after create', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      const handle = await lifecycle.create();

      expect(lifecycle.reuse()).toBe(handle);
    });
  });

  describe('WorkerPoolHandle.broadcast', () => {
    it('can broadcast a message without throwing', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      const handle = await lifecycle.create();

      expect(() => handle.broadcast({ kind: 'ping' })).not.toThrow();
    });
  });

  describe('WorkerPoolHandle.terminate', () => {
    it('reduces activeCount to zero', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      const handle = await lifecycle.create();
      await handle.terminate();

      expect(lifecycle.activeCount()).toBe(0);
    });
  });

  describe('WorkerPoolLifecycle.dispose', () => {
    it('reduces activeCount to zero', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      await lifecycle.create();
      await lifecycle.dispose();

      expect(lifecycle.activeCount()).toBe(0);
    });
  });

  describe('Pool size management', () => {
    it('caps activeCount at maxWorkers from config', async () => {
      installMockWorkerFactory();
      const config: Partial<AccelerationConfig> = { maxWorkers: 2 };
      const lifecycle = createWorkerPoolLifecycle(config as AccelerationConfig);
      await lifecycle.create();

      expect(lifecycle.activeCount()).toBeLessThanOrEqual(2);
    });
  });

  describe('Health checking and recovery', () => {
    it('reuse returns undefined after the handle is terminated', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      const handle = await lifecycle.create();
      await handle.terminate();

      expect(lifecycle.reuse()).toBeUndefined();
    });

    it('can create a new handle after the previous one is terminated', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      const firstHandle = await lifecycle.create();
      await firstHandle.terminate();
      const secondHandle = await lifecycle.create();

      expect(secondHandle).toMatchObject({
        broadcast: expect.any(Function),
        terminate: expect.any(Function),
      });
    });
  });

  describe('Integration with acceleration config', () => {
    it('does not create active workers when workers are disabled', async () => {
      installMockWorkerFactory();
      const config: Partial<AccelerationConfig> = { disableWorkers: true };
      const lifecycle = createWorkerPoolLifecycle(config as AccelerationConfig);
      await lifecycle.create();

      expect(lifecycle.activeCount()).toBe(0);
    });
  });

  describe('Observer notifications', () => {
    it('notifies the observer when workers are created', async () => {
      installMockWorkerFactory();
      const observer: AccelerationObserver = {
        onBackendChange: jest.fn(),
      };
      const lifecycle = createWorkerPoolLifecycle({}, observer);
      await lifecycle.create();

      expect(observer.onBackendChange).toHaveBeenCalled();
    });

    it('does not crash when observer has no onBackendChange callback', async () => {
      installMockWorkerFactory();
      const observer: AccelerationObserver = {};
      const lifecycle = createWorkerPoolLifecycle({}, observer);
      const handle = await lifecycle.create();

      expect(handle).toBeDefined();
    });

    it('falls back to Date.now() when global performance is unavailable', async () => {
      installMockWorkerFactory();
      const originalPerformance = globalRecord.performance;
      const dateNowSpy = jest.spyOn(Date, 'now').mockReturnValue(987_654);
      globalRecord.performance = undefined;

      try {
        const observer: AccelerationObserver = {
          onBackendChange: jest.fn(),
        };
        const lifecycle = createWorkerPoolLifecycle({}, observer);
        await lifecycle.create();

        expect(observer.onBackendChange).toHaveBeenCalledWith(
          expect.objectContaining({ timestamp: 987_654 }),
        );
      } finally {
        dateNowSpy.mockRestore();
        globalRecord.performance = originalPerformance;
      }
    });
  });

  describe('WorkerPoolHandle.broadcast', () => {
    it('no-ops when broadcasting a terminated handle', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle({
        maxWorkers: 1,
      } as AccelerationConfig);
      const handle = await lifecycle.create();
      const worker = (globalRecord.Worker as jest.Mock).mock.results[0]
        .value as MockWorker;

      await handle.terminate();
      handle.broadcast({ kind: 'ping' });

      expect(worker.postMessage).not.toHaveBeenCalled();
    });
  });

  describe('WorkerPoolHandle.terminate', () => {
    it('no-ops when terminating an already terminated handle', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle({
        maxWorkers: 1,
      } as AccelerationConfig);
      const handle = await lifecycle.create();
      const worker = (globalRecord.Worker as jest.Mock).mock.results[0]
        .value as MockWorker;

      await handle.terminate();
      await handle.terminate();

      expect(worker.terminate).toHaveBeenCalledTimes(1);
    });

    it('terminates a handle that is not the active handle', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle({
        maxWorkers: 1,
      } as AccelerationConfig);
      const handle = await lifecycle.create();
      const worker = (globalRecord.Worker as jest.Mock).mock.results[0]
        .value as MockWorker;

      // Prevent the first handle from clearing the active handle during a second create.
      const originalTerminate = handle.terminate;
      handle.terminate = jest.fn(() =>
        Promise.resolve(),
      ) as unknown as () => Promise<void>;
      await lifecycle.create();
      handle.terminate = originalTerminate;

      await handle.terminate();

      expect(worker.terminate).toHaveBeenCalled();
    });
  });

  describe('WorkerPoolLifecycle.create', () => {
    it('throws when the lifecycle has been disposed', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle();
      await lifecycle.dispose();

      await expect(lifecycle.create()).rejects.toThrow(
        'WorkerPoolLifecycle has been disposed',
      );
    });

    it('terminates the previous active handle before creating a new pool', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle({
        maxWorkers: 1,
      } as AccelerationConfig);
      await lifecycle.create();
      const firstWorker = (globalRecord.Worker as jest.Mock).mock.results[0]
        .value as MockWorker;

      await lifecycle.create();

      expect(firstWorker.terminate).toHaveBeenCalled();
    });

    it('returns an empty handle when the Worker constructor is missing', async () => {
      delete globalRecord.Worker;
      const lifecycle = createWorkerPoolLifecycle();
      await lifecycle.create();

      expect(lifecycle.activeCount()).toBe(0);
    });
  });

  describe('WorkerPoolLifecycle.dispose', () => {
    it('no-ops when disposing an already disposed lifecycle', async () => {
      installMockWorkerFactory();
      const lifecycle = createWorkerPoolLifecycle({
        maxWorkers: 1,
      } as AccelerationConfig);
      await lifecycle.create();
      const worker = (globalRecord.Worker as jest.Mock).mock.results[0]
        .value as MockWorker;

      await lifecycle.dispose();
      await lifecycle.dispose();

      expect(worker.terminate).toHaveBeenCalledTimes(1);
    });
  });
});
