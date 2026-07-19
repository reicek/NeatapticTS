/**
 * Red tests for the AccelerationManager lifecycle wrapper.
 *
 * `AccelerationManager` owns acceleration state over time: construction,
 * initialization, enable/disable transitions, status queries, observer
 * notifications, and teardown. It delegates backend selection to
 * `autoEnableAcceleration` and wraps a `LifecycleAccelerationPolicy` without
 * owning worker-pool resources.
 *
 * These tests import from `./acceleration.manager`, which does not exist yet,
 * so the suite fails with TS2307 until the implementation slice lands.
 */

jest.mock('./acceleration.gpu.device', () => ({
  requestGPUDevice: jest.fn().mockResolvedValue({} as GPUDevice),
}));

import { AccelerationManager } from './acceleration.manager';
import {
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  DEFAULT_ACCELERATION_WORKER_MIN_CORES,
} from './acceleration.constants';

/**
 * Configure the global navigator and cross-origin isolation state for a test.
 *
 * @param hardwareConcurrency - Logical core count to report.
 * @param hasGpu - Whether `navigator.gpu` should be present.
 */
const setNavigator = (hardwareConcurrency: number, hasGpu = true): void => {
  const navigatorLike: Record<string, unknown> = { hardwareConcurrency };

  if (hasGpu) {
    navigatorLike.gpu = {
      requestAdapter: jest.fn().mockResolvedValue({} as unknown as GPUAdapter),
    };
  }

  (globalThis as unknown as Record<string, unknown>).navigator =
    navigatorLike as unknown as Navigator;
  (globalThis as unknown as Record<string, unknown>).crossOriginIsolated = true;
};

afterEach(() => {
  jest.restoreAllMocks();
  const g = globalThis as unknown as Record<string, unknown>;
  delete g.navigator;
  delete g.crossOriginIsolated;
  delete g.telemetry;
});

describe('acceleration.manager', () => {
  describe('AccelerationManager', () => {
    it('exports the AccelerationManager class', () => {
      expect(typeof AccelerationManager).toBe('function');
    });

    it('constructs with default options', () => {
      const manager = new AccelerationManager();

      expect(manager).toBeInstanceOf(AccelerationManager);
    });

    it('constructs with a partial config', () => {
      const manager = new AccelerationManager({
        config: { disableGPU: true, backend: 'worker' },
      });

      expect(manager).toBeInstanceOf(AccelerationManager);
    });

    it('constructs with an observer', () => {
      const observer = { onBackendChange: jest.fn() };
      const manager = new AccelerationManager({ observer });

      expect(manager).toBeInstanceOf(AccelerationManager);
    });

    it('init() resolves to an AccelerationStatus', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      const status = await manager.init(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(status).toMatchObject({
        mode: expect.any(String),
        gpu: expect.any(Object),
        worker: expect.any(Object),
        cpu: expect.objectContaining({ available: true }),
      });
    });

    it('init() selects gpu when the environment supports it', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      const status = await manager.init(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(status.mode).toBe('gpu');
    });

    it('init() selects cpu when gpu and workers are disabled', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager({
        config: { disableGPU: true, disableWorkers: true },
      });

      const status = await manager.init(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(status.mode).toBe('cpu');
    });

    it('getStatus() returns the last resolved status after init()', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      const initialStatus = await manager.init(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );
      const queriedStatus = manager.getStatus();

      expect(queriedStatus).toBe(initialStatus);
    });

    it('getStatus() returns a default cpu status before init()', () => {
      const manager = new AccelerationManager();

      const status = manager.getStatus();

      expect(status.mode).toBe('cpu');
    });

    it('enable() transitions to the resolved backend after init()', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      const enabled = await manager.enable();

      expect(enabled).toBe(true);
    });

    it('enable() returns false before initialization', async () => {
      const manager = new AccelerationManager();

      const enabled = await manager.enable();

      expect(enabled).toBe(false);
    });

    it('disable() sets the active mode to cpu', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      await manager.enable();
      await manager.disable();

      expect(manager.getStatus().mode).toBe('cpu');
    });

    it('disable() is safe to call before init()', async () => {
      const manager = new AccelerationManager();

      await expect(manager.disable()).resolves.toBeUndefined();
    });

    it('teardown() resets the manager to a clean state', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      await manager.enable();
      await manager.teardown();

      expect(manager.getStatus().mode).toBe('cpu');
    });

    it('reEvaluate() updates status after a topology mutation', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      const updated = await manager.reEvaluate(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD * 2,
      );

      expect(updated).toBeInstanceOf(Object);
    });

    it('notifies the observer on backend selection during init()', async () => {
      setNavigator(8, true);
      const observer = { onBackendChange: jest.fn() };
      const manager = new AccelerationManager({ observer });

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);

      expect(observer.onBackendChange).toHaveBeenCalled();
    });

    it('notifies the observer when disable() changes the backend', async () => {
      setNavigator(8, true);
      const observer = { onBackendChange: jest.fn() };
      const manager = new AccelerationManager({ observer });

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      await manager.enable();
      observer.onBackendChange.mockClear();
      await manager.disable();

      expect(observer.onBackendChange).toHaveBeenCalled();
    });

    it('returns the existing status when init() is called twice', async () => {
      setNavigator(8, true);
      const manager = new AccelerationManager();

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      const secondStatus = await manager.init(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(secondStatus).toBe(manager.getStatus());
    });

    it('does not notify the observer again when init() is called twice', async () => {
      setNavigator(8, true);
      const observer = { onBackendChange: jest.fn() };
      const manager = new AccelerationManager({ observer });

      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);
      observer.onBackendChange.mockClear();
      await manager.init(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);

      expect(observer.onBackendChange).not.toHaveBeenCalled();
    });

    it('falls back to worker when gpu is disabled', async () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES, false);
      const manager = new AccelerationManager({ config: { disableGPU: true } });

      const status = await manager.init(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(status.mode).toBe('worker');
    });
  });
});
