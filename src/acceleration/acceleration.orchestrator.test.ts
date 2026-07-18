/**
 * Red tests for the generic acceleration auto-enable orchestrator.
 *
 * `autoEnableAcceleration` combines the GPU and worker auto-enable helpers into a
 * single unified decision. It returns an {@link AccelerationStatus} with the
 * selected mode, per-backend reports, CPU fallback, and gap reasons. It must
 * honour config overrides (`disableGPU`, `disableWorkers`) and emit observer
 * telemetry when a backend is selected.
 *
 * These tests import from `./acceleration.orchestrator`, which does not exist
 * yet, so the suite fails with TS2307 until the implementation slice lands.
 */

jest.mock('./acceleration.gpu.device', () => ({
  requestGPUDevice: jest.fn().mockResolvedValue({} as GPUDevice),
}));

import { autoEnableAcceleration } from './acceleration.orchestrator';
import { requestGPUDevice } from './acceleration.gpu.device';
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

describe('acceleration.orchestrator', () => {
  describe('autoEnableAcceleration', () => {
    it('returns a status object with mode, gpu, worker and cpu reports', async () => {
      setNavigator(8, true);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(status).toMatchObject({
        mode: expect.any(String),
        gpu: expect.objectContaining({
          available: expect.any(Boolean),
          reason: expect.any(String),
        }),
        worker: expect.objectContaining({
          available: expect.any(Boolean),
          count: expect.any(Number),
          reason: expect.any(String),
        }),
        cpu: expect.objectContaining({ available: true }),
      });
    });

    it('selects gpu mode when both gpu and worker are available', async () => {
      setNavigator(8, true);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(status.mode).toBe('gpu');
    });

    it('selects worker mode when gpu is disabled and worker is available', async () => {
      setNavigator(8, true);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        config: { disableGPU: true },
      });

      expect(status.mode).toBe('worker');
    });

    it('selects gpu mode when workers are disabled and gpu is available', async () => {
      setNavigator(8, true);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        config: { disableWorkers: true },
      });

      expect(status.mode).toBe('gpu');
    });

    it('falls back to cpu when both gpu and workers are disabled', async () => {
      setNavigator(8, true);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        config: { disableGPU: true, disableWorkers: true },
      });

      expect(status.mode).toBe('cpu');
    });

    it('exposes a non-null gpu device when gpu is enabled', async () => {
      setNavigator(8, true);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(status.gpu.device).not.toBeNull();
    });

    it('reports a positive worker count when workers are enabled', async () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES, false);

      const status = await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(status.worker.count).toBeGreaterThan(0);
    });

    it('does not call requestGPUDevice when disableGPU is true', async () => {
      setNavigator(8, true);
      jest.mocked(requestGPUDevice).mockClear();

      await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        config: { disableGPU: true },
      });

      expect(requestGPUDevice).not.toHaveBeenCalled();
    });

    it('notifies the observer when a backend is selected', async () => {
      setNavigator(8, true);
      const observer = { onBackendChange: jest.fn() };

      await autoEnableAcceleration({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        observer,
      });

      expect(observer.onBackendChange).toHaveBeenCalled();
    });

    it('falls back to cpu for a small network with no available accelerator', async () => {
      setNavigator(1, false);

      const status = await autoEnableAcceleration({ nodeCount: 16 });

      expect(status.mode).toBe('cpu');
    });

    it('falls back to cpu when gpu and worker report no reasons', async () => {
      jest.resetModules();
      jest.doMock('./acceleration.gpu', () => ({
        autoEnableGpu: jest.fn().mockResolvedValue({
          enabled: false,
          gpuDevice: null,
          notified: false,
          reason: '',
        }),
      }));
      jest.doMock('./acceleration.workers', () => ({
        autoEnableWorker: jest.fn().mockResolvedValue({
          enabled: false,
          workerCount: 0,
          notified: false,
          reason: '',
        }),
      }));

      const { autoEnableAcceleration: autoEnableAccelerationMocked } =
        await import('./acceleration.orchestrator');

      const status = await autoEnableAccelerationMocked({ nodeCount: 2_048 });

      expect(status.mode).toBe('cpu');
    });
  });
});
