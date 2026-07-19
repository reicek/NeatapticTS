/**
 * Red tests for the generic worker auto-enable helpers.
 *
 * `shouldAutoEnableWorker` decides whether worker-thread evaluation should be
 * activated based on network size, batch parallelism, and the host's logical
 * core count. `autoEnableWorker` probes the host environment and, when eligible,
 * reports how many workers would be used.
 *
 * These tests import from `./acceleration.workers`, which does not exist yet, so
 * the suite fails with TS2307 "Cannot find module" until the implementation
 * slice (P4S3-05) lands.
 */

import {
  shouldAutoEnableWorker,
  autoEnableWorker,
  type WorkerAutoEnableResult,
  type AutoEnableWorkerOptions,
} from './acceleration.workers';
import {
  DEFAULT_ACCELERATION_WORKER_MIN_CORES,
  DEFAULT_ACCELERATION_MAX_WORKERS,
} from './acceleration.constants';

/**
 * Mock the browser runtime facts that worker auto-enable probes.
 *
 * Sets `globalThis.navigator.hardwareConcurrency` and
 * `globalThis.crossOriginIsolated` so the unit tests can exercise eligibility
 * without spawning real worker threads.
 */
const setNavigator = (
  hardwareConcurrency: number,
  crossOriginIsolated = true,
): void => {
  const globalLike = globalThis as unknown as Record<string, unknown>;

  globalLike.navigator = {
    hardwareConcurrency,
  } as unknown as Navigator;

  globalLike.crossOriginIsolated = crossOriginIsolated;
};

afterEach(() => {
  jest.restoreAllMocks();
  const globalLike = globalThis as unknown as Record<string, unknown>;
  delete globalLike.navigator;
  delete globalLike.crossOriginIsolated;
  delete globalLike.telemetry;
});

describe('acceleration.workers', () => {
  describe('shouldAutoEnableWorker', () => {
    it('returns true when nodeCount meets the worker threshold and cores meet workerMinCores', () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES + 2, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 100_000,
        } as AutoEnableWorkerOptions),
      ).toBe(true);
    });

    it('returns true when batchParallelCount meets the worker threshold', () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES + 2, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 0,
          batchParallelCount: 100_000,
        } as AutoEnableWorkerOptions),
      ).toBe(true);
    });

    it('returns false when both nodeCount and batchParallelCount are below the threshold', () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES + 2, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 0,
          batchParallelCount: 0,
        } as AutoEnableWorkerOptions),
      ).toBe(false);
    });

    it('returns false when disableWorkers is true', () => {
      setNavigator(8, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 100_000,
          config: { disableWorkers: true },
        } as AutoEnableWorkerOptions),
      ).toBe(false);
    });

    it('returns false when hasActiveWorker is true', () => {
      setNavigator(8, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 100_000,
          config: { hasActiveWorker: true },
        } as AutoEnableWorkerOptions),
      ).toBe(false);
    });

    it('returns false when hardwareConcurrency is below workerMinCores', () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES - 1, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 100_000,
        } as AutoEnableWorkerOptions),
      ).toBe(false);
    });

    it('honors a custom workerMinCores override', () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES, true);

      expect(
        shouldAutoEnableWorker({
          nodeCount: 100_000,
          config: { workerMinCores: 1 },
        } as AutoEnableWorkerOptions),
      ).toBe(true);
    });
  });

  describe('autoEnableWorker', () => {
    it('returns a structured result with enabled, workerCount, notified and reason', async () => {
      setNavigator(8, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result).toMatchObject({
        enabled: expect.any(Boolean),
        workerCount: expect.any(Number),
        notified: expect.any(Boolean),
        reason: expect.any(String),
      });
    });

    it('returns enabled true when eligible and the environment supports workers', async () => {
      setNavigator(8, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.enabled).toBe(true);
    });

    it('returns a positive workerCount when eligible and the environment supports workers', async () => {
      setNavigator(8, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.workerCount).toBeGreaterThan(0);
    });

    it('returns enabled false when disableWorkers is true', async () => {
      setNavigator(8, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
        config: { disableWorkers: true },
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.enabled).toBe(false);
    });

    it('returns enabled false when the network is below the auto-enable threshold', async () => {
      setNavigator(8, true);

      const result = (await autoEnableWorker({
        nodeCount: 0,
        batchParallelCount: 0,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.enabled).toBe(false);
    });

    it('returns the below-threshold and unavailable reason when both conditions hold', async () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES - 1, true);

      const result = (await autoEnableWorker({
        nodeCount: 0,
        batchParallelCount: 0,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result).toMatchObject({
        enabled: false,
        reason: 'Network below auto-enable threshold and workers not available',
      });
    });

    it('returns notified true when workers are available but the network is below the threshold', async () => {
      setNavigator(8, true);

      const result = (await autoEnableWorker({
        nodeCount: 0,
        batchParallelCount: 0,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.notified).toBe(true);
    });

    it('returns enabled false and notified false when workers are unavailable', async () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES - 1, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result).toMatchObject({
        enabled: false,
        notified: false,
      });
    });

    it('returns workerCount 0 when not enabled', async () => {
      setNavigator(DEFAULT_ACCELERATION_WORKER_MIN_CORES - 1, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.workerCount).toBe(0);
    });

    it('caps workerCount at maxWorkers', async () => {
      setNavigator(16, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
        config: { maxWorkers: DEFAULT_ACCELERATION_MAX_WORKERS },
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.workerCount).toBe(DEFAULT_ACCELERATION_MAX_WORKERS);
    });

    it('reserves one core for the main thread', async () => {
      const hardwareConcurrency = 6;
      setNavigator(hardwareConcurrency, true);

      const result = (await autoEnableWorker({
        nodeCount: 100_000,
        config: { maxWorkers: 1_000 },
      } as AutoEnableWorkerOptions)) as WorkerAutoEnableResult;

      expect(result.workerCount).toBe(hardwareConcurrency - 1);
    });
  });
});
