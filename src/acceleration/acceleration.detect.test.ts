/**
 * Red tests for the generic acceleration detection primitive.
 *
 * Probes host capability (WebGPU and hardware concurrency) and returns a
 * structured status object. The tests use a fake `navigator` so they run in
 * Node/Jest without a real browser environment.
 */

import { detectAcceleration } from './acceleration.detect';
import {
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  DEFAULT_ACCELERATION_MAX_WORKERS,
  DEFAULT_ACCELERATION_WORKER_MIN_CORES,
} from './acceleration.constants';

interface NavigatorFixture {
  hardwareConcurrency: number;
  gpuRequestAdapter?: jest.MockedFunction<() => Promise<GPUAdapter | null>>;
  crossOriginIsolated?: boolean;
}

const setGlobal = <T>(key: string, value: T): void => {
  (globalThis as unknown as Record<string, T>)[key] = value;
};

const setNavigator = (fixture: NavigatorFixture): void => {
  setGlobal('crossOriginIsolated', fixture.crossOriginIsolated ?? true);

  const navigatorLike: Record<string, unknown> = {
    hardwareConcurrency: fixture.hardwareConcurrency,
  };

  if (fixture.gpuRequestAdapter) {
    navigatorLike.gpu = { requestAdapter: fixture.gpuRequestAdapter };
  }

  setGlobal('navigator', navigatorLike as unknown as Navigator);
};

afterEach(() => {
  jest.restoreAllMocks();
  const g = globalThis as unknown as Record<string, unknown>;
  delete g.navigator;
  delete g.crossOriginIsolated;
  delete g.telemetry;
});

describe('acceleration.detect', () => {
  describe('detectAcceleration', () => {
    it('returns a structured status with selected mode, backend reports, cpu fallback and gapReasons', () => {
      setNavigator({
        hardwareConcurrency: 8,
        gpuRequestAdapter: jest
          .fn()
          .mockResolvedValue({} as unknown as GPUAdapter),
      });

      const status = detectAcceleration({}, 1024);

      expect(status).toMatchObject({
        mode: expect.any(String),
        gpu: {
          available: expect.any(Boolean),
          reason: expect.any(String),
        },
        worker: {
          available: expect.any(Boolean),
          count: expect.any(Number),
          reason: expect.any(String),
        },
        cpu: { available: true },
        gapReasons: expect.any(Array),
      });
    });

    it('reports gpu available when adapter resolves and node count meets the threshold', () => {
      setNavigator({
        hardwareConcurrency: 8,
        gpuRequestAdapter: jest
          .fn()
          .mockResolvedValue({} as unknown as GPUAdapter),
      });

      const status = detectAcceleration(
        {},
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(status.gpu).toMatchObject({
        available: true,
        reason: expect.stringMatching(/available|ready/i),
      });
    });

    it('reports gpu unavailable when node count is below the gpuNodeThreshold', () => {
      setNavigator({
        hardwareConcurrency: 8,
        gpuRequestAdapter: jest
          .fn()
          .mockResolvedValue({} as unknown as GPUAdapter),
      });

      const status = detectAcceleration(
        {},
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1,
      );

      expect(status.gpu).toMatchObject({
        available: false,
        reason: expect.stringMatching(/threshold|small/i),
      });
    });

    it('reports gpu unavailable when disableGPU is true', () => {
      setNavigator({
        hardwareConcurrency: 8,
        gpuRequestAdapter: jest
          .fn()
          .mockResolvedValue({} as unknown as GPUAdapter),
      });

      const status = detectAcceleration(
        { disableGPU: true },
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD * 2,
      );

      expect(status.gpu).toMatchObject({
        available: false,
        reason: expect.stringMatching(/disable|override|off/i),
      });
    });

    it('reports worker available when cores meet workerMinCores and crossOriginIsolated is true', () => {
      setNavigator({
        hardwareConcurrency: 4,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({}, 1024);

      expect(status.worker).toMatchObject({
        available: true,
        count: expect.any(Number),
        reason: expect.stringMatching(/available|ready/i),
      });
    });

    it('reports worker unavailable when crossOriginIsolated is false', () => {
      setNavigator({
        hardwareConcurrency: 8,
        crossOriginIsolated: false,
      });

      const status = detectAcceleration({}, 1024);

      expect(status.worker).toMatchObject({
        available: false,
        count: 0,
        reason: expect.stringMatching(/isolate|coop|coep|origin/i),
      });
    });

    it('caps worker count at maxWorkers', () => {
      setNavigator({
        hardwareConcurrency: 16,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({ maxWorkers: 2 }, 1024);

      expect(status.worker.count).toBe(2);
    });

    it('reserves one logical core for the main thread when computing worker count', () => {
      setNavigator({
        hardwareConcurrency: 6,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration(
        { maxWorkers: DEFAULT_ACCELERATION_MAX_WORKERS * 2 },
        1024,
      );

      expect(status.worker.count).toBe(5);
    });

    it('reports worker unavailable when hardwareConcurrency is below workerMinCores', () => {
      setNavigator({
        hardwareConcurrency: DEFAULT_ACCELERATION_WORKER_MIN_CORES - 1,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({}, 1024);

      expect(status.worker.available).toBe(false);
    });

    it('reports worker unavailable when disableWorkers is true', () => {
      setNavigator({
        hardwareConcurrency: 8,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({ disableWorkers: true }, 1024);

      expect(status.worker).toMatchObject({
        available: false,
        count: 0,
        reason: expect.stringMatching(/disable|override|off/i),
      });
    });

    it('selects cpu mode when both gpu and worker are unavailable', () => {
      setNavigator({
        hardwareConcurrency: 1,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({}, 1024);

      expect(status.mode).toBe('cpu');
    });

    it('includes a non-empty gapReasons array when at least one backend is unavailable', () => {
      setNavigator({
        hardwareConcurrency: 1,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({}, 1024);

      expect((status.gapReasons ?? []).length).toBeGreaterThan(0);
    });

    it('exposes gapReasons as an array when all requested backends are available', () => {
      setNavigator({
        hardwareConcurrency: 8,
        crossOriginIsolated: true,
        gpuRequestAdapter: jest
          .fn()
          .mockResolvedValue({} as unknown as GPUAdapter),
      });

      const status = detectAcceleration(
        {},
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      expect(Array.isArray(status.gapReasons)).toBe(true);
    });

    it('uses singular worker wording when exactly one worker is available', () => {
      setNavigator({
        hardwareConcurrency: 2,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration({}, 1024);

      expect(status.worker.reason).toBe('Workers available (1 worker)');
    });

    it('applies default empty config and zero node count when called with no arguments', () => {
      setNavigator({
        hardwareConcurrency: 8,
        crossOriginIsolated: true,
      });

      const status = detectAcceleration();

      expect(status.mode).toBe('worker');
    });

    it('falls back to one core when navigator is unavailable', () => {
      const g = globalThis as unknown as Record<string, unknown>;
      delete g.navigator;
      setGlobal('crossOriginIsolated', true);

      const status = detectAcceleration({}, 1024);

      expect(status.worker.available).toBe(false);
    });
  });
});
