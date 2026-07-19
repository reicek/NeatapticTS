/**
 * Red tests for acceleration-mode resolution (Phase 3 contract).
 *
 * `resolveAccelerationMode` takes a detected {@link AccelerationStatus} and an
 * options bag, then returns a structured {@link AccelerationStatus} whose active
 * mode reflects GPU, worker, CPU or mixed-mode verdict. Worker ownership beats
 * GPU availability unless an explicit `backend` override is supplied.
 *
 * These tests import from `./acceleration.resolve`, which does not exist yet, so
 * TypeScript compilation fails with TS2307 until the implementation slice lands.
 */

import { detectAcceleration } from './acceleration.detect';
import { NoopAccelerationObserver } from './acceleration.observer';
import type {
  AccelerationBackendChangeEvent,
  AccelerationObserver,
} from './acceleration.observer';
import { resolveAccelerationMode } from './acceleration.resolve';
import type { AccelerationStatus } from './acceleration.types';
import { DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD } from './acceleration.constants';

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

const gpuAvailableStatus = (nodeCount: number = 2048): AccelerationStatus => {
  setNavigator({
    hardwareConcurrency: 8,
    crossOriginIsolated: true,
    gpuRequestAdapter: jest.fn().mockResolvedValue({} as unknown as GPUAdapter),
  });
  return detectAcceleration({}, nodeCount);
};

const workerOnlyStatus = (nodeCount: number = 2048): AccelerationStatus => {
  setNavigator({
    hardwareConcurrency: 4,
    crossOriginIsolated: true,
  });
  return detectAcceleration({}, nodeCount);
};

const cpuOnlyStatus = (nodeCount: number = 2048): AccelerationStatus => {
  setNavigator({
    hardwareConcurrency: 1,
    crossOriginIsolated: true,
  });
  return detectAcceleration({}, nodeCount);
};

const belowThresholdNoWorkers = (nodeCount: number): AccelerationStatus => {
  setNavigator({
    hardwareConcurrency: 1,
    crossOriginIsolated: true,
    gpuRequestAdapter: jest.fn().mockResolvedValue({} as unknown as GPUAdapter),
  });
  return detectAcceleration({}, nodeCount);
};

afterEach(() => {
  jest.restoreAllMocks();
  const g = globalThis as unknown as Record<string, unknown>;
  delete g.navigator;
  delete g.crossOriginIsolated;
});

describe('acceleration.resolve', () => {
  describe('resolveAccelerationMode', () => {
    it('returns a structured AccelerationStatus preserving capabilities and gapReasons', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status);

      expect(resolved).toMatchObject({
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

    it('resolves to cpu when the detected status reports cpu mode', () => {
      const status = cpuOnlyStatus();

      const resolved = resolveAccelerationMode(status);

      expect(resolved.mode).toBe('cpu');
    });

    it('resolves to gpu when gpu is available and no worker owns execution', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, {
        hasActiveWorker: false,
      });

      expect(resolved.mode).toBe('gpu');
    });

    it('resolves to worker when an active worker pool owns execution', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, {
        hasActiveWorker: true,
      });

      expect(resolved.mode).toBe('worker');
    });

    it('prefers worker over gpu when both are available and a worker is active', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, {
        hasActiveWorker: true,
      });

      expect(resolved.mode).toBe('worker');
    });

    it('returns gpu when both gpu and workers are available but no worker is active', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, {
        hasActiveWorker: false,
      });

      expect(resolved.mode).toBe('gpu');
    });

    it('falls back to worker when gpu is unavailable but workers are available', () => {
      const status = workerOnlyStatus();

      const resolved = resolveAccelerationMode(status);

      expect(resolved.mode).toBe('worker');
    });

    it('falls back to cpu when an active worker is claimed but workers are unavailable', () => {
      const status = cpuOnlyStatus();

      const resolved = resolveAccelerationMode(status, {
        hasActiveWorker: true,
      });

      expect(resolved.mode).toBe('cpu');
    });

    it('does not crash when observer lacks onBackendChange and resolved mode differs', () => {
      const status = gpuAvailableStatus();
      const observer: AccelerationObserver = { onFallback: jest.fn() };

      const resolved = resolveAccelerationMode(status, {
        backend: 'cpu',
        observer,
      });

      expect(resolved.mode).toBe('cpu');
    });

    it('honours an explicit backend override of cpu', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, { backend: 'cpu' });

      expect(resolved.mode).toBe('cpu');
    });

    it('honours an explicit backend override of worker', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, { backend: 'worker' });

      expect(resolved.mode).toBe('worker');
    });

    it('honours an explicit backend override of gpu even when workers are active', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, {
        backend: 'gpu',
        hasActiveWorker: true,
      });

      expect(resolved.mode).toBe('gpu');
    });

    it('honours an explicit backend override of auto by using the detected mode', () => {
      const status = workerOnlyStatus();

      const resolved = resolveAccelerationMode(status, { backend: 'auto' });

      expect(resolved.mode).toBe(status.mode);
    });

    it('populates gapReasons in the returned status when capabilities are insufficient', () => {
      const status = cpuOnlyStatus();

      const resolved = resolveAccelerationMode(status);

      expect((resolved.gapReasons ?? []).length).toBeGreaterThan(0);
    });

    it('always reports cpu as available in the returned status', () => {
      const status = cpuOnlyStatus();

      const resolved = resolveAccelerationMode(status);

      expect(resolved.cpu).toMatchObject({ available: true });
    });

    it('emits a backend change event through the observer when the resolved mode differs', () => {
      const status = gpuAvailableStatus();
      const onBackendChange = jest.fn();
      const observer: AccelerationObserver = { onBackendChange };

      resolveAccelerationMode(status, {
        backend: 'cpu',
        observer,
      });

      expect(onBackendChange).toHaveBeenCalledWith(
        expect.objectContaining({
          previous: status.mode,
          current: 'cpu',
          reason: expect.any(String),
        }),
      );
    });

    it('does not emit a backend change event when the resolved mode matches the detected mode', () => {
      const status = cpuOnlyStatus();
      const onBackendChange = jest.fn();
      const observer: AccelerationObserver = { onBackendChange };

      resolveAccelerationMode(status, { observer });

      expect(onBackendChange).not.toHaveBeenCalled();
    });

    it('falls back to cpu when node count is below the gpu threshold', () => {
      const status = belowThresholdNoWorkers(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1,
      );

      const resolved = resolveAccelerationMode(status);

      expect(resolved.mode).toBe('cpu');
    });

    it('selects gpu when node count meets the gpu threshold and gpu is available', () => {
      const status = gpuAvailableStatus(
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

      const resolved = resolveAccelerationMode(status, {
        hasActiveWorker: false,
      });

      expect(resolved.mode).toBe('gpu');
    });

    it('defaults to cpu for a minimal status with no available backends', () => {
      const status: AccelerationStatus = {
        mode: 'cpu',
        gpu: { available: false, reason: 'no gpu' },
        worker: { available: false, count: 0, reason: 'no workers' },
      };

      const resolved = resolveAccelerationMode(status);

      expect(resolved.mode).toBe('cpu');
    });

    it('accepts the no-op observer as a safe default', () => {
      const status = gpuAvailableStatus();

      const resolved = resolveAccelerationMode(status, {
        observer: NoopAccelerationObserver,
      });

      expect(resolved.mode).toBeDefined();
    });

    describe('collectGapReasons short-circuit branches', () => {
      it('excludes the gpu reason when gpu is marked available', () => {
        const status: AccelerationStatus = {
          mode: 'cpu',
          gpu: { available: true, reason: 'gpu unavailable for test' },
          worker: { available: false, count: 0, reason: 'no workers' },
        };

        const resolved = resolveAccelerationMode(status);

        expect(resolved.gapReasons).not.toContain('gpu unavailable for test');
      });

      it('excludes the worker reason when worker is marked available', () => {
        const status: AccelerationStatus = {
          mode: 'cpu',
          gpu: { available: false, reason: 'no gpu' },
          worker: {
            available: true,
            count: 4,
            reason: 'worker unavailable for test',
          },
        };

        const resolved = resolveAccelerationMode(status);

        expect(resolved.gapReasons).not.toContain(
          'worker unavailable for test',
        );
      });
    });

    describe('buildChangeEvent timestamp fallback', () => {
      let originalPerformance: Performance;
      let capturedEvent: AccelerationBackendChangeEvent | undefined;

      beforeEach(() => {
        originalPerformance = globalThis.performance;
        setGlobal<Performance>('performance', {
          now: 'not-a-function',
        } as unknown as Performance);
        jest.spyOn(Date, 'now').mockReturnValue(1_234_567_890);
        capturedEvent = undefined;
        const observer: AccelerationObserver = {
          onBackendChange: (event) => {
            capturedEvent = event;
          },
        };
        const status: AccelerationStatus = {
          mode: 'cpu',
          gpu: { available: true, reason: 'gpu ready' },
          worker: { available: false, count: 0, reason: 'no workers' },
        };
        resolveAccelerationMode(status, { backend: 'gpu', observer });
      });

      afterEach(() => {
        setGlobal('performance', originalPerformance);
      });

      it('falls back to Date.now when performance.now is not a function', () => {
        expect(capturedEvent?.timestamp).toBe(1_234_567_890);
      });
    });
  });
});
