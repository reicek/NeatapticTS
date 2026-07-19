/**
 * Red tests for acceleration policies.
 *
 * `AccelerationPolicy` encodes the decision rules that map a network and host
 * capability to a backend choice. `LifecycleAccelerationPolicy` extends this
 * with hooks for initialization, mutation, and teardown so the backend can be
 * re-verified after structural changes.
 */

import {
  AccelerationPolicy,
  LifecycleAccelerationPolicy,
} from './acceleration.policy';
import { detectAcceleration } from './acceleration.detect';
import { NoopAccelerationObserver } from './acceleration.observer';
import type { AccelerationObserver } from './acceleration.observer';
import type {
  AccelerationConfig,
  AccelerationStatus,
  BackendMode,
} from './acceleration.types';
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
  delete g.telemetry;
});

describe('acceleration.policy', () => {
  describe('AccelerationPolicy', () => {
    it('exposes a default policy instance', () => {
      expect(AccelerationPolicy.default).toBeInstanceOf(AccelerationPolicy);
    });

    it('constructs with no arguments', () => {
      const policy = new AccelerationPolicy();
      expect(policy).toBeInstanceOf(AccelerationPolicy);
    });

    it('constructs with a partial config', () => {
      const policy = new AccelerationPolicy({ backend: 'cpu' });
      expect(policy).toBeInstanceOf(AccelerationPolicy);
    });

    it('decide returns a decision object with backend and reason strings', () => {
      const policy = new AccelerationPolicy();
      const status: AccelerationStatus = {
        mode: 'cpu',
        gpu: { available: false, reason: 'no gpu' },
        worker: { available: false, count: 0, reason: 'no workers' },
      };

      const decision = policy.decide(status);

      expect(decision).toMatchObject({
        backend: expect.any(String),
        reason: expect.any(String),
      });
    });

    it('decide returns cpu for a cpu-only status when backend is auto', () => {
      const policy = new AccelerationPolicy();
      const status: AccelerationStatus = {
        mode: 'cpu',
        gpu: { available: false, reason: 'no gpu' },
        worker: { available: false, count: 0, reason: 'no workers' },
      };

      const decision = policy.decide(status, undefined, { backend: 'auto' });

      expect(decision.backend).toBe('cpu');
    });

    it('decide honors explicit gpu override even when gpu unavailable in status', () => {
      const policy = new AccelerationPolicy();
      const status: AccelerationStatus = {
        mode: 'cpu',
        gpu: { available: false, reason: 'no gpu' },
        worker: { available: false, count: 0, reason: 'no workers' },
      };

      const decision = policy.decide(status, undefined, { backend: 'gpu' });

      expect(decision.backend).toBe('gpu');
    });

    it('decide honors explicit cpu override even when gpu is available', () => {
      const policy = new AccelerationPolicy();
      const status: AccelerationStatus = {
        mode: 'gpu',
        gpu: { available: true, reason: 'gpu ready' },
        worker: { available: false, count: 0, reason: 'no workers' },
      };

      const decision = policy.decide(status, undefined, { backend: 'cpu' });

      expect(decision.backend).toBe('cpu');
    });

    it('decide honors explicit worker override when workers are available', () => {
      const policy = new AccelerationPolicy();
      const status = workerOnlyStatus();

      const decision = policy.decide(status, undefined, {
        backend: 'worker' as unknown as BackendMode,
      });

      expect(decision.backend).toBe('worker');
    });

    it('decide returns a reason string explaining the decision', () => {
      const policy = new AccelerationPolicy();
      const status: AccelerationStatus = {
        mode: 'cpu',
        gpu: { available: false, reason: 'no gpu' },
        worker: { available: false, count: 0, reason: 'no workers' },
      };

      const decision = policy.decide(status, undefined, { backend: 'auto' });

      expect(decision.reason).toEqual(expect.any(String));
    });
  });

  describe('LifecycleAccelerationPolicy', () => {
    it('is an AccelerationPolicy subclass', () => {
      expect(new LifecycleAccelerationPolicy()).toBeInstanceOf(
        AccelerationPolicy,
      );
    });

    it('constructs with no arguments', () => {
      const policy = new LifecycleAccelerationPolicy();
      expect(policy).toBeInstanceOf(LifecycleAccelerationPolicy);
    });

    it('constructs with a partial config', () => {
      const policy = new LifecycleAccelerationPolicy({ backend: 'cpu' });
      expect(policy).toBeInstanceOf(LifecycleAccelerationPolicy);
    });

    it('isTopologyDirty starts false', () => {
      const policy = new LifecycleAccelerationPolicy();
      expect(policy.isTopologyDirty).toBe(false);
    });

    it('needsReverification starts false', () => {
      const policy = new LifecycleAccelerationPolicy();
      expect(policy.needsReverification).toBe(false);
    });

    it('onMutated marks topology dirty', () => {
      const policy = new LifecycleAccelerationPolicy();
      policy.onMutated();
      expect(policy.isTopologyDirty).toBe(true);
    });

    it('needsReverification is true after mutation', () => {
      const policy = new LifecycleAccelerationPolicy();
      policy.onMutated();
      expect(policy.needsReverification).toBe(true);
    });

    it('clearDirty clears the dirty flag', () => {
      const policy = new LifecycleAccelerationPolicy();
      policy.onMutated();
      policy.clearDirty();
      expect(policy.isTopologyDirty).toBe(false);
    });

    it('clearDirty also clears needsReverification', () => {
      const policy = new LifecycleAccelerationPolicy();
      policy.onMutated();
      policy.clearDirty();
      expect(policy.needsReverification).toBe(false);
    });

    it('evaluate returns a structured AccelerationStatus', () => {
      const policy = new LifecycleAccelerationPolicy();
      gpuAvailableStatus(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD);

      const resolved = policy.evaluate(
        {},
        DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      );

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

    it('evaluate selects gpu when gpu is available and no worker owns execution', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
      gpuAvailableStatus(nodeCount);

      const resolved = policy.evaluate({}, nodeCount);

      expect(resolved.mode).toBe('gpu');
    });

    it('evaluate selects worker when an active worker pool owns execution', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
      gpuAvailableStatus(nodeCount);

      const resolved = policy.evaluate(
        { hasActiveWorker: true } as Partial<AccelerationConfig> & {
          hasActiveWorker?: boolean;
        },
        nodeCount,
      );

      expect(resolved.mode).toBe('worker');
    });

    it('evaluate selects cpu when neither gpu nor worker are available', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1;
      belowThresholdNoWorkers(nodeCount);

      const resolved = policy.evaluate({}, nodeCount);

      expect(resolved.mode).toBe('cpu');
    });

    it('evaluate honors an explicit backend cpu override', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
      gpuAvailableStatus(nodeCount);

      const resolved = policy.evaluate({ backend: 'cpu' }, nodeCount);

      expect(resolved.mode).toBe('cpu');
    });

    it('evaluate notifies the observer when resolved mode differs from detected mode', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
      gpuAvailableStatus(nodeCount);
      const onBackendChange = jest.fn();
      const observer: AccelerationObserver = { onBackendChange };

      policy.evaluate({ backend: 'cpu' }, nodeCount, observer);

      expect(onBackendChange).toHaveBeenCalledWith(
        expect.objectContaining({
          previous: 'gpu',
          current: 'cpu',
          reason: expect.any(String),
        }),
      );
    });

    it('evaluate does not notify observer when resolved mode matches detected mode', () => {
      const policy = new LifecycleAccelerationPolicy();
      cpuOnlyStatus();
      const onBackendChange = jest.fn();
      const observer: AccelerationObserver = { onBackendChange };

      policy.evaluate({}, 1024, observer);

      expect(onBackendChange).not.toHaveBeenCalled();
    });

    it('evaluate accepts the no-op observer safely', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
      gpuAvailableStatus(nodeCount);

      const resolved = policy.evaluate({}, nodeCount, NoopAccelerationObserver);

      expect(resolved.mode).toBeDefined();
    });

    it('evaluate uses default parameters when called with no arguments', () => {
      const policy = new LifecycleAccelerationPolicy();

      const resolved = policy.evaluate();

      expect(resolved.mode).toBeDefined();
    });

    it('evaluate still returns a status after the topology is marked dirty', () => {
      const policy = new LifecycleAccelerationPolicy();
      const nodeCount = DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
      gpuAvailableStatus(nodeCount);
      policy.onMutated();

      const resolved = policy.evaluate({}, nodeCount);

      expect(resolved).toMatchObject({
        mode: expect.any(String),
      });
    });
  });
});
