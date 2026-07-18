/**
 * Red tests for the generic GPU auto-enable helpers.
 *
 * `shouldAutoEnableGpu` decides whether a GPU backend should be activated based
 * on network size and batch-parallel count. `autoEnableGpu` probes the host
 * and, when eligible, requests a WebGPU device. These tests import from
 * `./acceleration.gpu`, which does not exist yet, so the suite fails with
 * TS2307 until the implementation slice lands.
 */

jest.mock('./acceleration.gpu.device', () => ({
  requestGPUDevice: jest.fn().mockResolvedValue({} as GPUDevice),
}));

import {
  shouldAutoEnableGpu,
  autoEnableGpu,
  evaluateWeightVariantsOnGpu,
} from './acceleration.gpu';
import { requestGPUDevice } from './acceleration.gpu.device';
import {
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD,
} from './acceleration.constants';
import { DEFAULT_VARIANT_SCORER } from './acceleration.variants';
import type {
  VariantEvaluationNetwork,
  WeightVariant,
} from './acceleration.types';

const setNavigator = (hasGpu = true): void => {
  const navigatorLike: Record<string, unknown> = { hardwareConcurrency: 8 };

  if (hasGpu) {
    navigatorLike.gpu = {
      requestAdapter: jest.fn().mockResolvedValue({} as unknown as GPUAdapter),
    };
  }

  (globalThis as unknown as Record<string, unknown>).navigator =
    navigatorLike as unknown as Navigator;
};

afterEach(() => {
  jest.restoreAllMocks();
  const g = globalThis as unknown as Record<string, unknown>;
  delete g.navigator;
  delete g.telemetry;
});

describe('acceleration.gpu', () => {
  describe('shouldAutoEnableGpu', () => {
    it('returns true when node count meets the threshold', () => {
      expect(shouldAutoEnableGpu(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD)).toBe(
        true,
      );
    });

    it('returns false when node count is below the threshold', () => {
      expect(
        shouldAutoEnableGpu(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1),
      ).toBe(false);
    });

    it('returns true when batch parallel count meets the threshold', () => {
      expect(
        shouldAutoEnableGpu(
          0,
          DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD,
        ),
      ).toBe(true);
    });

    it('returns false when both node count and batch count are below thresholds', () => {
      expect(
        shouldAutoEnableGpu(
          DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1,
          DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD - 1,
        ),
      ).toBe(false);
    });

    it('returns false when disableGPU is true', () => {
      expect(
        shouldAutoEnableGpu(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD * 2, 0, {
          disableGPU: true,
        }),
      ).toBe(false);
    });

    it('honours custom gpuNodeThreshold overrides', () => {
      expect(shouldAutoEnableGpu(500, 0, { gpuNodeThreshold: 400 })).toBe(true);
    });

    it('honours custom gpuBatchParallelThreshold overrides', () => {
      expect(shouldAutoEnableGpu(0, 4, { gpuBatchParallelThreshold: 4 })).toBe(
        true,
      );
    });
  });

  describe('autoEnableGpu', () => {
    it('returns a structured result with enabled, gpuDevice, notified and reason', async () => {
      setNavigator(true);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(result).toMatchObject({
        enabled: expect.any(Boolean),
        gpuDevice: expect.anything(),
        notified: expect.any(Boolean),
        reason: expect.any(String),
      });
    });

    it('returns enabled true and a non-null device when eligible and GPU is available', async () => {
      setNavigator(true);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(result.enabled).toBe(true);
    });

    it('returns a non-null gpuDevice when GPU is auto-enabled', async () => {
      setNavigator(true);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(result.gpuDevice).not.toBeNull();
    });

    it('calls requestGPUDevice when GPU is eligible and available', async () => {
      setNavigator(true);

      await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(requestGPUDevice).toHaveBeenCalled();
    });

    it('returns enabled false when disableGPU is true', async () => {
      setNavigator(true);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        config: { disableGPU: true },
      });

      expect(result.enabled).toBe(false);
    });

    it('does not call requestGPUDevice when disableGPU is true', async () => {
      setNavigator(true);
      jest.mocked(requestGPUDevice).mockClear();

      await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
        config: { disableGPU: true },
      });

      expect(requestGPUDevice).not.toHaveBeenCalled();
    });

    it('returns enabled false when node count is below the threshold', async () => {
      setNavigator(true);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1,
      });

      expect(result.enabled).toBe(false);
    });

    it('sets notified true when GPU is available but the network is below threshold', async () => {
      setNavigator(true);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1,
      });

      expect(result.notified).toBe(true);
    });

    it('returns enabled false and notified false when GPU is unavailable', async () => {
      setNavigator(false);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(result).toMatchObject({
        enabled: false,
        notified: false,
      });
    });

    it('returns the below-threshold fallback when GPU is unavailable', async () => {
      setNavigator(false);

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD - 1,
      });

      expect(result).toEqual({
        enabled: false,
        gpuDevice: null,
        notified: false,
        reason: 'Network below auto-enable threshold and GPU not available',
      });
    });

    it('returns enabled false when the device request fails', async () => {
      setNavigator(true);
      jest
        .mocked(requestGPUDevice)
        .mockRejectedValue(new Error('GPU device request failed'));

      const result = await autoEnableGpu({
        nodeCount: DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
      });

      expect(result.enabled).toBe(false);
    });
  });

  describe('evaluateWeightVariantsOnGpu', () => {
    it('returns per-variant scores in input order', async () => {
      const connection0 = { weight: 0.1 };
      const network = {
        nodes: [],
        connections: [connection0, { weight: 0.2 }],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;
      const variants: WeightVariant[] = [
        { weightIndex: 0, delta: 0 },
        { weightIndex: 1, delta: 0 },
      ];
      const device = {} as GPUDevice;

      const scores = await evaluateWeightVariantsOnGpu(
        network,
        variants,
        [[0.5, 0.5]],
        [1.0],
        DEFAULT_VARIANT_SCORER,
        device,
      );

      scores.forEach((score) => expect(score).toBeCloseTo(-0.16, 5));
    });

    it('restores the original gpuDevice after evaluation', async () => {
      const previousDevice = { label: 'previous' } as unknown as GPUDevice;
      const device = { label: 'current' } as unknown as GPUDevice;
      const network = {
        nodes: [],
        connections: [{ weight: 0.1 }],
        gpuDevice: previousDevice,
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;

      await evaluateWeightVariantsOnGpu(
        network,
        [{ weightIndex: 0, delta: 0.05 }],
        [[0.5, 0.5]],
        [1.0],
        DEFAULT_VARIANT_SCORER,
        device,
      );

      expect(network.gpuDevice).toBe(previousDevice);
    });

    it('restores connection weights after each variant is scored', async () => {
      const connection = { weight: 0.1 };
      const network = {
        nodes: [],
        connections: [connection],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;
      const device = {} as GPUDevice;

      await evaluateWeightVariantsOnGpu(
        network,
        [{ weightIndex: 0, delta: 0.05 }],
        [[0.5, 0.5]],
        [1.0],
        DEFAULT_VARIANT_SCORER,
        device,
      );

      expect(connection.weight).toBe(0.1);
    });

    it('propagates activation errors and still restores the gpuDevice', async () => {
      const previousDevice = { label: 'previous' } as unknown as GPUDevice;
      const device = { label: 'current' } as unknown as GPUDevice;
      const network = {
        nodes: [],
        connections: [{ weight: 0.1 }],
        gpuDevice: previousDevice,
        activate: jest.fn().mockRejectedValue(new Error('GPU activate failed')),
      } as unknown as VariantEvaluationNetwork;

      await expect(
        evaluateWeightVariantsOnGpu(
          network,
          [{ weightIndex: 0, delta: 0.05 }],
          [[0.5, 0.5]],
          [1.0],
          DEFAULT_VARIANT_SCORER,
          device,
        ),
      ).rejects.toThrow('GPU activate failed');

      expect(network.gpuDevice).toBe(previousDevice);
    });

    it('skips missing connections without mutating weights', async () => {
      const connection = { weight: 0.1 };
      const network = {
        nodes: [],
        connections: [connection],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;
      const device = {} as GPUDevice;

      const scores = await evaluateWeightVariantsOnGpu(
        network,
        [{ weightIndex: 5, delta: 0.05 }],
        [[0.5, 0.5]],
        [1.0],
        DEFAULT_VARIANT_SCORER,
        device,
      );

      expect(connection.weight).toBe(0.1);
      scores.forEach((score) => expect(score).toBeCloseTo(-0.16, 5));
    });
  });
});
