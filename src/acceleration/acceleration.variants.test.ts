/**
 * Red tests for the generic async weight-variant evaluator.
 *
 * `evaluateWeightVariantsAsync` lives in `src/acceleration/acceleration.variants.ts`
 * and dispatches variant evaluation to GPU, worker, or CPU backends. This suite
 * defines the expected async contract before the implementation slice lands.
 *
 * Tests import from `./acceleration.variants`, which does not exist yet, so the
 * suite fails with TS2307 until P8S3-05.
 */

import {
  evaluateWeightVariantsAsync,
  DEFAULT_VARIANT_SCORER,
  type WeightVariantResult,
  type VariantEvaluationNetwork,
  type WeightVariant,
} from './acceleration.variants';
import type {
  AccelerationConfig,
  AccelerationStatus,
} from './acceleration.types';
import * as accelerationOrchestrator from './acceleration.orchestrator';
import { DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD } from './acceleration.constants';

/**
 * Build a deterministic mock network surface that satisfies
 * `VariantEvaluationNetwork` without importing from `src/architecture/`.
 */
function buildMockNetwork(): VariantEvaluationNetwork {
  return {
    nodes: [],
    connections: [],
    activate: jest.fn().mockResolvedValue([0.5, 0.5]),
  } as unknown as VariantEvaluationNetwork;
}

/**
 * Build a node list large enough to satisfy the GPU node threshold.
 *
 * The production code only checks `network.nodes.length`, so a typed array
 * of nulls is sufficient for the mock surface.
 */
function gpuSizedNodes(): unknown[] {
  return new Array(DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD).fill(null);
}

/**
 * Build a small list of deterministic weight variants.
 */
function buildMockVariants(count = 4): WeightVariant[] {
  return Array.from({ length: count }, (_, index) => ({
    weightIndex: index,
    delta: 0.05 * (index + 1),
  }));
}

/**
 * Build a config object that requests parallel variant dispatch.
 */
function buildParallelConfig(count: number): AccelerationConfig {
  return { parallelVariantCount: count };
}

/**
 * Build a deterministic acceleration status for a given backend mode.
 *
 * @param mode - Selected backend.
 * @param gpuDevice - Optional fake WebGPU device; when supplied with `mode: 'gpu'`,
 *                    the evaluator follows the real GPU dispatch path instead of
 *                    silently falling back to CPU.
 * @returns A status object compatible with `evaluateWeightVariantsAsync`.
 */
function buildAccelerationStatus(
  mode: 'cpu' | 'gpu' | 'worker',
  gpuDevice: GPUDevice | null = null,
): AccelerationStatus {
  return {
    mode,
    gpu: {
      available: mode === 'gpu',
      reason: mode === 'gpu' ? 'mock gpu available' : 'mock gpu unavailable',
      device: gpuDevice ?? undefined,
    },
    worker: {
      available: mode === 'worker',
      reason:
        mode === 'worker' ? 'mock worker available' : 'mock worker unavailable',
      count: mode === 'worker' ? 2 : 0,
    },
    cpu: { available: true, reason: 'mock cpu fallback' },
  };
}

describe('acceleration.variants', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('evaluateWeightVariantsAsync', () => {
    it('returns a Promise', () => {
      const network = buildMockNetwork();
      const variants = buildMockVariants();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = evaluateWeightVariantsAsync(
        network,
        variants,
        inputs,
        target,
      );

      expect(result).toBeInstanceOf(Promise);
    });

    it('resolves to a result with best index, score, per-variant scores, and backend metadata', async () => {
      const network = buildMockNetwork();
      const variants = buildMockVariants();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result: WeightVariantResult = await evaluateWeightVariantsAsync(
        network,
        variants,
        inputs,
        target,
        DEFAULT_VARIANT_SCORER,
        42,
      );

      expect(result).toMatchObject({
        bestIndex: expect.any(Number),
        bestScore: expect.any(Number),
        scores: expect.any(Array),
        metadata: expect.objectContaining({
          backend: expect.any(String),
          variantCount: expect.any(Number),
          scaleDivisor: expect.any(Number),
          scorer: expect.any(String),
        }),
      });
    });

    it('uses the default scorer when no custom scorer is provided', async () => {
      const network = buildMockNetwork();
      const variants = buildMockVariants();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateWeightVariantsAsync(
        network,
        variants,
        inputs,
        target,
        undefined,
        42,
      );

      expect(result.metadata.scorer).toBe('default');
    });

    it('reports a custom scorer in metadata when one is supplied', async () => {
      const network = buildMockNetwork();
      const variants = buildMockVariants();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];
      const customScorer = jest.fn().mockReturnValue(1);

      const result = await evaluateWeightVariantsAsync(
        network,
        variants,
        inputs,
        target,
        customScorer,
        42,
      );

      expect(result.metadata.scorer).toBe('custom');
    });

    it('propagates a rejected promise from the network activate path', async () => {
      const network = buildMockNetwork();
      network.activate = jest
        .fn()
        .mockRejectedValue(new Error('activate failed'));
      const variants = buildMockVariants();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      await expect(
        evaluateWeightVariantsAsync(network, variants, inputs, target),
      ).rejects.toThrow('activate failed');
    });

    it('returns zero from the default scorer when outputs or target are empty', () => {
      expect(DEFAULT_VARIANT_SCORER([], [1.0])).toBe(0);
      expect(DEFAULT_VARIANT_SCORER([[0.5]], [])).toBe(0);
    });

    it('emits a telemetry event when an observer with onTelemetry is supplied', async () => {
      const network = buildMockNetwork();
      const variants = buildMockVariants();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];
      const onTelemetry = jest.fn();

      await evaluateWeightVariantsAsync(
        network,
        variants,
        inputs,
        target,
        undefined,
        42,
        undefined,
        { onTelemetry },
      );

      expect(onTelemetry).toHaveBeenCalledWith(
        expect.objectContaining({
          backend: expect.any(String),
          inferenceMs: 0,
        }),
      );
    });

    it('returns sensible defaults when no variants are supplied', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateWeightVariantsAsync(
        network,
        [],
        inputs,
        target,
      );

      expect(result).toMatchObject({
        bestIndex: 0,
        bestScore: 0,
        scores: [],
        metadata: expect.objectContaining({ variantCount: 0, scaleDivisor: 1 }),
      });
    });

    it('restores connection weights after evaluating variants with connections', async () => {
      const connection = { weight: 0.1 };
      const network = {
        nodes: [],
        connections: [connection],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;
      const variants: WeightVariant[] = [{ weightIndex: 0, delta: 0.05 }];
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      await evaluateWeightVariantsAsync(network, variants, inputs, target);

      expect(connection.weight).toBe(0.1);
    });

    it('selects a later variant when it scores strictly higher', async () => {
      const network = buildMockNetwork();
      const variants: WeightVariant[] = [
        { weightIndex: 0, delta: 0 },
        { weightIndex: 1, delta: 0 },
      ];
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];
      const customScorer = jest
        .fn()
        .mockReturnValueOnce(0.5)
        .mockReturnValueOnce(0.9);

      const result = await evaluateWeightVariantsAsync(
        network,
        variants,
        inputs,
        target,
        customScorer as jest.Mock,
      );

      expect(result.bestIndex).toBe(1);
    });

    it('keeps the default sequential path when config is omitted', async () => {
      const network = buildMockNetwork();
      const variants = buildMockVariants();

      const result = await evaluateWeightVariantsAsync(
        network,
        variants,
        [[0.5, 0.5]],
        [1.0],
      );

      expect(result.metadata.backend).toBe('cpu');
    });

    it('short-circuits to CPU when backend is explicitly set to cpu', async () => {
      const spy = jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('cpu'));

      const result = await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'cpu' },
      );

      expect(spy).not.toHaveBeenCalled();
      expect(result.metadata.backend).toBe('cpu');
    });

    it('emits onBackendChange with previous null and current cpu when backend is explicitly cpu', async () => {
      const onBackendChange = jest.fn();

      await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'cpu' },
        { onBackendChange },
      );

      expect(onBackendChange).toHaveBeenCalledWith(
        expect.objectContaining({
          previous: null,
          current: 'cpu',
        }),
      );
    });

    it('uses the sequential path when parallelVariantCount is 1', async () => {
      const result = await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { parallelVariantCount: 1 },
      );

      expect(result.scores).toHaveLength(buildMockVariants().length);
      expect(result.metadata.backend).toBe('cpu');
    });

    it('dispatches variants concurrently when parallelVariantCount is greater than 1', async () => {
      const connection0 = { weight: 0.1 };
      const connection1 = { weight: 0.2 };
      let resolveFirst: ((value: number[]) => void) | undefined;
      const firstPromise = new Promise<number[]>((resolve) => {
        resolveFirst = resolve;
      });
      let secondActivateStarted = false;

      const network = {
        nodes: [],
        connections: [connection0, connection1],
        activate: jest
          .fn()
          .mockImplementationOnce(() => firstPromise)
          .mockImplementationOnce(() => {
            secondActivateStarted = true;
            return Promise.resolve([0.6, 0.6]);
          }),
      } as unknown as VariantEvaluationNetwork;

      const variants: WeightVariant[] = [
        { weightIndex: 0, delta: 0.05 },
        { weightIndex: 1, delta: 0.05 },
      ];
      const evaluationPromise = evaluateWeightVariantsAsync(
        network,
        variants,
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        buildParallelConfig(2),
      );

      // Yield to the event loop so the async backend-resolution and
      // concurrent batch dispatch finish before we assert that the second
      // variant's activate call has already started.
      await new Promise<void>((resolve) => setImmediate(resolve));

      expect(secondActivateStarted).toBe(true);

      resolveFirst!([0.6, 0.6]);
      await evaluationPromise;
    });

    it('returns scores in the same order as input variants when parallelVariantCount is greater than 1', async () => {
      const connection0 = { weight: 0.1 };
      const connection1 = { weight: 0.2 };
      const network = {
        nodes: [],
        connections: [connection0, connection1],
        activate: jest
          .fn()
          .mockReturnValueOnce(Promise.resolve([0.4, 0.4]))
          .mockReturnValueOnce(Promise.resolve([0.6, 0.6])),
      } as unknown as VariantEvaluationNetwork;

      const variants: WeightVariant[] = [
        { weightIndex: 0, delta: 0 },
        { weightIndex: 1, delta: 0 },
      ];
      const customScorer = jest
        .fn()
        .mockReturnValueOnce(-0.5)
        .mockReturnValueOnce(-0.1);

      const result = await evaluateWeightVariantsAsync(
        network,
        variants,
        [[0.5, 0.5]],
        [1.0],
        customScorer as jest.Mock,
        42,
        buildParallelConfig(2),
      );

      expect(result.scores).toEqual([-0.5, -0.1]);
    });

    it('restores the first connection weight after a concurrent batch', async () => {
      const connection0 = { weight: 0.1 };
      const connection1 = { weight: 0.2 };
      const network = {
        nodes: [],
        connections: [connection0, connection1],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;

      const variants: WeightVariant[] = [
        { weightIndex: 0, delta: 0.05 },
        { weightIndex: 1, delta: 0.05 },
      ];

      await evaluateWeightVariantsAsync(
        network,
        variants,
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        buildParallelConfig(2),
      );

      expect(connection0.weight).toBe(0.1);
    });

    it('restores the second connection weight after a concurrent batch', async () => {
      const connection0 = { weight: 0.1 };
      const connection1 = { weight: 0.2 };
      const network = {
        nodes: [],
        connections: [connection0, connection1],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;

      const variants: WeightVariant[] = [
        { weightIndex: 0, delta: 0.05 },
        { weightIndex: 1, delta: 0.05 },
      ];

      await evaluateWeightVariantsAsync(
        network,
        variants,
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        buildParallelConfig(2),
      );

      expect(connection1.weight).toBe(0.2);
    });

    it('calls autoEnableAcceleration when config selects GPU', async () => {
      const spy = jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu'));

      await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
      );

      expect(spy).toHaveBeenCalled();
    });

    it('reports GPU backend in metadata when GPU is selected by policy', async () => {
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu'));

      const result = await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
      );

      expect(result.metadata.backend).toBe('gpu');
    });

    it('reports worker backend in metadata when GPU is unavailable and worker is selected', async () => {
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('worker'));

      const result = await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
      );

      expect(result.metadata.backend).toBe('worker');
    });

    it('falls back to the CPU evaluator path when GPU is selected but no device is provided', async () => {
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu'));
      const connection = { weight: 0.1 };
      let activatedWithGPU = false;
      const network = {
        nodes: gpuSizedNodes(),
        connections: [connection],
        activate: jest
          .fn()
          .mockImplementation(
            (input: number[], options?: { useGPU?: boolean }) => {
              activatedWithGPU = options?.useGPU ?? false;
              return Promise.resolve([0.6, 0.6]);
            },
          ),
      } as unknown as VariantEvaluationNetwork;

      const result = await evaluateWeightVariantsAsync(
        network,
        [{ weightIndex: 0, delta: 0.05 }],
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
      );

      expect(activatedWithGPU).toBe(false);
      expect(connection.weight).toBe(0.1);
      expect(result.metadata.backend).toBe('gpu');
    });

    it('dispatches a single variant through the GPU evaluator when a device is present', async () => {
      const device = {} as GPUDevice;
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu', device));
      const connection = { weight: 0.1 };
      let activatedWithGPU = false;
      const network = {
        nodes: gpuSizedNodes(),
        connections: [connection],
        activate: jest
          .fn()
          .mockImplementation(
            (input: number[], options?: { useGPU?: boolean }) => {
              activatedWithGPU = options?.useGPU ?? false;
              return Promise.resolve([0.6, 0.6]);
            },
          ),
      } as unknown as VariantEvaluationNetwork;

      const result = await evaluateWeightVariantsAsync(
        network,
        [{ weightIndex: 0, delta: 0.05 }],
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
      );

      expect(activatedWithGPU).toBe(true);
      expect(connection.weight).toBe(0.1);
      expect(result.metadata.backend).toBe('gpu');
    });

    it('dispatches a single variant through the GPU evaluator in sequential mode when parallelVariantCount is 1', async () => {
      const device = {} as GPUDevice;
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu', device));
      const connection = { weight: 0.1 };
      let activatedWithGPU = false;
      const network = {
        nodes: gpuSizedNodes(),
        connections: [connection],
        activate: jest
          .fn()
          .mockImplementation(
            (input: number[], options?: { useGPU?: boolean }) => {
              activatedWithGPU = options?.useGPU ?? false;
              return Promise.resolve([0.6, 0.6]);
            },
          ),
      } as unknown as VariantEvaluationNetwork;

      const result = await evaluateWeightVariantsAsync(
        network,
        [{ weightIndex: 0, delta: 0.05 }],
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu', parallelVariantCount: 1 },
      );

      expect(activatedWithGPU).toBe(true);
      expect(connection.weight).toBe(0.1);
      expect(result.metadata.backend).toBe('gpu');
    });

    it('dispatches variant batches through the GPU evaluator when parallelVariantCount is greater than 1', async () => {
      const device = {} as GPUDevice;
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu', device));
      let gpuActivateCount = 0;
      const network = {
        nodes: gpuSizedNodes(),
        connections: [{ weight: 0.1 }, { weight: 0.2 }],
        activate: jest.fn().mockImplementation(() => {
          gpuActivateCount += 1;
          return Promise.resolve([0.6, 0.6]);
        }),
      } as unknown as VariantEvaluationNetwork;

      await evaluateWeightVariantsAsync(
        network,
        [
          { weightIndex: 0, delta: 0.05 },
          { weightIndex: 1, delta: 0.05 },
        ],
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu', parallelVariantCount: 2 },
      );

      expect(gpuActivateCount).toBe(2);
    });

    it('calls autoEnableAcceleration when falling back to CPU', async () => {
      const spy = jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('cpu'));

      await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
      );

      expect(spy).toHaveBeenCalled();
    });

    it('emits onBackendChange telemetry when a backend is selected', async () => {
      const onBackendChange = jest.fn();
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu'));

      await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
        { onBackendChange },
      );

      expect(onBackendChange).toHaveBeenCalled();
    });

    it('emits onFallback telemetry when GPU is unavailable and fallback occurs', async () => {
      const onFallback = jest.fn();
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('cpu'));

      await evaluateWeightVariantsAsync(
        buildMockNetwork(),
        buildMockVariants(),
        [[0.5, 0.5]],
        [1.0],
        undefined,
        42,
        { backend: 'gpu' },
        { onFallback },
      );

      expect(onFallback).toHaveBeenCalled();
    });
  });
});
