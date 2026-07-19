/**
 * Tests for the NGE juvenile multi-connection weight-variant evaluator.
 *
 * `src/neat/nge-juvenile/neat.nge-juvenile.variants.ts` builds
 * lifecycle-aware *patches* of weight perturbations, scores each patch as a
 * single local-search step, and reports the best patch's representative
 * perturbation as the winning single-connection variant.
 */

import type {
  AccelerationStatus,
  VariantEvaluationNetwork,
} from '../../acceleration/acceleration.types';
import * as accelerationOrchestrator from '../../acceleration/acceleration.orchestrator';
import { DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD } from '../../acceleration/acceleration.constants';
import {
  NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
} from './neat.nge-juvenile.constants';
import type { NgeLifecycleStage } from './neat.nge-juvenile.types';
import {
  buildVariants,
  evaluateNgeWeightVariants,
  resolveVariantCountForStage,
  resolveRepresentativeDelta,
  resolveEffectiveMagnitude,
  pickDistinctIndicesExcluding,
  type NgeWeightVariantPatch,
} from './neat.nge-juvenile.variants';

/**
 * Build a deterministic mock network surface without importing
 * `src/architecture/network`.
 */
function buildMockNetwork(connectionCount = 0) {
  return {
    nodes: [],
    connections: Array.from({ length: connectionCount }, (_, index) => ({
      weight: 0.01 * (index + 1),
    })),
    activate: jest.fn().mockResolvedValue([0.5, 0.5]),
  };
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
 * Build a deterministic acceleration status for a given backend mode.
 *
 * @param mode - Selected backend.
 * @param gpuDevice - Optional fake WebGPU device; when supplied with `mode: 'gpu'`,
 *                    the evaluator follows the GPU path.
 * @returns A status object compatible with `autoEnableAcceleration` mocks.
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

function extractPerturbationIndices(patch: NgeWeightVariantPatch): number[] {
  return patch.perturbations.map(({ weightIndex }) => weightIndex);
}

describe('nge-juvenile.variants', () => {
  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('evaluateNgeWeightVariants', () => {
    it('returns a Promise', () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = evaluateNgeWeightVariants(
        network,
        'juvenile',
        inputs,
        target,
        42,
      );

      expect(result).toBeInstanceOf(Promise);
    });

    it('resolves to a result with backend metadata', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        inputs,
        target,
        42,
      );

      expect(result.metadata).toMatchObject({
        backend: expect.any(String),
        variantCount: expect.any(Number),
        scorer: 'default',
        scaleDivisor: expect.any(Number),
      });
    });

    it('selects a higher variant count for the baby stage', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const babyResult = await evaluateNgeWeightVariants(
        network,
        'baby',
        inputs,
        target,
        42,
      );
      const adultResult = await evaluateNgeWeightVariants(
        network,
        'adult',
        inputs,
        target,
        42,
      );

      expect(babyResult.metadata.variantCount).toBeGreaterThanOrEqual(
        adultResult.metadata.variantCount,
      );
    });

    it('maps each lifecycle stage to its configured default variant count', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const embryoResult = await evaluateNgeWeightVariants(
        network,
        'embryo',
        inputs,
        target,
        42,
      );
      const juvenileResult = await evaluateNgeWeightVariants(
        network,
        'juvenile',
        inputs,
        target,
        42,
      );
      const equilibriumResult = await evaluateNgeWeightVariants(
        network,
        'equilibrium',
        inputs,
        target,
        42,
      );

      expect(embryoResult.metadata.variantCount).toBeGreaterThan(
        juvenileResult.metadata.variantCount,
      );
      expect(juvenileResult.metadata.variantCount).toBeGreaterThan(
        equilibriumResult.metadata.variantCount,
      );
    });

    it('propagates async evaluator failures as rejected promises', async () => {
      const network = {
        nodes: [],
        connections: [],
        activate: jest.fn().mockRejectedValue(new Error('nge variants failed')),
      };
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      await expect(
        evaluateNgeWeightVariants(network, 'baby', inputs, target, 42),
      ).rejects.toThrow('nge variants failed');
    });

    it('uses an overridden baby-stage variant count when supplied', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        inputs,
        target,
        42,
        {
          stageVariantCounts: { baby: 3 },
        },
      );

      expect(result.metadata.variantCount).toBe(3);
    });

    it('honors accelerationConfig.stageVariantCounts over defaults', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateNgeWeightVariants(
        network,
        'juvenile',
        inputs,
        target,
        42,
        {
          accelerationConfig: {
            stageVariantCounts: { juvenile: 7 },
          },
        },
      );

      expect(result.metadata.variantCount).toBe(7);
    });

    it('uses the baby stage count for the embryo stage via accelerationConfig', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateNgeWeightVariants(
        network,
        'embryo',
        inputs,
        target,
        42,
        {
          accelerationConfig: {
            stageVariantCounts: { baby: 5 },
          },
        },
      );

      expect(result.metadata.variantCount).toBe(5);
    });

    it('uses the adult stage count for the equilibrium stage via accelerationConfig', async () => {
      const network = buildMockNetwork();
      const inputs: number[][] = [[0.5, 0.5]];
      const target: number[] = [1.0];

      const result = await evaluateNgeWeightVariants(
        network,
        'equilibrium',
        inputs,
        target,
        42,
        {
          accelerationConfig: {
            stageVariantCounts: { adult: 6 },
          },
        },
      );

      expect(result.metadata.variantCount).toBe(6);
    });

    it('binds a GPU device during evaluation and restores the previous device afterward', async () => {
      const device = {} as GPUDevice;
      const previousDevice = {} as GPUDevice;
      let deviceDuringEvaluation: GPUDevice | null | undefined = undefined;

      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu', device));

      const network = {
        nodes: gpuSizedNodes(),
        connections: [{ weight: 0.1 }],
        gpuDevice: previousDevice,
        activate: jest.fn().mockImplementation(() => {
          deviceDuringEvaluation = network.gpuDevice;
          return Promise.resolve([0.6, 0.6]);
        }),
      } as unknown as VariantEvaluationNetwork;

      await evaluateNgeWeightVariants(network, 'baby', [[0.5, 0.5]], [1.0], 42);

      expect(deviceDuringEvaluation).toBe(device);
      expect(network.gpuDevice).toBe(previousDevice);
    });

    it('passes useGPU: true to network.activate when a GPU device is selected', async () => {
      const device = {} as GPUDevice;

      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu', device));

      const network = {
        nodes: gpuSizedNodes(),
        connections: [{ weight: 0.1 }],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;

      await evaluateNgeWeightVariants(network, 'baby', [[0.5, 0.5]], [1.0], 42);

      expect(network.activate).toHaveBeenCalledWith(
        [0.5, 0.5],
        expect.objectContaining({ useGPU: true }),
      );
    });

    it('passes useGPU: false to network.activate when no GPU device is selected', async () => {
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('cpu'));

      const network = {
        nodes: [],
        connections: [{ weight: 0.1 }],
        activate: jest.fn().mockResolvedValue([0.6, 0.6]),
      } as unknown as VariantEvaluationNetwork;

      await evaluateNgeWeightVariants(network, 'baby', [[0.5, 0.5]], [1.0], 42);

      expect(network.activate).toHaveBeenCalledWith(
        [0.5, 0.5],
        expect.objectContaining({ useGPU: false }),
      );
    });

    it('restores the previous GPU device even when activation throws', async () => {
      const device = {} as GPUDevice;
      const previousDevice = {} as GPUDevice;

      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('gpu', device));

      const network = {
        nodes: gpuSizedNodes(),
        connections: [{ weight: 0.1 }],
        gpuDevice: previousDevice,
        activate: jest
          .fn()
          .mockRejectedValue(new Error('gpu activation failed')),
      } as unknown as VariantEvaluationNetwork;

      await expect(
        evaluateNgeWeightVariants(network, 'baby', [[0.5, 0.5]], [1.0], 42),
      ).rejects.toThrow('gpu activation failed');

      expect(network.gpuDevice).toBe(previousDevice);
    });

    it('evaluates each CPU patch sequentially on the live network and restores weights', async () => {
      // Force CPU backend so the sequential live-network path is exercised.
      jest
        .spyOn(accelerationOrchestrator, 'autoEnableAcceleration')
        .mockResolvedValue(buildAccelerationStatus('cpu'));

      // Two-connection network whose output is the sum of connection weights.
      // Sequential apply/activate/undo means each patch is scored with only
      // its own perturbation applied, and the original weights are restored
      // before the next patch begins.
      const network: VariantEvaluationNetwork = {
        nodes: [],
        connections: [{ weight: 0.1 }, { weight: 0.2 }],
        activate: jest.fn().mockImplementation(function (this: typeof network) {
          const sum = (this.connections as { weight: number }[]).reduce(
            (acc, conn) => acc + conn.weight,
            0,
          );
          return Promise.resolve([sum]);
        }),
      };

      const result = await evaluateNgeWeightVariants(
        network,
        'equilibrium',
        [[0, 0]],
        [0.4],
        42,
      );

      // Adult/equilibrium default variant count is 2. Patch 0 lowers
      // connection 0 by the adult magnitude (0.05); patch 1 raises connection 1
      // by the same amount. Sequential scores are negative MSE values.
      expect(result.metadata.variantCount).toBe(
        NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
      );
      expect(result.scores[0]).toBeCloseTo(-0.0225, 6);
      expect(result.scores[1]).toBeCloseTo(-0.0025, 6);
      expect(result.scores[0]).not.toBeCloseTo(result.scores[1], 6);

      // The live network must be returned to its original state after scoring.
      expect(
        (network.connections as { weight: number }[]).map((c) => c.weight),
      ).toEqual([0.1, 0.2]);
    });
  });

  describe('resolveVariantCountForStage', () => {
    it('falls back to lifecycle defaults when no overrides are given', () => {
      expect(resolveVariantCountForStage('baby')).toBeGreaterThan(
        resolveVariantCountForStage('juvenile'),
      );
      expect(resolveVariantCountForStage('juvenile')).toBeGreaterThan(
        resolveVariantCountForStage('adult'),
      );
    });

    it('prefers explicit overrides over accelerationConfig and defaults', () => {
      expect(
        resolveVariantCountForStage(
          'baby',
          { baby: 2 },
          { stageVariantCounts: { baby: 99 } },
        ),
      ).toBe(2);
    });

    it('honors accelerationConfig.stageVariantCounts when no explicit override is supplied', () => {
      expect(
        resolveVariantCountForStage(
          'adult',
          {},
          { stageVariantCounts: { adult: 12 } },
        ),
      ).toBe(12);
    });

    it('clamps config counts to at least one', () => {
      expect(resolveVariantCountForStage('baby', { baby: 0 })).toBe(1);
    });
  });

  describe('resolveRepresentativeDelta', () => {
    it('N=16 spans [-magnitude, +magnitude] endpoint-inclusive', () => {
      const deltas = Array.from({ length: 16 }, (_, index) =>
        resolveRepresentativeDelta(index, 16, 0.15),
      );

      expect(deltas[0]).toBe(-0.15);
    });

    it('N=16 ends at +magnitude', () => {
      const deltas = Array.from({ length: 16 }, (_, index) =>
        resolveRepresentativeDelta(index, 16, 0.15),
      );

      expect(deltas[15]).toBe(0.15);
    });

    it('N=16 produces 16 unique deltas', () => {
      const deltas = Array.from({ length: 16 }, (_, index) =>
        resolveRepresentativeDelta(index, 16, 0.15),
      );

      expect(new Set(deltas).size).toBe(16);
    });

    it('N=16 is symmetric around zero', () => {
      const deltas = Array.from({ length: 16 }, (_, index) =>
        resolveRepresentativeDelta(index, 16, 0.15),
      );

      expect(deltas).toEqual(deltas.map((delta) => -delta).reverse());
    });

    it('N=1024 starts at -magnitude', () => {
      const deltas = Array.from({ length: 1024 }, (_, index) =>
        resolveRepresentativeDelta(index, 1024, 0.3),
      );

      expect(deltas[0]).toBe(-0.3);
    });

    it('N=1024 ends at +magnitude', () => {
      const deltas = Array.from({ length: 1024 }, (_, index) =>
        resolveRepresentativeDelta(index, 1024, 0.3),
      );

      expect(deltas[1023]).toBe(0.3);
    });

    it('N=1024 produces 1024 unique deltas', () => {
      const deltas = Array.from({ length: 1024 }, (_, index) =>
        resolveRepresentativeDelta(index, 1024, 0.3),
      );

      expect(new Set(deltas).size).toBe(1024);
    });

    it('N=1 returns +magnitude', () => {
      expect(resolveRepresentativeDelta(0, 1, 0.15)).toBe(0.15);
    });
  });

  describe('buildVariants', () => {
    it('exports a deterministic patch builder', () => {
      const network = buildMockNetwork(4);
      const patches = buildVariants(network, 'baby', 3, 42);

      expect(patches).toHaveLength(3);
      expect(patches[0]?.representative).toEqual(
        expect.objectContaining({
          weightIndex: expect.any(Number),
          delta: expect.any(Number),
        }),
      );
    });

    it('produces the same patches for the same seed', () => {
      const network = buildMockNetwork(40);
      const first = buildVariants(network, 'baby', 4, 123);
      const second = buildVariants(network, 'baby', 4, 123);

      for (let index = 0; index < first.length; index++) {
        expect(extractPerturbationIndices(first[index]!)).toEqual(
          extractPerturbationIndices(second[index]!),
        );
      }
    });

    it('produces different patches for different seeds', () => {
      const network = buildMockNetwork(100);
      const first = buildVariants(network, 'baby', 2, 1);
      const second = buildVariants(network, 'baby', 2, 2);

      const firstFlat = first.flatMap(extractPerturbationIndices);
      const secondFlat = second.flatMap(extractPerturbationIndices);
      expect(firstFlat).not.toEqual(secondFlat);
    });

    it('perturbs multiple connections per patch when the network is large enough', () => {
      const network = buildMockNetwork(100);
      const patches = buildVariants(network, 'baby', 4, 7);

      expect(patches.length).toBe(4);
      for (const patch of patches) {
        expect(patch.perturbations.length).toBeGreaterThanOrEqual(2);
      }
    });

    it('cycles the representative index over the connection list', () => {
      const network = buildMockNetwork(3);
      const patches = buildVariants(network, 'baby', 5, 0);

      const representatives = patches.map(
        (patch) => patch.representative.weightIndex,
      );
      expect(representatives).toEqual([0, 1, 2, 0, 1]);
    });

    describe('buildVariants representative deltas', () => {
      it('first representative delta is negative for count > 1', () => {
        const network = buildMockNetwork(2);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(patches[0]!.representative.delta).toBeLessThan(0);
      });

      it('last representative delta is +magnitude', () => {
        const network = buildMockNetwork(2);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(patches[15]!.representative.delta).toBe(0.15);
      });

      it('all representative deltas are unique', () => {
        const network = buildMockNetwork(2);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(
          new Set(patches.map((patch) => patch.representative.delta)).size,
        ).toBe(16);
      });

      it('representative deltas are symmetric', () => {
        const network = buildMockNetwork(2);
        const patches = buildVariants(network, 'baby', 16, 0);
        const deltas = patches.map((patch) => patch.representative.delta);

        expect(deltas).toEqual(deltas.map((delta) => -delta).reverse());
      });
    });

    it('bounds non-representative deltas by the stage mutation magnitude', () => {
      const network = buildMockNetwork(100);
      const patches = buildVariants(network, 'juvenile', 8, 0);

      for (const patch of patches) {
        for (const perturbation of patch.perturbations) {
          expect(Math.abs(perturbation.delta)).toBeLessThanOrEqual(0.11);
        }
      }
    });

    describe('networks with no connections', () => {
      it('returns 16 deterministic patches for count 16', () => {
        const network = buildMockNetwork(0);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(patches).toHaveLength(16);
      });

      it('starts the representative delta ladder at -magnitude', () => {
        const network = buildMockNetwork(0);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(patches[0]!.representative.delta).toBe(-0.15);
      });

      it('ends the representative delta ladder at +magnitude', () => {
        const network = buildMockNetwork(0);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(patches[15]!.representative.delta).toBe(0.15);
      });

      it('produces unique representative deltas for count 16', () => {
        const network = buildMockNetwork(0);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(
          new Set(patches.map((patch) => patch.representative.delta)).size,
        ).toBe(16);
      });

      it('assigns weight index 0 to every representative', () => {
        const network = buildMockNetwork(0);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(
          patches.every((patch) => patch.representative.weightIndex === 0),
        ).toBe(true);
      });

      it('returns only the representative perturbation when there are no connections', () => {
        const network = buildMockNetwork(0);
        const patches = buildVariants(network, 'baby', 16, 0);

        expect(patches.every((patch) => patch.perturbations.length === 1)).toBe(
          true,
        );
      });
    });
  });

  describe('pickDistinctIndicesExcluding', () => {
    it('retries when rejection sampling produces a duplicate index', () => {
      const rand = jest
        .fn()
        .mockReturnValueOnce(0.0) // index 0 — duplicate of exclude
        .mockReturnValueOnce(0.0) // index 0 — duplicate again
        .mockReturnValueOnce(0.9); // index 1 — distinct

      const result = pickDistinctIndicesExcluding(0, 1, 2, rand);

      expect(result).toEqual([1]);
      expect(rand).toHaveBeenCalledTimes(3);
    });
  });

  describe('variant patch constants', () => {
    it('no longer exports NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA', async () => {
      const constants =
        (await import('./neat.nge-juvenile.constants')) as unknown as Record<
          string,
          unknown
        >;
      expect(constants).not.toHaveProperty(
        'NGE_VARIANT_PATCH_REPRESENTATIVE_DELTA',
      );
    });

    it('exports NGE_VARIANT_WEIGHT_RANGE equal to 2', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(constants.NGE_VARIANT_WEIGHT_RANGE).toBe(2);
    });

    it('exports NGE_VARIANT_WIDTH_FACTOR_MAX equal to 1.5', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(constants.NGE_VARIANT_WIDTH_FACTOR_MAX).toBe(1.5);
    });

    it('exports NGE_VARIANT_SIZE_FACTOR_FLOOR equal to 0.1', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(constants.NGE_VARIANT_SIZE_FACTOR_FLOOR).toBe(0.1);
    });

    it('exports NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS equal to 400', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(constants.NGE_VARIANT_SIZE_FACTOR_REF_CONNECTIONS).toBe(400);
    });

    it('exports NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR equal to 4', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(constants.NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR).toBe(4);
    });
  });

  describe('resolveEffectiveMagnitude', () => {
    it('default baby fixed point (count=16, connections=5) stays at 0.15', () => {
      expect(resolveEffectiveMagnitude('baby', 16, 5)).toBe(0.15);
    });

    it('1024 override widens magnitude logarithmically for tiny networks (count=1024, connections=5)', () => {
      expect(resolveEffectiveMagnitude('baby', 1024, 5)).toBe(0.225);
    });

    it('large networks shrink magnitude (count=16, connections=500)', () => {
      expect(resolveEffectiveMagnitude('baby', 16, 500)).toBeCloseTo(0.134, 3);
    });

    it('result depends on connectionCount, not nodeCount', () => {
      expect(resolveEffectiveMagnitude('baby', 16, 5)).not.toBe(
        resolveEffectiveMagnitude('baby', 16, 500),
      );
    });
  });

  describe('lifecycle stage fallback and edge cases', () => {
    it('resolveVariantCountForStage returns the adult default for unknown stages', () => {
      const unknownStage = 'unknown-stage' as NgeLifecycleStage;
      expect(resolveVariantCountForStage(unknownStage)).toBe(
        NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
      );
    });

    it('evaluateNgeWeightVariants falls back to adult defaults for unknown stages', async () => {
      const network = buildMockNetwork(2);
      const unknownStage = 'unknown-stage' as NgeLifecycleStage;

      const result = await evaluateNgeWeightVariants(
        network,
        unknownStage,
        [[0.5, 0.5]],
        [1.0],
      );

      expect(result.metadata.variantCount).toBe(
        NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
      );
      expect(result.scores).toHaveLength(
        NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
      );
    });

    it('computes scaleDivisor from max representative delta', async () => {
      const network = buildMockNetwork(0);

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5, 0.5]],
        [1.0],
      );

      expect(result.metadata.variantCount).toBe(
        NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
      );
      expect(result.metadata.scaleDivisor).toBe(
        NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
      );
    });
  });

  describe('evaluateNgeWeightVariants branch coverage', () => {
    it('falls back to scaleDivisor 1 and negative-infinity bestScore when variant count resolves to NaN', async () => {
      const network = buildMockNetwork();

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5, 0.5]],
        [1.0],
        undefined,
        {
          stageVariantCounts: { baby: Number.NaN },
        },
      );

      expect(result.bestIndex).toBe(0);
      expect(result.bestScore).toBe(Number.NEGATIVE_INFINITY);
      expect(result.scores).toEqual([]);
      expect(result.metadata.variantCount).toBe(0);
      expect(result.metadata.scaleDivisor).toBe(1);
    });

    it('updates bestIndex and bestScore when a later patch scores higher', async () => {
      const connections = [{ weight: 0.1 }, { weight: 0.2 }];
      const network: VariantEvaluationNetwork = {
        nodes: [],
        connections,
        activate: async () => [
          connections.reduce((sum, connection) => sum + connection.weight, 0),
        ],
      };

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5]],
        [1.0],
        undefined,
        {
          accelerationConfig: { backend: 'cpu', parallelVariantCount: 1 },
        },
      );

      expect({
        bestIndex: result.bestIndex,
        bestScoreEqualsMax: result.bestScore === Math.max(...result.scores),
      }).toEqual({
        bestIndex: result.scores.indexOf(Math.max(...result.scores)),
        bestScoreEqualsMax: true,
      });
    });

    it('tolerates a connection disappearing between patch apply and restore', async () => {
      const connections = [{ weight: 0.5 }];
      const network: VariantEvaluationNetwork = {
        nodes: [],
        connections,
        activate: jest.fn().mockImplementation(() => {
          connections.length = 0;
          return [0.5, 0.5];
        }),
      };

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5, 0.5]],
        [1.0],
      );

      expect(result.scores).toHaveLength(
        NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
      );
    });

    it('reports custom scorer in metadata when a scoreFn is supplied', async () => {
      const network = buildMockNetwork(2);
      network.activate = jest.fn().mockResolvedValue([0.6, 0.6]);

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5, 0.5]],
        [1.0],
        42,
        {
          accelerationConfig: { backend: 'cpu' },
          scoreFn: (outputs: readonly number[][], target: readonly number[]) =>
            -Math.abs(outputs[0]![0]! - target[0]!),
        },
      );

      expect(result.metadata.scorer).toBe('custom');
    });

    it('uses the supplied custom scorer instead of the default scorer', async () => {
      const network = buildMockNetwork(2);
      network.activate = jest.fn().mockResolvedValue([0.6, 0.6]);

      const result = await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5, 0.5]],
        [1.0],
        42,
        {
          accelerationConfig: { backend: 'cpu' },
          scoreFn: (outputs: readonly number[][], target: readonly number[]) =>
            -Math.abs(outputs[0]![0]! - target[0]!),
        },
      );

      expect(result.bestScore).toBeCloseTo(-0.4, 6);
    });

    it('invokes the custom scorer for every evaluated patch', async () => {
      const customScorer = jest.fn().mockReturnValue(-1.5);
      const network = buildMockNetwork(2);
      network.activate = jest.fn().mockResolvedValue([0.6, 0.6]);

      await evaluateNgeWeightVariants(
        network,
        'baby',
        [[0.5, 0.5]],
        [1.0],
        42,
        {
          accelerationConfig: { backend: 'cpu' },
          scoreFn: customScorer,
        },
      );

      expect(customScorer).toHaveBeenCalledTimes(
        NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
      );
    });
  });
});
