/**
 * Red-phase test contracts for the NGE grow-stabilize Phase 5 Slice 5A refactor.
 *
 * These tests define the expected contract for the dynamic delta distribution and
 * adaptive weight-exhaustion gate (`src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`)
 * before the implementation exists. They cover:
 *
 * - `resolveAdaptiveHysteresis` thresholds (2/3/5 for node counts ≤200/≤500/>500)
 * - `isPlateauReached` guards (first-growth bypass, MIN/MAX stabilization ticks)
 * - `applyWeightMutations` rate with deterministic random
 * - `runNgeGrowStabilizeCycle` function existence and first-growth commit
 * - `buildWeightVariants(network, variantCount, stage)` parity with `buildVariants`
 * - `resolveExhaustionImprovementThreshold` adaptive improvement threshold
 * - `resolveExhaustionForceGrowthThreshold` adaptive exhaustion count
 * - `resolveStageFraction` and `resolveNoiseSigmaFraction` stage mappings
 *
 * Tests fail because the required exports and constants do not exist yet.
 * The failure reason is "missing implementation" (import error), not syntax error
 * or bad fixture.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */
import Network from '../../architecture/network';
import {
  applyWeightMutations,
  buildWeightVariants,
  computeGrowthThrottle,
  isPlateauReached,
  resolveAdaptiveHysteresis,
  resolveExhaustionForceGrowthThreshold,
  resolveExhaustionImprovementThreshold,
  resolveNoiseSigmaFraction,
  resolveStageFraction,
  runNgeGrowStabilizeCycle,
} from './neat.nge-juvenile.grow-stabilize';
import type {
  NgeGrowStabilizeInput,
  NgeHysteresisState,
} from './neat.nge-juvenile.types';
import {
  resolveRepresentativeDelta,
  resolveEffectiveMagnitude,
} from './neat.nge-juvenile.variants';

describe('NGE grow-stabilize cycle', () => {
  describe('resolveAdaptiveHysteresis', () => {
    it('returns 2 for node count ≤ 200 and 3 for 200 < count ≤ 500', () => {
      // Arrange — small and medium network node counts
      const smallCount = 100;
      const mediumCount = 300;

      // Act
      const smallThreshold = resolveAdaptiveHysteresis(smallCount);
      const mediumThreshold = resolveAdaptiveHysteresis(mediumCount);

      // Assert
      expect([smallThreshold, mediumThreshold]).toEqual([2, 3]);
    });

    it('returns 5 for node count > 500', () => {
      // Arrange — large network node count
      const largeCount = 600;

      // Act
      const threshold = resolveAdaptiveHysteresis(largeCount);

      // Assert
      expect(threshold).toBe(5);
    });
  });

  describe('isPlateauReached', () => {
    it('returns true when hasGrownBefore is false (first-growth bypass)', () => {
      // Arrange — first growth, no prior structural growth
      const scoreWindow = [1, 2, 3, 4, 5];
      const hasGrownBefore = false;
      const stabilizationTicksSinceGrowth = 0;

      // Act
      const result = isPlateauReached(
        scoreWindow,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      // Assert
      expect(result).toBe(true);
    });

    it('returns false when stabilizationTicksSinceGrowth < 5 (MIN guard)', () => {
      // Arrange — has grown before, still within minimum stabilization window
      const scoreWindow = [1, 2, 3, 4, 5];
      const hasGrownBefore = true;
      const stabilizationTicksSinceGrowth = 3; // < MIN_STABILIZATION_TICKS (5)

      // Act
      const result = isPlateauReached(
        scoreWindow,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      // Assert
      expect(result).toBe(false);
    });

    it('returns true when stabilizationTicksSinceGrowth ≥ 25 (MAX guard)', () => {
      // Arrange — has grown before, exceeded the max stabilization cap
      const scoreWindow = [1, 2, 3, 4, 5];
      const hasGrownBefore = true;
      const stabilizationTicksSinceGrowth = 25; // >= MAX_STABILIZATION_TICKS (25)

      // Act
      const result = isPlateauReached(
        scoreWindow,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      // Assert
      expect(result).toBe(true);
    });
  });

  describe('applyWeightMutations', () => {
    it('mutates all connections when random always returns below the rate threshold', () => {
      // Arrange — deterministic network and random that always returns 0.1
      // (below WEIGHT_MUTATION_RATE of 0.3), so every connection is selected.
      const network = new Network(4, 2, { seed: 42 });
      const deterministicRandom = () => 0.1;

      // Act
      const mutatedCount = applyWeightMutations(network, deterministicRandom);

      // Assert — all connections should be mutated
      expect(mutatedCount).toBe(network.connections.length);
    });
  });

  describe('runNgeGrowStabilizeCycle', () => {
    it('is exported as a function', () => {
      // Arrange — (imported at top of file)

      // Act
      const fnType = typeof runNgeGrowStabilizeCycle;

      // Assert
      expect(fnType).toBe('function');
    });

    it('performs first-growth unconditional commit when hasGrownBefore is false', async () => {
      // Arrange — seed network, first growth, positive score history
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: false,
        stabilizationTicksSinceGrowth: 0,
      });

      // Assert — first growth should always commit
      expect(result.committed).toBe(true);
    });
  });

  describe('buildWeightVariants', () => {
    it('is exported from grow-stabilize', () => {
      expect(typeof buildWeightVariants).toBe('function');
    });

    it('accepts a stage parameter', () => {
      const network = new Network(1, 1, { seed: 42 });
      const variants = buildWeightVariants(network, 16, 'baby');
      expect(variants).toBeDefined();
    });

    it('uses resolveEffectiveMagnitude and resolveRepresentativeDelta for baby', () => {
      const network = new Network(1, 1, { seed: 42 });
      const variants = buildWeightVariants(network, 16, 'baby');
      const magnitude = resolveEffectiveMagnitude(
        'baby',
        16,
        network.connections.length,
      );

      expect(variants[0]!.delta).toBe(
        resolveRepresentativeDelta(0, 16, magnitude),
      );
    });
  });

  describe('weight exhaustion gate', () => {
    it('increments consecutiveWeightExhaustion when no improvement exceeds baseline + threshold', async () => {
      const network = new Network(4, 2, { seed: 42 });
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        lifecycleStage: 'baby',
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
        baselineScore: 1_000,
      } as unknown as NgeGrowStabilizeInput);
      expect((result as any).consecutiveWeightExhaustion).toBe(1);
    });

    it('resets consecutiveWeightExhaustion to 0 when a variant improvement commits', async () => {
      const network = new Network(4, 2, { seed: 42 });
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        lifecycleStage: 'baby',
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
        baselineScore: -1_000,
      } as unknown as NgeGrowStabilizeInput);
      expect((result as any).consecutiveWeightExhaustion).toBe(0);
    });

    it('forces growth when consecutiveWeightExhaustion reaches the threshold', async () => {
      const network = new Network(4, 2, { seed: 42 });
      const runner = jest.fn().mockReturnValue({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 2,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none',
          cooldownWindowsRemaining: 0,
        },
      } as any);
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        lifecycleStage: 'baby',
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
        consecutiveWeightExhaustion: 3,
        lifecycleRunner: runner,
      } as unknown as NgeGrowStabilizeInput);
      expect(result.phase).toBe('growth');
    });

    it('resets the exhaustion counter when forced growth fires', async () => {
      const network = new Network(4, 2, { seed: 42 });
      const runner = jest.fn().mockReturnValue({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 2,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none',
          cooldownWindowsRemaining: 0,
        },
      } as any);
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        lifecycleStage: 'baby',
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
        consecutiveWeightExhaustion: 3,
        lifecycleRunner: runner,
      } as unknown as NgeGrowStabilizeInput);
      expect((result as any).consecutiveWeightExhaustion).toBe(0);
    });

    it('top-of-cycle exhaustion override fires before the stabilization branch is evaluated', async () => {
      const network = new Network(4, 2, { seed: 42 });
      network.activate = jest.fn().mockResolvedValue([0.5, 0.5]);
      const runner = jest.fn().mockReturnValue({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 2,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none',
          cooldownWindowsRemaining: 0,
        },
      } as any);
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        lifecycleStage: 'baby',
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
        consecutiveWeightExhaustion: 3,
        lifecycleRunner: runner,
      } as unknown as NgeGrowStabilizeInput);
      expect(result.phase).toBe('growth');
    });
  });

  // --- Coverage tests for uncovered branches ---

  describe('isPlateauReached additional branches', () => {
    it('returns false when scoreWindow is shorter than the plateau window size', () => {
      // Arrange — hasGrownBefore=true, ticks between MIN and MAX, short window
      const scoreWindow = [1, 2, 3]; // length 3 < PLATEAU_WINDOW_SIZE (5)
      const hasGrownBefore = true;
      const stabilizationTicksSinceGrowth = 10; // between MIN (5) and MAX (25)

      // Act
      const result = isPlateauReached(
        scoreWindow,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      // Assert
      expect(result).toBe(false);
    });

    it('returns true when variance falls below the threshold', () => {
      // Arrange — full window with identical scores → variance = 0
      const scoreWindow = [0.5, 0.5, 0.5, 0.5, 0.5];
      const hasGrownBefore = true;
      const stabilizationTicksSinceGrowth = 10;

      // Act
      const result = isPlateauReached(
        scoreWindow,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      // Assert
      expect(result).toBe(true);
    });

    it('returns false when variance is at or above the threshold', () => {
      // Arrange — full window with high variance scores
      const scoreWindow = [0, 1, 0, 1, 0]; // variance ≈ 0.24 > 0.1
      const hasGrownBefore = true;
      const stabilizationTicksSinceGrowth = 10;

      // Act
      const result = isPlateauReached(
        scoreWindow,
        hasGrownBefore,
        stabilizationTicksSinceGrowth,
      );

      // Assert
      expect(result).toBe(false);
    });
  });

  describe('applyWeightMutations no-op branch', () => {
    it('mutates zero connections when random always returns above the rate threshold', () => {
      // Arrange — random returns 0.99 (>= WEIGHT_MUTATION_RATE of 0.3)
      const network = new Network(4, 2, { seed: 42 });
      const highRandom = () => 0.99;

      // Act
      const mutatedCount = applyWeightMutations(network, highRandom);

      // Assert
      expect(mutatedCount).toBe(0);
    });
  });

  describe('computeGrowthThrottle', () => {
    it('returns shouldThrottle false for small networks', () => {
      // Arrange — small network under the large-network threshold
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const { shouldThrottle, interval } = computeGrowthThrottle(network, 1);

      // Assert
      expect({ shouldThrottle, interval }).toEqual({
        shouldThrottle: false,
        interval: 1,
      });
    });

    it('returns shouldThrottle true for large networks when tick is not on interval', () => {
      // Arrange — mock network with 1001 nodes (above threshold of 1000)
      const largeNetwork = {
        nodes: new Array(1001),
        connections: [],
      } as unknown as Network;

      // Act — sizeBudget = ceil(1001/1000) = 2, interval = 3 * 2 = 6, tick = 1
      const { shouldThrottle, interval } = computeGrowthThrottle(
        largeNetwork,
        1,
      );

      // Assert
      expect({ shouldThrottle, interval }).toEqual({
        shouldThrottle: true,
        interval: 6,
      });
    });

    it('returns shouldThrottle false for large networks when tick lands on interval', () => {
      // Arrange — mock network with 1001 nodes, interval = 6, tick = 6
      const largeNetwork = {
        nodes: new Array(1001),
        connections: [],
      } as unknown as Network;

      // Act
      const { shouldThrottle, interval } = computeGrowthThrottle(
        largeNetwork,
        6,
      );

      // Assert — 6 is a multiple of the computed interval
      expect({ shouldThrottle, interval }).toEqual({
        shouldThrottle: false,
        interval: 6,
      });
    });
  });

  describe('runNgeGrowStabilizeCycle additional branches', () => {
    it('uses adaptive hysteresis for very large networks during growth', async () => {
      // Arrange — mock network with 1200 nodes (above the 500-node tier)
      const largeNetwork = {
        nodes: new Array(1200),
        connections: new Array(10),
      } as unknown as Network;
      let capturedHysteresisWindowCount = 0;
      const capturingRunner: NgeGrowStabilizeInput['lifecycleRunner'] = (
        input,
      ) => {
        capturedHysteresisWindowCount = input.config.hysteresisWindowCount ?? 0;
        return {
          stage: 'juvenile' as const,
          applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' as const }],
          hysteresis: {
            growthPositiveWindowCount: 0,
            pruneUnderuseWindowCount: 0,
            lastMorphKind: 'none' as const,
            cooldownWindowsRemaining: 0,
          },
        };
      };

      // Act — force growth by hitting the max stabilization tick cap
      const result = await runNgeGrowStabilizeCycle({
        network: largeNetwork,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        lifecycleRunner: capturingRunner,
      });

      // Assert — resolveAdaptiveHysteresis(1200) returns the largest window
      expect(capturedHysteresisWindowCount).toBe(5);
      expect(result.committed).toBe(true);
    });

    it('enters stabilization phase with weight mutations when plateau is not reached', async () => {
      // Arrange — hasGrownBefore=true, ticks below MIN guard, qualityScoreHistory provided
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3, // < MIN_STABILIZATION_TICKS (5)
        qualityScoreHistory: [0.5, 0.5, 0.5],
        random: () => 0.1, // below WEIGHT_MUTATION_RATE → all connections mutated
      });

      // Assert
      expect(result.phase).toBe('stabilization');
    });

    it('reports no_weight_mutations when random returns above the rate threshold', async () => {
      // Arrange — stabilization phase with high random → zero mutations
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        random: () => 0.99,
      });

      // Assert
      expect(result.reason).toBe('no_weight_mutations');
    });

    it('uses injected lifecycleRunner during growth phase with provided hysteresis', async () => {
      // Arrange — hasGrownBefore=true, ticks at MAX → plateau=true → growth
      const network = new Network(4, 2, { seed: 42 });
      const hysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis,
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25, // >= MAX_STABILIZATION_TICKS
        hysteresis,
        lifecycleRunner: mockRunner,
      });

      // Assert
      expect(result.committed).toBe(true);
    });

    it('passes config overrides through resolveGrowStabilizeConfig to the lifecycle runner', async () => {
      // Arrange — growth phase with custom config moduleId
      const network = new Network(4, 2, { seed: 42 });
      let capturedModuleId = '';
      const capturingRunner = (input: { moduleId: string }) => {
        capturedModuleId = input.moduleId;
        return {
          stage: 'juvenile',
          applyOutcomes: [],
          hysteresis: {
            growthPositiveWindowCount: 0,
            pruneUnderuseWindowCount: 0,
            lastMorphKind: 'none' as const,
            cooldownWindowsRemaining: 0,
          },
        };
      };

      // Act
      await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: false,
        stabilizationTicksSinceGrowth: 0,
        config: { moduleId: 'test:override' },
        lifecycleRunner: capturingRunner,
      });

      // Assert
      expect(capturedModuleId).toBe('test:override');
    });

    it('maps edgePrune and compact apply outcomes to prune_edge operation', async () => {
      // Arrange — inject a lifecycleRunner returning edgePrune and compact outcomes
      const network = new Network(4, 2, { seed: 42 });
      const hysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [
          { status: 'applied', kind: 'edgePrune' },
          { status: 'applied', kind: 'compact' },
        ],
        hysteresis,
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        hysteresis,
        lifecycleRunner: mockRunner,
      });

      // Assert — both edgePrune and compact map to 'prune_edge'
      expect(result.operations).toEqual(['prune_edge', 'prune_edge']);
    });

    it('uses parallel weight-variant evaluation when effective variant count > 1', async () => {
      // Arrange — stabilization phase with parallel variant evaluation enabled
      // Seed 3 produces a weight variant whose improvement clears the baby-stage
      // adaptive threshold for this tiny input/target pair.
      const network = new Network(4, 2, { seed: 3 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        lifecycleStage: 'baby',
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
      });

      // Assert — the winning variant commits exactly one weight nudge
      expect(result).toMatchObject({
        phase: 'stabilization',
        committed: true,
        mutatedCount: 1,
        reason: 'weight_variant_committed',
        operations: ['param_nudge'],
      });
    });

    it('defaults to the baby lifecycle stage for variant evaluation when none is supplied', async () => {
      // Arrange — stabilization phase with parallel variants but no explicit stage
      // Seed chosen so the winning variant clears the adaptive threshold.
      const network = new Network(4, 2, { seed: 3 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
      });

      // Assert — default stage still routes to the parallel variant path
      expect(result).toMatchObject({
        phase: 'stabilization',
        committed: true,
        reason: 'weight_variant_committed',
        operations: ['param_nudge'],
      });
    });

    it('falls back to legacy weight mutations when effective variant count is one', async () => {
      // Arrange — parallel evaluation requested but with only one variant slot
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 1 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
        random: () => 0.1, // below WEIGHT_MUTATION_RATE → all connections mutated
      });

      // Assert — legacy path mutates every connection
      expect(result.mutatedCount).toBe(network.connections.length);
    });

    it('uses default baby variant count when no accelerationConfig is provided', async () => {
      // Arrange — stabilization phase with training data but no acceleration config
      // Seed chosen so the default 16 baby variants produce a commitable winner.
      const network = new Network(4, 2, { seed: 3 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
      });

      // Assert — default baby count (16) routes to the parallel variant path
      expect(result).toMatchObject({
        phase: 'stabilization',
        committed: true,
        reason: 'weight_variant_committed',
        operations: ['param_nudge'],
      });
    });

    it('falls back to a zero baseline when no score sources are provided', async () => {
      // Arrange — stabilization phase with no baselineScore, previousScore,
      // qualityScoreHistory, inputs, or target. This exercises the final `?? 0`
      // fallback in the stabilization baseline chain.
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
      } as unknown as NgeGrowStabilizeInput);

      // Assert — cycle reaches stabilization and applies weight mutations
      expect(result.phase).toBe('stabilization');
    });

    it('skips stabilization and forces growth after repeated stabilization failures', async () => {
      // Arrange — network below plateau threshold but with enough consecutive
      // stabilization failures to trigger the forced-growth bypass.
      const network = new Network(4, 2, { seed: 42 });
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [] as { status: string; kind: string }[],
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 1, // below MIN_STABILIZATION_TICKS
        consecutiveStabilizationFailures: 3,
        lifecycleRunner: mockRunner,
      });

      // Assert — growth is forced by stabilization failures, not by plateau
      expect(result.reason).toBe('forced_by_stabilization_failures');
    });
  });

  // --- Coverage iteration 4: final uncovered branches ---

  describe('hysteresis fallback and scoreHistory edge cases', () => {
    it('uses default hysteresis when none is provided during growth phase', async () => {
      // Arrange — growth phase with hasGrownBefore=true but no hysteresis provided
      const network = new Network(4, 2, { seed: 42 });
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 0,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none' as const,
          cooldownWindowsRemaining: 0,
        },
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        lifecycleRunner: mockRunner,
      });

      // Assert — cycle commits with default hysteresis fallback
      expect(result.committed).toBe(true);
    });

    it('commits growth with empty scoreHistory', async () => {
      // Arrange — growth phase with empty scoreHistory exercises length < 2 and length === 0 branches
      const network = new Network(4, 2, { seed: 42 });
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 0,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none' as const,
          cooldownWindowsRemaining: 0,
        },
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        lifecycleRunner: mockRunner,
      });

      // Assert
      expect(result.committed).toBe(true);
    });

    it('commits growth with single-element scoreHistory', async () => {
      // Arrange — growth phase with scoreHistory of length 1 exercises length < 2 branch with non-empty array
      const network = new Network(4, 2, { seed: 42 });
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' }],
        hysteresis: {
          growthPositiveWindowCount: 0,
          pruneUnderuseWindowCount: 0,
          lastMorphKind: 'none' as const,
          cooldownWindowsRemaining: 0,
        },
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [0.5],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        lifecycleRunner: mockRunner,
      });

      // Assert
      expect(result.committed).toBe(true);
    });
  });

  describe('mapOutcomesToOperations edge cases', () => {
    it('filters skipped outcomes and maps edgeDensify to add_edge', async () => {
      // Arrange — mock runner returns mixed outcomes: skipped nodeAdd and applied edgeDensify
      const network = new Network(4, 2, { seed: 42 });
      const hysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [
          { status: 'skipped', kind: 'nodeAdd' },
          { status: 'applied', kind: 'edgeDensify' },
        ],
        hysteresis,
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        hysteresis,
        lifecycleRunner: mockRunner,
      });

      // Assert — skipped outcome filtered out, only edgeDensify maps to 'add_edge'
      expect(result.operations).toEqual(['add_edge']);
    });

    it('returns empty operations when lifecycleRunner omits applyOutcomes', async () => {
      // Arrange — mock runner returns a result without applyOutcomes,
      // exercising the `?? []` nullish-coalescing fallback at line 354.
      const network = new Network(4, 2, { seed: 42 });
      const hysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };
      const mockRunner = () => ({
        stage: 'juvenile',
        hysteresis,
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        hysteresis,
        lifecycleRunner: mockRunner,
      });

      // Assert — no applyOutcomes → empty operations → not committed
      expect(result.committed).toBe(false);
    });

    it('returns empty operations when outcome kind is unhandled (slotExpand)', async () => {
      // Arrange — mock runner returns an applied outcome with kind 'slotExpand',
      // which does not match any handled kind in mapOutcomesToOperations,
      // exercising the implicit else branch at line 482.
      const network = new Network(4, 2, { seed: 42 });
      const hysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 0,
        lastMorphKind: 'none',
        cooldownWindowsRemaining: 0,
      };
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'slotExpand' }],
        hysteresis,
      });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        hysteresis,
        lifecycleRunner: mockRunner,
      });

      // Assert — slotExpand falls through all conditions → empty operations
      expect(result.operations).toEqual([]);
    });
  });

  describe('parallel variant edge branches', () => {
    it('does not commit a weight variant when the evaluator reports no winner', async () => {
      jest.resetModules();
      // `doMock` is a Jest CommonJS mocking API used here to temporarily swap
      // an ESM module import inside this test; @types/jest does not declare it.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (jest as any).doMock('./neat.nge-juvenile.variants', () => ({
        __esModule: true,
        ...jest.requireActual('./neat.nge-juvenile.variants'),
        evaluateNgeWeightVariants: jest.fn().mockResolvedValue({
          bestIndex: -1,
          bestScore: 0,
          scores: [],
          metadata: {
            backend: 'cpu',
            variantCount: 0,
            scaleDivisor: 1,
            scorer: 'default',
          },
        }),
      }));

      const { runNgeGrowStabilizeCycle } =
        await import('./neat.nge-juvenile.grow-stabilize');
      const network = new Network(4, 2, { seed: 42 });

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
      });

      expect(result).toMatchObject({
        phase: 'stabilization',
        committed: false,
        reason: 'no_weight_mutations',
        operations: [],
      });

      jest.resetModules();
    });

    it('skips committing when the winning variant points at a missing connection', async () => {
      const network = {
        nodes: [],
        connections: [],
        activate: jest.fn().mockResolvedValue([0.5, 0.5]),
      } as unknown as Network;

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 16 },
        },
        inputs: [[0.5, 0.5]],
        target: [1.0, 0.0],
      });

      expect(result).toMatchObject({
        phase: 'stabilization',
        committed: false,
        reason: 'no_weight_mutations',
        operations: [],
      });
    });

    it('falls back to effective variant count when evaluator metadata omits variantCount', async () => {
      jest.resetModules();
      // `doMock` is a Jest CommonJS mocking API used here to temporarily swap
      // an ESM module import inside this test; @types/jest does not declare it.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (jest as any).doMock('./neat.nge-juvenile.variants', () => ({
        __esModule: true,
        ...jest.requireActual('./neat.nge-juvenile.variants'),
        evaluateNgeWeightVariants: jest.fn().mockResolvedValue({
          bestIndex: -1,
          bestScore: 0,
          scores: [],
          metadata: {
            backend: 'cpu',
            // variantCount intentionally omitted to exercise the fallback
            scaleDivisor: 1,
            scorer: 'default',
          },
        }),
      }));

      const { runNgeGrowStabilizeCycle } =
        await import('./neat.nge-juvenile.grow-stabilize');
      const network = new Network(4, 2, { seed: 42 });

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.5, 0.5, 0.5],
        accelerationConfig: {
          backend: 'cpu',
          stageVariantCounts: { baby: 4 },
        },
        inputs: [[0.5, 0.5, 0.5, 0.5]],
        target: [1.0, 0.0],
      });

      expect(result).toMatchObject({
        phase: 'stabilization',
        committed: false,
        reason: 'no_weight_mutations',
        operations: [],
      });

      jest.resetModules();
    });
  });

  describe('resolveExhaustionImprovementThreshold', () => {
    it('uses magnitude scale for an unbounded score ceiling', () => {
      const threshold = resolveExhaustionImprovementThreshold(
        0.5,
        0.6,
        16,
        'baby',
        { current: 50, max: 100 },
        Infinity,
      );
      expect(threshold).toBeCloseTo(0.015, 5);
    });

    it('uses headroom scale for a bounded score ceiling', () => {
      const threshold = resolveExhaustionImprovementThreshold(
        0.5,
        0.6,
        16,
        'baby',
        { current: 50, max: 100 },
        1.0,
      );
      expect(threshold).toBeCloseTo(0.0125, 5);
    });

    it('applies baby stage fraction 0.02', () => {
      const threshold = resolveExhaustionImprovementThreshold(
        0.5,
        0.6,
        1,
        'baby',
        { current: 100, max: 100 },
        1.0,
      );
      expect(threshold).toBeCloseTo(0.01, 5);
    });

    it('applies juvenile stage fraction 0.01', () => {
      const threshold = resolveExhaustionImprovementThreshold(
        0.5,
        0.6,
        1,
        'juvenile',
        { current: 100, max: 100 },
        1.0,
      );
      expect(threshold).toBeCloseTo(0.005, 5);
    });

    it('applies adult stage fraction 0.006', () => {
      const threshold = resolveExhaustionImprovementThreshold(
        0.5,
        0.6,
        1,
        'adult',
        { current: 100, max: 100 },
        1.0,
      );
      expect(threshold).toBeCloseTo(0.003, 5);
    });

    it('zeros noise uplift when only one variant is evaluated', () => {
      const threshold = resolveExhaustionImprovementThreshold(
        100,
        110,
        1,
        'adult',
        { current: 100, max: 100 },
        Infinity,
      );
      expect(threshold).toBeCloseTo(0.66, 5);
    });

    it('caps noise uplift so very large variant counts do not inflate the threshold', () => {
      const smallCount = resolveExhaustionImprovementThreshold(
        100,
        110,
        16,
        'adult',
        { current: 100, max: 100 },
        Infinity,
      );
      const largeCount = resolveExhaustionImprovementThreshold(
        100,
        110,
        1024,
        'adult',
        { current: 100, max: 100 },
        Infinity,
      );
      expect(largeCount).toBeCloseTo(smallCount, 5);
    });

    it('raises threshold by 1.5x for a small network relative to its neuron budget', () => {
      const smallNetwork = resolveExhaustionImprovementThreshold(
        100,
        110,
        1,
        'adult',
        { current: 0, max: 100 },
        Infinity,
      );
      const largeNetwork = resolveExhaustionImprovementThreshold(
        100,
        110,
        1,
        'adult',
        { current: 100, max: 100 },
        Infinity,
      );
      expect(smallNetwork).toBeCloseTo(largeNetwork * 1.5, 5);
    });

    it('falls back to magnitude scale when baseline is within epsilon of the ceiling', () => {
      const nearCeiling = resolveExhaustionImprovementThreshold(
        0.9999999,
        0.5,
        1,
        'baby',
        { current: 100, max: 100 },
        1.0,
      );
      const midRange = resolveExhaustionImprovementThreshold(
        0.5,
        0.6,
        1,
        'baby',
        { current: 100, max: 100 },
        1.0,
      );
      expect(nearCeiling).toBeGreaterThan(midRange * 1.5);
    });

    it('falls back to a neutral neuron factor when neuronBudget.max is not finite or not positive', () => {
      // Arrange — baseline uses a full budget where current === max so the
      // finite branch also yields a neutral factor of 1.0.
      const finite = resolveExhaustionImprovementThreshold(
        100,
        110,
        1,
        'adult',
        { current: 100, max: 100 },
        Infinity,
      );

      // Act
      const infiniteMax = resolveExhaustionImprovementThreshold(
        100,
        110,
        1,
        'adult',
        { current: 100, max: Infinity },
        Infinity,
      );
      const zeroMax = resolveExhaustionImprovementThreshold(
        100,
        110,
        1,
        'adult',
        { current: 100, max: 0 },
        Infinity,
      );

      // Assert — all three produce the same threshold because the fallback
      // neuron factor is 1.0 when max is not finite or not positive.
      expect(infiniteMax).toBe(finite);
      expect(zeroMax).toBe(finite);
      expect(finite).toBeCloseTo(0.66, 5);
    });

    it('decays the threshold as consecutive weight exhaustion failures increase', () => {
      // Arrange — identical inputs except for consecutive failure count
      const baseParams = [
        100,
        110,
        16,
        'baby',
        { current: 50, max: 100 },
        Infinity,
      ] as const;

      // Act
      const lowFailures = resolveExhaustionImprovementThreshold(
        ...baseParams,
        0,
      );
      const highFailures = resolveExhaustionImprovementThreshold(
        ...baseParams,
        10,
      );

      // Assert — repeated failures shrink the improvement bar
      expect(highFailures).toBeLessThan(lowFailures);
    });
  });

  describe('resolveExhaustionForceGrowthThreshold', () => {
    it('sets baby default 16 variants to three consecutive exhaustion ticks', () => {
      expect(resolveExhaustionForceGrowthThreshold(16, false)).toBe(3);
    });

    it('sets 1024 variants to one consecutive exhaustion tick', () => {
      expect(resolveExhaustionForceGrowthThreshold(1024, false)).toBe(1);
    });

    it('clamps the raw exhaustion count to a minimum of one', () => {
      expect(
        resolveExhaustionForceGrowthThreshold(1024, false),
      ).toBeGreaterThanOrEqual(1);
    });

    it('clamps the raw exhaustion count to a maximum of eight', () => {
      expect(resolveExhaustionForceGrowthThreshold(1, false)).toBe(8);
    });

    it('doubles the threshold after bad growth, capped at sixteen', () => {
      expect(resolveExhaustionForceGrowthThreshold(1, true)).toBe(16);
    });

    it('floors the post-growth boosted threshold at four', () => {
      expect(resolveExhaustionForceGrowthThreshold(1024, true)).toBe(4);
    });
  });

  describe('resolveStageFraction', () => {
    it('returns baby fraction 0.02 for embryo stage', () => {
      expect(resolveStageFraction('embryo')).toBe(0.02);
    });

    it('returns baby fraction 0.02 for baby stage', () => {
      expect(resolveStageFraction('baby')).toBe(0.02);
    });

    it('returns juvenile fraction 0.01', () => {
      expect(resolveStageFraction('juvenile')).toBe(0.01);
    });

    it('returns adult fraction 0.006 for adult stage', () => {
      expect(resolveStageFraction('adult')).toBe(0.006);
    });

    it('returns adult fraction 0.006 for equilibrium stage', () => {
      expect(resolveStageFraction('equilibrium')).toBe(0.006);
    });
  });

  describe('resolveNoiseSigmaFraction', () => {
    it('returns baby noise sigma fraction 0.003 for embryo stage', () => {
      expect(resolveNoiseSigmaFraction('embryo')).toBe(0.003);
    });

    it('returns baby noise sigma fraction 0.003 for baby stage', () => {
      expect(resolveNoiseSigmaFraction('baby')).toBe(0.003);
    });

    it('returns juvenile noise sigma fraction 0.002', () => {
      expect(resolveNoiseSigmaFraction('juvenile')).toBe(0.002);
    });

    it('returns adult noise sigma fraction 0.001 for adult stage', () => {
      expect(resolveNoiseSigmaFraction('adult')).toBe(0.001);
    });

    it('returns adult noise sigma fraction 0.001 for equilibrium stage', () => {
      expect(resolveNoiseSigmaFraction('equilibrium')).toBe(0.001);
    });
  });

  describe('exhaustion constants', () => {
    it('no longer exports NGE_GROW_STABILIZE_IMPROVEMENT_THRESHOLD', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(
        (constants as any).NGE_GROW_STABILIZE_IMPROVEMENT_THRESHOLD,
      ).toBeUndefined();
    });

    it('no longer exports WEIGHT_VARIANT_DELTA from grow-stabilize', async () => {
      const gs = await import('./neat.nge-juvenile.grow-stabilize');
      expect((gs as any).WEIGHT_VARIANT_DELTA).toBeUndefined();
    });

    it('exports NGE_EXHAUSTION_SCORE_EPSILON equal to 1e-6', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_SCORE_EPSILON).toBe(1e-6);
    });

    it('exports NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY equal to 0.003', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY).toBe(
        0.003,
      );
    });

    it('exports NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE equal to 0.002', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(
        (constants as any).NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE,
      ).toBe(0.002);
    });

    it('exports NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT equal to 0.001', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT).toBe(
        0.001,
      );
    });

    it('exports NGE_EXHAUSTION_STAGE_FRACTION_BABY equal to 0.02', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_STAGE_FRACTION_BABY).toBe(0.02);
    });

    it('exports NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE equal to 0.01', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE).toBe(
        0.01,
      );
    });

    it('exports NGE_EXHAUSTION_STAGE_FRACTION_ADULT equal to 0.006', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_STAGE_FRACTION_ADULT).toBe(
        0.006,
      );
    });

    it('exports NGE_EXHAUSTION_TICK_BUDGET equal to 48', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_TICK_BUDGET).toBe(48);
    });

    it('exports NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS equal to 1', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS).toBe(1);
    });

    it('exports NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS equal to 8', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS).toBe(8);
    });

    it('exports NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS equal to 16', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(
        (constants as any).NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS,
      ).toBe(16);
    });

    it('exports NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST equal to 2.0', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect(
        (constants as any).NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST,
      ).toBe(2.0);
    });

    it('exports NGE_EXHAUSTION_NEURON_BUDGET_FACTOR equal to 0.5', async () => {
      const constants = await import('./neat.nge-juvenile.constants');
      expect((constants as any).NGE_EXHAUSTION_NEURON_BUDGET_FACTOR).toBe(0.5);
    });
  });

  describe('post-growth baseline guard', () => {
    it('activates the anti-runaway boost when the baseline drops below the pre-growth baseline', async () => {
      const network = new Network(4, 2, { seed: 42 });

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        qualityScoreHistory: [0.0],
        preGrowthBaseline: 1.0,
      } as unknown as NgeGrowStabilizeInput);

      expect(result.postGrowthThresholdActive).toBe(true);
    });

    it('expires the post-growth boost once stabilization ticks reach the maximum', async () => {
      const network = new Network(4, 2, { seed: 42 });
      const maxTicks = 25;

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: maxTicks,
        qualityScoreHistory: [0.5, 0.5, 0.5, 0.5, 0.5],
        postGrowthThresholdActive: true,
      } as unknown as NgeGrowStabilizeInput);

      expect(result.postGrowthThresholdActive).toBe(false);
    });

    it('does not activate the anti-runaway boost when the baseline stays within the pre-growth threshold', async () => {
      // Arrange — preGrowthBaseline is set and the resolved baseline (0 because
      // no score sources are provided) is NOT below preGrowthBaseline - threshold.
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 3,
        preGrowthBaseline: 0,
      } as unknown as NgeGrowStabilizeInput);

      // Assert — false branch keeps postGrowthThresholdActive false
      expect(result.postGrowthThresholdActive).toBe(false);
    });
  });

  describe('DF6–DF9 regression coverage: hysteresis fallback branches', () => {
    it('falls back to input hysteresis when the lifecycle runner omits hysteresis', async () => {
      // DF6/DF7/DF9 regression coverage — exercise `lifecycleResult.hysteresis ?? input.hysteresis`
      // on the growth path when the runner returns applyOutcomes but no hysteresis object.
      const network = new Network(4, 2, { seed: 42 });
      const inputHysteresis: NgeHysteresisState = {
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 3,
        lastMorphKind: 'nodeAdd',
        cooldownWindowsRemaining: 0,
      };
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [{ status: 'applied', kind: 'nodeAdd' as const }],
      });

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        hysteresis: inputHysteresis,
        lifecycleRunner: mockRunner,
      });

      expect(result.committed).toBe(true);
      expect(result.hysteresis).toMatchObject({
        growthPositiveWindowCount: 2,
        pruneUnderuseWindowCount: 3,
        lastMorphKind: 'nodeAdd',
      });
    });

    it('forces ADD_NODE on first growth when runner omits hysteresis and applyOutcomes', async () => {
      // DF8 regression coverage — first-growth guarantee plus the
      // `resultHysteresis?.pruneUnderuseWindowCount ?? 0` fallback branch.
      const network = new Network(4, 2, { seed: 42 });
      const initialNodeCount = network.nodes.length;
      const mockRunner = () => ({
        stage: 'juvenile',
        applyOutcomes: [],
      });

      const result = await runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: false,
        stabilizationTicksSinceGrowth: 0,
        lifecycleRunner: mockRunner,
      });

      expect(result.phase).toBe('growth');
      expect(result.committed).toBe(true);
      expect(result.operations).toEqual(['add_node']);
      expect(network.nodes.length).toBeGreaterThan(initialNodeCount);
      expect(result.hysteresis).toMatchObject({
        lastMorphKind: 'nodeAdd',
        growthPositiveWindowCount: 0,
        pruneUnderuseWindowCount: 0,
      });
    });
  });
});
