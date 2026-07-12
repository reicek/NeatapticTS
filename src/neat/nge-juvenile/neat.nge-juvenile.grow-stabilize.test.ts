/**
 * Red-phase test contracts for the NGE grow-stabilize cycle extraction.
 *
 * These tests define the expected contract for the extracted grow-stabilize
 * module (`src/neat/nge-juvenile/neat.nge-juvenile.grow-stabilize.ts`) before
 * the implementation exists. They cover:
 *
 * - `resolveAdaptiveHysteresis` thresholds (2/3/5 for node counts ≤200/≤500/>500)
 * - `isPlateauReached` guards (first-growth bypass, MIN/MAX stabilization ticks)
 * - `applyWeightMutations` rate with deterministic random
 * - `runNgeGrowStabilizeCycle` function existence and first-growth commit
 *
 * All tests fail because `./neat.nge-juvenile.grow-stabilize` does not exist yet.
 * The failure reason is "missing implementation" (import error), not syntax error
 * or bad fixture.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */
import Network from '../../architecture/network';
import {
  applyWeightMutations,
  computeGrowthThrottle,
  isPlateauReached,
  resolveAdaptiveHysteresis,
  runNgeGrowStabilizeCycle,
} from './neat.nge-juvenile.grow-stabilize';
import type { NgeHysteresisState } from './neat.nge-juvenile.types';

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

    it('performs first-growth unconditional commit when hasGrownBefore is false', () => {
      // Arrange — seed network, first growth, positive score history
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: false,
        stabilizationTicksSinceGrowth: 0,
      });

      // Assert — first growth should always commit
      expect(result.committed).toBe(true);
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
  });

  describe('runNgeGrowStabilizeCycle additional branches', () => {
    it('enters stabilization phase with weight mutations when plateau is not reached', () => {
      // Arrange — hasGrownBefore=true, ticks below MIN guard, qualityScoreHistory provided
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = runNgeGrowStabilizeCycle({
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

    it('reports no_weight_mutations when random returns above the rate threshold', () => {
      // Arrange — stabilization phase with high random → zero mutations
      const network = new Network(4, 2, { seed: 42 });

      // Act
      const result = runNgeGrowStabilizeCycle({
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

    it('uses injected lifecycleRunner during growth phase with provided hysteresis', () => {
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
      const result = runNgeGrowStabilizeCycle({
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

    it('passes config overrides through resolveGrowStabilizeConfig to the lifecycle runner', () => {
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
      runNgeGrowStabilizeCycle({
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

    it('maps edgePrune and compact apply outcomes to prune_edge operation', () => {
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
      const result = runNgeGrowStabilizeCycle({
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
  });

  // --- Coverage iteration 4: final uncovered branches ---

  describe('hysteresis fallback and scoreHistory edge cases', () => {
    it('uses default hysteresis when none is provided during growth phase', () => {
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
      const result = runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [1, 2, 3, 4],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        lifecycleRunner: mockRunner,
      });

      // Assert — cycle commits with default hysteresis fallback
      expect(result.committed).toBe(true);
    });

    it('commits growth with empty scoreHistory', () => {
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
      const result = runNgeGrowStabilizeCycle({
        network,
        scoreHistory: [],
        hasGrownBefore: true,
        stabilizationTicksSinceGrowth: 25,
        lifecycleRunner: mockRunner,
      });

      // Assert
      expect(result.committed).toBe(true);
    });

    it('commits growth with single-element scoreHistory', () => {
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
      const result = runNgeGrowStabilizeCycle({
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
    it('filters skipped outcomes and maps edgeDensify to add_edge', () => {
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
      const result = runNgeGrowStabilizeCycle({
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

    it('returns empty operations when lifecycleRunner omits applyOutcomes', () => {
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
      const result = runNgeGrowStabilizeCycle({
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

    it('returns empty operations when outcome kind is unhandled (slotExpand)', () => {
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
      const result = runNgeGrowStabilizeCycle({
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
});
