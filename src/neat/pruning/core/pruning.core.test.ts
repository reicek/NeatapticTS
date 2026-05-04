import type { NeatLikeForPruning } from './pruning.types';
import {
  applyAdaptivePruneLevelToPopulation,
  applyPruningToPopulation,
  computeMeanConnectionCount,
  computeMeanNodeCount,
  computeNextAdaptivePruneLevel,
  computePopulationMetrics,
  computeRampFraction,
  computeTargetRemainingMetric,
  computeTargetSparsityNow,
  initializeAdaptivePruningState,
  resolveActiveAdaptivePruningOptions,
  resolveActiveEvolutionPruningOptions,
  resolveAdaptivePruneBaseline,
  resolveObservedMetricValue,
  shouldAdjustAdaptivePruning,
} from './pruning.core';

type RecordedPruneCall = { method: string | undefined; sparsity: number };
type RecordedGenome = NeatLikeForPruning['population'][number] & {
  calls: RecordedPruneCall[];
};
type PruningCoreTestHost = Omit<NeatLikeForPruning, 'population'> & {
  population: RecordedGenome[];
};

function createRecordedGenome(input: {
  connectionCount?: number;
  nodeCount?: number;
  supportsPruning?: boolean;
}): RecordedGenome {
  const calls: RecordedPruneCall[] = [];
  const genome = {
    calls,
    connections: Array.from({ length: input.connectionCount ?? 0 }, () => ({})),
    nodes: Array.from({ length: input.nodeCount ?? 0 }, () => ({})),
  } as RecordedGenome;

  if (input.supportsPruning !== false) {
    genome.pruneToSparsity = (sparsity: number, method?: string) => {
      calls.push({ method, sparsity });
    };
  }

  return genome;
}

function createPruningCoreHost(
  input: {
    adaptivePruneLevel?: number;
    generation?: number;
    evolutionPruning?: NeatLikeForPruning['options']['evolutionPruning'];
    genomes?: RecordedGenome[];
  } = {},
): PruningCoreTestHost {
  return {
    _adaptivePruneLevel: input.adaptivePruneLevel,
    generation: input.generation ?? 0,
    options: {
      adaptivePruning: undefined,
      evolutionPruning: input.evolutionPruning,
    },
    population: input.genomes ?? [
      createRecordedGenome({ connectionCount: 4, nodeCount: 2 }),
      createRecordedGenome({ connectionCount: 6, nodeCount: 3 }),
    ],
  };
}

describe('neat pruning core chapter', () => {
  describe('resolveActiveEvolutionPruningOptions', () => {
    describe('given scheduled pruning is not configured', () => {
      it('returns null', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({
          evolutionPruning: undefined,
          generation: 4,
        });

        // Act
        const activeOptions = resolveActiveEvolutionPruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });

    describe('given the current generation falls outside the configured pruning interval', () => {
      it('returns null', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({
          evolutionPruning: {
            interval: 3,
            startGeneration: 1,
            targetSparsity: 0.4,
          },
          generation: 2,
        });

        // Act
        const activeOptions = resolveActiveEvolutionPruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });
  });

  describe('resolveActiveAdaptivePruningOptions', () => {
    describe('given adaptive pruning is disabled', () => {
      it('returns null', () => {
        // Arrange
        const pruningHost = createPruningCoreHost();
        pruningHost.options.adaptivePruning = { enabled: false };

        // Act
        const activeOptions = resolveActiveAdaptivePruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });

    describe('given adaptive pruning is not configured at all', () => {
      it('returns null via the ?? null fallback at line 168', () => {
        // Arrange: adaptivePruning is undefined → options.adaptivePruning ?? null = null
        const pruningHost = createPruningCoreHost();
        // adaptivePruning is already undefined in createPruningCoreHost

        // Act
        const activeOptions = resolveActiveAdaptivePruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });
  });

  describe('computeRampFraction', () => {
    describe('given scheduled pruning disables ramping', () => {
      it('returns the fully applied ramp fraction', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({ generation: 4 });

        // Act
        const rampFraction = computeRampFraction(pruningHost, {
          rampGenerations: 0,
          startGeneration: 2,
        });

        // Assert
        expect(rampFraction).toBe(1);
      });
    });
  });

  describe('applyPruningToPopulation', () => {
    describe('given one genome does not implement pruning support', () => {
      it('skips the unsupported genome and applies the configured method to the supported one', () => {
        // Arrange
        const unsupportedGenome = createRecordedGenome({
          connectionCount: 4,
          nodeCount: 2,
          supportsPruning: false,
        });
        const supportedGenome = createRecordedGenome({
          connectionCount: 6,
          nodeCount: 3,
        });
        const pruningHost = createPruningCoreHost({
          genomes: [unsupportedGenome, supportedGenome],
        });

        // Act
        applyPruningToPopulation(pruningHost, { method: 'snip' }, 0.4);

        // Assert
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [],
          [{ method: 'snip', sparsity: 0.4 }],
        ]);
      });
    });

    describe('given no method is specified in the pruning options', () => {
      it('defaults to magnitude pruning', () => {
        // Arrange: method omitted → ?? 'magnitude' right arm (line 143)
        const pruningHost = createPruningCoreHost();

        // Act
        applyPruningToPopulation(pruningHost, {}, 0.3);

        // Assert: method defaults to 'magnitude'
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [{ method: 'magnitude', sparsity: 0.3 }],
          [{ method: 'magnitude', sparsity: 0.3 }],
        ]);
      });
    });
  });

  describe('initializeAdaptivePruningState', () => {
    describe('given the shared adaptive prune level already exists', () => {
      it('keeps the existing level unchanged', () => {
        // Arrange
        const pruningHost = createPruningCoreHost({ adaptivePruneLevel: 0.3 });

        // Act
        initializeAdaptivePruningState(pruningHost);

        // Assert
        expect(pruningHost._adaptivePruneLevel).toBe(0.3);
      });
    });

    describe('given no adaptive prune level has been initialized yet', () => {
      it('sets the prune level to zero', () => {
        // Arrange: _adaptivePruneLevel is undefined → exercises line 190 TRUE arm
        const pruningHost = createPruningCoreHost();

        // Act
        initializeAdaptivePruningState(pruningHost);

        // Assert
        expect(pruningHost._adaptivePruneLevel).toBe(0);
      });
    });
  });

  describe('computeTargetSparsityNow', () => {
    describe('given options with no targetSparsity, no rampGenerations, and no startGeneration', () => {
      it('uses all defaults and returns zero target sparsity', () => {
        // Arrange: no targetSparsity (→ 0), no rampGenerations (→ 0, full ramp), no startGeneration (→ 0)
        const pruningHost = createPruningCoreHost({ generation: 0 });

        // Act
        const target = computeTargetSparsityNow(pruningHost, {});

        // Assert: 0 (targetSparsity default) × 1 (full ramp) = 0
        expect(target).toBe(0);
      });
    });
  });

  describe('computeMeanNodeCount', () => {
    describe('given an empty population', () => {
      it('returns zero without dividing by zero', () => {
        // Arrange: empty population exercises the || 1 guard
        const pruningHost = createPruningCoreHost({ genomes: [] });

        // Act
        const mean = computeMeanNodeCount(pruningHost);

        // Assert
        expect(mean).toBe(0);
      });
    });
  });

  describe('computeMeanConnectionCount', () => {
    describe('given an empty population', () => {
      it('returns zero without dividing by zero', () => {
        // Arrange: empty population exercises the || 1 guard
        const pruningHost = createPruningCoreHost({ genomes: [] });

        // Act
        const mean = computeMeanConnectionCount(pruningHost);

        // Assert
        expect(mean).toBe(0);
      });
    });
  });

  describe('resolveObservedMetricValue', () => {
    describe('given no metric is specified in options', () => {
      it('defaults to the connection-count metric', () => {
        // Arrange: metric omitted → ?? 'connections' fallback
        const metrics = { meanNodeCount: 5, meanConnectionCount: 10 };

        // Act
        const value = resolveObservedMetricValue({}, metrics);

        // Assert
        expect(value).toBe(10);
      });
    });

    describe('given metric is set to nodes', () => {
      it('returns the node-count metric', () => {
        // Arrange: metric = 'nodes' exercises the TRUE ternary arm at line 274
        const metrics = { meanNodeCount: 5, meanConnectionCount: 10 };

        // Act
        const value = resolveObservedMetricValue({ metric: 'nodes' }, metrics);

        // Assert
        expect(value).toBe(5);
      });
    });
  });

  describe('resolveAdaptivePruneBaseline', () => {
    describe('given the baseline was already initialized', () => {
      it('returns the existing baseline without overwriting it', () => {
        // Arrange: pre-seed _adaptivePruneBaseline to exercise the FALSE arm at line 295
        const pruningHost = createPruningCoreHost();
        (pruningHost as Record<string, unknown>)._adaptivePruneBaseline = 8;

        // Act
        const baseline = resolveAdaptivePruneBaseline(pruningHost, 12);

        // Assert: returns stored value, not currentMetricValue
        expect(baseline).toBe(8);
      });
    });
  });

  describe('computeTargetRemainingMetric', () => {
    describe('given no targetSparsity is specified', () => {
      it('uses the default 0.5 sparsity to compute the remaining target', () => {
        // Arrange: targetSparsity omitted → ?? 0.5 fallback
        // Act
        const remaining = computeTargetRemainingMetric({}, 10);

        // Assert: 10 * (1 - 0.5) = 5
        expect(remaining).toBe(5);
      });
    });
  });

  describe('shouldAdjustAdaptivePruning', () => {
    describe('given no tolerance is specified and baseline is zero', () => {
      it('uses default tolerance and clamps baseline to 1', () => {
        // Arrange: tolerance omitted → ?? 0.05, baseline = 0 → || 1
        // normalizedDiff = (5 - 3) / 1 = 2 > 0.05 → true
        // Act
        const result = shouldAdjustAdaptivePruning({}, 5, 3, 0);

        // Assert
        expect(result).toBe(true);
      });
    });
  });

  describe('computeNextAdaptivePruneLevel', () => {
    describe('given no adjustRate, no targetSparsity, and metric below target', () => {
      it('uses defaults and relaxes the prune level when below target', () => {
        // Arrange: adjustRate omitted → ?? 0.02, targetSparsity omitted → ?? 0.5
        // metric (2) < targetRemaining (5) → direction = -1 → relax
        // Act
        const nextLevel = computeNextAdaptivePruneLevel({}, 0.5, 2, 5);

        // Assert: 0.5 + 0.02 * -1 = 0.48
        expect(nextLevel).toBeCloseTo(0.48, 5);
      });
    });

    describe('given metric exceeds the target remaining complexity', () => {
      it('tightens the prune level by using the +1 adjustment direction', () => {
        // Arrange: metric (8) > targetRemaining (5) → direction = +1 (line 380 branch 0)
        const nextLevel = computeNextAdaptivePruneLevel(
          { adjustRate: 0.1, targetSparsity: 0.4 },
          0.3,
          8,
          5,
        );

        // Assert: 0.3 + 0.1 * 1 = 0.4
        expect(nextLevel).toBeCloseTo(0.4, 5);
      });
    });
  });

  describe('resolveActiveEvolutionPruningOptions', () => {
    describe('given evolution pruning is configured with no startGeneration and no interval', () => {
      it('returns the options when generation is 0 and all defaults align', () => {
        // Arrange: startGeneration omitted → ?? 0, interval omitted → ?? 1
        // generation=0: 0 >= 0 ✓, (0-0)%1 = 0 ✓ → returns options
        const pruningHost = createPruningCoreHost({
          evolutionPruning: { targetSparsity: 0.5 },
          generation: 0,
        });

        // Act
        const activeOptions = resolveActiveEvolutionPruningOptions(pruningHost);

        // Assert
        expect(activeOptions).not.toBeNull();
      });
    });

    describe('given the current generation is before the start generation', () => {
      it('returns null', () => {
        // Arrange: generation < startGeneration → exercises line 63 return null
        const pruningHost = createPruningCoreHost({
          evolutionPruning: { startGeneration: 5, targetSparsity: 0.5 },
          generation: 2,
        });

        // Act
        const activeOptions = resolveActiveEvolutionPruningOptions(pruningHost);

        // Assert
        expect(activeOptions).toBeNull();
      });
    });
  });

  describe('resolveActiveAdaptivePruningOptions', () => {
    describe('given adaptive pruning is enabled', () => {
      it('returns the adaptive pruning options', () => {
        // Arrange: covers the return options path at line 173
        const pruningHost = createPruningCoreHost();
        pruningHost.options.adaptivePruning = {
          enabled: true,
          targetSparsity: 0.4,
        };

        // Act
        const activeOptions = resolveActiveAdaptivePruningOptions(pruningHost);

        // Assert
        expect(activeOptions).not.toBeNull();
      });
    });
  });

  describe('computePopulationMetrics', () => {
    describe('given a population with known node and connection counts', () => {
      it('returns the mean node and connection counts', () => {
        // Arrange: covers computePopulationMetrics and its delegates
        const pruningHost = createPruningCoreHost({
          genomes: [
            createRecordedGenome({ nodeCount: 4, connectionCount: 8 }),
            createRecordedGenome({ nodeCount: 2, connectionCount: 4 }),
          ],
        });

        // Act
        const metrics = computePopulationMetrics(pruningHost);

        // Assert
        expect(metrics).toEqual({ meanNodeCount: 3, meanConnectionCount: 6 });
      });
    });
  });

  describe('computeRampFraction', () => {
    describe('given rampGenerations is positive', () => {
      it('computes the normalized progress through the ramp window', () => {
        // Arrange: rampGenerations=4, startGeneration=0, generation=2 → progress=0.5
        const pruningHost = createPruningCoreHost({ generation: 2 });

        // Act
        const rampFraction = computeRampFraction(pruningHost, {
          rampGenerations: 4,
          startGeneration: 0,
        });

        // Assert
        expect(rampFraction).toBeCloseTo(0.5, 5);
      });
    });

    describe('given rampGenerations is positive and no startGeneration is specified', () => {
      it('defaults startGeneration to 0 and computes progress correctly', () => {
        // Arrange: startGeneration omitted → ?? 0, covers line 120 right arm
        const pruningHost = createPruningCoreHost({ generation: 4 });

        // Act
        const rampFraction = computeRampFraction(pruningHost, {
          rampGenerations: 4,
        });

        // Assert: (4-0)/4 = 1.0
        expect(rampFraction).toBeCloseTo(1, 5);
      });
    });
  });

  describe('resolveAdaptivePruneBaseline', () => {
    describe('given the baseline has not been set yet', () => {
      it('initializes the baseline to the current metric value', () => {
        // Arrange: _adaptivePruneBaseline undefined → exercises line 296 initialization
        const pruningHost = createPruningCoreHost();

        // Act
        const baseline = resolveAdaptivePruneBaseline(pruningHost, 7);

        // Assert
        expect(baseline).toBe(7);
      });
    });
  });

  describe('applyAdaptivePruneLevelToPopulation', () => {
    describe('given one genome does not implement adaptive pruning support', () => {
      it('skips the unsupported genome and applies the shared magnitude prune level to the supported one', () => {
        // Arrange
        const unsupportedGenome = createRecordedGenome({
          connectionCount: 4,
          nodeCount: 2,
          supportsPruning: false,
        });
        const supportedGenome = createRecordedGenome({
          connectionCount: 6,
          nodeCount: 3,
        });
        const pruningHost = createPruningCoreHost({
          genomes: [unsupportedGenome, supportedGenome],
        });

        // Act
        applyAdaptivePruneLevelToPopulation(pruningHost, 0.25);

        // Assert
        expect(pruningHost.population.map((genome) => genome.calls)).toEqual([
          [],
          [{ method: 'magnitude', sparsity: 0.25 }],
        ]);
      });
    });
  });
});
