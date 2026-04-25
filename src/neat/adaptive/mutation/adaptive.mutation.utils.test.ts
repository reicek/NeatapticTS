import {
  DEFAULT_ADAPT_EVERY,
  DEFAULT_INITIAL_MUTATION_RATE,
  DEFAULT_MAX_MUTATION_AMOUNT,
  DEFAULT_MAX_MUTATION_RATE,
  DEFAULT_MIN_MUTATION_AMOUNT,
  DEFAULT_MIN_MUTATION_RATE,
  DEFAULT_MUTATION_AMOUNT,
  DEFAULT_MUTATION_AMOUNT_SIGMA,
  DEFAULT_MUTATION_SIGMA,
  MUTATION_SIGMA_SCALE,
  MUTATION_STRATEGY_ANNEAL,
  MUTATION_STRATEGY_EXPLORE_LOW,
  MUTATION_STRATEGY_TWO_TIER,
} from '../core/adaptive.core.constants';
import type {
  Genome,
  MutationOutcome,
  MutationPartitions,
  MutationSettings,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';
import {
  applyMutationsToPopulation,
  applyTwoTierAmountDelta,
  applyTwoTierDelta,
  applyTwoTierFallback,
  clampValue,
  collectScoredGenomes,
  resolveAmountDelta,
  resolveMutationSettings,
  resolveRandomSource,
  resolveRateDelta,
  shouldAdaptThisGeneration,
  shouldApplyTwoTierFallback,
  sortScoredGenomes,
} from './adaptive.mutation.utils';

function createGenome(input: {
  mutationAmount?: number | null;
  mutationRate?: number | null;
  score?: number;
}): Genome {
  return {
    _mutAmount: input.mutationAmount,
    _mutRate: input.mutationRate,
    score: input.score,
  };
}

function createMutationSettings(
  overrides: Partial<MutationSettings> = {},
): MutationSettings {
  return {
    adaptAmount: false,
    amountSigma: 0.25,
    generation: 12,
    initialRate: 0.5,
    maxAmount: 8,
    maxRate: 1,
    minAmount: 1,
    minRate: 0.1,
    mutationAmountDefault: 3,
    populationSize: 4,
    sigmaBase: 0.3,
    strategy: MUTATION_STRATEGY_TWO_TIER,
    ...overrides,
  };
}

function createAdaptiveController(
  overrides: Partial<NeatLikeWithAdaptive> = {},
): NeatLikeWithAdaptive {
  return {
    generation: 7,
    input: 2,
    options: {
      adaptiveMutation: {},
      mutationAmount: undefined,
    },
    output: 1,
    population: [],
    ...overrides,
  };
}

describe('adaptive mutation utility chapter', () => {
  describe('shouldAdaptThisGeneration', () => {
    describe('given the configured cadence does not divide the current generation', () => {
      it('skips the adaptive mutation pass', () => {
        // Arrange
        const generation = 5;
        const adaptiveMutation = { adaptEvery: 3 };

        // Act
        const shouldAdapt = shouldAdaptThisGeneration(
          generation,
          adaptiveMutation,
        );

        // Assert
        expect(shouldAdapt).toBe(false);
      });
    });
  });

  describe('collectScoredGenomes', () => {
    describe('given the population mixes numeric and missing scores', () => {
      it('keeps only the genomes with numeric scores', () => {
        // Arrange
        const population = [
          createGenome({ score: 0.2 }),
          createGenome({ score: undefined }),
          createGenome({ score: 0.8 }),
        ];

        // Act
        const scoredGenomes = collectScoredGenomes(population);

        // Assert
        expect(scoredGenomes).toEqual([
          population[0],
          population[2],
        ]);
      });
    });
  });

  describe('sortScoredGenomes', () => {
    describe('given one direct utility call includes missing score fallbacks', () => {
      it('sorts using zero as the fallback score', () => {
        // Arrange
        const scoredGenomes = [
          createGenome({ score: 2 }),
          createGenome({ score: undefined }),
          createGenome({ score: 1 }),
          createGenome({ score: undefined }),
        ];

        // Act
        const sortedScores = sortScoredGenomes(scoredGenomes).map(
          (genome) => genome.score ?? 0,
        );

        // Assert
        expect(sortedScores).toEqual([0, 0, 1, 2]);
      });
    });
  });

  describe('resolveMutationSettings', () => {
    describe('given the adaptive config omits every override', () => {
      it('returns the shared mutation defaults and host counters', () => {
        // Arrange
        const adaptiveController = createAdaptiveController({
          generation: 9,
          population: [
            createGenome({ score: 0.1 }),
            createGenome({ score: 0.2 }),
          ],
        });

        // Act
        const mutationSettings = resolveMutationSettings(
          adaptiveController,
          {},
        );

        // Assert
        expect(mutationSettings).toEqual({
          adaptAmount: false,
          amountSigma: DEFAULT_MUTATION_AMOUNT_SIGMA,
          generation: 9,
          initialRate: DEFAULT_INITIAL_MUTATION_RATE,
          maxAmount: DEFAULT_MAX_MUTATION_AMOUNT,
          maxRate: DEFAULT_MAX_MUTATION_RATE,
          minAmount: DEFAULT_MIN_MUTATION_AMOUNT,
          minRate: DEFAULT_MIN_MUTATION_RATE,
          mutationAmountDefault: DEFAULT_MUTATION_AMOUNT,
          populationSize: 2,
          sigmaBase: DEFAULT_MUTATION_SIGMA * MUTATION_SIGMA_SCALE,
          strategy: MUTATION_STRATEGY_TWO_TIER,
        });
      });
    });
  });

  describe('resolveRandomSource', () => {
    describe('given the host omits the deterministic RNG factory', () => {
      it('falls back to Math.random', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const mathRandomSpy = jest
          .spyOn(Math, 'random')
          .mockReturnValue(0.25);

        try {
          // Act
          const randomValue = resolveRandomSource(adaptiveController)();

          // Assert
          expect(randomValue).toBe(0.25);
        } finally {
          mathRandomSpy.mockRestore();
        }
      });
    });
  });

  describe('applyMutationsToPopulation', () => {
    describe('given one genome is missing a mutation rate', () => {
      it('skips the missing-rate genome while mutating the eligible one', () => {
        // Arrange
        const population = [
          createGenome({ mutationRate: undefined, score: 0.1 }),
          createGenome({ mutationRate: 0.5, score: 0.9 }),
        ];
        const partitions: MutationPartitions = {
          bottomHalf: [],
          topHalf: [],
        };
        const mutationSettings = createMutationSettings();

        // Act
        const mutationOutcome = applyMutationsToPopulation(
          population,
          partitions,
          mutationSettings,
          () => 1,
        );

        // Assert
        expect({
          mutationOutcome,
          mutationRates: population.map((genome) => genome._mutRate ?? null),
        }).toEqual({
          mutationOutcome: { hasDecrease: true, hasIncrease: false },
          mutationRates: [null, 0.2],
        });
      });
    });
  });

  describe('resolveRateDelta', () => {
    describe('given the configured strategy is not recognized', () => {
      it('returns the unclipped stochastic base delta', () => {
        // Arrange
        const mutationSettings = createMutationSettings({
          sigmaBase: 0.45,
          strategy: 'custom-strategy',
        });
        const genome = createGenome({ mutationRate: 0.4, score: 0.3 });

        // Act
        const rateDelta = resolveRateDelta(
          mutationSettings,
          () => 1,
          genome,
          0,
          new Set<Genome>([createGenome({ mutationRate: 0.5, score: 1 })]),
          new Set<Genome>([createGenome({ mutationRate: 0.5, score: 0 })]),
        );

        // Assert
        expect(rateDelta).toBe(0.45);
      });
    });
  });

  describe('applyTwoTierDelta', () => {
    describe('given both score partitions are present but the genome is in neither set', () => {
      it('keeps the original base delta', () => {
        // Arrange
        const candidateGenome = createGenome({ mutationRate: 0.5, score: 0.4 });
        const topGenome = createGenome({ mutationRate: 0.5, score: 0.9 });
        const bottomGenome = createGenome({ mutationRate: 0.5, score: 0.1 });

        // Act
        const rateDelta = applyTwoTierDelta(
          -0.2,
          candidateGenome,
          2,
          new Set<Genome>([topGenome]),
          new Set<Genome>([bottomGenome]),
        );

        // Assert
        expect(rateDelta).toBe(-0.2);
      });
    });
  });

  describe('resolveAmountDelta', () => {
    describe('given the strategy is not two-tier', () => {
      it('returns the raw amount delta without repartitioning', () => {
        // Arrange
        const mutationSettings = createMutationSettings({
          amountSigma: 0.4,
          strategy: MUTATION_STRATEGY_EXPLORE_LOW,
        });
        const genome = createGenome({ mutationAmount: 3, mutationRate: 0.5 });

        // Act
        const amountDelta = resolveAmountDelta(
          mutationSettings,
          () => 1,
          genome,
          1,
          new Set<Genome>(),
          new Set<Genome>(),
        );

        // Assert
        expect(amountDelta).toBe(0.4);
      });
    });
  });

  describe('applyTwoTierAmountDelta', () => {
    describe('given no score partitions exist and the genome index is even', () => {
      it('returns the positive absolute delta', () => {
        // Arrange
        const genome = createGenome({ mutationAmount: 3, mutationRate: 0.5 });

        // Act
        const amountDelta = applyTwoTierAmountDelta(
          -0.3,
          genome,
          0,
          new Set<Genome>(),
          new Set<Genome>(),
        );

        // Assert
        expect(amountDelta).toBe(0.3);
      });
    });

    describe('given no score partitions exist and the genome index is odd', () => {
      it('returns the negative absolute delta', () => {
        // Arrange
        const genome = createGenome({ mutationAmount: 3, mutationRate: 0.5 });

        // Act
        const amountDelta = applyTwoTierAmountDelta(
          0.3,
          genome,
          1,
          new Set<Genome>(),
          new Set<Genome>(),
        );

        // Assert
        expect(amountDelta).toBe(-0.3);
      });
    });

    describe('given both score partitions exist but the genome is in neither set', () => {
      it('keeps the original amount delta', () => {
        // Arrange
        const candidateGenome = createGenome({ mutationAmount: 3, mutationRate: 0.5 });
        const topGenome = createGenome({ mutationAmount: 3, mutationRate: 0.5, score: 1 });
        const bottomGenome = createGenome({ mutationAmount: 3, mutationRate: 0.5, score: 0 });

        // Act
        const amountDelta = applyTwoTierAmountDelta(
          -0.4,
          candidateGenome,
          2,
          new Set<Genome>([topGenome]),
          new Set<Genome>([bottomGenome]),
        );

        // Assert
        expect(amountDelta).toBe(-0.4);
      });
    });
  });

  describe('clampValue', () => {
    describe('given the value falls below the lower bound', () => {
      it('returns the minimum bound', () => {
        // Arrange
        const value = -0.2;

        // Act
        const clampedValue = clampValue(value, 0.1, 1);

        // Assert
        expect(clampedValue).toBe(0.1);
      });
    });

    describe('given the value exceeds the upper bound', () => {
      it('returns the maximum bound', () => {
        // Arrange
        const value = 1.2;

        // Act
        const clampedValue = clampValue(value, 0.1, 1);

        // Assert
        expect(clampedValue).toBe(1);
      });
    });
  });

  describe('shouldApplyTwoTierFallback', () => {
    describe('given the active strategy is not two-tier', () => {
      it('does not request the fallback rebalance', () => {
        // Arrange
        const mutationOutcome: MutationOutcome = {
          hasDecrease: false,
          hasIncrease: false,
        };

        // Act
        const shouldApplyFallback = shouldApplyTwoTierFallback(
          MUTATION_STRATEGY_ANNEAL,
          mutationOutcome,
        );

        // Assert
        expect(shouldApplyFallback).toBe(false);
      });
    });
  });

  describe('applyTwoTierFallback', () => {
    describe('given one population member lacks a mutation rate and the others exceed both clamp bounds', () => {
      it('skips the missing entry and clamps the remaining fallback adjustments', () => {
        // Arrange
        const population = [
          createGenome({ mutationRate: 0.95 }),
          createGenome({ mutationRate: null }),
          createGenome({ mutationRate: 0.15 }),
        ];
        const mutationSettings = createMutationSettings({
          maxRate: 1,
          minRate: 0.1,
          sigmaBase: 0.2,
        });

        // Act
        applyTwoTierFallback(population, mutationSettings);

        // Assert
        expect(population.map((genome) => genome._mutRate ?? null)).toEqual([
          1,
          null,
          0.1,
        ]);
      });
    });
  });
});