import {
  applyAdaptiveMutation,
  applyMutationsToPopulation,
  applyOperatorDecay as applyOperatorDecayFromFacade,
  applyTwoTierFallback,
  resolveMutationSettings,
  resolveOperatorDecay,
  shouldAdaptThisGeneration,
  shouldApplyTwoTierFallback,
} from './adaptive.mutation';
import {
  applyOperatorDecay,
  collectOperatorStatsEntries,
} from './adaptive.operator.utils';
import type {
  AdaptiveMutationConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

function createMutationController(input: {
  scores: number[];
  mutationRates: number[];
  adaptiveMutation: AdaptiveMutationConfig;
  mutationAmounts?: Array<number | undefined>;
  mutationAmountDefault?: number;
  randomValue: number;
}): NeatLikeWithAdaptive {
  return {
    options: {
      adaptiveMutation: input.adaptiveMutation,
      mutationAmount: input.mutationAmountDefault,
    },
    population: input.scores.map((score, genomeIndex) => ({
      score,
      _mutRate: input.mutationRates[genomeIndex],
      _mutAmount: input.mutationAmounts?.[genomeIndex],
    })),
    input: 3,
    output: 1,
    generation: 0,
    _getRNG: () => () => input.randomValue,
  };
}

function roundNumericValue(
  value: number | null | undefined,
): number | undefined {
  return typeof value === 'number' ? Number(value.toFixed(2)) : undefined;
}

describe('neat adaptive mutation chapter', () => {
  describe('adaptive.mutation facade re-exports', () => {
    it('exports applyAdaptiveMutation as a function', () => {
      expect(typeof applyAdaptiveMutation).toBe('function');
    });

    it('exports shouldAdaptThisGeneration as a function', () => {
      expect(typeof shouldAdaptThisGeneration).toBe('function');
    });

    it('exports resolveMutationSettings as a function', () => {
      expect(typeof resolveMutationSettings).toBe('function');
    });

    it('exports applyMutationsToPopulation as a function', () => {
      expect(typeof applyMutationsToPopulation).toBe('function');
    });

    it('exports shouldApplyTwoTierFallback as a function', () => {
      expect(typeof shouldApplyTwoTierFallback).toBe('function');
    });

    it('exports applyTwoTierFallback as a function', () => {
      expect(typeof applyTwoTierFallback).toBe('function');
    });

    it('exports resolveOperatorDecay as a function', () => {
      expect(typeof resolveOperatorDecay).toBe('function');
    });

    it('exports applyOperatorDecay as a function', () => {
      expect(typeof applyOperatorDecayFromFacade).toBe('function');
    });
  });

  describe('applyAdaptiveMutation', () => {
    describe('given the two-tier strategy with scored top and bottom halves', () => {
      it('pushes the lower-scoring genomes above the baseline rate and the higher-scoring genomes below it', () => {
        // Arrange
        const adaptiveController = createMutationController({
          scores: [1, 2, 3, 4],
          mutationRates: [0.5, 0.5, 0.5, 0.5],
          adaptiveMutation: {
            enabled: true,
            initialRate: 0.5,
            sigma: 0.1,
            strategy: 'twoTier',
            adaptEvery: 1,
            minRate: 0.01,
            maxRate: 1,
          },
          randomValue: 1,
        });

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect(
          adaptiveController.population.map((genome) =>
            roundNumericValue(genome._mutRate),
          ),
        ).toEqual([0.65, 0.65, 0.35, 0.35]);
      });
    });

    describe('given the two-tier strategy with only one genome', () => {
      it('rebalances the one-sided pressure through the fallback step', () => {
        // Arrange
        const adaptiveController = createMutationController({
          scores: [1],
          mutationRates: [0.5],
          adaptiveMutation: {
            enabled: true,
            strategy: 'twoTier',
            initialRate: 0.4,
            sigma: 0.2,
            minRate: 0.01,
            maxRate: 1,
          },
          randomValue: 1,
        });

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect(
          adaptiveController.population.map((genome) =>
            roundNumericValue(genome._mutRate),
          ),
        ).toEqual([0.5]);
      });
    });

    describe('given two-tier mutation with amount adaptation enabled', () => {
      it('keeps mutation amounts inside the configured bounds while pushing weaker genomes upward', () => {
        // Arrange
        const adaptiveController = createMutationController({
          scores: [1, 2, 3, 4],
          mutationRates: [0.5, 0.5, 0.5, 0.5],
          adaptiveMutation: {
            enabled: true,
            strategy: 'twoTier',
            initialRate: 0.5,
            sigma: 0.15,
            adaptAmount: true,
            amountSigma: 0.6,
            minAmount: 1,
            maxAmount: 6,
          },
          mutationAmountDefault: 3,
          randomValue: 1,
        });

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect(
          adaptiveController.population.map((genome) => genome._mutAmount),
        ).toEqual([4, 4, 2, 2]);
      });
    });

    describe('given the explore-low strategy with scored bottom and top halves', () => {
      it('gives the lower-scoring half a larger positive rate delta than the higher-scoring half', () => {
        // Arrange
        const adaptiveController = createMutationController({
          scores: [1, 2, 3, 4],
          mutationRates: [0.4, 0.4, 0.4, 0.4],
          adaptiveMutation: {
            enabled: true,
            strategy: 'exploreLow',
            initialRate: 0.4,
            sigma: 0.2,
            minRate: 0.01,
            maxRate: 1,
          },
          randomValue: 1,
        });

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect(
          adaptiveController.population.map((genome) =>
            roundNumericValue(genome._mutRate),
          ),
        ).toEqual([0.85, 0.85, 0.25, 0.25]);
      });
    });

    describe('given the anneal strategy late in a run', () => {
      it('shrinks the per-generation rate delta while keeping the result inside the configured bounds', () => {
        // Arrange
        const adaptiveController = createMutationController({
          scores: [1, 2, 3, 4],
          mutationRates: [0.6, 0.6, 0.6, 0.6],
          adaptiveMutation: {
            enabled: true,
            strategy: 'anneal',
            initialRate: 0.6,
            sigma: 0.3,
            minRate: 0.01,
            maxRate: 1,
          },
          randomValue: 1,
        });
        adaptiveController.generation = 40;

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect(
          adaptiveController.population.map((genome) =>
            roundNumericValue(genome._mutRate),
          ),
        ).toEqual([0.72, 0.72, 0.72, 0.72]);
      });
    });
  });

  describe('applyOperatorDecay', () => {
    describe('given tracked operator success and attempt counts', () => {
      it('scales both counters by the configured decay factor', () => {
        // Arrange
        const operatorStats = new Map([
          ['ADD_NODE', { success: 4, attempts: 6 }],
          ['SUB_CONN', { success: 2, attempts: 8 }],
        ]);
        const operatorEntries = collectOperatorStatsEntries(operatorStats);

        // Act
        applyOperatorDecay(operatorStats, operatorEntries, 0.5);

        // Assert
        expect(Array.from(operatorStats.entries())).toEqual([
          ['ADD_NODE', { success: 2, attempts: 3 }],
          ['SUB_CONN', { success: 1, attempts: 4 }],
        ]);
      });
    });
  });
});
