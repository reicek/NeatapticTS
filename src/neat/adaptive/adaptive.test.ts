import {
  applyAdaptiveMutation,
  applyAncestorUniqAdaptive,
  applyMinimalCriterionAdaptive,
  applyOperatorAdaptation,
} from './adaptive';
import type { NeatLikeWithAdaptive } from './core/adaptive.core.types';
import {
  applyRejection,
  collectScores,
  computeAcceptance,
  initializeThreshold,
  resolveTargetSettings,
  updateThreshold,
} from './acceptance/adaptive.minimal-criterion.utils';
import {
  applyUniquenessAdjustment,
  extractAncestorUniqueness,
  isCooldownSatisfied,
  resolveAdjustmentMagnitude,
  resolveUniquenessThresholds,
} from './lineage/adaptive.ancestor-uniqueness.utils';
import {
  applyMutationsToPopulation,
  applyTwoTierFallback,
  collectScoredGenomes,
  resolveMutationSettings,
  resolveRandomSource,
  shouldApplyTwoTierFallback,
  shouldAdaptThisGeneration,
  sortScoredGenomes,
  splitScoredGenomes,
} from './mutation/adaptive.mutation.utils';
import {
  applyOperatorDecay,
  collectOperatorStatsEntries,
  resolveOperatorDecay,
} from './mutation/adaptive.operator.utils';

jest.mock('./acceptance/adaptive.minimal-criterion.utils', () => ({
  applyRejection: jest.fn(),
  collectScores: jest.fn(),
  computeAcceptance: jest.fn(),
  initializeThreshold: jest.fn(),
  resolveTargetSettings: jest.fn(),
  updateThreshold: jest.fn(),
}));

jest.mock('./lineage/adaptive.ancestor-uniqueness.utils', () => ({
  applyUniquenessAdjustment: jest.fn(),
  extractAncestorUniqueness: jest.fn(),
  isCooldownSatisfied: jest.fn(),
  resolveAdjustmentMagnitude: jest.fn(),
  resolveUniquenessThresholds: jest.fn(),
}));

jest.mock('./mutation/adaptive.mutation.utils', () => ({
  applyMutationsToPopulation: jest.fn(),
  applyTwoTierFallback: jest.fn(),
  collectScoredGenomes: jest.fn(),
  resolveMutationSettings: jest.fn(),
  resolveRandomSource: jest.fn(),
  shouldAdaptThisGeneration: jest.fn(),
  shouldApplyTwoTierFallback: jest.fn(),
  sortScoredGenomes: jest.fn(),
  splitScoredGenomes: jest.fn(),
}));

jest.mock('./mutation/adaptive.operator.utils', () => ({
  applyOperatorDecay: jest.fn(),
  collectOperatorStatsEntries: jest.fn(),
  resolveOperatorDecay: jest.fn(),
}));

const mockedApplyRejection = jest.mocked(applyRejection);
const mockedCollectScores = jest.mocked(collectScores);
const mockedComputeAcceptance = jest.mocked(computeAcceptance);
const mockedInitializeThreshold = jest.mocked(initializeThreshold);
const mockedResolveTargetSettings = jest.mocked(resolveTargetSettings);
const mockedUpdateThreshold = jest.mocked(updateThreshold);
const mockedApplyUniquenessAdjustment = jest.mocked(applyUniquenessAdjustment);
const mockedExtractAncestorUniqueness = jest.mocked(extractAncestorUniqueness);
const mockedIsCooldownSatisfied = jest.mocked(isCooldownSatisfied);
const mockedResolveAdjustmentMagnitude = jest.mocked(resolveAdjustmentMagnitude);
const mockedResolveUniquenessThresholds = jest.mocked(resolveUniquenessThresholds);
const mockedApplyMutationsToPopulation = jest.mocked(applyMutationsToPopulation);
const mockedApplyTwoTierFallback = jest.mocked(applyTwoTierFallback);
const mockedCollectScoredGenomes = jest.mocked(collectScoredGenomes);
const mockedResolveMutationSettings = jest.mocked(resolveMutationSettings);
const mockedResolveRandomSource = jest.mocked(resolveRandomSource);
const mockedShouldApplyTwoTierFallback = jest.mocked(shouldApplyTwoTierFallback);
const mockedShouldAdaptThisGeneration = jest.mocked(shouldAdaptThisGeneration);
const mockedSortScoredGenomes = jest.mocked(sortScoredGenomes);
const mockedSplitScoredGenomes = jest.mocked(splitScoredGenomes);
const mockedApplyOperatorDecay = jest.mocked(applyOperatorDecay);
const mockedCollectOperatorStatsEntries = jest.mocked(
  collectOperatorStatsEntries,
);
const mockedResolveOperatorDecay = jest.mocked(resolveOperatorDecay);

type MinimalCriterionAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;
type AncestorUniqAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['ancestorUniqAdaptive']
>;
type AdaptiveMutationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['adaptiveMutation']
>;
type OperatorAdaptationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['operatorAdaptation']
>;

function createAdaptiveController(): NeatLikeWithAdaptive {
  return {
    options: {},
    population: [],
    input: 2,
    output: 1,
    generation: 0,
  };
}

describe('neat adaptive root chapter', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('applyMinimalCriterionAdaptive', () => {
    describe('given threshold initialization leaves the controller threshold undefined', () => {
      it('falls back to zero for acceptance and rejection calculations', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const minimalCriterionAdaptive: MinimalCriterionAdaptiveConfig = {
          enabled: true,
          targetAcceptance: 0.6,
          adjustRate: 0.1,
        };
        const tuning = {
          targetAcceptance: 0.6,
          adjustRate: 0.1,
        } as ReturnType<typeof resolveTargetSettings>;

        adaptiveController.options.minimalCriterionAdaptive =
          minimalCriterionAdaptive;
        mockedInitializeThreshold.mockImplementation(() => undefined);
        mockedCollectScores.mockReturnValue([0.2, 0.8]);
        mockedComputeAcceptance.mockReturnValue(0.5);
        mockedResolveTargetSettings.mockReturnValue(tuning);
        mockedUpdateThreshold.mockImplementation(() => undefined);

        // Act
        applyMinimalCriterionAdaptive.call(adaptiveController);

        // Assert
        expect({
          applyRejectionArgs: mockedApplyRejection.mock.calls[0],
          computeAcceptanceArgs: mockedComputeAcceptance.mock.calls[0],
        }).toEqual({
          applyRejectionArgs: [adaptiveController, 0],
          computeAcceptanceArgs: [[0.2, 0.8], 0],
        });
      });
    });
  });

  describe('applyAncestorUniqAdaptive', () => {
    describe('given the cooldown window has not elapsed yet', () => {
      it('returns before reading any lineage telemetry', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const ancestorUniqAdaptive: AncestorUniqAdaptiveConfig = {
          enabled: true,
          cooldown: 4,
          mode: 'epsilon',
        };

        adaptiveController.options.ancestorUniqAdaptive = ancestorUniqAdaptive;
        mockedIsCooldownSatisfied.mockReturnValue(false);

        // Act
        applyAncestorUniqAdaptive.call(adaptiveController);

        // Assert
        expect({
          adjustmentCalls: mockedApplyUniquenessAdjustment.mock.calls.length,
          extractCalls: mockedExtractAncestorUniqueness.mock.calls.length,
        }).toEqual({
          adjustmentCalls: 0,
          extractCalls: 0,
        });
      });
    });

    describe('given telemetry does not expose an ancestor uniqueness metric', () => {
      it('returns before applying any uniqueness adjustment', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const ancestorUniqAdaptive: AncestorUniqAdaptiveConfig = {
          enabled: true,
          cooldown: 2,
          mode: 'epsilon',
        };

        adaptiveController.options.ancestorUniqAdaptive = ancestorUniqAdaptive;
        mockedIsCooldownSatisfied.mockReturnValue(true);
        mockedExtractAncestorUniqueness.mockReturnValue(undefined);

        // Act
        applyAncestorUniqAdaptive.call(adaptiveController);

        // Assert
        expect({
          adjustmentCalls: mockedApplyUniquenessAdjustment.mock.calls.length,
          extractCalls: mockedExtractAncestorUniqueness.mock.calls.length,
        }).toEqual({
          adjustmentCalls: 0,
          extractCalls: 1,
        });
      });
    });

    describe('given telemetry exposes an ancestor uniqueness metric after cooldown', () => {
      it('resolves thresholds and delegates the adjustment with the computed values', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const ancestorUniqAdaptive: AncestorUniqAdaptiveConfig = {
          enabled: true,
          cooldown: 2,
          adjust: 0.15,
          lowThreshold: 0.2,
          highThreshold: 0.6,
          mode: 'epsilon',
        };
        const thresholds = {
          lowThreshold: 0.2,
          highThreshold: 0.6,
        } as ReturnType<
          typeof resolveUniquenessThresholds
        >;

        adaptiveController.options.ancestorUniqAdaptive = ancestorUniqAdaptive;
        mockedIsCooldownSatisfied.mockReturnValue(true);
        mockedExtractAncestorUniqueness.mockReturnValue(0.1);
        mockedResolveUniquenessThresholds.mockReturnValue(thresholds);
        mockedResolveAdjustmentMagnitude.mockReturnValue(0.15);

        // Act
        applyAncestorUniqAdaptive.call(adaptiveController);

        // Assert
        expect({
          adjustmentArgs: mockedApplyUniquenessAdjustment.mock.calls[0],
          resolveAdjustmentArgs: mockedResolveAdjustmentMagnitude.mock.calls[0],
          resolveThresholdArgs: mockedResolveUniquenessThresholds.mock.calls[0],
        }).toEqual({
          adjustmentArgs: [
            adaptiveController,
            ancestorUniqAdaptive,
            0.1,
            thresholds,
            0.15,
          ],
          resolveAdjustmentArgs: [ancestorUniqAdaptive],
          resolveThresholdArgs: [ancestorUniqAdaptive],
        });
      });
    });
  });

  describe('applyAdaptiveMutation', () => {
    describe('given the current generation does not satisfy the adaptation cadence', () => {
      it('returns before collecting scored genomes', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const adaptiveMutation: AdaptiveMutationConfig = {
          enabled: true,
          adaptEvery: 5,
        };

        adaptiveController.options.adaptiveMutation = adaptiveMutation;
        adaptiveController.generation = 3;
        mockedShouldAdaptThisGeneration.mockReturnValue(false);

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect({
          collectScoredGenomesCalls: mockedCollectScoredGenomes.mock.calls.length,
          shouldAdaptArgs: mockedShouldAdaptThisGeneration.mock.calls[0],
        }).toEqual({
          collectScoredGenomesCalls: 0,
          shouldAdaptArgs: [3, adaptiveMutation],
        });
      });
    });

    describe('given the current generation satisfies the adaptation cadence', () => {
      it('partitions scored genomes, applies mutations, and triggers the fallback when requested', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const adaptiveMutation: AdaptiveMutationConfig = {
          enabled: true,
          adaptEvery: 1,
          strategy: 'twoTier',
        };
        const scoredGenomes = [{ score: 3 }, { score: 1 }] as ReturnType<
          typeof collectScoredGenomes
        >;
        const sortedScoredGenomes = [{ score: 3 }, { score: 1 }] as ReturnType<
          typeof sortScoredGenomes
        >;
        const partitions = {
          topHalf: [{ score: 3 }],
          bottomHalf: [{ score: 1 }],
        } as ReturnType<typeof splitScoredGenomes>;
        const mutationSettings = {
          strategy: 'twoTier',
        } as ReturnType<typeof resolveMutationSettings>;
        const randomSource = () => 0.25;
        const mutationOutcome = {
          hasIncrease: false,
          hasDecrease: true,
        } as ReturnType<typeof applyMutationsToPopulation>;

        adaptiveController.options.adaptiveMutation = adaptiveMutation;
        adaptiveController.generation = 4;
        adaptiveController.population = [{ score: 1 }, { score: 3 }];
        mockedShouldAdaptThisGeneration.mockReturnValue(true);
        mockedCollectScoredGenomes.mockReturnValue(scoredGenomes);
        mockedSortScoredGenomes.mockReturnValue(sortedScoredGenomes);
        mockedSplitScoredGenomes.mockReturnValue(partitions);
        mockedResolveMutationSettings.mockReturnValue(mutationSettings);
        mockedResolveRandomSource.mockReturnValue(randomSource);
        mockedApplyMutationsToPopulation.mockReturnValue(mutationOutcome);
        mockedShouldApplyTwoTierFallback.mockReturnValue(true);

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect({
          applyMutationArgs: mockedApplyMutationsToPopulation.mock.calls[0],
          fallbackArgs: mockedApplyTwoTierFallback.mock.calls[0],
          settingsArgs: mockedResolveMutationSettings.mock.calls[0],
        }).toEqual({
          applyMutationArgs: [
            adaptiveController.population,
            partitions,
            mutationSettings,
            randomSource,
          ],
          fallbackArgs: [adaptiveController.population, mutationSettings],
          settingsArgs: [adaptiveController, adaptiveMutation],
        });
      });
    });

    describe('given the current generation satisfies the adaptation cadence but the fallback is unnecessary', () => {
      it('returns without applying the two-tier fallback', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const adaptiveMutation: AdaptiveMutationConfig = {
          enabled: true,
          adaptEvery: 1,
          strategy: 'twoTier',
        };
        const scoredGenomes = [{ score: 4 }, { score: 2 }] as ReturnType<
          typeof collectScoredGenomes
        >;
        const sortedScoredGenomes = [{ score: 2 }, { score: 4 }] as ReturnType<
          typeof sortScoredGenomes
        >;
        const partitions = {
          topHalf: [{ score: 4 }],
          bottomHalf: [{ score: 2 }],
        } as ReturnType<typeof splitScoredGenomes>;
        const mutationSettings = {
          strategy: 'twoTier',
        } as ReturnType<typeof resolveMutationSettings>;
        const randomSource = () => 0.5;
        const mutationOutcome = {
          hasIncrease: true,
          hasDecrease: true,
        } as ReturnType<typeof applyMutationsToPopulation>;

        adaptiveController.options.adaptiveMutation = adaptiveMutation;
        adaptiveController.generation = 8;
        adaptiveController.population = [{ score: 2 }, { score: 4 }];
        mockedShouldAdaptThisGeneration.mockReturnValue(true);
        mockedCollectScoredGenomes.mockReturnValue(scoredGenomes);
        mockedSortScoredGenomes.mockReturnValue(sortedScoredGenomes);
        mockedSplitScoredGenomes.mockReturnValue(partitions);
        mockedResolveMutationSettings.mockReturnValue(mutationSettings);
        mockedResolveRandomSource.mockReturnValue(randomSource);
        mockedApplyMutationsToPopulation.mockReturnValue(mutationOutcome);
        mockedShouldApplyTwoTierFallback.mockReturnValue(false);

        // Act
        applyAdaptiveMutation.call(adaptiveController);

        // Assert
        expect({
          applyMutationArgs: mockedApplyMutationsToPopulation.mock.calls[0],
          fallbackCalls: mockedApplyTwoTierFallback.mock.calls.length,
        }).toEqual({
          applyMutationArgs: [
            adaptiveController.population,
            partitions,
            mutationSettings,
            randomSource,
          ],
          fallbackCalls: 0,
        });
      });
    });
  });

  describe('applyOperatorAdaptation', () => {
    describe('given operator adaptation is disabled', () => {
      it('returns before resolving decay settings', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const operatorAdaptation: OperatorAdaptationConfig = {
          enabled: false,
          decay: 0.8,
        };

        adaptiveController.options.operatorAdaptation = operatorAdaptation;

        // Act
        applyOperatorAdaptation.call(adaptiveController);

        // Assert
        expect({
          applyDecayCalls: mockedApplyOperatorDecay.mock.calls.length,
          collectEntriesCalls: mockedCollectOperatorStatsEntries.mock.calls.length,
          resolveDecayCalls: mockedResolveOperatorDecay.mock.calls.length,
        }).toEqual({
          applyDecayCalls: 0,
          collectEntriesCalls: 0,
          resolveDecayCalls: 0,
        });
      });
    });

    describe('given operator adaptation is enabled but no operator statistics exist yet', () => {
      it('returns before collecting entries or applying decay', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const operatorAdaptation: OperatorAdaptationConfig = {
          enabled: true,
          decay: 0.75,
        };

        adaptiveController.options.operatorAdaptation = operatorAdaptation;

        // Act
        applyOperatorAdaptation.call(adaptiveController);

        // Assert
        expect({
          applyDecayCalls: mockedApplyOperatorDecay.mock.calls.length,
          collectEntriesCalls: mockedCollectOperatorStatsEntries.mock.calls.length,
          resolveDecayCalls: mockedResolveOperatorDecay.mock.calls.length,
        }).toEqual({
          applyDecayCalls: 0,
          collectEntriesCalls: 0,
          resolveDecayCalls: 0,
        });
      });
    });

    describe('given operator adaptation is enabled and operator statistics are present', () => {
      it('delegates decay using the resolved factor and collected entries', () => {
        // Arrange
        const adaptiveController = createAdaptiveController();
        const operatorAdaptation: OperatorAdaptationConfig = {
          enabled: true,
          decay: 0.7,
        };
        const operatorStats = new Map([
          ['addNode', { success: 3, attempts: 5 }],
        ]);
        const statsEntries = Array.from(operatorStats.entries()) as ReturnType<
          typeof collectOperatorStatsEntries
        >;

        adaptiveController.options.operatorAdaptation = operatorAdaptation;
        adaptiveController._operatorStats = operatorStats;
        mockedResolveOperatorDecay.mockReturnValue(0.7);
        mockedCollectOperatorStatsEntries.mockReturnValue(statsEntries);

        // Act
        applyOperatorAdaptation.call(adaptiveController);

        // Assert
        expect({
          applyDecayArgs: mockedApplyOperatorDecay.mock.calls[0],
          collectEntriesArgs: mockedCollectOperatorStatsEntries.mock.calls[0],
          resolveDecayArgs: mockedResolveOperatorDecay.mock.calls[0],
        }).toEqual({
          applyDecayArgs: [operatorStats, statsEntries, 0.7],
          collectEntriesArgs: [operatorStats],
          resolveDecayArgs: [operatorAdaptation],
        });
      });
    });
  });
});