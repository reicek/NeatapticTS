import {
  ADJUST_RATE_DEFAULT,
  TARGET_ACCEPTANCE_DEFAULT,
} from '../core/adaptive.core.constants';
import type { NeatLikeWithAdaptive } from '../core/adaptive.core.types';
import {
  applyRejection,
  collectScores,
  computeAcceptance,
  initializeThreshold,
  resolveTargetSettings,
  updateThreshold,
} from './adaptive.minimal-criterion.utils';

type AcceptanceConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;

function createAcceptanceController(
  scoreSnapshot: Array<number | undefined>,
  minimalCriterionAdaptive: AcceptanceConfig = {},
): NeatLikeWithAdaptive {
  return {
    options: { minimalCriterionAdaptive },
    population: scoreSnapshot.map((score) => ({ score })),
    input: 2,
    output: 1,
    generation: 0,
  } as NeatLikeWithAdaptive;
}

describe('adaptive minimal criterion utility chapter', () => {
  describe('initializeThreshold', () => {
    describe('given the controller already has a threshold', () => {
      it('keeps the existing threshold unchanged', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([], {
          initialThreshold: 0.1,
        });
        adaptiveController._mcThreshold = 0.4;

        // Act
        initializeThreshold(adaptiveController, {
          initialThreshold: 0.1,
        });

        // Assert
        expect(adaptiveController._mcThreshold).toBe(0.4);
      });
    });

    describe('given the controller has no threshold and no configured initial value', () => {
      it('seeds the threshold at zero', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([]);

        // Act
        initializeThreshold(adaptiveController, {});

        // Assert
        expect(adaptiveController._mcThreshold).toBe(0);
      });
    });
  });

  describe('collectScores', () => {
    describe('given genomes have missing controller-visible scores', () => {
      it('replaces the missing scores with zero in the snapshot', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([
          0.25,
          undefined,
          0.75,
        ]);

        // Act
        const scoreSnapshot = collectScores(adaptiveController);

        // Assert
        expect(scoreSnapshot).toEqual([0.25, 0, 0.75]);
      });
    });
  });

  describe('computeAcceptance', () => {
    describe('given no scores are available', () => {
      it('returns zero acceptance', () => {
        // Arrange
        const scoreSnapshot: number[] = [];

        // Act
        const acceptance = computeAcceptance(scoreSnapshot, 0.3);

        // Assert
        expect(acceptance).toBe(0);
      });
    });
  });

  describe('resolveTargetSettings', () => {
    describe('given the adaptive config omits both tuning values', () => {
      it('returns the default target acceptance and adjust rate', () => {
        // Arrange
        const minimalCriterionAdaptive = {};

        // Act
        const targetSettings = resolveTargetSettings(minimalCriterionAdaptive);

        // Assert
        expect(targetSettings).toEqual({
          adjustRate: ADJUST_RATE_DEFAULT,
          targetAcceptance: TARGET_ACCEPTANCE_DEFAULT,
        });
      });
    });
  });

  describe('updateThreshold', () => {
    describe('given acceptance exceeds the target band before a threshold has been initialized', () => {
      it('falls back to zero and keeps the threshold at zero', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([]);

        // Act
        updateThreshold(adaptiveController, 0.9, {
          adjustRate: 0.25,
          targetAcceptance: 0.5,
        });

        // Assert
        expect(adaptiveController._mcThreshold).toBe(0);
      });
    });

    describe('given acceptance falls below the target band', () => {
      it('decreases the persisted threshold', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([]);
        adaptiveController._mcThreshold = 0.4;

        // Act
        updateThreshold(adaptiveController, 0.2, {
          adjustRate: 0.25,
          targetAcceptance: 0.5,
        });

        // Assert
        expect(adaptiveController._mcThreshold).toBeCloseTo(0.3, 12);
      });
    });

    describe('given acceptance falls below the target band before a threshold has been initialized', () => {
      it('falls back to zero and keeps the threshold at zero', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([]);

        // Act
        updateThreshold(adaptiveController, 0.2, {
          adjustRate: 0.25,
          targetAcceptance: 0.5,
        });

        // Assert
        expect(adaptiveController._mcThreshold).toBe(0);
      });
    });

    describe('given acceptance stays inside the target band', () => {
      it('leaves the threshold unchanged', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([]);
        adaptiveController._mcThreshold = 0.4;

        // Act
        updateThreshold(adaptiveController, 0.5, {
          adjustRate: 0.25,
          targetAcceptance: 0.5,
        });

        // Assert
        expect(adaptiveController._mcThreshold).toBe(0.4);
      });
    });
  });

  describe('applyRejection', () => {
    describe('given genomes fall on both sides of the final threshold', () => {
      it('rewrites only the below-threshold controller-visible scores to zero', () => {
        // Arrange
        const adaptiveController = createAcceptanceController([
          0.2,
          0.5,
          undefined,
        ]);

        // Act
        applyRejection(adaptiveController, 0.3);

        // Assert
        expect(
          adaptiveController.population.map((genome) => genome.score),
        ).toEqual([0, 0.5, 0]);
      });
    });
  });
});
