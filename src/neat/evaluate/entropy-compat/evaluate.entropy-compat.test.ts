import { runEntropyCompatibilityTuning } from './evaluate.entropy-compat';
import type { NeatControllerForEval } from '../shared/evaluate.types';
import {
  COMPAT_THRESHOLD_DEFAULT,
  ENTROPY_TARGET_DEFAULT,
} from '../shared/evaluate.constants';

function createEvaluationController(input: {
  meanEntropy: number;
  compatibilityThreshold?: number;
}): NeatControllerForEval {
  return {
    options: {
      compatibilityThreshold: input.compatibilityThreshold ?? 3,
      entropyCompatTuning: {
        enabled: true,
        targetEntropy: 0.5,
        adjustRate: 0.2,
        deadband: 0.05,
        minThreshold: 0.5,
        maxThreshold: 6,
      },
    },
    population: [],
    fitness: async () => 0,
    _diversityStats: { meanEntropy: input.meanEntropy },
  };
}

describe('neat evaluate entropy-compat chapter', () => {
  describe('runEntropyCompatibilityTuning', () => {
    describe('given the observed mean entropy falls below the target band', () => {
      it('tightens the compatibility threshold for the next pass', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          meanEntropy: 0.1,
        });

        // Act
        runEntropyCompatibilityTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.compatibilityThreshold).toBeCloseTo(
          2.4,
        );
      });
    });

    describe('given the observed mean entropy rises above the target band', () => {
      it('widens the compatibility threshold for the next pass', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          meanEntropy: 1,
        });

        // Act
        runEntropyCompatibilityTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.compatibilityThreshold).toBeCloseTo(
          3.6,
        );
      });
    });

    describe('given entropy-compatibility tuning is disabled', () => {
      it('leaves the compatibility threshold unchanged', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          compatibilityThreshold: 2.7,
          meanEntropy: 0.1,
        });
        evaluationController.options.entropyCompatTuning = { enabled: false };

        // Act
        runEntropyCompatibilityTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.compatibilityThreshold).toBe(2.7);
      });
    });

    describe('given the diversity stats omit mean entropy', () => {
      it('leaves the compatibility threshold unchanged', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          compatibilityThreshold: 2.9,
          meanEntropy: 0.5,
        });
        Reflect.set(evaluationController, '_diversityStats', undefined);

        // Act
        runEntropyCompatibilityTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.compatibilityThreshold).toBe(2.9);
      });
    });

    describe('given tuning uses only shared defaults and mean entropy stays inside the deadband', () => {
      it('falls back to the shared default compatibility threshold', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          meanEntropy: ENTROPY_TARGET_DEFAULT,
        });
        evaluationController.options.compatibilityThreshold = undefined;
        evaluationController.options.entropyCompatTuning = { enabled: true };

        // Act
        runEntropyCompatibilityTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.compatibilityThreshold).toBe(
          COMPAT_THRESHOLD_DEFAULT,
        );
      });
    });
  });
});
