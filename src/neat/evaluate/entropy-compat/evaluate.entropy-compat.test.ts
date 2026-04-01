import { runEntropyCompatibilityTuning } from './evaluate.entropy-compat';
import type { NeatControllerForEval } from '../shared/evaluate.types';

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
  });
});
