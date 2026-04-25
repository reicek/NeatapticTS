import {
  ensureDiversityStatsContainer,
  runEntropySharingTuning,
} from './evaluate.entropy-sharing';
import type { NeatControllerForEval } from '../shared/evaluate.types';

function createEvaluationController(input: {
  varEntropy: number;
  sharingSigma?: number;
}): NeatControllerForEval {
  return {
    options: {
      sharingSigma: input.sharingSigma ?? 3,
      entropySharingTuning: {
        enabled: true,
        targetEntropyVar: 0.2,
        adjustRate: 0.2,
        minSigma: 0.5,
        maxSigma: 10,
      },
    },
    population: [],
    fitness: async () => 0,
    _diversityStats: { varEntropy: input.varEntropy },
  };
}

describe('neat evaluate entropy-sharing chapter', () => {
  describe('ensureDiversityStatsContainer', () => {
    describe('given diversity stats storage is missing', () => {
      it('creates the container lazily', () => {
        // Arrange
        const evaluationController = {
          fitness: async () => 0,
          options: {},
          population: [],
        } as NeatControllerForEval;

        // Act
        ensureDiversityStatsContainer(evaluationController);

        // Assert
        expect(evaluationController._diversityStats).toEqual({});
      });
    });
  });

  describe('runEntropySharingTuning', () => {
    describe('given the observed entropy variance falls below the low band', () => {
      it('shrinks the sharing sigma for the next pass', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          varEntropy: 0.01,
        });

        // Act
        runEntropySharingTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.sharingSigma).toBeCloseTo(2.4);
      });
    });

    describe('given the observed entropy variance rises above the high band', () => {
      it('grows the sharing sigma for the next pass', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          varEntropy: 1,
        });

        // Act
        runEntropySharingTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.sharingSigma).toBeCloseTo(3.6);
      });
    });

    describe('given the observed entropy variance stays within the target band', () => {
      it('keeps the current sharing sigma unchanged', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          sharingSigma: 3,
          varEntropy: 0.2,
        });

        // Act
        runEntropySharingTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController.options.sharingSigma).toBe(3);
      });
    });
  });
});
