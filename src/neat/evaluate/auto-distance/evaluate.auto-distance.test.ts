import { runAutoDistanceCoefficientTuning } from './evaluate.auto-distance';
import type { NeatControllerForEval } from '../shared/evaluate.types';

function createEvaluationController(input: {
  connectionCounts: number[];
  lastConnVar?: number | null;
}): NeatControllerForEval {
  return {
    options: {
      speciation: true,
      autoDistanceCoeffTuning: { enabled: true },
    },
    population: input.connectionCounts.map((connectionCount) => ({
      connections: Array.from({ length: connectionCount }, () => ({})),
    })),
    fitness: async () => 0,
    _lastConnVar: input.lastConnVar,
  };
}

describe('neat evaluate auto-distance chapter', () => {
  describe('runAutoDistanceCoefficientTuning', () => {
    describe('given the first enabled speciation pass', () => {
      it('bootstraps both structural-distance coefficients upward once', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          connectionCounts: [1, 2, 3],
        });

        // Act
        runAutoDistanceCoefficientTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect([
          evaluationController.options.excessCoeff,
          evaluationController.options.disjointCoeff,
        ]).toEqual([1.05, 1.05]);
      });
    });

    describe('given the first enabled speciation pass with fresh topology sizes', () => {
      it('stores the observed connection variance as the moving baseline', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          connectionCounts: [1, 2, 3],
        });

        // Act
        runAutoDistanceCoefficientTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(evaluationController._lastConnVar).toBeCloseTo(2 / 3);
      });
    });
  });
});
