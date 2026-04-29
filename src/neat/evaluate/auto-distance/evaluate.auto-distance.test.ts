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

    describe('given the stored variance baseline is null', () => {
      it('treats the pass as a fresh bootstrap run and reseeds the moving baseline', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          connectionCounts: [1, 2, 3],
          lastConnVar: null,
        });

        // Act
        runAutoDistanceCoefficientTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect({
          disjointCoeff: evaluationController.options.disjointCoeff,
          excessCoeff: evaluationController.options.excessCoeff,
          lastConnVar: Number(
            (evaluationController._lastConnVar ?? 0).toFixed(6),
          ),
        }).toEqual({
          disjointCoeff: 1.05,
          excessCoeff: 1.05,
          lastConnVar: 0.666667,
        });
      });
    });

    describe('given auto-distance tuning is not active for the pass', () => {
      it('returns before changing coefficients or the stored variance baseline', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          connectionCounts: [1, 2, 3],
          lastConnVar: 4,
        });
        evaluationController.options.speciation = false;
        evaluationController.options.excessCoeff = 2;
        evaluationController.options.disjointCoeff = 3;

        // Act
        runAutoDistanceCoefficientTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect({
          disjointCoeff: evaluationController.options.disjointCoeff,
          excessCoeff: evaluationController.options.excessCoeff,
          lastConnVar: evaluationController._lastConnVar,
        }).toEqual({
          disjointCoeff: 3,
          excessCoeff: 2,
          lastConnVar: 4,
        });
      });
    });

    describe('given the evaluated population is empty', () => {
      it('reduces the connection statistics to zero before applying the variance policy', () => {
        // Arrange
        const evaluationController = createEvaluationController({
          connectionCounts: [],
          lastConnVar: 1,
        });

        // Act
        runAutoDistanceCoefficientTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect({
          disjointCoeff: evaluationController.options.disjointCoeff,
          excessCoeff: evaluationController.options.excessCoeff,
          lastConnVar: evaluationController._lastConnVar,
        }).toEqual({
          disjointCoeff: 1.05,
          excessCoeff: 1.05,
          lastConnVar: 0,
        });
      });
    });

    describe('given topology variance collapses below the stored band', () => {
      it('increases both distance coefficients through the explicit and default coefficient paths', () => {
        // Arrange
        const controllerWithExplicitExcess = createEvaluationController({
          connectionCounts: [2, 2, 2],
          lastConnVar: 10,
        });
        controllerWithExplicitExcess.options.autoDistanceCoeffTuning = {
          adjustRate: 0.1,
          enabled: true,
          maxCoeff: 8,
        };
        controllerWithExplicitExcess.options.excessCoeff = 7.9;

        const controllerWithExplicitDisjoint = createEvaluationController({
          connectionCounts: [2, 2, 2],
          lastConnVar: 10,
        });
        controllerWithExplicitDisjoint.options.autoDistanceCoeffTuning = {
          adjustRate: 0.1,
          enabled: true,
          maxCoeff: 8,
        };
        controllerWithExplicitDisjoint.options.disjointCoeff = 7.9;

        // Act
        runAutoDistanceCoefficientTuning(
          controllerWithExplicitExcess,
          controllerWithExplicitExcess.options,
        );
        runAutoDistanceCoefficientTuning(
          controllerWithExplicitDisjoint,
          controllerWithExplicitDisjoint.options,
        );

        // Assert
        expect({
          explicitDisjointPath: {
            disjointCoeff: controllerWithExplicitDisjoint.options.disjointCoeff,
            excessCoeff: controllerWithExplicitDisjoint.options.excessCoeff,
          },
          explicitExcessPath: {
            disjointCoeff: controllerWithExplicitExcess.options.disjointCoeff,
            excessCoeff: controllerWithExplicitExcess.options.excessCoeff,
          },
        }).toEqual({
          explicitDisjointPath: {
            disjointCoeff: 8,
            excessCoeff: 1.1,
          },
          explicitExcessPath: {
            disjointCoeff: 1.1,
            excessCoeff: 8,
          },
        });
      });
    });

    describe('given topology variance spreads above the stored band', () => {
      it('decreases both distance coefficients through the explicit and default coefficient paths', () => {
        // Arrange
        const controllerWithExplicitExcess = createEvaluationController({
          connectionCounts: [0, 0, 4],
          lastConnVar: 1,
        });
        controllerWithExplicitExcess.options.autoDistanceCoeffTuning = {
          adjustRate: 0.5,
          enabled: true,
          minCoeff: 0.05,
        };
        controllerWithExplicitExcess.options.excessCoeff = 0.06;

        const controllerWithExplicitDisjoint = createEvaluationController({
          connectionCounts: [0, 0, 4],
          lastConnVar: 1,
        });
        controllerWithExplicitDisjoint.options.autoDistanceCoeffTuning = {
          adjustRate: 0.5,
          enabled: true,
          minCoeff: 0.05,
        };
        controllerWithExplicitDisjoint.options.disjointCoeff = 0.06;

        // Act
        runAutoDistanceCoefficientTuning(
          controllerWithExplicitExcess,
          controllerWithExplicitExcess.options,
        );
        runAutoDistanceCoefficientTuning(
          controllerWithExplicitDisjoint,
          controllerWithExplicitDisjoint.options,
        );

        // Assert
        expect({
          explicitDisjointPath: {
            disjointCoeff: controllerWithExplicitDisjoint.options.disjointCoeff,
            excessCoeff: controllerWithExplicitDisjoint.options.excessCoeff,
          },
          explicitExcessPath: {
            disjointCoeff: controllerWithExplicitExcess.options.disjointCoeff,
            excessCoeff: controllerWithExplicitExcess.options.excessCoeff,
          },
        }).toEqual({
          explicitDisjointPath: {
            disjointCoeff: 0.05,
            excessCoeff: 0.5,
          },
          explicitExcessPath: {
            disjointCoeff: 0.5,
            excessCoeff: 0.05,
          },
        });
      });
    });
  });
});
