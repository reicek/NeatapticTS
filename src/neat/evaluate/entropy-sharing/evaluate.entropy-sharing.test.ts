import {
  ensureDiversityStatsContainer,
  runEntropySharingTuning,
} from './evaluate.entropy-sharing';
import { ENTROPY_VAR_MIN_SIGMA_DEFAULT } from '../shared/evaluate.constants';
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

    describe('given diversity stats storage already exists', () => {
      it('leaves the existing container unchanged', () => {
        // Arrange
        const evaluationController = {
          fitness: async () => 0,
          options: {},
          population: [],
          _diversityStats: { varEntropy: 0.5 },
        } as NeatControllerForEval;

        // Act
        ensureDiversityStatsContainer(evaluationController);

        // Assert
        expect(evaluationController._diversityStats).toEqual({
          varEntropy: 0.5,
        });
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

    describe('given entropy sharing tuning is disabled', () => {
      it('returns early without adjusting sigma', () => {
        // Arrange
        const evaluationController = {
          fitness: async () => 0,
          options: {
            entropySharingTuning: { enabled: false },
            sharingSigma: 5,
          },
          population: [],
          _diversityStats: { varEntropy: 0.01 },
        } as NeatControllerForEval;

        // Act
        runEntropySharingTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert: early return → sigma unchanged
        expect(evaluationController.options.sharingSigma).toBe(5);
      });
    });

    describe('given variance entropy in diversity stats is not numeric', () => {
      it('returns early without adjusting sigma', () => {
        // Arrange
        const evaluationController = {
          fitness: async () => 0,
          options: { entropySharingTuning: { enabled: true }, sharingSigma: 5 },
          population: [],
          _diversityStats: {},
        } as NeatControllerForEval;

        // Act
        runEntropySharingTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert: early return → sigma unchanged
        expect(evaluationController.options.sharingSigma).toBe(5);
      });
    });

    describe('given sharingSigma is absent and tuning options use all defaults', () => {
      it('falls back to zero sigma and clamps to the default minimum', () => {
        // Arrange: no sharingSigma, entropySharingTuning has only enabled flag
        const evaluationController = {
          fitness: async () => 0,
          options: { entropySharingTuning: { enabled: true } },
          population: [],
          _diversityStats: { varEntropy: 0.001 },
        } as NeatControllerForEval;

        // Act: varEntropy=0.001 < targetVar(0.2)*LOW_BAND(0.9)=0.18 → shrink path
        // currentSigma ?? 0 = 0; result = Math.max(minSigma_default, 0 * 0.9) = 0.1
        runEntropySharingTuning(
          evaluationController,
          evaluationController.options,
        );

        // Assert: clamped to default minimum sigma
        expect(evaluationController.options.sharingSigma).toBe(
          ENTROPY_VAR_MIN_SIGMA_DEFAULT,
        );
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
