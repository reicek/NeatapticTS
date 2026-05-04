import { runFitnessEvaluation } from './evaluate.fitness';
import type {
  GenomeForEvaluation,
  NeatControllerForEval,
} from '../shared/evaluate.types';

type LineageAwareGenome = GenomeForEvaluation & {
  _parents: number[];
};

type ClearableGenome = GenomeForEvaluation & {
  clear: jest.Mock;
};

function createEvaluationController(): NeatControllerForEval {
  return {
    options: { fitnessPopulation: false },
    population: [
      { connections: [], _parents: [1, 2] },
      { connections: [], _parents: [3, 4] },
    ] as LineageAwareGenome[],
    fitness: async (genomeOrPopulation) => {
      if (Array.isArray(genomeOrPopulation)) return 0;

      return Array.isArray(genomeOrPopulation._parents)
        ? genomeOrPopulation._parents.length
        : 0;
    },
  };
}

function createPopulationEvaluationController(input?: { clear?: boolean }): {
  evaluationController: NeatControllerForEval;
  fitnessDelegate: jest.Mock;
  population: ClearableGenome[];
} {
  const population = [
    { connections: [], clear: jest.fn() },
    { connections: [], clear: jest.fn() },
  ] as ClearableGenome[];
  const fitnessDelegate = jest.fn(async () => 1);

  return {
    evaluationController: {
      options: {
        fitnessPopulation: true,
        clear: input?.clear ?? false,
      },
      population,
      fitness: fitnessDelegate,
    },
    fitnessDelegate,
    population,
  };
}

describe('neat evaluate fitness chapter', () => {
  describe('runFitnessEvaluation', () => {
    describe('given per-genome evaluation for genomes that already carry lineage metadata', () => {
      it('records numeric scores back onto each genome in place', async () => {
        // Arrange
        const evaluationController = createEvaluationController();

        // Act
        await runFitnessEvaluation(
          evaluationController,
          evaluationController.options,
        );

        // Assert
        expect(
          evaluationController.population.map((genome) => genome.score),
        ).toEqual([2, 2]);
      });
    });

    describe('given per-genome evaluation with the clear option enabled', () => {
      describe('when the genome exposes a clear method', () => {
        it('invokes clear on the genome before scoring it', async () => {
          // Arrange
          const clearFn = jest.fn();
          const evaluationController: NeatControllerForEval = {
            options: { fitnessPopulation: false, clear: true },
            population: [{ connections: [], clear: clearFn }],
            fitness: async () => 1,
          };

          // Act
          await runFitnessEvaluation(
            evaluationController,
            evaluationController.options,
          );

          // Assert
          expect(clearFn).toHaveBeenCalledTimes(1);
        });
      });
    });

    describe('given population-mode evaluation is enabled', () => {
      describe('when the delegate receives the full population together', () => {
        it('calls the fitness delegate once with the shared population array', async () => {
          // Arrange
          const { evaluationController, fitnessDelegate } =
            createPopulationEvaluationController();

          // Act
          await runFitnessEvaluation(
            evaluationController,
            evaluationController.options,
          );

          // Assert
          expect(fitnessDelegate.mock.calls).toEqual([
            [evaluationController.population],
          ]);
        });
      });
    });

    describe('given population-mode evaluation clears runtime state first', () => {
      describe('when the clear option is enabled', () => {
        it('invokes clear on every genome before the delegate runs', async () => {
          // Arrange
          const { evaluationController, population } =
            createPopulationEvaluationController({ clear: true });

          // Act
          await runFitnessEvaluation(
            evaluationController,
            evaluationController.options,
          );

          // Assert
          expect(
            population.map((genome) => genome.clear.mock.calls.length),
          ).toEqual([1, 1]);
        });
      });
    });
  });
});
