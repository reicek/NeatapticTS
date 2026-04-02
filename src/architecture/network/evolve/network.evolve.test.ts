import { Architect, Network } from '../../../neataptic';
import {
  NetworkEvolveDatasetCompatibilityError,
  NetworkEvolveStoppingConditionRequiredError,
} from './network.evolve.errors';

type TrainingSet = Parameters<Network['evolve']>[0];
type EvolutionSummary = Awaited<ReturnType<Network['evolve']>>;

jest.setTimeout(10000);

const xorTrainingSet: TrainingSet = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

describe('network evolve chapter', () => {
  describe('Network.evolve()', () => {
    describe('given the dataset is empty', () => {
      describe('when evolve() is called', () => {
        it('rejects with the dataset compatibility error type', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolvePromise = network.evolve([], {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
          });

          // Assert
          await expect(evolvePromise).rejects.toThrow(
            NetworkEvolveDatasetCompatibilityError,
          );
        });
      });
    });

    describe('given the dataset input size does not match the network', () => {
      describe('when evolve() is called', () => {
        it('rejects with the dataset compatibility error type', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);
          const invalidTrainingSet: TrainingSet = [
            { input: [0], output: [0] },
            { input: [1], output: [1] },
          ];

          // Act
          const evolvePromise = network.evolve(invalidTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
          });

          // Assert
          await expect(evolvePromise).rejects.toThrow(
            NetworkEvolveDatasetCompatibilityError,
          );
        });
      });
    });

    describe('given the dataset output size does not match the network', () => {
      describe('when evolve() is called', () => {
        it('rejects with the dataset compatibility error type', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);
          const invalidTrainingSet: TrainingSet = [
            { input: [0, 1], output: [0, 1] },
            { input: [1, 0], output: [1, 0] },
          ];

          // Act
          const evolvePromise = network.evolve(invalidTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
          });

          // Assert
          await expect(evolvePromise).rejects.toThrow(
            NetworkEvolveDatasetCompatibilityError,
          );
        });
      });
    });

    describe('given no stopping condition is provided', () => {
      describe('when evolve() is called', () => {
        it('rejects with the stopping-condition-required error type', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolvePromise = network.evolve(xorTrainingSet, {});

          // Assert
          await expect(evolvePromise).rejects.toThrow(
            NetworkEvolveStoppingConditionRequiredError,
          );
        });
      });
    });

    describe('given a reachable dataset and bounded stop conditions', () => {
      let network: Network;
      let evolutionSummary: EvolutionSummary;

      beforeEach(async () => {
        // Arrange
        network = Architect.perceptron(2, 4, 1);

        // Act
        evolutionSummary = await network.evolve(xorTrainingSet, {
          iterations: 2,
          error: 0.5,
          amount: 1,
          threads: 1,
          popsize: 2,
        });
      });

      describe('when reading the error field', () => {
        it('returns a numeric error summary', () => {
          // Assert
          expect(typeof evolutionSummary.error).toBe('number');
        });
      });

      describe('when reading the iterations field', () => {
        it('stays within the configured maximum', () => {
          // Assert
          expect(evolutionSummary.iterations).toBeLessThanOrEqual(2);
        });
      });

      describe('when reading the time field', () => {
        it('returns a numeric elapsed time', () => {
          // Assert
          expect(typeof evolutionSummary.time).toBe('number');
        });
      });
    });

    describe('given only an error stop condition is provided', () => {
      describe('when evolve() is called', () => {
        it('returns zero iterations', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolutionSummary = await network.evolve(xorTrainingSet, {
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
          });

          // Assert
          expect(evolutionSummary.iterations).toBe(0);
        });
      });
    });

    describe('given only an iterations stop condition is provided', () => {
      describe('when evolve() is called', () => {
        it('performs exactly one generation when iterations is one', async () => {
          // Arrange
          const network = new Network(1, 1, { seed: 471 });
          const trainingSet: TrainingSet = [{ input: [0.1], output: [0.2] }];

          // Act
          const evolutionSummary = await network.evolve(trainingSet, {
            iterations: 1,
            popsize: 6,
          });

          // Assert
          expect(evolutionSummary.iterations).toBe(1);
        });
      });
    });

    describe('given a schedule callback is configured', () => {
      describe('when evolve() is called', () => {
        it('invokes the schedule function at the configured interval', async () => {
          // Arrange
          const network = new Network(1, 1, { seed: 472 });
          const trainingSet: TrainingSet = [{ input: [0.9], output: [0.1] }];
          const scheduleSpy = jest.fn();

          // Act
          await network.evolve(trainingSet, {
            iterations: 1,
            schedule: { iterations: 1, function: scheduleSpy },
            log: 1,
          });

          // Assert
          expect(scheduleSpy).toHaveBeenCalledTimes(1);
        });
      });
    });

    describe('given POWER selection is requested', () => {
      describe('when evolve() is called', () => {
        it('returns a numeric error summary', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolutionSummary = await network.evolve(xorTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
            selection: 'POWER',
          });

          // Assert
          expect(typeof evolutionSummary.error).toBe('number');
        });
      });
    });

    describe('given FITNESS_PROPORTIONATE selection is requested', () => {
      describe('when evolve() is called', () => {
        it('returns a numeric error summary', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolutionSummary = await network.evolve(xorTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
            selection: 'FITNESS_PROPORTIONATE',
          });

          // Assert
          expect(typeof evolutionSummary.error).toBe('number');
        });
      });
    });

    describe('given TOURNAMENT selection is requested', () => {
      describe('when evolve() is called', () => {
        it('returns a numeric error summary', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolutionSummary = await network.evolve(xorTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
            selection: { name: 'TOURNAMENT', size: 2, probability: 1 },
          });

          // Assert
          expect(typeof evolutionSummary.error).toBe('number');
        });
      });
    });

    describe('given a maxNodes constraint is configured', () => {
      describe('when evolve() is called', () => {
        it('returns a numeric error summary', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const evolutionSummary = await network.evolve(xorTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 2,
            maxNodes: 3,
          });

          // Assert
          expect(typeof evolutionSummary.error).toBe('number');
        });
      });
    });

    describe('given the configured population size is empty', () => {
      describe('when evolve() is called', () => {
        it('still returns a numeric error summary', async () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);
          Reflect.set(network, 'population', []);

          // Act
          const evolutionSummary = await network.evolve(xorTrainingSet, {
            iterations: 2,
            error: 0.5,
            amount: 1,
            threads: 1,
            popsize: 0,
          });

          // Assert
          expect(typeof evolutionSummary.error).toBe('number');
        });
      });
    });
  });
});
