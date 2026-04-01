import {
  DEFAULT_ELITISM,
  DEFAULT_MAX_CONNS,
  DEFAULT_MAX_GATES,
  DEFAULT_MAX_NODES,
  DEFAULT_MUTATION_AMOUNT,
  DEFAULT_MUTATION_RATE,
  DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
  DEFAULT_POPULATION_SIZE,
  DEFAULT_PROVENANCE,
} from '../neat.defaults.constants';
import {
  initializeNeatConstructor,
  type NeatInitializationHost,
} from './neat.init';

type InitializationHostHarness = NeatInitializationHost & {
  createPool: jest.Mock;
};

function createInitializationHost(): InitializationHostHarness {
  return {
    options: {},
    population: [],
    _lineageEnabled: false,
    _getRNG() {
      return () => 0.5;
    },
    createPool: jest.fn(),
  } as unknown as InitializationHostHarness;
}

describe('neat init chapter', () => {
  describe('initializeNeatConstructor', () => {
    describe('given the caller passes an empty options bag', () => {
      it('materializes the public constructor defaults onto that bag', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = {};

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: {},
          defaults: DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
        });

        // Assert
        expect(optionBag).toEqual(
          expect.objectContaining({
            popsize: DEFAULT_POPULATION_SIZE,
            equal: false,
            clear: false,
            elitism: DEFAULT_ELITISM,
            provenance: DEFAULT_PROVENANCE,
            mutationRate: DEFAULT_MUTATION_RATE,
            mutationAmount: DEFAULT_MUTATION_AMOUNT,
            fitnessPopulation: false,
            selection: expect.anything(),
            crossover: expect.anything(),
            mutation: expect.anything(),
            maxNodes: DEFAULT_MAX_NODES,
            maxConns: DEFAULT_MAX_CONNS,
            maxGates: DEFAULT_MAX_GATES,
          }),
        );
      });
    });

    describe('given explicit population and elitism settings are provided', () => {
      it('preserves those values and still bootstraps the initial pool', () => {
        // Arrange
        const initializationHost = createInitializationHost();
        const optionBag = { popsize: 10, elitism: 2 };

        // Act
        initializeNeatConstructor(initializationHost, {
          optionBag,
          rawOptions: { ...optionBag },
          defaults: DEFAULT_NEAT_CONSTRUCTOR_DEFAULTS,
        });

        // Assert
        expect({
          popsize: optionBag.popsize,
          elitism: optionBag.elitism,
          createPoolCalls: initializationHost.createPool.mock.calls,
        }).toEqual({
          popsize: 10,
          elitism: 2,
          createPoolCalls: [[null]],
        });
      });
    });
  });
});
