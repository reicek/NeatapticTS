import * as methods from '../../../methods/methods';
import type {
  EvolveOptions,
  EvolutionSettings,
  NeatRuntime,
} from '../network.types';
import Network from '../network';
import * as evolveFitnessUtils from './network.evolve.fitness.utils';
import {
  applySmallPopulationHeuristics,
  configureNeatOptions,
  createEvolutionConfig,
  prepareFitnessFunction,
  warnIfNoBestGenomeMayOccur,
} from './network.evolve.setup.utils';

describe('network evolve setup utility chapter', () => {
  describe('createEvolutionConfig', () => {
    describe('given no schedule is configured', () => {
      it('returns undefined', () => {
        // Arrange
        const evolutionSettings: EvolutionSettings = {
          amount: 1,
          clear: false,
          cost: methods.Cost.mse,
          growth: 0,
          log: 0,
          schedule: undefined,
          targetError: 0.01,
          threads: 1,
        };

        // Act
        const result = createEvolutionConfig(evolutionSettings);

        // Assert
        expect(result).toBeUndefined();
      });
    });

    describe('given a schedule callback is configured', () => {
      it('returns the structured evolution summary config', () => {
        // Arrange
        const scheduleCallback = jest.fn();
        const evolutionSettings: EvolutionSettings = {
          amount: 3,
          clear: true,
          cost: methods.Cost.mse,
          growth: 0.25,
          log: 5,
          schedule: {
            function: scheduleCallback,
            iterations: 2,
          },
          targetError: 0.01,
          threads: 2,
        };

        // Act
        const evolutionConfig = createEvolutionConfig(evolutionSettings);

        // Assert
        expect(evolutionConfig).toEqual({
          amount: 3,
          clear: true,
          cost: methods.Cost.mse,
          growth: 0.25,
          log: 5,
          schedule: {
            function: scheduleCallback,
            iterations: 2,
          },
          targetError: 0.01,
          threads: 2,
        });
      });
    });
  });

  describe('configureNeatOptions', () => {
    describe('given populationSize is provided without the legacy popsize alias', () => {
      it('hydrates the legacy alias while preserving the explicit speciation flag', () => {
        // Arrange
        const network = new Network(1, 1, { seed: 801 });
        const evolveOptions: EvolveOptions = {
          populationSize: 7,
          speciation: true,
        };

        // Act
        configureNeatOptions(network, evolveOptions);

        // Assert
        expect({
          network: evolveOptions.network,
          popsize: evolveOptions.popsize,
          speciation: evolveOptions.speciation,
        }).toEqual({
          network,
          popsize: 7,
          speciation: true,
        });
      });
    });
  });

  describe('warnIfNoBestGenomeMayOccur', () => {
    describe('given _warnIfNoBestGenome is absent on the neat instance', () => {
      it('returns early without throwing', () => {
        // Arrange – _warnIfNoBestGenome absent → line 250 TRUE arm
        const neatInstance = {
          // _warnIfNoBestGenome intentionally absent
        } as unknown as NeatRuntime;

        // Act & Assert (must not throw)
        expect(() =>
          warnIfNoBestGenomeMayOccur(neatInstance, { iterations: 0 }),
        ).not.toThrow();
      });
    });
  });

  describe('applySmallPopulationHeuristics', () => {
    describe('given popsize is at or below the small-population threshold', () => {
      it('assigns default mutation rate and amount when they are not already set', () => {
        // Arrange – popsize=5 triggers the heuristic, undefined rates → lines 277-279
        const neatInstance = {
          options: {
            // mutationRate and mutationAmount intentionally absent
          },
        } as unknown as NeatRuntime;

        // Act
        applySmallPopulationHeuristics(neatInstance, { popsize: 5 });

        // Assert
        expect({
          mutationRate: neatInstance.options.mutationRate,
          mutationAmount: neatInstance.options.mutationAmount,
        }).toEqual({
          mutationRate: expect.any(Number),
          mutationAmount: expect.any(Number),
        });
      });
    });
  });

  describe('prepareFitnessFunction', () => {
    describe('given multithreaded evolution is requested', () => {
      it('delegates to the multithread fitness builder and returns its resolved worker setup', async () => {
        // Arrange
        const expectedFitnessFunction = jest.fn();
        const trainingSet = [{ input: [0.2], output: [0.8] }];
        const resolvedSettings: EvolutionSettings = {
          amount: 3,
          clear: false,
          cost: methods.Cost.mse,
          growth: 0.15,
          log: 5,
          schedule: undefined,
          targetError: 0.01,
          threads: 2,
        };
        const evolveOptions: EvolveOptions = { threads: 2 };
        const buildMultiThreadFitnessSpy = jest
          .spyOn(evolveFitnessUtils, 'buildMultiThreadFitness')
          .mockResolvedValue({
            fitnessFunction: expectedFitnessFunction,
            threads: 2,
          });

        // Act
        const fitnessSetup = await prepareFitnessFunction(
          trainingSet,
          resolvedSettings,
          evolveOptions,
        );

        // Assert
        expect({
          fitnessSetup,
          buildMultiThreadFitnessArguments:
            buildMultiThreadFitnessSpy.mock.calls[0],
        }).toEqual({
          fitnessSetup: {
            fitnessFunction: expectedFitnessFunction,
            threads: 2,
          },
          buildMultiThreadFitnessArguments: [
            trainingSet,
            methods.Cost.mse,
            3,
            0.15,
            2,
            evolveOptions,
          ],
        });
      });
    });
  });
});
