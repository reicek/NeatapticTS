import * as methods from '../../../methods/methods';
import type { EvolveOptions, EvolutionSettings } from '../network.types';
import Network from '../network';
import * as evolveFitnessUtils from './network.evolve.fitness.utils';
import {
  configureNeatOptions,
  createEvolutionConfig,
  prepareFitnessFunction,
} from './network.evolve.setup.utils';

describe('network evolve setup utility chapter', () => {
  describe('createEvolutionConfig', () => {
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