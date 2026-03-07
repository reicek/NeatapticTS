import Neat from '../../../../src/neat.ts';
import Architect from '../../../../src/architecture/architect.ts';
import * as methods from '../../../../src/methods/methods.ts';
import {
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';
import {
  FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT,
  FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT,
  FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE,
  FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE,
} from './trainer.constants';
import type {
  FlappyTrainerNeatController,
  FlappyTrainerRuntimeState,
  FlappyTrainerSetup,
} from './trainer.types';

/**
 * Creates immutable setup values for the trainer.
 *
 * @returns Default trainer setup values used for NEAT configuration.
 */
export function createTrainerSetup(): FlappyTrainerSetup {
  return {
    inputSize: FLAPPY_NETWORK_INPUT_SIZE,
    outputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
    populationSize: FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE,
    elitismCount: FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT,
  };
}

/**
 * Creates mutable runtime state container.
 *
 * @returns Fresh runtime state used by loop orchestration.
 */
export function createTrainerRuntimeState(): FlappyTrainerRuntimeState {
  return {
    shouldStop: false,
    latestGenerationReport: undefined,
  };
}

/**
 * Builds the NEAT controller with baseline options.
 *
 * @param trainerSetup - Immutable trainer setup values.
 * @returns Typed NEAT controller used by the trainer loop.
 */
export function createNeatController(
  trainerSetup: FlappyTrainerSetup,
): FlappyTrainerNeatController {
  const neatInstance = new Neat(
    trainerSetup.inputSize,
    trainerSetup.outputSize,
    resolveNoopFitness,
    {
      popsize: trainerSetup.populationSize,
      elitism: trainerSetup.elitismCount,
      mutationRate: FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE,
      mutationAmount: FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT,
      mutation: methods.mutation.FFW,
      network: Architect.perceptron(
        trainerSetup.inputSize,
        ...FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
        trainerSetup.outputSize,
      ),
      fitnessPopulation: true,
      speciation: true,
      multiObjective: { enabled: false },
      novelty: { enabled: false },
    },
  );

  return neatInstance as never as FlappyTrainerNeatController;
}

/**
 * Trivial baseline fitness used before attaching population evaluator.
 *
 * @returns Constant zero fitness.
 */
function resolveNoopFitness(): number {
  return 0;
}
