/**
 * Bootstrap helpers for the trainer's static shape and initial NEAT runtime.
 *
 * The trainer keeps setup separate from the main entry file so configuration and
 * controller construction can be read, tested, and tuned without also reading
 * the outer loop. This boundary answers a simple question: what does the demo
 * need before the first generation can run?
 */
import Neat from '../../../src/neat.ts';
import * as methods from '../../../src/methods/methods.ts';
import {
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';
import {
  DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  buildExampleArchitectureProfileNetwork,
  resolveExampleArchitectureProfile,
} from '../../architectureProfiles';
import {
  FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT,
  FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT,
  FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE,
  FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE,
} from './trainer.constants';

const FLAPPY_TRAINER_RECURRENT_MUTATION_METHODS = [
  ...methods.mutation.FFW,
  methods.mutation.ADD_BACK_CONN,
  methods.mutation.SUB_BACK_CONN,
  methods.mutation.ADD_SELF_CONN,
  methods.mutation.SUB_SELF_CONN,
];
import type {
  FlappyTrainerNeatController,
  FlappyTrainerRuntimeState,
  FlappyTrainerSetup,
} from './trainer.types';

/**
 * Creates immutable setup values for the trainer.
 *
 * Educational note:
 * The setup object freezes the core training shape up front: input width,
 * output width, population size, and elitism count. Centralizing those values
 * makes the rest of the trainer read as policy rather than configuration noise.
 *
 * @returns Default trainer setup values used for NEAT configuration.
 */
export function createTrainerSetup(): FlappyTrainerSetup {
  const profileId = DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID;
  const resolvedProfile = resolveExampleArchitectureProfile(
    'flappy-bird',
    profileId,
  );
  return {
    architectureProfileId: profileId,
    isRecurrent: resolvedProfile.recurrent,
    inputSize: FLAPPY_NETWORK_INPUT_SIZE,
    outputSize: FLAPPY_NETWORK_OUTPUT_SIZE,
    populationSize: FLAPPY_TRAINER_DEFAULT_POPULATION_SIZE,
    elitismCount: FLAPPY_TRAINER_DEFAULT_ELITISM_COUNT,
  };
}

/**
 * Creates mutable runtime state container.
 *
 * The runtime state is intentionally tiny. It only tracks stop intent and the
 * latest report so the outer loop can remain easy to reason about.
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
 * Educational note:
 * The trainer enables population-level fitness mode because the quality of a
 * Flappy policy depends on fair comparison across shared seed batches, not on a
 * one-network-at-a-time scoring callback.
 *
 * @example
 * ```ts
 * const trainerSetup = createTrainerSetup();
 * const neatController = createNeatController(trainerSetup);
 * ```
 *
 * @param trainerSetup - Immutable trainer setup values.
 * @returns Typed NEAT controller used by the trainer loop.
 */
export function createNeatController(
  trainerSetup: FlappyTrainerSetup,
): FlappyTrainerNeatController {
  const resolvedProfile = resolveExampleArchitectureProfile(
    'flappy-bird',
    trainerSetup.architectureProfileId,
  );
  const seedNetwork = buildExampleArchitectureProfileNetwork(
    'flappy-bird',
    trainerSetup.architectureProfileId,
  );
  const neatInstance = new Neat(
    trainerSetup.inputSize,
    trainerSetup.outputSize,
    resolveNoopFitness,
    {
      popsize: trainerSetup.populationSize,
      elitism: trainerSetup.elitismCount,
      mutationRate: FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_RATE,
      mutationAmount: FLAPPY_TRAINER_NEAT_INITIAL_MUTATION_AMOUNT,
      allowRecurrent: resolvedProfile.recurrent,
      mutation: resolvedProfile.recurrent
        ? FLAPPY_TRAINER_RECURRENT_MUTATION_METHODS
        : methods.mutation.FFW,
      network: seedNetwork,
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
 * This placeholder keeps controller construction simple. The real staged
 * evaluator is attached immediately afterward by the fitness service, so this
 * function exists only to satisfy the generic controller's constructor contract.
 *
 * @returns Constant zero fitness.
 */
function resolveNoopFitness(): number {
  return 0;
}
