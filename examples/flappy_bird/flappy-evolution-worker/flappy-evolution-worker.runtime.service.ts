import { Neat, methods } from '../../../src/neataptic';
import { evaluateFlappyFitness } from '../flappyEvaluation';
import type { WorkerInitMessage } from './flappy-evolution-worker.types';
import {
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
} from '../constants/constants';
import {
  DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  buildExampleArchitectureProfileNetwork,
} from '../../architectureProfiles';

/**
 * Creates and configures the worker-local NEAT runtime used by browser evolution playback.
 *
 * Educational note:
 * The browser worker reuses the same core NeatapticTS runtime as the Node-side
 * trainer, but trims configuration down to the pieces needed for an interactive
 * example: deterministic seeding, feed-forward mutation policy, and a fitness
 * function that favors quick browser-visible iteration.
 *
 * The resulting runtime is both the evolution engine and the source of the
 * population that later playback requests visualize.
 *
 * For background reading, the Wikipedia article on "Neuroevolution of
 * augmenting topologies" is a useful overview of the family of ideas this demo
 * is exercising, even though the repository implements its own detailed runtime
 * behavior and modern extensions.
 *
 * @example
 * ```ts
 * const neatRuntime = createInitializedWorkerRuntime({
 *   populationSize: 50,
 *   elitismCount: 10,
 *   rngSeed: 12345,
 * });
 * ```
 *
 * @param initPayload - Initialization values from the browser host.
 * @returns Initialized NEAT runtime.
 */
export function createInitializedWorkerRuntime(
  initPayload: WorkerInitMessage['payload'],
): Neat {
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;
  const seedNetwork = buildExampleArchitectureProfileNetwork(
    'flappy-bird',
    initPayload.architectureProfileId ??
      DEFAULT_FLAPPY_ARCHITECTURE_PROFILE_ID,
  );

  const neatRuntime = new Neat(inputSize, outputSize, () => 0, {
    popsize: initPayload.populationSize,
    elitism: initPayload.elitismCount,
    mutationRate: 0.75,
    mutationAmount: 2,
    allowRecurrent: false,
    mutation: methods.mutation.FFW,
    network: seedNetwork,
    speciation: true,
    multiObjective: { enabled: false },
    novelty: { enabled: false },
  });

  neatRuntime.fitness = (network) =>
    evaluateFlappyFitness(network, {
      enableEarlyTermination: true,
      maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    });

  neatRuntime.restoreRNGState(initPayload.rngSeed);
  return neatRuntime;
}
