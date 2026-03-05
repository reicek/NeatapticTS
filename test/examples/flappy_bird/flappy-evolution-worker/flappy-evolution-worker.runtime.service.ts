import { Neat, methods } from '../../../../src/neataptic';
import Architect from '../../../../src/architecture/architect';
import { evaluateFlappyFitness } from '../flappyEvaluation';
import type { WorkerInitMessage } from './flappy-evolution-worker.types';
import {
  FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
  FLAPPY_NETWORK_INPUT_SIZE,
  FLAPPY_NETWORK_OUTPUT_SIZE,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
} from '../constants/constants';

/**
 * Creates and configures the worker-local NEAT runtime used by browser evolution playback.
 *
 * @param initPayload - Initialization values from the browser host.
 * @returns Initialized NEAT runtime.
 */
export function createInitializedWorkerRuntime(
  initPayload: WorkerInitMessage['payload'],
): Neat {
  const inputSize = FLAPPY_NETWORK_INPUT_SIZE;
  const outputSize = FLAPPY_NETWORK_OUTPUT_SIZE;

  const neatRuntime = new Neat(inputSize, outputSize, () => 0, {
    popsize: initPayload.populationSize,
    elitism: initPayload.elitismCount,
    mutationRate: 0.75,
    mutationAmount: 2,
    mutation: methods.mutation.FFW,
    network: Architect.perceptron(
      inputSize,
      ...FLAPPY_NETWORK_HIDDEN_LAYER_SIZES,
      outputSize,
    ),
    speciation: true,
    multiObjective: { enabled: false },
    novelty: { enabled: false },
  });

  neatRuntime.fitness = (network) =>
    evaluateFlappyFitness(network, {
      enableEarlyTermination: true,
      maxFrames: FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
    });

  neatRuntime.restoreRNGState(initPayload.rngSeed);
  return neatRuntime;
}
