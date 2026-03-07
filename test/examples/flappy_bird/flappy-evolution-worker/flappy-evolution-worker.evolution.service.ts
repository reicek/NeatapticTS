import type { Neat } from '../../../../src/neataptic';
import type Network from '../../../../src/architecture/network';
import type { WorkerGenerationReadyMessage } from './flappy-evolution-worker.types';

/**
 * Dependencies required to evolve one generation and prepare host payload output.
 */
export interface WorkerEvolutionServiceOptions {
  initializationPromise?: Promise<void>;
  neatRuntime: Neat | undefined;
  isStopped: () => boolean;
  warmStartGenerationZeroIfNeeded: (neatController: Neat) => Promise<void>;
  setCurrentPopulation: (population: Network[]) => void;
}

/**
 * Evolves one generation and creates the compact generation-ready response payload.
 *
 * @param options - Evolution dependencies and runtime state accessors.
 * @returns Generation-ready worker response payload.
 */
export async function evolveAndBuildGenerationReadyMessage(
  options: WorkerEvolutionServiceOptions,
): Promise<WorkerGenerationReadyMessage> {
  const {
    initializationPromise,
    neatRuntime,
    isStopped,
    warmStartGenerationZeroIfNeeded,
    setCurrentPopulation,
  } = options;

  if (initializationPromise) {
    await initializationPromise;
  }

  if (!neatRuntime || isStopped()) {
    throw new Error('Evolution worker runtime is not initialized.');
  }

  await warmStartGenerationZeroIfNeeded(neatRuntime);

  const bestNetwork = (await neatRuntime.evolve()) as Network;
  const runtimeNeat = neatRuntime as unknown as { population?: Network[] };
  setCurrentPopulation(
    Array.isArray(runtimeNeat.population)
      ? runtimeNeat.population
      : [bestNetwork],
  );

  return {
    type: 'generation-ready',
    payload: {
      generation: neatRuntime.generation,
      bestFitness: Number(bestNetwork.score ?? 0),
      bestNetworkJson: bestNetwork.toJSON(),
    },
  };
}
