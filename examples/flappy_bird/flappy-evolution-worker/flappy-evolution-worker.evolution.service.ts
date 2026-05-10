import {
  exportTransferableInferencePayload,
  getTransferList,
  type Neat,
} from '../../../src/neataptic';
import type Network from '../../../src/architecture/network';
import type { WorkerGenerationReadyMessage } from './flappy-evolution-worker.types';

/**
 * Dependencies required to evolve one generation and prepare host payload output.
 *
 * Educational note:
 * This interface isolates the evolution step from the worker entrypoint. That
 * makes the README easier to follow: the entrypoint owns protocol orchestration,
 * while this service owns one well-defined "run generation -> publish summary"
 * slice of behavior.
 */
export interface WorkerEvolutionServiceOptions {
  architectureProfileId: WorkerGenerationReadyMessage['payload']['architectureProfileId'];
  initializationPromise?: Promise<void>;
  neatRuntime: Neat | undefined;
  isStopped: () => boolean;
  warmStartGenerationZeroIfNeeded: (neatController: Neat) => Promise<void>;
  setCurrentPopulation: (population: Network[]) => void;
}

/**
 * Evolves one generation and creates the compact generation-ready response payload.
 *
 * Educational note:
 * The worker does not stream the whole population back to the UI after each
 * evolution step. Instead it emits a compact summary containing the generation
 * index, best fitness, typed transferable inference payloads for playback, and
 * the temporary JSON visualization bridge used by the host-side network panel.
 *
 * @example
 * ```ts
 * const generationMessage = await evolveAndBuildGenerationReadyMessage({
 *   initializationPromise,
 *   neatRuntime,
 *   isStopped: () => false,
 *   warmStartGenerationZeroIfNeeded,
 *   setCurrentPopulation,
 * });
 * ```
 *
 * @param options - Evolution dependencies and runtime state accessors.
 * @returns Generation-ready worker response payload.
 */
export async function evolveAndBuildGenerationReadyMessage(
  options: WorkerEvolutionServiceOptions,
): Promise<WorkerGenerationReadyMessage> {
  const {
    architectureProfileId,
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

  const runtimeNeat = neatRuntime as unknown as { population?: Network[] };

  const bestNetwork = (await neatRuntime.evolve()) as Network;
  setCurrentPopulation(
    Array.isArray(runtimeNeat.population)
      ? runtimeNeat.population
      : [bestNetwork],
  );

  return {
    type: 'generation-ready',
    payload: {
      architectureProfileId,
      generation: neatRuntime.generation,
      bestFitness: Number(bestNetwork.score ?? 0),
      bestNetworkJson: bestNetwork.toJSON(),
      bestNetworkPayload: exportTransferableInferencePayload(bestNetwork),
      populationNetworksJson: Array.isArray(runtimeNeat.population)
        ? runtimeNeat.population.map((populationNetwork) =>
            populationNetwork.toJSON(),
          )
        : [bestNetwork.toJSON()],
      populationNetworkPayloads: Array.isArray(runtimeNeat.population)
        ? runtimeNeat.population.map((populationNetwork) =>
            exportTransferableInferencePayload(populationNetwork),
          )
        : [exportTransferableInferencePayload(bestNetwork)],
    },
  };
}

/**
 * Collect the transferable buffers owned by one generation-ready response payload.
 *
 * The transfer list intentionally includes only the typed-array inference
 * payloads. The temporary JSON visualization bridge remains structured-clone
 * data so the browser can continue rebuilding network-view models separately.
 *
 * @param workerMessage - Generation-ready worker response payload.
 * @returns Transfer list for `postMessage(...)`.
 */
export function resolveGenerationReadyMessageTransferList(
  workerMessage: WorkerGenerationReadyMessage,
): ArrayBuffer[] {
  const bestNetworkTransferList = workerMessage.payload.bestNetworkPayload
    ? getTransferList(workerMessage.payload.bestNetworkPayload)
    : [];
  const populationTransferList =
    workerMessage.payload.populationNetworkPayloads?.flatMap(
      (populationNetworkPayload) => getTransferList(populationNetworkPayload),
    ) ?? [];

  return [...bestNetworkTransferList, ...populationTransferList];
}
