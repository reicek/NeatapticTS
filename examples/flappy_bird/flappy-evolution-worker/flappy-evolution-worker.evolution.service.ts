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
  /** Whether generation zero should be released for playback before the first full evolve pass. */
  publishStartupPopulationBeforeFirstEvolution?: boolean;
  /** Marks the startup population as already published once the worker sends it to the host. */
  markStartupPopulationPublished?: () => void;
}

/**
 * Creates the next compact generation-ready response payload.
 *
 * Educational note:
 * The first browser-visible population should not wait for a full recurrent
 * selection batch or optional warm-start assist. When the caller opts in,
 * generation zero is released immediately so playback can begin promptly. Later
 * requests run the bounded warm-start assist, the normal NEAT `evolve()` pass,
 * and emit the same compact summary shape: generation index, best fitness,
 * transferable inference payloads for playback, and the temporary JSON
 * visualization bridge used by the host network panel.
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
    publishStartupPopulationBeforeFirstEvolution,
    markStartupPopulationPublished,
  } = options;

  if (initializationPromise) {
    await initializationPromise;
  }

  if (!neatRuntime || isStopped()) {
    throw new Error('Evolution worker runtime is not initialized.');
  }

  const runtimeNeat = neatRuntime as unknown as { population?: Network[] };
  const startupPopulation = resolveRuntimePopulation(runtimeNeat);

  if (
    shouldPublishStartupPopulationBeforeEvolution(
      neatRuntime.generation,
      startupPopulation,
      publishStartupPopulationBeforeFirstEvolution,
    )
  ) {
    const bestStartupNetwork =
      resolveBestNetworkFromPopulation(startupPopulation);

    setCurrentPopulation(startupPopulation);
    markStartupPopulationPublished?.();

    return buildGenerationReadyMessage({
      architectureProfileId,
      bestNetwork: bestStartupNetwork,
      generation: neatRuntime.generation,
      population: startupPopulation,
    });
  }

  await runBestEffortWarmStart(warmStartGenerationZeroIfNeeded, neatRuntime);

  const bestNetwork = (await neatRuntime.evolve()) as Network;
  const evolvedPopulation = resolveRuntimePopulation(runtimeNeat, bestNetwork);
  setCurrentPopulation(evolvedPopulation);

  return buildGenerationReadyMessage({
    architectureProfileId,
    bestNetwork,
    generation: neatRuntime.generation,
    population: evolvedPopulation,
  });
}

/**
 * Resolves whether the worker should release generation zero before evolving.
 *
 * @param generation - Current NEAT generation index.
 * @param population - Current worker population snapshot.
 * @param publishStartupPopulationBeforeFirstEvolution - Caller startup-release flag.
 * @returns True when generation zero can be published immediately for playback.
 */
function shouldPublishStartupPopulationBeforeEvolution(
  generation: number,
  population: readonly Network[],
  publishStartupPopulationBeforeFirstEvolution: boolean | undefined,
): boolean {
  return (
    Boolean(publishStartupPopulationBeforeFirstEvolution) &&
    generation === 0 &&
    population.length > 0
  );
}

/**
 * Resolves the current runtime population with an optional best-network fallback.
 *
 * @param runtimeNeat - Runtime population holder.
 * @param fallbackNetwork - Best network returned by an evolution pass.
 * @returns Non-empty population when one is available.
 */
function resolveRuntimePopulation(
  runtimeNeat: { population?: Network[] },
  fallbackNetwork?: Network,
): Network[] {
  if (
    Array.isArray(runtimeNeat.population) &&
    runtimeNeat.population.length > 0
  ) {
    return runtimeNeat.population;
  }

  return fallbackNetwork ? [fallbackNetwork] : [];
}

/**
 * Chooses the best network from a population snapshot using available scores.
 *
 * @param population - Population snapshot to scan.
 * @returns Highest-scored network, falling back to the first network when scores are equal.
 */
function resolveBestNetworkFromPopulation(
  population: readonly Network[],
): Network {
  const firstPopulationNetwork = population[0];
  if (!firstPopulationNetwork) {
    throw new Error('Cannot publish a Flappy generation without a network.');
  }

  return population.reduce((bestNetwork, populationNetwork) => {
    const bestScore = Number(bestNetwork.score ?? 0);
    const populationScore = Number(populationNetwork.score ?? 0);

    return populationScore > bestScore ? populationNetwork : bestNetwork;
  }, firstPopulationNetwork);
}

/**
 * Builds the generation-ready worker response from a population snapshot.
 *
 * @param options - Generation metadata plus selected best network and population.
 * @returns Generation-ready worker response payload.
 */
function buildGenerationReadyMessage(options: {
  architectureProfileId: WorkerGenerationReadyMessage['payload']['architectureProfileId'];
  generation: number;
  bestNetwork: Network;
  population: Network[];
}): WorkerGenerationReadyMessage {
  return {
    type: 'generation-ready',
    payload: {
      architectureProfileId: options.architectureProfileId,
      generation: options.generation,
      bestFitness: Number(options.bestNetwork.score ?? 0),
      bestNetworkJson: options.bestNetwork.toJSON(),
      bestNetworkPayload: exportTransferableInferencePayload(
        options.bestNetwork,
      ),
      populationNetworksJson: options.population.map((populationNetwork) =>
        populationNetwork.toJSON(),
      ),
      populationNetworkPayloads: options.population.map((populationNetwork) =>
        exportTransferableInferencePayload(populationNetwork),
      ),
    },
  };
}

/**
 * Runs generation-zero warm-start as an optional assist before regular evolution.
 *
 * @param warmStartGenerationZeroIfNeeded - Warm-start callback for the active runtime.
 * @param neatRuntime - Runtime that should evolve even when warm-start fails.
 * @returns Promise resolved after warm-start succeeds or is skipped.
 */
async function runBestEffortWarmStart(
  warmStartGenerationZeroIfNeeded: (neatController: Neat) => Promise<void>,
  neatRuntime: Neat,
): Promise<void> {
  try {
    await warmStartGenerationZeroIfNeeded(neatRuntime);
  } catch {
    // The browser demo should keep teaching/evolving even when the optional prior fails.
  }
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
