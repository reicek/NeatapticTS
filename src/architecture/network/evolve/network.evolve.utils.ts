import Network from '../../network/network';
import type { EvolveOptions, TrainingSample } from '../network.types';
import {
  applySmallPopulationHeuristics,
  assertEvolutionDatasetCompatibility,
  configureNeatOptions,
  createNeatInstance,
  getNormalizedOptions,
  prepareFitnessFunction,
  resolveEvolutionSettings,
  resolveStopConditions,
  warnIfNoBestGenomeMayOccur,
} from './network.evolve.setup.utils';
import { runEvolutionLoop } from './network.evolve.loop.utils';
import {
  adoptBestGenomeOrWarn,
  buildEvolutionSummary,
  terminateWorkersSafely,
} from './network.evolve.finalize.utils';

/**
 * Evolves a network with a NEAT-style search loop until an error target or generation limit is reached.
 *
 * Overview:
 * - This method treats the current network as a *seed genome* and explores better variants.
 * - Candidate genomes are scored by prediction error plus a structural complexity penalty.
 * - The best discovered genome is copied back into the current instance (in-place upgrade).
 *
 * Typical usage guidance:
 * - Use `error` when you care about reaching a quality threshold.
 * - Use `iterations` when you need deterministic runtime bounds.
 * - Use both when you want "stop when good enough, otherwise cap time" behavior.
 * - Increase `threads` only when worker support exists and dataset evaluation is expensive.
 *
 * @param this - Bound Network instance that receives the best evolved structure.
 * @param set - Supervised samples; sample input/output dimensions must match network I/O.
 * @param options - Evolution hyperparameters and stop conditions.
 * @returns Final summary containing best error estimate, generations processed, and elapsed milliseconds.
 *
 * @example
 * ```ts
 * const summary = await network.evolve(trainingSet, {
 *   error: 0.02,
 *   iterations: 500,
 *   growth: 0.0005,
 *   threads: 2,
 * });
 * console.log(summary.error, summary.iterations, summary.time);
 * ```
 */
export async function evolveNetwork(
  this: Network,
  set: TrainingSample[],
  options: EvolveOptions = {},
): Promise<{ error: number; iterations: number; time: number }> {
  // Step 1: Validate dataset shape and normalize options.
  assertEvolutionDatasetCompatibility(this, set);
  const normalizedOptions = getNormalizedOptions(options);
  const startTime = Date.now();

  // Step 2: Resolve scalar settings and stopping conditions.
  const settings = resolveEvolutionSettings(normalizedOptions);
  const stopConditions = resolveStopConditions(
    normalizedOptions,
    settings.targetError,
  );

  // Step 3: Prepare fitness evaluation (single-thread or worker-based).
  const fitnessSetup = await prepareFitnessFunction(
    set,
    settings,
    normalizedOptions,
  );

  // Step 4: Configure and instantiate NEAT runtime.
  configureNeatOptions(this, normalizedOptions);
  const neat = await createNeatInstance(
    this,
    fitnessSetup.fitnessFunction,
    normalizedOptions,
  );
  warnIfNoBestGenomeMayOccur(neat, normalizedOptions);
  applySmallPopulationHeuristics(neat, normalizedOptions);

  // Step 5: Execute evolution loop.
  const loopResult = await runEvolutionLoop(
    neat,
    settings,
    stopConditions.targetError,
    normalizedOptions.iterations,
  );

  // Step 6: Adopt best genome and cleanup resources.
  adoptBestGenomeOrWarn(this, neat, loopResult.bestGenome, settings.clear);
  terminateWorkersSafely(normalizedOptions);

  // Step 7: Return final evolve summary.
  return buildEvolutionSummary(loopResult.error, neat.generation, startTime);
}
