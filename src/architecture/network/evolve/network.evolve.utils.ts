/**
 * Network-level neuroevolution loop for improving one runnable graph in place.
 *
 * This chapter sits between the lightweight `Network` facade and the broader
 * `neat/` subsystem. Call `network.evolve()` when the question is "improve
 * this graph against a supervised dataset" without first wiring a separate
 * experiment harness. The method treats the current network as a seed genome,
 * configures a short-lived NEAT runtime around it, evaluates descendants, then
 * copies the best discovered structure back into the original instance.
 *
 * The folder is split by the same stages the public call executes. `setup`
 * validates dataset shape and resolves defaults. `fitness` turns prediction
 * error into a comparable score. `loop` owns generation-to-generation stopping
 * logic. `finalize` adopts the winning genome and tears down worker resources.
 * Keeping those shelves separate makes the README read like an evolution run
 * rather than an alphabetical pile of helpers.
 *
 * ```mermaid
 * flowchart LR
 *   Seed[Seed Network] --> Normalize[Normalize options and dataset]
 *   Normalize --> Fitness[Build fitness evaluator]
 *   Fitness --> Neat[Create temporary NEAT runtime]
 *   Neat --> Loop[Run evolve loop]
 *   Loop --> Adopt[Adopt best genome into original network]
 * ```
 *
 * Use this boundary when you want a bounded local search over network
 * structure, not a long-lived population controller. `error` answers "stop
 * when good enough", `iterations` answers "stop after this many generations",
 * and `growth` answers "how much should extra structure cost while searching".
 *
 * For compact background reading on the wider search family behind this folder,
 * see Wikipedia contributors,
 * [Evolutionary algorithm](https://en.wikipedia.org/wiki/Evolutionary_algorithm).
 * The implementation here is intentionally narrower: it keeps the public call
 * site small while reusing the repo's NEAT runtime under the hood.
 *
 * Example: stop when the network gets below an error target or hits a
 * generation cap.
 *
 * ```ts
 * const summary = await network.evolve(trainingSet, {
 *   error: 0.02,
 *   iterations: 500,
 *   growth: 0.0005,
 * });
 * ```
 *
 * Example: cap the search more tightly and fan evaluation across workers when
 * dataset scoring dominates.
 *
 * ```ts
 * const summary = await network.evolve(trainingSet, {
 *   iterations: 120,
 *   threads: 2,
 *   log: 20,
 * });
 * ```
 */
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
