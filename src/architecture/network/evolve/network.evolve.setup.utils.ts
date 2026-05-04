import Network from '../../network/network';
import * as methods from '../../../methods/methods';
import type {
  EvolutionConfig,
  EvolutionFitnessFunction,
  EvolutionSettings,
  EvolutionStopConditions,
  EvolveCostFunction as CostFunction,
  EvolveOptions,
  FitnessSetup,
  NeatRuntime,
  TrainingSample,
} from '../network.types';
import {
  buildMultiThreadFitness,
  buildSingleThreadFitness,
} from './network.evolve.fitness.utils';
import {
  DATASET_COMPATIBILITY_ERROR_MESSAGE,
  DEFAULT_EVALUATION_AMOUNT,
  DEFAULT_GROWTH,
  DEFAULT_LOG_INTERVAL,
  DEFAULT_TARGET_ERROR,
  DEFAULT_THREAD_COUNT,
  SMALL_POPULATION_MUTATION_AMOUNT,
  SMALL_POPULATION_MUTATION_RATE,
  SMALL_POPULATION_THRESHOLD,
  STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE,
  DISABLED_TARGET_ERROR,
  ZERO_ITERATIONS,
} from './network.evolve.utils.types';
import {
  NetworkEvolveDatasetCompatibilityError,
  NetworkEvolveStoppingConditionRequiredError,
} from './network.evolve.errors';

/**
 * Validate dataset existence and dimensional compatibility with network I/O.
 *
 * @param network - Network being evolved.
 * @param dataSet - Supervised dataset.
 * @returns Nothing.
 */
export function assertEvolutionDatasetCompatibility(
  network: Network,
  dataSet: TrainingSample[],
): void {
  if (!dataSet || dataSet.length === 0) {
    throw new NetworkEvolveDatasetCompatibilityError(
      DATASET_COMPATIBILITY_ERROR_MESSAGE,
    );
  }

  const firstSample = dataSet[0];
  const inputMatches = firstSample.input.length === network.input;
  const outputMatches = firstSample.output.length === network.output;
  if (!inputMatches || !outputMatches) {
    throw new NetworkEvolveDatasetCompatibilityError(
      DATASET_COMPATIBILITY_ERROR_MESSAGE,
    );
  }
}

/**
 * Ensure options object exists.
 *
 * @param evolveOptions - Incoming evolve options.
 * @returns Safe options object.
 */
export function getNormalizedOptions(
  evolveOptions: EvolveOptions,
): EvolveOptions {
  return evolveOptions;
}

/**
 * Resolve normalized scalar settings with defaults.
 *
 * @param evolveOptions - Evolve options object.
 * @returns Normalized scalar settings.
 */
export function resolveEvolutionSettings(
  evolveOptions: EvolveOptions,
): EvolutionSettings {
  return {
    targetError: evolveOptions.error ?? DEFAULT_TARGET_ERROR,
    growth: evolveOptions.growth ?? DEFAULT_GROWTH,
    cost: evolveOptions.cost ?? methods.Cost.mse,
    amount: evolveOptions.amount ?? DEFAULT_EVALUATION_AMOUNT,
    log: evolveOptions.log ?? DEFAULT_LOG_INTERVAL,
    schedule: evolveOptions.schedule,
    clear: evolveOptions.clear ?? false,
    threads:
      typeof evolveOptions.threads === 'undefined'
        ? DEFAULT_THREAD_COUNT
        : evolveOptions.threads,
  };
}

/**
 * Resolve stopping-condition semantics while preserving legacy behavior.
 *
 * @param evolveOptions - Evolve options object.
 * @param initialTargetError - Target error resolved from options.
 * @returns Final stop conditions.
 */
export function resolveStopConditions(
  evolveOptions: EvolveOptions,
  initialTargetError: number,
): EvolutionStopConditions {
  let resolvedTargetError = initialTargetError;

  const iterationsMissing = typeof evolveOptions.iterations === 'undefined';
  const errorMissing = typeof evolveOptions.error === 'undefined';
  if (iterationsMissing && errorMissing) {
    throw new NetworkEvolveStoppingConditionRequiredError(
      STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE,
    );
  }

  if (errorMissing) {
    resolvedTargetError = DISABLED_TARGET_ERROR;
  } else if (iterationsMissing) {
    evolveOptions.iterations = ZERO_ITERATIONS;
  }

  return { targetError: resolvedTargetError };
}

/**
 * Build optional structured evolution config summary.
 *
 * @param settingsToSummarize - Scalar evolution settings.
 * @returns Optional summary config.
 */
export function createEvolutionConfig(
  settingsToSummarize: EvolutionSettings,
): EvolutionConfig | undefined {
  if (!settingsToSummarize.schedule) return undefined;

  return {
    targetError: settingsToSummarize.targetError,
    growth: settingsToSummarize.growth,
    cost: settingsToSummarize.cost,
    amount: settingsToSummarize.amount,
    log: settingsToSummarize.log,
    schedule: settingsToSummarize.schedule,
    clear: settingsToSummarize.clear,
    threads: settingsToSummarize.threads,
  };
}

/**
 * Build fitness function according to threading configuration.
 *
 * @param dataSet - Supervised dataset.
 * @param resolvedSettings - Scalar evolution settings.
 * @param evolveOptions - Evolve options object.
 * @returns Fitness function and resolved thread count.
 */
export async function prepareFitnessFunction(
  dataSet: TrainingSample[],
  resolvedSettings: EvolutionSettings,
  evolveOptions: EvolveOptions,
): Promise<FitnessSetup> {
  if (resolvedSettings.threads === DEFAULT_THREAD_COUNT) {
    return {
      fitnessFunction: buildSingleThreadFitness(
        dataSet,
        resolvedSettings.cost as CostFunction,
        resolvedSettings.amount,
        resolvedSettings.growth,
      ),
      threads: DEFAULT_THREAD_COUNT,
    };
  }

  const multiThreadSetup = await buildMultiThreadFitness(
    dataSet,
    resolvedSettings.cost,
    resolvedSettings.amount,
    resolvedSettings.growth,
    resolvedSettings.threads,
    evolveOptions,
  );

  return {
    fitnessFunction: multiThreadSetup.fitnessFunction,
    threads: multiThreadSetup.threads,
  };
}

/**
 * Normalize options used by NEAT constructor.
 *
 * @param network - Network instance being evolved.
 * @param evolveOptions - Evolve options object.
 * @returns Nothing.
 */
export function configureNeatOptions(
  network: Network,
  evolveOptions: EvolveOptions,
): void {
  evolveOptions.network = network;
  if (evolveOptions.populationSize != null && evolveOptions.popsize == null)
    evolveOptions.popsize = evolveOptions.populationSize;
  if (typeof evolveOptions.speciation === 'undefined')
    evolveOptions.speciation = false;
}

/**
 * Lazy-load and create NEAT instance.
 *
 * @param network - Network instance being evolved.
 * @param fitnessFunction - Prepared fitness evaluator.
 * @param evolveOptions - Evolve options object.
 * @returns Constructed NEAT instance.
 */
export async function createNeatInstance(
  network: Network,
  fitnessFunction: EvolutionFitnessFunction,
  evolveOptions: EvolveOptions,
): Promise<NeatRuntime> {
  const { default: Neat } = await import('../../../neat');
  return new Neat(
    network.input,
    network.output,
    fitnessFunction,
    evolveOptions,
  ) as unknown as NeatRuntime;
}

/**
 * Emit warning when zero-iteration configuration may produce no best genome.
 *
 * @param neatInstance - Active NEAT instance.
 * @param evolveOptions - Evolve options object.
 * @returns Nothing.
 */
export function warnIfNoBestGenomeMayOccur(
  neatInstance: NeatRuntime,
  evolveOptions: EvolveOptions,
): void {
  if (
    typeof evolveOptions.iterations !== 'number' ||
    evolveOptions.iterations !== ZERO_ITERATIONS
  )
    return;

  if (!neatInstance._warnIfNoBestGenome) return;

  try {
    neatInstance._warnIfNoBestGenome();
  } catch {
    // Ignore warning errors
  }
}

/**
 * Increase mutation aggressiveness for tiny populations.
 *
 * @param neatInstance - Active NEAT instance.
 * @param evolveOptions - Evolve options object.
 * @returns Nothing.
 */
export function applySmallPopulationHeuristics(
  neatInstance: NeatRuntime,
  evolveOptions: EvolveOptions,
): void {
  if (
    !evolveOptions.popsize ||
    evolveOptions.popsize > SMALL_POPULATION_THRESHOLD
  )
    return;

  neatInstance.options.mutationRate =
    neatInstance.options.mutationRate ?? SMALL_POPULATION_MUTATION_RATE;
  neatInstance.options.mutationAmount =
    neatInstance.options.mutationAmount ?? SMALL_POPULATION_MUTATION_AMOUNT;
}
