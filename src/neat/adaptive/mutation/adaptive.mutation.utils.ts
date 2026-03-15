import {
  ANNEAL_BASELINE_GENERATIONS,
  ANNEAL_PROGRESS_MAX,
  DEFAULT_ADAPT_EVERY,
  DEFAULT_INITIAL_MUTATION_RATE,
  DEFAULT_MAX_MUTATION_AMOUNT,
  DEFAULT_MAX_MUTATION_RATE,
  DEFAULT_MIN_MUTATION_AMOUNT,
  DEFAULT_MIN_MUTATION_RATE,
  DEFAULT_MUTATION_AMOUNT,
  DEFAULT_MUTATION_AMOUNT_SIGMA,
  DEFAULT_MUTATION_SIGMA,
  EXPLORE_LOW_DECREASE_MULTIPLIER,
  EXPLORE_LOW_INCREASE_MULTIPLIER,
  HALF_INDEX_DIVISOR,
  MUTATION_SIGMA_SCALE,
  MUTATION_STRATEGY_ANNEAL,
  MUTATION_STRATEGY_EXPLORE_LOW,
  MUTATION_STRATEGY_TWO_TIER,
  RNG_CENTER_OFFSET,
  RNG_SPREAD_MULTIPLIER,
  ZERO,
} from '../core/adaptive.core.constants';
import type {
  AdaptiveMutationConfig,
  Genome,
  MutationOutcome,
  MutationPartitions,
  MutationSettings,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

/**
 * Check whether mutation adaptation should run this generation.
 *
 * @param generation - Current generation index.
 * @param config - Adaptive mutation configuration.
 * @returns True if adaptation should run.
 */
export function shouldAdaptThisGeneration(
  generation: number,
  config: AdaptiveMutationConfig,
): boolean {
  const adaptEvery = config.adaptEvery ?? DEFAULT_ADAPT_EVERY;
  return adaptEvery <= DEFAULT_ADAPT_EVERY || generation % adaptEvery === ZERO;
}

/**
 * Collect genomes with numeric scores.
 *
 * @param population - Population of genomes.
 * @returns Scored genomes.
 */
export function collectScoredGenomes(population: Genome[]): Genome[] {
  return population.filter((genome) => typeof genome.score === 'number');
}

/**
 * Sort scored genomes in ascending score order.
 *
 * @param scoredGenomes - Scored genomes.
 * @returns Sorted genomes.
 */
export function sortScoredGenomes(scoredGenomes: Genome[]): Genome[] {
  return scoredGenomes.toSorted(
    (leftGenome, rightGenome) =>
      (leftGenome.score ?? ZERO) - (rightGenome.score ?? ZERO),
  );
}

/**
 * Split scored genomes into top and bottom halves.
 *
 * @param scoredGenomes - Sorted scored genomes.
 * @returns Partitions used by strategy rules.
 */
export function splitScoredGenomes(
  scoredGenomes: Genome[],
): MutationPartitions {
  const halfIndex = Math.floor(scoredGenomes.length / HALF_INDEX_DIVISOR);
  const bottomHalf = scoredGenomes.slice(ZERO, halfIndex);
  const topHalf = scoredGenomes.slice(halfIndex);
  return { topHalf, bottomHalf };
}

/**
 * Resolve mutation settings derived from configuration and engine state.
 *
 * @param engine - NEAT engine instance.
 * @param config - Adaptive mutation configuration.
 * @returns Resolved mutation settings.
 */
export function resolveMutationSettings(
  engine: NeatLikeWithAdaptive,
  config: AdaptiveMutationConfig,
): MutationSettings {
  const sigmaBase =
    (config.sigma ?? DEFAULT_MUTATION_SIGMA) * MUTATION_SIGMA_SCALE;
  const minRate = config.minRate ?? DEFAULT_MIN_MUTATION_RATE;
  const maxRate = config.maxRate ?? DEFAULT_MAX_MUTATION_RATE;
  const strategy = config.strategy ?? MUTATION_STRATEGY_TWO_TIER;
  const initialRate = config.initialRate ?? DEFAULT_INITIAL_MUTATION_RATE;
  const adaptAmount = config.adaptAmount ?? false;
  const amountSigma = config.amountSigma ?? DEFAULT_MUTATION_AMOUNT_SIGMA;
  const minAmount = config.minAmount ?? DEFAULT_MIN_MUTATION_AMOUNT;
  const maxAmount = config.maxAmount ?? DEFAULT_MAX_MUTATION_AMOUNT;
  const mutationAmountDefault =
    engine.options.mutationAmount ?? DEFAULT_MUTATION_AMOUNT;
  const generation = engine.generation;
  const populationSize = engine.population.length;

  return {
    strategy,
    sigmaBase,
    minRate,
    maxRate,
    initialRate,
    adaptAmount,
    amountSigma,
    minAmount,
    maxAmount,
    mutationAmountDefault,
    generation,
    populationSize,
  };
}

/**
 * Resolve a random source that matches the legacy RNG usage.
 *
 * @param engine - NEAT engine instance.
 * @returns Random number provider.
 */
export function resolveRandomSource(
  engine: NeatLikeWithAdaptive,
): () => number {
  const rngFactory = engine._getRNG ?? (() => Math.random);
  return () => rngFactory()();
}

/**
 * Apply mutation updates to the population.
 *
 * @param population - Full population to mutate.
 * @param partitions - Scored partitions.
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @returns Mutation outcome flags.
 */
export function applyMutationsToPopulation(
  population: Genome[],
  partitions: MutationPartitions,
  settings: MutationSettings,
  randomSource: () => number,
): MutationOutcome {
  const topHalfSet = new Set(partitions.topHalf);
  const bottomHalfSet = new Set(partitions.bottomHalf);
  let hasIncrease = false;
  let hasDecrease = false;

  for (let genomeIndex = ZERO; genomeIndex < population.length; genomeIndex++) {
    const genome = population[genomeIndex];
    if (genome._mutRate === undefined || genome._mutRate === null) continue;

    const rateDelta = resolveRateDelta(
      settings,
      randomSource,
      genome,
      genomeIndex,
      topHalfSet,
      bottomHalfSet,
    );
    const mutationRate = clampValue(
      genome._mutRate + rateDelta,
      settings.minRate,
      settings.maxRate,
    );

    if (mutationRate > settings.initialRate) hasIncrease = true;
    if (mutationRate < settings.initialRate) hasDecrease = true;

    genome._mutRate = mutationRate;

    if (settings.adaptAmount) {
      applyMutationAmount(
        genome,
        settings,
        randomSource,
        genomeIndex,
        topHalfSet,
        bottomHalfSet,
      );
    }
  }

  return { hasIncrease, hasDecrease };
}

/**
 * Resolve mutation-rate delta based on strategy.
 *
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Signed mutation rate delta.
 */
export function resolveRateDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  const baseDelta = createRandomDelta(settings.sigmaBase, randomSource);

  if (settings.strategy === MUTATION_STRATEGY_TWO_TIER) {
    return applyTwoTierDelta(
      baseDelta,
      genome,
      genomeIndex,
      topHalfSet,
      bottomHalfSet,
    );
  }

  if (settings.strategy === MUTATION_STRATEGY_EXPLORE_LOW) {
    return applyExploreLowDelta(baseDelta, genome, bottomHalfSet);
  }

  if (settings.strategy === MUTATION_STRATEGY_ANNEAL) {
    return applyAnnealDelta(baseDelta, settings);
  }

  return baseDelta;
}

/**
 * Create a signed random delta scaled by sigma.
 *
 * @param sigmaBase - Sigma scaling factor.
 * @param randomSource - Random number provider.
 * @returns Signed delta.
 */
export function createRandomDelta(
  sigmaBase: number,
  randomSource: () => number,
): number {
  const baseUnit = randomSource() * RNG_SPREAD_MULTIPLIER - RNG_CENTER_OFFSET;
  return baseUnit * sigmaBase;
}

/**
 * Apply two-tier adjustments to a delta.
 *
 * @param baseDelta - Base random delta.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Adjusted delta.
 */
export function applyTwoTierDelta(
  baseDelta: number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  if (!topHalfSet.size || !bottomHalfSet.size) {
    const isEvenIndex = genomeIndex % HALF_INDEX_DIVISOR === ZERO;
    return isEvenIndex ? Math.abs(baseDelta) : -Math.abs(baseDelta);
  }

  if (topHalfSet.has(genome)) return -Math.abs(baseDelta);
  if (bottomHalfSet.has(genome)) return Math.abs(baseDelta);
  return baseDelta;
}

/**
 * Apply explore-low adjustments to a delta.
 *
 * @param baseDelta - Base random delta.
 * @param genome - Current genome.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Adjusted delta.
 */
export function applyExploreLowDelta(
  baseDelta: number,
  genome: Genome,
  bottomHalfSet: Set<Genome>,
): number {
  if (bottomHalfSet.has(genome)) {
    return Math.abs(baseDelta * EXPLORE_LOW_INCREASE_MULTIPLIER);
  }

  return -Math.abs(baseDelta * EXPLORE_LOW_DECREASE_MULTIPLIER);
}

/**
 * Apply annealing adjustments to a delta.
 *
 * @param baseDelta - Base random delta.
 * @param settings - Resolved settings.
 * @returns Adjusted delta.
 */
export function applyAnnealDelta(
  baseDelta: number,
  settings: MutationSettings,
): number {
  const progress = Math.min(
    ANNEAL_PROGRESS_MAX,
    settings.generation /
      (ANNEAL_BASELINE_GENERATIONS + settings.populationSize),
  );
  return baseDelta * (ANNEAL_PROGRESS_MAX - progress);
}

/**
 * Apply mutation-amount adjustments to a genome.
 *
 * @param genome - Current genome.
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns {void}
 */
export function applyMutationAmount(
  genome: Genome,
  settings: MutationSettings,
  randomSource: () => number,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): void {
  const amountDelta = resolveAmountDelta(
    settings,
    randomSource,
    genome,
    genomeIndex,
    topHalfSet,
    bottomHalfSet,
  );
  const mutationAmount = clampValue(
    Math.round(
      (genome._mutAmount ?? settings.mutationAmountDefault) + amountDelta,
    ),
    settings.minAmount,
    settings.maxAmount,
  );
  genome._mutAmount = mutationAmount;
}

/**
 * Resolve mutation-amount delta based on strategy.
 *
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Signed mutation amount delta.
 */
export function resolveAmountDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  const baseDelta = createRandomDelta(settings.amountSigma, randomSource);

  if (settings.strategy === MUTATION_STRATEGY_TWO_TIER) {
    return applyTwoTierAmountDelta(
      baseDelta,
      genome,
      genomeIndex,
      topHalfSet,
      bottomHalfSet,
    );
  }

  return baseDelta;
}

/**
 * Apply two-tier adjustments to amount delta.
 *
 * @param baseDelta - Base random delta.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Adjusted delta.
 */
export function applyTwoTierAmountDelta(
  baseDelta: number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  if (!topHalfSet.size || !bottomHalfSet.size) {
    const isEvenIndex = genomeIndex % HALF_INDEX_DIVISOR === ZERO;
    return isEvenIndex ? Math.abs(baseDelta) : -Math.abs(baseDelta);
  }

  if (bottomHalfSet.has(genome)) return Math.abs(baseDelta);
  if (topHalfSet.has(genome)) return -Math.abs(baseDelta);
  return baseDelta;
}

/**
 * Clamp a value between min and max bounds.
 *
 * @param value - Value to clamp.
 * @param min - Minimum bound.
 * @param max - Maximum bound.
 * @returns Clamped value.
 */
export function clampValue(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/**
 * Determine whether a two-tier fallback is needed.
 *
 * @param strategy - Mutation strategy identifier.
 * @param outcome - Mutation outcome flags.
 * @returns True if fallback should run.
 */
export function shouldApplyTwoTierFallback(
  strategy: string,
  outcome: MutationOutcome,
): boolean {
  if (strategy !== MUTATION_STRATEGY_TWO_TIER) return false;
  return !(outcome.hasIncrease && outcome.hasDecrease);
}

/**
 * Apply two-tier fallback balancing.
 *
 * @param population - Population of genomes.
 * @param settings - Resolved settings.
 * @returns {void}
 */
export function applyTwoTierFallback(
  population: Genome[],
  settings: MutationSettings,
): void {
  const halfIndex = Math.floor(population.length / HALF_INDEX_DIVISOR);

  for (let genomeIndex = ZERO; genomeIndex < population.length; genomeIndex++) {
    const genome = population[genomeIndex];
    if (genome._mutRate === undefined || genome._mutRate === null) continue;

    const adjustedRate =
      genomeIndex < halfIndex
        ? genome._mutRate + settings.sigmaBase
        : genome._mutRate - settings.sigmaBase;
    genome._mutRate = clampValue(
      adjustedRate,
      settings.minRate,
      settings.maxRate,
    );
  }
}
