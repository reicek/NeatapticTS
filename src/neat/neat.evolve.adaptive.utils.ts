/*
 * ESLint configuration for intentional `any` usage in NEAT evolution adaptive utils
 *
 * This file mirrors the evolution module's runtime metadata handling,
 * where dynamic properties are attached to genomes/species at runtime.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */

import type { NeatControllerForEvolution } from './neat.evolve.types';

/**
 * Apply adaptive complexity controllers if available.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export async function applyAdaptiveComplexityControllers(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Apply complexity budget controller (optional).
  try {
    const { applyComplexityBudget } = await import('./neat.adaptive');
    applyComplexityBudget.call(internal as never);
  } catch {
    // Empty catch: adaptive complexity budget may not be configured.
  }
  // Step 2: Apply phased complexity controller (optional).
  try {
    const { applyPhasedComplexity } = await import('./neat.adaptive');
    applyPhasedComplexity.call(internal as never);
  } catch {
    // Empty catch: phased complexity may not be configured.
  }
}

/**
 * Apply minimal criterion adaptive controller if available.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export async function applyMinimalCriterionAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Delegate to optional module.
  try {
    const { applyMinimalCriterionAdaptive } = await import('./neat.adaptive');
    applyMinimalCriterionAdaptive.call(internal as never);
  } catch {
    // Empty catch: minimal criterion adaptation may not be configured.
  }
}

/**
 * Apply ancestor uniqueness adaptation if available.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export async function applyAncestorUniqAdaptiveSafe(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Delegate to adaptive module when present.
  try {
    const adaptiveModule = await import('./neat.adaptive');
    adaptiveModule.applyAncestorUniqAdaptive.call(internal as never);
  } catch {
    // Empty catch: ancestor uniqueness adaptation is optional.
  }
}

/**
 * Apply auto-compatibility tuning if enabled.
 * @param internal - NEAT controller instance.
 * @param config - Tuning constants.
 * @returns void.
 */
export function applyAutoCompatibilityTuning(
  internal: NeatControllerForEvolution,
  config: {
    targetMin: number;
    adjustRate: number;
    minCoeff: number;
    maxCoeff: number;
    randomScale: number;
  },
): void {
  // Step 1: Guard for missing config.
  try {
    const options = internal.options as any;
    if (!options.autoCompatTuning?.enabled) return;
    // Step 2: Compute target species count and error.
    const target =
      options.autoCompatTuning.target ??
      options.targetSpecies ??
      Math.max(
        config.targetMin,
        Math.round(Math.sqrt(internal.population.length)),
      );
    const observed = (internal._species?.length ?? 0) || 1;
    const error = target - observed;
    const rate = options.autoCompatTuning.adjustRate ?? config.adjustRate;
    const minCoeff = options.autoCompatTuning.minCoeff ?? config.minCoeff;
    const maxCoeff = options.autoCompatTuning.maxCoeff ?? config.maxCoeff;
    // Step 3: Compute adjustment factor.
    let factor = 1 - rate * Math.sign(error);
    if (error === 0) {
      factor = 1 + (internal._getRNG()() - 0.5) * rate * config.randomScale;
    }
    // Step 4: Apply coefficient updates.
    options.excessCoeff = Math.min(
      maxCoeff,
      Math.max(minCoeff, options.excessCoeff * factor),
    );
    options.disjointCoeff = Math.min(
      maxCoeff,
      Math.max(minCoeff, options.disjointCoeff * factor),
    );
  } catch {
    // Empty catch: auto-compatibility tuning is optional.
  }
}

/**
 * Apply pruning and mutation phases.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export async function applyPruningAndMutation(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Apply evolution-time pruning.
  try {
    const pruningModule = await import('./neat.pruning');
    pruningModule.applyEvolutionPruning.call(internal as never);
  } catch {
    // Empty catch: evolution-time pruning is optional.
  }
  // Step 2: Apply adaptive pruning.
  try {
    const pruningModule = await import('./neat.pruning');
    pruningModule.applyAdaptivePruning.call(internal as never);
  } catch {
    // Empty catch: adaptive pruning is optional.
  }
  // Step 3: Mutate population.
  await internal.mutate?.();
  // Step 4: Apply adaptive mutation if available.
  try {
    const adaptiveModule = await import('./neat.adaptive');
    adaptiveModule.applyAdaptiveMutation.call(internal as never);
  } catch {
    // Empty catch: genome-level adaptive mutation is optional.
  }
}

/**
 * Invalidate compatibility caches after mutations.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function invalidateCompatibilityCaches(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Remove cached compatibility on each genome.
  internal.population.forEach((genome: any) => {
    if (genome._compatCache) delete genome._compatCache;
  });
}

/**
 * Adapt the re-enable probability based on recent success ratios.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function adaptReenableProbability(
  internal: NeatControllerForEvolution,
  config: {
    minSamples: number;
    target: number;
    min: number;
    max: number;
    deltaScale: number;
  },
): void {
  // Step 1: Guard for disabled configuration.
  if (internal.options.reenableProb === undefined) return;
  // Step 2: Aggregate success and attempt counters.
  let reenableSuccessTotal = 0;
  let reenableAttemptsTotal = 0;
  for (const genome of internal.population) {
    reenableSuccessTotal += (genome as any)._reenableSuccess || 0;
    reenableAttemptsTotal += (genome as any)._reenableAttempts || 0;
    (genome as any)._reenableSuccess = 0;
    (genome as any)._reenableAttempts = 0;
  }
  // Step 3: Adjust only with sufficient sample size.
  if (reenableAttemptsTotal > config.minSamples) {
    const ratio = reenableSuccessTotal / reenableAttemptsTotal;
    const delta = ratio - config.target;
    internal.options.reenableProb = Math.min(
      config.max,
      Math.max(
        config.min,
        (internal.options.reenableProb ?? config.target) -
          delta * config.deltaScale,
      ),
    );
  }
}

/**
 * Apply operator adaptation if available.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export async function applyOperatorAdaptationSafe(
  internal: NeatControllerForEvolution,
): Promise<void> {
  // Step 1: Delegate to adaptive module when present.
  try {
    const adaptiveModule = await import('./neat.adaptive');
    adaptiveModule.applyOperatorAdaptation.call(internal as never);
  } catch {
    // Empty catch: operator adaptation is optional.
  }
}
