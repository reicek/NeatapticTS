/*
 * ESLint configuration for intentional `any` usage in NEAT evolution module
 *
 * This file uses `any` strategically for:
 * 1. Runtime genome metadata properties that are dynamically added during evolution
 *    (_id, _parents, _depth, _moRank, _moCrowd, _sharedFitness, etc.)
 * 2. Species members and population arrays that contain mixed metadata
 *    - Runtime behavior guarantees type safety beyond what TypeScript can infer
 * 3. Dynamic multi-objective optimization structures (paretoFronts, objective accessors)
 *    - Complex nested structures with varying runtime shapes
 * 4. Telemetry and diversity stat calculations
 *    - Generic accessor functions that work across different genome properties
 * 5. Type system bridging between GenomeWithMetadata and Network
 *    - Where runtime contracts are sound but TypeScript can't prove it statically
 *
 * All `any` usage here is intentional, documented, and necessary for the evolution architecture.
 */

import Network from '../architecture/network';
import type { NeatControllerForEvolution } from './neat.evolve.types';
import {
  addOffspring,
  addSpeciatedOffspring,
  addUnspeciatedOffspring,
  applyAdaptiveComplexityControllers,
  applyAncestorUniqAdaptiveSafe,
  applyAutoCompatibilityTuning,
  applyDynamicObjectiveSchedule,
  applyElitism,
  applyFitnessSuppressionForTests,
  applyGlobalStagnationInjectionIfNeeded,
  applyMinimalCriterionAdaptiveSafe,
  applyOperatorAdaptationSafe,
  applyPruningAndMutation,
  applyProvenance,
  applySpeciationAndSharingIfEnabled,
  adaptReenableProbability,
  buildFittestSnapshot,
  buildFreshGenomeForStagnation,
  buildNextPopulation,
  captureObjectiveImportanceSnapshot,
  clearPopulationScores,
  computeDiversityStatsSafely,
  computeElapsedTime,
  ensurePopulationEvaluated,
  ensureSpeciesHistorySnapshot,
  enforcePopulationConstraints,
  invalidateCompatibilityCaches,
  processMultiObjective,
  recordSpeciesHistorySnapshot,
  recordTelemetryIfEnabled,
  resetObjectivesCache,
  resolveStartTime,
  trackGlobalImprovement,
  updateGlobalBestTracking,
  updateObjectiveScheduleAndAges,
  updateSpeciesStagnationIfEnabled,
} from './neat.evolve.utils';

/** Maximum number of Pareto archive snapshots to retain. */
export const EVOLVE_PARETO_ARCHIVE_MAX = 200;
/** Minimum target front size used for adaptive epsilon tuning. */
export const EVOLVE_TARGET_FRONT_MIN = 3;
/** Upper ratio threshold for Pareto front size vs target. */
export const EVOLVE_TARGET_FRONT_UPPER_RATIO = 1.2;
/** Lower ratio threshold for Pareto front size vs target. */
export const EVOLVE_TARGET_FRONT_LOWER_RATIO = 0.8;
/** Default adjustment step for dominance epsilon. */
export const EVOLVE_DEFAULT_EPSILON_ADJUST = 0.002;
/** Default minimum dominance epsilon. */
export const EVOLVE_DEFAULT_EPSILON_MIN = 0;
/** Default maximum dominance epsilon. */
export const EVOLVE_DEFAULT_EPSILON_MAX = 0.5;
/** Default cooldown (generations) between epsilon adjustments. */
export const EVOLVE_DEFAULT_EPSILON_COOLDOWN = 2;
/** Default prune window (generations) for inactive objectives. */
export const EVOLVE_PRUNE_WINDOW_DEFAULT = 5;
/** Default inactive objective range epsilon. */
export const EVOLVE_PRUNE_RANGE_EPS_DEFAULT = 1e-6;
/** Default young species threshold (generations). */
export const EVOLVE_YOUNG_THRESHOLD_DEFAULT = 5;
/** Default young species fitness multiplier. */
export const EVOLVE_YOUNG_MULTIPLIER_DEFAULT = 1.3;
/** Default old species threshold (generations). */
export const EVOLVE_OLD_THRESHOLD_DEFAULT = 30;
/** Default old species fitness multiplier. */
export const EVOLVE_OLD_MULTIPLIER_DEFAULT = 0.7;
/** Default minimum offspring per species. */
export const EVOLVE_MIN_OFFSPRING_DEFAULT = 1;
/** Default survival threshold for parent selection. */
export const EVOLVE_SURVIVAL_THRESHOLD_DEFAULT = 0.5;
/** Guard limit for cross-species mating selection retries. */
export const EVOLVE_CROSS_SPECIES_GUARD_LIMIT = 5;
/** Default auto-entropy activation generation. */
export const EVOLVE_AUTO_ENTROPY_ADD_AT = 3;
/** Fraction of population to replace during global stagnation injection. */
export const EVOLVE_GLOBAL_STAGNATION_REPLACE_FRACTION = 0.2;
/** Minimum samples required to adjust re-enable probability. */
export const EVOLVE_REENABLE_MIN_SAMPLES = 20;
/** Target re-enable success ratio. */
export const EVOLVE_REENABLE_TARGET = 0.3;
/** Scale factor for re-enable probability adjustment. */
export const EVOLVE_REENABLE_DELTA_SCALE = 0.1;
/** Minimum re-enable probability. */
export const EVOLVE_REENABLE_MIN = 0.05;
/** Maximum re-enable probability. */
export const EVOLVE_REENABLE_MAX = 0.9;
/** Minimum target species when auto-tuning compatibility coefficients. */
export const EVOLVE_AUTO_COMPAT_TARGET_MIN = 2;
/** Default auto-compatibility adjust rate. */
export const EVOLVE_AUTO_COMPAT_ADJUST_RATE = 0.01;
/** Default minimum compatibility coefficient. */
export const EVOLVE_AUTO_COMPAT_MIN_COEFF = 0.1;
/** Default maximum compatibility coefficient. */
export const EVOLVE_AUTO_COMPAT_MAX_COEFF = 5;
/** Random scale factor used when auto-compatibility has zero error. */
export const EVOLVE_AUTO_COMPAT_RANDOM_SCALE = 0.5;
/** Maximum number of species history snapshots to retain. */
export const EVOLVE_SPECIES_HISTORY_MAX = 200;

/**
 * Run a single evolution step for this NEAT population.
 *
 * This method performs a full generation update: evaluation (if needed),
 * adaptive hooks, speciation and fitness sharing, multi-objective
 * processing, elitism/provenance, offspring allocation (within or without
 * species), mutation, pruning, and telemetry recording. It mutates the
 * controller state (`this.population`, `this.generation`, and telemetry
 * caches) and returns a copy of the best discovered `Network` for the
 * generation.
 *
 * Important side-effects:
 * - Replaces `this.population` with the newly constructed generation.
 * - Increments `this.generation`.
 * - May register or remove dynamic objectives via adaptive controllers.
 *
 * Example:
 * // assuming `neat` is an instance with configured population/options
 * await neat.evolve();
 * console.log('generation:', neat.generation);
 *
 * @this {NeatControllerForEvolution} the NEAT instance (contains population, options, RNG, etc.)
 * @returns {Promise<Network>} a deep-cloned Network representing the best genome
 *                              in the previous generation (useful for evaluation)
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6 Instinct: neuro-evolution on steroids by Thomas Wagenaar}
 */
export async function evolve(
  this: NeatControllerForEvolution,
): Promise<Network> {
  const internal = this as unknown as NeatControllerForEvolution;
  const startTime = resolveStartTime();

  // Step 1: Ensure population has scores before any selection/sorting.
  await ensurePopulationEvaluated(internal);
  // Step 2: Reset objective cache so dynamic schedules can rebuild.
  resetObjectivesCache(internal);
  // Step 3: Apply adaptive complexity controllers (optional hooks).
  await applyAdaptiveComplexityControllers(internal);
  // Step 4: Sort population by current fitness/criteria.
  internal.sort();
  // Step 5: Track best score for global stagnation logic.
  updateGlobalBestTracking(internal);
  // Step 6: Apply minimal-criterion adaptation if configured.
  await applyMinimalCriterionAdaptiveSafe(internal);
  // Step 7: Compute diversity stats for adaptive/telemetry use.
  computeDiversityStatsSafely(internal);
  // Step 8: Run multi-objective ranking and archives if enabled.
  if (internal.options.multiObjective?.enabled) {
    processMultiObjective(internal, {
      paretoArchiveMax: EVOLVE_PARETO_ARCHIVE_MAX,
      targetFrontMin: EVOLVE_TARGET_FRONT_MIN,
      targetFrontUpperRatio: EVOLVE_TARGET_FRONT_UPPER_RATIO,
      targetFrontLowerRatio: EVOLVE_TARGET_FRONT_LOWER_RATIO,
      defaultEpsilonAdjust: EVOLVE_DEFAULT_EPSILON_ADJUST,
      defaultEpsilonMin: EVOLVE_DEFAULT_EPSILON_MIN,
      defaultEpsilonMax: EVOLVE_DEFAULT_EPSILON_MAX,
      defaultEpsilonCooldown: EVOLVE_DEFAULT_EPSILON_COOLDOWN,
      pruneWindowDefault: EVOLVE_PRUNE_WINDOW_DEFAULT,
      pruneRangeEpsDefault: EVOLVE_PRUNE_RANGE_EPS_DEFAULT,
    });
  }
  // Step 9: Apply ancestor-uniqueness adaptation (optional).
  await applyAncestorUniqAdaptiveSafe(internal);
  // Step 10: Speciate, share fitness, and snapshot species history.
  await applySpeciationAndSharingIfEnabled(internal, {
    applyAutoCompatibilityTuning: () =>
      applyAutoCompatibilityTuning(internal, {
        targetMin: EVOLVE_AUTO_COMPAT_TARGET_MIN,
        adjustRate: EVOLVE_AUTO_COMPAT_ADJUST_RATE,
        minCoeff: EVOLVE_AUTO_COMPAT_MIN_COEFF,
        maxCoeff: EVOLVE_AUTO_COMPAT_MAX_COEFF,
        randomScale: EVOLVE_AUTO_COMPAT_RANDOM_SCALE,
      }),
    recordSpeciesHistorySnapshot: () =>
      recordSpeciesHistorySnapshot(internal, EVOLVE_SPECIES_HISTORY_MAX),
  });

  // Step 11: Snapshot best genome into a Network.
  const fittest = buildFittestSnapshot(internal);
  // Step 12: Recompute diversity stats for telemetry snapshot.
  computeDiversityStatsSafely(internal);
  // Step 13: Update objective schedules and age tracking.
  await updateObjectiveScheduleAndAges(internal, {
    applyDynamicObjectiveSchedule: (currentObjectiveKeys) =>
      applyDynamicObjectiveSchedule(internal, currentObjectiveKeys, {
        autoEntropyAddAt: EVOLVE_AUTO_ENTROPY_ADD_AT,
      }),
  });
  // Step 14: Apply test-specific objective suppression if needed.
  applyFitnessSuppressionForTests(internal);
  // Step 15: Capture objective importance metrics for telemetry.
  captureObjectiveImportanceSnapshot(internal);
  // Step 16: Record telemetry entry if enabled.
  await recordTelemetryIfEnabled(internal, fittest);
  // Step 17: Track global improvement for stagnation windows.
  trackGlobalImprovement(internal, fittest);

  // Step 18: Build next population via elitism, provenance, offspring.
  const nextPopulation = await buildNextPopulation(internal, {
    addOffspring: (nextPopulation) =>
      addOffspring(internal, nextPopulation, {
        addSpeciatedOffspring: (nextPopulation, remainingSlots) =>
          addSpeciatedOffspring(internal, nextPopulation, remainingSlots, {
            minOffspringDefault: EVOLVE_MIN_OFFSPRING_DEFAULT,
            survivalThresholdDefault: EVOLVE_SURVIVAL_THRESHOLD_DEFAULT,
            youngThresholdDefault: EVOLVE_YOUNG_THRESHOLD_DEFAULT,
            youngMultiplierDefault: EVOLVE_YOUNG_MULTIPLIER_DEFAULT,
            oldThresholdDefault: EVOLVE_OLD_THRESHOLD_DEFAULT,
            oldMultiplierDefault: EVOLVE_OLD_MULTIPLIER_DEFAULT,
            crossSpeciesGuardLimit: EVOLVE_CROSS_SPECIES_GUARD_LIMIT,
          }),
        addUnspeciatedOffspring: (nextPopulation, remainingSlots) =>
          addUnspeciatedOffspring(internal, nextPopulation, remainingSlots),
      }),
    applyElitism: (nextPopulation) => applyElitism(internal, nextPopulation),
    applyProvenance: (nextPopulation) =>
      applyProvenance(internal, nextPopulation),
  });
  // Step 19: Enforce structural constraints on new genomes.
  await enforcePopulationConstraints(internal, nextPopulation);
  // Step 20: Replace population with new generation.
  internal.population = nextPopulation as never;

  // Step 21: Apply pruning and mutation phases.
  await applyPruningAndMutation(internal);
  // Step 22: Clear compatibility caches after structural changes.
  invalidateCompatibilityCaches(internal);
  //   Step 23: Reset scores to force re-evaluation next generation.
  clearPopulationScores(internal);
  // Step 24: Increment generation counter.
  internal.generation++;
  // Step 25: Update species stagnation tracking.
  updateSpeciesStagnationIfEnabled(internal);
  // Step 26: Inject diversity if global stagnation threshold hit.
  await applyGlobalStagnationInjectionIfNeeded(internal, {
    buildFreshGenomeForStagnation: () =>
      buildFreshGenomeForStagnation(internal),
    replaceFraction: EVOLVE_GLOBAL_STAGNATION_REPLACE_FRACTION,
  });
  // Step 27: Adapt re-enable probability based on recent stats.
  adaptReenableProbability(internal, {
    minSamples: EVOLVE_REENABLE_MIN_SAMPLES,
    target: EVOLVE_REENABLE_TARGET,
    min: EVOLVE_REENABLE_MIN,
    max: EVOLVE_REENABLE_MAX,
    deltaScale: EVOLVE_REENABLE_DELTA_SCALE,
  });
  // Step 28: Apply operator adaptation (optional hook).
  await applyOperatorAdaptationSafe(internal);

  // Step 29: Store elapsed evolve duration for telemetry.
  internal._lastEvolveDuration = computeElapsedTime(startTime);
  // Step 30: Ensure at least a minimal species history snapshot exists.
  ensureSpeciesHistorySnapshot(internal, EVOLVE_SPECIES_HISTORY_MAX);

  // Step 31: Return the best network snapshot from this generation.
  return fittest;
}
