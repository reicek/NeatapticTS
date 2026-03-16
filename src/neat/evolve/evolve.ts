/**
 * The evolution chapter is where one NEAT generation actually turns into the next.
 *
 * If the root `Neat` controller explains the public lifecycle, this module explains the
 * orchestration core that makes that lifecycle real. The exported `evolve()` function is
 * intentionally long in responsibility but short in local decision-making: it reads like a
 * stage manager for one generation and delegates the detailed work to focused helpers under
 * `adaptive/`, `objectives/`, `population/`, `runtime/`, `speciation/`, and `telemetry/`.
 * That split matters because the controller has to coordinate many policies at once without
 * collapsing them into one opaque algorithm blob.
 *
 * Read this chapter when you want to answer questions such as:
 *
 * - When does the controller force evaluation before selection starts?
 * - Which adaptive hooks run before and after ranking?
 * - Where do speciation, offspring construction, mutation, pruning, and telemetry fit?
 * - Why does the method return the best network from the previous generation instead of the
 *   newly mutated population?
 *
 * The high-level flow is easier to retain if you treat it as five phases:
 *
 * 1. make the current population comparable,
 * 2. update adaptive and objective policies,
 * 3. rank and summarize the current generation,
 * 4. construct the next population,
 * 5. finalize runtime state for the next loop.
 *
 * ```mermaid
 * flowchart TD
 *   Evaluate[Evaluate current population] --> Adapt[Apply adaptive controllers]
 *   Adapt --> Rank[Sort, rank, and speciate]
 *   Rank --> Snapshot[Snapshot best genome and telemetry evidence]
 *   Snapshot --> Rebuild[Build next population via elitism, provenance, and offspring]
 *   Rebuild --> Mutate[Apply pruning, mutation, and cache invalidation]
 *   Mutate --> Finalize[Advance generation and maintain runtime history]
 * ```
 *
 * Reading order:
 * - start with {@link evolve} for the generation spine,
 * - jump into `runtime/` when you need evaluation and timekeeping semantics,
 * - jump into `speciation/` and `population/` when parent selection or offspring allocation is the question,
 * - jump into `adaptive/` when you need to understand dynamic policy changes,
 * - jump into `telemetry/` when you want to know which evidence gets recorded during the step.
 */
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

import Network from '../../architecture/network';
import { processMultiObjective } from '../multiobjective/category/multiobjective.category';
import {
  applyAdaptiveComplexityControllers,
  applyAncestorUniqAdaptiveSafe,
  applyAutoCompatibilityTuning,
  applyMinimalCriterionAdaptiveSafe,
  applyOperatorAdaptationSafe,
  applyPruningAndMutation,
  adaptReenableProbability,
  invalidateCompatibilityCaches,
} from './adaptive/evolve.adaptive.utils';
import {
  updateObjectiveScheduleAndAges,
  applyFitnessSuppressionForTests,
  captureObjectiveImportanceSnapshot,
  applyDynamicObjectiveSchedule,
  resetObjectivesCache,
} from './objectives/evolve.objectives.utils';
import {
  addOffspring,
  addSpeciatedOffspring,
  addUnspeciatedOffspring,
  applyElitism,
  applyProvenance,
  buildNextPopulation,
  enforcePopulationConstraints,
} from './population/evolve.population.utils';
import {
  buildFittestSnapshot,
  clearPopulationScores,
  computeElapsedTime,
  ensurePopulationEvaluated,
  resolveStartTime,
  trackGlobalImprovement,
  updateGlobalBestTracking,
} from './runtime/evolve.runtime.utils';
import {
  applyGlobalStagnationInjectionIfNeeded,
  applySpeciationAndSharingIfEnabled,
  buildFreshGenomeForStagnation,
  ensureSpeciesHistorySnapshot,
  recordSpeciesHistorySnapshot,
  updateSpeciesStagnationIfEnabled,
} from './speciation/evolve.speciation.utils';
import {
  recordTelemetryIfEnabled,
  computeDiversityStatsSafely,
} from './telemetry/evolve.telemetry.utils';
import type { NeatControllerForEvolution } from './evolve.types';

/**
 * Maximum number of Pareto archive snapshots to retain.
 *
 * This keeps multi-objective history useful for inspection without letting archive state
 * grow unbounded during long runs.
 */
export const EVOLVE_PARETO_ARCHIVE_MAX = 200;
/**
 * Minimum target front size used for adaptive epsilon tuning.
 *
 * Small Pareto fronts are easy to overfit, so the epsilon controller never aims below this floor.
 */
export const EVOLVE_TARGET_FRONT_MIN = 3;
/**
 * Upper ratio threshold for Pareto front size vs target.
 *
 * When the first front grows beyond this band, the controller can tighten epsilon to recover pressure.
 */
export const EVOLVE_TARGET_FRONT_UPPER_RATIO = 1.2;
/**
 * Lower ratio threshold for Pareto front size vs target.
 *
 * When the first front shrinks below this band, the controller can relax epsilon to avoid over-pruning.
 */
export const EVOLVE_TARGET_FRONT_LOWER_RATIO = 0.8;
/**
 * Default adjustment step for dominance epsilon.
 *
 * The value is intentionally small because epsilon changes should steer ranking gradually rather than jerk it.
 */
export const EVOLVE_DEFAULT_EPSILON_ADJUST = 0.002;
/** Minimum dominance epsilon floor used by adaptive Pareto tuning. */
export const EVOLVE_DEFAULT_EPSILON_MIN = 0;
/** Maximum dominance epsilon ceiling used by adaptive Pareto tuning. */
export const EVOLVE_DEFAULT_EPSILON_MAX = 0.5;
/**
 * Default cooldown in generations between epsilon adjustments.
 *
 * This prevents the controller from reacting to every short-lived fluctuation in front width.
 */
export const EVOLVE_DEFAULT_EPSILON_COOLDOWN = 2;
/**
 * Default inactivity window used before adaptive objective pruning considers removal.
 */
export const EVOLVE_PRUNE_WINDOW_DEFAULT = 5;
/**
 * Default numerical range epsilon for deciding whether an objective has effectively gone flat.
 */
export const EVOLVE_PRUNE_RANGE_EPS_DEFAULT = 1e-6;
/** Generation threshold below which a species is still treated as young. */
export const EVOLVE_YOUNG_THRESHOLD_DEFAULT = 5;
/** Fitness-sharing multiplier applied to species that are still in their early growth window. */
export const EVOLVE_YOUNG_MULTIPLIER_DEFAULT = 1.3;
/** Generation threshold after which a species is treated as old for age-based fitness shaping. */
export const EVOLVE_OLD_THRESHOLD_DEFAULT = 30;
/** Fitness-sharing multiplier applied to older species so stale lineages lose selection privilege. */
export const EVOLVE_OLD_MULTIPLIER_DEFAULT = 0.7;
/**
 * Minimum offspring allocation reserved for a surviving species during speciated reproduction.
 */
export const EVOLVE_MIN_OFFSPRING_DEFAULT = 1;
/**
 * Survivor fraction used when choosing the parent pool inside each species.
 */
export const EVOLVE_SURVIVAL_THRESHOLD_DEFAULT = 0.5;
/**
 * Retry limit for cross-species parent sampling before the controller falls back to a safer path.
 */
export const EVOLVE_CROSS_SPECIES_GUARD_LIMIT = 5;
/**
 * Default generation at which automatic entropy objective scheduling becomes eligible.
 */
export const EVOLVE_AUTO_ENTROPY_ADD_AT = 3;
/**
 * Fraction of the population replaced with fresh genomes when global stagnation rescue triggers.
 */
export const EVOLVE_GLOBAL_STAGNATION_REPLACE_FRACTION = 0.2;
/**
 * Minimum re-enable observations required before the controller trusts its adaptation signal.
 */
export const EVOLVE_REENABLE_MIN_SAMPLES = 20;
/** Desired success ratio for connection re-enable attempts during adaptive mutation control. */
export const EVOLVE_REENABLE_TARGET = 0.3;
/**
 * Scale factor that converts re-enable success error into a probability update.
 */
export const EVOLVE_REENABLE_DELTA_SCALE = 0.1;
/** Lower bound for adaptive connection re-enable probability. */
export const EVOLVE_REENABLE_MIN = 0.05;
/** Upper bound for adaptive connection re-enable probability. */
export const EVOLVE_REENABLE_MAX = 0.9;
/**
 * Minimum target species count used by automatic compatibility tuning.
 *
 * The controller never tries to collapse diversity below this floor when adjusting coefficients.
 */
export const EVOLVE_AUTO_COMPAT_TARGET_MIN = 2;
/**
 * Default rate used when nudging compatibility coefficients toward the desired species count.
 */
export const EVOLVE_AUTO_COMPAT_ADJUST_RATE = 0.01;
/** Minimum compatibility coefficient allowed during automatic tuning. */
export const EVOLVE_AUTO_COMPAT_MIN_COEFF = 0.1;
/** Maximum compatibility coefficient allowed during automatic tuning. */
export const EVOLVE_AUTO_COMPAT_MAX_COEFF = 5;
/**
 * Random perturbation scale used when compatibility tuning has no directional error to follow.
 */
export const EVOLVE_AUTO_COMPAT_RANDOM_SCALE = 0.5;
/**
 * Maximum number of species-history snapshots to retain for telemetry and later export.
 */
export const EVOLVE_SPECIES_HISTORY_MAX = 200;

/**
 * Run a single evolution step for this NEAT population.
 *
 * This is the orchestration spine for one full generation update. The method does not try to
 * implement every evolutionary policy inline; instead, it coordinates the major phases in a fixed
 * order so the rest of the NEAT controller can remain modular and inspectable.
 *
 * The lifecycle is easiest to read in seven stages:
 *
 * 1. ensure the current population has fresh evaluation scores,
 * 2. run adaptive controllers that may change complexity or objective policy,
 * 3. rank the current generation through sorting, multi-objective processing, and speciation,
 * 4. capture the best-network snapshot plus telemetry evidence while the generation is still intact,
 * 5. build the next population through elitism, provenance, and offspring allocation,
 * 6. mutate and prune the newly built population,
 * 7. finalize runtime bookkeeping for the next call.
 *
 * A subtle but important design choice is that the returned {@link Network} represents the best
 * genome from the generation that was just analyzed, not from the population after mutation. That
 * makes the return value a stable evaluation artifact: callers can inspect or replay the winning
 * candidate without worrying that post-selection mutation has already changed it.
 *
 * The method mutates controller-owned state such as `this.population`, `this.generation`,
 * compatibility caches, telemetry buffers, objective snapshots, and species-history state.
 * In other words, call this when you intend to advance the controller, not when you merely want a
 * read-only score refresh.
 *
 * Important side-effects:
 * - Replaces `this.population` with the newly constructed generation.
 * - Increments `this.generation`.
 * - May register or remove dynamic objectives via adaptive controllers.
 * - Refreshes telemetry, diversity, species-history, and timing snapshots.
 *
 * Use the surrounding helper folders as the next reading map:
 * - `runtime/` explains evaluation readiness and loop timing.
 * - `adaptive/` explains policy changes that respond to stagnation or controller statistics.
 * - `speciation/` explains fitness sharing, compatibility tuning, and species history.
 * - `population/` explains how elitism, provenance, and offspring fill the next generation.
 * - `telemetry/` explains which traces are recorded while this method runs.
 *
 * Example:
 *
 * ```ts
 * // assuming `neat` is an instance with configured population/options
 * const bestNetwork = await neat.evolve();
 * console.log('generation:', neat.generation);
 * console.log('output nodes:', bestNetwork.output);
 * ```
 *
 * @this {NeatControllerForEvolution} the NEAT instance (contains population, options, RNG, etc.)
 * @returns {Promise<Network>} a deep-cloned network snapshot representing the best genome from the
 * previous generation, ready for inspection or external evaluation
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
