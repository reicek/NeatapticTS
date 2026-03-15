/**
 * Adaptive NEAT controllers for complexity, acceptance, lineage diversity,
 * and mutation-rate tuning.
 *
 * This root entrypoint stays intentionally small: it explains the high-level
 * adaptive story, while the detailed heuristics now live in focused teaching
 * folders for complexity, mutation, acceptance, lineage, and shared runtime
 * vocabulary.
 */
import type { NeatLikeWithAdaptive as NeatLikeWithAdaptiveType } from './core/adaptive.core.types';
import { applyComplexityBudgetSchedule } from './complexity/adaptive.complexity.utils';
import {
  initializePhaseState,
  togglePhaseIfNeeded,
} from './complexity/adaptive.phases.utils';
import {
  initializeThreshold,
  collectScores,
  computeAcceptance,
  resolveTargetSettings,
  updateThreshold,
  applyRejection,
} from './acceptance/adaptive.minimal-criterion.utils';
import {
  isCooldownSatisfied,
  extractAncestorUniqueness,
  resolveUniquenessThresholds,
  resolveAdjustmentMagnitude,
  applyUniquenessAdjustment,
} from './lineage/adaptive.ancestor-uniqueness.utils';
import {
  shouldAdaptThisGeneration,
  collectScoredGenomes,
  sortScoredGenomes,
  splitScoredGenomes,
  resolveMutationSettings,
  resolveRandomSource,
  applyMutationsToPopulation,
  shouldApplyTwoTierFallback,
  applyTwoTierFallback,
} from './mutation/adaptive.mutation.utils';
import {
  resolveOperatorDecay,
  collectOperatorStatsEntries,
  applyOperatorDecay,
} from './mutation/adaptive.operator.utils';

export type { NeatLikeWithAdaptive } from './core/adaptive.core.types';

/**
 * Apply complexity budget scheduling to the evolving population.
 *
 * This routine updates `this.options.maxNodes` (and optionally
 * `this.options.maxConns`) according to a configured complexity budget
 * strategy. Two modes are supported:
 *
 * - `adaptive`: reacts to recent population improvement (or stagnation)
 *   by increasing or decreasing the current complexity cap using
 *   heuristics such as slope (linear trend) of recent best scores,
 *   novelty, and configured increase/stagnation factors.
 * - `linear` (default behaviour when not `adaptive`): linearly ramps
 *   the budget from `maxNodesStart` to `maxNodesEnd` over a horizon.
 *
 * Internal state used/maintained on the `this` object:
 * - `_cbHistory`: rolling window of best scores used to compute trends.
 * - `_cbMaxNodes`: current complexity budget for nodes.
 * - `_cbMaxConns`: current complexity budget for connections (optional).
 *
 * The method is intended to be called on the NEAT engine instance with
 * `this` bound appropriately (i.e. a NeatapticTS `Neat`-like object).
 *
 * @this {{
 *   options: any,
 *   population: Array<{score?: number}>,
 *   input: number,
 *   output: number,
 *   generation: number,
 *   _noveltyArchive?: any[]
 * }} NeatEngine
 *
 * @returns {void} Updates `this.options.maxNodes` and possibly
 * `this.options.maxConns` in-place; no value is returned.
 *
 * @example
 * // inside a training loop where `engine` is your Neat instance:
 * engine.applyComplexityBudget();
 * // engine.options.maxNodes now holds the adjusted complexity cap
 *
 * @remarks
 * This method intentionally uses lightweight linear-regression slope
 * estimation to detect improvement trends. It clamps growth/decay and
 * respects explicit `minNodes`/`maxNodesEnd` if provided. When used in
 * an educational setting, this helps learners observe how stronger
 * selection pressure or novelty can influence allowed network size.
 */
export function applyComplexityBudget(this: NeatLikeWithAdaptiveType) {
  if (!this.options.complexityBudget?.enabled) return;

  // Step 1: Resolve the active configuration.
  const config = this.options.complexityBudget;
  // Step 2: Delegate to the scheduling helper.
  applyComplexityBudgetSchedule(this, config);
}

/**
 * Toggle phased complexity mode between 'complexify' and 'simplify'.
 *
 * Phased complexity supports alternating periods where the algorithm
 * is encouraged to grow (complexify) or shrink (simplify) network
 * structures. This can help escape local minima or reduce bloat.
 *
 * The current phase and its start generation are stored on `this` as
 * `_phase` and `_phaseStartGeneration` so the state persists across
 * generations.
 *
 * @this {{ options: any, generation: number }} NeatEngine
 * @returns {void} Mutates `this._phase` and `this._phaseStartGeneration`.
 *
 * @example
 * // Called once per generation to update the phase state
 * engine.applyPhasedComplexity();
 */
export function applyPhasedComplexity(this: NeatLikeWithAdaptiveType) {
  if (!this.options.phasedComplexity?.enabled) return;
  const phaseConfig = this.options.phasedComplexity;

  // Step 1: Ensure phase state exists.
  initializePhaseState(this, phaseConfig);
  // Step 2: Flip phase when the window expires.
  togglePhaseIfNeeded(this, phaseConfig);
}

/**
 * Apply adaptive minimal criterion (MC) acceptance.
 *
 * This method maintains an MC threshold used to decide whether an
 * individual genome is considered acceptable. It adapts the threshold
 * based on the proportion of the population that meets the current
 * threshold, trying to converge to a target acceptance rate.
 *
 * Behavior summary:
 * - Initializes `_mcThreshold` from configuration if undefined.
 * - Computes the proportion of genomes with score >= threshold.
 * - Adjusts threshold multiplicatively by `adjustRate` to move the
 *   observed proportion towards `targetAcceptance`.
 * - Sets `g.score = 0` for genomes that fall below the final threshold
 *   — effectively rejecting them from selection.
 *
 * @this {{ options: any, population: Array<{score?: number}>, _mcThreshold?: number }} NeatEngine
 * @returns {void}
 *
 * @example
 * // Example config snippet used by the engine
 * // options.minimalCriterionAdaptive = { enabled: true, initialThreshold: 0.1, targetAcceptance: 0.5, adjustRate: 0.1 }
 * engine.applyMinimalCriterionAdaptive();
 *
 * @notes
 * Use MC carefully: setting an overly high initial threshold can cause
 * mass-rejection early in evolution. The multiplicative update keeps
 * changes smooth and conservative.
 */
export function applyMinimalCriterionAdaptive(this: NeatLikeWithAdaptiveType) {
  if (!this.options.minimalCriterionAdaptive?.enabled) return;

  // Step 1: Resolve configuration and seed threshold state.
  const config = this.options.minimalCriterionAdaptive;

  initializeThreshold(this, config);

  // Step 2: Measure acceptance and update threshold.
  const scoreSnapshot = collectScores(this);
  const acceptance = computeAcceptance(scoreSnapshot, this._mcThreshold ?? 0);
  const tuning = resolveTargetSettings(config);

  updateThreshold(this, acceptance, tuning);
  // Step 3: Apply rejection below the final threshold.
  applyRejection(this, this._mcThreshold ?? 0);
}

/**
 * Adaptive adjustments based on ancestor uniqueness telemetry.
 *
 * This helper inspects the most recent telemetry lineage block (if
 * available) for an `ancestorUniq` metric indicating how unique
 * ancestry is across the population. If ancestry uniqueness drifts
 * outside configured thresholds, the method will adjust either the
 * multi-objective dominance epsilon (if `mode === 'epsilon'`) or the
 * lineage pressure strength (if `mode === 'lineagePressure'`).
 *
 * Typical usage: keep population lineage diversity within a healthy
 * band. Low ancestor uniqueness means too many genomes share ancestors
 * (risking premature convergence); high uniqueness might indicate
 * excessive divergence.
 *
 * @this {{ options: any, generation: number, _telemetry?: any[], _lastAncestorUniqAdjustGen?: number }} NeatEngine
 * @returns {void}
 *
 * @example
 * // Adjusts `options.multiObjective.dominanceEpsilon` when configured
 * engine.applyAncestorUniqAdaptive();
 *
 * @remarks
 * This method respects a `cooldown` so adjustments are not made every
 * generation. Values are adjusted multiplicatively for gentle change.
 */
export const applyAncestorUniqAdaptive = function (
  this: NeatLikeWithAdaptiveType,
) {
  if (!this.options.ancestorUniqAdaptive?.enabled) return;
  // Step 1: Resolve configuration and cooldown.
  const config = this.options.ancestorUniqAdaptive;

  if (!isCooldownSatisfied(this, config)) return;

  // Step 2: Pull the latest lineage metric.
  const ancestorUniq = extractAncestorUniqueness(this);
  if (ancestorUniq === undefined) return;

  // Step 3: Compute thresholds and apply adjustments.
  const thresholds = resolveUniquenessThresholds(config);
  const adjustMagnitude = resolveAdjustmentMagnitude(config);

  applyUniquenessAdjustment(
    this,
    config,
    ancestorUniq,
    thresholds,
    adjustMagnitude,
  );
};

/**
 * Self-adaptive per-genome mutation tuning.
 *
 * This function implements several strategies to adjust each genome's
 * internal mutation rate (`g._mutRate`) and optionally its mutation
 * amount (`g._mutAmount`) over time. Strategies include:
 * - `twoTier`: push top and bottom halves in opposite directions to
 *   create exploration/exploitation balance.
 * - `exploreLow`: preferentially increase mutation for lower-scoring
 *   genomes to promote exploration.
 * - `anneal`: gradually reduce mutation deltas over time.
 *
 * The method reads `this.options.adaptiveMutation` for configuration
 * and mutates genomes in-place.
 *
 * @this {{ options: any, population: Array<any>, generation: number, _getRNG: () => () => number }} NeatEngine
 * @returns {void}
 *
 * @example
 * // configuration example:
 * // options.adaptiveMutation = { enabled: true, initialRate: 0.5, adaptEvery: 1, strategy: 'twoTier', minRate: 0.01, maxRate: 1 }
 * engine.applyAdaptiveMutation();
 *
 * @notes
 * - Each genome must already expose `_mutRate` to be adapted. The
 *   function leaves genomes without `_mutRate` untouched.
 * - Randomness is used to propose changes; seeding the RNG allows for
 *   reproducible experiments.
 */
export const applyAdaptiveMutation = function (this: NeatLikeWithAdaptiveType) {
  if (!this.options.adaptiveMutation?.enabled) return;
  // Step 1: Resolve configuration and cadence.
  const config = this.options.adaptiveMutation;
  const shouldAdapt = shouldAdaptThisGeneration(this.generation, config);
  if (!shouldAdapt) return;

  // Step 2: Prepare partitions and settings.
  const scoredGenomes = collectScoredGenomes(this.population);
  const sortedScoredGenomes = sortScoredGenomes(scoredGenomes);
  const partitions = splitScoredGenomes(sortedScoredGenomes);
  const settings = resolveMutationSettings(this, config);
  const randomSource = resolveRandomSource(this);

  // Step 3: Apply mutations across the population.
  const mutationOutcome = applyMutationsToPopulation(
    this.population,
    partitions,
    settings,
    randomSource,
  );

  // Step 4: Apply fallback balancing when needed.
  if (shouldApplyTwoTierFallback(settings.strategy, mutationOutcome)) {
    applyTwoTierFallback(this.population, settings);
  }
};

/**
 * Decay operator adaptation statistics (success/attempt counters).
 *
 * Many adaptive operator-selection schemes keep running tallies of how
 * successful each operator has been. This helper applies an exponential
 * moving-average style decay to those counters so older outcomes
 * progressively matter less.
 *
 * The `_operatorStats` map on `this` is expected to contain values of
 * the shape `{ success: number, attempts: number }` keyed by operator
 * id/name.
 *
 * @this {{ options: any, _operatorStats: Map<any, {success:number,attempts:number}> }} NeatEngine
 * @returns {void}
 *
 * @example
 * engine.applyOperatorAdaptation();
 */
export const applyOperatorAdaptation = function (
  this: NeatLikeWithAdaptiveType,
) {
  if (!this.options.operatorAdaptation?.enabled) return;

  // Step 1: Resolve configuration and state.
  const operatorConfig = this.options.operatorAdaptation;
  const operatorStats = this._operatorStats;
  if (!operatorStats) return;

  // Step 2: Decay the operator statistics.
  const decayFactor = resolveOperatorDecay(operatorConfig);
  const statsEntries = collectOperatorStatsEntries(operatorStats);

  applyOperatorDecay(operatorStats, statsEntries, decayFactor);
};
