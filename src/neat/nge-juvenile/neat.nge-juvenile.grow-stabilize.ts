/**
 * NGE grow-stabilize cycle.
 *
 * This module owns the core grow-stabilize adaptation cycle extracted from the
 * racing curriculum's runtime adaptation engine. It provides pure decision
 * functions (plateau detection, adaptive hysteresis, weight mutations, growth
 * throttle) and a single orchestrator (`runNgeGrowStabilizeCycle`) that
 * sequences one adaptation tick.
 *
 * The orchestrator accepts plain numeric score history and a live mutable
 * network, keeping the core free of demo-specific types (e.g.
 * `RacingQualitySignal`). App layers convert composite signals to scalar
 * numbers before calling the core cycle.
 *
 * ## Determinism note
 *
 * When a deterministic `random` source is supplied, weight mutation selection
 * is reproducible. When a `lifecycleRunner` is injected, the caller controls
 * the lifecycle execution, enabling test doubles and cycle breaking.
 *
 * ## Background reading
 *
 * - NEAT and topology-evolving neuroevolution:
 *   K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through
 *   Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2,
 *   pp. 99-127, 2002.
 *   [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)
 * - Growth/stabilization as an explore–exploit tradeoff:
 *   [Wikipedia — Exploration–exploitation dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma)
 * - Hysteresis in control systems:
 *   [Wikipedia — Hysteresis](https://en.wikipedia.org/wiki/Hysteresis).
 * - Plateau detection via rolling-window variance:
 *   [Wikipedia — Variance](https://en.wikipedia.org/wiki/Variance).
 * - Mean squared error:
 *   [Wikipedia — Mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error)
 *
 * ```mermaid
 * stateDiagram-v2
 *   [*] --> PlateauCheck
 *   PlateauCheck --> Stabilization : not plateaued
 *   PlateauCheck --> Growth : plateaued or first growth
 *   Stabilization --> [*] : weight mutations applied
 *   Growth --> [*] : lifecycle morphs applied
 * ```
 */

import type Network from '../../architecture/network';
import { DEFAULT_VARIANT_SCORER } from '../../acceleration/acceleration.variants';
import type {
  VariantScorer,
  WeightVariant,
} from '../../acceleration/acceleration.variants';
import { mutation } from '../../methods/mutation/mutation';
import { runNgeLifecycle } from '../neat.nge-lifecycle';
import { resolveGrowStabilizeConfig } from './neat.nge-juvenile.config';
import {
  NGE_EXHAUSTION_DECAY_FLOOR_TIERS,
  NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS,
  NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS,
  NGE_EXHAUSTION_NEURON_BUDGET_FACTOR,
  NGE_EXHAUSTION_NOISE_MULTIPLIER_CAP,
  NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT,
  NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY,
  NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE,
  NGE_EXHAUSTION_NOISE_SIGMA_TIERS,
  NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST,
  NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS,
  NGE_EXHAUSTION_SCORE_EPSILON,
  NGE_EXHAUSTION_STAGE_FRACTION_ADULT,
  NGE_EXHAUSTION_STAGE_FRACTION_BABY,
  NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE,
  NGE_EXHAUSTION_THRESHOLD_DECAY_FLOOR,
  NGE_EXHAUSTION_THRESHOLD_DECAY_RATE,
  NGE_EXHAUSTION_TIER_FRACTIONS,
  NGE_EXHAUSTION_TICK_BUDGET,
  NGE_GROW_STABILIZE_FORCE_GROWTH_AFTER_FAILED_STABILIZATIONS,
  NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS,
  NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
  NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS,
  NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS,
  NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD,
  NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE,
  NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT,
  NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE,
  NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE,
} from './neat.nge-juvenile.constants';
import {
  evaluateNgeWeightVariants,
  resolveEffectiveMagnitude,
  resolveRepresentativeDelta,
  resolveVariantCountForStage,
} from './neat.nge-juvenile.variants';
import type {
  NgeGrowthBudget,
  NgeGrowStabilizeConfig,
  NgeGrowStabilizeInput,
  NgeGrowStabilizeResult,
  NgeHysteresisState,
  NgeLifecycleStage,
  NgeModuleMetricsSnapshot,
  NgePruneBudget,
} from './neat.nge-juvenile.types';

// ──────────────────────────────────────────────────────────────────────
// Weight-exhaustion gate helpers
// ──────────────────────────────────────────────────────────────────────

/**
 * Resolve a tiered numeric value based on current neuron count.
 *
 * Clamps the count to non-negative values and selects the first tier whose
 * `maxNeurons` upper bound exceeds the count. If no tier matches (for example an
 * empty tier list), the fallback value is returned.
 *
 * @param currentNeurons - Number of neurons in the network (clamped to >= 0).
 * @param tiers - Ordered tier table with `maxNeurons` upper bounds. Each tier
 *   carries either a `fraction` or a `floor` value.
 * @param fallback - Value returned when no tier matches.
 * @returns The value belonging to the matched tier, or the fallback.
 *
 * @example
 * ```ts
 * const fraction = resolveNeuronTierFraction(150, NGE_EXHAUSTION_TIER_FRACTIONS, 0.02);
 * console.log(fraction); // 0.02
 * ```
 */
function resolveNeuronTierFraction(
  currentNeurons: number,
  tiers:
    | readonly { maxNeurons: number; fraction: number }[]
    | readonly { maxNeurons: number; floor: number }[],
  fallback: number,
): number {
  const clamped = Math.max(0, currentNeurons);
  const tier = tiers.find((t) => clamped < t.maxNeurons);
  if (tier === undefined) return fallback;
  if ('fraction' in tier) return tier.fraction;
  return tier.floor;
}

/**
 * Resolve the relative improvement fraction for a lifecycle stage.
 *
 * Baby/embryo networks get the largest bar (2%), juvenile networks get a
 * tighter bar (1%), and adult/equilibrium networks get the tightest bar
 * (0.6%). This implements the "grow fast past baby, picky in middle, slower
 * adult" intent by requiring larger improvements early and smaller
 * improvements later.
 *
 * When `currentNeurons` is supplied for a baby/embryo network, the fraction is
 * resolved from `NGE_EXHAUSTION_TIER_FRACTIONS` so tiny newborn networks get a
 * lower bar than larger pre-juvenile networks. Omitting `currentNeurons`
 * preserves the legacy single-value behavior.
 *
 * @param stage - Current NGE lifecycle stage.
 * @param currentNeurons - Optional current neuron count for baby/embryo tier
 *   resolution.
 * @returns Relative improvement fraction for the stage.
 *
 * @example
 * ```ts
 * const fraction = resolveStageFraction('baby');
 * console.log(fraction); // 0.02
 * ```
 */
export function resolveStageFraction(
  stage: NgeLifecycleStage,
  currentNeurons?: number,
): number {
  if (stage === 'embryo' || stage === 'baby') {
    if (currentNeurons !== undefined) {
      return resolveNeuronTierFraction(
        currentNeurons,
        NGE_EXHAUSTION_TIER_FRACTIONS,
        NGE_EXHAUSTION_STAGE_FRACTION_BABY,
      );
    }
    return NGE_EXHAUSTION_STAGE_FRACTION_BABY;
  }
  if (stage === 'juvenile') return NGE_EXHAUSTION_STAGE_FRACTION_JUVENILE;
  return NGE_EXHAUSTION_STAGE_FRACTION_ADULT;
}

/**
 * Resolve the noise-sigma fraction for a lifecycle stage.
 *
 * The noise-sigma fraction scales the adaptive threshold by the expected
 * statistical noise from evaluating a finite number of variants. Early stages
 * get a larger fraction (0.003) because they evaluate more variants and need
 * a higher uplift; later stages get a smaller fraction (0.001).
 *
 * When `currentNeurons` is supplied for a baby/embryo network, the fraction is
 * resolved from `NGE_EXHAUSTION_NOISE_SIGMA_TIERS` so tiny newborn networks get
 * a larger noise allowance that shrinks as the network grows.
 *
 * @param stage - Current NGE lifecycle stage.
 * @param currentNeurons - Optional current neuron count for baby/embryo tier
 *   resolution.
 * @returns Noise-sigma fraction for the stage.
 *
 * @example
 * ```ts
 * const sigmaFraction = resolveNoiseSigmaFraction('juvenile');
 * console.log(sigmaFraction); // 0.002
 * ```
 */
export function resolveNoiseSigmaFraction(
  stage: NgeLifecycleStage,
  currentNeurons?: number,
): number {
  if (stage === 'embryo' || stage === 'baby') {
    if (currentNeurons !== undefined) {
      return resolveNeuronTierFraction(
        currentNeurons,
        NGE_EXHAUSTION_NOISE_SIGMA_TIERS,
        NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY,
      );
    }
    return NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_BABY;
  }
  if (stage === 'juvenile') return NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_JUVENILE;
  return NGE_EXHAUSTION_NOISE_SIGMA_FRACTION_ADULT;
}

/**
 * Resolve the adaptive improvement threshold used by the weight-exhaustion
 * gate.
 *
 * The threshold combines:
 *
 * - a stage-relative improvement bar (tier-aware for baby/embryo networks),
 * - a noise-aware uplift that grows with the number of evaluated variants,
 * - a neuron-budget factor that biases small networks toward structural growth,
 * - a per-failure decay whose floor rises with network size.
 *
 * When the score ceiling is finite and both baseline and best score are below
 * it, the threshold scales with remaining headroom; otherwise it scales with
 * the absolute score magnitude.
 *
 * @param baseline - Score before evaluating variants.
 * @param bestScore - Best score observed across all variants.
 * @param variantCount - Number of variants evaluated.
 * @param stage - Current NGE lifecycle stage.
 * @param neuronBudget - Current and maximum neuron counts.
 * @param scoreCeiling - Known score ceiling, or `Infinity` when absent.
 * @param consecutiveFailures - Optional number of consecutive failed
 *   stabilization ticks. Each failure decays the threshold so that
 *   near-converged scores can still commit useful weight variants.
 * @returns Adaptive improvement threshold; a variant commits when
 *   `bestScore > baseline + threshold`.
 *
 * @example
 * ```ts
 * const threshold = resolveExhaustionImprovementThreshold(
 *   0.5, 0.6, 16, 'baby', { current: 10, max: 100 }, Infinity,
 * );
 * console.log(threshold > 0); // true
 * ```
 */
export function resolveExhaustionImprovementThreshold(
  baseline: number,
  bestScore: number,
  variantCount: number,
  stage: NgeLifecycleStage,
  neuronBudget: { current: number; max: number },
  scoreCeiling: number,
  consecutiveFailures = 0,
): number {
  const epsilon = NGE_EXHAUSTION_SCORE_EPSILON;
  const useMagnitude =
    !Number.isFinite(scoreCeiling) ||
    baseline >= scoreCeiling - epsilon ||
    bestScore >= scoreCeiling - epsilon;
  const scoreScale = useMagnitude
    ? Math.max(Math.abs(baseline), Math.abs(bestScore), epsilon)
    : Math.max(scoreCeiling - baseline, scoreCeiling - bestScore, epsilon);
  const stageFraction = resolveStageFraction(stage, neuronBudget.current);
  const relativeBar = stageFraction * scoreScale;
  const noiseSigmaFraction = resolveNoiseSigmaFraction(
    stage,
    neuronBudget.current,
  );
  const noiseSigma = noiseSigmaFraction * scoreScale;
  const noiseMultiplier =
    variantCount <= 1
      ? 0
      : Math.min(
          Math.sqrt(2 * Math.log(variantCount)),
          NGE_EXHAUSTION_NOISE_MULTIPLIER_CAP,
        );
  const noiseUplift = noiseSigma * noiseMultiplier;
  const neuronFactor =
    Number.isFinite(neuronBudget.max) && neuronBudget.max > 0
      ? Math.min(
          2.0,
          Math.max(
            0.5,
            1.0 +
              NGE_EXHAUSTION_NEURON_BUDGET_FACTOR *
                (1.0 - neuronBudget.current / neuronBudget.max),
          ),
        )
      : 1.0;
  const decayFloor = resolveNeuronTierFraction(
    neuronBudget.current,
    NGE_EXHAUSTION_DECAY_FLOOR_TIERS,
    NGE_EXHAUSTION_THRESHOLD_DECAY_FLOOR,
  );
  const decay = Math.max(
    decayFloor,
    1.0 - NGE_EXHAUSTION_THRESHOLD_DECAY_RATE * consecutiveFailures,
  );
  return Math.max(relativeBar, noiseUplift) * neuronFactor * decay;
}

/**
 * Resolve the number of consecutive weight-exhaustion ticks before structural
 * growth is forced.
 *
 * The raw count is `ceil(tickBudget / variantCount)`, clamped to the allowed
 * [min, max] range. After a bad growth event the limit is doubled (capped at
 * 16, floored at 4) to prevent the network from over-tuning weights instead of
 * adding useful structure.
 *
 * @param variantCount - Number of parallel variants evaluated.
 * @param postGrowthBoostActive - Whether the post-growth anti-runaway boost
 *   is active.
 * @returns Allowed consecutive exhaustion ticks before forcing growth.
 *
 * @example
 * ```ts
 * const limit = resolveExhaustionForceGrowthThreshold(16, false);
 * console.log(limit); // 3
 * ```
 */
export function resolveExhaustionForceGrowthThreshold(
  variantCount: number,
  postGrowthBoostActive: boolean,
): number {
  const raw = Math.ceil(NGE_EXHAUSTION_TICK_BUDGET / Math.max(1, variantCount));
  const base = Math.min(
    Math.max(raw, NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS),
    NGE_EXHAUSTION_MAX_CONSECUTIVE_TICKS,
  );
  if (!postGrowthBoostActive) return base;
  return Math.max(
    NGE_EXHAUSTION_MIN_CONSECUTIVE_TICKS * 4,
    Math.min(
      base * NGE_EXHAUSTION_POST_GROWTH_EXHAUSTION_BOOST,
      NGE_EXHAUSTION_POST_GROWTH_MAX_CONSECUTIVE_TICKS,
    ),
  );
}

// ──────────────────────────────────────────────────────────────────────
// Pure decision functions
// ──────────────────────────────────────────────────────────────────────

/**
 * Resolve the adaptive hysteresis window count based on the live network
 * node count. Smaller networks use a lower threshold (2 consecutive
 * positive-quality windows) to accelerate early growth, while larger
 * networks require more sustained evidence (5 windows) before committing
 * to further structural expansion.
 *
 * @param nodeCount - Current total node count in the live network.
 * @returns Hysteresis window count: 2 for ≤ 200 nodes, 3 for ≤ 500, 5 for > 500.
 *
 * @example
 * ```ts
 * const hysteresis = resolveAdaptiveHysteresis(150);
 * console.log(hysteresis); // 2
 * ```
 */
export function resolveAdaptiveHysteresis(nodeCount: number): number {
  if (nodeCount <= 200) {
    return 2;
  }
  if (nodeCount <= 500) {
    return 3;
  }
  return 5;
}

/**
 * Determine whether the quality score has plateaued based on a rolling
 * window of recent baseline scores.
 *
 * Before the first structural growth, the function always returns `true` to
 * allow initial network development without waiting for a full score window.
 * After the first growth, the network is considered plateaued when the
 * rolling window is full and its variance falls below the threshold.
 *
 * Time-boxed stabilization: a minimum number of ticks must elapse before
 * plateau can fire (preventing premature growth), and a maximum number of
 * ticks forces growth re-entry even if the variance remains above threshold.
 *
 * @param scoreWindow - Rolling window of recent baseline quality scores.
 * @param hasGrownBefore - Whether the network has already undergone at least
 *   one structural growth phase.
 * @param stabilizationTicksSinceGrowth - Ticks elapsed in the stabilization
 *   phase since the last structural growth.
 * @returns `true` when growth should proceed (first growth, stabilized
 *   plateau, or time-box cap exceeded), `false` when the network is still
 *   stabilizing after growth.
 *
 * @example
 * ```ts
 * const plateaued = isPlateauReached([0.5, 0.51, 0.49, 0.5, 0.5], true, 10);
 * console.log(plateaued); // true (low variance after min ticks)
 * ```
 */
export function isPlateauReached(
  scoreWindow: readonly number[],
  hasGrownBefore: boolean,
  stabilizationTicksSinceGrowth: number,
): boolean {
  if (!hasGrownBefore) {
    return true;
  }

  // Time-box cap: after max stabilization ticks (25), force growth re-entry
  // even if the score has not plateaued. This prevents indefinite
  // stabilization when the quality signal remains noisy.
  if (
    stabilizationTicksSinceGrowth >= NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS
  ) {
    return true;
  }

  // Minimum guard: require at least min stabilization ticks before plateau
  // can fire. This gives the network time to learn its new structure
  // before allowing further structural growth.
  if (
    stabilizationTicksSinceGrowth < NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS
  ) {
    return false;
  }

  if (scoreWindow.length < NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE) {
    return false;
  }

  const mean =
    scoreWindow.reduce(
      (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
      0,
    ) / scoreWindow.length;

  let variance = 0;
  for (const score of scoreWindow) {
    const deviation = score - mean;
    variance += deviation * deviation;
  }
  variance /= scoreWindow.length;

  return variance < NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD;
}

/**
 * Apply random weight perturbations to existing connections.
 *
 * Each connection is independently selected for mutation with probability
 * equal to the weight mutation rate. Selected connections have their weight
 * perturbed by a random amount in the range
 * [-magnitude, +magnitude]. This helps the network learn to use its current
 * structure during the stabilization phase between structural growth phases.
 *
 * @param network - The network whose connections to perturb.
 * @param random - Random number generator returning a float in [0, 1).
 * @param magnitude - Optional override for the perturbation magnitude. When
 *   omitted, the default grow-stabilize weight mutation magnitude is used.
 * @returns The number of connections that were mutated.
 *
 * @example
 * ```ts
 * const mutated = applyWeightMutations(network, Math.random);
 * console.log(mutated); // e.g. 3
 * ```
 */
export function applyWeightMutations(
  network: Network,
  random: () => number,
  magnitude?: number,
): number {
  const effectiveMagnitude =
    magnitude ?? NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE;
  let mutatedCount = 0;
  for (const connection of network.connections) {
    if (random() < NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE) {
      const delta = (random() * 2 - 1) * effectiveMagnitude;
      connection.weight += delta;
      mutatedCount++;
    }
  }
  return mutatedCount;
}

/**
 * Compute whether the growth lifecycle should be throttled for the current tick.
 *
 * When the network exceeds the large-network node threshold, the effective
 * throttle interval scales with network size so that larger networks get
 * progressively longer back-off intervals. This preserves real-time
 * performance by preventing the lifecycle from running every tick at scale.
 *
 * @param network - Live controller network whose size determines throttling.
 * @param tick - Current fixed-timestep tick used for interval gating.
 * @returns Throttle decision with the computed interval.
 *
 * @example
 * ```ts
 * const { shouldThrottle } = computeGrowthThrottle(network, 42);
 * console.log(shouldThrottle); // false for small networks
 * ```
 */
export function computeGrowthThrottle(
  network: Network,
  tick: number,
): { shouldThrottle: boolean; interval: number } {
  const nodeCount = network.nodes.length;
  if (nodeCount <= NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD) {
    return { shouldThrottle: false, interval: 1 };
  }

  // Scale the throttle interval based on network size budget.
  const sizeBudget = Math.ceil(
    nodeCount / NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
  );
  const interval =
    NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS * sizeBudget;
  const shouldThrottle = tick % interval !== 0;

  return { shouldThrottle, interval };
}

// ──────────────────────────────────────────────────────────────────────
// Orchestrator
// ──────────────────────────────────────────────────────────────────────

/**
 * Run one NGE grow-stabilize adaptation cycle.
 *
 * This orchestrator encapsulates the plateau-detection decision and either:
 *
 * - **Stabilization phase**: applies weight perturbations to existing
 *   connections so the network can learn to use its current structure.
 * - **Growth phase**: builds module metrics, a growth budget, and a prune
 *   budget from the live network state, then delegates to the NGE lifecycle
 *   runner to plan and apply structural morphs.
 *
 * For the very first growth (`hasGrownBefore` is `false`), the plateau check
 * is bypassed and the hysteresis gate is pre-satisfied so the lifecycle
 * produces candidate morphs immediately — the network needs capacity before
 * stabilization can tune it. If the lifecycle still returns no applied
 * operations on that first call, the cycle forces a single `ADD_NODE` mutation
 * so the network cannot remain stuck at its starting size.
 *
 * Weight-exhaustion detection is stateful. Supply `consecutiveWeightExhaustion`
 * and `postGrowthThresholdActive` so the cycle can count failed variant ticks
 * and activate the post-growth anti-runaway boost. A captured
 * `preGrowthBaseline` is compared against the stabilization baseline; a large
 * drop activates the boost and consumes the captured value.
 *
 * When stabilization repeatedly fails to commit weight variants, the caller
 * can pass a non-zero `consecutiveStabilizationFailures` count. Once it meets
 * or exceeds `NGE_GROW_STABILIZE_FORCE_GROWTH_AFTER_FAILED_STABILIZATIONS`,
 * the cycle skips the stabilization phase and forces a growth attempt with
 * reason `forced_by_stabilization_failures`. This prevents the network from
 * staying stuck in local weight-tuning optima.
 *
 * The adaptive improvement threshold used during variant evaluation decays as
 * `consecutiveWeightExhaustion` increases, controlled by
 * `NGE_EXHAUSTION_THRESHOLD_DECAY_RATE` and floored by
 * `NGE_EXHAUSTION_THRESHOLD_DECAY_FLOOR`. The decay lowers the bar so that
 * near-converged scores can still commit useful weight variants.
 *
 * The returned result includes `actualVariantCount`, which reports the number of
 * weight variants that were actually evaluated during stabilization. When
 * variants are not evaluated (for example because training data are missing or
 * growth was forced), the field is zero.
 *
 * Score-space alignment matters. When a custom `scoreFn` is supplied, it is
 * used for both the baseline and the variant evaluations. The caller must ensure
 * the scorer returns values in the same semantic space and direction as the
 * supplied `baselineScore`; otherwise the commit inequality
 * `bestScore > baselineScore + threshold` can never be satisfied. A common
 * mistake is comparing a positive task-specific quality score with the default
 * negative mean-squared-error scorer.
 *
 * The caller is responsible for pre-mutation score evaluation, network
 * snapshot/rollback, and post-mutation score evaluation. The cycle only
 * handles the core decision and mutation application; commit/rollback based
 * on score improvement remains the caller's responsibility.
 *
 * @param input - Grow-stabilize cycle input with required network,
 *   scoreHistory, hasGrownBefore, and stabilizationTicksSinceGrowth.
 * @returns Promise resolving to the result describing whether the cycle
 *   committed, which phase it entered, what operations were applied, and the
 *   updated exhaustion/hysteresis state. When stabilization evaluated weight
 *   variants, `actualVariantCount` reports the count that were used.
 *
 * @example
 * ```ts
 * const result = await runNgeGrowStabilizeCycle({
 *   network,
 *   scoreHistory: [1, 2, 3, 4],
 *   hasGrownBefore: false,
 *   stabilizationTicksSinceGrowth: 0,
 * });
 * console.log(result.committed); // true (first growth)
 * ```
 */
export async function runNgeGrowStabilizeCycle(
  input: NgeGrowStabilizeInput,
): Promise<NgeGrowStabilizeResult> {
  const ctx = resolveCycleContext(input);

  if (ctx.shouldStabilize) {
    return runStabilizationPhase(input, ctx);
  }

  return runGrowthPhase(input, ctx);
}

// ──────────────────────────────────────────────────────────────────────
// Helpers (below the fold)
// ──────────────────────────────────────────────────────────────────────

/**
 * Build default module metrics from numeric score history and live network state.
 *
 * @param scoreHistory - Rolling numeric score history.
 * @param network - Live controller network.
 * @param moduleId - Module identifier for the metrics snapshot.
 * @returns NGE module metrics for the lifecycle focus scorer.
 * @internal
 */
function buildDefaultMetrics(
  scoreHistory: readonly number[],
  network: Network,
  moduleId: string,
): NgeModuleMetricsSnapshot {
  const scoreTrend =
    scoreHistory.length >= 2 ? scoreHistory.at(-1)! - scoreHistory[0]! : 0;
  const scoreMean =
    scoreHistory.length > 0
      ? scoreHistory.reduce(
          (accumulatedScore, scoreValue) => accumulatedScore + scoreValue,
          0,
        ) / scoreHistory.length
      : 0;

  return {
    moduleId,
    utilization: Math.min(scoreMean, 1),
    rewardDelta: scoreTrend,
    novelty: 0,
    stabilityAge: 0,
    wiringCost: network.nodes.length + network.connections.length,
  };
}

/**
 * Build a default growth budget from the live network and resolved config.
 *
 * @param network - Live controller network.
 * @param config - Resolved grow-stabilize config.
 * @returns NGE growth budget for the lifecycle apply phase.
 * @internal
 */
function buildDefaultBudget(
  network: Network,
  config: NgeGrowStabilizeConfig,
): NgeGrowthBudget {
  return {
    maxNodes: config.maxNodes,
    maxEdges: config.maxConnections,
    maxEpisodicSlots: config.maxEpisodicSlots,
    currentNodeCount: network.nodes.length,
    currentEdgeCount: network.connections.length,
    currentEpisodicSlotCount: 0,
  };
}

/**
 * Build a default prune budget from the live network.
 *
 * @param network - Live controller network.
 * @returns NGE prune budget for the lifecycle apply phase.
 * @internal
 */
function buildDefaultPruneBudget(network: Network): NgePruneBudget {
  return {
    minEdges: 0,
    minNodes: 1,
    costExemptEdgeIds: [],
    currentEdgeCount: network.connections.length,
    currentNodeCount: network.nodes.length,
    currentWiringCost: network.nodes.length + network.connections.length,
  };
}

/**
 * Build deterministic weight variants that mirror the evaluator's internal
 * variant list.
 *
 * The parallel evaluator restores connection weights after each variant, so
 * the grow-stabilize cycle must reconstruct the same variant list to commit
 * the winning delta. This builder uses the same endpoint-inclusive delta
 * distribution and effective-magnitude scaling as the evaluator so that
 * reconstruction is guaranteed to match the evaluated slot.
 *
 * @param network - Network surface whose connection list is used for indexing.
 * @param variantCount - Number of parallel variants to reconstruct.
 * @param stage - Current NGE lifecycle stage; controls effective magnitude.
 * @returns Array of deterministic weight variants.
 *
 * @example
 * ```ts
 * const variants = buildWeightVariants(network, 4, 'juvenile');
 * // variants[0] targets connection 0 with a small negative delta;
 * // variants[3] targets connection 3 with the largest positive delta.
 * ```
 */
export function buildWeightVariants(
  network: Network,
  variantCount: number,
  stage: NgeLifecycleStage,
): WeightVariant[] {
  const connectionCount = network.connections.length;
  const effectiveMagnitude = resolveEffectiveMagnitude(
    stage,
    variantCount,
    connectionCount,
  );
  const variants: WeightVariant[] = [];

  for (let index = 0; index < variantCount; index++) {
    const weightIndex = connectionCount > 0 ? index % connectionCount : 0;
    variants.push({
      weightIndex,
      delta: resolveRepresentativeDelta(
        index,
        variantCount,
        effectiveMagnitude,
      ),
    });
  }

  return variants;
}

/**
 * Map lifecycle apply outcomes to operation name strings.
 *
 * @param outcomes - Apply outcomes from the lifecycle result.
 * @returns Operation strings for telemetry, excluding skipped morphs.
 * @internal
 */
function mapOutcomesToOperations(
  outcomes: readonly { status: string; kind: string }[],
): string[] {
  const operations: string[] = [];
  for (const outcome of outcomes) {
    if (outcome.status !== 'applied') continue;
    if (outcome.kind === 'edgeDensify') operations.push('add_edge');
    else if (outcome.kind === 'nodeAdd') operations.push('add_node');
    else if (outcome.kind === 'edgePrune' || outcome.kind === 'compact')
      operations.push('prune_edge');
  }
  return operations;
}

/**
 * Score the live network on a provided input/target pair using the same
 * variant scorer that the parallel evaluator uses.
 *
 * This keeps the weight-exhaustion improvement baseline in the same score
 * units as the variant scores, so a better-than-baseline variant can actually
 * win the commit decision. Callers may supply a task-specific scorer (for
 * example, to reduce multi-dimensional controller outputs to a scalar).
 *
 * @param network - Live network to evaluate.
 * @param inputs - Input rows, one per evaluation sample.
 * @param target - Target output vector or scalar target values.
 * @param scoreFn - Scorer to use; defaults to {@link DEFAULT_VARIANT_SCORER}.
 * @returns Baseline score in the scorer's units (higher is better).
 * @internal
 */
async function evaluateNetworkScore(
  network: Network,
  inputs: number[][],
  target: number[],
  scoreFn: VariantScorer = DEFAULT_VARIANT_SCORER,
): Promise<number> {
  const outputs: number[][] = [];
  for (const input of inputs) {
    const output = await Promise.resolve(network.activate(input));
    outputs.push([...output]);
  }
  return scoreFn(outputs, target);
}

// ──────────────────────────────────────────────────────────────────────
// Orchestrator helpers (extracted from runNgeGrowStabilizeCycle)
// ──────────────────────────────────────────────────────────────────────

/** Sub-result from the stabilization phase variant/fallback paths. */
interface StabilizationSubResult {
  readonly mutatedCount: number;
  readonly reason: string;
  readonly operations: readonly string[];
  readonly nextExhaustion: number;
  readonly nextPostGrowthActive: boolean;
  readonly bestVariantScore: number | undefined;
  readonly threshold: number | undefined;
  readonly actualVariantCount: number;
}

/** Result of attempting to commit a weight variant. */
interface VariantCommitResult {
  readonly committed: boolean;
  readonly reason: string;
  readonly operations: readonly string[];
  readonly nextExhaustion: number;
  readonly nextPostGrowthActive: boolean;
}

/** Result of the first-growth guarantee check. */
interface FirstGrowthResult {
  readonly operations: string[];
  readonly applyOutcomes: readonly { status: string; kind: string }[];
  readonly forced: boolean;
}

/** Result of checking and consuming the pre-growth baseline. */
interface PostGrowthCheck {
  readonly nextPostGrowthActive: boolean;
  readonly nextPreGrowthBaseline: number | undefined;
}

/** Shared context resolved once per cycle for both phases. */
interface CycleContext {
  readonly network: Network;
  readonly hasGrownBefore: boolean;
  readonly stabilizationTicksSinceGrowth: number;
  readonly random: () => number;
  readonly runner:
    | NonNullable<NgeGrowStabilizeInput['lifecycleRunner']>
    | typeof runNgeLifecycle;
  readonly config: NgeGrowStabilizeConfig;
  readonly consecutiveWeightExhaustion: number;
  readonly postGrowthThresholdActive: boolean;
  readonly consecutiveStabilizationFailures: number;
  readonly stage: NgeLifecycleStage;
  readonly effectiveVariantCount: number;
  readonly hasTrainingData: boolean;
  readonly shouldEvaluateVariants: boolean;
  readonly exhaustionLimit: number;
  readonly forceGrowthByStabilizationFailures: boolean;
  readonly plateauReached: boolean;
  readonly shouldStabilize: boolean;
}

/**
 * Return `value` when defined, otherwise `fallback`.
 *
 * @internal
 */
function withDefault<T>(value: T | undefined, fallback: T): T {
  return value ?? fallback;
}

/**
 * Return the first defined number in the list, or 0 when all are undefined.
 *
 * @internal
 */
function firstDefined(...values: readonly (number | undefined)[]): number {
  for (const value of values) {
    if (value !== undefined) return value;
  }
  return 0;
}

/**
 * Whether the input carries usable training data for variant evaluation.
 *
 * @internal
 */
function hasTrainingDataInput(input: NgeGrowStabilizeInput): boolean {
  return (
    input.inputs !== undefined &&
    input.inputs.length > 0 &&
    input.target !== undefined &&
    input.target.length > 0
  );
}

/**
 * Whether parallel weight-variant evaluation should run for the current stage.
 *
 * @internal
 */
function shouldEvaluateVariantsForStage(
  variantCount: number,
  hasData: boolean,
): boolean {
  return variantCount > 1 && hasData;
}

/**
 * Whether the cycle should enter the stabilization phase.
 *
 * @internal
 */
function shouldRunStabilization(
  plateauReached: boolean,
  exhaustion: number,
  limit: number,
  forceStabilization: boolean,
): boolean {
  return !plateauReached && exhaustion < limit && !forceStabilization;
}

/**
 * Resolve the fallback baseline score from previous score or quality history.
 *
 * @internal
 */
function resolveFallbackScore(input: NgeGrowStabilizeInput): number {
  return firstDefined(input.previousScore, input.qualityScoreHistory?.at(-1));
}

/**
 * Resolve the baseline quality score using a cascade of fallbacks.
 *
 * @internal
 */
async function resolveBaselineScore(
  input: NgeGrowStabilizeInput,
  network: Network,
  hasData: boolean,
): Promise<number> {
  if (input.baselineScore !== undefined) return input.baselineScore;
  if (hasData) {
    return evaluateNetworkScore(
      network,
      input.inputs!,
      input.target!,
      withDefault(input.scoreFn, DEFAULT_VARIANT_SCORER),
    );
  }
  return resolveFallbackScore(input);
}

/**
 * Whether the variant evaluation guard has failed.
 *
 * @internal
 */
function isVariantGuardFailed(
  bestIndex: number,
  bestScore: number,
  variantCount: number,
): boolean {
  return (
    bestIndex < 0 || !Number.isFinite(bestScore) || bestIndex >= variantCount
  );
}

/**
 * Build the result for a failed variant guard.
 *
 * @internal
 */
function buildGuardFailedResult(
  exhaustion: number,
  postGrowthActive: boolean,
): StabilizationSubResult {
  return {
    mutatedCount: 0,
    reason: 'no_weight_mutations',
    operations: [],
    nextExhaustion: exhaustion + 1,
    nextPostGrowthActive: postGrowthActive,
    bestVariantScore: undefined,
    threshold: undefined,
    actualVariantCount: 0,
  };
}

/**
 * Attempt to commit the best weight variant to the network.
 *
 * @internal
 */
function tryCommitVariant(
  network: Network,
  variants: readonly WeightVariant[],
  bestIndex: number,
  bestScore: number,
  baselineScore: number,
  threshold: number,
  exhaustion: number,
  postGrowthActive: boolean,
): VariantCommitResult {
  const bestVariant = variants[bestIndex];
  const weightIndex = bestVariant.weightIndex;
  if (
    bestScore > baselineScore + threshold &&
    weightIndex < network.connections.length
  ) {
    network.connections[weightIndex].weight += bestVariant.delta;
    return {
      committed: true,
      reason: 'weight_variant_committed',
      operations: ['param_nudge'],
      nextExhaustion: 0,
      nextPostGrowthActive: false,
    };
  }
  return {
    committed: false,
    reason: 'no_weight_mutations',
    operations: [],
    nextExhaustion: exhaustion + 1,
    nextPostGrowthActive: postGrowthActive,
  };
}

/**
 * Evaluate weight variants and attempt to commit the best one.
 *
 * @internal
 */
async function evaluateAndCommitVariant(
  input: NgeGrowStabilizeInput,
  network: Network,
  stage: NgeLifecycleStage,
  effectiveVariantCount: number,
  baselineScore: number,
  consecutiveWeightExhaustion: number,
  postGrowthThresholdActive: boolean,
  config: NgeGrowStabilizeConfig,
): Promise<StabilizationSubResult> {
  const variantResult = await evaluateNgeWeightVariants(
    network,
    stage,
    input.inputs!,
    input.target!,
    undefined,
    {
      accelerationConfig: input.accelerationConfig,
      stageVariantCounts: {
        [stage]: NGE_GROW_STABILIZE_STABILIZATION_VARIANT_COUNT,
      },
      scoreFn: input.scoreFn,
    },
  );
  const actualVariantCount = withDefault(
    variantResult.metadata?.variantCount,
    effectiveVariantCount,
  );
  const variants = buildWeightVariants(network, actualVariantCount, stage);
  const bestIndex = variantResult.bestIndex;
  const bestScore = variantResult.bestScore;

  if (isVariantGuardFailed(bestIndex, bestScore, variants.length)) {
    return buildGuardFailedResult(
      consecutiveWeightExhaustion,
      postGrowthThresholdActive,
    );
  }

  const scoreCeiling = withDefault(
    input.scoreCeiling,
    Number.POSITIVE_INFINITY,
  );
  const neuronBudget = {
    current: network.nodes.length,
    max: withDefault(input.maxNeurons, config.maxNodes),
  };
  const threshold = resolveExhaustionImprovementThreshold(
    baselineScore,
    bestScore,
    actualVariantCount,
    stage,
    neuronBudget,
    scoreCeiling,
    consecutiveWeightExhaustion,
  );
  const commit = tryCommitVariant(
    network,
    variants,
    bestIndex,
    bestScore,
    baselineScore,
    threshold,
    consecutiveWeightExhaustion,
    postGrowthThresholdActive,
  );
  return {
    mutatedCount: +commit.committed,
    reason: commit.reason,
    operations: commit.operations,
    nextExhaustion: commit.nextExhaustion,
    nextPostGrowthActive: commit.nextPostGrowthActive,
    bestVariantScore: bestScore,
    threshold,
    actualVariantCount,
  };
}

/**
 * Apply fallback random weight mutations when variant evaluation is skipped.
 *
 * @internal
 */
function applyFallbackWeightMutations(
  network: Network,
  random: () => number,
  exhaustion: number,
  postGrowthActive: boolean,
): StabilizationSubResult {
  const mutatedCount = applyWeightMutations(network, random);
  return {
    mutatedCount,
    reason:
      mutatedCount > 0 ? 'weight_mutation_committed' : 'no_weight_mutations',
    operations: mutatedCount > 0 ? ['param_nudge'] : [],
    nextExhaustion: exhaustion + 1,
    nextPostGrowthActive: postGrowthActive,
    bestVariantScore: undefined,
    threshold: undefined,
    actualVariantCount: 0,
  };
}

/**
 * Check and consume the pre-growth baseline, activating the post-growth boost
 * when the new baseline drops below the pre-growth value.
 *
 * @internal
 */
function checkPostGrowthBaseline(
  preGrowthBaseline: number | undefined,
  baselineScore: number,
  improvementThreshold: number,
  currentPostGrowthActive: boolean,
): PostGrowthCheck {
  if (preGrowthBaseline === undefined) {
    return {
      nextPostGrowthActive: currentPostGrowthActive,
      nextPreGrowthBaseline: undefined,
    };
  }
  const triggered = baselineScore < preGrowthBaseline - improvementThreshold;
  return {
    nextPostGrowthActive: triggered || currentPostGrowthActive,
    nextPreGrowthBaseline: undefined,
  };
}

/**
 * Resolve all shared state for the grow-stabilize cycle.
 *
 * @internal
 */
function resolveCycleContext(input: NgeGrowStabilizeInput): CycleContext {
  const config = resolveGrowStabilizeConfig(input.config);
  const stage = withDefault(input.lifecycleStage, 'baby');
  const postGrowthThresholdActive = withDefault(
    input.postGrowthThresholdActive,
    false,
  );
  const consecutiveWeightExhaustion = withDefault(
    input.consecutiveWeightExhaustion,
    0,
  );
  const consecutiveStabilizationFailures = withDefault(
    input.consecutiveStabilizationFailures,
    0,
  );
  const effectiveVariantCount = resolveVariantCountForStage(
    stage,
    undefined,
    input.accelerationConfig,
  );
  const hasTrainingData = hasTrainingDataInput(input);
  const shouldEvaluateVariants = shouldEvaluateVariantsForStage(
    effectiveVariantCount,
    hasTrainingData,
  );
  const exhaustionLimit = resolveExhaustionForceGrowthThreshold(
    effectiveVariantCount,
    postGrowthThresholdActive,
  );
  const forceGrowthByStabilizationFailures =
    consecutiveStabilizationFailures >=
    NGE_GROW_STABILIZE_FORCE_GROWTH_AFTER_FAILED_STABILIZATIONS;
  const plateauReached = isPlateauReached(
    withDefault(input.qualityScoreHistory, []),
    input.hasGrownBefore,
    input.stabilizationTicksSinceGrowth,
  );
  const shouldStabilize = shouldRunStabilization(
    plateauReached,
    consecutiveWeightExhaustion,
    exhaustionLimit,
    forceGrowthByStabilizationFailures,
  );
  return {
    network: input.network,
    hasGrownBefore: input.hasGrownBefore,
    stabilizationTicksSinceGrowth: input.stabilizationTicksSinceGrowth,
    random: withDefault(input.random, Math.random),
    runner: input.lifecycleRunner ?? runNgeLifecycle,
    config,
    consecutiveWeightExhaustion,
    postGrowthThresholdActive,
    consecutiveStabilizationFailures,
    stage,
    effectiveVariantCount,
    hasTrainingData,
    shouldEvaluateVariants,
    exhaustionLimit,
    forceGrowthByStabilizationFailures,
    plateauReached,
    shouldStabilize,
  };
}

/**
 * Run the stabilization phase: weight tuning or variant evaluation.
 *
 * @internal
 */
async function runStabilizationPhase(
  input: NgeGrowStabilizeInput,
  ctx: CycleContext,
): Promise<NgeGrowStabilizeResult> {
  const baselineScore = await resolveBaselineScore(
    input,
    ctx.network,
    ctx.hasTrainingData,
  );

  const subResult = ctx.shouldEvaluateVariants
    ? await evaluateAndCommitVariant(
        input,
        ctx.network,
        ctx.stage,
        ctx.effectiveVariantCount,
        baselineScore,
        ctx.consecutiveWeightExhaustion,
        ctx.postGrowthThresholdActive,
        ctx.config,
      )
    : applyFallbackWeightMutations(
        ctx.network,
        ctx.random,
        ctx.consecutiveWeightExhaustion,
        ctx.postGrowthThresholdActive,
      );

  const postGrowth = checkPostGrowthBaseline(
    input.preGrowthBaseline,
    baselineScore,
    ctx.config.improvementThreshold,
    subResult.nextPostGrowthActive,
  );

  return {
    committed: subResult.mutatedCount > 0,
    phase: 'stabilization',
    reason: subResult.reason,
    operations: subResult.operations,
    stabilizationTicksSinceGrowth: ctx.stabilizationTicksSinceGrowth + 1,
    mutatedCount: subResult.mutatedCount,
    networkSizeAfter: {
      nodes: ctx.network.nodes.length,
      connections: ctx.network.connections.length,
    },
    consecutiveWeightExhaustion: subResult.nextExhaustion,
    postGrowthThresholdActive: postGrowth.nextPostGrowthActive,
    preGrowthBaseline: postGrowth.nextPreGrowthBaseline,
    bestVariantScore: subResult.bestVariantScore,
    threshold: subResult.threshold,
    actualVariantCount: subResult.actualVariantCount,
    consecutiveStabilizationFailures: ctx.consecutiveStabilizationFailures,
  };
}

/**
 * Resolve lifecycle hysteresis for the growth phase.
 *
 * @internal
 */
function resolveLifecycleHysteresis(
  isFirstGrowth: boolean,
  inputHysteresis: NgeHysteresisState | undefined,
  adaptiveHysteresis: number,
): NgeHysteresisState {
  if (isFirstGrowth) {
    return {
      growthPositiveWindowCount: adaptiveHysteresis,
      pruneUnderuseWindowCount: 0,
      lastMorphKind: 'none',
      cooldownWindowsRemaining: 0,
    };
  }
  return withDefault(inputHysteresis, {
    growthPositiveWindowCount: 0,
    pruneUnderuseWindowCount: 0,
    lastMorphKind: 'none',
    cooldownWindowsRemaining: 0,
  });
}

/**
 * Apply the first-growth guarantee: force a node addition when the lifecycle
 * produced no operations on the very first growth.
 *
 * @internal
 */
function applyFirstGrowthGuarantee(
  network: Network,
  isFirstGrowth: boolean,
  operations: string[],
  applyOutcomes: readonly { status: string; kind: string }[],
): FirstGrowthResult {
  if (!isFirstGrowth || operations.length > 0) {
    return { operations, applyOutcomes, forced: false };
  }
  network.mutate(mutation.ADD_NODE);
  const fallbackOutcomes = [
    ...applyOutcomes,
    { status: 'applied', kind: 'nodeAdd' },
  ];
  return {
    operations: mapOutcomesToOperations(fallbackOutcomes),
    applyOutcomes: fallbackOutcomes,
    forced: true,
  };
}

/**
 * Apply forced first-growth hysteresis overrides.
 *
 * @internal
 */
function applyForcedFirstGrowthHysteresis(
  resultHysteresis: NgeHysteresisState | undefined,
  forced: boolean,
  config: NgeGrowStabilizeConfig,
): NgeHysteresisState | undefined {
  if (!forced) return resultHysteresis;
  return {
    ...resultHysteresis,
    lastMorphKind: 'nodeAdd',
    cooldownWindowsRemaining: config.lifecycleCooldownWindowCount,
    growthPositiveWindowCount: 0,
    pruneUnderuseWindowCount: withDefault(
      resultHysteresis?.pruneUnderuseWindowCount,
      0,
    ),
  };
}

/**
 * Resolve the growth reason string.
 *
 * @internal
 */
function resolveGrowthReason(
  forcedByStabilizationFailures: boolean,
  forcedByExhaustion: boolean,
  operationCount: number,
): string {
  if (forcedByStabilizationFailures) return 'forced_by_stabilization_failures';
  if (forcedByExhaustion) return 'forced_by_weight_exhaustion';
  if (operationCount > 0) return 'committed';
  return 'no_candidate_operations';
}

/**
 * Run the growth phase: structural morphs via the lifecycle runner.
 *
 * @internal
 */
async function runGrowthPhase(
  input: NgeGrowStabilizeInput,
  ctx: CycleContext,
): Promise<NgeGrowStabilizeResult> {
  const forcedByExhaustion =
    !ctx.plateauReached &&
    ctx.consecutiveWeightExhaustion >= ctx.exhaustionLimit;
  const forcedByStabilizationFailures =
    !ctx.plateauReached && ctx.forceGrowthByStabilizationFailures;

  const metrics = buildDefaultMetrics(
    input.scoreHistory,
    ctx.network,
    ctx.config.moduleId,
  );
  const budget = buildDefaultBudget(ctx.network, ctx.config);
  const pruneBudget = buildDefaultPruneBudget(ctx.network);
  const adaptiveHysteresis = resolveAdaptiveHysteresis(
    ctx.network.nodes.length,
  );
  const isFirstGrowth = !ctx.hasGrownBefore;
  const lifecycleHysteresis = resolveLifecycleHysteresis(
    isFirstGrowth,
    input.hysteresis,
    adaptiveHysteresis,
  );

  const lifecycleResult = ctx.runner({
    stage: 'juvenile',
    moduleId: ctx.config.moduleId,
    metrics,
    budget,
    config: {
      hysteresisWindowCount: adaptiveHysteresis,
      cooldownWindowCount: 5,
      maxStructuralEditsPerStep: ctx.config.maxStructuralEditsPerStep,
    },
    hysteresis: lifecycleHysteresis,
    network: ctx.network,
    pruneBudget,
  });

  const initialApplyOutcomes = withDefault(lifecycleResult.applyOutcomes, []);
  const initialOperations = mapOutcomesToOperations(initialApplyOutcomes);
  const firstGrowth = applyFirstGrowthGuarantee(
    ctx.network,
    isFirstGrowth,
    initialOperations,
    initialApplyOutcomes,
  );
  const operations = firstGrowth.operations;
  const resultHysteresis = applyForcedFirstGrowthHysteresis(
    lifecycleResult.hysteresis ?? input.hysteresis,
    firstGrowth.forced,
    ctx.config,
  );

  const baselineScore = await resolveBaselineScore(
    input,
    ctx.network,
    ctx.hasTrainingData,
  );

  return {
    committed: operations.length > 0,
    phase: 'growth',
    reason: resolveGrowthReason(
      forcedByStabilizationFailures,
      forcedByExhaustion,
      operations.length,
    ),
    operations,
    stabilizationTicksSinceGrowth: 0,
    mutatedCount: 0,
    networkSizeAfter: {
      nodes: ctx.network.nodes.length,
      connections: ctx.network.connections.length,
    },
    hysteresis: resultHysteresis,
    consecutiveWeightExhaustion: 0,
    postGrowthThresholdActive: false,
    preGrowthBaseline: baselineScore,
    bestVariantScore: undefined,
    threshold: undefined,
    actualVariantCount: 0,
    consecutiveStabilizationFailures: 0,
  };
}
