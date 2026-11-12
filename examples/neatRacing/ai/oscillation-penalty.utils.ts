/**
 * NeatRacing shared oscillation predicate (slice `P2-S3-B`, AC-030).
 *
 * THE LAW (spec section 9.5, LD-31 / E3): the racing oscillation retune has
 * TWO coupled mechanisms — the MULTIPLICATIVE oscillation penalty inside the
 * variant scorer and the ADDITIVE growth-commit threshold boost — and BOTH
 * are driven by ONE shared predicate:
 *
 * - ENGAGE iff `nodeCount >= the current level's promotion threshold`
 *   (TP-07: 300 at lvl 1 -> 2, 600 at lvl 2 -> 3) AND the oscillation signal
 *   has been sustained over the TP-09 grace window;
 * - HYSTERESIS (no step changes): the engagement ramps IN over the grace
 *   window while the signal is sustained and ramps OUT over the same window
 *   while it is absent, so a network crossing the node-count boundary never
 *   sees an instantaneous score jump (the original rollback-cascade trigger);
 * - EVALUATED ONCE PER ADAPTATION TICK (never per variant): the caller
 *   evaluates the predicate once per tick and applies the resulting
 *   engagement to the baseline AND the variant score SYMMETRICALLY, so the
 *   variant-vs-baseline comparison stays fair;
 * - THE DEADBAND IS PRESERVED: `RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15`
 *   lives in the metric detectors (`detectSteeringOscillation` in
 *   `runtime.adaptation.ts`), so a gentle chicane correction still yields a
 *   metric of exactly 0 and is never penalized;
 * - BELOW-THRESHOLD NETWORKS ARE NEVER PENALIZED: the node-count gate is a
 *   HARD gate — under the level's promotion threshold the law is fully
 *   disengaged and the hysteresis state is reset, so a young network can
 *   never accumulate penalty debt and trip a rollback cascade (AC-030).
 *
 * THE INTENTIONAL BOUNDARY ASYMMETRY (spec section 9.5 — do NOT "fix"): the
 * predicate engages at `nodeCount >= threshold` while promotion (section 9.4)
 * fires at `nodeCount > threshold`. The penalty is therefore already ramping
 * when a promotable crossing occurs, so there is NO step change in the
 * penalty at the promotion instant. The two operators MUST NOT be normalized
 * to a common comparison.
 *
 * THE LEVEL RE-BASE ON PROMOTION (a consequence of the level-relative
 * floor — NOT a bug and NOT a second asymmetry): the threshold is the
 * CURRENT level's promotion threshold, so a level 1 -> 2 promotion re-bases
 * the floor 300 -> 600 and the law disengages instantly at that boundary.
 * The drop is a DISENGAGEMENT (the score rises; it never falls), so it can
 * never cascade into a rollback, and the round-boundary state reset (AC-031)
 * owns the state cleanup that follows a level change.
 *
 * THE CONSTANT OWNERSHIP (no dual path): this module is the canonical owner
 * of the penalty weight, the commit gate, the commit boost and the TP-07
 * promotion node thresholds of the AI lane. `runtime.adaptation.ts` still
 * declares its own copies of the first three because that file is OUT of the
 * P2-S3-B packet boundary (READ-ONLY in this slice); the slice that wires
 * this predicate into the runtime MUST delete those three declarations there
 * and import them from here IN THE SAME STEP. This module deliberately does
 * NOT import them back from `runtime.adaptation.ts` — an import in that
 * direction would create a cycle once the runtime imports this predicate.
 *
 * THE REALM: pure, headless and deterministic — no RNG, no wall-clock reads,
 * no DOM, no worker globals. The same (input, state) pair always yields the
 * same engagement.
 *
 * @module
 */

/* ------------------------------------------------------------------------- *
 * Section 9.5 — the oscillation law's named constants
 * ------------------------------------------------------------------------- */

/**
 * Weight applied to the oscillation metric when the shared predicate removes
 * score from a baseline or a variant (AC-030).  The penalty is MULTIPLICATIVE
 * and PROPORTIONAL to the score, so it damps oscillation without inverting a
 * positive score into a negative one.
 */
export const RACING_OSCILLATION_PENALTY_WEIGHT = 0.25;

/**
 * Oscillation gate of the additive growth-commit threshold boost: the boost is
 * granted only while the oscillation metric is strictly ABOVE this value.
 * At or below the gate the growth-commit threshold is left unchanged.
 */
export const RACING_OSCILLATION_COMMIT_THRESHOLD = 0.5;

/**
 * Small additive boost added to the growth-commit threshold while the shared
 * predicate is engaged and the oscillation metric is above
 * {@link RACING_OSCILLATION_COMMIT_THRESHOLD}.  The boost is scaled by the
 * hysteresis ramp (never a step change) and is NOT multiplied by the metric.
 */
export const RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST = 0.03;

/**
 * TP-09 grace window of the shared predicate, counted in PREDICATE
 * EVALUATIONS (adaptation ticks) and NOT in raw simulation ticks.  The
 * engagement ramps in over this window while the oscillation signal is
 * sustained and ramps out over the same window while it is absent, so a
 * single noisy adaptation tick can never engage (or disengage) the law.
 * Arbitrary starting tuning point (like TP-07), tuned in Phase 2.
 */
export const NEATRACING_OSCILLATION_GRACE_TICKS = 4;

/**
 * TP-07 promotion node threshold of the lvl-1 -> lvl-2 band: the node count
 * at which the oscillation law engages for a session at level 1.  The
 * promotion itself fires at a STRICTLY LARGER node count (spec section 9.4)
 * — the asymmetry is intentional (see the module header).
 */
export const NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2 = 300;

/**
 * TP-07 promotion node threshold of the lvl-2 -> lvl-3 band.  It is also the
 * floor of the level-3 band, because the level ladder caps at level 3
 * (`NEATRACING_HUMAN_MAX_LEVEL`) and there is no further promotion to gate
 * on.
 */
export const NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3 = 600;

/* ------------------------------------------------------------------------- *
 * Section 9.5 — the shared predicate's public types
 * ------------------------------------------------------------------------- */

/**
 * The shared predicate's hysteresis state at one adaptation tick.
 *
 * The state is the ONLY memory the predicate keeps: it is threaded from one
 * tick to the next by the caller, which is what makes the engagement
 * hysteretic (a ramp, never a step change).
 */
export interface OscillationPredicateState {
  /** The engagement strength in [0, 1]: 0 = fully disengaged, 1 = fully engaged. */
  readonly ramp: number;
  /**
   * Consecutive predicate evaluations with an ACTIVE oscillation signal,
   * capped at the TP-09 grace window.  Reset to 0 by an inactive evaluation.
   */
  readonly sustainedTicks: number;
}

/**
 * The observation the shared predicate consumes at one adaptation tick.
 */
export interface OscillationEvaluationInput {
  /** Live node count of the evaluated network. */
  readonly nodeCount: number;
  /**
   * The oscillation metric — the maximum of the steering oscillation and the
   * score-window oscillation.  The metric is expected to be DEADBAND-FILTERED
   * already (the `RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15` deadband lives
   * in the metric detectors), so a metric of exactly 0 means "no oscillation".
   */
  readonly oscillationMetric: number;
  /** The session level whose promotion node threshold gates the law. */
  readonly level: number;
}

/**
 * The shared predicate's verdict for one adaptation tick.
 *
 * The verdict is evaluated ONCE per adaptation tick and then applied to the
 * baseline AND the variant score, which is what keeps the comparison fair.
 */
export interface OscillationEngagement {
  /** The hysteresis state AFTER this evaluation — the next tick's input state. */
  readonly state: OscillationPredicateState;
  /** True while the law is engaged at a non-zero strength. */
  readonly engaged: boolean;
  /** The engagement strength in [0, 1] (the hysteresis ramp). */
  readonly ramp: number;
  /** The fraction of a score removed by the oscillation penalty, in [0, 1]. */
  readonly penaltyFraction: number;
  /** The additive growth-commit threshold boost (0, or the boost scaled by the ramp). */
  readonly commitBoost: number;
}

/**
 * The symmetrically penalized score pair produced by
 * {@link applyOscillationPenaltySymmetrically}.
 */
export interface SymmetricOscillationScores {
  /** The penalized baseline score. */
  readonly baselineScore: number;
  /** The penalized variant score. */
  readonly variantScore: number;
}

/* ------------------------------------------------------------------------- *
 * Section 9.5 — the shared predicate's public surface
 * ------------------------------------------------------------------------- */

/**
 * Creates the initial (fully disengaged) hysteresis state of the shared
 * oscillation predicate.
 *
 * The returned state is the seed of one adaptation session's predicate
 * memory; it carries no penalty debt, so the very first evaluation of a
 * session starts from zero engagement.
 *
 * @returns The fresh, fully disengaged predicate state.
 *
 * @example
 * ```ts
 * const state = createOscillationPredicateState();
 * // { ramp: 0, sustainedTicks: 0 }
 * ```
 */
export function createOscillationPredicateState(): OscillationPredicateState {
  return { ramp: 0, sustainedTicks: 0 };
}

/**
 * Resolves the TP-07 promotion node threshold that gates the oscillation law
 * for a session level.
 *
 * Level 1 is gated by the lvl-1 -> lvl-2 band (300 nodes), level 2 by the
 * lvl-2 -> lvl-3 band (600 nodes) and level 3 keeps the lvl-2 -> lvl-3 band
 * as its floor because the level ladder caps at level 3.  An unknown level
 * falls back to the highest band, which is the conservative choice (a
 * network is penalized less eagerly, never more).
 *
 * @param level - The session level of the evaluated network.
 * @returns The node count at which the oscillation law engages for that level.
 *
 * @example
 * ```ts
 * resolveOscillationPromotionNodeThreshold(1); // 300
 * resolveOscillationPromotionNodeThreshold(2); // 600
 * ```
 */
export function resolveOscillationPromotionNodeThreshold(
  level: number,
): number {
  return (
    PROMOTION_NODE_THRESHOLDS_BY_LEVEL[level] ??
    NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3
  );
}

/**
 * Evaluates the shared oscillation predicate for ONE adaptation tick.
 *
 * The predicate is the single gate of BOTH coupled mechanisms: the
 * multiplicative score penalty (via {@link OscillationEngagement.penaltyFraction})
 * and the additive growth-commit threshold boost (via
 * {@link OscillationEngagement.commitBoost}).
 *
 * The engagement is hysteretic: an active oscillation signal ramps the
 * engagement UP by one grace step per evaluation and an inactive signal ramps
 * it DOWN by one grace step per evaluation, so neither the node-count
 * boundary nor a single noisy tick can move the penalty in a step.
 *
 * @param input - The tick observation (node count, oscillation metric, level).
 * @param state - The previous tick's hysteresis state (the session seed for
 *   the first tick).
 * @returns The engagement verdict, carrying the next tick's state.
 *
 * @example
 * ```ts
 * const state = createOscillationPredicateState();
 * const engagement = evaluateOscillationEngagement(
 *   { nodeCount: 300, oscillationMetric: 0.8, level: 1 },
 *   state,
 * );
 * // engagement.ramp === 0.25 (one grace step of the four-step window)
 * ```
 */
export function evaluateOscillationEngagement(
  input: OscillationEvaluationInput,
  state: OscillationPredicateState,
): OscillationEngagement {
  // Step 1: the node-count gate is a HARD gate — below the level's promotion
  // threshold the law is fully disengaged and the hysteresis state is reset,
  // so a below-threshold network never carries penalty debt upward.
  const threshold = resolveOscillationPromotionNodeThreshold(input.level);
  if (input.nodeCount < threshold) {
    return createDisengagedEngagement();
  }

  // Step 2: advance the hysteresis ramp over the TP-09 grace window.
  const nextState = advanceOscillationState(state, input.oscillationMetric);

  // Step 3: derive the penalty fraction and the commit boost from the ramp.
  return buildOscillationEngagement(nextState, input.oscillationMetric);
}

/**
 * Applies the oscillation penalty to ONE score.
 *
 * The penalty is multiplicative and proportional to the score, so it damps
 * the oscillation without inverting a positive score into a negative one.
 * Apply the SAME engagement to the baseline and to the variant score — that
 * is what makes the comparison symmetric.
 *
 * @param score - The unpenalized score (baseline or variant).
 * @param engagement - The engagement verdict of the current adaptation tick.
 * @returns The penalized score.
 *
 * @example
 * ```ts
 * const penalized = applyOscillationPenalty(10, engagement);
 * // 10 * (1 - engagement.penaltyFraction)
 * ```
 */
export function applyOscillationPenalty(
  score: number,
  engagement: OscillationEngagement,
): number {
  return score * (1 - engagement.penaltyFraction);
}

/**
 * Applies the oscillation penalty SYMMETRICALLY to the baseline and the
 * variant score of one adaptation tick.
 *
 * AC-030 requires the penalty to be applied to both sides of the comparison:
 * the penalty is evaluated ONCE per adaptation tick (never per variant) and
 * the resulting engagement damps the baseline and the variant by the SAME
 * fraction, so the variant-vs-baseline ordering survives the penalty.
 *
 * @param baselineScore - The unpenalized baseline score.
 * @param variantScore - The unpenalized variant score.
 * @param engagement - The engagement verdict of the current adaptation tick.
 * @returns The penalized baseline and variant scores.
 *
 * @example
 * ```ts
 * const scores = applyOscillationPenaltySymmetrically(10, 11, engagement);
 * // { baselineScore: 10 * (1 - f), variantScore: 11 * (1 - f) }
 * ```
 */
export function applyOscillationPenaltySymmetrically(
  baselineScore: number,
  variantScore: number,
  engagement: OscillationEngagement,
): SymmetricOscillationScores {
  return {
    baselineScore: applyOscillationPenalty(baselineScore, engagement),
    variantScore: applyOscillationPenalty(variantScore, engagement),
  };
}

/* ------------------------------------------------------------------------- *
 * Section 9.5 — private helpers (below the fold)
 * ------------------------------------------------------------------------- */

/** The first level of the level ladder (TP-12) — gated by the lvl-2 band. */
const LEVEL_FIRST = 1;

/** The second level of the level ladder — gated by the lvl-3 band. */
const LEVEL_SECOND = 2;

/** The final level of the level ladder (TP-12 cap) — no further promotion. */
const LEVEL_FINAL = 3;

/**
 * The TP-07 promotion node threshold per session level (the single table of
 * the fixed mapping — no per-level branching anywhere else).
 */
const PROMOTION_NODE_THRESHOLDS_BY_LEVEL: Readonly<Record<number, number>> = {
  [LEVEL_FIRST]: NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
  [LEVEL_SECOND]: NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
  [LEVEL_FINAL]: NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
};

/** The hysteresis step of one grace tick (a quarter of the grace window). */
const OSCILLATION_RAMP_STEP = 1 / NEATRACING_OSCILLATION_GRACE_TICKS;

/**
 * Advances the hysteresis state by one evaluation: an active oscillation
 * signal ramps the engagement up, an inactive signal ramps it down.  Neither
 * direction can move the ramp by more than one grace step, so the penalty
 * never steps.
 *
 * @param state - The previous tick's hysteresis state.
 * @param oscillationMetric - The tick's (deadband-filtered) oscillation metric.
 * @returns The next tick's hysteresis state.
 */
function advanceOscillationState(
  state: OscillationPredicateState,
  oscillationMetric: number,
): OscillationPredicateState {
  if (!isOscillationSignalActive(oscillationMetric)) {
    return {
      ramp: Math.max(0, state.ramp - OSCILLATION_RAMP_STEP),
      sustainedTicks: 0,
    };
  }

  return {
    ramp: Math.min(1, state.ramp + OSCILLATION_RAMP_STEP),
    sustainedTicks: Math.min(
      state.sustainedTicks + 1,
      NEATRACING_OSCILLATION_GRACE_TICKS,
    ),
  };
}

/**
 * Reports whether the tick's oscillation metric carries an oscillation
 * signal.  A metric of exactly 0 is the deadband-filtered "smooth driving"
 * verdict and never engages the law.
 *
 * @param oscillationMetric - The tick's (deadband-filtered) oscillation metric.
 * @returns True when an oscillation signal is present.
 */
function isOscillationSignalActive(oscillationMetric: number): boolean {
  return oscillationMetric > 0;
}

/**
 * Builds the engagement verdict from the advanced hysteresis state.
 *
 * @param state - The hysteresis state after this evaluation.
 * @param oscillationMetric - The tick's (deadband-filtered) oscillation metric.
 * @returns The engagement verdict of the tick.
 */
function buildOscillationEngagement(
  state: OscillationPredicateState,
  oscillationMetric: number,
): OscillationEngagement {
  return {
    state,
    engaged: state.ramp > 0,
    ramp: state.ramp,
    penaltyFraction:
      state.ramp * oscillationMetric * RACING_OSCILLATION_PENALTY_WEIGHT,
    commitBoost: resolveCommitBoost(state.ramp, oscillationMetric),
  };
}

/**
 * Resolves the additive growth-commit threshold boost of the tick: the boost
 * is granted only above the commit gate and is scaled by the hysteresis ramp
 * so it ramps in and out instead of stepping.
 *
 * @param ramp - The tick's engagement strength in [0, 1].
 * @param oscillationMetric - The tick's (deadband-filtered) oscillation metric.
 * @returns The additive growth-commit threshold boost.
 */
function resolveCommitBoost(ramp: number, oscillationMetric: number): number {
  if (oscillationMetric <= RACING_OSCILLATION_COMMIT_THRESHOLD) {
    return 0;
  }

  return ramp * RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST;
}

/**
 * Builds the fully disengaged verdict of a below-threshold tick.  It also
 * resets the hysteresis state, so a network that drops under the promotion
 * threshold never carries engagement debt back over the boundary.
 *
 * @returns The disengaged engagement verdict.
 */
function createDisengagedEngagement(): OscillationEngagement {
  return {
    state: createOscillationPredicateState(),
    engaged: false,
    ramp: 0,
    penaltyFraction: 0,
    commitBoost: 0,
  };
}
