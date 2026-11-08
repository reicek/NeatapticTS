/**
 * Green tests for the NeatRacing shared oscillation predicate
 * (`examples/neatRacing/ai/oscillation-penalty.utils.ts` — Step 03, slice
 * P2-S3-B, TDD lane green-only).
 *
 * Contracts pinned here (AC-030):
 * - ONE shared predicate drives BOTH coupled mechanisms — the multiplicative
 *   oscillation penalty and the additive growth-commit threshold boost (E3).
 * - The node-count gate is the CURRENT LEVEL's promotion threshold (TP-07:
 *   300 at lvl 1 -> 2, 600 at lvl 2 -> 3), replacing the retired fixed
 *   200-neuron floor: below-threshold networks are NEVER penalized and carry
 *   NO penalty debt upward, so they can never trip a rollback cascade.
 * - The INTENTIONAL boundary asymmetry: the predicate engages at
 *   `nodeCount >= threshold` while promotion fires at `nodeCount > threshold`.
 *   The two operators MUST NOT be normalized — the penalty must already be
 *   ramping when a promotable crossing occurs.
 * - HYSTERESIS (no step changes): the engagement ramps IN over the TP-09
 *   grace window while the oscillation signal is sustained and ramps OUT over
 *   the same window while it is absent.  A node-count boundary crossing
 *   therefore sees ONE grace step of penalty, never the full weight — that is
 *   the original rollback-cascade trigger, closed.
 * - The penalty is evaluated ONCE per adaptation tick and applied
 *   SYMMETRICALLY to the baseline and the variant score, so the
 *   variant-vs-baseline comparison stays fair.
 * - The `RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15` deadband is preserved:
 *   a gentle chicane correction yields a metric of exactly 0 and never
 *   engages the law.
 *
 * The suite deliberately does NOT depend on the runtime's own copies of the
 * penalty-law constants (`runtime.adaptation.ts`): the shared predicate is the
 * canonical owner of those values, and the slice that wires the predicate into
 * the runtime deletes the runtime's copies in the same step.
 *
 * @module
 */

import {
  NEATRACING_OSCILLATION_GRACE_TICKS,
  NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
  NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
  RACING_OSCILLATION_COMMIT_THRESHOLD,
  RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
  RACING_OSCILLATION_PENALTY_WEIGHT,
  applyOscillationPenalty,
  applyOscillationPenaltySymmetrically,
  createOscillationPredicateState,
  evaluateOscillationEngagement,
  resolveOscillationPromotionNodeThreshold,
  type OscillationEngagement,
  type OscillationEvaluationInput,
  type OscillationPredicateState,
} from './oscillation-penalty.utils';
import {
  RACING_OSCILLATION_MIN_MEAN_STEERING,
  detectSteeringOscillation,
} from './runtime.adaptation';

// ── Fixtures ─────────────────────────────────────────────────────────────────

/** The first level of the NeatRacing level ladder (TP-12). */
const LEVEL_FIRST = 1;

/** The second level of the NeatRacing level ladder (TP-12). */
const LEVEL_SECOND = 2;

/** The final level of the NeatRacing level ladder (TP-12 cap). */
const LEVEL_FINAL = 3;

/** A level outside the ladder — exercises the conservative fallback band. */
const LEVEL_UNKNOWN = 99;

/** An oscillation metric ABOVE the growth-commit gate (0.5). */
const METRIC_OSCILLATING = 0.8;

/** An oscillation metric BELOW the growth-commit gate but above the deadband. */
const METRIC_MILD = 0.2;

/** The metric the deadband filter publishes for a gentle correction. */
const METRIC_DEADBANDED = 0;

/** The retired fixed neuron floor the level threshold replaces (AC-030). */
const RETIRED_FIXED_NODE_FLOOR = 200;

/** The baseline score used by the penalty and symmetry fixtures. */
const BASELINE_SCORE = 10;

/** The variant score used by the symmetry fixtures (above the baseline). */
const VARIANT_SCORE = 11;

/** One grace step of the engagement ramp (a quarter of the grace window). */
const RAMP_STEP = 1 / NEATRACING_OSCILLATION_GRACE_TICKS;

/** A gentle chicane correction: sign changes, but below the 0.15 deadband. */
const GENTLE_CHICANE_ANGLES: readonly number[] = [0.1, -0.1, 0.1, -0.1];

/** An aggressive zig-zag: sign changes above the deadband (metric of 1). */
const AGGRESSIVE_ZIGZAG_ANGLES: readonly number[] = [0.6, -0.6, 0.6, -0.6];

/**
 * Builds one adaptation-tick observation for the shared predicate.
 *
 * @param nodeCount - The live node count of the evaluated network.
 * @param oscillationMetric - The deadband-filtered oscillation metric.
 * @param level - The session level whose promotion threshold gates the law.
 * @returns The tick observation.
 */
function buildInput(
  nodeCount: number,
  oscillationMetric: number,
  level: number = LEVEL_FIRST,
): OscillationEvaluationInput {
  return { nodeCount, oscillationMetric, level };
}

/**
 * Evaluates the shared predicate for a run of adaptation ticks against a
 * CONSTANT observation, threading the hysteresis state from tick to tick.
 *
 * @param input - The tick observation, repeated for every tick.
 * @param tickCount - The number of evaluations to run (at least one).
 * @param seedState - The session's initial hysteresis state.
 * @returns The engagement verdict of the LAST tick.
 */
function evaluateTicks(
  input: OscillationEvaluationInput,
  tickCount: number,
  seedState: OscillationPredicateState = createOscillationPredicateState(),
): OscillationEngagement {
  let state = seedState;
  let engagement = evaluateOscillationEngagement(input, state);

  for (let tick = 1; tick < tickCount; tick += 1) {
    state = engagement.state;
    engagement = evaluateOscillationEngagement(input, state);
  }

  return engagement;
}

/**
 * Collects the per-tick engagement ramp of a full ramp-IN / ramp-OUT sweep:
 * four sustained ticks followed by four inactive ticks.
 *
 * @returns The per-tick engagement strengths of the sweep.
 */
function collectRampSweep(): number[] {
  const ramp: number[] = [];
  const rampingIn = buildInput(
    NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
    METRIC_OSCILLATING,
  );
  const rampingOut = buildInput(
    NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
    METRIC_DEADBANDED,
  );
  let state = createOscillationPredicateState();

  for (let tick = 0; tick < NEATRACING_OSCILLATION_GRACE_TICKS; tick += 1) {
    const engagement = evaluateOscillationEngagement(rampingIn, state);
    ramp.push(engagement.ramp);
    state = engagement.state;
  }

  for (let tick = 0; tick < NEATRACING_OSCILLATION_GRACE_TICKS; tick += 1) {
    const engagement = evaluateOscillationEngagement(rampingOut, state);
    ramp.push(engagement.ramp);
    state = engagement.state;
  }

  return ramp;
}

/**
 * Reports the largest absolute step between two consecutive values of a
 * trace.
 *
 * @param trace - The sampled values, in chronological order.
 * @returns The largest absolute consecutive difference.
 */
function findMaxStep(trace: readonly number[]): number {
  let maxStep = 0;

  for (let index = 1; index < trace.length; index += 1) {
    const previous = trace[index - 1] ?? 0;
    const current = trace[index] ?? 0;
    maxStep = Math.max(maxStep, Math.abs(current - previous));
  }

  return maxStep;
}

// ── Tests ────────────────────────────────────────────────────────────────────

describe('promotion node threshold (TP-07)', () => {
  it('gates level 1 at the lvl-1 -> lvl-2 promotion threshold', () => {
    expect(resolveOscillationPromotionNodeThreshold(LEVEL_FIRST)).toBe(
      NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
    );
  });

  it('gates level 2 at the lvl-2 -> lvl-3 promotion threshold', () => {
    expect(resolveOscillationPromotionNodeThreshold(LEVEL_SECOND)).toBe(
      NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
    );
  });

  it('keeps the lvl-2 -> lvl-3 band as the floor of the capped level-3 ladder', () => {
    expect(resolveOscillationPromotionNodeThreshold(LEVEL_FINAL)).toBe(
      NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
    );
  });

  it('falls back to the highest band for an unknown level', () => {
    expect(resolveOscillationPromotionNodeThreshold(LEVEL_UNKNOWN)).toBe(
      NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
    );
  });
});

describe('node-count gate (AC-030 — below-threshold never penalized)', () => {
  it('starts fully disengaged with no penalty debt', () => {
    expect(createOscillationPredicateState()).toEqual({
      ramp: 0,
      sustainedTicks: 0,
    });
  });

  it('never engages a network below the level promotion threshold', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2 - 1,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement).toEqual({
      state: { ramp: 0, sustainedTicks: 0 },
      engaged: false,
      ramp: 0,
      penaltyFraction: 0,
      commitBoost: 0,
    });
  });

  it('drops the hysteresis state when the network falls back below the threshold', () => {
    const engaged = evaluateTicks(
      buildInput(NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2, METRIC_OSCILLATING),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
    const fallen = evaluateOscillationEngagement(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2 - 1,
        METRIC_OSCILLATING,
      ),
      engaged.state,
    );

    expect(fallen.state).toEqual({ ramp: 0, sustainedTicks: 0 });
  });

  it('retires the fixed 200-neuron floor in favour of the level threshold', () => {
    const engagement = evaluateTicks(
      buildInput(RETIRED_FIXED_NODE_FLOOR, METRIC_OSCILLATING),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.ramp).toBe(0);
  });

  it('engages at exactly the promotion threshold (the intentional >= asymmetry)', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      1,
    );

    expect(engagement.ramp).toBeCloseTo(RAMP_STEP, 12);
  });

  it('keeps a level-2 network below its own threshold unpenalized', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3 - 1,
        METRIC_OSCILLATING,
        LEVEL_SECOND,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.ramp).toBe(0);
  });

  it('engages a level-2 network at the lvl-2 -> lvl-3 threshold', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3,
        METRIC_OSCILLATING,
        LEVEL_SECOND,
      ),
      1,
    );

    expect(engagement.ramp).toBeCloseTo(RAMP_STEP, 12);
  });
});

describe('hysteresis grace window (TP-09)', () => {
  it('ramps the engagement up by one grace step per sustained evaluation', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      3,
    );

    expect(engagement.ramp).toBeCloseTo(3 * RAMP_STEP, 12);
  });

  it('reaches full engagement only at the end of the grace window', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.ramp).toBeCloseTo(1, 12);
  });

  it('caps the sustained tick counter at the grace window', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS + 5,
    );

    expect(engagement.state.sustainedTicks).toBe(
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
  });

  it('ramps the engagement down by one grace step per inactive evaluation', () => {
    const engaged = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
    const fading = evaluateOscillationEngagement(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_DEADBANDED,
      ),
      engaged.state,
    );

    expect(fading.ramp).toBeCloseTo(1 - RAMP_STEP, 12);
  });

  it('resets the sustained tick counter on an inactive evaluation', () => {
    const engaged = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
    const fading = evaluateOscillationEngagement(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_DEADBANDED,
      ),
      engaged.state,
    );

    expect(fading.state.sustainedTicks).toBe(0);
  });

  it('changes the engagement by at most one grace step per tick', () => {
    expect(findMaxStep(collectRampSweep())).toBeCloseTo(RAMP_STEP, 12);
  });

  it('ramps the penalty in over the grace window when a grown network crosses the threshold', () => {
    // THE ROLLBACK-CASCADE GUARD: a network that grew across the promotion
    // threshold sees ONE grace step of penalty on the crossing tick — never
    // the full weight, so the score never jumps in a single tick.
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      1,
    );

    expect(engagement.penaltyFraction).toBeCloseTo(
      METRIC_OSCILLATING * RACING_OSCILLATION_PENALTY_WEIGHT * RAMP_STEP,
      12,
    );
  });
});

describe('penalty fraction (multiplicative, proportional)', () => {
  it('scales the penalty fraction by the ramp, the metric and the penalty weight', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.penaltyFraction).toBeCloseTo(
      RACING_OSCILLATION_PENALTY_WEIGHT * METRIC_OSCILLATING,
      12,
    );
  });

  it('never removes more than the penalty weight of the score', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.penaltyFraction).toBeLessThanOrEqual(
      RACING_OSCILLATION_PENALTY_WEIGHT,
    );
  });

  it('damps a positive score without inverting its sign', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(applyOscillationPenalty(BASELINE_SCORE, engagement)).toBeGreaterThan(0);
  });

  it('damps the score by exactly the penalty fraction', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(applyOscillationPenalty(BASELINE_SCORE, engagement)).toBeCloseTo(
      BASELINE_SCORE * (1 - engagement.penaltyFraction),
      12,
    );
  });

  it('leaves a disengaged score untouched', () => {
    const disengaged = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2 - 1,
        METRIC_OSCILLATING,
      ),
      1,
    );

    expect(applyOscillationPenalty(BASELINE_SCORE, disengaged)).toBe(
      BASELINE_SCORE,
    );
  });
});

describe('symmetry (AC-030 — one engagement, both sides)', () => {
  it('applies the same fraction to the baseline and the variant score', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
    const scores = applyOscillationPenaltySymmetrically(
      BASELINE_SCORE,
      VARIANT_SCORE,
      engagement,
    );

    expect(scores.baselineScore / BASELINE_SCORE).toBeCloseTo(
      scores.variantScore / VARIANT_SCORE,
      12,
    );
  });

  it('preserves the baseline-versus-variant ordering', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
    const scores = applyOscillationPenaltySymmetrically(
      BASELINE_SCORE,
      VARIANT_SCORE,
      engagement,
    );

    expect(scores.variantScore).toBeGreaterThan(scores.baselineScore);
  });

  it('applies the single-tick engagement to both sides of the comparison', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );
    const scores = applyOscillationPenaltySymmetrically(
      BASELINE_SCORE,
      VARIANT_SCORE,
      engagement,
    );

    expect(scores.baselineScore).toBeCloseTo(
      applyOscillationPenalty(BASELINE_SCORE, engagement),
      12,
    );
  });
});

describe('growth-commit threshold boost (the additive mechanism)', () => {
  it('grants no boost at the commit threshold', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        RACING_OSCILLATION_COMMIT_THRESHOLD,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.commitBoost).toBe(0);
  });

  it('grants no boost below the commit threshold', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_MILD,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.commitBoost).toBe(0);
  });

  it('grants the boost above the commit threshold', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.commitBoost).toBeCloseTo(
      RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
      12,
    );
  });

  it('scales the boost down with the ramp (no step change)', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      1,
    );

    expect(engagement.commitBoost).toBeCloseTo(
      RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST * RAMP_STEP,
      12,
    );
  });
});

describe('deadband (RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15)', () => {
  it('pins the steering deadband at 0.15', () => {
    expect(RACING_OSCILLATION_MIN_MEAN_STEERING).toBe(0.15);
  });

  it('never engages on a deadband-filtered metric of exactly zero', () => {
    const engagement = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_DEADBANDED,
      ),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.ramp).toBe(0);
  });

  it('keeps a gentle chicane correction unpenalized end to end', () => {
    const metric = detectSteeringOscillation(GENTLE_CHICANE_ANGLES);
    const engagement = evaluateTicks(
      buildInput(NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2, metric),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.penaltyFraction).toBe(0);
  });

  it('engages on an aggressive zig-zag above the deadband', () => {
    const metric = detectSteeringOscillation(AGGRESSIVE_ZIGZAG_ANGLES);
    const engagement = evaluateTicks(
      buildInput(NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2, metric),
      NEATRACING_OSCILLATION_GRACE_TICKS,
    );

    expect(engagement.penaltyFraction).toBeCloseTo(
      RACING_OSCILLATION_PENALTY_WEIGHT,
      12,
    );
  });
});

describe('purity and determinism', () => {
  it('never mutates the caller-owned hysteresis state', () => {
    const state = createOscillationPredicateState();

    evaluateOscillationEngagement(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      state,
    );

    expect(state).toEqual({ ramp: 0, sustainedTicks: 0 });
  });

  it('is deterministic across repeated identical evaluations', () => {
    const first = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      3,
    );
    const second = evaluateTicks(
      buildInput(
        NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2,
        METRIC_OSCILLATING,
      ),
      3,
    );

    expect(second).toEqual(first);
  });

  it('pins the shared penalty law values and the grace window verbatim', () => {
    expect([
      RACING_OSCILLATION_PENALTY_WEIGHT,
      RACING_OSCILLATION_COMMIT_THRESHOLD,
      RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST,
      NEATRACING_OSCILLATION_GRACE_TICKS,
    ]).toEqual([0.25, 0.5, 0.03, 4]);
  });
});
