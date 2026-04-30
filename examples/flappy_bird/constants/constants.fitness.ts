/**
 * Flappy reward-shaping constants.
 *
 * These values tune the learning signal seen by evolution. Separating them from
 * world/physics constants keeps behavior tuning explicit and easier to teach.
 *
 * Design philosophy: centering quality is the primary signal. An agent that
 * stays aligned with the next gap center — measured against the gap width, not
 * absolute world height — earns more signal per frame than one that merely
 * survives. Survival and pipe-passing remain important, but centering dominates
 * the dense-shaping channel so the population does not converge on passive or
 * erratic drift strategies.
 */

/** Fitness bonus added per pipe successfully passed. */
export const FLAPPY_FITNESS_BONUS_PER_PIPE = 1_000;

/**
 * Fraction of raw frame-survival reward kept in total fitness.
 *
 * Lower values reduce the incentive to merely stay alive and increase pressure
 * to center on gaps and pass pipes cleanly.
 */
export const FLAPPY_FITNESS_SURVIVAL_WEIGHT = 0.45;

/**
 * Per-frame reward weight for staying vertically aligned with the next gap.
 *
 * Increased to make gap alignment a primary per-frame signal so that birds
 * tracking the ideal center path score noticeably higher than drifting birds.
 */
export const FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME = 3.0;

/** Reward scale for reducing distance to the next pipe between consecutive frames. */
export const FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT = 320;

/** Reward scale for reducing vertical error to the next gap center. */
export const FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT = 320;

/**
 * Per-frame reward weight for keeping the bird inside next-gap clearance.
 *
 * Increased to make clearance (gap-width-normalized position) the primary
 * per-frame centering signal. This strongly rewards birds that stay near
 * the gap center and penalizes those that drift to the edges.
 */
export const FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME = 2.5;

/**
 * Per-frame reward weight for maintaining controllable vertical velocity.
 *
 * Increased to penalize monotonic-direction behavior (always rising or always
 * falling) more aggressively. Birds with wild or one-directional velocity
 * profiles score lower than those with controlled flight.
 */
export const FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME = 1.5;

/** Terminal bonus based on final alignment with the next gap center. */
export const FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT = 180;

/** Terminal bonus based on final progress toward the next pipe. */
export const FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT = 80;

/**
 * Per-frame reward weight for gap-width-normalized centering quality.
 *
 * This term uses the normalizedNextGapClearance feature, which is proportional
 * to how well-centered the bird is relative to the current gap size rather than
 * absolute world height. This is the dominant centering signal — a bird that
 * stays at gap center earns full reward regardless of gap size, while one near
 * the edges earns near zero.
 */
export const FLAPPY_FITNESS_GAP_CENTERING_QUALITY_WEIGHT_PER_FRAME = 3.0;

/**
 * Minimum frames survived before the early-death penalty is waived.
 *
 * Set below the ~104-frame natural-fall time so that untrained birds falling
 * straight to the floor are not catastrophically penalized before NEAT can
 * explore better policies. Monotonic strategies that die unusually quickly
 * (ceiling rockets or instant drops) still trigger the penalty.
 *
 * Previous value of 120 was above the natural-fall time, which crushed NARX
 * signal to near-zero on early generations. Lowering to 80 lets the density
 * shaping guide evolution without the early-death gate blocking the signal.
 */
export const FLAPPY_FITNESS_EARLY_DEATH_FRAME_THRESHOLD = 80;

/**
 * Fitness multiplier applied when a bird dies before the early-death threshold.
 *
 * Set below 1 so extremely short-lived monotonic deaths score lower than any
 * reasonable centering strategy. A value of 0.4 (vs the prior 0.1) keeps the
 * penalty meaningful while preserving enough signal for NEAT to distinguish
 * between bad and worse short-lived strategies.
 */
export const FLAPPY_FITNESS_EARLY_DEATH_PENALTY_MULTIPLIER = 0.4;
