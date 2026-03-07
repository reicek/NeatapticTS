/**
 * Flappy reward-shaping constants.
 *
 * These values tune the learning signal seen by evolution. Separating them from
 * world/physics constants keeps behavior tuning explicit and easier to teach.
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

/** Per-frame reward weight for staying vertically aligned with the next gap. */
export const FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME = 0.9;

/** Reward scale for reducing distance to the next pipe between consecutive frames. */
export const FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT = 320;

/** Reward scale for reducing vertical error to the next gap center. */
export const FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT = 320;

/** Per-frame reward weight for keeping the bird inside next-gap clearance. */
export const FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME = 0.35;

/** Per-frame reward weight for pre-aligning with the second upcoming gap. */
export const FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME = 0.2;

/** Per-frame reward weight for maintaining controllable vertical velocity. */
export const FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME = 0.2;

/** Terminal bonus based on final alignment with the next gap center. */
export const FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT = 180;

/** Terminal bonus based on final progress toward the next pipe. */
export const FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT = 80;
