/** Default difficulty scale for rollouts when caller does not provide one. */
export const FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE = 1;

/** Default grace period (frames) before early termination checks begin. */
export const FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES = 160;

/** Default consecutive unrecoverable frames required for early termination. */
export const FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES = 24;

/** Default pipe-progress target used when normalizing rollout fitness. */
export const FLAPPY_EVALUATION_DEFAULT_PIPE_PROGRESS_TARGET = 20;

/** Dense shaping normalization factor per survived frame. */
export const FLAPPY_EVALUATION_DENSE_SHAPING_FRAMES_NORMALIZER = 4.5;

/** Survival channel weight in normalized fitness composition. */
export const FLAPPY_EVALUATION_NORMALIZED_SURVIVAL_WEIGHT = 2_600;

/** Pipe-progress channel weight in normalized fitness composition. */
export const FLAPPY_EVALUATION_NORMALIZED_PROGRESS_WEIGHT = 1_400;

/** Dense-shaping channel weight in normalized fitness composition. */
export const FLAPPY_EVALUATION_NORMALIZED_DENSE_WEIGHT = 1_200;

/** Terminal-shaping channel weight in normalized fitness composition. */
export const FLAPPY_EVALUATION_NORMALIZED_TERMINAL_WEIGHT = 600;

/** Robust fitness penalty multiplier applied to standard deviation. */
export const FLAPPY_EVALUATION_ROBUST_STDDEV_PENALTY = 0.35;

/** Unrecoverable clearance threshold used by early termination heuristic. */
export const FLAPPY_EVALUATION_UNRECOVERABLE_CLEARANCE_THRESHOLD = -0.7;

/** Lower-gap delta threshold used by early termination heuristic. */
export const FLAPPY_EVALUATION_UNRECOVERABLE_BELOW_GAP_DELTA = 0.45;

/** Falling-speed threshold used by early termination heuristic. */
export const FLAPPY_EVALUATION_UNRECOVERABLE_FALLING_VELOCITY = 0.5;

/** Upper-gap delta threshold used by early termination heuristic. */
export const FLAPPY_EVALUATION_UNRECOVERABLE_ABOVE_GAP_DELTA = -0.45;

/** Rising-speed threshold used by early termination heuristic. */
export const FLAPPY_EVALUATION_UNRECOVERABLE_RISING_VELOCITY = -0.5;

/** Seed-mix additive constant used to decorrelate nearby genome ids. */
export const FLAPPY_EVALUATION_SEED_MIX_XOR_SALT = 0x9e3779b9;

/** Seed-mix first multiplicative avalanche constant. */
export const FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_A = 0x85ebca6b;

/** Seed-mix second multiplicative avalanche constant. */
export const FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_B = 0xc2b2ae35;
