/**
 * Default number of nearest neighbors used when computing novelty.
 *
 * Small values keep novelty sensitive to local behavioral differences without requiring a large
 * archive or population.
 */
export const NOVELTY_DEFAULT_NEIGHBORS = 3;

/**
 * Default blend factor used when mixing novelty into an existing fitness score.
 *
 * A mid-range value keeps novelty influential without letting exploratory behavior completely drown
 * out task performance.
 */
export const NOVELTY_DEFAULT_BLEND = 0.3;

/**
 * Maximum number of descriptors retained in the novelty archive.
 *
 * The cap keeps novelty history useful for exploration while preventing unbounded memory growth.
 */
export const NOVELTY_ARCHIVE_CAP = 200;

/**
 * Target variance used by entropy-sharing tuning.
 *
 * The controller nudges sharing sigma toward a population whose entropy spread is neither too flat
 * nor too unstable.
 */
export const ENTROPY_VAR_TARGET_DEFAULT = 0.2;

/** Default step size used when entropy-sharing tuning increases or decreases sharing sigma. */
export const ENTROPY_VAR_ADJUST_DEFAULT = 0.1;

/** Lower bound for the sharing sigma used by entropy-sharing adaptation. */
export const ENTROPY_VAR_MIN_SIGMA_DEFAULT = 0.1;

/** Upper bound for the sharing sigma used by entropy-sharing adaptation. */
export const ENTROPY_VAR_MAX_SIGMA_DEFAULT = 10;

/** Lower tolerance band for deciding that observed entropy variance is meaningfully low. */
export const ENTROPY_VAR_LOW_BAND = 0.9;

/** Upper tolerance band for deciding that observed entropy variance is meaningfully high. */
export const ENTROPY_VAR_HIGH_BAND = 1.1;

/**
 * Target mean entropy used when tuning the compatibility threshold.
 *
 * The goal is to keep speciation pressure near a stable diversity level instead of drifting toward
 * either species collapse or fragmentation.
 */
export const ENTROPY_TARGET_DEFAULT = 0.5;

/** Deadband around the entropy target where compatibility tuning intentionally does nothing. */
export const ENTROPY_DEADBAND_DEFAULT = 0.05;

/** Default rate used when compatibility tuning nudges the threshold upward or downward. */
export const ENTROPY_ADJUST_DEFAULT = 0.05;

/** Baseline compatibility threshold used when no explicit value is configured. */
export const COMPAT_THRESHOLD_DEFAULT = 3;

/** Minimum compatibility threshold allowed during automatic compatibility tuning. */
export const COMPAT_MIN_THRESHOLD_DEFAULT = 0.5;

/** Maximum compatibility threshold allowed during automatic compatibility tuning. */
export const COMPAT_MAX_THRESHOLD_DEFAULT = 10;

/**
 * Default rate used when auto distance-coefficient tuning rebalances structural distance weights.
 */
export const AUTO_COEFF_ADJUST_DEFAULT = 0.05;

/** Minimum structural-distance coefficient allowed during automatic tuning. */
export const AUTO_COEFF_MIN_DEFAULT = 0.05;

/** Maximum structural-distance coefficient allowed during automatic tuning. */
export const AUTO_COEFF_MAX_DEFAULT = 8;

/** Baseline structural-distance coefficient used before any automatic tuning occurs. */
export const DISTANCE_COEFF_DEFAULT = 1;

/** Multiplier below which observed variance is treated as a meaningful decrease. */
export const VARIANCE_DECREASE_THRESHOLD = 0.95;

/** Multiplier above which observed variance is treated as a meaningful increase. */
export const VARIANCE_INCREASE_THRESHOLD = 1.05;
