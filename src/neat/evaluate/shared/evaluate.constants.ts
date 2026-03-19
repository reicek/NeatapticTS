/**
 * Shared default constants for the NEAT evaluate chapter.
 *
 * These constants anchor the small policy loops used across evaluation. The
 * root evaluate chapter explains the full scoring-and-adaptation flow, while
 * the helper chapters consume these values as stable defaults for novelty,
 * entropy-sharing, entropy-compatibility, and auto-distance tuning.
 *
 * Read this chapter when you want to answer questions such as:
 * - Which defaults shape novelty exploration before the user configures
 *   anything?
 * - What target values and bands anchor the entropy-based tuning loops?
 * - Which bounds keep compatibility and distance tuning from drifting too far?
 * - How does the evaluate subtree keep its small policy helpers aligned with
 *   one another?
 *
 * The exports below group into four families:
 * - novelty defaults,
 * - entropy-sharing defaults,
 * - entropy-compatibility defaults,
 * - auto-distance and variance-baseline defaults.
 */

// Shared evaluation constants begin here.

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

/**
 * Default step size used when entropy-sharing tuning increases or decreases sharing sigma.
 *
 * A modest default keeps the sigma controller responsive without letting one noisy variance read
 * swing the sharing radius too aggressively.
 */
export const ENTROPY_VAR_ADJUST_DEFAULT = 0.1;

/**
 * Lower bound for the sharing sigma used by entropy-sharing adaptation.
 *
 * The lower clamp prevents the sharing radius from shrinking so far that later sharing pressure
 * becomes hypersensitive to tiny entropy fluctuations.
 */
export const ENTROPY_VAR_MIN_SIGMA_DEFAULT = 0.1;

/**
 * Upper bound for the sharing sigma used by entropy-sharing adaptation.
 *
 * The upper clamp prevents the radius from widening until entropy sharing loses practical
 * contrast across the population.
 */
export const ENTROPY_VAR_MAX_SIGMA_DEFAULT = 10;

/**
 * Lower tolerance band for deciding that observed entropy variance is meaningfully low.
 *
 * Values below this multiplier mark a population whose entropy spread is flatter than the tuning
 * target expects.
 */
export const ENTROPY_VAR_LOW_BAND = 0.9;

/**
 * Upper tolerance band for deciding that observed entropy variance is meaningfully high.
 *
 * Values above this multiplier mark a population whose entropy spread is noisier than the tuning
 * target expects.
 */
export const ENTROPY_VAR_HIGH_BAND = 1.1;

/**
 * Target mean entropy used when tuning the compatibility threshold.
 *
 * The goal is to keep speciation pressure near a stable diversity level instead of drifting toward
 * either species collapse or fragmentation.
 */
export const ENTROPY_TARGET_DEFAULT = 0.5;

/**
 * Deadband around the entropy target where compatibility tuning intentionally does nothing.
 *
 * The deadband reduces threshold jitter by treating small entropy deviations as normal noise
 * rather than signals that demand a policy change.
 */
export const ENTROPY_DEADBAND_DEFAULT = 0.05;

/**
 * Default rate used when compatibility tuning nudges the threshold upward or downward.
 *
 * A conservative step size keeps the compatibility-threshold loop gradual enough that later
 * speciation reads remain interpretable from one generation to the next.
 */
export const ENTROPY_ADJUST_DEFAULT = 0.05;

/**
 * Baseline compatibility threshold used when no explicit value is configured.
 *
 * This serves as the neutral starting point before entropy-based tuning or explicit user policy
 * begins to reshape species pressure.
 */
export const COMPAT_THRESHOLD_DEFAULT = 3;

/**
 * Minimum compatibility threshold allowed during automatic compatibility tuning.
 *
 * The lower clamp prevents the threshold from collapsing until even small structural differences
 * force unnecessary species fragmentation.
 */
export const COMPAT_MIN_THRESHOLD_DEFAULT = 0.5;

/**
 * Maximum compatibility threshold allowed during automatic compatibility tuning.
 *
 * The upper clamp prevents compatibility from becoming so permissive that species boundaries lose
 * practical meaning.
 */
export const COMPAT_MAX_THRESHOLD_DEFAULT = 10;

/**
 * Default rate used when auto distance-coefficient tuning rebalances structural distance weights.
 *
 * This shared step size controls how quickly excess and disjoint coefficients react when topology
 * variance drifts away from the recent baseline.
 */
export const AUTO_COEFF_ADJUST_DEFAULT = 0.05;

/**
 * Minimum structural-distance coefficient allowed during automatic tuning.
 *
 * The lower bound keeps structural differences meaningful even when the auto-distance policy is
 * softening species pressure.
 */
export const AUTO_COEFF_MIN_DEFAULT = 0.05;

/**
 * Maximum structural-distance coefficient allowed during automatic tuning.
 *
 * The upper bound prevents excess and disjoint penalties from dominating every later compatibility
 * comparison.
 */
export const AUTO_COEFF_MAX_DEFAULT = 8;

/**
 * Baseline structural-distance coefficient used before any automatic tuning occurs.
 *
 * This is the neutral starting point for the structural-distance fold before variance-driven
 * updates begin to reshape it.
 */
export const DISTANCE_COEFF_DEFAULT = 1;

/**
 * Multiplier below which observed variance is treated as a meaningful decrease.
 *
 * Drops below this band tell the auto-distance loop that topology sizes are converging relative to
 * the recent baseline.
 */
export const VARIANCE_DECREASE_THRESHOLD = 0.95;

/**
 * Multiplier above which observed variance is treated as a meaningful increase.
 *
 * Values above this band tell the auto-distance loop that topology sizes are spreading relative to
 * the recent baseline.
 */
export const VARIANCE_INCREASE_THRESHOLD = 1.05;
