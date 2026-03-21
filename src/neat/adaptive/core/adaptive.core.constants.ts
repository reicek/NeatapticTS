/**
 * Shared defaults, labels, and numeric guard rails for the adaptive subtree.
 *
 * The adaptive chapters reuse this file so they can talk about policy with one
 * stable vocabulary for thresholds, strategy names, phase labels, and fallback
 * numbers instead of redefining those values inline.
 *
 * Read this after the contract map in `adaptive.core.types.ts`. The types file
 * answers "which knobs and scratch fields exist?" while this file answers
 * "what should those knobs and scratch fields default to when the caller does
 * not override them?"
 *
 * Read the exports as four families rather than one long constant shelf:
 *
 * - tiny numeric helpers such as `ZERO`, `ONE`, and `NEGATIVE_ONE` keep the
 *   utility math explicit without scattering magic numbers,
 * - schedule defaults such as `DEFAULT_IMPROVEMENT_WINDOW`,
 *   `DEFAULT_CB_INCREASE_FACTOR`, and `PHASE_LENGTH_DEFAULT` shape how quickly
 *   adaptive controllers react,
 * - mode labels such as `COMPLEXITY_MODE_ADAPTIVE`, `PHASE_COMPLEXIFY`, and
 *   `MUTATION_STRATEGY_ANNEAL` give the helper files one shared vocabulary,
 * - clamp and multiplier values define the safe operating envelope for
 *   acceptance tuning, lineage pressure, and mutation adaptation.
 *
 * The goal is not to memorize every export. The goal is to see that the
 * adaptive subtree reuses one glossary for tiny math helpers, schedule timing,
 * mode names, and safety clamps instead of scattering unrelated literals across
 * each control loop.
 */

/** Zero baseline reused by tiny adaptive arithmetic helpers. */
export const ZERO = 0;
/** Unit baseline reused by clamp, ratio, and fallback calculations. */
export const ONE = 1;
/** Small divisor and offset used by split and normalization helpers. */
export const TWO = 2;
/** Small threshold used by helpers that need a minimal multi-sample floor. */
export const THREE = 3;
/** Small multiplier reused by adaptive-budget growth defaults. */
export const FOUR = 4;
/** Small count baseline reused by archive-size and cooldown defaults. */
export const FIVE = 5;
/** Round-number default reused by windows and phase lengths. */
export const TEN = 10;
/** Large round-number default for long-horizon scheduling. */
export const ONE_HUNDRED = 100;
/** Last-index sentinel reused when helpers need the final recorded item. */
export const NEGATIVE_ONE = -1;
/** Default score-history window for trend-aware complexity budgeting. */
export const DEFAULT_IMPROVEMENT_WINDOW = TEN;
/** Minimum history length to compute improvement. */
export const HISTORY_MIN_IMPROVEMENT_COUNT = TWO;
/** Minimum history length to compute slope. */
export const HISTORY_MIN_SLOPE_COUNT = THREE;
/** Default multiplier used when adaptive complexity budgeting detects improvement. */
export const DEFAULT_CB_INCREASE_FACTOR = 1.1;
/** Default multiplier used when adaptive complexity budgeting responds to stagnation. */
export const DEFAULT_CB_STAGNATION_FACTOR = 0.95;
/** Slope boost multiplier for adaptive increase factor. */
export const SLOPE_BOOST_MULTIPLIER = 0.05;
/** Slope penalty multiplier for stagnation factor. */
export const SLOPE_PENALTY_MULTIPLIER = 0.03;
/** Clamp magnitude for slope normalization. */
export const SLOPE_NORMALIZE_CLAMP = TWO;
/** Novelty archive minimum size. */
export const NOVELTY_ARCHIVE_MIN_SIZE = FIVE;
/** Novelty factor when archive is small. */
export const NOVELTY_FACTOR_SMALL = 0.9;
/** Novelty factor when archive is sufficient. */
export const NOVELTY_FACTOR_DEFAULT = ONE;
/** Offset added to input/output for minimal topology. */
export const MINIMAL_TOPOLOGY_OFFSET = TWO;
/** Default budget growth multiplier. */
export const BUDGET_GROWTH_MULTIPLIER = FOUR;
/** Default horizon for linear schedule. */
export const LINEAR_HORIZON_DEFAULT = ONE_HUNDRED;
/** Maximum progress ratio for scheduling. */
export const PROGRESS_RATIO_MAX = ONE;
/** Default phase length in generations for phased complexify/simplify schedules. */
export const PHASE_LENGTH_DEFAULT = TEN;
/** Default acceptance target for adaptive minimal-criterion control. */
export const TARGET_ACCEPTANCE_DEFAULT = 0.5;
/** Default adjustment step for adaptive minimal-criterion threshold updates. */
export const ADJUST_RATE_DEFAULT = 0.1;
/** Upper multiplier used when the controller nudges acceptance pressure upward. */
export const ACCEPTANCE_UPPER_MULTIPLIER = 1.05;
/** Lower multiplier used when the controller relaxes acceptance pressure. */
export const ACCEPTANCE_LOWER_MULTIPLIER = 0.95;
/** Fallback denominator to avoid divide-by-zero. */
export const DENOMINATOR_FALLBACK = ONE;
/** Default operator decay factor. */
export const OPERATOR_DECAY_DEFAULT = 0.9;
/** Mode label for feedback-driven complexity-budget scheduling. */
export const COMPLEXITY_MODE_ADAPTIVE = 'adaptive';
/** Mode label for pre-planned linear complexity-budget scheduling. */
export const COMPLEXITY_MODE_LINEAR = 'linear';
/** Phase label for the structure-growth side of phased complexity. */
export const PHASE_COMPLEXIFY = 'complexify';
/** Phase label for the structure-pruning side of phased complexity. */
export const PHASE_SIMPLIFY = 'simplify';
/** Mode label for epsilon-style ancestor-uniqueness feedback. */
export const ANCESTOR_UNIQ_MODE_EPSILON = 'epsilon';
/** Mode label for ancestor-uniqueness control via lineage-pressure tuning. */
export const ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE = 'lineagePressure';
/** Lineage pressure spread mode. */
export const LINEAGE_PRESSURE_MODE_SPREAD = 'spread';
/** Default cooldown (generations) for ancestor-uniqueness adjustments. */
export const DEFAULT_ANCESTOR_UNIQ_COOLDOWN = 5;
/** Default lower bound for acceptable ancestor uniqueness. */
export const DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD = 0.25;
/** Default upper bound for acceptable ancestor uniqueness. */
export const DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD = 0.55;
/** Default adjustment magnitude for uniqueness nudges. */
export const DEFAULT_ANCESTOR_UNIQ_ADJUST = 0.01;
/** Default lineage pressure strength when initializing the option. */
export const DEFAULT_LINEAGE_PRESSURE_STRENGTH = 0.01;
/** Multiplier when increasing lineage pressure strength. */
export const LINEAGE_PRESSURE_INCREASE_MULTIPLIER = 1.15;
/** Multiplier when decreasing lineage pressure strength. */
export const LINEAGE_PRESSURE_DECREASE_MULTIPLIER = 0.9;
/** Default cadence for refreshing per-genome adaptive mutation settings. */
export const DEFAULT_ADAPT_EVERY = 1;
/** Default perturbation spread for adaptive mutation-rate updates. */
export const DEFAULT_MUTATION_SIGMA = 0.05;
/** Scale applied to mutation sigma for perturbations. */
export const MUTATION_SIGMA_SCALE = 1.5;
/** Default lower clamp for per-genome adaptive mutation rates. */
export const DEFAULT_MIN_MUTATION_RATE = 0.01;
/** Default upper clamp for per-genome adaptive mutation rates. */
export const DEFAULT_MAX_MUTATION_RATE = 1;
/** Default initial mutation rate used before adaptive balancing specializes genomes. */
export const DEFAULT_INITIAL_MUTATION_RATE = 0.5;
/** Default mutation amount when genome value is missing. */
export const DEFAULT_MUTATION_AMOUNT = 1;
/** Default perturbation spread for adaptive mutation-amount updates. */
export const DEFAULT_MUTATION_AMOUNT_SIGMA = 0.25;
/** Default minimum mutation amount. */
export const DEFAULT_MIN_MUTATION_AMOUNT = 1;
/** Default maximum mutation amount. */
export const DEFAULT_MAX_MUTATION_AMOUNT = 10;
/** Random range multiplier for signed deltas. */
export const RNG_SPREAD_MULTIPLIER = 2;
/** Random offset for signed deltas. */
export const RNG_CENTER_OFFSET = 1;
/** Multiplicative boost for explore-low strategy (bottom half). */
export const EXPLORE_LOW_INCREASE_MULTIPLIER = 1.5;
/** Multiplicative decay for explore-low strategy (top half). */
export const EXPLORE_LOW_DECREASE_MULTIPLIER = 0.5;
/** Baseline generations for annealing progress. */
export const ANNEAL_BASELINE_GENERATIONS = 50;
/** Maximum progress ratio used in annealing. */
export const ANNEAL_PROGRESS_MAX = 1;
/** Divisor used to split populations in half. */
export const HALF_INDEX_DIVISOR = 2;
/** Strategy label for ranking-sensitive two-tier adaptive mutation. */
export const MUTATION_STRATEGY_TWO_TIER = 'twoTier';
/** Strategy label for boosting structural risk on the lower-ranked half. */
export const MUTATION_STRATEGY_EXPLORE_LOW = 'exploreLow';
/** Strategy label for annealed mutation pressure across run progress. */
export const MUTATION_STRATEGY_ANNEAL = 'anneal';
