/** Constant: zero value. */
export const ZERO = 0;
/** Constant: one value. */
export const ONE = 1;
/** Constant: two value. */
export const TWO = 2;
/** Constant: three value. */
export const THREE = 3;
/** Constant: four value. */
export const FOUR = 4;
/** Constant: five value. */
export const FIVE = 5;
/** Constant: ten value. */
export const TEN = 10;
/** Constant: one hundred value. */
export const ONE_HUNDRED = 100;
/** Constant: negative one for last index. */
export const NEGATIVE_ONE = -1;
/** Default score history window. */
export const DEFAULT_IMPROVEMENT_WINDOW = TEN;
/** Minimum history length to compute improvement. */
export const HISTORY_MIN_IMPROVEMENT_COUNT = TWO;
/** Minimum history length to compute slope. */
export const HISTORY_MIN_SLOPE_COUNT = THREE;
/** Default increase factor for adaptive schedule. */
export const DEFAULT_CB_INCREASE_FACTOR = 1.1;
/** Default stagnation factor for adaptive schedule. */
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
/** Default phase length in generations. */
export const PHASE_LENGTH_DEFAULT = TEN;
/** Default target acceptance in minimal criterion. */
export const TARGET_ACCEPTANCE_DEFAULT = 0.5;
/** Default adjustment rate in minimal criterion. */
export const ADJUST_RATE_DEFAULT = 0.1;
/** Upper acceptance multiplier. */
export const ACCEPTANCE_UPPER_MULTIPLIER = 1.05;
/** Lower acceptance multiplier. */
export const ACCEPTANCE_LOWER_MULTIPLIER = 0.95;
/** Fallback denominator to avoid divide-by-zero. */
export const DENOMINATOR_FALLBACK = ONE;
/** Default operator decay factor. */
export const OPERATOR_DECAY_DEFAULT = 0.9;
/** Complexity budget adaptive mode string. */
export const COMPLEXITY_MODE_ADAPTIVE = 'adaptive';
/** Complexity budget linear mode string. */
export const COMPLEXITY_MODE_LINEAR = 'linear';
/** Phase label for complexify. */
export const PHASE_COMPLEXIFY = 'complexify';
/** Phase label for simplify. */
export const PHASE_SIMPLIFY = 'simplify';
/** Ancestor uniqueness epsilon mode. */
export const ANCESTOR_UNIQ_MODE_EPSILON = 'epsilon';
/** Ancestor uniqueness lineage pressure mode. */
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
/** Default adapt-every cadence for adaptive mutation. */
export const DEFAULT_ADAPT_EVERY = 1;
/** Default mutation sigma for adaptive mutation. */
export const DEFAULT_MUTATION_SIGMA = 0.05;
/** Scale applied to mutation sigma for perturbations. */
export const MUTATION_SIGMA_SCALE = 1.5;
/** Default minimum per-genome mutation rate. */
export const DEFAULT_MIN_MUTATION_RATE = 0.01;
/** Default maximum per-genome mutation rate. */
export const DEFAULT_MAX_MUTATION_RATE = 1;
/** Default initial mutation rate used for balance checks. */
export const DEFAULT_INITIAL_MUTATION_RATE = 0.5;
/** Default mutation amount when genome value is missing. */
export const DEFAULT_MUTATION_AMOUNT = 1;
/** Default mutation amount sigma for perturbations. */
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
/** Strategy identifier for two-tier mutation. */
export const MUTATION_STRATEGY_TWO_TIER = 'twoTier';
/** Strategy identifier for explore-low mutation. */
export const MUTATION_STRATEGY_EXPLORE_LOW = 'exploreLow';
/** Strategy identifier for annealed mutation. */
export const MUTATION_STRATEGY_ANNEAL = 'anneal';
