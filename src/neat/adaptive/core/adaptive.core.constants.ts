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
/** Unit baseline reused by clamp, ratio, fallback, and normalisation calculations. */
export const ONE = 1;
/** Small divisor and offset used by split and normalization helpers. */
export const TWO = 2;
/** Small threshold used by helpers that need a minimal multi-sample floor. */
export const THREE = 3;
/** Small numeric constant reused by adaptive-budget growth defaults and initialisation helpers. */
export const FOUR = 4;
/** Small count baseline reused by archive-size, cooldown, and neighbor-count defaults. */
export const FIVE = 5;
/** Round-number constant reused by improvement windows and default phase lengths. */
export const TEN = 10;
/** Large round-number constant used as a default for long-horizon adaptive scheduling. */
export const ONE_HUNDRED = 100;
/** Last-index sentinel reused when helpers need the final recorded item. */
export const NEGATIVE_ONE = -1;
/** Default score-history window length used by trend-aware adaptive complexity budgeting. */
export const DEFAULT_IMPROVEMENT_WINDOW = TEN;
/** Minimum score-history length required to compute an improvement signal reliably. */
export const HISTORY_MIN_IMPROVEMENT_COUNT = TWO;
/** Minimum score-history length required to compute a trend slope value. */
export const HISTORY_MIN_SLOPE_COUNT = THREE;
/** Default multiplier applied when adaptive complexity budgeting detects a fitness improvement. */
export const DEFAULT_CB_INCREASE_FACTOR = 1.1;
/** Default multiplier used when adaptive complexity budgeting responds to stagnation. */
export const DEFAULT_CB_STAGNATION_FACTOR = 0.95;
/** Slope boost multiplier applied to the adaptive increase factor under improving fitness trends. */
export const SLOPE_BOOST_MULTIPLIER = 0.05;
/** Slope penalty multiplier applied to the stagnation factor under declining fitness trends. */
export const SLOPE_PENALTY_MULTIPLIER = 0.03;
/** Clamp magnitude applied when normalizing fitness trend slopes to bounded values. */
export const SLOPE_NORMALIZE_CLAMP = TWO;
/** Minimum number of entries required in the novelty archive for reliable scoring. */
export const NOVELTY_ARCHIVE_MIN_SIZE = FIVE;
/** Novelty scaling factor applied when the archive holds too few neighbor candidates. */
export const NOVELTY_FACTOR_SMALL = 0.9;
/** Novelty scaling factor applied when the archive holds enough neighbor candidates. */
export const NOVELTY_FACTOR_DEFAULT = ONE;
/** Offset added to input and output width for minimal feed-forward topology sizing. */
export const MINIMAL_TOPOLOGY_OFFSET = TWO;
/** Default complexity budget growth multiplier applied when the controller detects sustained fitness improvement. */
export const BUDGET_GROWTH_MULTIPLIER = FOUR;
/** Default generation horizon used by the pre-planned linear complexity-budget schedule. */
export const LINEAR_HORIZON_DEFAULT = ONE_HUNDRED;
/** Maximum progress ratio clamp for linear and annealing complexity schedule calculations. */
export const PROGRESS_RATIO_MAX = ONE;
/** Default phase length in generations for phased complexify and simplify schedule cycling. */
export const PHASE_LENGTH_DEFAULT = TEN;
/** Default acceptance probability target used by adaptive minimal-criterion acceptance control. */
export const TARGET_ACCEPTANCE_DEFAULT = 0.5;
/** Default adjustment step size for adaptive minimal-criterion acceptance threshold updates. */
export const ADJUST_RATE_DEFAULT = 0.1;
/** Upper multiplier used when the controller nudges acceptance pressure upward. */
export const ACCEPTANCE_UPPER_MULTIPLIER = 1.05;
/** Lower multiplier applied when the adaptive controller relaxes acceptance threshold pressure. */
export const ACCEPTANCE_LOWER_MULTIPLIER = 0.95;
/** Fallback denominator substituted when the real denominator is zero or missing. */
export const DENOMINATOR_FALLBACK = ONE;
/** Default decay factor applied per generation to unused adaptive mutation operator weights. */
export const OPERATOR_DECAY_DEFAULT = 0.9;
/** Mode label for feedback-driven adaptive complexity-budget scheduling with slope detection. */
export const COMPLEXITY_MODE_ADAPTIVE = 'adaptive';
/** Mode label for pre-planned linear complexity-budget scheduling with a fixed generation horizon. */
export const COMPLEXITY_MODE_LINEAR = 'linear';
/** Phase label for the structure-growth side of phased complexity scheduling. */
export const PHASE_COMPLEXIFY = 'complexify';
/** Phase label for the structure-pruning side of phased complexity scheduling. */
export const PHASE_SIMPLIFY = 'simplify';
/** Mode label for epsilon-style ancestor-uniqueness feedback, adjusting the pressure threshold directly. */
export const ANCESTOR_UNIQ_MODE_EPSILON = 'epsilon';
/** Mode label for ancestor-uniqueness control achieved through lineage-pressure strength tuning. */
export const ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE = 'lineagePressure';
/** Lineage pressure spread mode label for population-wide diversity pressure distribution. */
export const LINEAGE_PRESSURE_MODE_SPREAD = 'spread';
/** Default generation cooldown between consecutive ancestor-uniqueness acceptance threshold adjustment steps. */
export const DEFAULT_ANCESTOR_UNIQ_COOLDOWN = 5;
/** Default lower bound for acceptable ancestor uniqueness, below which pressure is increased. */
export const DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD = 0.25;
/** Default upper bound for acceptable ancestor uniqueness, above which pressure is relaxed. */
export const DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD = 0.55;
/** Default adjustment magnitude applied when nudging ancestor-uniqueness pressure thresholds upward or downward. */
export const DEFAULT_ANCESTOR_UNIQ_ADJUST = 0.01;
/** Default lineage pressure strength applied when initializing the ancestor-uniqueness option. */
export const DEFAULT_LINEAGE_PRESSURE_STRENGTH = 0.01;
/** Multiplier applied when the controller increases lineage pressure strength to drive diversity. */
export const LINEAGE_PRESSURE_INCREASE_MULTIPLIER = 1.15;
/** Multiplier applied when the controller decreases lineage pressure strength after over-pressure. */
export const LINEAGE_PRESSURE_DECREASE_MULTIPLIER = 0.9;
/** Default generation cadence for refreshing per-genome adaptive mutation rate settings. */
export const DEFAULT_ADAPT_EVERY = 1;
/** Default Gaussian perturbation spread used for adaptive mutation-rate self-adaptation rate evolution. */
export const DEFAULT_MUTATION_SIGMA = 0.05;
/** Scale factor applied to mutation sigma when computing self-adaptation perturbation bounds. */
export const MUTATION_SIGMA_SCALE = 1.5;
/** Default lower clamp applied to per-genome adaptive mutation rate evolution. */
export const DEFAULT_MIN_MUTATION_RATE = 0.01;
/** Default upper clamp applied to per-genome adaptive mutation rate evolution. */
export const DEFAULT_MAX_MUTATION_RATE = 1;
/** Default initial mutation rate used before adaptive balancing specializes genomes. */
export const DEFAULT_INITIAL_MUTATION_RATE = 0.5;
/** Default mutation amount applied when a genome's target value is missing or not set. */
export const DEFAULT_MUTATION_AMOUNT = 1;
/** Default Gaussian perturbation spread used for adaptive mutation-amount self-adaptation rate evolution. */
export const DEFAULT_MUTATION_AMOUNT_SIGMA = 0.25;
/** Default lower bound for the per-genome adaptive mutation amount value. */
export const DEFAULT_MIN_MUTATION_AMOUNT = 1;
/** Default upper bound for the per-genome adaptive mutation amount value. */
export const DEFAULT_MAX_MUTATION_AMOUNT = 10;
/** Random range multiplier used when computing signed self-adaptation perturbation deltas. */
export const RNG_SPREAD_MULTIPLIER = 2;
/** Random center offset subtracted when mapping uniform samples to signed perturbation deltas. */
export const RNG_CENTER_OFFSET = 1;
/** Multiplicative boost applied to bottom-half genomes under the explore-low adaptive strategy. */
export const EXPLORE_LOW_INCREASE_MULTIPLIER = 1.5;
/** Multiplicative decay applied to top-half genomes under the explore-low adaptive strategy. */
export const EXPLORE_LOW_DECREASE_MULTIPLIER = 0.5;
/** Baseline generation count used when computing annealing schedule progress towards maximum. */
export const ANNEAL_BASELINE_GENERATIONS = 50;
/** Maximum progress ratio clamp applied in annealing schedule pressure calculations. */
export const ANNEAL_PROGRESS_MAX = 1;
/** Divisor applied when splitting ranked populations into top and bottom halves. */
export const HALF_INDEX_DIVISOR = 2;
/** Strategy label for ranking-sensitive two-tier adaptive mutation with separate top and bottom behavior. */
export const MUTATION_STRATEGY_TWO_TIER = 'twoTier';
/** Strategy label for boosting structural risk on the lower-ranked population half. */
export const MUTATION_STRATEGY_EXPLORE_LOW = 'exploreLow';
/** Strategy label for annealed mutation pressure that decreases across run progress. */
export const MUTATION_STRATEGY_ANNEAL = 'anneal';
