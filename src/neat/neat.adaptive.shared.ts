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

/**
 * Minimal interface for NEAT instances with adaptive features.
 * Exported for use in tests and type-safe function calls.
 */
export interface NeatLikeWithAdaptive {
  /** Adaptive and scheduling configuration options for NEAT behaviors. */
  options: {
    /** Complexity-budget scheduling configuration. */
    complexityBudget?: {
      /** Enables complexity budget scheduling. */
      enabled?: boolean;
      /** Scheduling mode (for example: 'adaptive' or 'linear'). */
      mode?: string;
      /** Window size used to measure improvement trends. */
      improvementWindow?: number;
      /** Multiplicative increase factor applied on improvement. */
      increaseFactor?: number;
      /** Multiplicative decay factor applied on stagnation. */
      stagnationFactor?: number;
      /** Starting node cap for linear/initial budgeting. */
      maxNodesStart?: number;
      /** Ending node cap for linear/maximum budgeting. */
      maxNodesEnd?: number;
      /** Minimum node cap allowed during budgeting. */
      minNodes?: number;
      /** Starting connection cap for linear/initial budgeting. */
      maxConnsStart?: number;
      /** Ending connection cap for linear/maximum budgeting. */
      maxConnsEnd?: number;
      /** Horizon (in generations) for linear schedules. */
      horizon?: number;
    };
    /** Phased complexity configuration (complexify/simplify cycles). */
    phasedComplexity?: {
      /** Enables phased complexity scheduling. */
      enabled?: boolean;
      /** Explicit phase definitions with generation and caps. */
      phases?: Array<{
        /** Generation at which the phase begins. */
        generation: number;
        /** Node cap to apply during the phase. */
        maxNodes?: number;
        /** Connection cap to apply during the phase. */
        maxConns?: number;
      }>;
      /** Default duration (in generations) for a phase. */
      phaseLength?: number;
      /** Initial phase name (for example: 'complexify'). */
      initialPhase?: string;
    };
    /** Minimal-criterion configuration for acceptance thresholds. */
    minimalCriterion?: {
      /** Enables minimal-criterion gating. */
      enabled?: boolean;
      /** Strategy mode for minimal criterion. */
      mode?: string;
      /** Target complexity threshold for acceptance. */
      targetComplexity?: number;
      /** Learning rate for threshold adjustments. */
      learningRate?: number;
    };
    /** Adaptive minimal-criterion configuration. */
    minimalCriterionAdaptive?: {
      /** Enables adaptive minimal-criterion thresholding. */
      enabled?: boolean;
      /** Initial threshold for acceptance. */
      initialThreshold?: number;
      /** Target acceptance proportion in the population. */
      targetAcceptance?: number;
      /** Adjustment rate for threshold updates. */
      adjustRate?: number;
    };
    /** Ancestor-uniqueness configuration (static). */
    ancestorUniqueness?: {
      /** Enables ancestor-uniqueness tracking. */
      enabled?: boolean;
      /** Target ancestor-uniqueness rate. */
      targetRate?: number;
      /** Learning rate for uniqueness adjustments. */
      learningRate?: number;
    };
    /** Adaptive ancestor-uniqueness configuration. */
    ancestorUniqAdaptive?: {
      /** Enables adaptive ancestor-uniqueness adjustments. */
      enabled?: boolean;
      /** Cooldown (in generations) between adjustments. */
      cooldown?: number;
      /** Lower bound triggering an increase in diversity pressure. */
      lowThreshold?: number;
      /** Upper bound triggering a decrease in diversity pressure. */
      highThreshold?: number;
      /** Adjustment magnitude applied when thresholds are crossed. */
      adjust?: number;
      /** Strategy mode (for example: 'epsilon' or 'lineagePressure'). */
      mode?: string;
    };
    /** Self-adaptive mutation configuration. */
    adaptiveMutation?: {
      /** Enables per-genome mutation adaptation. */
      enabled?: boolean;
      /** Learning rate for mutation adaptation. */
      learningRate?: number;
      /** Minimum mutation rate allowed. */
      min?: number;
      /** Maximum mutation rate allowed. */
      max?: number;
      /** Adapt mutation parameters every N generations. */
      adaptEvery?: number;
      /** Standard deviation for mutation perturbations. */
      sigma?: number;
      /** Minimum per-genome mutation rate (clamp). */
      minRate?: number;
      /** Maximum per-genome mutation rate (clamp). */
      maxRate?: number;
      /** Adaptation strategy identifier. */
      strategy?: string;
      /** Enables mutation amount adaptation. */
      adaptAmount?: boolean;
      /** Minimum mutation amount allowed. */
      minAmount?: number;
      /** Maximum mutation amount allowed. */
      maxAmount?: number;
      /** Initial mutation rate used as a baseline. */
      initialRate?: number;
      /** Standard deviation for mutation-amount perturbations. */
      amountSigma?: number;
    };
    /** Adaptive operator-selection configuration. */
    operatorAdaptation?: {
      /** Enables operator adaptation. */
      enabled?: boolean;
      /** Learning rate for operator statistics updates. */
      learningRate?: number;
      /** Alpha parameter for operator adaptation. */
      alpha?: number;
      /** Decay factor for operator success/attempt statistics. */
      decay?: number;
    };
    /** Multi-objective configuration options. */
    multiObjective?: {
      /** Adaptive epsilon configuration for multi-objective selection. */
      adaptiveEpsilon?: {
        /** Enables adaptive epsilon. */
        enabled?: boolean;
      };
      /** Dominance epsilon for Pareto comparisons. */
      dominanceEpsilon?: number;
    };
    /** Lineage pressure configuration. */
    lineagePressure?: {
      /** Enables lineage pressure. */
      enabled?: boolean;
      /** Lineage pressure mode. */
      mode?: string;
      /** Lineage pressure strength value. */
      strength?: number;
    };
    /** Global maximum node cap. */
    maxNodes?: number;
    /** Global maximum connection cap. */
    maxConns?: number;
    /** Base mutation rate. */
    mutationRate?: number;
    /** Base mutation amount. */
    mutationAmount?: number;
  };
  /** Current population of genomes. */
  population: Array<{
    /** Fitness score for the genome. */
    score?: number;
    /** Per-genome mutation rate override. */
    _mutRate?: number | null;
    /** Per-genome mutation amount override. */
    _mutAmount?: number | null;
    /** Additional dynamic genome fields. */
    [key: string]: unknown;
  }>;
  /** Number of input nodes in the network. */
  input: number;
  /** Number of output nodes in the network. */
  output: number;
  /** Current generation index. */
  generation: number;
  /** Rolling history of best scores for complexity budgeting. */
  _cbHistory?: number[];
  /** Current complexity budget for nodes. */
  _cbMaxNodes?: number;
  /** Current complexity budget for connections. */
  _cbMaxConns?: number;
  /** Novelty archive used by some adaptive strategies. */
  _noveltyArchive?: unknown[];
  /** Minimal criterion baseline value. */
  _mcBaseline?: number;
  /** Minimal criterion current level. */
  _mcLevel?: number;
  /** Last observed inbreeding count. */
  _lastInbreedingCount?: number;
  /** Inbreeding penalty applied to selection. */
  _inbreedingPenalty?: number;
  /** Adaptive mutation rate computed by strategy. */
  _mutationRateAdaptive?: number;
  /** Operator success/attempt statistics by operator id. */
  _operatorStats?: Map<string, { success: number; attempts: number }>;
  /** Current phase name for phased complexity. */
  _phase?: string;
  /** Generation when the current phase started. */
  _phaseStartGeneration?: number;
  /** Minimal criterion acceptance threshold. */
  _mcThreshold?: number;
  /** Generation of the last ancestor-uniqueness adjustment. */
  _lastAncestorUniqAdjustGen?: number;
  /** Telemetry history (including lineage metrics). */
  _telemetry?: Array<{ lineage?: { ancestorUniq?: number } }>;
  /** RNG provider for deterministic randomness. */
  _getRNG?: () => () => number;
}

export type ComplexityBudgetConfig = NonNullable<
  NeatLikeWithAdaptive['options']['complexityBudget']
>;
export type PhasedComplexityConfig = NonNullable<
  NeatLikeWithAdaptive['options']['phasedComplexity']
>;
export type MinimalCriterionAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;
export type AncestorUniqAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['ancestorUniqAdaptive']
>;
export type AdaptiveMutationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['adaptiveMutation']
>;
export type OperatorAdaptationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['operatorAdaptation']
>;

export type MutationSettings = {
  strategy: string;
  sigmaBase: number;
  minRate: number;
  maxRate: number;
  initialRate: number;
  adaptAmount: boolean;
  amountSigma: number;
  minAmount: number;
  maxAmount: number;
  mutationAmountDefault: number;
  generation: number;
  populationSize: number;
};

export type MutationOutcome = { hasIncrease: boolean; hasDecrease: boolean };
export type MutationPartitions = { topHalf: Genome[]; bottomHalf: Genome[] };
export type Genome = NeatLikeWithAdaptive['population'][number];
