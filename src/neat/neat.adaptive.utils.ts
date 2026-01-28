import { EPSILON } from './neat.constants';

/** Constant: zero value. */
const ZERO = 0;
/** Constant: one value. */
const ONE = 1;
/** Constant: two value. */
const TWO = 2;
/** Constant: three value. */
const THREE = 3;
/** Constant: four value. */
const FOUR = 4;
/** Constant: five value. */
const FIVE = 5;
/** Constant: ten value. */
const TEN = 10;
/** Constant: one hundred value. */
const ONE_HUNDRED = 100;
/** Constant: negative one for last index. */
const NEGATIVE_ONE = -1;
/** Default score history window. */
const DEFAULT_IMPROVEMENT_WINDOW = TEN;
/** Minimum history length to compute improvement. */
const HISTORY_MIN_IMPROVEMENT_COUNT = TWO;
/** Minimum history length to compute slope. */
const HISTORY_MIN_SLOPE_COUNT = THREE;
/** Default increase factor for adaptive schedule. */
const DEFAULT_CB_INCREASE_FACTOR = 1.1;
/** Default stagnation factor for adaptive schedule. */
const DEFAULT_CB_STAGNATION_FACTOR = 0.95;
/** Slope boost multiplier for adaptive increase factor. */
const SLOPE_BOOST_MULTIPLIER = 0.05;
/** Slope penalty multiplier for stagnation factor. */
const SLOPE_PENALTY_MULTIPLIER = 0.03;
/** Clamp magnitude for slope normalization. */
const SLOPE_NORMALIZE_CLAMP = TWO;
/** Novelty archive minimum size. */
const NOVELTY_ARCHIVE_MIN_SIZE = FIVE;
/** Novelty factor when archive is small. */
const NOVELTY_FACTOR_SMALL = 0.9;
/** Novelty factor when archive is sufficient. */
const NOVELTY_FACTOR_DEFAULT = ONE;
/** Offset added to input/output for minimal topology. */
const MINIMAL_TOPOLOGY_OFFSET = TWO;
/** Default budget growth multiplier. */
const BUDGET_GROWTH_MULTIPLIER = FOUR;
/** Default horizon for linear schedule. */
const LINEAR_HORIZON_DEFAULT = ONE_HUNDRED;
/** Maximum progress ratio for scheduling. */
const PROGRESS_RATIO_MAX = ONE;
/** Default phase length in generations. */
const PHASE_LENGTH_DEFAULT = TEN;
/** Default target acceptance in minimal criterion. */
const TARGET_ACCEPTANCE_DEFAULT = 0.5;
/** Default adjustment rate in minimal criterion. */
const ADJUST_RATE_DEFAULT = 0.1;
/** Upper acceptance multiplier. */
const ACCEPTANCE_UPPER_MULTIPLIER = 1.05;
/** Lower acceptance multiplier. */
const ACCEPTANCE_LOWER_MULTIPLIER = 0.95;
/** Fallback denominator to avoid divide-by-zero. */
const DENOMINATOR_FALLBACK = ONE;
/** Default operator decay factor. */
const OPERATOR_DECAY_DEFAULT = 0.9;
/** Complexity budget adaptive mode string. */
const COMPLEXITY_MODE_ADAPTIVE = 'adaptive';
/** Complexity budget linear mode string. */
const COMPLEXITY_MODE_LINEAR = 'linear';
/** Phase label for complexify. */
const PHASE_COMPLEXIFY = 'complexify';
/** Phase label for simplify. */
const PHASE_SIMPLIFY = 'simplify';
/** Ancestor uniqueness epsilon mode. */
const ANCESTOR_UNIQ_MODE_EPSILON = 'epsilon';
/** Ancestor uniqueness lineage pressure mode. */
const ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE = 'lineagePressure';
/** Lineage pressure spread mode. */
const LINEAGE_PRESSURE_MODE_SPREAD = 'spread';

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

type ComplexityBudgetConfig = NonNullable<
  NeatLikeWithAdaptive['options']['complexityBudget']
>;
type PhasedComplexityConfig = NonNullable<
  NeatLikeWithAdaptive['options']['phasedComplexity']
>;
type MinimalCriterionAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['minimalCriterionAdaptive']
>;
type AncestorUniqAdaptiveConfig = NonNullable<
  NeatLikeWithAdaptive['options']['ancestorUniqAdaptive']
>;
type AdaptiveMutationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['adaptiveMutation']
>;
type OperatorAdaptationConfig = NonNullable<
  NeatLikeWithAdaptive['options']['operatorAdaptation']
>;

type MutationSettings = {
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

type MutationOutcome = { hasIncrease: boolean; hasDecrease: boolean };
type MutationPartitions = { topHalf: Genome[]; bottomHalf: Genome[] };
type Genome = NeatLikeWithAdaptive['population'][number];

/**
 * Apply the complexity budget schedule for the configured mode.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function applyComplexityBudgetSchedule(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  if (config.mode === COMPLEXITY_MODE_ADAPTIVE) {
    applyAdaptiveSchedule(engine, config);
    return;
  }

  applyLinearSchedule(engine, config);
}

/**
 * Apply adaptive complexity budget scheduling.
 *
 * @param engine - NEAT engine instance with adaptive state.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function applyAdaptiveSchedule(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  const history = updateScoreHistory(engine, config);
  const trends = computeTrends(history);
  const factors = computeAdjustmentFactors(config, trends, history);
  const noveltyFactor = computeNoveltyFactor(engine);

  initializeNodeBudget(engine, config);
  adjustNodeBudget(engine, config, trends, factors, noveltyFactor, history);
  clampNodeBudget(engine, config);
  engine.options.maxNodes = engine._cbMaxNodes;

  if (config.maxConnsStart) {
    initializeConnectionBudget(engine, config);
    adjustConnectionBudget(
      engine,
      config,
      trends,
      factors,
      noveltyFactor,
      history,
    );
    engine.options.maxConns = engine._cbMaxConns;
  }
}

/**
 * Apply linear complexity budget scheduling.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function applyLinearSchedule(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  const minimalTopology =
    engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
  const startBudget = config.maxNodesStart ?? minimalTopology;
  const endBudget =
    config.maxNodesEnd ?? startBudget * BUDGET_GROWTH_MULTIPLIER;
  const horizonGens = config.horizon ?? LINEAR_HORIZON_DEFAULT;
  const progress = Math.min(
    PROGRESS_RATIO_MAX,
    engine.generation / horizonGens,
  );

  engine.options.maxNodes = Math.floor(
    startBudget + (endBudget - startBudget) * progress,
  );
}

/**
 * Update rolling score history with current best score.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns Rolling history array after update.
 */
export function updateScoreHistory(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): number[] {
  if (!engine._cbHistory) engine._cbHistory = [];

  const currentBestScore = engine.population[ZERO]?.score ?? ZERO;
  engine._cbHistory.push(currentBestScore);

  const windowSize = config.improvementWindow ?? DEFAULT_IMPROVEMENT_WINDOW;
  if (engine._cbHistory.length > windowSize) {
    engine._cbHistory.shift();
  }

  return engine._cbHistory;
}

/**
 * Compute improvement and slope trends from score history.
 *
 * @param history - Rolling history of best scores.
 * @returns Trend metrics (improvement and slope).
 */
export function computeTrends(history: number[]): {
  improvement: number;
  slope: number;
} {
  const improvement =
    history.length >= HISTORY_MIN_IMPROVEMENT_COUNT
      ? (history.at(NEGATIVE_ONE) ?? ZERO) - history[ZERO]
      : ZERO;
  const slope =
    history.length >= HISTORY_MIN_SLOPE_COUNT ? computeSlope(history) : ZERO;
  return { improvement, slope };
}

/**
 * Compute linear regression slope using ordinary least squares.
 *
 * @param history - Rolling history of best scores.
 * @returns OLS slope estimate.
 */
export function computeSlope(history: number[]): number {
  const count = history.length;
  let sumIndices = ZERO;
  let sumScores = ZERO;
  let sumIndexScore = ZERO;
  let sumIndexSquared = ZERO;

  for (let index = ZERO; index < count; index++) {
    sumIndices += index;
    sumScores += history[index];
    sumIndexScore += index * history[index];
    sumIndexSquared += index * index;
  }

  const denominator = count * sumIndexSquared - sumIndices * sumIndices;
  const safeDenominator = denominator || DENOMINATOR_FALLBACK;
  return (count * sumIndexScore - sumIndices * sumScores) / safeDenominator;
}

/**
 * Compute adjustment factors for budget growth and decay.
 *
 * @param config - Complexity budget configuration.
 * @param trends - Improvement and slope metrics.
 * @param history - Rolling history of best scores.
 * @returns Adjustment factors (increase and stagnation multipliers).
 */
export function computeAdjustmentFactors(
  config: ComplexityBudgetConfig,
  trends: { improvement: number; slope: number },
  history: number[],
): { increaseFactor: number; stagnationFactor: number } {
  const baseIncrease = config.increaseFactor ?? DEFAULT_CB_INCREASE_FACTOR;
  const baseStagnation =
    config.stagnationFactor ?? DEFAULT_CB_STAGNATION_FACTOR;
  const normalizedSlope = normalizeSlope(trends.slope, history[ZERO]);

  const increaseFactor =
    baseIncrease + SLOPE_BOOST_MULTIPLIER * Math.max(ZERO, normalizedSlope);
  const stagnationFactor =
    baseStagnation -
    SLOPE_PENALTY_MULTIPLIER * Math.max(ZERO, -normalizedSlope);

  return { increaseFactor, stagnationFactor };
}

/**
 * Normalize slope magnitude relative to initial score.
 *
 * @param slope - Raw OLS slope.
 * @param initialScore - First score in history window.
 * @returns Normalized slope clamped to [-2, 2].
 */
export function normalizeSlope(slope: number, initialScore: number): number {
  return Math.min(
    SLOPE_NORMALIZE_CLAMP,
    Math.max(
      -SLOPE_NORMALIZE_CLAMP,
      slope / (Math.abs(initialScore) + EPSILON),
    ),
  );
}

/**
 * Compute novelty factor based on archive size.
 *
 * @param engine - NEAT engine instance.
 * @returns Novelty multiplier (0.9 if archive small, 1.0 otherwise).
 */
export function computeNoveltyFactor(engine: NeatLikeWithAdaptive): number {
  return (engine._noveltyArchive?.length ?? ZERO) > NOVELTY_ARCHIVE_MIN_SIZE
    ? NOVELTY_FACTOR_DEFAULT
    : NOVELTY_FACTOR_SMALL;
}

/**
 * Initialize node budget if undefined.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function initializeNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  if (engine._cbMaxNodes === undefined) {
    const minimalTopology =
      engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
    engine._cbMaxNodes = config.maxNodesStart ?? minimalTopology;
  }
}

/**
 * Adjust node budget based on trends and factors.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @param trends - Improvement and slope metrics.
 * @param factors - Adjustment factors.
 * @param noveltyFactor - Novelty multiplier.
 * @param history - Rolling history for window checks.
 * @returns {void}
 */
export function adjustNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
  trends: { improvement: number; slope: number },
  factors: { increaseFactor: number; stagnationFactor: number },
  noveltyFactor: number,
  history: number[],
): void {
  const windowSize = config.improvementWindow ?? DEFAULT_IMPROVEMENT_WINDOW;
  const isImproving = trends.improvement > ZERO || trends.slope > ZERO;
  const isWindowFull = history.length === windowSize;

  if (isImproving) {
    const maxCap =
      config.maxNodesEnd ?? engine._cbMaxNodes! * BUDGET_GROWTH_MULTIPLIER;
    const proposed = Math.floor(
      engine._cbMaxNodes! * factors.increaseFactor * noveltyFactor,
    );
    engine._cbMaxNodes = Math.min(maxCap, proposed);
  } else if (isWindowFull) {
    const minimalTopology =
      engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
    const minCap = config.minNodes ?? minimalTopology;
    const proposed = Math.floor(engine._cbMaxNodes! * factors.stagnationFactor);
    engine._cbMaxNodes = Math.max(minCap, proposed);
  }
}

/**
 * Clamp node budget to configured minimum.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function clampNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  const minimalTopology =
    engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
  const minAllowed = config.minNodes ?? minimalTopology;
  engine._cbMaxNodes = Math.max(minAllowed, engine._cbMaxNodes!);
}

/**
 * Initialize connection budget if undefined.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function initializeConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  if (engine._cbMaxConns === undefined) {
    engine._cbMaxConns = config.maxConnsStart!;
  }
}

/**
 * Adjust connection budget based on trends and factors.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @param trends - Improvement and slope metrics.
 * @param factors - Adjustment factors.
 * @param noveltyFactor - Novelty multiplier.
 * @param history - Rolling history for window checks.
 * @returns {void}
 */
export function adjustConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
  trends: { improvement: number; slope: number },
  factors: { increaseFactor: number; stagnationFactor: number },
  noveltyFactor: number,
  history: number[],
): void {
  const windowSize = config.improvementWindow ?? DEFAULT_IMPROVEMENT_WINDOW;
  const isImproving = trends.improvement > ZERO || trends.slope > ZERO;
  const isWindowFull = history.length === windowSize;

  if (isImproving) {
    const maxCap =
      config.maxConnsEnd ?? engine._cbMaxConns! * BUDGET_GROWTH_MULTIPLIER;
    const proposed = Math.floor(
      engine._cbMaxConns! * factors.increaseFactor * noveltyFactor,
    );
    engine._cbMaxConns = Math.min(maxCap, proposed);
  } else if (isWindowFull) {
    const minCap = config.maxConnsStart!;
    const proposed = Math.floor(engine._cbMaxConns! * factors.stagnationFactor);
    engine._cbMaxConns = Math.max(minCap, proposed);
  }
}

/**
 * Ensure phase state is initialized.
 *
 * @param engine - NEAT engine instance.
 * @param config - Phased complexity configuration.
 * @returns {void}
 */
export function initializePhaseState(
  engine: NeatLikeWithAdaptive,
  config: PhasedComplexityConfig,
): void {
  if (engine._phase) return;

  // Step 1: Select initial phase and record start generation.
  const initialPhase = config.initialPhase ?? PHASE_COMPLEXIFY;
  engine._phase = initialPhase;
  engine._phaseStartGeneration = engine.generation;
}

/**
 * Toggle phase if the current phase has exceeded its length.
 *
 * @param engine - NEAT engine instance.
 * @param config - Phased complexity configuration.
 * @returns {void}
 */
export function togglePhaseIfNeeded(
  engine: NeatLikeWithAdaptive,
  config: PhasedComplexityConfig,
): void {
  const phaseLength = config.phaseLength ?? PHASE_LENGTH_DEFAULT;
  const elapsed = engine.generation - (engine._phaseStartGeneration ?? ZERO);
  if (elapsed < phaseLength) return;

  // Step 1: Toggle phase.
  engine._phase = resolveNextPhase(engine._phase ?? PHASE_COMPLEXIFY);
  // Step 2: Reset phase start generation.
  engine._phaseStartGeneration = engine.generation;
}

/**
 * Resolve next phase name.
 *
 * @param currentPhase - Current phase label.
 * @returns Next phase label.
 */
export function resolveNextPhase(currentPhase: string): string {
  return currentPhase === PHASE_COMPLEXIFY ? PHASE_SIMPLIFY : PHASE_COMPLEXIFY;
}

/**
 * Initialize MC threshold if missing.
 *
 * @param engine - NEAT engine instance.
 * @param config - Minimal-criterion adaptive configuration.
 * @returns {void}
 */
export function initializeThreshold(
  engine: NeatLikeWithAdaptive,
  config: MinimalCriterionAdaptiveConfig,
): void {
  if (engine._mcThreshold !== undefined) return;

  const initialThreshold = config.initialThreshold ?? ZERO;
  engine._mcThreshold = initialThreshold;
}

/**
 * Collect population scores into a snapshot array.
 *
 * @param engine - NEAT engine instance.
 * @returns Array of scores (missing scores treated as 0).
 */
export function collectScores(engine: NeatLikeWithAdaptive): number[] {
  return engine.population.map((genome) => genome.score ?? ZERO);
}

/**
 * Compute acceptance metrics for the current threshold.
 *
 * @param scores - Population score snapshot.
 * @param threshold - Current MC threshold.
 * @returns Acceptance proportion.
 */
export function computeAcceptance(scores: number[], threshold: number): number {
  if (!scores.length) return ZERO;

  const acceptedCount = scores.filter((score) => score >= threshold).length;
  return acceptedCount / scores.length;
}

/**
 * Resolve target acceptance and adjust rate settings.
 *
 * @param config - Minimal-criterion adaptive configuration.
 * @returns Target settings.
 */
export function resolveTargetSettings(config: MinimalCriterionAdaptiveConfig): {
  targetAcceptance: number;
  adjustRate: number;
} {
  const targetAcceptance = config.targetAcceptance ?? TARGET_ACCEPTANCE_DEFAULT;
  const adjustRate = config.adjustRate ?? ADJUST_RATE_DEFAULT;
  return { targetAcceptance, adjustRate };
}

/**
 * Update the MC threshold based on acceptance proportion.
 *
 * @param engine - NEAT engine instance.
 * @param acceptance - Observed acceptance proportion.
 * @param tuning - Target acceptance and adjustment settings.
 * @returns {void}
 */
export function updateThreshold(
  engine: NeatLikeWithAdaptive,
  acceptance: number,
  tuning: { targetAcceptance: number; adjustRate: number },
): void {
  const upperBound = tuning.targetAcceptance * ACCEPTANCE_UPPER_MULTIPLIER;
  const lowerBound = tuning.targetAcceptance * ACCEPTANCE_LOWER_MULTIPLIER;

  if (acceptance > upperBound) {
    engine._mcThreshold =
      (engine._mcThreshold ?? ZERO) * (ONE + tuning.adjustRate);
    return;
  }

  if (acceptance < lowerBound) {
    engine._mcThreshold =
      (engine._mcThreshold ?? ZERO) * (ONE - tuning.adjustRate);
  }
}

/**
 * Zero scores below the final threshold.
 *
 * @param engine - NEAT engine instance.
 * @param threshold - Final MC threshold.
 * @returns {void}
 */
export function applyRejection(
  engine: NeatLikeWithAdaptive,
  threshold: number,
): void {
  for (const genome of engine.population) {
    if ((genome.score ?? ZERO) < threshold) genome.score = ZERO;
  }
}

/**
 * Determine whether the cooldown window has elapsed.
 *
 * @param engine - NEAT engine instance.
 * @param config - Ancestor uniqueness adaptive configuration.
 * @returns True when adjustment is allowed.
 */
export function isCooldownSatisfied(
  engine: NeatLikeWithAdaptive,
  config: AncestorUniqAdaptiveConfig,
): boolean {
  const cooldown = config.cooldown ?? DEFAULT_ANCESTOR_UNIQ_COOLDOWN;
  const lastAdjustGeneration = engine._lastAncestorUniqAdjustGen ?? ZERO;

  return engine.generation - lastAdjustGeneration >= cooldown;
}

/**
 * Extract the latest ancestor-uniqueness metric from telemetry.
 *
 * @param engine - NEAT engine instance.
 * @returns Ancestor uniqueness value or undefined when missing.
 */
export function extractAncestorUniqueness(
  engine: NeatLikeWithAdaptive,
): number | undefined {
  const lineageTelemetry = engine._telemetry?.at(NEGATIVE_ONE)?.lineage;
  const ancestorUniqueness = lineageTelemetry?.ancestorUniq;

  return typeof ancestorUniqueness === 'number'
    ? ancestorUniqueness
    : undefined;
}

/**
 * Resolve thresholds for ancestor-uniqueness decisions.
 *
 * @param config - Ancestor uniqueness adaptive configuration.
 * @returns Threshold bounds.
 */
export function resolveUniquenessThresholds(
  config: AncestorUniqAdaptiveConfig,
): { lowThreshold: number; highThreshold: number } {
  const lowThreshold =
    config.lowThreshold ?? DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD;
  const highThreshold =
    config.highThreshold ?? DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD;

  return { lowThreshold, highThreshold };
}

/**
 * Resolve adjustment magnitude for nudging controlled parameters.
 *
 * @param config - Ancestor uniqueness adaptive configuration.
 * @returns Adjustment magnitude.
 */
export function resolveAdjustmentMagnitude(
  config: AncestorUniqAdaptiveConfig,
): number {
  return config.adjust ?? DEFAULT_ANCESTOR_UNIQ_ADJUST;
}

/**
 * Apply an adjustment for the configured mode.
 *
 * @param engine - NEAT engine instance.
 * @param config - Ancestor uniqueness adaptive configuration.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @param adjustMagnitude - Adjustment magnitude.
 * @returns {void}
 */
export function applyUniquenessAdjustment(
  engine: NeatLikeWithAdaptive,
  config: AncestorUniqAdaptiveConfig,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number },
  adjustMagnitude: number,
): void {
  if (config.mode === ANCESTOR_UNIQ_MODE_EPSILON) {
    applyEpsilonAdjustment(engine, ancestorUniq, thresholds, adjustMagnitude);

    return;
  }

  if (config.mode === ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE) {
    applyLineagePressureAdjustment(engine, ancestorUniq, thresholds);
  }
}

/**
 * Apply dominance-epsilon adjustments when configured.
 *
 * @param engine - NEAT engine instance.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @param adjustMagnitude - Adjustment magnitude.
 * @returns {void}
 */
export function applyEpsilonAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number },
  adjustMagnitude: number,
): void {
  if (!engine.options.multiObjective?.adaptiveEpsilon?.enabled) return;

  const shouldIncrease = ancestorUniq < thresholds.lowThreshold;
  const shouldDecrease = ancestorUniq > thresholds.highThreshold;
  if (!shouldIncrease && !shouldDecrease) return;

  const currentEpsilon = engine.options.multiObjective.dominanceEpsilon ?? ZERO;
  engine.options.multiObjective.dominanceEpsilon = shouldIncrease
    ? currentEpsilon + adjustMagnitude
    : Math.max(ZERO, currentEpsilon - adjustMagnitude);

  recordAdjustment(engine);
}

/**
 * Apply lineage pressure strength adjustments.
 *
 * @param engine - NEAT engine instance.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @returns {void}
 */
export function applyLineagePressureAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number },
): void {
  const lineagePressureState = ensureLineagePressureState(engine);

  const shouldIncrease = ancestorUniq < thresholds.lowThreshold;
  const shouldDecrease = ancestorUniq > thresholds.highThreshold;
  if (!shouldIncrease && !shouldDecrease) return;

  const currentStrength =
    lineagePressureState.strength ?? DEFAULT_LINEAGE_PRESSURE_STRENGTH;
  lineagePressureState.strength = shouldIncrease
    ? currentStrength * LINEAGE_PRESSURE_INCREASE_MULTIPLIER
    : currentStrength * LINEAGE_PRESSURE_DECREASE_MULTIPLIER;

  if (shouldIncrease) lineagePressureState.mode = LINEAGE_PRESSURE_MODE_SPREAD;

  recordAdjustment(engine);
}

/**
 * Ensure lineage pressure state is available.
 *
 * @param engine - NEAT engine instance.
 * @returns Lineage pressure configuration object.
 */
export function ensureLineagePressureState(
  engine: NeatLikeWithAdaptive,
): NonNullable<NeatLikeWithAdaptive['options']['lineagePressure']> {
  if (!engine.options.lineagePressure) {
    engine.options.lineagePressure = {
      enabled: true,
      mode: LINEAGE_PRESSURE_MODE_SPREAD,
      strength: DEFAULT_LINEAGE_PRESSURE_STRENGTH,
    };
  }

  return engine.options.lineagePressure;
}

/**
 * Record the generation when an adjustment is applied.
 *
 * @param engine - NEAT engine instance.
 * @returns {void}
 */
export function recordAdjustment(engine: NeatLikeWithAdaptive): void {
  engine._lastAncestorUniqAdjustGen = engine.generation;
}

/**
 * Check whether mutation adaptation should run this generation.
 *
 * @param generation - Current generation index.
 * @param config - Adaptive mutation configuration.
 * @returns True if adaptation should run.
 */
export function shouldAdaptThisGeneration(
  generation: number,
  config: AdaptiveMutationConfig,
): boolean {
  const adaptEvery = config.adaptEvery ?? DEFAULT_ADAPT_EVERY;
  return adaptEvery <= DEFAULT_ADAPT_EVERY || generation % adaptEvery === ZERO;
}

/**
 * Collect genomes with numeric scores.
 *
 * @param population - Population of genomes.
 * @returns Scored genomes.
 */
export function collectScoredGenomes(population: Genome[]): Genome[] {
  return population.filter((genome) => typeof genome.score === 'number');
}

/**
 * Sort scored genomes in ascending score order.
 *
 * @param scoredGenomes - Scored genomes.
 * @returns Sorted genomes.
 */
export function sortScoredGenomes(scoredGenomes: Genome[]): Genome[] {
  return scoredGenomes.toSorted(
    (leftGenome, rightGenome) =>
      (leftGenome.score ?? ZERO) - (rightGenome.score ?? ZERO),
  );
}

/**
 * Split scored genomes into top and bottom halves.
 *
 * @param scoredGenomes - Sorted scored genomes.
 * @returns Partitions used by strategy rules.
 */
export function splitScoredGenomes(
  scoredGenomes: Genome[],
): MutationPartitions {
  const halfIndex = Math.floor(scoredGenomes.length / HALF_INDEX_DIVISOR);
  const bottomHalf = scoredGenomes.slice(ZERO, halfIndex);
  const topHalf = scoredGenomes.slice(halfIndex);
  return { topHalf, bottomHalf };
}

/**
 * Resolve mutation settings derived from configuration and engine state.
 *
 * @param engine - NEAT engine instance.
 * @param config - Adaptive mutation configuration.
 * @returns Resolved mutation settings.
 */
export function resolveMutationSettings(
  engine: NeatLikeWithAdaptive,
  config: AdaptiveMutationConfig,
): MutationSettings {
  const sigmaBase =
    (config.sigma ?? DEFAULT_MUTATION_SIGMA) * MUTATION_SIGMA_SCALE;
  const minRate = config.minRate ?? DEFAULT_MIN_MUTATION_RATE;
  const maxRate = config.maxRate ?? DEFAULT_MAX_MUTATION_RATE;
  const strategy = config.strategy ?? MUTATION_STRATEGY_TWO_TIER;
  const initialRate = config.initialRate ?? DEFAULT_INITIAL_MUTATION_RATE;
  const adaptAmount = config.adaptAmount ?? false;
  const amountSigma = config.amountSigma ?? DEFAULT_MUTATION_AMOUNT_SIGMA;
  const minAmount = config.minAmount ?? DEFAULT_MIN_MUTATION_AMOUNT;
  const maxAmount = config.maxAmount ?? DEFAULT_MAX_MUTATION_AMOUNT;
  const mutationAmountDefault =
    engine.options.mutationAmount ?? DEFAULT_MUTATION_AMOUNT;
  const generation = engine.generation;
  const populationSize = engine.population.length;

  return {
    strategy,
    sigmaBase,
    minRate,
    maxRate,
    initialRate,
    adaptAmount,
    amountSigma,
    minAmount,
    maxAmount,
    mutationAmountDefault,
    generation,
    populationSize,
  };
}

/**
 * Resolve a random source that matches the legacy RNG usage.
 *
 * @param engine - NEAT engine instance.
 * @returns Random number provider.
 */
export function resolveRandomSource(
  engine: NeatLikeWithAdaptive,
): () => number {
  const rngFactory = engine._getRNG ?? (() => Math.random);
  return () => rngFactory()();
}

/**
 * Apply mutation updates to the population.
 *
 * @param population - Full population to mutate.
 * @param partitions - Scored partitions.
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @returns Mutation outcome flags.
 */
export function applyMutationsToPopulation(
  population: Genome[],
  partitions: MutationPartitions,
  settings: MutationSettings,
  randomSource: () => number,
): MutationOutcome {
  const topHalfSet = new Set(partitions.topHalf);
  const bottomHalfSet = new Set(partitions.bottomHalf);
  let hasIncrease = false;
  let hasDecrease = false;

  for (let genomeIndex = ZERO; genomeIndex < population.length; genomeIndex++) {
    const genome = population[genomeIndex];
    if (genome._mutRate === undefined || genome._mutRate === null) continue;

    const rateDelta = resolveRateDelta(
      settings,
      randomSource,
      genome,
      genomeIndex,
      topHalfSet,
      bottomHalfSet,
    );
    const mutationRate = clampValue(
      genome._mutRate + rateDelta,
      settings.minRate,
      settings.maxRate,
    );

    if (mutationRate > settings.initialRate) hasIncrease = true;
    if (mutationRate < settings.initialRate) hasDecrease = true;

    genome._mutRate = mutationRate;

    if (settings.adaptAmount) {
      applyMutationAmount(
        genome,
        settings,
        randomSource,
        genomeIndex,
        topHalfSet,
        bottomHalfSet,
      );
    }
  }

  return { hasIncrease, hasDecrease };
}

/**
 * Resolve mutation-rate delta based on strategy.
 *
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Signed mutation rate delta.
 */
export function resolveRateDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  const baseDelta = createRandomDelta(settings.sigmaBase, randomSource);

  if (settings.strategy === MUTATION_STRATEGY_TWO_TIER) {
    return applyTwoTierDelta(
      baseDelta,
      genome,
      genomeIndex,
      topHalfSet,
      bottomHalfSet,
    );
  }

  if (settings.strategy === MUTATION_STRATEGY_EXPLORE_LOW) {
    return applyExploreLowDelta(baseDelta, genome, bottomHalfSet);
  }

  if (settings.strategy === MUTATION_STRATEGY_ANNEAL) {
    return applyAnnealDelta(baseDelta, settings);
  }

  return baseDelta;
}

/**
 * Create a signed random delta scaled by sigma.
 *
 * @param sigmaBase - Sigma scaling factor.
 * @param randomSource - Random number provider.
 * @returns Signed delta.
 */
export function createRandomDelta(
  sigmaBase: number,
  randomSource: () => number,
): number {
  const baseUnit = randomSource() * RNG_SPREAD_MULTIPLIER - RNG_CENTER_OFFSET;
  return baseUnit * sigmaBase;
}

/**
 * Apply two-tier adjustments to a delta.
 *
 * @param baseDelta - Base random delta.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Adjusted delta.
 */
export function applyTwoTierDelta(
  baseDelta: number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  if (!topHalfSet.size || !bottomHalfSet.size) {
    const isEvenIndex = genomeIndex % HALF_INDEX_DIVISOR === ZERO;
    return isEvenIndex ? Math.abs(baseDelta) : -Math.abs(baseDelta);
  }

  if (topHalfSet.has(genome)) return -Math.abs(baseDelta);
  if (bottomHalfSet.has(genome)) return Math.abs(baseDelta);
  return baseDelta;
}

/**
 * Apply explore-low adjustments to a delta.
 *
 * @param baseDelta - Base random delta.
 * @param genome - Current genome.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Adjusted delta.
 */
export function applyExploreLowDelta(
  baseDelta: number,
  genome: Genome,
  bottomHalfSet: Set<Genome>,
): number {
  if (bottomHalfSet.has(genome)) {
    return Math.abs(baseDelta * EXPLORE_LOW_INCREASE_MULTIPLIER);
  }

  return -Math.abs(baseDelta * EXPLORE_LOW_DECREASE_MULTIPLIER);
}

/**
 * Apply annealing adjustments to a delta.
 *
 * @param baseDelta - Base random delta.
 * @param settings - Resolved settings.
 * @returns Adjusted delta.
 */
export function applyAnnealDelta(
  baseDelta: number,
  settings: MutationSettings,
): number {
  const progress = Math.min(
    ANNEAL_PROGRESS_MAX,
    settings.generation /
      (ANNEAL_BASELINE_GENERATIONS + settings.populationSize),
  );
  return baseDelta * (ANNEAL_PROGRESS_MAX - progress);
}

/**
 * Apply mutation-amount adjustments to a genome.
 *
 * @param genome - Current genome.
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns {void}
 */
export function applyMutationAmount(
  genome: Genome,
  settings: MutationSettings,
  randomSource: () => number,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): void {
  const amountDelta = resolveAmountDelta(
    settings,
    randomSource,
    genome,
    genomeIndex,
    topHalfSet,
    bottomHalfSet,
  );
  const mutationAmount = clampValue(
    Math.round(
      (genome._mutAmount ?? settings.mutationAmountDefault) + amountDelta,
    ),
    settings.minAmount,
    settings.maxAmount,
  );
  genome._mutAmount = mutationAmount;
}

/**
 * Resolve mutation-amount delta based on strategy.
 *
 * @param settings - Resolved settings.
 * @param randomSource - Random number provider.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Signed mutation amount delta.
 */
export function resolveAmountDelta(
  settings: MutationSettings,
  randomSource: () => number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  const baseDelta = createRandomDelta(settings.amountSigma, randomSource);

  if (settings.strategy === MUTATION_STRATEGY_TWO_TIER) {
    return applyTwoTierAmountDelta(
      baseDelta,
      genome,
      genomeIndex,
      topHalfSet,
      bottomHalfSet,
    );
  }

  return baseDelta;
}

/**
 * Apply two-tier adjustments to amount delta.
 *
 * @param baseDelta - Base random delta.
 * @param genome - Current genome.
 * @param genomeIndex - Genome index.
 * @param topHalfSet - Lookup for top-half genomes.
 * @param bottomHalfSet - Lookup for bottom-half genomes.
 * @returns Adjusted delta.
 */
export function applyTwoTierAmountDelta(
  baseDelta: number,
  genome: Genome,
  genomeIndex: number,
  topHalfSet: Set<Genome>,
  bottomHalfSet: Set<Genome>,
): number {
  if (!topHalfSet.size || !bottomHalfSet.size) {
    const isEvenIndex = genomeIndex % HALF_INDEX_DIVISOR === ZERO;
    return isEvenIndex ? Math.abs(baseDelta) : -Math.abs(baseDelta);
  }

  if (bottomHalfSet.has(genome)) return Math.abs(baseDelta);
  if (topHalfSet.has(genome)) return -Math.abs(baseDelta);
  return baseDelta;
}

/**
 * Clamp a value between min and max bounds.
 *
 * @param value - Value to clamp.
 * @param min - Minimum bound.
 * @param max - Maximum bound.
 * @returns Clamped value.
 */
export function clampValue(value: number, min: number, max: number): number {
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

/**
 * Determine whether a two-tier fallback is needed.
 *
 * @param strategy - Mutation strategy identifier.
 * @param outcome - Mutation outcome flags.
 * @returns True if fallback should run.
 */
export function shouldApplyTwoTierFallback(
  strategy: string,
  outcome: MutationOutcome,
): boolean {
  if (strategy !== MUTATION_STRATEGY_TWO_TIER) return false;
  return !(outcome.hasIncrease && outcome.hasDecrease);
}

/**
 * Apply two-tier fallback balancing.
 *
 * @param population - Population of genomes.
 * @param settings - Resolved settings.
 * @returns {void}
 */
export function applyTwoTierFallback(
  population: Genome[],
  settings: MutationSettings,
): void {
  const halfIndex = Math.floor(population.length / HALF_INDEX_DIVISOR);

  for (let genomeIndex = ZERO; genomeIndex < population.length; genomeIndex++) {
    const genome = population[genomeIndex];
    if (genome._mutRate === undefined || genome._mutRate === null) continue;

    const adjustedRate =
      genomeIndex < halfIndex
        ? genome._mutRate + settings.sigmaBase
        : genome._mutRate - settings.sigmaBase;
    genome._mutRate = clampValue(
      adjustedRate,
      settings.minRate,
      settings.maxRate,
    );
  }
}

/**
 * Resolve the decay factor for operator statistics.
 *
 * @param config - Operator adaptation configuration.
 * @returns Decay factor for exponential smoothing.
 */
export function resolveOperatorDecay(config: OperatorAdaptationConfig): number {
  return config.decay ?? OPERATOR_DECAY_DEFAULT;
}

/**
 * Collect operator statistic entries for processing.
 *
 * @param stats - Operator statistics map.
 * @returns Array of operator stat entries.
 */
export function collectOperatorStatsEntries(
  stats: Map<string, { success: number; attempts: number }>,
): Array<[string, { success: number; attempts: number }]> {
  return Array.from(stats.entries());
}

/**
 * Apply exponential decay to each operator statistic entry.
 *
 * @param stats - Operator statistics map.
 * @param entries - Operator stat entries to update.
 * @param decay - Decay factor.
 * @returns {void}
 */
export function applyOperatorDecay(
  stats: Map<string, { success: number; attempts: number }>,
  entries: Array<[string, { success: number; attempts: number }]>,
  decay: number,
): void {
  for (const [operatorId, operatorStat] of entries) {
    const nextStat = decayOperatorStat(operatorStat, decay);
    stats.set(operatorId, nextStat);
  }
}

/**
 * Apply decay to a single operator statistic record.
 *
 * @param operatorStat - Operator statistic record.
 * @param decay - Decay factor.
 * @returns Decayed operator statistic record.
 */
export function decayOperatorStat(
  operatorStat: { success: number; attempts: number },
  decay: number,
): { success: number; attempts: number } {
  return {
    success: operatorStat.success * decay,
    attempts: operatorStat.attempts * decay,
  };
}
