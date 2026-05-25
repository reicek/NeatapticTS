/**
 * Error message emitted when the supplied dataset dimensions do not match the network input or output size.
 */
export const DATASET_COMPATIBILITY_ERROR_MESSAGE =
  'Dataset is invalid or dimensions do not match network input/output size!';

/**
 * Error message emitted when an evolution call is started with neither an iteration limit nor an error target specified.
 */
export const STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE =
  'At least one stopping condition (`iterations` or `error`) must be specified for evolution.';

/**
 * Default target error threshold used when no explicit error stopping condition is provided to the evolve call.
 */
export const DEFAULT_TARGET_ERROR = 0.05;

/**
 * Default per-connection complexity growth penalty applied when computing fitness-adjusted complexity scores in the evolve loop.
 */
export const DEFAULT_GROWTH = 0.0001;

/**
 * Default number of repeated fitness evaluations used when no explicit evaluation amount is specified per genome.
 */
export const DEFAULT_EVALUATION_AMOUNT = 1;

/**
 * Default generation logging frequency; zero disables per-generation log output during the evolve loop.
 */
export const DEFAULT_LOG_INTERVAL = 0;

/**
 * Default worker thread count used when no explicit thread override is provided to single-thread evolve calls.
 */
export const DEFAULT_THREAD_COUNT = 1;

/**
 * Sentinel error value indicating that error-based stopping is explicitly disabled and only iteration limits apply.
 */
export const DISABLED_TARGET_ERROR = -1;

/**
 * Explicit zero used as an initial iteration counter and for stopping-condition comparisons during evolve loop entry.
 */
export const ZERO_ITERATIONS = 0;

/**
 * Population size threshold below which the evolve loop applies more aggressive fallback mutation rates and amounts.
 */
export const SMALL_POPULATION_THRESHOLD = 10;

/**
 * Mutation rate fallback applied when the active population falls below the small-population threshold during evolution.
 */
export const SMALL_POPULATION_MUTATION_RATE = 0.5;

/**
 * Mutation amount fallback applied when the active population falls below the small-population threshold during evolution.
 */
export const SMALL_POPULATION_MUTATION_AMOUNT = 1;

/**
 * Maximum number of consecutive NaN or Infinity fitness values tolerated before the evolve loop aborts early.
 */
export const MAX_CONSECUTIVE_INVALID_ERRORS = 5;

/**
 * Shared summary payload returned by the evolve loop containing the best error, generation count, and wall-clock time.
 */
export type EvolutionSummary = {
  error: number;
  iterations: number;
  time: number;
};

/**
 * Structural node, connection, and gate counts used by complexity growth-penalty heuristics during fitness adjustment.
 */
export type GenomeStructureCounts = {
  nodeCount: number;
  connectionCount: number;
  gateCount: number;
};
