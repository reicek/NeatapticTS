/**
 * Shared dataset compatibility error message.
 */
export const DATASET_COMPATIBILITY_ERROR_MESSAGE =
  'Dataset is invalid or dimensions do not match network input/output size!';

/**
 * Shared evolve stopping-condition validation error.
 */
export const STOPPING_CONDITION_REQUIRED_ERROR_MESSAGE =
  'At least one stopping condition (`iterations` or `error`) must be specified for evolution.';

/**
 * Default target error used when omitted.
 */
export const DEFAULT_TARGET_ERROR = 0.05;

/**
 * Default complexity growth penalty.
 */
export const DEFAULT_GROWTH = 0.0001;

/**
 * Default repeated evaluation amount.
 */
export const DEFAULT_EVALUATION_AMOUNT = 1;

/**
 * Default logging frequency value.
 */
export const DEFAULT_LOG_INTERVAL = 0;

/**
 * Default single-thread worker count.
 */
export const DEFAULT_THREAD_COUNT = 1;

/**
 * Sentinel target error indicating that error-based stopping is disabled.
 */
export const DISABLED_TARGET_ERROR = -1;

/**
 * Explicit zero-iteration value.
 */
export const ZERO_ITERATIONS = 0;

/**
 * Population threshold considered "small" for mutation heuristics.
 */
export const SMALL_POPULATION_THRESHOLD = 10;

/**
 * Mutation rate fallback used for very small populations.
 */
export const SMALL_POPULATION_MUTATION_RATE = 0.5;

/**
 * Mutation amount fallback used for very small populations.
 */
export const SMALL_POPULATION_MUTATION_AMOUNT = 1;

/**
 * Maximum consecutive invalid errors tolerated before loop abort.
 */
export const MAX_CONSECUTIVE_INVALID_ERRORS = 5;

/**
 * Shared evolution summary payload.
 */
export type EvolutionSummary = {
  error: number;
  iterations: number;
  time: number;
};

/**
 * Structural counts used by complexity heuristics.
 */
export type GenomeStructureCounts = {
  nodeCount: number;
  connectionCount: number;
  gateCount: number;
};
