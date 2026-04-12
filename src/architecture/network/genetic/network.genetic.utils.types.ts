/**
 * Shared constants and tiny contracts for the genetic crossover helpers.
 *
 * This file keeps the "small but important" numbers used across the crossover
 * implementation named and documented.
 *
 * Most of these are not user-facing tuning knobs. They are intended to:
 *
 * - keep randomness decisions easy to audit in tests,
 * - centralize compatibility error messaging,
 * - make sentinel values (like "no gater") explicit.
 */

/**
 * Canonical threshold used for random binary parent/gene choice.
 */
export const RANDOM_BINARY_SELECTION_THRESHOLD = 0.5;

/**
 * Default probability for re-enabling disabled genes during crossover.
 */
export const DEFAULT_REENABLE_PROBABILITY = 0.25;

/**
 * Sentinel index representing that no gater node is assigned.
 */
export const NO_GATER_INDEX = -1;

/**
 * First element index used when reading newly created connections.
 */
export const FIRST_INDEX = 0;

/**
 * Shared compatibility error message for crossover parent validation.
 */
export const PARENT_COMPATIBILITY_ERROR_MESSAGE =
  'Parent networks must have the same input and output sizes for crossover.';

/**
 * Shared random generator signature for genetic operators.
 */
export type RandomGenerator = () => number;
