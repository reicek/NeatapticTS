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
 * Canonical 50% probability threshold for binary parent or gene selection during crossover; values above this choose one parent, values below choose the other.
 */
export const RANDOM_BINARY_SELECTION_THRESHOLD = 0.5;

/**
 * Default probability for re-enabling a disabled gene when both parents carry it in their gene lists during NEAT crossover.
 */
export const DEFAULT_REENABLE_PROBABILITY = 0.25;

/**
 * Sentinel index value indicating that a connection has no gating node assigned after crossover or network construction.
 */
export const NO_GATER_INDEX = -1;

/**
 * Zero-based first-element index used when reading the first item from newly created connection arrays after crossover gene assembly.
 */
export const FIRST_INDEX = 0;

/**
 * Shared error message text for the crossover parent compatibility guard; thrown when parent networks have mismatched input or output dimension counts.
 */
export const PARENT_COMPATIBILITY_ERROR_MESSAGE =
  'Parent networks must have the same input and output sizes for crossover.';

/**
 * Shared random number generator signature consumed by genetic operators; each call returns a uniform float in the half-open interval [0, 1).
 */
export type RandomGenerator = () => number;
