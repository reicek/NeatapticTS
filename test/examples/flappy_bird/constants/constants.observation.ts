/**
 * Shared observation-normalization constants.
 *
 * These values are used by both browser playback and headless environment
 * simulation when building normalized feature vectors.
 */

/**
 * Small epsilon divisor guard for world/physics normalization.
 *
 * Prevents division by near-zero values when view-dependent scales are very
 * small, keeping feature values numerically stable.
 */
export const FLAPPY_NORMALIZATION_EPSILON = 0.001;
