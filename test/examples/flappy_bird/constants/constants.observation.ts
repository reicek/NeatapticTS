/**
 * Shared observation-normalization constants.
 *
 * These values are used by both browser playback and headless environment
 * simulation when building normalized feature vectors.
 *
 * Keeping normalization constants shared is important because even tiny drift in
 * feature scaling would make trainer, worker, and browser policies "see"
 * different worlds.
 */

/**
 * Small epsilon divisor guard for world/physics normalization.
 *
 * Prevents division by near-zero values when view-dependent scales are very
 * small, keeping feature values numerically stable.
 *
 * If you want a quick refresher on why feature scaling matters, the Wikipedia
 * article on "feature scaling" is a useful background reference.
 */
export const FLAPPY_NORMALIZATION_EPSILON = 0.001;
