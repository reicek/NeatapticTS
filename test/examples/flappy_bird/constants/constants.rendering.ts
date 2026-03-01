/**
 * Shared rendering-policy constants.
 *
 * These values are consumed by browser playback drawing but intentionally live
 * in shared Flappy constants to keep legacy imports stable during migration.
 */

/**
 * Fraction of parent bird opacity used for drawing trails.
 *
 * For example, `0.5` means a bird at 20% opacity gets a 10% opacity trail.
 */
export const FLAPPY_TRAIL_OPACITY_FACTOR = 0.5;
