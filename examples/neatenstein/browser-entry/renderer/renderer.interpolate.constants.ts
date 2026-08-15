/**
 * Shared interpolation constants for the Neatenstein render-side state
 * interpolation.
 *
 * Centralises the snapshot name labels and alpha bounds used by
 * {@link module:./interpolate} so they can be referenced without inline magic
 * strings and numbers.
 *
 * @module
 */

/**
 * Label identifying the previous simulation snapshot in validation messages.
 *
 * Used by {@link module:./interpolate} when copying or reading fields from the
 * `previous` argument so error messages reference the correct source.
 */
export const INTERP_PREVIOUS = 'previous';

/**
 * Label identifying the current simulation snapshot in validation messages.
 *
 * Used by {@link module:./interpolate} when copying or reading fields from the
 * `current` argument so error messages reference the correct source.
 */
export const INTERP_CURRENT = 'current';

/** Lower bound for the interpolation alpha blend factor. */
export const MIN_INTERPOLATION_ALPHA = 0;

/** Upper bound for the interpolation alpha blend factor. */
export const MAX_INTERPOLATION_ALPHA = 1;