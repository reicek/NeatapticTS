/**
 * Default curriculum scale used when callers do not provide one.
 *
 * A value of `1` means full adaptive difficulty behavior is enabled.
 */
export const FLAPPY_SHARED_DEFAULT_DIFFICULTY_SCALE = 1;

/**
 * Small positive epsilon used to guard divisions in normalized timing features.
 *
 * The epsilon avoids unstable divide-by-zero behavior when distances or speeds
 * collapse toward zero during normalization.
 */
export const FLAPPY_SHARED_DEFAULT_NORMALIZATION_EPSILON = 0.001;
