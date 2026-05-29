/**
 * Rolling evaluation-window length used by the adult plateau detector. Defaults to 6 evaluation steps.
 */
export const NGE_ADULT_DEFAULT_PLATEAU_WINDOW = 6;

/**
 * Smallest meaningful reward improvement that still counts as positive adult return.
 */
export const NGE_ADULT_DEFAULT_MARGINAL_EPSILON = 0.01;

/**
 * Rolling evaluation-window length used by the adult gain-stability detector. Defaults to 5 evaluation steps.
 */
export const NGE_ADULT_DEFAULT_GAIN_STABILITY_WINDOW = 5;

/**
 * Allowed gain deviation before one adult zone is considered unstable again.
 */
export const NGE_ADULT_DEFAULT_GAIN_STABILITY_TOLERANCE = 0.05;

/**
 * Residual growth-budget fraction preserved while the adult phase biases toward prune and compact.
 */
export const NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR = 0.1;

/**
 * Minimum normalized focus score required before adult growth remains eligible at all.
 */
export const NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR = 0.6;
