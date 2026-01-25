/** Default neighbor count for novelty calculation. */
export const NOVELTY_DEFAULT_NEIGHBORS = 3;
/** Default blend factor for novelty vs. fitness. */
export const NOVELTY_DEFAULT_BLEND = 0.3;
/** Maximum number of entries stored in the novelty archive. */
export const NOVELTY_ARCHIVE_CAP = 200;
/** Default target variance for entropy sharing. */
export const ENTROPY_VAR_TARGET_DEFAULT = 0.2;
/** Default adjustment rate for entropy sharing. */
export const ENTROPY_VAR_ADJUST_DEFAULT = 0.1;
/** Default minimum sigma for entropy sharing. */
export const ENTROPY_VAR_MIN_SIGMA_DEFAULT = 0.1;
/** Default maximum sigma for entropy sharing. */
export const ENTROPY_VAR_MAX_SIGMA_DEFAULT = 10;
/** Lower band multiplier for entropy variance tuning. */
export const ENTROPY_VAR_LOW_BAND = 0.9;
/** Upper band multiplier for entropy variance tuning. */
export const ENTROPY_VAR_HIGH_BAND = 1.1;
/** Default target entropy for compatibility tuning. */
export const ENTROPY_TARGET_DEFAULT = 0.5;
/** Default deadband for compatibility tuning. */
export const ENTROPY_DEADBAND_DEFAULT = 0.05;
/** Default adjustment rate for compatibility tuning. */
export const ENTROPY_ADJUST_DEFAULT = 0.05;
/** Default compatibility threshold when not provided. */
export const COMPAT_THRESHOLD_DEFAULT = 3;
/** Default minimum compatibility threshold. */
export const COMPAT_MIN_THRESHOLD_DEFAULT = 0.5;
/** Default maximum compatibility threshold. */
export const COMPAT_MAX_THRESHOLD_DEFAULT = 10;
/** Default adjustment rate for auto distance coefficient tuning. */
export const AUTO_COEFF_ADJUST_DEFAULT = 0.05;
/** Default minimum coefficient for auto distance coefficient tuning. */
export const AUTO_COEFF_MIN_DEFAULT = 0.05;
/** Default maximum coefficient for auto distance coefficient tuning. */
export const AUTO_COEFF_MAX_DEFAULT = 8;
/** Default coefficient value when not provided. */
export const DISTANCE_COEFF_DEFAULT = 1;
/** Variance decrease threshold multiplier. */
export const VARIANCE_DECREASE_THRESHOLD = 0.95;
/** Variance increase threshold multiplier. */
export const VARIANCE_INCREASE_THRESHOLD = 1.05;
