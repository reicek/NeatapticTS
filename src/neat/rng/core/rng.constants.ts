/**
 * Constants used by the deterministic xorshift RNG helper.
 *
 * Read these values when you want to understand the fixed numeric choices that
 * shape seed guarding, time scrambling, and integer-to-float normalization.
 */
export const RNG_TIME_SCRAMBLE_CONSTANT = 0x9e3779b1;

/** Fallback seed used when the derived seed would be zero (xorshift cannot use 0). */
export const RNG_DEFAULT_SEED_FALLBACK = 0x1a2b3c4d;

/** Bit-shift values for the xorshift32 variant. */
export const RNG_SHIFT_LEFT_PRIMARY = 13;
export const RNG_SHIFT_RIGHT_PRIMARY = 17;
export const RNG_SHIFT_LEFT_SECONDARY = 5;

/** Divisor used to normalize the 32-bit integer state into [0, 1). */
export const RNG_NORMALIZATION_DIVISOR = 0xffffffff;

/** Minimum population offset added before scrambling to avoid zero seeds. */
export const RNG_POPULATION_OFFSET = 1;
