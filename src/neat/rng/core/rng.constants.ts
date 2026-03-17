/**
 * Constants used by the deterministic xorshift RNG helper.
 *
 * Read these values when you want to understand the fixed numeric choices that
 * shape seed guarding, time scrambling, and integer-to-float normalization.
 * The root RNG chapter uses these constants to make the replay contract
 * inspectable: none of these values are magic once you understand whether they
 * control initial seeding, xorshift mutation, or float normalization.
 */
export const RNG_TIME_SCRAMBLE_CONSTANT = 0x9e3779b1;

/**
 * Fallback seed used when the derived or restored seed would otherwise be zero.
 *
 * Xorshift32 cannot advance from a zero state, so this constant is the guarded
 * non-zero escape hatch that keeps initialization and restore flows valid.
 */
export const RNG_DEFAULT_SEED_FALLBACK = 0x1a2b3c4d;

/** Left-shift used by the first xorshift32 mixing step. */
export const RNG_SHIFT_LEFT_PRIMARY = 13;
/** Right-shift used by the middle xorshift32 mixing step. */
export const RNG_SHIFT_RIGHT_PRIMARY = 17;
/** Left-shift used by the final xorshift32 mixing step. */
export const RNG_SHIFT_LEFT_SECONDARY = 5;

/**
 * Divisor used to normalize the 32-bit integer state into the `[0, 1)` range.
 *
 * This is the final step that turns a deterministic integer state transition
 * into the floating-point random samples consumed by the controller.
 */
export const RNG_NORMALIZATION_DIVISOR = 0xffffffff;

/**
 * Minimum population offset added before time scrambling during default seeding.
 *
 * The offset keeps empty or tiny populations from collapsing the derived seed
 * toward zero too easily during initialization.
 */
export const RNG_POPULATION_OFFSET = 1;
