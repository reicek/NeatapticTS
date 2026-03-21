/**
 * Odd scramble factor used while deriving a default seed from time and host
 * context.
 *
 * This constant helps mix the fallback seed path before the xorshift stream is
 * ever created. It matters only when callers did not already provide an RNG,
 * explicit seed, or restored numeric state.
 */
export const RNG_TIME_SCRAMBLE_CONSTANT = 0x9e3779b1;

/**
 * Fallback seed used when the derived or restored seed would otherwise be zero.
 *
 * Xorshift32 cannot advance from a zero state, so this constant is the guarded
 * non-zero escape hatch that keeps initialization and restore flows valid.
 * It is the last-resort seed, not the normal source of entropy.
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
 * into the floating-point random samples consumed by the controller. Keeping it
 * named makes the integer-state phase and the outward-facing sample phase read
 * like two explicit steps instead of one opaque formula.
 */
export const RNG_NORMALIZATION_DIVISOR = 0xffffffff;

/**
 * Minimum population offset added before time scrambling during default seeding.
 *
 * The offset keeps empty or tiny populations from collapsing the derived seed
 * toward zero too easily during initialization. It exists to stabilize the
 * fallback path, not to encode any meaningful NEAT population heuristic.
 */
export const RNG_POPULATION_OFFSET = 1;
