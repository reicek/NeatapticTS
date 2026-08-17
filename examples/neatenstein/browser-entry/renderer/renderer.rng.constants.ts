/**
 * Shared random-number-generator constants for the Neatenstein renderer.
 *
 * Centralises the Park-Miller LCG primes and derez hash primes so they are
 * defined once and shared across {@link module:./map}, {@link module:./pulse},
 * and {@link module:./derez} without duplication.
 *
 * @module
 */

/**
 * Modulus for the Park-Miller minimal standard LCG.
 *
 * This is the largest Mersenne prime that fits in a signed 32-bit integer
 * (2³¹ − 1). Used by map generation and pulse emission to produce
 * deterministic pseudo-random sequences from a seed.
 */
export const PARK_MILLER_MODULUS = 2_147_483_647;

/**
 * Multiplier for the Park-Miller minimal standard LCG.
 *
 * Combined with {@link PARK_MILLER_MODULUS}, this produces the canonical
 * Park-Miller (1988) minimal standard generator.
 */
export const PARK_MILLER_MULTIPLIER = 16_807;

/**
 * First derez hash prime, multiplied with the X coordinate.
 *
 * Part of the deterministic derez noise hash used to vary enemy sprite
 * appearance per-cell.
 */
export const DEREZ_HASH_PRIME_1 = 374761393;

/**
 * Second derez hash prime, multiplied with the Y coordinate.
 *
 * Part of the deterministic derez noise hash used to vary enemy sprite
 * appearance per-cell.
 */
export const DEREZ_HASH_PRIME_2 = 668265263;

/**
 * Third derez hash prime, multiplied with the seed.
 *
 * Part of the deterministic derez noise hash used to vary enemy sprite
 * appearance per-cell.
 */
export const DEREZ_HASH_PRIME_3 = 2246822519;

/**
 * Modulus (divisor) for the derez hash, equal to 2³².
 *
 * The hash result is divided by this value to normalise the output into
 * the `[0, 1)` range.
 */
export const DEREZ_HASH_MODULUS = 0x100000000;
