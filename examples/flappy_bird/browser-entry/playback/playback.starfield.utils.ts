import {
  FLAPPY_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR,
  FLAPPY_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL,
  FLAPPY_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY,
  FLAPPY_STARFIELD_XORSHIFT_RIGHT_SHIFT,
} from '../../constants/constants';

/**
 * Creates a deterministic pseudo-random generator for starfield tile layouts.
 *
 * @param seed - Unsigned integer seed.
 * @returns Function that yields values in the range [0, 1).
 */
export function createSeededRandom(seed: number): () => number {
  let randomState = seed >>> 0;

  return () => {
    // Step 1: Apply xorshift32 state transitions.
    randomState ^= randomState << FLAPPY_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY;
    randomState >>>= 0;
    randomState ^= randomState >> FLAPPY_STARFIELD_XORSHIFT_RIGHT_SHIFT;
    randomState >>>= 0;
    randomState ^= randomState << FLAPPY_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL;
    randomState >>>= 0;

    // Step 2: Normalize 32-bit unsigned state into [0, 1).
    return randomState / FLAPPY_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR;
  };
}

/**
 * Resolves positive modulo suitable for horizontal tiling offsets.
 *
 * @param value - Input value to wrap.
 * @param modulo - Modulus base.
 * @returns Wrapped value in [0, modulo).
 */
export function positiveModulo(value: number, modulo: number): number {
  const remainder = value % modulo;
  return remainder < 0 ? remainder + modulo : remainder;
}
