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
    randomState ^= randomState << 13;
    randomState >>>= 0;
    randomState ^= randomState >> 17;
    randomState >>>= 0;
    randomState ^= randomState << 5;
    randomState >>>= 0;

    // Step 2: Normalize 32-bit unsigned state into [0, 1).
    return randomState / 0x1_0000_0000;
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
