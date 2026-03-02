/**
 * Deterministic RNG used by the Flappy Bird demo.
 *
 * Using a local RNG keeps evaluations reproducible regardless of global
 * `Math.random()` state and makes comparisons between genomes more stable.
 */

/** Minimal interface needed by the simulation to sample randomness. */
export interface FlappyRng {
  /** @returns A floating-point number in the range [0, 1). */
  nextFloat01(): number;

  /**
   * @param minInclusive - Smallest integer that can be returned.
   * @param maxExclusive - One past the largest integer that can be returned.
   * @returns An integer in [minInclusive, maxExclusive).
   */
  nextInt(minInclusive: number, maxExclusive: number): number;
}

/**
 * Create a deterministic xorshift32 RNG.
 *
 * @param seed - Unsigned 32-bit seed.
 * @returns RNG instance.
 *
 * @example
 * const rng = createXorshift32(123);
 * rng.nextFloat01();
 */
export function createXorshift32(seed: number): FlappyRng {
  let state = toUint32(seed) || 0x6d2b79f5;

  return {
    nextFloat01() {
      state = xorshift32(state);
      // Convert to [0, 1) using 2^32.
      return state / 4_294_967_296;
    },
    nextInt(minInclusive, maxExclusive) {
      const minInt = Math.trunc(minInclusive);
      const maxInt = Math.trunc(maxExclusive);
      if (!Number.isFinite(minInt) || !Number.isFinite(maxInt)) {
        throw new Error('nextInt bounds must be finite numbers');
      }
      if (maxInt <= minInt) {
        throw new Error('nextInt expects maxExclusive > minInclusive');
      }

      const span = maxInt - minInt;
      const sample = this.nextFloat01();
      return minInt + Math.floor(sample * span);
    },
  };

  /** @param value - Number to coerce. @returns Unsigned 32-bit value. */
  function toUint32(value: number): number {
    // >>> 0 coerces to uint32.
    return (value >>> 0) as number;
  }

  /** @param value - uint32. @returns next uint32. */
  function xorshift32(value: number): number {
    let next = value >>> 0;
    next ^= (next << 13) >>> 0;
    next ^= next >>> 17;
    next ^= (next << 5) >>> 0;
    return next >>> 0;
  }
}
