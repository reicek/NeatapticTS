/**
 * Deterministic hash-seed utility for the Neatenstein co-evolution harness.
 *
 * Produces a stable 32-bit unsigned integer from a `(seed, generation,
 * variantId)` tuple. Used by both the worker-side Neat population creation
 * (P3S2) and the main-runner episode evaluation (P5S1) to seed Neat
 * populations deterministically without creating a circular module
 * dependency between `arms-race.ts` and `main-runner.ts`.
 *
 * @module
 */

/**
 * Produce a deterministic 32-bit unsigned hash from the seed, generation, and
 * optional variant id.
 *
 * The formula `((seed * 100003 + generation) * 100003 + (variantId ?? 0)) >>> 0`
 * uses two multiplications by the prime constant `100003` to distribute bits
 * across the 32-bit range. The unsigned right-shift (`>>> 0`) coerces the
 * result to a non-negative 32-bit integer.
 *
 * @param seed - Base seed value.
 * @param generation - Current co-evolution generation (non-negative integer).
 * @param variantId - Optional variant discriminator (defaults to 0).
 * @returns A 32-bit unsigned integer hash suitable for seeding a Neat
 *   population or an episode RNG.
 *
 * @example
 * ```ts
 * const popSeed = hashSeed(gameState.seed, gameState.generation);
 * const neatPop = new Neat(12, 5, fitnessFn, { popsize: 4, seed: popSeed });
 * ```
 */
export function hashSeed(
  seed: number,
  generation: number,
  variantId?: number,
): number {
  return ((seed * 100003 + generation) * 100003 + (variantId ?? 0)) >>> 0;
}
