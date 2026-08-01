/**
 * Deterministic seed-pack generation for the Neatenstein asymmetric
 * co-evolution harness.
 *
 * Every variant evaluated in a single generation sees the same frozen seed
 * stream so that fitness differences reflect the variant, not environmental
 * variance. The pack is fully deterministic from the generation number and
 * variant count alone, making episodes replay-safe without storing per-variant
 * RNG state.
 *
 * @module
 */

import type { SeedPack } from './types.ts';
import { NEATENSTEIN_MLP_VARIANT_COUNT } from './constants.ts';

/**
 * Configuration for {@link createSeedPack}.
 */
export interface CreateSeedPackOptions {
  /** Generation the seed pack belongs to (non-negative integer). */
  generation: number;
  /** Number of deterministic seeds to generate (defaults to the MLP variant count). */
  variantCount?: number;
}

/** LCG multiplier from the classic Park-Miller minimal standard. */
const SEED_PACK_LCG_MULTIPLIER = 16_807;

/** LCG modulus (a Mersenne prime) kept within 31-bit signed range. */
const SEED_PACK_LCG_MODULUS = 2_147_483_647;

/** Arbitrary non-zero offset that breaks the zero-state fixed point. */
const SEED_PACK_LCG_OFFSET = 1_234_567_890;

/**
 * Mix a generation number into a positive 32-bit LCG seed.
 *
 * Uses a golden-ratio-derived multiplier so adjacent generations produce
 * uncorrelated initial states.
 *
 * @param generation - Generation to seed from.
 * @returns A positive integer LCG state.
 */
function createLcgSeed(generation: number): number {
  return ((generation + 1) * 2_654_435_761) >>> 0;
}

/**
 * Advance one step of the deterministic LCG.
 *
 * @param state - Current LCG state.
 * @returns Next LCG state.
 */
function nextLcgState(state: number): number {
  return (
    (state * SEED_PACK_LCG_MULTIPLIER + SEED_PACK_LCG_OFFSET) %
    SEED_PACK_LCG_MODULUS
  );
}

/**
 * Generate an ordered list of deterministic seeds for a generation.
 *
 * @param generation - Generation to generate seeds for.
 * @param count - Number of seeds to generate.
 * @returns Ordered array of integer seeds.
 */
function generateDeterministicSeeds(
  generation: number,
  count: number,
): number[] {
  const seeds: number[] = [];
  let state = createLcgSeed(generation);

  for (let index = 0; index < count; index++) {
    state = nextLcgState(state);
    seeds.push(state);
  }

  return seeds;
}

/**
 * Create a deterministic seed pack for the given generation.
 *
 * The same `generation` and `variantCount` always produce the same ordered
 * seeds, ensuring that every variant evaluated in a generation experiences
 * identical environmental randomness.
 *
 * @param options - Seed pack configuration.
 * @param options.generation - Generation the pack belongs to.
 * @param options.variantCount - Number of seeds to generate; defaults to
 *   {@link NEATENSTEIN_MLP_VARIANT_COUNT}.
 * @returns A frozen seed pack tied to the requested generation.
 * @throws Error when `generation` is negative or `variantCount` is not a
 *   positive integer.
 *
 * @example
 * ```ts
 * const pack = createSeedPack({ generation: 5, variantCount: 32 });
 * console.log(pack.seeds.length); // 32
 * ```
 */
export function createSeedPack({
  generation,
  variantCount = NEATENSTEIN_MLP_VARIANT_COUNT,
}: CreateSeedPackOptions): SeedPack {
  if (
    !Number.isFinite(generation) ||
    generation < 0 ||
    !Number.isInteger(generation)
  ) {
    throw new Error('generation must be a non-negative integer');
  }
  if (
    !Number.isFinite(variantCount) ||
    !Number.isInteger(variantCount) ||
    variantCount <= 0
  ) {
    throw new Error('variantCount must be a positive integer');
  }

  return makeSeedPack({
    generation,
    seeds: generateDeterministicSeeds(generation, variantCount),
  });
}

/**
 * Build a frozen seed pack from a generation label and a seed list.
 *
 * @param pack - Raw seed pack data.
 * @returns Frozen {@link SeedPack}.
 */
function makeSeedPack(pack: { generation: number; seeds: number[] }): SeedPack {
  const seeds = Object.freeze([...pack.seeds]);
  return Object.freeze({
    generation: pack.generation,
    seeds,
  }) as unknown as SeedPack;
}

/**
 * Create a deterministic enemy seed pack from a single seed.
 *
 * The pack contains one seed per MLP enemy variant so every variant in a
 * generation is evaluated against the same frozen environmental randomness.
 * Calling this with the same seed always produces the same ordered, frozen
 * pack.
 *
 * @param seed - Deterministic seed for the generation.
 * @returns A frozen seed pack with {@link NEATENSTEIN_MLP_VARIANT_COUNT} seeds.
 *
 * @example
 * ```ts
 * const pack = makeEnemySeedPack(123);
 * console.log(pack.seeds.length); // 32
 * ```
 */
export function makeEnemySeedPack(seed: number): SeedPack {
  if (!Number.isFinite(seed) || !Number.isInteger(seed)) {
    throw new Error('seed must be a finite integer');
  }

  const seeds = generateDeterministicSeeds(seed, NEATENSTEIN_MLP_VARIANT_COUNT);
  return makeSeedPack({ generation: seed, seeds });
}
