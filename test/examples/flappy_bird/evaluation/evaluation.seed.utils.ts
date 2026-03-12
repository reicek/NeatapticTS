import {
  FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_A,
  FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_B,
  FLAPPY_EVALUATION_SEED_MIX_XOR_SALT,
} from './evaluation.constants';

/**
 * Mixes a genome identifier into a stable uint32 rollout seed.
 *
 * This keeps evaluation deterministic per genome while still spreading nearby
 * genome ids across the RNG state space to reduce correlated rollouts.
 *
 * If you want background reading, the Wikipedia article on "hash function"
 * gives a reasonable intuition for why a few avalanche-style mixing steps help
 * nearby ids map to less-correlated seed values.
 *
 * @param genomeId - Genome id from NEAT bookkeeping.
 * @returns Mixed uint32 seed.
 */
export function mixGenomeEvaluationSeed(genomeId: number): number {
  let mixedSeed = (genomeId >>> 0) ^ FLAPPY_EVALUATION_SEED_MIX_XOR_SALT;
  mixedSeed ^= mixedSeed >>> 16;
  mixedSeed = Math.imul(mixedSeed, FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_A);
  mixedSeed ^= mixedSeed >>> 13;
  mixedSeed = Math.imul(mixedSeed, FLAPPY_EVALUATION_SEED_MIX_MULTIPLIER_B);
  mixedSeed ^= mixedSeed >>> 16;
  return mixedSeed >>> 0;
}
