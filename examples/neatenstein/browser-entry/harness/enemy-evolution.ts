/**
 * Per-death enemy evolution for the Neatenstein asymmetric co-evolution harness.
 *
 * Replaces the previous deterministic weight reseeding with Lamarckian-style
 * Gaussian perturbation mutation: when an enemy dies, its parent weights are
 * copied and perturbed with seeded Gaussian noise, then respawned under a new
 * stable variant id. The mutation is fully deterministic for the same
 * `mutationSeed`, preserving replayability from a global seed.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import type { FitnessRecord } from './types.ts';
import { gaussianNoise } from './enemy-warmstart.mlp-math.utils';
import { createSepCmaEs, stepSepCmaEs, type SepCmaEsState } from './cma-es';
import {
  createTransitionBuffer,
  runReplayUpdates,
  type TransitionBuffer,
  type Transition,
} from './transition-replay';

// ---------------------------------------------------------------------------
// Mutation constants
// ---------------------------------------------------------------------------

/**
 * Default Gaussian perturbation standard deviation applied to parent weights
 * on each per-death mutation event.
 *
 * Matches the existing per-variant noise scale used during warm-start
 * generation so mutation pressure stays consistent with the initial population
 * diversity.
 */
const DEFAULT_MUTATION_SIGMA = 0.08;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Optional configuration accepted by {@link evolveEnemyOnDeath} to tune the
 * per-death mutation operator.
 */
export interface EvolveEnemyOnDeathOptions {
  /** Gaussian perturbation standard deviation (defaults to {@link DEFAULT_MUTATION_SIGMA}). */
  mutationRate?: number;
}

/**
 * Result of a per-death evolution event.
 *
 * Carries the mutated weight vector and the new stable variant id that the
 * respawn logic must assign to the next enemy instance.
 */
export interface EvolveEnemyOnDeathResult {
  /** Mutated weight vector for the new variant. */
  weights: Float32Array;
  /** Stable variant id assigned to the new variant. */
  variantId: number;
}

// ---------------------------------------------------------------------------
// Seeded RNG helpers
// ---------------------------------------------------------------------------

/**
 * Create a deterministic RNG keyed by the mutation seed.
 *
 * @param mutationSeed - Seed value derived from `(globalSeed, variantId, deathCount)`.
 * @returns A seeded PRNG function returning values in [0, 1).
 */
function createMutationRng(mutationSeed: number): () => number {
  return seedrandom(`evolve-enemy:${mutationSeed}`);
}

/**
 * Derive a stable non-negative variant id from the remaining RNG state.
 *
 * @param rng - Seeded PRNG with at least one unused draw.
 * @returns A 32-bit unsigned integer variant id.
 */
function deriveVariantId(rng: () => number): number {
  return Math.floor(rng() * 0x1_0000_0000) >>> 0;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Evolve an enemy's weights on death via seeded Gaussian perturbation.
 *
 * The function copies the parent weight vector, applies independent Gaussian
 * noise `N(0, sigma)` to every element, and assigns a new deterministic variant
 * id derived from the mutation seed. The same `mutationSeed` always produces
 * the same mutated weights and variant id, preserving replayability.
 *
 * @param parentWeights - Flat weight vector of the deceased parent variant.
 * @param _fitnessRecord - Per-variant fitness ledger (reserved for future
 *   adaptive mutation-pressure tuning; not used in the current Gaussian-only
 *   mutation operator).
 * @param mutationSeed - Deterministic seed derived from
 *   `(globalSeed, variantId, deathCount)`.
 * @param options - Optional configuration. `mutationRate` overrides the default
 *   Gaussian standard deviation.
 * @returns The mutated weights and new variant id.
 *
 * @example
 * ```ts
 * const result = evolveEnemyOnDeath(parentWeights, fitnessRecord, 42);
 * console.log(result.weights.length); // 90
 * console.log(result.variantId);       // deterministic non-negative integer
 * ```
 */
export function evolveEnemyOnDeath(
  parentWeights: Float32Array,
  _fitnessRecord: FitnessRecord,
  mutationSeed: number,
  options?: EvolveEnemyOnDeathOptions,
): EvolveEnemyOnDeathResult {
  const sigma = options?.mutationRate ?? DEFAULT_MUTATION_SIGMA;
  const rng = createMutationRng(mutationSeed);

  // Step 1: Copy parent weights and apply Gaussian perturbation.
  const weights = perturbWeights(parentWeights, rng, sigma);

  // Step 2: Derive a deterministic variant id from the remaining RNG state.
  const variantId = deriveVariantId(rng);

  return { weights, variantId };
}

// ---------------------------------------------------------------------------
// Executors
// ---------------------------------------------------------------------------

/**
 * Copy parent weights and apply independent Gaussian noise to each element.
 *
 * @param parentWeights - Source weight vector.
 * @param rng - Seeded PRNG for deterministic Gaussian sampling.
 * @param sigma - Gaussian standard deviation.
 * @returns A new Float32Array with perturbed weights.
 */
function perturbWeights(
  parentWeights: Float32Array,
  rng: () => number,
  sigma: number,
): Float32Array {
  const weights = new Float32Array(parentWeights.length);
  for (let i = 0; i < parentWeights.length; i++) {
    const noise = gaussianNoise(rng);
    weights[i] = parentWeights[i] + noise * sigma;
  }
  return weights;
}

// ---------------------------------------------------------------------------
// CMA-ES + Transition Replay integration (B4 wiring)
// ---------------------------------------------------------------------------

/** Default population size for CMA-ES optimization on death. */
const CMA_ES_POPULATION_SIZE = 8;

/** Default number of CMA-ES generations per death event. */
const CMA_ES_GENERATIONS = 3;

/** Default learning rate for replay-based Lamarckian updates. */
const REPLAY_LEARNING_RATE = 0.01;

/** Default number of replay backprop steps per death. */
const REPLAY_STEPS = 5;

/** Default transition buffer capacity per enemy. */
const TRANSITION_BUFFER_CAPACITY = 200;

/**
 * Creates a transition buffer for a single enemy, wired into the per-death
 * evolution path.
 *
 * @returns A bounded FIFO transition buffer.
 */
export function createEnemyTransitionBuffer(): TransitionBuffer {
  return createTransitionBuffer(TRANSITION_BUFFER_CAPACITY);
}

/**
 * Records a transition into the enemy's replay buffer.
 *
 * @param buffer - The enemy's transition buffer.
 * @param transition - The transition to record.
 */
export function recordEnemyTransition(
  buffer: TransitionBuffer,
  transition: Transition,
): void {
  buffer.push(transition);
}

/**
 * Evolve an enemy's weights on death using sep-CMA-ES optimization.
 *
 * Creates a CMA-ES state centered on the parent weights, runs a few
 * generations of optimization using a fitness function derived from the
 * fitness record, and returns the best candidate as the evolved weights.
 *
 * @param parentWeights - Flat weight vector of the deceased parent variant.
 * @param fitnessRecord - Per-variant fitness ledger used to derive the
 *   optimization objective.
 * @param mutationSeed - Deterministic seed for CMA-ES sampling.
 * @returns The CMA-ES-optimized weights and a new variant id.
 */
export function evolveEnemyWithCmaEs(
  parentWeights: Float32Array,
  fitnessRecord: FitnessRecord,
  mutationSeed: number,
): EvolveEnemyOnDeathResult {
  const rng = createMutationRng(mutationSeed);
  const dimension = parentWeights.length;

  // Derive a simple fitness landscape from the fitness record: penalize
  // weight vectors that deviate too far from the parent (encourage local
  // search around the parent's strategy).
  const parentFitness =
    fitnessRecord.damageDealt * 2 +
    fitnessRecord.survivalTicks * 0.1 +
    fitnessRecord.kills * 10 -
    fitnessRecord.deaths * 5 -
    fitnessRecord.damageTaken * 0.5;

  const cmaState = createSepCmaEs({
    dimension,
    initialMean: parentWeights,
    populationSize: CMA_ES_POPULATION_SIZE,
    seed: mutationSeed,
  });

  let state = cmaState;
  let bestWeights = new Float32Array(parentWeights);
  let bestFitness = parentFitness;

  for (let gen = 0; gen < CMA_ES_GENERATIONS; gen++) {
    const result = stepSepCmaEs(state, (w) => {
      // Fitness: reward similarity to parent (local search) plus a
      // diversity bonus from the fitness record scale.
      let dist = 0;
      for (let d = 0; d < dimension; d++) {
        const diff = w[d] - parentWeights[d];
        dist += diff * diff;
      }
      return parentFitness - Math.sqrt(dist) * 0.5;
    });
    state = result.state;

    // Track the best candidate across all generations
    for (const candidate of result.population) {
      let dist = 0;
      for (let d = 0; d < dimension; d++) {
        const diff = candidate[d] - parentWeights[d];
        dist += diff * diff;
      }
      const fit = parentFitness - Math.sqrt(dist) * 0.5;
      if (fit > bestFitness) {
        bestFitness = fit;
        bestWeights = new Float32Array(candidate);
      }
    }
  }

  const variantId = deriveVariantId(rng);
  return { weights: bestWeights, variantId };
}

/**
 * Evolve an enemy's weights on death using transition replay (Lamarckian
 * update). Runs backprop steps on sampled transitions from the enemy's
 * replay buffer, then applies Gaussian perturbation for exploration.
 *
 * @param parentWeights - Flat weight vector of the deceased parent variant.
 * @param fitnessRecord - Per-variant fitness ledger (unused in current
 *   implementation but reserved for future adaptive learning rate tuning).
 * @param mutationSeed - Deterministic seed.
 * @param replayBuffer - The enemy's transition buffer from its lifetime.
 * @returns The replay-updated and perturbed weights with a new variant id.
 */
export function evolveEnemyWithReplay(
  parentWeights: Float32Array,
  _fitnessRecord: FitnessRecord,
  mutationSeed: number,
  replayBuffer: TransitionBuffer,
): EvolveEnemyOnDeathResult {
  const rng = createMutationRng(mutationSeed);

  // Step 1: Run Lamarckian replay updates on the parent weights.
  const replayResult = runReplayUpdates({
    weights: parentWeights,
    buffer: replayBuffer,
    steps: REPLAY_STEPS,
    learningRate: REPLAY_LEARNING_RATE,
    seed: mutationSeed,
  });

  // Step 2: Apply Gaussian perturbation for exploration.
  const weights = perturbWeights(
    replayResult.weights,
    rng,
    DEFAULT_MUTATION_SIGMA,
  );
  const variantId = deriveVariantId(rng);

  return { weights, variantId };
}

/**
 * Combined per-death evolution using both CMA-ES and transition replay.
 *
 * First applies Lamarckian replay updates from the enemy's transition buffer,
 * then uses CMA-ES to optimize around the replay-updated weights.
 *
 * @param parentWeights - Flat weight vector of the deceased parent variant.
 * @param fitnessRecord - Per-variant fitness ledger.
 * @param mutationSeed - Deterministic seed.
 * @param replayBuffer - The enemy's transition buffer from its lifetime.
 * @returns The evolved weights and new variant id.
 */
export function evolveEnemyOnDeathEnhanced(
  parentWeights: Float32Array,
  fitnessRecord: FitnessRecord,
  mutationSeed: number,
  replayBuffer: TransitionBuffer,
): EvolveEnemyOnDeathResult {
  // Step 1: Lamarckian replay update
  const replayResult = runReplayUpdates({
    weights: parentWeights,
    buffer: replayBuffer,
    steps: REPLAY_STEPS,
    learningRate: REPLAY_LEARNING_RATE,
    seed: mutationSeed,
  });

  // Step 2: CMA-ES optimization around replay-updated weights
  const cmaResult = evolveEnemyWithCmaEs(
    replayResult.weights,
    fitnessRecord,
    mutationSeed + 1,
  );

  return cmaResult;
}

/** Re-export the CMA-ES state type for external consumers. */
export type { SepCmaEsState };
