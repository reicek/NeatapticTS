/**
 * Weight-only MLP backend for the Neatenstein enemy population.
 *
 * This module materializes a small population of feed-forward enemy variants
 * that share the fixed topology declared in {@link NEATENSTEIN_MLP_TOPOLOGY}.
 * The population is evolved in place: every
 * {@link NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS} generations the champion
 * snapshot is regenerated deterministically from the population seed, while
 * intermediate generations reuse the previous snapshot unchanged.
 *
 * @module
 */

import seedrandom from 'seedrandom';

import {
  NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS,
  NEATENSTEIN_MLP_TOPOLOGY,
  NEATENSTEIN_MLP_VARIANT_COUNT,
} from './constants';
import type {
  EnemyPopulation,
  EnemyVariant,
  MlpSnapshot,
  Snapshot,
} from './types';

/**
 * Options accepted by {@link createMlpEnemyPopulation}.
 */
export interface CreateMlpEnemyPopulationOptions {
  /** Deterministic seed used to generate the initial variant weights. */
  seed?: number;
}

/**
 * Create a weight-only MLP enemy population.
 *
 * The returned population exposes the common {@link EnemyPopulation} contract
 * plus an {@link MlpEnemyPopulation.update | update} method used by the harness
 * to gate snapshot refreshes every 5 generations.
 *
 * @param options - Population configuration. Defaults to `seed: 0` so zero-
 *   argument calls still produce a deterministic population.
 * @returns An MLP enemy population with 32 weight-only variants.
 *
 * @example
 * ```ts
 * const population = createMlpEnemyPopulation({ seed: 7 });
 * const variant = population.sample(0) as EnemyVariant;
 * console.log(variant.weights.length); // 80
 * ```
 */
export function createMlpEnemyPopulation(
  options: CreateMlpEnemyPopulationOptions = {},
): MlpEnemyPopulation {
  const seed = options.seed ?? 0;
  const variants = createVariants(seed);
  let championSnapshot: MlpSnapshot = createSnapshot(variants[0].weights);

  return {
    kind: 'mlp',
    size: NEATENSTEIN_MLP_VARIANT_COUNT,
    sample(index: number): unknown {
      // Index-stable sampling: any index maps deterministically to a variant.
      const safeIndex =
        Number.isFinite(index) && index >= 0
          ? index % NEATENSTEIN_MLP_VARIANT_COUNT
          : 0;
      return variants[safeIndex] as unknown;
    },
    snapshot: () => championSnapshot,
    update: ({ generation }: { generation: number }): Snapshot => {
      if (generation % NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS === 0) {
        championSnapshot = createSnapshot(
          createChampionWeights(seed, generation),
        );
      }
      return championSnapshot;
    },
  };
}

/**
 * MLP enemy population returned by {@link createMlpEnemyPopulation}.
 */
export interface MlpEnemyPopulation extends EnemyPopulation {
  /**
   * Advance the population snapshot on refresh generations.
   *
   * @param context - Current generation context.
   * @returns The population snapshot. The same reference is returned when no
   *   refresh happens; a new reference is returned on refresh generations.
   */
  update: (context: { generation: number }) => Snapshot;
}

/**
 * Allowed weight-only mutation operator types for the MLP enemy backend.
 *
 * The MLP backend uses a fixed 8→6→4→2 topology, so structural operators such
 * as add-node or add-connection would corrupt the feed-forward shape. This
 * allowlist is the single source of truth for operator types that are safe to
 * apply to an MLP enemy.
 */
const MLP_ALLOWED_MUTATION_TYPES = new Set(['weight', 'weights', 'perturb']);

/**
 * Guard that rejects structural mutation operators for MLP enemies.
 *
 * MLP enemies evolve weights only on a fixed topology. Any operator that would
 * add, remove, or rewire nodes/connections must be rejected so the feed-forward
 * shape stays intact.
 *
 * @param operator - Mutation operator descriptor.
 * @returns `true` when the operator is a safe weight-only mutation, `false` when
 *   it would alter topology.
 *
 * @example
 * ```ts
 * guardMlpStructuralMutation({ type: 'weight' }); // true
 * guardMlpStructuralMutation({ type: 'add-node' }); // false
 * ```
 */
export function guardMlpStructuralMutation(operator: {
  type: string;
}): boolean {
  return MLP_ALLOWED_MUTATION_TYPES.has(operator.type);
}

/**
 * Build all 32 deterministic variants for the population seed.
 *
 * @param seed - Population seed.
 * @returns Ordered list of enemy variants.
 */
function createVariants(seed: number): EnemyVariant[] {
  const variants: EnemyVariant[] = [];
  for (let i = 0; i < NEATENSTEIN_MLP_VARIANT_COUNT; i++) {
    variants.push({
      id: i,
      weights: createVariantWeights(seed, i),
    });
  }
  return variants;
}

/**
 * Generate a fresh weight vector for one variant.
 *
 * Weights are sampled from a per-variant seeded PRNG so the same `seed` and
 * `variantId` always produce the same vector, while different ids are extremely
 * unlikely to collide.
 *
 * @param seed - Population seed.
 * @param variantId - Stable variant index.
 * @returns A new weight vector sized for the fixed MLP topology.
 */
function createVariantWeights(seed: number, variantId: number): Float32Array {
  const rng = seedrandom(`${seed}:variant:${variantId}`);
  const weightCount = countWeights(NEATENSTEIN_MLP_TOPOLOGY);
  const weights = new Float32Array(weightCount);
  for (let i = 0; i < weightCount; i++) {
    weights[i] = rng() * 2 - 1;
  }
  return weights;
}

/**
 * Generate the champion weight vector for a refresh generation.
 *
 * @param seed - Population seed.
 * @param generation - Generation at which the snapshot refreshes.
 * @returns A new deterministic champion weight vector.
 */
function createChampionWeights(seed: number, generation: number): Float32Array {
  const rng = seedrandom(`${seed}:refresh:${generation}`);
  const weightCount = countWeights(NEATENSTEIN_MLP_TOPOLOGY);
  const weights = new Float32Array(weightCount);
  for (let i = 0; i < weightCount; i++) {
    weights[i] = rng() * 2 - 1;
  }
  return weights;
}

/**
 * Count the total number of connection weights for a fully-connected feed-
 * forward topology.
 *
 * @param topology - Ordered layer sizes.
 * @returns Total weight count.
 */
function countWeights(topology: readonly number[]): number {
  let total = 0;
  for (let i = 0; i < topology.length - 1; i++) {
    total += topology[i] * topology[i + 1];
  }
  return total;
}

/**
 * Wrap a weight vector in the MLP snapshot shape.
 *
 * @param weights - Champion weight vector.
 * @returns A serializable MLP snapshot.
 */
function createSnapshot(weights: Float32Array): MlpSnapshot {
  return {
    kind: 'mlp',
    weights: new Float32Array(weights),
  };
}
