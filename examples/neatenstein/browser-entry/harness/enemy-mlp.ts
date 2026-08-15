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

import {
  NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS,
  NEATENSTEIN_MLP_TOPOLOGY,
  NEATENSTEIN_MLP_VARIANT_COUNT,
} from './constants';
import { SNAPSHOT_KIND_MLP } from '../constants';
import {
  MLP_ALLOWED_MUTATION_TYPES,
  NEATENSTEIN_MLP_OUTPUT_LABELS,
  CHAMPION_SEED_PRIME,
  OUTPUT_PRECISION,
} from './enemy-mlp.constants';
import type {
  EnemyVariant,
  MlpSnapshot,
  Snapshot,
  CreateMlpEnemyPopulationOptions,
  MlpEnemyPopulation,
} from './types';
import { warmStartTemplate, warmStartWeights } from './enemy-warmstart';

/**
 * Options accepted by {@link createMlpEnemyPopulation}.
 *
 * @deprecated Import from `./types` instead. This re-export preserves the
 *   public API for existing consumers.
 */
export type { CreateMlpEnemyPopulationOptions } from './types';

/**
 * MLP enemy population returned by {@link createMlpEnemyPopulation}.
 *
 * @deprecated Import from `./types` instead. This re-export preserves the
 * public API for existing consumers.
 */
export type { MlpEnemyPopulation } from './types';

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
 * console.log(variant.weights.length); // 90
 * ```
 */
export function createMlpEnemyPopulation(
  options: CreateMlpEnemyPopulationOptions = {},
): MlpEnemyPopulation {
  const seed = options.seed ?? 0;
  const variants = createVariants(seed);
  let championSnapshot: MlpSnapshot = createSnapshot(variants[0].weights);

  return {
    kind: SNAPSHOT_KIND_MLP,
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
 * Guard that rejects structural mutation operators for MLP enemies.
 */
const MLP_ALLOWED_MUTATION_SET = new Set(MLP_ALLOWED_MUTATION_TYPES);

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
  return MLP_ALLOWED_MUTATION_SET.has(operator.type);
}

/**
 * Ordered output labels for the MLP enemy backend.
 *
 * The four outputs produced by {@link activateMlp} are mapped to:
 * move, strafe, turn, and fire.
 *
 * @deprecated Import `NEATENSTEIN_MLP_OUTPUT_LABELS` from `./enemy-mlp.constants`
 *   instead. This re-export preserves the public API.
 */
export { NEATENSTEIN_MLP_OUTPUT_LABELS } from './enemy-mlp.constants';

/**
 * Activate the fixed-topology MLP for a set of world inputs.
 *
 * The weight vector must include all connection weights followed by the
 * per-layer bias terms, in layer order. The default topology is
 * {@link NEATENSTEIN_MLP_TOPOLOGY} (6→6→4→4), which requires 90 values:
 * 76 connection weights plus 14 biases.
 *
 * @param weights - Flat weight vector (connections + biases).
 * @param inputs - Input vector matching the first layer size.
 * @param topology - Layer sizes; defaults to the fixed enemy topology.
 * @returns Float32Array of outputs for the final layer.
 *
 * @throws Error when `inputs` or `weights` do not match the topology.
 *
 * @example
 * ```ts
 * const out = activateMlp(
 *   new Float32Array(90),
 *   new Float32Array(6),
 * );
 * console.log(out.length); // 4
 * ```
 */
export function activateMlp(
  weights: Float32Array,
  inputs: Float32Array,
  topology: readonly number[] = NEATENSTEIN_MLP_TOPOLOGY,
): Float32Array {
  if (inputs.length !== topology[0]) {
    throw new Error(
      `MLP input size ${inputs.length} does not match topology input ${topology[0]}`,
    );
  }
  const expected = countParameters(topology);
  if (weights.length !== expected) {
    throw new Error(
      `MLP weight vector length ${weights.length} does not match expected ${expected}`,
    );
  }

  let activations = new Float32Array(inputs);
  let offset = 0;
  for (let layer = 1; layer < topology.length; layer++) {
    const inSize = topology[layer - 1];
    const outSize = topology[layer];
    const next = new Float32Array(outSize);
    for (let o = 0; o < outSize; o++) {
      let sum = 0;
      for (let i = 0; i < inSize; i++) {
        sum += activations[i] * weights[offset + o * inSize + i];
      }
      sum += weights[offset + inSize * outSize + o];
      next[o] = Math.tanh(sum);
    }
    offset += inSize * outSize + outSize;
    activations = next;
  }
  return activations;
}

/**
 * Map raw MLP outputs to a labelled action record.
 *
 * @param outputs - Raw output vector from {@link activateMlp}.
 * @param labels - Ordered output labels; defaults to
 *   {@link NEATENSTEIN_MLP_OUTPUT_LABELS}.
 * @returns Record keyed by label with the corresponding output value.
 *
 * @throws Error when `outputs` and `labels` have different lengths.
 *
 * @example
 * ```ts
 * const actions = interpretMlpOutputs(new Float32Array([0.1, 0.2, 0.3, 0.4]));
 * console.log(actions.move); // 0.1
 * ```
 */
export function interpretMlpOutputs(
  outputs: Float32Array,
  labels: readonly string[] = NEATENSTEIN_MLP_OUTPUT_LABELS,
): Record<string, number> {
  if (outputs.length !== labels.length) {
    throw new Error(
      `Output length ${outputs.length} does not match label count ${labels.length}`,
    );
  }
  const result: Record<string, number> = {};
  for (let i = 0; i < labels.length; i++) {
    result[labels[i]] = Number(Number(outputs[i]).toPrecision(OUTPUT_PRECISION));
  }
  return result;
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
      weights: warmStartWeights(seed, i),
    });
  }
  return variants;
}

/**
 * Generate the champion weight vector for a refresh generation.
 *
 * The champion is re-warm-started deterministically from a generation-unique
 * seed so that every refresh produces a fresh trained template.
 *
 * @param seed - Population seed.
 * @param generation - Generation at which the snapshot refreshes.
 * @returns A new deterministic champion weight vector.
 */
function createChampionWeights(seed: number, generation: number): Float32Array {
  return warmStartTemplate(seed + generation * CHAMPION_SEED_PRIME);
}

/**
 * Count the total number of parameters for a fully-connected feed-forward
 * topology with per-layer bias.
 *
 * For each adjacent pair of layers this includes every connection weight plus
 * one bias per output neuron.
 *
 * @param topology - Ordered layer sizes.
 * @returns Total parameter count (connection weights + biases).
 */
export function countParameters(topology: readonly number[]): number {
  let total = 0;
  for (let i = 0; i < topology.length - 1; i++) {
    total += topology[i] * topology[i + 1] + topology[i + 1];
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
    kind: SNAPSHOT_KIND_MLP,
    weights: new Float32Array(weights),
  };
}
