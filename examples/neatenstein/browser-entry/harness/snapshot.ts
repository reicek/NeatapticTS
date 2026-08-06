/**
 * Rolling opponent snapshot pool for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * The harness keeps two enemy backends (MLP and SWARM) in tension with the
 * main NEAT agent. To prevent overfitting to a single opponent, each backend
 * refreshes its champion snapshot on a deterministic cadence:
 *
 * - MLP enemy snapshots refresh every
 *   {@link NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS} generations.
 * - SWARM enemy snapshots refresh every
 *   {@link NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS} generations.
 *
 * These refresh gates are pure functions of the current generation so the
 * cadence is deterministic and replay-safe from a seed.
 *
 * @module
 */

import {
  NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS,
  NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS,
} from './constants.ts';
import type { EnemyPopulation, MlpSnapshot } from './types.ts';

/**
 * Rolling store of frozen enemy snapshots indexed by variant id.
 *
 * Snapshots are deep-copied from the live population when
 * {@link refreshEnemySnapshots} is called and are never mutated afterwards.
 * This keeps evaluation barriers isolated from in-place population evolution.
 */
const enemySnapshotStore = new Map<number, MlpSnapshot>();

/**
 * Refresh the rolling enemy snapshot store from a live population.
 *
 * Each variant in the population is deep-copied into a frozen snapshot so
 * subsequent mutations of the population weights do not leak into stored
 * snapshots. Only the MLP backend is supported; other backends leave the
 * store empty.
 *
 * @param population - Live enemy population to snapshot.
 *
 * @example
 * ```ts
 * const population = createMlpEnemyPopulation({ seed: 1 });
 * refreshEnemySnapshots(population);
 * const snapshot = getEnemySnapshot(0);
 * console.log(snapshot.weights.length); // 90
 * ```
 */
export function refreshEnemySnapshots(population: EnemyPopulation): void {
  enemySnapshotStore.clear();

  if (population.kind !== 'mlp') {
    return;
  }

  for (let variantId = 0; variantId < population.size; variantId++) {
    const variant = population.sample(variantId) as {
      weights: Float32Array;
    };
    const frozenWeights = Object.freeze(Array.from(variant.weights));
    const snapshot: MlpSnapshot = {
      kind: 'mlp',
      weights: frozenWeights as unknown as Float32Array,
    };
    enemySnapshotStore.set(variantId, Object.freeze(snapshot));
  }
}

/**
 * Return the frozen snapshot for a previously refreshed enemy variant.
 *
 * @param variantId - Variant index within the enemy population.
 * @returns Frozen {@link MlpSnapshot} for the variant.
 * @throws Error when no snapshot has been refreshed for the variant.
 *
 * The returned `weights` are a flat `Float32Array` sized for the fixed MLP
 * topology ({@link NEATENSTEIN_MLP_TOPOLOGY} = 90 params for [6,6,4,4]). They
 * can be passed directly to {@link activateMlp} without materializing an
 * `INetwork` — the rollout in `enemy-runner.ts` relies on this direct usage.
 *
 * @example
 * ```ts
 * refreshEnemySnapshots(population);
 * const snapshot = getEnemySnapshot(0);
 * ```
 */
export function getEnemySnapshot(variantId: number): MlpSnapshot {
  const snapshot = enemySnapshotStore.get(variantId);
  if (!snapshot) {
    throw new Error(
      `No enemy snapshot for variant ${variantId}; call refreshEnemySnapshots first.`,
    );
  }
  return snapshot;
}

/**
 * Decide whether the MLP enemy population should produce a new champion snapshot
 * for the current generation.
 *
 * The MLP backend evolves weights on a fixed topology and changes more slowly
 * than the SWARM backend, so its snapshot is refreshed every 5 generations.
 * Generation 0 is treated as the initial refresh boundary.
 *
 * @param generation - Current co-evolution generation (non-negative integer).
 * @returns `true` when the MLP snapshot should be refreshed.
 *
 * @example
 * ```ts
 * shouldRefreshMlpSnapshot(4); // false
 * shouldRefreshMlpSnapshot(5); // true
 * shouldRefreshMlpSnapshot(10); // true
 * ```
 */
export function shouldRefreshMlpSnapshot(generation: number): boolean {
  return (
    generation >= 0 &&
    generation % NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS === 0
  );
}

/**
 * Decide whether the SWARM enemy population should produce a new champion
 * snapshot for the current generation.
 *
 * The SWARM backend is a fast-moving WeightSharedCohort that can shift
 * behaviour quickly, so its snapshot is refreshed every 3 generations.
 * Generation 0 is treated as the initial refresh boundary.
 *
 * @param generation - Current co-evolution generation (non-negative integer).
 * @returns `true` when the SWARM snapshot should be refreshed.
 *
 * @example
 * ```ts
 * shouldRefreshSwarmSnapshot(2); // false
 * shouldRefreshSwarmSnapshot(3); // true
 * shouldRefreshSwarmSnapshot(6); // true
 * ```
 */
export function shouldRefreshSwarmSnapshot(generation: number): boolean {
  return (
    generation >= 0 &&
    generation % NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS === 0
  );
}
