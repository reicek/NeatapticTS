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
