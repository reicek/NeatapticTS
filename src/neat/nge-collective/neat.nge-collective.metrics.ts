import { safeStructuredClone } from '../../utils/safeStructuredClone';
import type {
  OpponentSnapshot,
  OpponentSnapshotPool,
} from './neat.nge-collective.types';

export type { OpponentSnapshot, OpponentSnapshotPool };

/**
 * @module neat.nge-collective.metrics
 *
 * Observability metrics and rolling tournament infrastructure for NGE Phase G.
 *
 * This module provides two independent observability primitives:
 *
 * ### Role-divergence metric
 *
 * `computeRoleDivergenceMetric` computes the L1 (Manhattan) distance between two
 * module-size distributions. In a collective of agents grown from identical DNA,
 * non-zero divergence indicates spontaneous role differentiation — foragers evolving
 * larger sensory modules while defenders grow larger motor modules, for example.
 * A score of `0` means structurally indistinguishable compositions.
 *
 * ### Opponent snapshot pool
 *
 * `createOpponentSnapshotPool` and `addOpponentSnapshot` implement a fixed-capacity
 * rolling circular buffer of deep-cloned opponent payloads. This supports rolling
 * tournament evaluation: new generations are tested against a diverse history of past
 * opponents rather than only the current generation's best.
 *
 * Pool invariants:
 * - Snapshots are ordered oldest-first in `pool.snapshots`.
 * - When the pool is at capacity, the oldest snapshot is evicted before the new one is added.
 * - Each payload is deep-cloned at registration time via `safeStructuredClone`; the original
 *   object may be mutated freely after the call without affecting the stored snapshot.
 *
 * ```mermaid
 * graph LR
 *   A["addOpponentSnapshot(pool, id, payload, tick)"]
 *   B["safeStructuredClone(payload)"]
 *   C["OpponentSnapshot\n{ agentId, snapshot, frozenAt }"]
 *   D["pool.snapshots (oldest → newest)"]
 *   E["evict oldest if at capacity"]
 *   A --> B --> C --> D
 *   D -->|"length > capacity"| E
 * ```
 */

/**
 * Computes a role-divergence metric between two module-size distributions using
 * the L1 (Manhattan) distance — the sum of absolute per-slot differences.
 *
 * A value of `0` indicates identical distributions. Larger values indicate greater
 * structural divergence between the two agents' module compositions.
 *
 * @param distributionA - Ordered module-size counts for agent A.
 * @param distributionB - Ordered module-size counts for agent B.
 * @returns Non-negative divergence score (`0` when distributions are equal).
 *
 * @example
 * ```ts
 * computeRoleDivergenceMetric([10, 5], [10, 5]); // 0
 * computeRoleDivergenceMetric([20, 0], [0, 20]); // 40
 * ```
 */
export function computeRoleDivergenceMetric(
  distributionA: number[],
  distributionB: number[],
): number {
  // Step 1: Walk over the union of both distributions and accumulate absolute differences.
  const maxLength = Math.max(distributionA.length, distributionB.length);
  let divergence = 0;

  for (let index = 0; index < maxLength; index++) {
    divergence += Math.abs(
      (distributionA[index] ?? 0) - (distributionB[index] ?? 0),
    );
  }

  return divergence;
}

/**
 * Creates an empty opponent snapshot pool with the given capacity.
 *
 * The pool is a rolling circular buffer: when at capacity, the oldest snapshot is
 * evicted in FIFO order to make room for each new addition.
 *
 * @param capacity - Maximum number of snapshots retained at any time.
 * @returns A new `OpponentSnapshotPool` with an empty snapshot list.
 *
 * @example
 * ```ts
 * const pool = createOpponentSnapshotPool(5);
 * // pool.capacity === 5, pool.snapshots.length === 0
 * ```
 */
export function createOpponentSnapshotPool(
  capacity: number,
): OpponentSnapshotPool {
  return { capacity, snapshots: [] };
}

/**
 * Adds a new snapshot to the opponent pool, evicting the oldest if the pool is at capacity.
 *
 * The payload is deep-cloned via `safeStructuredClone` at registration time so that
 * post-registration mutations to the original object are **not** reflected in the stored snapshot.
 *
 * Returns a **new** `OpponentSnapshotPool`; the original pool is **not** mutated.
 *
 * @param pool - Current snapshot pool.
 * @param agentId - Stable identifier for the agent being snapshotted.
 * @param payload - Arbitrary agent state to freeze at registration time.
 * @param frozenAt - Generation tick at which this snapshot is registered.
 * @returns A new pool containing the added snapshot, rotated if the pool was at capacity.
 *
 * @example
 * ```ts
 * const updated = addOpponentSnapshot(pool, 'agent:alpha', { fitness: 42 }, 3);
 * // updated.snapshots[0].agentId === 'agent:alpha'
 * // updated.snapshots[0].frozenAt === 3
 * ```
 */
export function addOpponentSnapshot(
  pool: OpponentSnapshotPool,
  agentId: string,
  payload: Record<string, unknown>,
  frozenAt: number,
): OpponentSnapshotPool {
  // Step 1: Deep-clone the payload to enforce snapshot immutability.
  const frozenSnapshot: OpponentSnapshot = {
    agentId,
    snapshot: safeStructuredClone(payload) as Readonly<Record<string, unknown>>,
    frozenAt,
  };

  // Step 2: Append the new snapshot to the ordered list.
  const extendedSnapshots = [...pool.snapshots, frozenSnapshot];

  // Step 3: Rotate out the oldest snapshot if the pool has exceeded capacity.
  const trimmedSnapshots =
    extendedSnapshots.length > pool.capacity
      ? extendedSnapshots.slice(extendedSnapshots.length - pool.capacity)
      : extendedSnapshots;

  return { capacity: pool.capacity, snapshots: trimmedSnapshots };
}
