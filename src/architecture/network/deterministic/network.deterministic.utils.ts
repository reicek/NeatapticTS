import type Network from '../../network';
import type { RNGSnapshot } from './network.deterministic.utils.types';
import { setSeed as setupSeed } from './network.deterministic.setup.utils';
import {
  snapshotRNG as lifecycleSnapshotRng,
  restoreRNG as lifecycleRestoreRng,
} from './network.deterministic.lifecycle.utils';
import {
  getRNGState as stateGetRngState,
  setRNGState as stateSetRngState,
  getRandomFn as stateGetRandomFunction,
} from './network.deterministic.state.utils';

export type { RNGSnapshot } from './network.deterministic.utils.types';

/**
 * Sets deterministic randomness for a network by installing a seed-backed RNG.
 *
 * Overview:
 * - Use this before training, mutation, or any stochastic operation when you need repeatable runs.
 * - The same seed and operation order produce the same random sequence and reproducible outcomes.
 * - This method delegates to setup utilities so behavior stays centralized across deterministic APIs.
 *
 * @param this - Bound network instance whose RNG state is being initialized.
 * @param seed - Seed value used to derive deterministic RNG state (low 32 bits are applied).
 * @returns Nothing.
 *
 * @example
 * ```ts
 * network.setSeed(42);
 * ```
 */
export function setSeed(this: Network, seed: number): void {
  // Step 1: Delegate deterministic seed setup to setup helpers.
  setupSeed.call(this, seed);
}

/**
 * Captures the current deterministic RNG lifecycle state as a portable snapshot.
 *
 * Overview:
 * - Use this before temporary experiments, branching simulations, or stateful debug sessions.
 * - The snapshot preserves enough information to resume from the same deterministic point later.
 * - This is useful when comparing alternate algorithm branches from an identical random timeline.
 *
 * @param this - Bound network instance whose RNG lifecycle state is captured.
 * @returns Snapshot containing deterministic progress metadata and RNG state payload.
 *
 * @example
 * ```ts
 * const snapshot = network.snapshotRNG();
 * ```
 */
export function snapshotRNG(this: Network): RNGSnapshot {
  // Step 1: Delegate lifecycle snapshot flow.
  return lifecycleSnapshotRng.call(this);
}

/**
 * Restores deterministic RNG lifecycle behavior from a provided RNG function.
 *
 * Overview:
 * - Use this when replaying deterministic flows after custom serialization, hydration, or test setup.
 * - The restored RNG function becomes the active random source used by the network lifecycle helpers.
 * - This keeps deterministic plumbing explicit when external code owns RNG reconstruction.
 *
 * @param this - Bound network instance receiving the restored RNG lifecycle function.
 * @param fn - Deterministic RNG function to install (expected to return values in `[0, 1)`).
 * @returns Nothing.
 *
 * @example
 * ```ts
 * network.restoreRNG(restoredRandomFunction);
 * ```
 */
export function restoreRNG(this: Network, fn: () => number): void {
  // Step 1: Delegate lifecycle restore flow.
  lifecycleRestoreRng.call(this, fn);
}

/**
 * Returns the current deterministic RNG numeric state, when available.
 *
 * Overview:
 * - Use this for lightweight checkpointing when full lifecycle snapshots are unnecessary.
 * - The value can be persisted and later reapplied through `setRNGState`.
 * - This is commonly used by tests that assert deterministic continuity across operations.
 *
 * @param this - Bound network instance queried for deterministic RNG numeric state.
 * @returns Numeric RNG state value, or `undefined` when no deterministic state exists yet.
 *
 * @example
 * ```ts
 * const state = network.getRNGState();
 * ```
 */
export function getRNGState(this: Network): number | undefined {
  // Step 1: Delegate state accessor flow.
  return stateGetRngState.call(this);
}

/**
 * Applies a deterministic RNG numeric state to continue from a known checkpoint.
 *
 * Overview:
 * - Pair this with `getRNGState` to pause/resume deterministic sequences.
 * - Useful for reproducible tests, multi-stage training workflows, and deterministic replay.
 * - Delegation keeps the write path consistent with the rest of deterministic state utilities.
 *
 * @param this - Bound network instance receiving deterministic RNG state.
 * @param state - Numeric RNG state checkpoint to install.
 * @returns Nothing.
 *
 * @example
 * ```ts
 * network.setRNGState(savedState);
 * ```
 */
export function setRNGState(this: Network, state: number): void {
  // Step 1: Delegate state mutator flow.
  stateSetRngState.call(this, state);
}

/**
 * Returns the active deterministic RNG function currently attached to the network.
 *
 * Overview:
 * - Use this when tooling or diagnostics need direct RNG access.
 * - Returning the function allows advanced integration code to inspect or reuse the random stream.
 * - For most persistence workflows, prefer `snapshotRNG` and `getRNGState` over direct function plumbing.
 *
 * @param this - Bound network instance queried for active RNG function.
 * @returns Active RNG function, or `undefined` when deterministic RNG is not initialized.
 *
 * @example
 * ```ts
 * const randomFn = network.getRandomFn();
 * ```
 */
export function getRandomFn(this: Network): (() => number) | undefined {
  // Step 1: Delegate function accessor flow.
  return stateGetRandomFunction.call(this);
}

/**
 * Default export bundle for convenient named imports.
 */
export default {
  setSeed,
  snapshotRNG,
  restoreRNG,
  getRNGState,
  setRNGState,
  getRandomFn,
};
