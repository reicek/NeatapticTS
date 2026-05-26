/**
 * Deterministic RNG chapter for replayable `Network` behavior.
 *
 * This folder keeps random state explicit so stochastic network behavior can be
 * repeated instead of merely hoped for. The same graph may use randomness for
 * dropout, stochastic depth, weight noise, training-time perturbations, or
 * mutation-adjacent helpers. Deterministic utilities give those flows a common
 * seed, checkpoint, and restore surface.
 *
 * The important design split is between setup, state access, and lifecycle
 * capture. `setSeed()` installs a reproducible random stream. `getRNGState()`
 * and `setRNGState()` expose lightweight numeric checkpoints for pause-resume
 * workflows. `snapshotRNG()` and `restoreRNG()` support the slightly richer
 * lifecycle case where external tooling wants to branch or reconstruct the
 * active random function more deliberately.
 *
 * This chapter matters because deterministic behavior is only useful when it is
 * applied consistently. A seed is not enough if callers cannot checkpoint the
 * current stream, resume it later, or understand which helper owns the active
 * random function. By keeping those responsibilities together, the network
 * surface stays reproducible across tests, debugging sessions, and long-running
 * experiments.
 *
 * Another useful lens is to think in terms of branching timelines. Training or
 * evolution often needs to ask, "what happens if I continue from exactly this
 * stochastic point but change one later decision?" The deterministic helpers in
 * this folder are the bridge that makes that question answerable instead of
 * approximate.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Seed[setSeed]:::accent --> RandomOps[stochastic network operations]:::base
 *   RandomOps --> State[getRNGState checkpoint]:::base
 *   RandomOps --> Snapshot[snapshotRNG lifecycle snapshot]:::base
 *   State --> Resume[setRNGState resume]:::accent
 *   Snapshot --> Restore[restoreRNG external restore]:::accent
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   DeterministicChapter[deterministic/]:::accent --> Setup[setup utils<br/>install seed-backed RNG]:::base
 *   DeterministicChapter --> State[state utils<br/>read and write numeric state]:::base
 *   DeterministicChapter --> Lifecycle[lifecycle utils<br/>snapshot and restore]:::base
 * ```
 *
 * For background on why this chapter can stay reproducible with a small state
 * surface, see Wikipedia contributors,
 * [Pseudorandom number generator](https://en.wikipedia.org/wiki/Pseudorandom_number_generator).
 * The exported helpers here are about managing one deterministic stream, not
 * about generating cryptographic randomness.
 *
 * Example: initialize one reproducible random stream and checkpoint its numeric
 * state for later reuse.
 *
 * ```ts
 * network.setSeed(42);
 * const checkpoint = network.getRNGState();
 *
 * if (checkpoint !== undefined) {
 *   network.setRNGState(checkpoint);
 * }
 * ```
 *
 * Example: capture a portable snapshot before branching into a debugging or
 * replay workflow.
 *
 * ```ts
 * network.setSeed(7);
 * const snapshot = network.snapshotRNG();
 *
 * console.log(snapshot.state, snapshot.step);
 * ```
 *
 * Practical reading order:
 *
 * 1. Start here for the public deterministic helpers and their intended use.
 * 2. Continue into `network.deterministic.setup.utils.ts` for seed-backed RNG
 *    installation.
 * 3. Continue into `network.deterministic.state.utils.ts` for numeric
 *    checkpoint accessors.
 * 4. Finish in `network.deterministic.lifecycle.utils.ts` when you want the
 *    richer snapshot and restore plumbing.
 */

import type Network from '../../network/network';
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
 * - Use this before training, mutation, or stochastic operations when you need repeatable runs.
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
 * Default export bundle for the deterministic RNG utilities chapter.
 *
 * Bundles the seed, checkpoint, and restore helpers so the network facade can
 * bind them as methods without importing each function individually.
 */
const networkDeterministicUtils = {
  setSeed,
  snapshotRNG,
  restoreRNG,
  getRNGState,
  setRNGState,
  getRandomFn,
};
export default networkDeterministicUtils;
