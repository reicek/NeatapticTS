import {
  exportRngState,
  importRngState,
  restoreRngState,
  sampleRandomSequence,
  snapshotRngState,
  type RngHost,
} from '../rng';

/**
 * Public RNG maintenance facade helpers for the stable `Neat` entrypoint.
 *
 * These helpers expose deterministic replay and lightweight random sampling
 * without forcing the main [src/neat.ts](src/neat.ts) facade to carry the
 * lower-level RNG state plumbing inline. Grouping them here keeps the public
 * class orchestration-focused while preserving the same debugging and replay
 * affordances that tests and examples already use.
 *
 * Invariant: this boundary only manages deterministic RNG state progression
 * for replay and diagnostics. It does not change evolution policy, mutation
 * behavior, or selection semantics.
 */

/**
 * Narrow `Neat` host surface required by the public RNG facade.
 *
 * This stays intentionally small because the facade only forwards state
 * snapshot, restore, export, and sampling calls into the existing RNG helpers.
 */
export interface NeatRngFacadeHost extends RngHost {}

/**
 * Return the current opaque RNG numeric state used by the instance.
 *
 * This is mainly useful for deterministic replay, test fixtures, and bug
 * reports that need to resume the same random subsequence.
 *
 * @param host - `Neat` instance exposing RNG state.
 * @returns Numeric RNG state or `undefined` when the RNG has not been initialized.
 */
export function snapshotRNGState(host: NeatRngFacadeHost): number | undefined {
  return snapshotRngState(host);
}

/**
 * Restore a previously captured RNG state.
 *
 * The next RNG access will continue from the restored subsequence, which makes
 * it useful for replaying evolution bugs or comparing deterministic traces.
 *
 * @param host - `Neat` instance exposing RNG state.
 * @param state - Numeric or string seed snapshot to restore.
 * @returns Nothing. The helper mutates the host RNG state in place.
 */
export function restoreRNGState(
  host: NeatRngFacadeHost,
  state: number | string | undefined,
): void {
  restoreRngState(host, state);
}

/**
 * Import an RNG state using the legacy compatibility name.
 *
 * This mirrors `restoreRNGState` so older callers can keep using the stable
 * public method name while the implementation stays centralized.
 *
 * @param host - `Neat` instance exposing RNG state.
 * @param state - Numeric or string seed snapshot to restore.
 * @returns Nothing. The helper mutates the host RNG state in place.
 */
export function importRNGState(
  host: NeatRngFacadeHost,
  state: number | string | undefined,
): void {
  importRngState(host, state);
}

/**
 * Export the current RNG state for persistence or debugging.
 *
 * @param host - `Neat` instance exposing RNG state.
 * @returns Numeric RNG state or `undefined` when the RNG has not been initialized.
 */
export function exportRNGState(host: NeatRngFacadeHost): number | undefined {
  return exportRngState(host);
}

/**
 * Produce deterministic random samples from the instance RNG.
 *
 * This read-only helper is useful for tests and diagnostics that want to prove
 * two runs share the same random subsequence after restoring a snapshot.
 *
 * @param host - `Neat` instance exposing RNG state.
 * @param sampleCount - Number of random values to generate.
 * @returns Deterministic random samples in the range `[0, 1)`.
 *
 * @example
 * ```ts
 * const snapshot = neat.snapshotRNGState();
 * const firstSequence = neat.sampleRandom(3);
 * neat.restoreRNGState(snapshot);
 * const repeatedSequence = neat.sampleRandom(3);
 * console.log(firstSequence, repeatedSequence);
 * ```
 */
export function sampleRandom(
  host: NeatRngFacadeHost,
  sampleCount: number,
): number[] {
  return sampleRandomSequence(host, sampleCount);
}
