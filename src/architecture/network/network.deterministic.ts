import type Network from '../network';

/**
 * Deterministic pseudo‑random number generation (PRNG) utilities for {@link Network}.
 *
 * Why this module exists:
 *  - Facilitates reproducible evolutionary runs / gradient training by allowing explicit seeding.
 *  - Centralizes RNG state management & snapshot/restore operations (useful for rollbacks or
 *    deterministic tests around mutation sequences).
 *  - Keeps the core Network class focused by extracting ancillary RNG concerns.
 *
 * Implementation notes:
 *  - Uses a small, fast 32‑bit xorshift / mix style generator (same semantics as the legacy inline version)
 *    combining an additive Weyl sequence step plus a few avalanche-style integer mixes.
 *  - Not cryptographically secure. Do not use for security / fairness sensitive applications.
 *  - Produces floating point numbers in [0,1) with 2^32 (~4.29e9) discrete possible mantissa states.
 *
 * Public surface:
 *  - {@link setSeed}: Initialize deterministic generator with a numeric seed.
 *  - {@link snapshotRNG}: Capture current training step + raw internal RNG state.
 *  - {@link restoreRNG}: Provide an externally saved RNG function (advanced) & clear stored state.
 *  - {@link getRNGState} / {@link setRNGState}: Low-level accessors for the internal 32‑bit state word.
 *  - {@link getRandomFn}: Retrieve the active random() function reference (primarily for tests / tooling).
 *
 * Design rationale:
 *  - Storing both a state integer (_rngState) and a function (_rand) allows hot-swapping alternative
 *    RNG implementations (e.g., for benchmarking or pluggable randomness strategies) without rewriting
 *    callsites inside Network algorithms.
 *
 * @module network.deterministic
 */

/** Shape of an RNG snapshot object. */
export interface RNGSnapshot {
  step: number | undefined;
  state: number | undefined;
}

/**
 * Seed the internal PRNG and install a deterministic random() implementation on the Network instance.
 *
 * Process:
 *  1. Coerce the provided seed to an unsigned 32‑bit integer (>>> 0) for predictable wraparound behavior.
 *  2. Define an inline closure that advances an internal 32‑bit state using:
 *       a. A Weyl increment (adding constant 0x6D2B79F5 each call) ensuring full-period traversal of
 *          the 32‑bit space when combined with mixing.
 *       b. Two rounds of xorshift / integer mixing (xor, shifts, multiplications) to decorrelate bits.
 *       c. Normalization to [0,1) by dividing the final 32‑bit unsigned integer by 2^32.
 *
 * Bit-mixing explanation (rough intuition):
 *  - XOR with shifted versions spreads high-order entropy to lower bits.
 *  - Multiplication (Math.imul) with carefully chosen odd constants introduces non-linear mixing.
 *  - The final right shift & xor avalanche aims to reduce sequential correlation.
 *
 * @param this - Bound {@link Network} instance.
 * @param seed - Any finite number; only its lower 32 bits are used.
 * @example
 * net.setSeed(1234);
 * const a = net.getRandomFn()(); // deterministic given the seed
 * net.setSeed(1234);
 * const b = net.getRandomFn()(); // a === b
 */
export function setSeed(this: Network, seed: number): void {
  // Store 32-bit unsigned state (bitwise ops in JS operate on signed 32-bit but we keep consistency via >>> 0).
  (this as any)._rngState = seed >>> 0;
  // Install PRNG closure referencing _rngState by name for mutation on each invocation.
  (this as any)._rand = () => {
    // Add Weyl constant (chosen odd constant) & coerce to uint32 wraparound.
    (this as any)._rngState = ((this as any)._rngState + 0x6d2b79f5) >>> 0;
    // First mix: xor with shifted self and multiply (Math.imul preserves 32-bit overflow semantics).
    let r = Math.imul(
      (this as any)._rngState ^ ((this as any)._rngState >>> 15),
      1 | (this as any)._rngState
    );
    // Second mix: avalanche style bit diffusion.
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    // Final xor/shift; convert to unsigned, then scale to [0,1).
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296; // 2^32
  };
}

/**
 * Capture a snapshot of the RNG state together with the network's training step.
 *
 * Useful for implementing speculative evolutionary mutations where you may revert both the
 * structural change and the randomness timeline if accepting/rejecting a candidate.
 *
 * @param this - Bound {@link Network} instance.
 * @returns Object containing current training step & 32‑bit RNG state (both possibly undefined if unseeded).
 * @example
 * const snap = net.snapshotRNG();
 * // ... perform operations
 * net.setRNGState(snap.state!);
 */
export function snapshotRNG(this: Network): RNGSnapshot {
  return { step: (this as any)._trainingStep, state: (this as any)._rngState };
}

/**
 * Restore a previously captured RNG function implementation (advanced usage).
 *
 * This does NOT rehydrate _rngState (it explicitly sets it to undefined). Intended for scenarios
 * where a caller has customly serialized a full RNG closure or wants to inject a deterministic stub.
 * If you only need to restore the raw state word produced by {@link snapshotRNG}, prefer
 * {@link setRNGState} instead.
 *
 * @param this - Bound {@link Network} instance.
 * @param fn - Function returning a pseudo‑random number in [0,1). Caller guarantees determinism if required.
 * @example
 * const original = net.getRandomFn();
 * net.restoreRNG(() => 0.5); // force constant RNG for a test
 * // ... test invariants ...
 * net.restoreRNG(original); // restore
 */
export function restoreRNG(this: Network, fn: () => number): void {
  (this as any)._rand = fn;
  (this as any)._rngState = undefined;
}

/**
 * Get the current internal 32‑bit RNG state value.
 *
 * @param this - Bound {@link Network} instance.
 * @returns Unsigned 32‑bit state integer or undefined if generator not yet seeded or was reset.
 */
export function getRNGState(this: Network): number | undefined {
  return (this as any)._rngState as number | undefined;
}

/**
 * Explicitly set (override) the internal 32‑bit RNG state without changing the generator function.
 *
 * This is a low‑level operation; typical clients should call {@link setSeed}. Provided for advanced
 * replay functionality where the same PRNG algorithm is assumed but you want to resume exactly at a
 * known state word.
 *
 * @param this - Bound {@link Network} instance.
 * @param state - Any finite number (only low 32 bits used). Ignored if not numeric.
 */
export function setRNGState(this: Network, state: number): void {
  if (typeof state === 'number') (this as any)._rngState = state >>> 0;
}

/**
 * Retrieve the active random function reference (for testing, instrumentation, or swapping).
 *
 * Mutating the returned function's closure variables (if any) is not recommended; prefer using
 * higher-level APIs (setSeed / restoreRNG) to manage state.
 *
 * @param this - Bound {@link Network} instance.
 * @returns Function producing numbers in [0,1). May be undefined if never seeded (call setSeed first).
 */
export function getRandomFn(this: Network): (() => number) | undefined {
  return (this as any)._rand as () => number;
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
