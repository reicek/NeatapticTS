import type Network from '../../network';
import type {
  DeterministicNetworkInternals as NetworkInternals,
  RNGSnapshot,
} from '../network.types';
export type { RNGSnapshot } from '../network.types';

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
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Normalize and persist unsigned 32-bit seed state.
  const normalizedSeedState = toUint32(seed);
  setInternalSeedState(networkInternal, normalizedSeedState);

  // Step 2: Install deterministic PRNG closure.
  setRandomFunction(
    networkInternal,
    createDeterministicRandomFunction(networkInternal),
  );

  /**
   * Convert a numeric value into unsigned 32-bit state.
   *
   * @param numericValue - Numeric value to normalize.
   * @returns Unsigned 32-bit representation.
   */
  function toUint32(numericValue: number): number {
    return numericValue >>> 0;
  }

  /**
   * Assign normalized seed state to internal RNG storage.
   *
   * @param internalState - Runtime network internals used by deterministic RNG pipeline.
   * @param normalizedState - Unsigned 32-bit state.
   * @returns Nothing.
   */
  function setInternalSeedState(
    internalState: NetworkInternals,
    normalizedState: number,
  ): void {
    internalState._rngState = normalizedState;
  }

  /**
   * Assign the active random function reference.
   *
   * @param internalState - Runtime network internals used by deterministic RNG pipeline.
   * @param randomFunction - PRNG function returning values in [0,1).
   * @returns Nothing.
   */
  function setRandomFunction(
    internalState: NetworkInternals,
    randomFunction: () => number,
  ): void {
    internalState._rand = randomFunction;
  }

  /**
   * Create deterministic PRNG function bound to provided internal state holder.
   *
   * @param internalState - Runtime network internals used by deterministic RNG pipeline.
   * @returns PRNG function returning values in [0,1).
   */
  function createDeterministicRandomFunction(
    internalState: NetworkInternals,
  ): () => number {
    return () => {
      // Step 1: Advance state by Weyl increment.
      internalState._rngState = advanceStateWithWeylIncrement(
        internalState._rngState,
      );

      // Step 2: Apply integer avalanche mixing.
      const mixedState = mixStateWord(internalState._rngState);

      // Step 3: Convert to normalized float in [0,1).
      return toUnitInterval(mixedState);
    };
  }

  /**
   * Advance state using a fixed Weyl increment with uint32 wraparound.
   *
   * @param currentState - Current state word (possibly undefined).
   * @returns Next unsigned 32-bit state.
   */
  function advanceStateWithWeylIncrement(
    currentState: number | undefined,
  ): number {
    return ((currentState ?? 0) + 0x6d2b79f5) >>> 0;
  }

  /**
   * Mix state word with xorshift and multiplication steps.
   *
   * @param stateWord - Unsigned 32-bit state word.
   * @returns Mixed unsigned 32-bit word.
   */
  function mixStateWord(stateWord: number): number {
    let mixedWord = Math.imul(stateWord ^ (stateWord >>> 15), 1 | stateWord);
    mixedWord ^=
      mixedWord + Math.imul(mixedWord ^ (mixedWord >>> 7), 61 | mixedWord);
    return (mixedWord ^ (mixedWord >>> 14)) >>> 0;
  }

  /**
   * Convert unsigned 32-bit word to float in [0,1).
   *
   * @param unsignedWord - Unsigned 32-bit word.
   * @returns Unit-interval floating-point value.
   */
  function toUnitInterval(unsignedWord: number): number {
    return unsignedWord / 4294967296;
  }
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
  const networkInternal = this as unknown as NetworkInternals;
  return {
    step: networkInternal._trainingStep,
    state: networkInternal._rngState,
  };
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
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Replace active random function.
  setRandomFunction(networkInternal, fn);

  // Step 2: Reset stored raw state because closure-backed function is now external.
  clearStoredRngState(networkInternal);

  /**
   * Assign active random function reference.
   *
   * @param internalState - Runtime network internals used by deterministic RNG pipeline.
   * @param randomFunction - PRNG function returning values in [0,1).
   * @returns Nothing.
   */
  function setRandomFunction(
    internalState: NetworkInternals,
    randomFunction: () => number,
  ): void {
    internalState._rand = randomFunction;
  }

  /**
   * Clear stored numeric RNG state.
   *
   * @param internalState - Runtime network internals used by deterministic RNG pipeline.
   * @returns Nothing.
   */
  function clearStoredRngState(internalState: NetworkInternals): void {
    internalState._rngState = undefined;
  }
}

/**
 * Get the current internal 32‑bit RNG state value.
 *
 * @param this - Bound {@link Network} instance.
 * @returns Unsigned 32‑bit state integer or undefined if generator not yet seeded or was reset.
 */
export function getRNGState(this: Network): number | undefined {
  const networkInternal = this as unknown as NetworkInternals;
  return networkInternal._rngState;
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
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Guard against non-numeric values.
  if (!isNumericState(state)) return;

  // Step 2: Persist normalized uint32 state.
  networkInternal._rngState = toUint32(state);

  /**
   * Check whether incoming state is numeric.
   *
   * @param candidateState - Candidate state value.
   * @returns True when state is numeric.
   */
  function isNumericState(candidateState: number): boolean {
    return typeof candidateState === 'number';
  }

  /**
   * Convert numeric state to unsigned 32-bit representation.
   *
   * @param numericState - Numeric state value.
   * @returns Unsigned 32-bit state.
   */
  function toUint32(numericState: number): number {
    return numericState >>> 0;
  }
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
  const networkInternal = this as unknown as NetworkInternals;
  return networkInternal._rand;
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
