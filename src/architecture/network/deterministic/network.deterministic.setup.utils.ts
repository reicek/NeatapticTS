import type Network from '../../network/network';
import {
  RNG_WEYL_INCREMENT,
  UINT32_NORMALIZER,
  type NetworkInternals,
} from './network.deterministic.utils.types';

/**
 * Seed the internal PRNG and install a deterministic random() implementation on the Network instance.
 *
 * @param this - Bound Network instance.
 * @param seed - A finite number; only its lower 32 bits are used.
 * @returns Nothing.
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
}

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
  return ((currentState ?? 0) + RNG_WEYL_INCREMENT) >>> 0;
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
  return unsignedWord / UINT32_NORMALIZER;
}
