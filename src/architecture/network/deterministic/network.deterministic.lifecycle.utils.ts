import type Network from '../../network/network';
import type {
  NetworkInternals,
  RNGSnapshot,
} from './network.deterministic.utils.types';

/**
 * Capture a snapshot of the RNG state together with the network's training step.
 *
 * @param this - Bound Network instance.
 * @returns Object containing current training step and 32-bit RNG state.
 */
export function snapshotRNG(this: Network): RNGSnapshot {
  const networkInternal = this as unknown as NetworkInternals;
  return {
    step: networkInternal._trainingStep,
    state: networkInternal._rngState,
  };
}

/**
 * Restore a previously captured RNG function implementation and clear stored numeric state.
 *
 * @param this - Bound Network instance.
 * @param fn - Function returning a pseudo-random number in [0,1).
 * @returns Nothing.
 */
export function restoreRNG(this: Network, fn: () => number): void {
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Replace active random function.
  setRandomFunction(networkInternal, fn);

  // Step 2: Reset stored raw state because closure-backed function is now external.
  clearStoredRngState(networkInternal);
}

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
