import type Network from '../../network/network';
import type { NetworkInternals } from './network.deterministic.utils.types';

/**
 * Get the current internal 32-bit RNG state value.
 *
 * @param this - Bound Network instance.
 * @returns Unsigned 32-bit state integer or undefined if generator not yet seeded or was reset.
 */
export function getRNGState(this: Network): number | undefined {
  const networkInternal = this as unknown as NetworkInternals;
  return networkInternal._rngState;
}

/**
 * Explicitly set (override) the internal 32-bit RNG state without changing the generator function.
 *
 * @param this - Bound Network instance.
 * @param state - Any finite number (only low 32 bits used). Ignored if not numeric.
 * @returns Nothing.
 */
export function setRNGState(this: Network, state: number): void {
  const networkInternal = this as unknown as NetworkInternals;

  // Step 1: Guard against non-numeric values.
  if (!isNumericState(state)) return;

  // Step 2: Persist normalized uint32 state.
  networkInternal._rngState = toUint32(state);
}

/**
 * Retrieve the active random function reference.
 *
 * @param this - Bound Network instance.
 * @returns Function producing numbers in [0,1). May be undefined if never seeded.
 */
export function getRandomFn(this: Network): (() => number) | undefined {
  const networkInternal = this as unknown as NetworkInternals;
  return networkInternal._rand;
}

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
