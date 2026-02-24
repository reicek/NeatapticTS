import type {
  DeterministicNetworkInternals as DeterministicNetworkInternalsType,
  RNGSnapshot,
} from '../network.types';

export type { RNGSnapshot };

/**
 * Internal deterministic network state shape used across deterministic utility modules.
 */
export type NetworkInternals = DeterministicNetworkInternalsType;

/**
 * Fixed Weyl increment used to advance deterministic PRNG state.
 */
export const RNG_WEYL_INCREMENT = 0x6d2b79f5;

/**
 * Divisor used to normalize uint32 PRNG output into [0, 1).
 */
export const UINT32_NORMALIZER = 4_294_967_296;
