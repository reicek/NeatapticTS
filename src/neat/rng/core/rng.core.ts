/**
 * Deterministic RNG primitives used by NEAT replay and diagnostics.
 *
 * This chapter collects the low-level pieces: the host contract, the xorshift
 * constants, and the helpers that create, restore, and sample a reproducible
 * random stream.
 */
export type { RngHost } from './rng.types';
export {
  getOrCreateRng,
  snapshotRngState,
  restoreRngState,
  importRngState,
  exportRngState,
  sampleRandomSequence,
} from './rng.utils';
