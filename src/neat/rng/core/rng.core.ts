/**
 * Deterministic RNG primitives used by NEAT replay and diagnostics.
 *
 * This barrel is the compact reading map for the RNG core layer. Together, the
 * files re-exported here answer one question: how does a NEAT run make random
 * choices without giving up deterministic replay?
 *
 * The story is intentionally split into three shelves:
 *
 * 1. `rng.types.ts` defines the tiny host seam that replay depends on,
 * 2. `rng.utils.ts` turns that seam into a live stream, checkpoint, and
 *    restore flow,
 * 3. `rng.constants.ts` names the fixed xorshift and seed-guarding choices so
 *    the algorithm is inspectable instead of magical.
 *
 * Read the chapter in that order when you want the whole lifecycle: host
 * state, stream creation, snapshot or export, then later restore.
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
