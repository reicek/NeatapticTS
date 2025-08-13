import type Network from '../network';

/**
 * Network statistics accessors.
 *
 * Currently exposes a single helper for retrieving the most recent regularization / stochasticity
 * metrics snapshot recorded during training or evaluation. The internal `_lastStats` field (on the
 * Network instance, typed as any) is expected to be populated elsewhere in the training loop with
 * values such as:
 *  - l1Penalty, l2Penalty
 *  - dropoutApplied (fraction of units dropped last pass)
 *  - weightNoiseStd (effective std dev used if noise injected)
 *  - sparsityRatio, prunedConnections
 *  - any custom user extensions (object is not strictly typed to allow experimentation)
 *
 * Design decision: We return a deep copy to prevent external mutation of internal accounting state.
 * If the object is large and copying becomes a bottleneck, future versions could offer a freeze
 * option or incremental diff interface.
 */

/**
 * Deep clone utility with a resilient fallback strategy.
 *
 * Priority order:
 *  1. Use native structuredClone when available (handles typed arrays, dates, etc.).
 *  2. Fallback to JSON serialize/deserialize (sufficient for plain data objects).
 *  3. If serialization fails (rare circular or unsupported types), a second JSON attempt is made
 *     inside the catch to avoid throwing and to preserve backwards compatibility (will still throw
 *     if fundamentally non-serializable).
 *
 * NOTE: This is intentionally minimal; for richer cloning semantics consider a dedicated utility.
 */
function deepCloneValue<T>(value: T): T {
  try {
    return (globalThis as any).structuredClone
      ? (globalThis as any).structuredClone(value)
      : JSON.parse(JSON.stringify(value));
  } catch {
    // Fallback: attempt JSON path again; if it fails this will throw—acceptable for edge cases.
    return JSON.parse(JSON.stringify(value));
  }
}

/**
 * Obtain the last recorded regularization / stochastic statistics snapshot.
 *
 * Returns a defensive deep copy so callers can inspect metrics without risking mutation of the
 * internal `_lastStats` object maintained by the training loop (e.g., during pruning, dropout, or
 * noise scheduling updates).
 *
 * @returns A deep-cloned stats object or null if no stats have been recorded yet.
 */
export function getRegularizationStats(this: Network) {
  /** Raw internal stats reference (may be undefined if never set). */
  const lastStatsSnapshot = (this as any)._lastStats;
  return lastStatsSnapshot ? deepCloneValue(lastStatsSnapshot) : null;
}

export default { getRegularizationStats };
