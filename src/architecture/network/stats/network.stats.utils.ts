import type Network from '../../network/network';
/**
 * Contract for testNetwork.
 */
export { testNetwork } from './network.stats.test.utils';
import { safeStructuredClone } from '../../../utils/safeStructuredClone';
import type { StatsNetworkProps as NetworkStatsProps } from '../network.types';

/**
 * Network statistics accessors.
 *
 * Currently exposes a single helper for retrieving the most recent regularization / stochasticity
 * metrics snapshot recorded during training or evaluation. The internal `_lastStats` field on the
 * Network instance is read through the local `NetworkStatsProps` bridge and is expected to be
 * populated elsewhere in the training loop with
 * values such as:
 *  - l1Penalty, l2Penalty
 *  - dropoutApplied (fraction of units dropped last pass)
 *  - weightNoiseStd (effective std dev used if noise injected)
 *  - sparsityRatio, prunedConnections
 *  - custom user extensions (the object stays intentionally open for experimentation)
 *
 * Design decision: We return a deep copy to prevent external mutation of internal accounting state.
 * If the object is large and copying becomes a bottleneck, future versions could offer a freeze
 * option or incremental diff interface.
 */

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
  const lastStatsSnapshot = (this as unknown as NetworkStatsProps)._lastStats;
  return lastStatsSnapshot ? safeStructuredClone(lastStatsSnapshot) : null;
}

/**
 * Default export bundle for the network statistics utilities chapter.
 *
 * Bundles getRegularizationStats so the network facade can bind it as a method
 * without importing it individually.
 */
const networkStatsUtils = { getRegularizationStats };
export default networkStatsUtils;
