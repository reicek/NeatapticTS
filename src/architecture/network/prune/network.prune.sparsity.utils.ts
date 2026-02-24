import type Network from '../../network';
import type { NetworkPruningProps } from '../network.types';

/**
 * Read baseline used for sparsity reporting.
 * @param currentNetwork - Network to inspect.
 * @returns Baseline connection count when available.
 */
export function readInitialSparsityBaseline(
  currentNetwork: Network,
): number | undefined {
  return (currentNetwork as unknown as NetworkPruningProps)
    ._initialConnectionCount;
}

/**
 * Convert current density into sparsity ratio.
 * @param currentConnectionCount - Current connection count.
 * @param baselineConnectionCount - Baseline connection count.
 * @returns Sparsity ratio in [0,1] for valid baselines.
 */
export function calculateSparsityFromBaseline(
  currentConnectionCount: number,
  baselineConnectionCount: number,
): number {
  return 1 - currentConnectionCount / baselineConnectionCount;
}
