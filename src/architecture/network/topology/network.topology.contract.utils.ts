import type Network from '../../network/network';
import type {
  NetworkRuntimeProps,
  NetworkTopologyIntent,
  TopologyNetworkProps,
} from '../network.types';

type TopologyContractNetworkProps = TopologyNetworkProps & NetworkRuntimeProps;

/**
 * Read the public topology intent preserved on a network instance.
 *
 * This accessor keeps the semantic contract visible to callers even though the
 * lower-level runtime ultimately enforces acyclicity through booleans and cache
 * invalidation.
 *
 * @param this Target network instance.
 * @returns Current topology intent.
 */
export function getTopologyIntent(this: Network): NetworkTopologyIntent {
  return (
    (this as unknown as TopologyContractNetworkProps)._topologyIntent ??
    'unconstrained'
  );
}

/**
 * Set the public topology intent and synchronize low-level runtime flags.
 *
 * Updating the semantic contract also updates acyclic enforcement and marks the
 * topological cache dirty so later activation paths rebuild consistent state.
 *
 * @param this Target network instance.
 * @param topologyIntent Desired topology intent.
 * @returns Nothing.
 */
export function setTopologyIntent(
  this: Network,
  topologyIntent: NetworkTopologyIntent,
): void {
  const topologyContractNetwork =
    this as unknown as TopologyContractNetworkProps;

  // Step 1: Persist the public topology contract.
  topologyContractNetwork._topologyIntent = topologyIntent;

  // Step 2: Keep the low-level acyclic guard aligned with the public contract.
  topologyContractNetwork._enforceAcyclic = topologyIntent === 'feed-forward';

  // Step 3: Mark topology caches dirty so later activation rebuilds coherent state.
  topologyContractNetwork._topoDirty = true;
}

/**
 * Toggle low-level acyclic enforcement while preserving a coherent public contract.
 *
 * This exists for backward compatibility with callers that still use the legacy
 * boolean API instead of the semantic `topologyIntent` field.
 *
 * @param this Target network instance.
 * @param flag Whether to enforce acyclic connectivity.
 * @returns Nothing.
 */
export function setEnforceAcyclic(this: Network, flag: boolean): void {
  const topologyContractNetwork =
    this as unknown as TopologyContractNetworkProps;

  // Step 1: Preserve backward compatibility for callers using the legacy toggle.
  topologyContractNetwork._enforceAcyclic = !!flag;

  // Step 2: Keep the public topology intent synchronized with the active runtime contract.
  topologyContractNetwork._topologyIntent = flag
    ? 'feed-forward'
    : 'unconstrained';

  // Step 3: Mark topology caches dirty so acyclic mode changes rebuild ordering safely.
  topologyContractNetwork._topoDirty = true;
}
