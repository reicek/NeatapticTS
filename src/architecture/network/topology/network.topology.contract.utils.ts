import type Network from '../../network/network';
import type {
  NetworkRuntimeProps,
  NetworkTopologyIntent,
  TopologyNetworkProps,
} from '../network.types';

type TopologyContractNetworkProps = TopologyNetworkProps & NetworkRuntimeProps;

/**
 * Minimal runtime surface needed to read the active feed-forward contract.
 *
 * Some callers have a full `Network` instance with `getTopologyIntent()`, while
 * others only hold a narrow runtime genome shape with the low-level acyclic flag.
 * This contract keeps both shapes usable from one small helper.
 */
export interface FeedForwardTopologyContractCarrier {
  /** Optional topology-intent accessor used by full Network instances. */
  getTopologyIntent?: () => NetworkTopologyIntent;
}

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
 * Check whether a runtime shape currently carries the feed-forward contract.
 *
 * The helper is intentionally conservative when callers are in a mismatched
 * transitional state: either an explicit `feed-forward` intent or a truthy
 * `_enforceAcyclic` flag is treated as a feed-forward contract. That keeps
 * mutation and crossover helpers from introducing recurrent structure into a
 * genome that still advertises acyclic semantics anywhere on its runtime seam.
 *
 * @param carrier Narrow runtime shape or full network instance.
 * @returns True when feed-forward semantics are currently enforced.
 */
export function hasFeedForwardTopologyContract(
  carrier: FeedForwardTopologyContractCarrier,
): boolean {
  const runtimeCarrier = carrier as Record<string, unknown>;
  const topologyIntent =
    typeof carrier.getTopologyIntent === 'function'
      ? carrier.getTopologyIntent()
      : (runtimeCarrier._topologyIntent as NetworkTopologyIntent | undefined);

  return topologyIntent === 'feed-forward' || runtimeCarrier._enforceAcyclic === true;
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
