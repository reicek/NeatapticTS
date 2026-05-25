import Network from '../../network/network';
import Connection from '../../connection';
import Node from '../../node';
import type {
  CompactPayloadContext,
  CompactSerializedNetworkTuple,
  NetworkInternals,
  NetworkInternalsWithDropout,
  NodeInternals,
  ResolvedNetworkSizeContext,
} from './network.serialize.utils.types';
import { DEFAULT_NUMERIC_VALUE } from './network.serialize.utils.types';

/**
 * Cast a public network instance to the internal serializer runtime shape so low-level deserialization helpers can access mutable fields without duplicating bridge-cast logic.
 *
 * @param network - Network instance.
 * @returns Runtime internals.
 * @remarks
 * This is a type-level adapter used inside serialization internals only.
 * It does not clone or validate the runtime object.
 */
export function asNetworkInternals(network: Network): NetworkInternals {
  return network as unknown as NetworkInternals;
}

/**
 * Cast a public network instance to serializer internals that include optional dropout metadata so verbose restore and export paths can read dropout fields consistently.
 *
 * @param network - Network instance.
 * @returns Runtime internals with optional dropout.
 * @remarks
 * This exists because dropout is optional in runtime but required in verbose JSON payloads.
 */
export function asNetworkInternalsWithDropout(
  network: Network,
): NetworkInternalsWithDropout {
  return network as unknown as NetworkInternalsWithDropout;
}

/**
 * Cast a node instance to its internal runtime representation so serializer helpers can read and write persisted node metadata through one shared bridge.
 *
 * @param node - Node instance.
 * @returns Node internals.
 * @remarks
 * This helper centralizes serializer-side node internals access.
 */
export function asNodeInternals(node: Node): NodeInternals {
  return node as unknown as NodeInternals;
}

/**
 * Normalizes a compact tuple payload into a named object context.
 *
 * This improves readability in orchestration code by replacing positional tuple access
 * with semantically named fields.
 *
 * @param data - Compact tuple payload.
 * @returns Normalized payload context.
 * @remarks
 * No validation is performed here; callers should ensure `data` follows compact tuple ordering.
 * @example
 * ```ts
 * const compactPayload = createCompactPayloadContext(compactTuple);
 * ```
 */
export function createCompactPayloadContext(
  data: CompactSerializedNetworkTuple,
): CompactPayloadContext {
  const [
    activations,
    states,
    squashes,
    connections,
    serializedInput,
    serializedOutput,
    nodeGeneIds,
    topologyIntent,
  ] = data;

  return {
    activations,
    states,
    squashes,
    connections,
    serializedInput,
    serializedOutput,
    nodeGeneIds,
    topologyIntent,
  };
}

/**
 * Resolve effective input and output dimensions using optional explicit overrides so compact payload restore paths can apply caller-provided sizes deterministically.
 *
 * When an override is provided, it takes precedence over serialized values.
 *
 * @param compactPayload - Compact payload context.
 * @param inputSizeOverride - Optional input override.
 * @param outputSizeOverride - Optional output override.
 * @returns Resolved network size context.
 * @remarks
 * Missing serialized dimensions are normalized by `resolveSizeOverride` to maintain deterministic defaults.
 */
export function resolveNetworkSize(
  compactPayload: CompactPayloadContext,
  inputSizeOverride?: number,
  outputSizeOverride?: number,
): ResolvedNetworkSizeContext {
  return {
    input: resolveSizeOverride(
      inputSizeOverride,
      compactPayload.serializedInput,
    ),
    output: resolveSizeOverride(
      outputSizeOverride,
      compactPayload.serializedOutput,
    ),
  };
}

/**
 * Resolve one size value with override-first semantics so deserialization can prioritize explicit caller intent while preserving serialized fallbacks when overrides are absent.
 *
 * @param overrideValue - Optional explicit override.
 * @param serializedValue - Serialized fallback value.
 * @returns Effective size.
 * @remarks
 * Returns `0` when both values are absent/invalid according to serializer defaults.
 */
export function resolveSizeOverride(
  overrideValue: number | undefined,
  serializedValue: number,
): number {
  if (typeof overrideValue === 'number') {
    return overrideValue;
  }
  return serializedValue;
}

/**
 * Create a fresh network instance for deserialize workflows so restoration code can hydrate graph state onto a clean runtime object.
 *
 * @param input - Input size.
 * @param output - Output size.
 * @returns New network instance.
 * @remarks
 * This helper does not execute serialized payload code; it only calls the local `Network` constructor.
 */
export function createNetworkInstance(input: number, output: number): Network {
  return new Network(input, output);
}

/**
 * Clear mutable runtime collections before reconstruction so node, connection, self-connection, and gate arrays are reset to a predictable empty baseline.
 *
 * @param networkInternals - Runtime internals.
 * @returns Nothing.
 * @remarks
 * Call before node/connection rebuild so deserialization starts from a clean state.
 */
export function resetMutableRuntimeCollections(
  networkInternals: NetworkInternals,
): void {
  networkInternals.nodes = [];
  networkInternals.connections = [];
  networkInternals.selfconns = [];
  networkInternals.gates = [];
}

/**
 * Check whether a candidate node index falls inside the valid array bounds so restore logic can reject malformed serialized endpoint references.
 *
 * @param nodes - Node list.
 * @param index - Candidate index.
 * @returns True when index is valid.
 * @remarks
 * Bounds are inclusive of `0` and exclusive of `nodes.length`.
 */
export function isNodeIndexInBounds(nodes: Node[], index: number): boolean {
  return index >= DEFAULT_NUMERIC_VALUE && index < nodes.length;
}

/**
 * Check whether a candidate index value is finite so endpoint validation can reject NaN and infinite references before bounds checks execute.
 *
 * @param index - Candidate index value.
 * @returns True when finite number.
 * @remarks
 * Use with bounds checks when validating serialized endpoint references.
 */
export function isFiniteIndex(index: number): boolean {
  return Number.isFinite(index);
}

/**
 * Writes a restored node gene id when serialized identity data is available.
 *
 * Compact restore paths construct fresh runtime nodes first, then overwrite the
 * temporary constructor-assigned ids with persisted historical ids.
 *
 * @param node - Restored runtime node.
 * @param geneId - Persisted stable gene id.
 * @returns Nothing.
 */
export function hydrateNodeGeneIdWhenProvided(
  node: Node,
  geneId: number | null | undefined,
): void {
  if (typeof geneId !== 'number') {
    return;
  }

  asNodeInternals(node).geneId = geneId;
}

/**
 * Restores persisted connection identity onto a freshly created runtime connection.
 *
 * Import paths still build edges through `connect()` so graph bookkeeping stays
 * centralized. This helper then reapplies the persisted innovation and enabled state.
 *
 * @param createdConnection - Newly created runtime connection.
 * @param identity - Persisted identity metadata.
 * @returns Nothing.
 */
export function applyRestoredConnectionIdentity(
  createdConnection: Connection | undefined,
  identity: { innovation?: number; enabled?: boolean },
): void {
  if (!createdConnection) {
    return;
  }

  if (typeof identity.innovation === 'number') {
    createdConnection.innovation = identity.innovation;
  }

  if (typeof identity.enabled === 'boolean') {
    (createdConnection as Connection & { enabled?: boolean }).enabled =
      identity.enabled;
  }
}

/**
 * Advances static node and connection counters past all restored historical ids.
 *
 * Without this step, a fresh process could deserialize a high-id genome and then
 * allocate colliding `geneId` or `innovation` values on the next mutation.
 *
 * @param networkInternals - Restored mutable network internals.
 * @returns Nothing.
 */
export function syncRestoredHistoricalCounters(
  networkInternals: NetworkInternals,
): void {
  const maxObservedGeneId = networkInternals.nodes.reduce(
    (currentMaxGeneId, nodeReference) => {
      return Math.max(currentMaxGeneId, asNodeInternals(nodeReference).geneId!);
    },
    DEFAULT_NUMERIC_VALUE,
  );
  const maxObservedInnovation = networkInternals.connections
    .concat(networkInternals.selfconns)
    .reduce(
      (currentMaxInnovation, connectionReference) =>
        Math.max(currentMaxInnovation, connectionReference.innovation),
      DEFAULT_NUMERIC_VALUE,
    );

  Node.syncGeneIdCounter(maxObservedGeneId);
  Connection.syncInnovationCounter(maxObservedInnovation);
}
