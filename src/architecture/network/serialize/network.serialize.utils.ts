import type Network from '../../network/network';
import Connection from '../../connection';
import type {
  CompactSerializedNetworkTuple,
  NetworkJSON,
} from './network.serialize.utils.types';
export type { SerializedConnection } from '../network.types';
import type { NetworkArchitectureDescriptor } from '../network.types';
import { describeArchitecture } from '../topology/network.topology.architecture.utils';
import {
  collectNodeActivations,
  collectNodeSquashKeys,
  collectNodeStates,
  collectSerializedConnections,
  rebuildConnectionsFromCompactPayload,
  rebuildNodesFromCompactPayload,
  refreshNodeIndices,
} from './network.serialize.compact.utils';
import {
  appendJsonForwardConnections,
  appendJsonNodesAndSelfConnections,
  createEmptyNetworkJson,
  rebuildConnectionsFromJsonPayload,
  rebuildNodesFromJsonPayload,
  resolveDropout,
  validateNetworkJsonOrThrow,
  warnWhenJsonFormatVersionIsUnknown,
} from './network.serialize.json.utils';
import {
  asNetworkInternals,
  asNetworkInternalsWithDropout,
  createCompactPayloadContext,
  createNetworkInstance,
  resetMutableRuntimeCollections,
  resolveNetworkSize,
} from './network.serialize.runtime.utils';

/**
 * Serializes a network instance into the compact tuple format.
 *
 * Use this format when payload size and serialization speed matter more than readability.
 * The tuple layout is positional and optimized for transport/storage efficiency.
 *
 * @param this - Bound network instance.
 * @returns Compact tuple payload containing activations, states, squash keys, connections, and input/output sizes.
 * @remarks
 * The tuple is deterministic for a fixed runtime node ordering because indices are refreshed before collection.
 * This compact format is intentionally lossy with respect to object-level metadata outside the captured fields.
 * Prefer `toJSONImpl` for human-readable payloads or long-lived interoperability.
 *
 * Time complexity is $O(N + C)$ where $N$ is node count and $C$ is connection count.
 *
 * Compact payloads are plain data and are never evaluated as code, but they still cross a trust boundary.
 * Validate and sanitize data before calling `deserialize` when payload origin is untrusted.
 * @example
 * ```ts
 * import Network from '../../network';
 * import { deserialize, serialize } from './network.serialize.utils';
 *
 * const sourceNetwork = new Network(2, 1);
 * const compactTuple = serialize.call(sourceNetwork);
 * const rebuiltNetwork = deserialize(compactTuple);
 * ```
 */
export function serialize(this: Network): CompactSerializedNetworkTuple {
  const networkInternals = asNetworkInternals(this);

  // Step 1: Refresh canonical node indices used by serialized references.
  refreshNodeIndices(networkInternals.nodes);

  // Step 2: Collect per-node scalar arrays aligned by index.
  const activations = collectNodeActivations(networkInternals.nodes);
  const states = collectNodeStates(networkInternals.nodes);
  const squashes = collectNodeSquashKeys(networkInternals.nodes);

  // Step 3: Flatten all connection groups into compact connection records.
  const serializedConnections = collectSerializedConnections(networkInternals);

  // Step 4: Return compact tuple with IO shape metadata.
  return [
    activations,
    states,
    squashes,
    serializedConnections,
    networkInternals.input,
    networkInternals.output,
  ];
}

/**
 * Rebuilds a network instance from compact tuple form.
 *
 * Use this importer for compact payloads produced by `serialize`.
 * Optional `inputSize` and `outputSize` let callers enforce shape overrides at import time.
 *
 * @param data - Compact tuple payload.
 * @param inputSize - Optional input-size override that takes precedence over serialized input.
 * @param outputSize - Optional output-size override that takes precedence over serialized output.
 * @returns Reconstructed network instance.
 * @remarks
 * Endpoint and gater indices that are out of bounds are skipped with warnings so import can continue.
 * Unknown activation keys fall back to identity to keep reconstruction deterministic and non-throwing.
 *
 * Compact payloads do not carry an explicit format version. For cross-version persistence contracts,
 * prefer `toJSONImpl` and `fromJSONImpl`.
 *
 * Treat incoming payloads as untrusted input unless provenance is guaranteed.
 * @throws TypeError When `data` is not iterable tuple-like input at runtime.
 * @example
 * ```ts
 * import { deserialize } from './network.serialize.utils';
 *
 * const rebuiltNetwork = deserialize(compactTuple, 2, 1);
 * ```
 */
export const deserialize = (
  data: CompactSerializedNetworkTuple,
  inputSize?: number,
  outputSize?: number,
): Network => {
  // Step 1: Normalize tuple payload and resolve effective network IO dimensions.
  const compactPayload = createCompactPayloadContext(data);
  const resolvedSize = resolveNetworkSize(
    compactPayload,
    inputSize,
    outputSize,
  );

  // Step 2: Create empty network runtime and reset mutable collections.
  const rebuiltNetwork = createNetworkInstance(
    resolvedSize.input,
    resolvedSize.output,
  );
  const networkInternals = asNetworkInternals(rebuiltNetwork);
  resetMutableRuntimeCollections(networkInternals);

  // Step 3: Rebuild nodes and their runtime scalar fields.
  rebuildNodesFromCompactPayload(networkInternals, {
    activations: compactPayload.activations,
    states: compactPayload.states,
    squashes: compactPayload.squashes,
    input: resolvedSize.input,
    output: resolvedSize.output,
  });

  // Step 4: Rebuild all compact connections and restore gating links.
  rebuildConnectionsFromCompactPayload({
    networkInternals,
    serializedConnections: compactPayload.connections,
  });

  return rebuiltNetwork;
};

/**
 * Serializes a network instance into the verbose JSON format.
 *
 * Use this format when you need human-readable snapshots, explicit schema versioning,
 * and better forward/backward compatibility handling.
 *
 * @param this - Bound network instance.
 * @returns Versioned JSON payload with shape metadata, nodes, and connections.
 * @remarks
 * The payload includes `formatVersion` so readers can detect schema drift and apply migration logic.
 * Node indices are re-canonicalized before connection export, producing deterministic endpoint mapping.
 *
 * Compared to compact tuples, JSON is larger but easier to inspect, diff, and store in versioned artifacts.
 *
 * Serialization is data-only and does not include executable code.
 * @example
 * ```ts
 * import Network from '../../network';
 * import { fromJSONImpl, toJSONImpl } from './network.serialize.utils';
 *
 * const sourceNetwork = new Network(3, 1);
 * const snapshotJson = toJSONImpl.call(sourceNetwork);
 * const rebuiltNetwork = fromJSONImpl(snapshotJson);
 * ```
 */
export function toJSONImpl(this: Network): NetworkJSON {
  const networkInternals = asNetworkInternalsWithDropout(this);

  // Step 1: Create baseline JSON shell.
  const networkJson = createEmptyNetworkJson(networkInternals);

  // Step 2: Rebuild node list and append self-connections.
  appendJsonNodesAndSelfConnections(networkInternals, networkJson);

  // Step 3: Append non-self connections.
  appendJsonForwardConnections(networkInternals, networkJson);

  // Step 4: Attach optional architecture metadata for diagnostics consumers.
  networkJson.architecture = describeArchitecture(this);

  return networkJson;
}

/**
 * Reconstructs a network instance from the verbose JSON payload.
 *
 * This importer validates payload shape, restores dropout and topology, and then rebuilds
 * connections, gating relationships, and optional enabled flags.
 *
 * @param json - Verbose JSON payload.
 * @returns Reconstructed network instance.
 * @remarks
 * Unknown `formatVersion` values emit a warning and import still proceeds on a best-effort basis.
 * Invalid connection rows or gater indices are skipped with warnings to preserve recoverability.
 *
 * Use this path for versioned persistence contracts and interop with tools that expect readable JSON.
 *
 * Security note: do not deserialize untrusted blobs without an application-level validation policy.
 * @throws Error When `json` root is missing or not an object.
 * @example
 * ```ts
 * import { fromJSONImpl } from './network.serialize.utils';
 *
 * const rebuiltNetwork = fromJSONImpl(snapshotJson);
 * ```
 */
export const fromJSONImpl = (json: NetworkJSON): Network => {
  // Step 1: Validate payload contract and report unknown versions.
  validateNetworkJsonOrThrow(json);
  warnWhenJsonFormatVersionIsUnknown(json.formatVersion);

  // Step 2: Create empty runtime network and reset mutable collections.
  const rebuiltNetwork = createNetworkInstance(json.input, json.output);
  const networkInternals = asNetworkInternalsWithDropout(rebuiltNetwork);
  networkInternals.dropout = resolveDropout(json.dropout);
  resetMutableRuntimeCollections(networkInternals);

  // Step 3: Rebuild nodes from JSON entries.
  rebuildNodesFromJsonPayload({
    networkInternals,
    nodeJsonEntries: json.nodes,
  });

  // Step 4: Rebuild JSON connections, gating links, and enabled flags.
  rebuildConnectionsFromJsonPayload({
    networkInternals,
    connectionJsonEntries: json.connections,
  });

  // Step 5: Restore the public topology contract when the payload carries one.
  if (json.topologyIntent) {
    rebuiltNetwork.setTopologyIntent(json.topologyIntent);
  }

  // Step 6: Hydrate optional architecture metadata when valid.
  applyHydratedArchitectureDescriptor(rebuiltNetwork, json.architecture);

  return rebuiltNetwork;
};

/**
 * Applies hydrated architecture metadata to runtime network when shape is valid.
 *
 * @param network - Rebuilt network instance.
 * @param architectureDescriptor - Optional serialized descriptor.
 * @returns Nothing.
 */
function applyHydratedArchitectureDescriptor(
  network: Network,
  architectureDescriptor: NetworkArchitectureDescriptor | undefined,
): void {
  if (!isArchitectureDescriptorShapeValid(architectureDescriptor)) {
    return;
  }

  const runtimeNetwork = network as unknown as {
    _serializedArchitectureDescriptor?: NetworkArchitectureDescriptor;
  };
  runtimeNetwork._serializedArchitectureDescriptor = architectureDescriptor;
}

/**
 * @param architectureDescriptor - Optional descriptor candidate.
 * @returns True when minimal descriptor shape is valid.
 */
function isArchitectureDescriptorShapeValid(
  architectureDescriptor: NetworkArchitectureDescriptor | undefined,
): architectureDescriptor is NetworkArchitectureDescriptor {
  if (!architectureDescriptor || typeof architectureDescriptor !== 'object') {
    return false;
  }

  if (!Array.isArray(architectureDescriptor.hiddenLayerSizes)) {
    return false;
  }

  const hasValidSource =
    architectureDescriptor.source === 'layer-metadata' ||
    architectureDescriptor.source === 'graph-topology' ||
    architectureDescriptor.source === 'inferred';

  return hasValidSource;
}

/**
 * Re-exported connection constructor used by tooling that needs to inspect connection internals.
 */
export { Connection };
