import type Network from '../../network/network';
import Connection from '../../connection';
import type Node from '../../node';
import type {
  CompressedSerializedNetworkArchive,
  CompressedSerializedNetworkArchiveOptions,
  CompressedSerializedNetwork,
  CompactSerializedNetworkTuple,
  NetworkJSON,
  ParameterLayoutEntry,
  ParameterLayoutV1,
  ParameterVector,
} from './network.serialize.utils.types';
export type {
  ParameterLayoutEntry,
  ParameterLayoutV1,
  ParameterVector,
} from './network.serialize.utils.types';
export type { SerializedConnection } from '../network.types';
import type {
  NetworkArchitectureDescriptor,
  NetworkJSONExtensions,
} from '../network.types';
import { synchronizeTemporalDescriptorExtensions } from '../network.temporal.extensions.utils';
import { describeArchitecture } from '../topology/network.topology.architecture.utils';
import {
  collectNodeActivations,
  collectNodeGeneIds,
  collectNodeSquashKeys,
  collectNodeStates,
  collectSerializedConnections,
  rebuildConnectionsFromCompactPayload,
  rebuildNodesFromCompactPayload,
  refreshNodeIndices,
} from './network.serialize.compact.utils';
import {
  compressSerializedConnections,
  COMPRESSED_NETWORK_FORMAT,
  createCompressedArchiveDecodeMetrics,
  createCompressedArchiveEncodeMetrics,
  type CompressedArchiveDecodeOptions,
  type CompressedArchiveDecodeResult,
  type CompressedArchiveEncodeResult,
  createCompressedNetworkArchive,
  createCompressedNetworkArchiveAsync,
  decodeArchivePayloadBase64,
  decompressSerializedConnections,
  estimateSerializedByteLength,
  parseCompressedNetworkArchive,
  parseCompressedNetworkArchiveAsync,
} from './network.serialize.compression.utils';
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
  asNodeInternals,
  asNetworkInternalsWithDropout,
  createCompactPayloadContext,
  createNetworkInstance,
  resetMutableRuntimeCollections,
  resolveNetworkSize,
  syncRestoredHistoricalCounters,
} from './network.serialize.runtime.utils';

type RuntimeNetworkWithSerializedExtensions = Network & {
  _serializedExtensions?: NetworkJSONExtensions;
};

type ParameterLayoutBiasDescriptor = {
  kind: 'bias';
  nodeId: number;
};

type ParameterLayoutWeightDescriptor = {
  kind: 'weight';
  from: number;
  innovation?: number;
  to: number;
};

type ParameterLayoutWeightOrderingIdentity = {
  descriptor: ParameterLayoutWeightDescriptor;
  fromGeneId: number;
  innovation: number | null;
  orderingIdentityKey: string;
  toGeneId: number;
};

type ParameterRuntimeBinding = {
  descriptor: ParameterLayoutEntry;
  readValue: () => number;
  writeValue: (parameterValue: number) => void;
};

type ParameterRuntimeContext = {
  layout: ParameterLayoutV1;
  runtimeBindings: ParameterRuntimeBinding[];
};

const NEUTRAL_CONNECTION_GAIN = 1;
const NEUTRAL_NODE_RESPONSE = 1;

/**
 * Build the deterministic descriptor order for parameter-layout version `1`.
 *
 * Version `1` folds all bias entries before all weight entries. Biases are
 * ordered by stable node gene id. Weights prefer connection innovation when it
 * exists and fall back to stable endpoint gene ids when it does not.
 * Ambiguous or missing ordering identity fails explicitly instead of inheriting
 * incidental runtime array order. For a fixed topology with stable historical
 * identities, the result is ordered deterministic on the same runtime.
 * This helper describes layout order only; it is not a cross-runtime exact
 * replay promise.
 *
 * @param network - Live network instance to inspect.
 * @returns Versioned ordered layout descriptors for the current topology.
 * @example
 * ```ts
 * const layout = createParameterLayoutV1(network);
 * console.log(layout.entries[0]);
 * ```
 */
export function createParameterLayoutV1(network: Network): ParameterLayoutV1 {
  const networkInternals = asNetworkInternals(network);

  // Step 1: Resolve bias descriptors from stable node gene ids.
  const biasEntries = collectParameterLayoutBiasEntries(networkInternals);

  // Step 2: Resolve weight descriptors from explicit stable edge identity.
  const weightEntries = collectParameterLayoutWeightEntries(networkInternals);

  // Step 3: Return the stable versioned layout with the documented fold order.
  return {
    version: 1,
    entries: [...biasEntries, ...weightEntries],
  };
}

/**
 * Export one versioned parameter vector from a live network runtime.
 *
 * Version `1` keeps ordered `ParameterLayoutV1` metadata and the aligned bias
 * and weight scalars together so same-runtime imports can verify compatibility
 * before mutating another network. The payload includes every live node bias
 * and every live forward-connection or self-connection weight in layout order,
 * including disabled connections when they still exist in the runtime graph.
 * Weight slots still prefer innovation-backed descriptors and fall back to
 * stable endpoint gene ids when a live connection has no innovation id.
 * Export rejects non-neutral `node.response` and `connection.gain` explicitly
 * instead of flattening those deferred families into the weights-and-biases v1
 * payload.
 *
 * @param network - Live network instance to export.
 * @returns Ordered layout metadata plus aligned scalar values for same-runtime deterministic roundtrips.
 */
export function toParameterVector(network: Network): ParameterVector {
  const parameterRuntimeContext = createParameterRuntimeContext(network);

  // Step 1: Read the live scalar values in the exact layout order.
  const values = Float64Array.from(
    parameterRuntimeContext.runtimeBindings,
    (runtimeBinding) => runtimeBinding.readValue(),
  );

  // Step 2: Return layout metadata and values as one versioned payload.
  return {
    layout: parameterRuntimeContext.layout,
    values,
  };
}

/**
 * Import one versioned parameter vector into a compatible live network runtime.
 *
 * The target layout is rebuilt fresh and validated against the incoming vector
 * before any bias or weight mutation occurs. Version `1` applies only live
 * node biases and live forward-connection or self-connection weights, so
 * disabled connections still consume slots whenever they still exist in the
 * target runtime graph. Ordered descriptor compatibility stays innovation-first
 * and falls back to stable endpoint gene ids when an innovation id is absent,
 * which keeps same-runtime imports aligned with the export layout contract.
 * Import rejects non-neutral `node.response` and `connection.gain` explicitly
 * and also rejects version, entry-count, values-length, or descriptor mismatch
 * before mutation.
 *
 * @param network - Live target network instance.
 * @param parameterVector - Versioned payload to apply.
 * @returns Nothing. The target runtime mutates only after compatibility checks pass.
 */
export function fromParameterVector(
  network: Network,
  parameterVector: ParameterVector,
): void {
  const parameterRuntimeContext = createParameterRuntimeContext(network);

  // Step 1: Reject incompatible payloads before mutating the target runtime.
  assertCompatibleParameterVector(
    parameterVector,
    parameterRuntimeContext.layout,
  );

  // Step 2: Apply the aligned scalar values after the full compatibility gate.
  parameterRuntimeContext.runtimeBindings.forEach(
    (runtimeBinding, bindingIndex) => {
      runtimeBinding.writeValue(parameterVector.values[bindingIndex]!);
    },
  );
}

function collectParameterLayoutBiasEntries(
  networkInternals: ReturnType<typeof asNetworkInternals>,
): ParameterLayoutBiasDescriptor[] {
  const biasEntries = networkInternals.nodes
    .map((node) => createParameterLayoutBiasDescriptor(node))
    .toSorted(
      (leftBiasEntry, rightBiasEntry) =>
        leftBiasEntry.nodeId - rightBiasEntry.nodeId,
    );

  assertDistinctBiasOrdering(biasEntries);
  return biasEntries;
}

function createParameterLayoutBiasDescriptor(
  node: Node,
): ParameterLayoutBiasDescriptor {
  return {
    kind: 'bias',
    nodeId: readRequiredLayoutNodeGeneId(node, 'bias node'),
  };
}

function assertDistinctBiasOrdering(
  biasEntries: ReadonlyArray<ParameterLayoutBiasDescriptor>,
): void {
  const seenNodeIds = new Set<number>();

  for (const biasEntry of biasEntries) {
    if (seenNodeIds.has(biasEntry.nodeId)) {
      throw new Error(
        `ParameterLayoutV1 requires unique bias node ids. Duplicate node id ${biasEntry.nodeId} is ambiguous.`,
      );
    }

    seenNodeIds.add(biasEntry.nodeId);
  }
}

function collectParameterLayoutWeightEntries(
  networkInternals: ReturnType<typeof asNetworkInternals>,
): ParameterLayoutWeightDescriptor[] {
  const weightOrderingIdentities = [
    ...networkInternals.connections,
    ...networkInternals.selfconns,
  ]
    .map((connectionInstance) =>
      createParameterLayoutWeightOrderingIdentity(connectionInstance),
    )
    .toSorted(compareParameterLayoutWeightOrderingIdentity);

  assertDistinctWeightOrdering(weightOrderingIdentities);
  return weightOrderingIdentities.map(
    (weightOrderingIdentity) => weightOrderingIdentity.descriptor,
  );
}

function createParameterLayoutWeightOrderingIdentity(
  connectionInstance: Connection,
): ParameterLayoutWeightOrderingIdentity {
  const fromGeneId = readRequiredLayoutNodeGeneId(
    connectionInstance.from,
    'weight source node',
  );
  const toGeneId = readRequiredLayoutNodeGeneId(
    connectionInstance.to,
    'weight target node',
  );
  const innovation = Number.isFinite(connectionInstance.innovation)
    ? connectionInstance.innovation
    : null;

  return {
    descriptor:
      innovation === null
        ? {
            kind: 'weight',
            from: fromGeneId,
            to: toGeneId,
          }
        : {
            kind: 'weight',
            from: fromGeneId,
            innovation,
            to: toGeneId,
          },
    fromGeneId,
    innovation,
    orderingIdentityKey:
      innovation === null
        ? `fallback:${fromGeneId}->${toGeneId}`
        : `innovation:${innovation}`,
    toGeneId,
  };
}

function compareParameterLayoutWeightOrderingIdentity(
  leftWeightIdentity: ParameterLayoutWeightOrderingIdentity,
  rightWeightIdentity: ParameterLayoutWeightOrderingIdentity,
): number {
  const innovationDifference = compareOptionalInnovations(
    leftWeightIdentity.innovation,
    rightWeightIdentity.innovation,
  );

  if (innovationDifference !== 0) {
    return innovationDifference;
  }

  if (leftWeightIdentity.fromGeneId !== rightWeightIdentity.fromGeneId) {
    return leftWeightIdentity.fromGeneId - rightWeightIdentity.fromGeneId;
  }

  return leftWeightIdentity.toGeneId - rightWeightIdentity.toGeneId;
}

function compareOptionalInnovations(
  leftInnovation: number | null,
  rightInnovation: number | null,
): number {
  const leftHasInnovation = leftInnovation !== null;
  const rightHasInnovation = rightInnovation !== null;

  if (leftHasInnovation && rightHasInnovation) {
    const innovationDifference = leftInnovation - rightInnovation;

    if (innovationDifference !== 0) {
      return innovationDifference;
    }
  }

  if (leftHasInnovation !== rightHasInnovation) {
    return leftHasInnovation ? -1 : 1;
  }

  return 0;
}

function assertDistinctWeightOrdering(
  weightOrderingIdentities: ReadonlyArray<ParameterLayoutWeightOrderingIdentity>,
): void {
  const seenOrderingIdentityKeys = new Set<string>();

  for (const weightOrderingIdentity of weightOrderingIdentities) {
    if (
      seenOrderingIdentityKeys.has(weightOrderingIdentity.orderingIdentityKey)
    ) {
      throw new Error(
        `ParameterLayoutV1 requires unique stable weight identities. Duplicate identity ${weightOrderingIdentity.orderingIdentityKey} is ambiguous.`,
      );
    }

    seenOrderingIdentityKeys.add(weightOrderingIdentity.orderingIdentityKey);
  }
}

function readRequiredLayoutNodeGeneId(
  node: Node,
  nodeRoleLabel: string,
): number {
  const nodeGeneId = asNodeInternals(node).geneId;

  if (typeof nodeGeneId !== 'number') {
    throw new Error(
      `ParameterLayoutV1 requires a stable ${nodeRoleLabel} gene id.`,
    );
  }

  return nodeGeneId;
}

function createParameterRuntimeContext(
  network: Network,
): ParameterRuntimeContext {
  const networkInternals = asNetworkInternals(network);

  // Step 1: Reject unsupported parameter families before creating a payload.
  assertSupportedParameterFamilies(networkInternals);

  // Step 2: Rebuild the live deterministic layout for this runtime.
  const layout = createParameterLayoutV1(network);

  // Step 3: Resolve one ordered read-write binding per layout descriptor.
  const runtimeBindings = collectParameterRuntimeBindings(
    networkInternals,
    layout,
  );

  // Step 4: Return the ordered binding context for export or import.
  return {
    layout,
    runtimeBindings,
  };
}

function assertSupportedParameterFamilies(
  networkInternals: ReturnType<typeof asNetworkInternals>,
): void {
  networkInternals.nodes.forEach((node, nodeIndex) => {
    if (node.response !== NEUTRAL_NODE_RESPONSE) {
      throw new Error(
        `ParameterVector v1 does not support non-neutral node.response. Node at runtime index ${nodeIndex} has response ${node.response}.`,
      );
    }
  });

  [...networkInternals.connections, ...networkInternals.selfconns].forEach(
    (connection, connectionIndex) => {
      if (
        connection.gain !== NEUTRAL_CONNECTION_GAIN &&
        connection.gater === null
      ) {
        throw new Error(
          `ParameterVector v1 does not support non-neutral connection.gain. Connection at runtime index ${connectionIndex} has gain ${connection.gain}.`,
        );
      }
    },
  );
}

function collectParameterRuntimeBindings(
  networkInternals: ReturnType<typeof asNetworkInternals>,
  layout: ParameterLayoutV1,
): ParameterRuntimeBinding[] {
  const nodesByGeneId = createNodesByGeneId(networkInternals.nodes);
  const connectionsByDescriptorKey =
    createConnectionsByDescriptorKey(networkInternals);

  return layout.entries.map((layoutEntry) =>
    createParameterRuntimeBinding(
      layoutEntry,
      nodesByGeneId,
      connectionsByDescriptorKey,
    ),
  );
}

function createNodesByGeneId(nodes: ReadonlyArray<Node>): Map<number, Node> {
  return new Map(
    nodes.map((node) => [
      readRequiredLayoutNodeGeneId(node, 'bias node'),
      node,
    ]),
  );
}

function createConnectionsByDescriptorKey(
  networkInternals: ReturnType<typeof asNetworkInternals>,
): Map<string, Connection> {
  return new Map(
    [...networkInternals.connections, ...networkInternals.selfconns].map(
      (connection) => {
        const weightDescriptor =
          createParameterLayoutWeightOrderingIdentity(connection).descriptor;

        return [
          createParameterLayoutEntryKey(weightDescriptor),
          connection,
        ] as const;
      },
    ),
  );
}

function createParameterRuntimeBinding(
  layoutEntry: ParameterLayoutEntry,
  nodesByGeneId: ReadonlyMap<number, Node>,
  connectionsByDescriptorKey: ReadonlyMap<string, Connection>,
): ParameterRuntimeBinding {
  if (layoutEntry.kind === 'bias') {
    const matchingNode = nodesByGeneId.get(layoutEntry.nodeId)!;

    return {
      descriptor: layoutEntry,
      readValue: () => matchingNode.bias,
      writeValue: (parameterValue) => {
        matchingNode.bias = parameterValue;
      },
    };
  }

  const descriptorKey = createParameterLayoutEntryKey(layoutEntry);
  const matchingConnection = connectionsByDescriptorKey.get(descriptorKey)!;

  return {
    descriptor: layoutEntry,
    readValue: () => matchingConnection.weight,
    writeValue: (parameterValue) => {
      matchingConnection.weight = parameterValue;
    },
  };
}

function assertCompatibleParameterVector(
  parameterVector: ParameterVector,
  targetLayout: ParameterLayoutV1,
): void {
  if (parameterVector.layout.version !== targetLayout.version) {
    throw new Error(
      `ParameterVector layout version mismatch. Expected version ${targetLayout.version} but received ${parameterVector.layout.version}.`,
    );
  }

  if (parameterVector.layout.entries.length !== targetLayout.entries.length) {
    throw new Error(
      `ParameterVector layout entry count mismatch. Expected ${targetLayout.entries.length} entries but received ${parameterVector.layout.entries.length}.`,
    );
  }

  if (parameterVector.values.length !== targetLayout.entries.length) {
    throw new Error(
      `ParameterVector values length mismatch. Expected ${targetLayout.entries.length} values but received ${parameterVector.values.length}.`,
    );
  }

  parameterVector.layout.entries.forEach((layoutEntry, entryIndex) => {
    const targetLayoutEntry = targetLayout.entries[entryIndex]!;

    if (!doParameterLayoutEntriesMatch(layoutEntry, targetLayoutEntry)) {
      throw new Error(
        `ParameterVector descriptor mismatch at index ${entryIndex}. Expected ${createParameterLayoutEntryKey(targetLayoutEntry)} but received ${createParameterLayoutEntryKey(layoutEntry)}.`,
      );
    }
  });
}

function doParameterLayoutEntriesMatch(
  leftLayoutEntry: ParameterLayoutEntry,
  rightLayoutEntry: ParameterLayoutEntry,
): boolean {
  return (
    leftLayoutEntry.kind === rightLayoutEntry.kind &&
    createParameterLayoutEntryKey(leftLayoutEntry) ===
      createParameterLayoutEntryKey(rightLayoutEntry)
  );
}

function createParameterLayoutEntryKey(
  layoutEntry: ParameterLayoutEntry,
): string {
  if (layoutEntry.kind === 'bias') {
    return `bias:${layoutEntry.nodeId}`;
  }

  return typeof layoutEntry.innovation === 'number'
    ? `weight:innovation:${layoutEntry.innovation}`
    : `weight:fallback:${layoutEntry.from}->${layoutEntry.to}`;
}

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
 * Historical node ids and topology intent are appended in optional trailing slots so older consumers that
 * only read the first six positions keep working.
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
  const nodeGeneIds = collectNodeGeneIds(networkInternals.nodes);

  // Step 3: Flatten all connection groups into compact connection records.
  const serializedConnections = collectSerializedConnections(networkInternals);

  // Step 4: Return compact tuple with IO shape metadata and trailing identity slots.
  return [
    activations,
    states,
    squashes,
    serializedConnections,
    networkInternals.input,
    networkInternals.output,
    nodeGeneIds,
    networkInternals._topologyIntent,
  ];
}

/**
 * Serializes a network instance into the compressed compact format.
 *
 * This path keeps round-trip semantics identical to `serialize()` while
 * replacing the object-per-connection payload with one array-oriented block.
 *
 * @param this - Bound network instance.
 * @returns Compressed compact payload.
 */
export function serializeCompressed(
  this: Network,
): CompressedSerializedNetwork {
  const networkInternals = asNetworkInternals(this);

  // Step 1: Build the exact structural snapshot through the verbose serializer.
  const jsonSnapshot = toJSONImpl.call(this);

  // Step 2: Capture live runtime activation and recurrent-state arrays.
  const activations = collectNodeActivations(networkInternals.nodes);
  const states = collectNodeStates(networkInternals.nodes);

  // Step 3: Compress the verbose JSON connection rows.
  const compressedConnections = compressSerializedConnections(
    jsonSnapshot.connections,
  );

  // Step 4: Return the additive compressed payload.
  return {
    activations,
    architecture: jsonSnapshot.architecture,
    connections: compressedConnections,
    dropout: jsonSnapshot.dropout,
    extensions: jsonSnapshot.extensions,
    format: COMPRESSED_NETWORK_FORMAT,
    formatVersion: jsonSnapshot.formatVersion,
    input: networkInternals.input,
    nodes: jsonSnapshot.nodes,
    output: networkInternals.output,
    states,
    topologyIntent: jsonSnapshot.topologyIntent,
  };
}

/**
 * Serializes a network instance into the compressed archive wrapper.
 *
 * This is the Node-side storage path for `serializeCompressed()`: it first
 * builds the exact compressed JSON payload, then applies gzip or zstd above
 * that payload without changing replay semantics.
 *
 * @param this - Bound network instance.
 * @param options - Optional archive compression settings.
 * @returns Archived compressed payload.
 */
export function serializeCompressedArchive(
  this: Network,
  options?: CompressedSerializedNetworkArchiveOptions,
): CompressedSerializedNetworkArchive {
  // Step 1: Build the exact compressed JSON payload.
  const compressedPayload = serializeCompressed.call(this);

  // Step 2: Wrap the payload in the Node-side archive codec.
  return createCompressedNetworkArchive(compressedPayload, options);
}

/**
 * Serialize a network archive and report size plus encode-time metrics for deterministic storage and transport audits.
 * The metrics payload helps compare archive codecs without changing the underlying compressed network contract.
 *
 * @param this - Bound network instance.
 * @param options - Optional archive compression settings.
 * @returns Archived payload plus encode metrics.
 */
export function serializeCompressedArchiveWithMetrics(
  this: Network,
  options?: CompressedSerializedNetworkArchiveOptions,
): CompressedArchiveEncodeResult<CompressedSerializedNetworkArchive> {
  const startedAt = performance.now();

  // Step 1: Build the exact compressed JSON payload.
  const compressedPayload = serializeCompressed.call(this);

  // Step 2: Wrap the payload in the archive codec.
  const archive = createCompressedNetworkArchive(compressedPayload, options);
  const encodeTimeMs = performance.now() - startedAt;

  // Step 3: Report byte-size and encode timing metrics alongside the archive.
  return {
    archive,
    metrics: createCompressedArchiveEncodeMetrics(
      estimateSerializedByteLength(compressedPayload),
      decodeArchivePayloadBase64(archive.payload).length,
      encodeTimeMs,
    ),
  };
}

/**
 * Serializes a network instance into the compressed archive wrapper with async runtime codecs.
 *
 * Browser runtimes prefer the archive stream path so payload compression can stay
 * off the synchronous main-thread lane, while Node falls back to the existing
 * archive owner when browser streams are unavailable.
 *
 * @param this - Bound network instance.
 * @param options - Optional archive compression settings.
 * @returns Archived compressed payload.
 */
export async function serializeCompressedArchiveAsync(
  this: Network,
  options?: CompressedSerializedNetworkArchiveOptions,
): Promise<CompressedSerializedNetworkArchive> {
  // Step 1: Build the exact compressed JSON payload.
  const compressedPayload = serializeCompressed.call(this);

  // Step 2: Wrap the payload in the best available async archive codec.
  return createCompressedNetworkArchiveAsync(compressedPayload, options);
}

/**
 * Serialize a network archive with async codecs and report size plus encode-time metrics for responsive environments.
 * This variant keeps the same archive semantics while allowing non-blocking compression paths in browser runtimes.
 *
 * @param this - Bound network instance.
 * @param options - Optional archive compression settings.
 * @returns Archived payload plus encode metrics.
 */
export async function serializeCompressedArchiveAsyncWithMetrics(
  this: Network,
  options?: CompressedSerializedNetworkArchiveOptions,
): Promise<CompressedArchiveEncodeResult<CompressedSerializedNetworkArchive>> {
  const startedAt = performance.now();

  // Step 1: Build the exact compressed JSON payload.
  const compressedPayload = serializeCompressed.call(this);

  // Step 2: Wrap the payload in the best available async archive codec.
  const archive = await createCompressedNetworkArchiveAsync(
    compressedPayload,
    options,
  );
  const encodeTimeMs = performance.now() - startedAt;

  // Step 3: Report byte-size and encode timing metrics alongside the archive.
  return {
    archive,
    metrics: createCompressedArchiveEncodeMetrics(
      estimateSerializedByteLength(compressedPayload),
      decodeArchivePayloadBase64(archive.payload).length,
      encodeTimeMs,
    ),
  };
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
    nodeGeneIds: compactPayload.nodeGeneIds,
    input: resolvedSize.input,
    output: resolvedSize.output,
  });
  rebuiltNetwork.refreshExplicitIORoles();

  // Step 4: Rebuild all compact connections and restore gating links.
  rebuildConnectionsFromCompactPayload({
    networkInternals,
    serializedConnections: compactPayload.connections,
  });

  // Step 5: Restore optional topology intent and advance historical counters.
  if (compactPayload.topologyIntent) {
    rebuiltNetwork.setTopologyIntent(compactPayload.topologyIntent);
  }
  syncRestoredHistoricalCounters(networkInternals);

  return rebuiltNetwork;
};

/**
 * Rebuild a network instance from the compressed compact payload and restore runtime activation/state vectors after structure import.
 * The function validates payload format first so incompatible compressed data fails before partial reconstruction occurs.
 *
 * @param data - Compressed compact payload.
 * @param inputSize - Optional input-size override.
 * @param outputSize - Optional output-size override.
 * @returns Reconstructed network instance.
 */
export const deserializeCompressed = (
  data: CompressedSerializedNetwork,
  inputSize?: number,
  outputSize?: number,
): Network => {
  if (data.format !== COMPRESSED_NETWORK_FORMAT) {
    throw new TypeError('Invalid compressed network payload format.');
  }

  const rebuiltNetwork = fromJSONImpl({
    architecture: data.architecture,
    connections: decompressSerializedConnections(data.connections),
    dropout: data.dropout,
    extensions: data.extensions,
    formatVersion: data.formatVersion,
    input: inputSize ?? data.input,
    nodes: data.nodes,
    output: outputSize ?? data.output,
    topologyIntent: data.topologyIntent,
  });

  applySerializedRuntimeState(rebuiltNetwork, data.activations, data.states);

  return rebuiltNetwork;
};

/**
 * Rebuild a network instance from the compressed archive wrapper by inflating archive bytes and delegating to compressed deserialization.
 * This keeps archive-specific decode concerns separate from structural reconstruction and runtime-state restoration.
 *
 * @param data - Archived compressed payload.
 * @param inputSize - Optional input-size override.
 * @param outputSize - Optional output-size override.
 * @returns Reconstructed network instance.
 */
export const deserializeCompressedArchive = (
  data: CompressedSerializedNetworkArchive,
  inputSize?: number,
  outputSize?: number,
): Network => {
  // Step 1: Inflate the archive back into the exact compressed JSON payload.
  const compressedPayload = parseCompressedNetworkArchive(data);

  // Step 2: Reuse the existing compressed deserialize flow.
  return deserializeCompressed(compressedPayload, inputSize, outputSize);
};

/**
 * Rebuild a network archive and report size plus decode-time metrics for import performance and payload diagnostics.
 * The returned metrics quantify archive inflation and reconstruction overhead alongside the rebuilt runtime.
 *
 * @param data - Archived compressed payload.
 * @param inputSize - Optional input-size override.
 * @param outputSize - Optional output-size override.
 * @returns Rebuilt network plus decode metrics.
 */
export const deserializeCompressedArchiveWithMetrics = (
  data: CompressedSerializedNetworkArchive,
  inputSize?: number,
  outputSize?: number,
): CompressedArchiveDecodeResult<Network> => {
  const compressedByteLength = decodeArchivePayloadBase64(data.payload).length;
  const startedAt = performance.now();

  // Step 1: Inflate the archive back into the exact compressed JSON payload.
  const compressedPayload = parseCompressedNetworkArchive(data);

  // Step 2: Reuse the existing compressed deserialize flow.
  const rebuiltNetwork = deserializeCompressed(
    compressedPayload,
    inputSize,
    outputSize,
  );
  const decodeTimeMs = performance.now() - startedAt;

  // Step 3: Report byte-size and decode timing metrics alongside the network.
  return {
    metrics: createCompressedArchiveDecodeMetrics(
      estimateSerializedByteLength(compressedPayload),
      compressedByteLength,
      decodeTimeMs,
    ),
    value: rebuiltNetwork,
  };
};

/**
 * Rebuilds a network instance from the compressed archive wrapper with async runtime codecs.
 *
 * Browser runtimes prefer the archive stream path so payload hydration can stay
 * off the synchronous main-thread lane, while Node falls back to the existing
 * archive owner when browser streams are unavailable.
 *
 * @param data - Archived compressed payload.
 * @param inputSize - Optional input-size override.
 * @param outputSize - Optional output-size override.
 * @returns Reconstructed network instance.
 */
export const deserializeCompressedArchiveAsync = async (
  data: CompressedSerializedNetworkArchive,
  inputSize?: number,
  outputSize?: number,
  options: CompressedArchiveDecodeOptions = {},
): Promise<Network> => {
  // Step 1: Inflate the archive back into the exact compressed JSON payload.
  const compressedPayload = await parseCompressedNetworkArchiveAsync(
    data,
    options,
  );

  // Step 2: Reuse the existing compressed deserialize flow.
  return deserializeCompressed(compressedPayload, inputSize, outputSize);
};

/**
 * Rebuild a network archive with async codecs and report size plus decode-time metrics for streaming or browser import paths.
 * This helper preserves deterministic reconstruction while exposing decode telemetry for responsiveness tuning.
 *
 * @param data - Archived compressed payload.
 * @param inputSize - Optional input-size override.
 * @param outputSize - Optional output-size override.
 * @param options - Optional incremental decode callbacks.
 * @returns Rebuilt network plus decode metrics.
 */
export const deserializeCompressedArchiveAsyncWithMetrics = async (
  data: CompressedSerializedNetworkArchive,
  inputSize?: number,
  outputSize?: number,
  options: CompressedArchiveDecodeOptions = {},
): Promise<CompressedArchiveDecodeResult<Network>> => {
  const compressedByteLength = decodeArchivePayloadBase64(data.payload).length;
  const startedAt = performance.now();

  // Step 1: Inflate the archive back into the exact compressed JSON payload.
  const compressedPayload = await parseCompressedNetworkArchiveAsync(
    data,
    options,
  );

  // Step 2: Reuse the existing compressed deserialize flow.
  const rebuiltNetwork = deserializeCompressed(
    compressedPayload,
    inputSize,
    outputSize,
  );
  const decodeTimeMs = performance.now() - startedAt;

  // Step 3: Report byte-size and decode timing metrics alongside the network.
  return {
    metrics: createCompressedArchiveDecodeMetrics(
      estimateSerializedByteLength(compressedPayload),
      compressedByteLength,
      decodeTimeMs,
    ),
    value: rebuiltNetwork,
  };
};

/**
 * Restore live activation and recurrent-state scalars after structural import.
 *
 * @param rebuiltNetwork - Reconstructed runtime network.
 * @param activations - Activation values aligned to node order.
 * @param states - Recurrent state values aligned to node order.
 * @returns Nothing.
 */
function applySerializedRuntimeState(
  rebuiltNetwork: Network,
  activations: number[],
  states: number[],
): void {
  const networkInternals = asNetworkInternals(rebuiltNetwork);

  if (
    activations.length !== networkInternals.nodes.length ||
    states.length !== networkInternals.nodes.length
  ) {
    throw new TypeError('Compressed runtime state length is invalid.');
  }

  networkInternals.nodes.forEach((nodeReference, nodeIndex) => {
    const nodeInternals = asNodeInternals(nodeReference);

    nodeInternals.activation = activations[nodeIndex]!;
    nodeInternals.state = states[nodeIndex]!;
  });
}

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

  // Step 5: Normalize explicit temporal descriptors against the live graph.
  synchronizeTemporalDescriptorExtensions(
    this as RuntimeNetworkWithSerializedExtensions,
  );

  // Step 6: Re-emit any explicit extension bag hydrated previously.
  applySerializedExtensionBag(
    this as RuntimeNetworkWithSerializedExtensions,
    networkJson,
  );

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
  rebuiltNetwork.refreshExplicitIORoles();

  // Step 4: Rebuild JSON connections, gating links, and enabled flags.
  rebuildConnectionsFromJsonPayload({
    networkInternals,
    connectionJsonEntries: json.connections,
  });

  // Step 5: Restore the public topology contract and advance historical counters.
  if (json.topologyIntent) {
    rebuiltNetwork.setTopologyIntent(json.topologyIntent);
  }
  syncRestoredHistoricalCounters(networkInternals);

  // Step 6: Hydrate optional architecture metadata when valid.
  applyHydratedArchitectureDescriptor(rebuiltNetwork, json.architecture);

  // Step 7: Hydrate the generic extension bag when its top-level shape is valid.
  applyHydratedExtensionBag(
    rebuiltNetwork as RuntimeNetworkWithSerializedExtensions,
    json.extensions,
  );

  return rebuiltNetwork;
};

/**
 * Re-emits the hydrated generic extension bag on verbose JSON snapshots.
 *
 * @param runtimeNetwork - Runtime network that may carry hydrated extensions.
 * @param networkJson - Target JSON payload.
 * @returns Nothing.
 */
function applySerializedExtensionBag(
  runtimeNetwork: RuntimeNetworkWithSerializedExtensions,
  networkJson: NetworkJSON,
): void {
  const clonedExtensions = cloneNetworkJsonExtensions(
    runtimeNetwork._serializedExtensions,
  );
  if (!clonedExtensions) {
    return;
  }

  networkJson.extensions = clonedExtensions;
}

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
 * Hydrates the generic extension bag onto the runtime network when shape is valid.
 *
 * @param runtimeNetwork - Runtime network receiving the extension bag.
 * @param extensions - Optional serialized extension bag.
 * @returns Nothing.
 */
function applyHydratedExtensionBag(
  runtimeNetwork: RuntimeNetworkWithSerializedExtensions,
  extensions: NetworkJSON['extensions'],
): void {
  const clonedExtensions = cloneNetworkJsonExtensions(extensions);
  if (!clonedExtensions) {
    Reflect.deleteProperty(runtimeNetwork, '_serializedExtensions');
    return;
  }

  runtimeNetwork._serializedExtensions = clonedExtensions;
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
 * Clones one generic network extension bag when the top-level shape is valid.
 *
 * @param extensions - Optional serialized extension bag.
 * @returns Cloned extension bag or `undefined` when shape is invalid.
 */
function cloneNetworkJsonExtensions(
  extensions: NetworkJSON['extensions'],
): NetworkJSONExtensions | undefined {
  if (!extensions) {
    return undefined;
  }

  const hasValidVersion =
    Number.isInteger(extensions.version) && extensions.version > 0;
  const hasValidValues = isPlainObjectRecord(extensions.values);

  if (!hasValidVersion || !hasValidValues) {
    return undefined;
  }

  return {
    version: extensions.version,
    values: structuredClone(extensions.values),
  };
}

function isPlainObjectRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Re-exported connection constructor used by tooling that needs to inspect connection internals.
 */
export { Connection };
