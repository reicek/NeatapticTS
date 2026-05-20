import type {
  CompressedSerializedConnectionBlock,
  CompressedSerializedNetworkArchive,
  CompressedSerializedNetworkArchiveCompression,
  CompressedSerializedNetworkArchiveOptions,
  CompressedSerializedConnectionWeights,
  CompressedSerializedIndexRun,
  CompressedSerializedNetwork,
  CompactConnectionRebuildContext,
  CompactNodeRebuildContext,
  CompactPayloadContext,
  CompactSerializedNetworkTuple,
  ConnectionInternalsWithEnabled,
  JsonConnectionRebuildContext,
  JsonNodeRebuildContext,
  NetworkInternalsWithDropout,
  NetworkJSON,
  NetworkJSONConnection,
  NetworkJSONNode,
  ResolvedNetworkSizeContext,
  SerializeNetworkInternals as NetworkInternals,
  SerializeNodeInternals as NodeInternals,
  SerializedConnection,
} from '../network.types';

export type {
  CompressedSerializedConnectionBlock,
  CompressedSerializedNetworkArchive,
  CompressedSerializedNetworkArchiveCompression,
  CompressedSerializedNetworkArchiveOptions,
  CompressedSerializedConnectionWeights,
  CompressedSerializedIndexRun,
  CompressedSerializedNetwork,
  CompactConnectionRebuildContext,
  CompactNodeRebuildContext,
  CompactPayloadContext,
  CompactSerializedNetworkTuple,
  ConnectionInternalsWithEnabled,
  JsonConnectionRebuildContext,
  JsonNodeRebuildContext,
  NetworkInternals,
  NetworkInternalsWithDropout,
  NetworkJSON,
  NetworkJSONConnection,
  NetworkJSONNode,
  NodeInternals,
  ResolvedNetworkSizeContext,
  SerializedConnection,
};

/**
 * One ordered descriptor inside parameter-layout version `1`.
 *
 * Bias entries use the stable node gene id in `nodeId`.
 * Weight entries prefer the live connection `innovation` when it exists and
 * otherwise fall back to the stable endpoint gene ids in `from` and `to`.
 * Duplicate innovation ids or duplicate fallback endpoint pairs are rejected
 * as ambiguous instead of inheriting incidental container order.
 */
export type ParameterLayoutEntry =
  | { kind: 'bias'; nodeId: number }
  | { kind: 'weight'; from: number; to: number; innovation?: number };

/**
 * Deterministic parameter-layout descriptor owned by the network serialize boundary.
 *
 * Version `1` keeps one stable fold order: all bias entries first, then all
 * weight entries. For a fixed topology with stable historical ids, this gives
 * ordered determinism on the same runtime.
 * The layout documents descriptor order only; it does not claim cross-runtime
 * exact replay by itself.
 */
export interface ParameterLayoutV1 {
  /** Layout schema version. */
  version: 1;
  /** Ordered parameter descriptors for the current network topology. */
  entries: ParameterLayoutEntry[];
}

/**
 * Versioned parameter payload for same-runtime vector roundtrips.
 *
 * Layout metadata and scalar values travel together so imports can reject
 * incompatible payloads before mutating a live network. Version `1` exports
 * exactly one bias slot for every live runtime node and one weight slot for
 * every live forward connection plus self-connection in `ParameterLayoutV1`
 * order, including disabled connections when they still exist in the runtime
 * graph.
 *
 * Weight descriptors still prefer `innovation` and otherwise fall back to the
 * stable endpoint gene ids in `from` and `to`, so the same runtime-owned slot
 * identity survives export and import even when a live connection has no
 * innovation id. For a fixed topology with stable historical ids, this payload
 * is ordered deterministic on the same runtime. It does not claim cross-runtime
 * exact replay by itself.
 *
 * Non-neutral `node.response` and `connection.gain` remain explicit rejection
 * cases for the runtime helpers instead of silently widening the weights-and-
 * biases v1 contract.
 */
export interface ParameterVector {
  /** Ordered descriptor metadata aligned with `values`. */
  layout: ParameterLayoutV1;
  /** Scalar parameter values in the exact order described by `layout.entries`. */
  values: Float64Array;
}

/**
 * Default format version for verbose serialization payloads.
 *
 * Consumers can use this value to identify the JSON schema revision.
 *
 * @remarks
 * Bump this value only when the `NetworkJSON` shape changes in a way that affects compatibility.
 * Readers should treat unknown versions as a migration signal.
 */
export const NETWORK_JSON_FORMAT_VERSION = 4;

/**
 * Default numeric fallback for absent scalar values.
 *
 * This keeps import logic deterministic when optional numeric fields are missing.
 *
 * @remarks
 * Zero is chosen as the neutral fallback for index and scalar reconstruction paths.
 */
export const DEFAULT_NUMERIC_VALUE = 0;

/**
 * Index of the first created connection returned by `connect()`.
 *
 * The runtime API returns an array, and serializer logic consistently reads index `0`.
 *
 * @remarks
 * Keeping this as a named constant avoids positional magic numbers in rebuild helpers.
 */
export const FIRST_CONNECTION_INDEX = 0;

/**
 * Node type literal used for input layer nodes.
 */
export const NODE_TYPE_INPUT = 'input';

/**
 * Node type literal used for hidden layer nodes.
 */
export const NODE_TYPE_HIDDEN = 'hidden';

/**
 * Node type literal used for output layer nodes.
 */
export const NODE_TYPE_OUTPUT = 'output';

/**
 * Fallback activation key used when no mapping is found.
 *
 * Identity is selected as the safest non-disruptive activation fallback.
 *
 * @remarks
 * This preserves deserialization continuity when custom or unknown activations are encountered.
 */
export const FALLBACK_ACTIVATION_KEY = 'identity';

/**
 * Error text emitted for invalid verbose JSON payload roots.
 *
 * @remarks
 * Keep this message stable for tooling that pattern-matches deserialize failures.
 */
export const ERROR_INVALID_NETWORK_JSON = 'Invalid JSON for network.';

/**
 * Warning emitted for unknown verbose format versions.
 *
 * @remarks
 * Imports continue after this warning to support best-effort compatibility workflows.
 */
export const WARNING_UNKNOWN_FORMAT_VERSION =
  'fromJSONImpl: Unknown formatVersion, attempting import.';

/**
 * Warning emitted when compact deserialize sees an invalid gater index.
 */
export const WARNING_INVALID_GATER_DURING_DESERIALIZE =
  'Invalid gater index encountered during deserialize; skipping gater assignment.';

/**
 * Warning emitted when compact deserialize sees invalid edge endpoints.
 */
export const WARNING_INVALID_CONNECTION_DURING_DESERIALIZE =
  'Invalid connection indices encountered during deserialize; skipping connection.';

/**
 * Warning emitted when JSON deserialize sees invalid edge endpoints.
 */
export const WARNING_INVALID_CONNECTION_DURING_FROM_JSON =
  'Invalid connection indices encountered during fromJSONImpl; skipping connection.';

/**
 * Warning emitted when JSON deserialize sees an invalid gater index.
 */
export const WARNING_INVALID_GATER_DURING_FROM_JSON =
  'Invalid gater index encountered during fromJSONImpl; skipping gater assignment.';

/**
 * Prefix used when warning about unknown activation keys.
 */
export const WARNING_UNKNOWN_SQUASH_PREFIX = 'Unknown squash function';

/**
 * Suffix used when warning about unknown activation keys.
 */
export const WARNING_UNKNOWN_SQUASH_SUFFIX =
  'encountered during deserialization. Falling back to identity.';
