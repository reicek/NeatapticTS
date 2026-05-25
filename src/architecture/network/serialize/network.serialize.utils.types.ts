import type {
  CompressedSerializedConnectionBlock,
  CompressedSerializedConnectionWeights,
  CompressedSerializedIndexRun,
  CompressedSerializedNetwork,
  CompressedSerializedNetworkArchive,
  CompressedSerializedNetworkArchiveCompression,
  CompressedSerializedNetworkArchiveOptions,
  CompactConnectionRebuildContext,
  CompactNodeRebuildContext,
  CompactPayloadContext,
  CompactSerializedNetworkTuple,
  ConnectionInternalsWithEnabled,
  JsonConnectionRebuildContext,
  JsonNodeRebuildContext,
  NetworkInternalsWithDropout,
  NetworkJSON as NetworkJsonContract,
  NetworkJSONConnection,
  NetworkJSONNode,
  ResolvedNetworkSizeContext,
  SerializeNetworkInternals as NetworkInternals,
  SerializeNodeInternals as NodeInternals,
  SerializedConnection,
} from '../network.types';

export type {
  CompressedSerializedConnectionBlock,
  CompressedSerializedConnectionWeights,
  CompressedSerializedIndexRun,
  CompressedSerializedNetwork,
  CompressedSerializedNetworkArchive,
  CompressedSerializedNetworkArchiveCompression,
  CompressedSerializedNetworkArchiveOptions,
  CompactConnectionRebuildContext,
  CompactNodeRebuildContext,
  CompactPayloadContext,
  CompactSerializedNetworkTuple,
  ConnectionInternalsWithEnabled,
  JsonConnectionRebuildContext,
  JsonNodeRebuildContext,
  NetworkInternals,
  NetworkInternalsWithDropout,
  NetworkJSONConnection,
  NetworkJSONNode,
  NodeInternals,
  ResolvedNetworkSizeContext,
  SerializedConnection,
};

/**
 * Canonical verbose network JSON contract used across serializer and deserializer entry points in persistence, migration, and diagnostics workflows.
 * This local alias keeps payload-shape ownership discoverable from the serialize boundary without requiring deep type imports.
 */
export type NetworkJSON = NetworkJsonContract;

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
 * incompatible payloads before mutating a live network.
 */
export interface ParameterVector {
  /** Ordered descriptor metadata aligned with `values`. */
  layout: ParameterLayoutV1;
  /** Scalar parameter values in the exact order described by `layout.entries`. */
  values: Float64Array;
}

/**
 * Current verbose `NetworkJSON` schema version used by serializer and deserializer boundaries to coordinate compatibility checks and migration warnings.
 * Readers should treat mismatches as migration signals rather than silently assuming field parity.
 */
export const NETWORK_JSON_FORMAT_VERSION = 4;

/**
 * Neutral numeric fallback applied whenever optional scalar fields are absent in serialized payloads during rebuild normalization.
 * Using a single constant keeps rebuild defaults predictable across compact and verbose import paths.
 */
export const DEFAULT_NUMERIC_VALUE = 0;

/**
 * Index of the first created connection returned by `connect()`.
 *
 * The runtime API returns an array, and serializer logic consistently reads index `0`.
 */
export const FIRST_CONNECTION_INDEX = 0;

/**
 * Node type literal for input layer nodes used during JSON serialization and deserialization rebuilds.
 */
export const NODE_TYPE_INPUT = 'input';

/**
 * Node type literal for hidden layer nodes used during JSON serialization and deserialization rebuilds.
 */
export const NODE_TYPE_HIDDEN = 'hidden';

/**
 * Node type literal for output layer nodes used during JSON serialization and deserialization rebuilds.
 */
export const NODE_TYPE_OUTPUT = 'output';

/**
 * Activation key used when a serialized squash name cannot be resolved to any registered runtime activation function safely.
 * Identity preserves import continuity while warning paths surface the unknown key to diagnostics.
 */
export const FALLBACK_ACTIVATION_KEY = 'identity';

/**
 * Stable error text thrown when verbose JSON root validation fails before any field-level reconstruction logic executes.
 * Keeping this message deterministic helps tests and diagnostics tooling match invalid-payload failures.
 */
export const ERROR_INVALID_NETWORK_JSON = 'Invalid JSON for network.';

/**
 * Warning emitted for unknown verbose format versions encountered during JSON import boundaries with best-effort fallback semantics.
 * Keep this message stable so diagnostics tooling can classify unknown-format imports consistently.
 */
export const WARNING_UNKNOWN_FORMAT_VERSION =
  'fromJSONImpl: Unknown formatVersion, attempting import.';

/**
 * Warning emitted when compact deserialize encounters an invalid gater index and skips the gater assignment silently.
 */
export const WARNING_INVALID_GATER_DURING_DESERIALIZE =
  'Invalid gater index encountered during deserialize; skipping gater assignment.';

/**
 * Warning emitted when compact deserialize encounters invalid edge endpoints and skips the connection reconstruction silently.
 */
export const WARNING_INVALID_CONNECTION_DURING_DESERIALIZE =
  'Invalid connection indices encountered during deserialize; skipping connection.';

/**
 * Warning emitted when JSON verbose deserialize encounters invalid edge endpoints and skips the connection reconstruction silently.
 */
export const WARNING_INVALID_CONNECTION_DURING_FROM_JSON =
  'Invalid connection indices encountered during fromJSONImpl; skipping connection.';

/**
 * Warning emitted when JSON verbose deserialize encounters an invalid gater index and skips the gater assignment silently.
 */
export const WARNING_INVALID_GATER_DURING_FROM_JSON =
  'Invalid gater index encountered during fromJSONImpl; skipping gater assignment.';

/**
 * Prefix used when emitting a warning about an unknown activation squash key encountered during deserialization.
 */
export const WARNING_UNKNOWN_SQUASH_PREFIX = 'Unknown squash function';

/**
 * Suffix appended when emitting a warning about an unknown activation squash key encountered during deserialization.
 */
export const WARNING_UNKNOWN_SQUASH_SUFFIX =
  'encountered during deserialization. Falling back to identity.';
