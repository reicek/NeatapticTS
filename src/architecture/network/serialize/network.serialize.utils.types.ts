import type {
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
 * Default format version for verbose serialization payloads.
 *
 * Consumers can use this value to identify the JSON schema revision.
 *
 * @remarks
 * Bump this value only when the `NetworkJSON` shape changes in a way that affects compatibility.
 * Readers should treat unknown versions as a migration signal.
 */
export const NETWORK_JSON_FORMAT_VERSION = 2;

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
