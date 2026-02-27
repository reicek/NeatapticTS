import Network from '../../network';
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
 * Casts a network instance to the internal runtime shape used by serializer helpers.
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
 * Casts a network instance to internals that include optional dropout metadata.
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
 * Casts a node instance to its internal runtime representation.
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
  ] = data;

  return {
    activations,
    states,
    squashes,
    connections,
    serializedInput,
    serializedOutput,
  };
}

/**
 * Resolves effective input/output dimensions using optional explicit overrides.
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
 * Resolves one size value with override-first semantics.
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
  return serializedValue || DEFAULT_NUMERIC_VALUE;
}

/**
 * Creates a new network instance for deserialize workflows.
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
 * Clears mutable runtime collections before reconstruction.
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
 * Checks whether an index is inside the bounds of a node array.
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
 * Checks whether a candidate index value is a finite number.
 *
 * @param index - Candidate index value.
 * @returns True when finite number.
 * @remarks
 * Use with bounds checks when validating serialized endpoint references.
 */
export function isFiniteIndex(index: number): boolean {
  return Number.isFinite(index);
}
