import Node from '../../node';
import Connection from '../../connection';
import {
  resolveActivationFunction,
  resolveActivationKey,
} from './network.serialize.activation.utils';
import {
  applyRestoredConnectionIdentity,
  asNodeInternals,
  isFiniteIndex,
  isNodeIndexInBounds,
} from './network.serialize.runtime.utils';
import type {
  ConnectionInternalsWithEnabled,
  JsonConnectionRebuildContext,
  JsonNodeRebuildContext,
  NetworkInternals,
  NetworkInternalsWithDropout,
  NetworkJSON,
  NetworkJSONConnection,
  NetworkJSONNode,
  NodeInternals,
} from './network.serialize.utils.types';
import {
  DEFAULT_NUMERIC_VALUE,
  ERROR_INVALID_NETWORK_JSON,
  FIRST_CONNECTION_INDEX,
  NETWORK_JSON_FORMAT_VERSION,
  WARNING_INVALID_CONNECTION_DURING_FROM_JSON,
  WARNING_INVALID_GATER_DURING_FROM_JSON,
  WARNING_UNKNOWN_FORMAT_VERSION,
} from './network.serialize.utils.types';
import { NetworkSerializeInvalidJsonError } from './network.serialize.errors';

const NEUTRAL_NODE_RESPONSE = 1;

/**
 * Creates an empty verbose JSON shell from runtime internals.
 *
 * The shell includes format and shape metadata and is filled in by subsequent
 * node and connection append steps.
 *
 * @param networkInternals - Runtime internals with optional dropout.
 * @returns Empty JSON shell with `formatVersion` and scalar metadata initialized.
 * @remarks
 * `formatVersion` is written from a single source-of-truth constant so exporters stay consistent.
 * @example
 * ```ts
 * const networkJson = createEmptyNetworkJson(networkInternals);
 * ```
 */
export function createEmptyNetworkJson(
  networkInternals: NetworkInternalsWithDropout,
): NetworkJSON {
  return {
    formatVersion: NETWORK_JSON_FORMAT_VERSION,
    input: networkInternals.input,
    output: networkInternals.output,
    dropout: resolveDropout(networkInternals.dropout),
    topologyIntent: networkInternals._topologyIntent,
    nodes: [],
    connections: [],
  };
}

/**
 * Resolves dropout with a numeric fallback when the value is absent.
 *
 * @param dropout - Optional dropout value.
 * @returns Effective dropout.
 * @remarks
 * A missing value is normalized to `0` so downstream consumers can treat `dropout` as required numeric data.
 */
export function resolveDropout(dropout: number | undefined): number {
  return dropout || DEFAULT_NUMERIC_VALUE;
}

/**
 * Appends JSON node entries and optional self-connections.
 *
 * Node indices are refreshed during this step so connection records can reference
 * stable numeric endpoints.
 *
 * @param networkInternals - Runtime internals.
 * @param networkJson - Target JSON accumulator.
 * @returns Nothing.
 * @remarks
 * This mutates `networkJson` in place and preserves node iteration order as exported topology order.
 */
export function appendJsonNodesAndSelfConnections(
  networkInternals: NetworkInternals,
  networkJson: NetworkJSON,
): void {
  networkInternals.nodes.forEach((node, nodeIndex) => {
    const nodeInternals = asNodeInternals(node);
    nodeInternals.index = nodeIndex;

    networkJson.nodes.push(createJsonNode(node, nodeInternals, nodeIndex));
    appendJsonSelfConnectionWhenPresent(nodeInternals, nodeIndex, networkJson);
  });
}

/**
 * Appends JSON entries for forward connections.
 *
 * Non-finite endpoint indices are ignored to prevent malformed output records.
 *
 * @param networkInternals - Runtime internals.
 * @param networkJson - JSON accumulator.
 * @returns Nothing.
 * @remarks
 * This mutates `networkJson` in place and appends only valid forward connections.
 * Self-connections are handled separately by `appendJsonNodesAndSelfConnections`.
 */
export function appendJsonForwardConnections(
  networkInternals: NetworkInternals,
  networkJson: NetworkJSON,
): void {
  networkInternals.connections.forEach((connectionInstance) => {
    const sourceIndex = asNodeInternals(connectionInstance.from).index;
    const targetIndex = asNodeInternals(connectionInstance.to).index;

    if (!isFiniteIndex(sourceIndex) || !isFiniteIndex(targetIndex)) {
      return;
    }

    networkJson.connections.push(
      createJsonConnection(
        connectionInstance,
        sourceIndex,
        targetIndex,
      ),
    );
  });
}

/**
 * Validates the verbose JSON payload root shape.
 *
 * @param json - Payload candidate.
 * @returns Nothing.
 * @throws Error When payload root is missing or not an object.
 * @remarks
 * This intentionally performs a lightweight root check.
 * Field-level compatibility checks happen in rebuild helpers with best-effort recovery.
 */
export function validateNetworkJsonOrThrow(json: NetworkJSON): void {
  if (!json || typeof json !== 'object') {
    throw new NetworkSerializeInvalidJsonError(ERROR_INVALID_NETWORK_JSON);
  }
}

/**
 * Warns when incoming verbose format version differs from the expected one.
 *
 * @param formatVersion - Incoming format version.
 * @returns Nothing.
 * @remarks
 * Unknown versions are warning-only so applications can attempt controlled migration paths.
 * For strict schema enforcement, add an application-level gate before calling `fromJSONImpl`.
 */
export function warnWhenJsonFormatVersionIsUnknown(
  formatVersion: number,
): void {
  if (formatVersion !== NETWORK_JSON_FORMAT_VERSION) {
    console.warn(WARNING_UNKNOWN_FORMAT_VERSION);
  }
}

/**
 * Rebuilds runtime nodes from verbose JSON entries.
 *
 * @param jsonNodeContext - JSON node rebuild context.
 * @returns Nothing.
 * @remarks
 * Node type, bias, squash function, and optional gene identifiers are restored.
 * Unknown squash names resolve through activation fallback logic to keep import non-throwing.
 * @example
 * ```ts
 * rebuildNodesFromJsonPayload({ networkInternals, nodeJsonEntries });
 * ```
 */
export function rebuildNodesFromJsonPayload(
  jsonNodeContext: JsonNodeRebuildContext,
): void {
  jsonNodeContext.nodeJsonEntries.forEach((nodeJsonEntry, nodeIndex) => {
    const rebuiltNode = createNodeWithType(nodeJsonEntry.type);
    hydrateNodeFromJsonEntry(rebuiltNode, nodeJsonEntry, nodeIndex);
    jsonNodeContext.networkInternals.nodes.push(rebuiltNode);
  });
}

/**
 * Rebuilds runtime connections from verbose JSON entries.
 *
 * Invalid entries are skipped with warnings so import continues for valid records.
 *
 * @param jsonConnectionContext - JSON connection rebuild context.
 * @returns Nothing.
 * @remarks
 * Endpoint validation and optional gater/enabled restoration are applied per row.
 * One invalid row does not stop processing of remaining rows.
 * @example
 * ```ts
 * rebuildConnectionsFromJsonPayload({
 *   networkInternals,
 *   connectionJsonEntries,
 * });
 * ```
 */
export function rebuildConnectionsFromJsonPayload(
  jsonConnectionContext: JsonConnectionRebuildContext,
): void {
  jsonConnectionContext.connectionJsonEntries.forEach((connectionJsonEntry) => {
    rebuildOneJsonConnection(
      jsonConnectionContext.networkInternals,
      connectionJsonEntry,
    );
  });
}

/**
 * Creates one JSON node entry.
 *
 * @param node - Runtime node.
 * @param nodeInternals - Node internals.
 * @param nodeIndex - Canonical index.
 * @returns JSON node entry.
 */
function createJsonNode(
  node: Node,
  nodeInternals: NodeInternals,
  nodeIndex: number,
): NetworkJSONNode {
  return {
    type: node.type,
    bias: nodeInternals.bias,
    ...(isFiniteNonNeutralNodeResponse(nodeInternals.response)
      ? { response: nodeInternals.response }
      : {}),
    squash: resolveActivationKey(nodeInternals.squash),
    index: nodeIndex,
    geneId: nodeInternals.geneId,
  };
}

/**
 * Appends JSON self-connection when node has one.
 *
 * @param nodeInternals - Node internals.
 * @param nodeIndex - Node index.
 * @param networkJson - JSON accumulator.
 * @returns Nothing.
 */
function appendJsonSelfConnectionWhenPresent(
  nodeInternals: NodeInternals,
  nodeIndex: number,
  networkJson: NetworkJSON,
): void {
  const selfConnection = nodeInternals.connections.self[FIRST_CONNECTION_INDEX];
  if (!selfConnection) {
    return;
  }

  networkJson.connections.push(
    createJsonConnection(
      selfConnection,
      nodeIndex,
      nodeIndex,
    ),
  );
}

/**
 * Creates one JSON connection entry.
 *
 * @param connectionInstance - Runtime connection carrying the historical identity to persist.
 * @param from - Source index.
 * @param to - Target index.
 * @returns JSON connection entry.
 */
function createJsonConnection(
  connectionInstance: Connection,
  from: number,
  to: number,
): NetworkJSONConnection {
  return {
    from,
    to,
    weight: connectionInstance.weight,
    gain: connectionInstance.gain,
    gater: resolveGaterIndex(connectionInstance.gater),
    enabled: isConnectionEnabled(connectionInstance),
    innovation: connectionInstance.innovation,
    fromGeneId: asNodeInternals(connectionInstance.from).geneId,
    toGeneId: asNodeInternals(connectionInstance.to).geneId,
    gaterGeneId: connectionInstance.gater
      ? asNodeInternals(connectionInstance.gater).geneId
      : null,
  };
}

/**
 * Resolves gater node index from gater reference.
 *
 * @param gaterNode - Optional gater node.
 * @returns Gater index or null.
 */
function resolveGaterIndex(gaterNode: Node | null): number | null {
  if (!gaterNode) {
    return null;
  }
  return asNodeInternals(gaterNode).index;
}

/**
 * Resolves enabled status from optional connection flag.
 *
 * @param connectionInstance - Connection instance.
 * @returns True when connection is enabled.
 */
function isConnectionEnabled(connectionInstance: Connection): boolean {
  return (
    (connectionInstance as ConnectionInternalsWithEnabled).enabled !== false
  );
}

/**
 * Hydrates one node from JSON node entry.
 *
 * @param rebuiltNode - Node to hydrate.
 * @param nodeJsonEntry - JSON node entry.
 * @param nodeIndex - Canonical node index.
 * @returns Nothing.
 */
function hydrateNodeFromJsonEntry(
  rebuiltNode: Node,
  nodeJsonEntry: NetworkJSONNode,
  nodeIndex: number,
): void {
  const nodeInternals = asNodeInternals(rebuiltNode);
  nodeInternals.bias = nodeJsonEntry.bias;
  nodeInternals.response =
    typeof nodeJsonEntry.response === 'number' &&
    Number.isFinite(nodeJsonEntry.response)
      ? nodeJsonEntry.response
      : NEUTRAL_NODE_RESPONSE;
  nodeInternals.squash = resolveActivationFunction(nodeJsonEntry.squash);
  nodeInternals.index = nodeIndex;
  if (typeof nodeJsonEntry.geneId === 'number') {
    nodeInternals.geneId = nodeJsonEntry.geneId;
  }
}

function isFiniteNonNeutralNodeResponse(response: unknown): response is number {
  return (
    typeof response === 'number' &&
    Number.isFinite(response) &&
    response !== NEUTRAL_NODE_RESPONSE
  );
}

/**
 * Rebuilds one verbose JSON connection entry.
 *
 * @param networkInternals - Runtime internals.
 * @param connectionJsonEntry - JSON connection entry.
 * @returns Nothing.
 */
function rebuildOneJsonConnection(
  networkInternals: NetworkInternals,
  connectionJsonEntry: NetworkJSONConnection,
): void {
  if (!isJsonConnectionShapeValid(connectionJsonEntry)) {
    return;
  }

  if (
    !isJsonConnectionInNodeBounds(networkInternals.nodes, connectionJsonEntry)
  ) {
    console.warn(WARNING_INVALID_CONNECTION_DURING_FROM_JSON);
    return;
  }

  const sourceNode = networkInternals.nodes[connectionJsonEntry.from];
  const targetNode = networkInternals.nodes[connectionJsonEntry.to];
  const createdConnection = createConnection(
    networkInternals,
    sourceNode,
    targetNode,
    connectionJsonEntry.weight,
  );
  applyRestoredConnectionIdentity(createdConnection, connectionJsonEntry);

  assignJsonGaterWhenValid(
    networkInternals,
    connectionJsonEntry.gater,
    createdConnection,
  );
  assignJsonEnabledFlagWhenProvided(
    createdConnection,
    connectionJsonEntry.enabled,
  );
  assignJsonGainWhenProvided(createdConnection, connectionJsonEntry.gain);
}

/**
 * Checks that JSON connection has numeric endpoint fields.
 *
 * @param connectionJsonEntry - JSON connection entry.
 * @returns True when endpoint fields are numbers.
 */
function isJsonConnectionShapeValid(
  connectionJsonEntry: NetworkJSONConnection,
): boolean {
  return (
    typeof connectionJsonEntry.from === 'number' &&
    typeof connectionJsonEntry.to === 'number'
  );
}

/**
 * Checks JSON connection indices against node list bounds.
 *
 * @param nodes - Node list.
 * @param connectionJsonEntry - JSON connection entry.
 * @returns True when both indices are valid.
 */
function isJsonConnectionInNodeBounds(
  nodes: Node[],
  connectionJsonEntry: NetworkJSONConnection,
): boolean {
  return (
    isNodeIndexInBounds(nodes, connectionJsonEntry.from) &&
    isNodeIndexInBounds(nodes, connectionJsonEntry.to)
  );
}

/**
 * Creates one connection and returns first created instance.
 *
 * @param networkInternals - Runtime internals.
 * @param sourceNode - Source node.
 * @param targetNode - Target node.
 * @param weight - Connection weight.
 * @returns Created connection or undefined.
 */
function createConnection(
  networkInternals: NetworkInternals,
  sourceNode: Node,
  targetNode: Node,
  weight: number,
): Connection | undefined {
  return networkInternals.connect(sourceNode, targetNode, weight)[
    FIRST_CONNECTION_INDEX
  ];
}

/**
 * Assigns JSON gater when connection and gater index are valid.
 *
 * @param networkInternals - Runtime internals.
 * @param gaterIndex - Optional gater index.
 * @param createdConnection - Created connection.
 * @returns Nothing.
 */
function assignJsonGaterWhenValid(
  networkInternals: NetworkInternals,
  gaterIndex: number | null,
  createdConnection: Connection | undefined,
): void {
  if (!createdConnection || gaterIndex == null) {
    return;
  }

  if (!isNodeIndexInBounds(networkInternals.nodes, gaterIndex)) {
    console.warn(WARNING_INVALID_GATER_DURING_FROM_JSON);
    return;
  }

  networkInternals.gate(networkInternals.nodes[gaterIndex], createdConnection);
}

/**
 * Assigns enabled flag when value is provided.
 *
 * @param createdConnection - Created connection.
 * @param enabled - Optional enabled value.
 * @returns Nothing.
 */
function assignJsonEnabledFlagWhenProvided(
  createdConnection: Connection | undefined,
  enabled: boolean,
): void {
  if (!createdConnection || typeof enabled === 'undefined') {
    return;
  }

  (createdConnection as ConnectionInternalsWithEnabled).enabled = enabled;
}

/**
 * Assigns a restored connection gain when one was serialized explicitly.
 *
 * @param createdConnection - Created connection.
 * @param gain - Optional serialized gain.
 * @returns Nothing.
 */
function assignJsonGainWhenProvided(
  createdConnection: Connection | undefined,
  gain: number | undefined,
): void {
  if (
    !createdConnection ||
    typeof gain !== 'number' ||
    !Number.isFinite(gain)
  ) {
    return;
  }

  createdConnection.gain = gain;
}

/**
 * Creates one node with provided type.
 *
 * @param nodeType - Node type.
 * @returns New node.
 */
function createNodeWithType(nodeType: string): Node {
  return new Node(nodeType);
}
