import type Network from '../../network';
import Node from '../../node';
import Connection from '../../connection';
import * as methods from '../../../methods/methods';
import type { ActivationFunction } from '../../../methods/activation.utils';
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
export type { SerializedConnection } from '../network.types';

/**
 * Default format version for verbose serialization.
 */
const NETWORK_JSON_FORMAT_VERSION = 2;

/**
 * Default numeric fallback.
 */
const DEFAULT_NUMERIC_VALUE = 0;

/**
 * Index of first returned element from connect().
 */
const FIRST_CONNECTION_INDEX = 0;

/**
 * Node type literal for input nodes.
 */
const NODE_TYPE_INPUT = 'input';

/**
 * Node type literal for hidden nodes.
 */
const NODE_TYPE_HIDDEN = 'hidden';

/**
 * Node type literal for output nodes.
 */
const NODE_TYPE_OUTPUT = 'output';

/**
 * Fallback activation key when no mapping is found.
 */
const FALLBACK_ACTIVATION_KEY = 'identity';

/**
 * Dynamic module path for Network constructor.
 */
const NETWORK_MODULE_PATH = '../../network';

/**
 * Error emitted for invalid verbose JSON payloads.
 */
const ERROR_INVALID_NETWORK_JSON = 'Invalid JSON for network.';

/**
 * Warning emitted for unknown verbose format versions.
 */
const WARNING_UNKNOWN_FORMAT_VERSION =
  'fromJSONImpl: Unknown formatVersion, attempting import.';

/**
 * Warning emitted when compact deserialize sees invalid gater index.
 */
const WARNING_INVALID_GATER_DURING_DESERIALIZE =
  'Invalid gater index encountered during deserialize; skipping gater assignment.';

/**
 * Warning emitted when compact deserialize sees invalid edge endpoints.
 */
const WARNING_INVALID_CONNECTION_DURING_DESERIALIZE =
  'Invalid connection indices encountered during deserialize; skipping connection.';

/**
 * Warning emitted when JSON deserialize sees invalid edge endpoints.
 */
const WARNING_INVALID_CONNECTION_DURING_FROM_JSON =
  'Invalid connection indices encountered during fromJSONImpl; skipping connection.';

/**
 * Warning emitted when JSON deserialize sees invalid gater index.
 */
const WARNING_INVALID_GATER_DURING_FROM_JSON =
  'Invalid gater index encountered during fromJSONImpl; skipping gater assignment.';

/**
 * Warning template prefix for unknown activation keys.
 */
const WARNING_UNKNOWN_SQUASH_PREFIX = 'Unknown squash function';

/**
 * Warning template suffix for unknown activation keys.
 */
const WARNING_UNKNOWN_SQUASH_SUFFIX =
  'encountered during deserialization. Falling back to identity.';

/**
 * Instance-level lightweight serializer used primarily for fast inter-thread transfer.
 *
 * @param this - Bound network instance.
 * @returns Compact serialized tuple payload.
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
 * Rebuilds a Network from compact tuple form.
 *
 * @param data - Compact tuple payload.
 * @param inputSize - Optional input override.
 * @param outputSize - Optional output override.
 * @returns Reconstructed network.
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
 * Verbose JSON export (stable formatVersion).
 *
 * @param this - Bound network instance.
 * @returns Verbose structural JSON payload.
 */
export function toJSONImpl(this: Network): NetworkJSON {
  const networkInternals = asNetworkInternalsWithDropout(this);

  // Step 1: Create baseline JSON shell.
  const networkJson = createEmptyNetworkJson(networkInternals);

  // Step 2: Rebuild node list and append self-connections.
  appendJsonNodesAndSelfConnections(networkInternals, networkJson);

  // Step 3: Append non-self connections.
  appendJsonForwardConnections(networkInternals, networkJson);

  return networkJson;
}

/**
 * Reconstructs a Network from verbose JSON payload.
 *
 * @param json - Verbose JSON payload.
 * @returns Reconstructed network.
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

  return rebuiltNetwork;
};

/**
 * Resolves canonical activation key from function reference.
 *
 * @param squashFunction - Activation function instance.
 * @returns Activation key.
 */
function resolveActivationKey(squashFunction: ActivationFunction): string {
  const activationEntry = findActivationEntryByReference(squashFunction);
  if (activationEntry) {
    return activationEntry[0];
  }

  const activationName = resolveNamedActivationFromFunction(squashFunction);
  if (activationName) {
    return activationName;
  }

  return FALLBACK_ACTIVATION_KEY;
}

/**
 * Resolves activation function from stored key or function name.
 *
 * @param squashName - Activation key or function name.
 * @returns Activation function.
 */
function resolveActivationFunction(
  squashName: string | undefined,
): ActivationFunction {
  const activationByKey = findActivationByKey(squashName);
  if (activationByKey) {
    return activationByKey;
  }

  const activationByName = findActivationByFunctionName(squashName);
  if (activationByName) {
    return activationByName;
  }

  warnUnknownSquashName(squashName);
  return methods.Activation.identity;
}

/**
 * Casts network instance to internal runtime shape.
 *
 * @param network - Network instance.
 * @returns Runtime internals.
 */
function asNetworkInternals(network: Network): NetworkInternals {
  return network as unknown as NetworkInternals;
}

/**
 * Casts network instance to internals with optional dropout.
 *
 * @param network - Network instance.
 * @returns Runtime internals with optional dropout.
 */
function asNetworkInternalsWithDropout(
  network: Network,
): NetworkInternalsWithDropout {
  return network as unknown as NetworkInternalsWithDropout;
}

/**
 * Casts node to internal runtime shape.
 *
 * @param node - Node instance.
 * @returns Node internals.
 */
function asNodeInternals(node: Node): NodeInternals {
  return node as unknown as NodeInternals;
}

/**
 * Creates compact payload context from tuple input.
 *
 * @param data - Compact tuple payload.
 * @returns Normalized payload context.
 */
function createCompactPayloadContext(
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
 * Resolves effective input/output dimensions with optional overrides.
 *
 * @param compactPayload - Compact payload context.
 * @param inputSizeOverride - Optional input override.
 * @param outputSizeOverride - Optional output override.
 * @returns Resolved network size context.
 */
function resolveNetworkSize(
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
 * Resolves one size value preferring explicit override.
 *
 * @param overrideValue - Optional explicit override.
 * @param serializedValue - Serialized fallback value.
 * @returns Effective size.
 */
function resolveSizeOverride(
  overrideValue: number | undefined,
  serializedValue: number,
): number {
  if (typeof overrideValue === 'number') {
    return overrideValue;
  }
  return serializedValue || DEFAULT_NUMERIC_VALUE;
}

/**
 * Creates a network instance using runtime constructor loading.
 *
 * @param input - Input size.
 * @param output - Output size.
 * @returns New network instance.
 */
function createNetworkInstance(input: number, output: number): Network {
  // eslint-disable-next-line @typescript-eslint/no-require-imports -- Dynamic require needed to avoid circular dependency
  const { default: NetworkConstructor } = require(NETWORK_MODULE_PATH);
  return new NetworkConstructor(input, output) as Network;
}

/**
 * Clears mutable runtime collections before reconstruction.
 *
 * @param networkInternals - Runtime internals.
 * @returns Nothing.
 */
function resetMutableRuntimeCollections(
  networkInternals: NetworkInternals,
): void {
  networkInternals.nodes = [];
  networkInternals.connections = [];
  networkInternals.selfconns = [];
  networkInternals.gates = [];
}

/**
 * Refreshes node.index for every node in list.
 *
 * @param nodes - Node list.
 * @returns Nothing.
 */
function refreshNodeIndices(nodes: Node[]): void {
  nodes.forEach((nodeReference, nodeIndex) => {
    asNodeInternals(nodeReference).index = nodeIndex;
  });
}

/**
 * Collects node activation values in positional order.
 *
 * @param nodes - Node list.
 * @returns Activation list.
 */
function collectNodeActivations(nodes: Node[]): number[] {
  return nodes.map(
    (nodeReference) => asNodeInternals(nodeReference).activation,
  );
}

/**
 * Collects node state values in positional order.
 *
 * @param nodes - Node list.
 * @returns State list.
 */
function collectNodeStates(nodes: Node[]): number[] {
  return nodes.map((nodeReference) => asNodeInternals(nodeReference).state);
}

/**
 * Collects node squash keys in positional order.
 *
 * @param nodes - Node list.
 * @returns Squash-key list.
 */
function collectNodeSquashKeys(nodes: Node[]): string[] {
  return nodes.map((nodeReference) =>
    resolveActivationKey(asNodeInternals(nodeReference).squash),
  );
}

/**
 * Collects serialized connections from forward and self groups.
 *
 * @param networkInternals - Runtime internals.
 * @returns Serialized connection list.
 */
function collectSerializedConnections(
  networkInternals: NetworkInternals,
): SerializedConnection[] {
  const allConnections = collectAllConnections(networkInternals);
  return allConnections.map(serializeOneConnection);
}

/**
 * Collects all runtime connections into a single list.
 *
 * @param networkInternals - Runtime internals.
 * @returns Combined connections.
 */
function collectAllConnections(
  networkInternals: NetworkInternals,
): Connection[] {
  return networkInternals.connections.concat(networkInternals.selfconns);
}

/**
 * Serializes one connection into compact indexed form.
 *
 * @param connectionInstance - Runtime connection.
 * @returns Serialized connection record.
 */
function serializeOneConnection(
  connectionInstance: Connection,
): SerializedConnection {
  return {
    from: asNodeInternals(connectionInstance.from).index,
    to: asNodeInternals(connectionInstance.to).index,
    weight: connectionInstance.weight,
    gater: connectionInstance.gater
      ? asNodeInternals(connectionInstance.gater).index
      : null,
  };
}

/**
 * Rebuilds nodes from compact payload.
 *
 * @param networkInternals - Runtime internals.
 * @param compactNodeContext - Compact node rebuild context.
 * @returns Nothing.
 */
function rebuildNodesFromCompactPayload(
  networkInternals: NetworkInternals,
  compactNodeContext: CompactNodeRebuildContext,
): void {
  compactNodeContext.activations.forEach((activation, nodeIndex) => {
    const nodeType = resolveNodeTypeFromCompactIndex(
      nodeIndex,
      compactNodeContext.activations.length,
      compactNodeContext.input,
      compactNodeContext.output,
    );
    const rebuiltNode = createNodeWithType(nodeType);
    hydrateNodeStateFromCompactPayload(
      rebuiltNode,
      activation,
      compactNodeContext.states[nodeIndex],
      compactNodeContext.squashes[nodeIndex],
      nodeIndex,
    );
    networkInternals.nodes.push(rebuiltNode);
  });
}

/**
 * Resolves node type from compact tuple position.
 *
 * @param nodeIndex - Node index.
 * @param totalNodeCount - Total node count.
 * @param input - Input size.
 * @param output - Output size.
 * @returns Node type string.
 */
function resolveNodeTypeFromCompactIndex(
  nodeIndex: number,
  totalNodeCount: number,
  input: number,
  output: number,
): string {
  if (nodeIndex < input) {
    return NODE_TYPE_INPUT;
  }
  if (nodeIndex >= totalNodeCount - output) {
    return NODE_TYPE_OUTPUT;
  }
  return NODE_TYPE_HIDDEN;
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

/**
 * Hydrates node runtime state from compact tuple values.
 *
 * @param rebuiltNode - Node to hydrate.
 * @param activation - Activation value.
 * @param state - State value.
 * @param squashName - Activation key.
 * @param nodeIndex - Canonical node index.
 * @returns Nothing.
 */
function hydrateNodeStateFromCompactPayload(
  rebuiltNode: Node,
  activation: number,
  state: number,
  squashName: string | undefined,
  nodeIndex: number,
): void {
  const nodeInternals = asNodeInternals(rebuiltNode);
  nodeInternals.activation = activation;
  nodeInternals.state = state;
  nodeInternals.squash = resolveActivationFunction(
    typeof squashName === 'string' ? squashName : undefined,
  );
  nodeInternals.index = nodeIndex;
}

/**
 * Rebuilds connections from compact payload records.
 *
 * @param compactConnectionContext - Compact connection rebuild context.
 * @returns Nothing.
 */
function rebuildConnectionsFromCompactPayload(
  compactConnectionContext: CompactConnectionRebuildContext,
): void {
  compactConnectionContext.serializedConnections.forEach(
    (serializedConnection) => {
      rebuildOneCompactConnection(
        compactConnectionContext.networkInternals,
        serializedConnection,
      );
    },
  );
}

/**
 * Rebuilds one compact serialized connection.
 *
 * @param networkInternals - Runtime internals.
 * @param serializedConnection - Serialized connection record.
 * @returns Nothing.
 */
function rebuildOneCompactConnection(
  networkInternals: NetworkInternals,
  serializedConnection: SerializedConnection,
): void {
  if (
    !isSerializedConnectionInNodeBounds(networkInternals, serializedConnection)
  ) {
    console.warn(WARNING_INVALID_CONNECTION_DURING_DESERIALIZE);
    return;
  }

  const sourceNode = networkInternals.nodes[serializedConnection.from];
  const targetNode = networkInternals.nodes[serializedConnection.to];
  const createdConnection = createConnection(
    networkInternals,
    sourceNode,
    targetNode,
    serializedConnection.weight,
  );
  assignCompactGaterWhenValid(
    networkInternals,
    serializedConnection.gater,
    createdConnection,
  );
}

/**
 * Checks compact connection bounds against current node list.
 *
 * @param networkInternals - Runtime internals.
 * @param serializedConnection - Serialized connection record.
 * @returns True when endpoints are valid.
 */
function isSerializedConnectionInNodeBounds(
  networkInternals: NetworkInternals,
  serializedConnection: SerializedConnection,
): boolean {
  return (
    serializedConnection.from >= DEFAULT_NUMERIC_VALUE &&
    serializedConnection.to >= DEFAULT_NUMERIC_VALUE &&
    serializedConnection.from < networkInternals.nodes.length &&
    serializedConnection.to < networkInternals.nodes.length
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
 * Assigns compact gater when both connection and gater index are valid.
 *
 * @param networkInternals - Runtime internals.
 * @param gaterIndex - Optional gater index.
 * @param createdConnection - Created connection.
 * @returns Nothing.
 */
function assignCompactGaterWhenValid(
  networkInternals: NetworkInternals,
  gaterIndex: number | null,
  createdConnection: Connection | undefined,
): void {
  if (!createdConnection || gaterIndex == null) {
    return;
  }

  if (!isNodeIndexInBounds(networkInternals.nodes, gaterIndex)) {
    console.warn(WARNING_INVALID_GATER_DURING_DESERIALIZE);
    return;
  }

  networkInternals.gate(networkInternals.nodes[gaterIndex], createdConnection);
}

/**
 * Creates empty JSON shell from runtime internals.
 *
 * @param networkInternals - Runtime internals with optional dropout.
 * @returns Empty JSON shell.
 */
function createEmptyNetworkJson(
  networkInternals: NetworkInternalsWithDropout,
): NetworkJSON {
  return {
    formatVersion: NETWORK_JSON_FORMAT_VERSION,
    input: networkInternals.input,
    output: networkInternals.output,
    dropout: resolveDropout(networkInternals.dropout),
    nodes: [],
    connections: [],
  };
}

/**
 * Resolves dropout value with numeric fallback.
 *
 * @param dropout - Optional dropout value.
 * @returns Effective dropout.
 */
function resolveDropout(dropout: number | undefined): number {
  return dropout || DEFAULT_NUMERIC_VALUE;
}

/**
 * Appends JSON nodes and their optional self-connections.
 *
 * @param networkInternals - Runtime internals.
 * @param networkJson - Target JSON accumulator.
 * @returns Nothing.
 */
function appendJsonNodesAndSelfConnections(
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
      nodeIndex,
      nodeIndex,
      selfConnection.weight,
      resolveGaterIndex(selfConnection.gater),
      isConnectionEnabled(selfConnection),
    ),
  );
}

/**
 * Appends JSON entries for forward connection list.
 *
 * @param networkInternals - Runtime internals.
 * @param networkJson - JSON accumulator.
 * @returns Nothing.
 */
function appendJsonForwardConnections(
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
        sourceIndex,
        targetIndex,
        connectionInstance.weight,
        resolveGaterIndex(connectionInstance.gater),
        isConnectionEnabled(connectionInstance),
      ),
    );
  });
}

/**
 * Creates one JSON connection entry.
 *
 * @param from - Source index.
 * @param to - Target index.
 * @param weight - Connection weight.
 * @param gater - Optional gater index.
 * @param enabled - Enabled status.
 * @returns JSON connection entry.
 */
function createJsonConnection(
  from: number,
  to: number,
  weight: number,
  gater: number | null,
  enabled: boolean,
): NetworkJSONConnection {
  return {
    from,
    to,
    weight,
    gater,
    enabled,
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
 * Validates verbose JSON payload and throws on invalid root shape.
 *
 * @param json - Payload candidate.
 * @returns Nothing.
 */
function validateNetworkJsonOrThrow(json: NetworkJSON): void {
  if (!json || typeof json !== 'object') {
    throw new Error(ERROR_INVALID_NETWORK_JSON);
  }
}

/**
 * Warns when verbose format version differs from expected.
 *
 * @param formatVersion - Incoming format version.
 * @returns Nothing.
 */
function warnWhenJsonFormatVersionIsUnknown(formatVersion: number): void {
  if (formatVersion !== NETWORK_JSON_FORMAT_VERSION) {
    console.warn(WARNING_UNKNOWN_FORMAT_VERSION);
  }
}

/**
 * Rebuilds nodes from verbose JSON payload.
 *
 * @param jsonNodeContext - JSON node rebuild context.
 * @returns Nothing.
 */
function rebuildNodesFromJsonPayload(
  jsonNodeContext: JsonNodeRebuildContext,
): void {
  jsonNodeContext.nodeJsonEntries.forEach((nodeJsonEntry, nodeIndex) => {
    const rebuiltNode = createNodeWithType(nodeJsonEntry.type);
    hydrateNodeFromJsonEntry(rebuiltNode, nodeJsonEntry, nodeIndex);
    jsonNodeContext.networkInternals.nodes.push(rebuiltNode);
  });
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
  nodeInternals.squash = resolveActivationFunction(nodeJsonEntry.squash);
  nodeInternals.index = nodeIndex;
  if (typeof nodeJsonEntry.geneId === 'number') {
    nodeInternals.geneId = nodeJsonEntry.geneId;
  }
}

/**
 * Rebuilds connections from verbose JSON payload.
 *
 * @param jsonConnectionContext - JSON connection rebuild context.
 * @returns Nothing.
 */
function rebuildConnectionsFromJsonPayload(
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

  assignJsonGaterWhenValid(
    networkInternals,
    connectionJsonEntry.gater,
    createdConnection,
  );
  assignJsonEnabledFlagWhenProvided(
    createdConnection,
    connectionJsonEntry.enabled,
  );
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
 * Checks whether index is within array bounds.
 *
 * @param nodes - Node list.
 * @param index - Candidate index.
 * @returns True when index is valid.
 */
function isNodeIndexInBounds(nodes: Node[], index: number): boolean {
  return index >= DEFAULT_NUMERIC_VALUE && index < nodes.length;
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
 * Finds activation entry by function reference.
 *
 * @param squashFunction - Activation function instance.
 * @returns Activation entry or undefined.
 */
function findActivationEntryByReference(
  squashFunction: ActivationFunction,
): [string, ActivationFunction] | undefined {
  return Object.entries(methods.Activation).find(
    ([, activationFunction]) => activationFunction === squashFunction,
  );
}

/**
 * Resolves activation name from function.name when non-empty.
 *
 * @param squashFunction - Activation function instance.
 * @returns Activation name or undefined.
 */
function resolveNamedActivationFromFunction(
  squashFunction: ActivationFunction,
): string | undefined {
  if (
    typeof squashFunction?.name === 'string' &&
    squashFunction.name.length > 0
  ) {
    return squashFunction.name;
  }
  return undefined;
}

/**
 * Resolves activation by direct key lookup.
 *
 * @param squashName - Activation key.
 * @returns Activation function or undefined.
 */
function findActivationByKey(
  squashName: string | undefined,
): ActivationFunction | undefined {
  if (!squashName) {
    return undefined;
  }
  return methods.Activation[squashName];
}

/**
 * Resolves activation by matching function.name.
 *
 * @param squashName - Activation function name.
 * @returns Activation function or undefined.
 */
function findActivationByFunctionName(
  squashName: string | undefined,
): ActivationFunction | undefined {
  const activationEntry = Object.entries(methods.Activation).find(
    ([, activationFunction]) => activationFunction.name === squashName,
  );
  return activationEntry?.[1];
}

/**
 * Warns about unknown activation and fallback to identity.
 *
 * @param squashName - Unknown activation name.
 * @returns Nothing.
 */
function warnUnknownSquashName(squashName: string | undefined): void {
  console.warn(
    `${WARNING_UNKNOWN_SQUASH_PREFIX} '${String(squashName)}' ${WARNING_UNKNOWN_SQUASH_SUFFIX}`,
  );
}

/**
 * Checks whether value is a finite index.
 *
 * @param index - Candidate index value.
 * @returns True when finite number.
 */
function isFiniteIndex(index: number): boolean {
  return Number.isFinite(index);
}

export { Connection }; // re-export for potential external tooling needing innovation IDs
