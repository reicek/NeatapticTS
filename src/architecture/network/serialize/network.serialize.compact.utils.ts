import Node from '../../node';
import Connection from '../../connection';
import {
  resolveActivationFunction,
  resolveActivationKey,
} from './network.serialize.activation.utils';
import {
  asNodeInternals,
  isNodeIndexInBounds,
} from './network.serialize.runtime.utils';
import type {
  CompactConnectionRebuildContext,
  CompactNodeRebuildContext,
  NetworkInternals,
  SerializedConnection,
} from './network.serialize.utils.types';
import {
  DEFAULT_NUMERIC_VALUE,
  FIRST_CONNECTION_INDEX,
  NODE_TYPE_HIDDEN,
  NODE_TYPE_INPUT,
  NODE_TYPE_OUTPUT,
  WARNING_INVALID_CONNECTION_DURING_DESERIALIZE,
  WARNING_INVALID_GATER_DURING_DESERIALIZE,
} from './network.serialize.utils.types';

/**
 * Refreshes `node.index` for each node in list order.
 *
 * Canonical indices are required so compact connection records can store endpoints
 * and gaters as stable numeric positions.
 *
 * @param nodes - Node list.
 * @returns Nothing.
 * @remarks
 * Call this immediately before compact export so endpoint references remain deterministic.
 *
 * Time complexity is $O(N)$.
 */
export function refreshNodeIndices(nodes: Node[]): void {
  nodes.forEach((nodeReference, nodeIndex) => {
    asNodeInternals(nodeReference).index = nodeIndex;
  });
}

/**
 * Collects node activation values in positional order.
 *
 * @param nodes - Node list.
 * @returns Activation list aligned to node indices.
 * @remarks
 * The returned array is index-stable relative to `nodes` order and is intended for compact tuple slot `0`.
 */
export function collectNodeActivations(nodes: Node[]): number[] {
  return nodes.map(
    (nodeReference) => asNodeInternals(nodeReference).activation,
  );
}

/**
 * Collects node state values in positional order.
 *
 * @param nodes - Node list.
 * @returns State list aligned to node indices.
 * @remarks
 * The returned array is intended for compact tuple slot `1`.
 */
export function collectNodeStates(nodes: Node[]): number[] {
  return nodes.map((nodeReference) => asNodeInternals(nodeReference).state);
}

/**
 * Collects node activation keys in positional order.
 *
 * Each function reference is normalized to a string key for compact transfer.
 *
 * @param nodes - Node list.
 * @returns Squash-key list aligned to node indices.
 * @remarks
 * Unknown activation references resolve to a stable fallback key (`identity`) via activation helpers.
 * This keeps compact payloads portable across runtimes where function identity can differ.
 */
export function collectNodeSquashKeys(nodes: Node[]): string[] {
  return nodes.map((nodeReference) =>
    resolveActivationKey(asNodeInternals(nodeReference).squash),
  );
}

/**
 * Collects compact connection records from forward and self connection groups.
 *
 * @param networkInternals - Runtime internals.
 * @returns Serialized connection list.
 * @remarks
 * Connection endpoints and optional gaters are stored as numeric indices.
 * Callers should refresh node indices first to avoid stale references.
 *
 * Time complexity is $O(C)$ where $C$ is total forward + self connections.
 */
export function collectSerializedConnections(
  networkInternals: NetworkInternals,
): SerializedConnection[] {
  const allConnections = collectAllConnections(networkInternals);
  return allConnections.map(serializeOneConnection);
}

/**
 * Rebuilds runtime nodes from compact payload arrays.
 *
 * Node type is inferred from index position relative to input/output boundaries.
 *
 * @param networkInternals - Runtime internals.
 * @param compactNodeContext - Compact node rebuild context.
 * @returns Nothing.
 * @remarks
 * The importer assumes aligned arrays (`activations`, `states`, `squashes`) by index.
 * Missing squash keys fall back to identity activation through activation resolver logic.
 *
 * This operation is linear in node count.
 * @example
 * ```ts
 * import { rebuildNodesFromCompactPayload } from './network.serialize.compact.utils';
 *
 * rebuildNodesFromCompactPayload(networkInternals, compactNodeContext);
 * ```
 */
export function rebuildNodesFromCompactPayload(
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
 * Rebuilds runtime connections from compact connection records.
 *
 * Invalid endpoint or gater indices are skipped with warnings to preserve import flow.
 *
 * @param compactConnectionContext - Compact connection rebuild context.
 * @returns Nothing.
 * @remarks
 * This is best-effort import behavior: one malformed connection row does not abort the whole rebuild.
 * Invalid rows are dropped and diagnostic warnings are emitted.
 * @example
 * ```ts
 * import { rebuildConnectionsFromCompactPayload } from './network.serialize.compact.utils';
 *
 * rebuildConnectionsFromCompactPayload({
 *   networkInternals,
 *   serializedConnections,
 * });
 * ```
 */
export function rebuildConnectionsFromCompactPayload(
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
