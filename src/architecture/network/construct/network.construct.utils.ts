import Connection from '../../connection';
import Group from '../../group';
import Layer from '../../layer';
import Node from '../../node';
import type Network from '../network';
import type {
  ActivationSchedule,
  ActivationSchedulingDiagnostics,
  NetworkConstructorOptions,
  NetworkTopologyIntent,
} from '../network.types';
import { computeTopoOrder } from '../topology/network.topology.utils';
import {
  NetworkConstructAmbiguousNodeIdError,
  NetworkConstructCycleModeError,
  NetworkConstructDuplicateEdgeError,
  NetworkConstructInputNodeIncomingEdgeError,
  NetworkConstructIsolatedHiddenNodeError,
  NetworkConstructMissingReferencedNodeError,
  NetworkConstructNoInputNodesError,
  NetworkConstructNoOutputNodesError,
  NetworkConstructOutputNodeGatedConnectionError,
  NetworkConstructNodeIdResolutionError,
  NetworkConstructOutputNodeOutgoingEdgeError,
  NetworkConstructSelfEdgeError,
} from './network.construct.errors';
import type {
  ConstructDiagnostics,
  ConstructGraphSnapshot,
  ConstructNodeId,
  ConstructOptions,
  ConstructPart,
  ConstructResult,
} from './network.construct.utils.types';

const DEFAULT_SCHEDULE_ITERATIONS = 1;
const INPUT_NODE_ROLE = 'input';
const OUTPUT_NODE_ROLE = 'output';
const FEED_FORWARD_TOPOLOGY_INTENT: NetworkTopologyIntent = 'feed-forward';
const UNCONSTRAINED_TOPOLOGY_INTENT: NetworkTopologyIntent = 'unconstrained';

type NetworkConstructor = new (
  input: number,
  output: number,
  options?: NetworkConstructorOptions,
) => Network;

interface NormalizedConstructOptions extends ConstructOptions {
  mode: 'acyclic' | 'recurrent';
  allowIsolatedHiddenNodes: boolean;
  validate: {
    allowOutputNodeOutgoingEdges: boolean;
    forbidDuplicateEdges: boolean;
    forbidSelfEdges: boolean;
  };
}

type RuntimeConstructNetwork = {
  nodes: Node[];
  connections: Connection[];
  selfconns: Connection[];
  gates: Connection[];
  _activationSchedule: ActivationSchedule | null;
  _activationSchedulingDiagnostics: ActivationSchedulingDiagnostics | null;
  _topoOrder: Node[] | null;
  _topoDirty: boolean;
  _slabDirty: boolean;
  _adjDirty: boolean;
  _nodeIndexDirty: boolean;
  layers?: unknown[];
};

/**
 * Construct one runnable `Network` from mixed primitive parts.
 *
 * The builder flattens `Node`, `Group`, and `Layer` inputs into one canonical
 * runtime graph, validates that every referenced edge remains inside the
 * provided part boundary, preserves explicit input/output ordering, and then
 * compiles the existing network scheduling cache in either acyclic or recurrent
 * mode.
 *
 * @param this Network constructor used to instantiate the runtime graph.
 * @param parts Mixed architecture parts to flatten.
 * @param options Optional construct-time validation, ordering, and runtime flags.
 * @returns Materialized runtime plus lightweight diagnostics and a detached graph snapshot.
 *
 * @example
 * ```ts
 * const leftSensor = new Node('input');
 * const rightSensor = new Node('input');
 * const hidden = new Group(2);
 * const readout = Layer.dense(1, 'output');
 *
 * leftSensor.describe({ label: 'leftSensor' });
 * rightSensor.describe({ label: 'rightSensor' });
 * readout.nodes[0].describe({ label: 'readout' });
 *
 * leftSensor.connect(hidden);
 * rightSensor.connect(hidden);
 * hidden.connect(readout);
 *
 * const { network } = Network.construct(
 *   [hidden, rightSensor, readout, leftSensor],
 *   {
 *     inputNodes: ['leftSensor', 'rightSensor'],
 *     outputNodes: ['readout'],
 *   },
 * );
 * ```
 */
export function constructNetwork(
  this: NetworkConstructor,
  parts: readonly ConstructPart[],
  options: ConstructOptions = {},
): ConstructResult {
  // Step 1: Normalize defaults and flatten the mixed primitive surface.
  const normalizedOptions = normalizeConstructOptions(options);
  const includedNodes = collectIncludedNodes(parts);
  const referencedConnections = collectReferencedConnections(includedNodes);

  // Step 2: Validate that every collected edge stays inside the provided part boundary.
  validateReferencedConnections(
    includedNodes,
    referencedConnections,
    normalizedOptions,
  );

  // Step 3: Resolve the public input/output ordering contract and hidden-node slice.
  const nodeLabelsByName = buildNodeLabelsByName(includedNodes);
  const inputNodes = resolveRoleNodes(
    includedNodes,
    nodeLabelsByName,
    INPUT_NODE_ROLE,
    normalizedOptions.inputNodes,
  );
  const outputNodes = resolveRoleNodes(
    includedNodes,
    nodeLabelsByName,
    OUTPUT_NODE_ROLE,
    normalizedOptions.outputNodes,
  );
  validateRolePresence(inputNodes, outputNodes);
  validateNoRoleOverlap(inputNodes, outputNodes);
  validateRoleEdgeBoundaries(
    inputNodes,
    outputNodes,
    referencedConnections,
    normalizedOptions,
  );

  const orderedNodes = createOrderedNodeList(
    includedNodes,
    inputNodes,
    outputNodes,
  );
  validateHiddenConnectivity(
    orderedNodes,
    inputNodes,
    outputNodes,
    referencedConnections,
    normalizedOptions,
  );

  // Step 4: Materialize the public runtime and compile its deterministic schedule.
  const orderedConnections = orderConnections(referencedConnections);
  const network = materializeNetwork.call(
    this,
    orderedNodes,
    orderedConnections,
    normalizedOptions,
  );
  computeTopoOrder.call(network);

  const schedulingDiagnostics = network.getActivationSchedulingDiagnostics();
  validateCompiledMode(
    normalizedOptions.mode,
    schedulingDiagnostics,
    orderedConnections,
  );

  // Step 5: Return the runtime plus lightweight construction diagnostics.
  const diagnostics = createConstructDiagnostics(
    network,
    schedulingDiagnostics,
  );

  return {
    network,
    diagnostics,
    graph: createConstructGraphSnapshot(
      network,
      orderedConnections,
      normalizedOptions.mode,
      diagnostics,
    ),
  };
}

function normalizeConstructOptions(
  options: ConstructOptions,
): NormalizedConstructOptions {
  return {
    ...options,
    mode: options.mode ?? 'acyclic',
    allowIsolatedHiddenNodes: options.allowIsolatedHiddenNodes === true,
    validate: {
      allowOutputNodeOutgoingEdges:
        options.validate?.allowOutputNodeOutgoingEdges === true,
      forbidDuplicateEdges: options.validate?.forbidDuplicateEdges !== false,
      forbidSelfEdges: options.validate?.forbidSelfEdges === true,
    },
  };
}

function collectIncludedNodes(parts: readonly ConstructPart[]): Node[] {
  const includedNodes = new Set<Node>();

  for (const part of parts) {
    if (part instanceof Node) {
      includedNodes.add(part);
      continue;
    }

    if (part instanceof Group || part instanceof Layer) {
      part.nodes.forEach((node) => {
        includedNodes.add(node);
      });
    }
  }

  return [...includedNodes];
}

function collectReferencedConnections(nodes: readonly Node[]): Connection[] {
  const referencedConnections = new Set<Connection>();

  for (const node of nodes) {
    addConnectionCollectionToSet(node.connections.in, referencedConnections);
    addConnectionCollectionToSet(node.connections.out, referencedConnections);
    addConnectionCollectionToSet(node.connections.self, referencedConnections);
    addConnectionCollectionToSet(node.connections.gated, referencedConnections);
  }

  return [...referencedConnections];
}

function addConnectionCollectionToSet(
  connections: readonly Connection[],
  referencedConnections: Set<Connection>,
): void {
  for (const connection of connections) {
    referencedConnections.add(connection);
  }
}

function buildNodeLabelsByName(
  includedNodes: readonly Node[],
): Map<string, Node[]> {
  const nodeLabelsByName = new Map<string, Node[]>();

  for (const node of includedNodes) {
    if (!node.label) {
      continue;
    }

    const labeledNodes = nodeLabelsByName.get(node.label) ?? [];
    labeledNodes.push(node);
    nodeLabelsByName.set(node.label, labeledNodes);
  }

  return nodeLabelsByName;
}

function resolveRoleNodes(
  includedNodes: readonly Node[],
  nodeLabelsByName: ReadonlyMap<string, Node[]>,
  requestedRole: typeof INPUT_NODE_ROLE | typeof OUTPUT_NODE_ROLE,
  requestedNodeIds: readonly ConstructNodeId[] | undefined,
): Node[] {
  if (!requestedNodeIds?.length) {
    return includedNodes
      .filter((node) => node.type === requestedRole)
      .toSorted(compareNodesByGeneId);
  }

  const resolvedNodes = requestedNodeIds.map((requestedNodeId) =>
    resolveRequestedNode(
      requestedNodeId,
      includedNodes,
      nodeLabelsByName,
      requestedRole,
    ),
  );

  validateUniqueResolvedNodes(resolvedNodes, requestedRole);
  validateRoleCoverage(includedNodes, resolvedNodes, requestedRole);

  return resolvedNodes;
}

function resolveRequestedNode(
  requestedNodeId: ConstructNodeId,
  includedNodes: readonly Node[],
  nodeLabelsByName: ReadonlyMap<string, Node[]>,
  requestedRole: typeof INPUT_NODE_ROLE | typeof OUTPUT_NODE_ROLE,
): Node {
  const resolvedNode =
    typeof requestedNodeId === 'number'
      ? includedNodes.find((node) => node.geneId === requestedNodeId)
      : resolveNodeByLabel(requestedNodeId, nodeLabelsByName);

  if (!resolvedNode) {
    throw new NetworkConstructNodeIdResolutionError(
      `Could not resolve ${requestedRole} node id ${formatRequestedNodeId(requestedNodeId)} from the provided parts.`,
    );
  }

  if (resolvedNode.type !== requestedRole) {
    throw new NetworkConstructNodeIdResolutionError(
      `Node ${formatNodeIdentity(resolvedNode)} cannot be used as a ${requestedRole} node because its runtime role is ${resolvedNode.type}.`,
    );
  }

  return resolvedNode;
}

function resolveNodeByLabel(
  requestedLabel: string,
  nodeLabelsByName: ReadonlyMap<string, Node[]>,
): Node | undefined {
  const matchingNodes = nodeLabelsByName.get(requestedLabel);

  if (!matchingNodes?.length) {
    return undefined;
  }

  if (matchingNodes.length > 1) {
    throw new NetworkConstructAmbiguousNodeIdError(
      `Node label ${formatRequestedNodeId(requestedLabel)} matched multiple nodes. Labels used as explicit construct ids must be unique within the provided parts.`,
    );
  }

  return matchingNodes[0];
}

function validateUniqueResolvedNodes(
  resolvedNodes: readonly Node[],
  requestedRole: typeof INPUT_NODE_ROLE | typeof OUTPUT_NODE_ROLE,
): void {
  const resolvedNodeSet = new Set(resolvedNodes);

  if (resolvedNodeSet.size === resolvedNodes.length) {
    return;
  }

  throw new NetworkConstructNodeIdResolutionError(
    `Explicit ${requestedRole} node ids must resolve to unique nodes.`,
  );
}

function validateRoleCoverage(
  includedNodes: readonly Node[],
  resolvedRoleNodes: readonly Node[],
  requestedRole: typeof INPUT_NODE_ROLE | typeof OUTPUT_NODE_ROLE,
): void {
  const resolvedRoleSet = new Set(resolvedRoleNodes);
  const uncoveredRoleNodes = includedNodes.filter(
    (node) => node.type === requestedRole && !resolvedRoleSet.has(node),
  );

  if (!uncoveredRoleNodes.length) {
    return;
  }

  throw new NetworkConstructNodeIdResolutionError(
    `Explicit ${requestedRole} node ids must cover every ${requestedRole} node in the provided parts. Missing: ${formatNodeIdentityList(uncoveredRoleNodes)}.`,
  );
}

function validateRolePresence(
  inputNodes: readonly Node[],
  outputNodes: readonly Node[],
): void {
  if (!inputNodes.length) {
    throw new NetworkConstructNoInputNodesError(
      'Constructed graph must include at least one input node.',
    );
  }

  if (!outputNodes.length) {
    throw new NetworkConstructNoOutputNodesError(
      'Constructed graph must include at least one output node.',
    );
  }
}

function validateNoRoleOverlap(
  inputNodes: readonly Node[],
  outputNodes: readonly Node[],
): void {
  const inputNodeSet = new Set(inputNodes);
  const overlappingNodes = outputNodes.filter((node) => inputNodeSet.has(node));

  if (!overlappingNodes.length) {
    return;
  }

  throw new NetworkConstructNodeIdResolutionError(
    `Input and output node selections must not overlap. Shared nodes: ${formatNodeIdentityList(overlappingNodes)}.`,
  );
}

function validateRoleEdgeBoundaries(
  inputNodes: readonly Node[],
  outputNodes: readonly Node[],
  referencedConnections: readonly Connection[],
  options: NormalizedConstructOptions,
): void {
  const inputNodeSet = new Set(inputNodes);
  const inputIncomingConnections = orderConnections(
    referencedConnections.filter((connection) =>
      inputNodeSet.has(connection.to),
    ),
  );

  if (inputIncomingConnections.length) {
    throw new NetworkConstructInputNodeIncomingEdgeError(
      'Input nodes must be pure sources in construct-from-parts graphs. ' +
        `Incoming edges: ${formatConnectionIdentityList(inputIncomingConnections)}.`,
    );
  }

  if (options.validate.allowOutputNodeOutgoingEdges) {
    return;
  }

  const outputNodeSet = new Set(outputNodes);
  const outputOutgoingConnections = orderConnections(
    referencedConnections.filter((connection) =>
      outputNodeSet.has(connection.from),
    ),
  );

  if (outputOutgoingConnections.length) {
    throw new NetworkConstructOutputNodeOutgoingEdgeError(
      'Output nodes must be pure sinks unless validate.allowOutputNodeOutgoingEdges is true. ' +
        `Outgoing edges: ${formatConnectionIdentityList(outputOutgoingConnections)}.`,
    );
  }

  const outputGatedConnections = orderConnections(
    referencedConnections.filter(
      (connection) =>
        connection.gater !== null && outputNodeSet.has(connection.gater),
    ),
  );

  if (!outputGatedConnections.length) {
    return;
  }

  throw new NetworkConstructOutputNodeGatedConnectionError(
    'Output nodes must be pure sinks unless validate.allowOutputNodeOutgoingEdges is true. ' +
      `Gated connections: ${formatGatedConnectionIdentityList(outputGatedConnections)}.`,
  );
}

function createOrderedNodeList(
  includedNodes: readonly Node[],
  inputNodes: readonly Node[],
  outputNodes: readonly Node[],
): Node[] {
  const inputNodeSet = new Set(inputNodes);
  const outputNodeSet = new Set(outputNodes);
  const hiddenNodes = includedNodes
    .filter((node) => !inputNodeSet.has(node) && !outputNodeSet.has(node))
    .toSorted(compareNodesByGeneId);

  return [...inputNodes, ...hiddenNodes, ...outputNodes];
}

function validateReferencedConnections(
  includedNodes: readonly Node[],
  referencedConnections: readonly Connection[],
  options: NormalizedConstructOptions,
): void {
  const includedNodeSet = new Set(includedNodes);
  const edgeCountsByKey = new Map<string, number>();

  for (const connection of referencedConnections) {
    validateConnectionEndpoints(connection, includedNodeSet);

    if (options.validate.forbidSelfEdges && isSelfConnection(connection)) {
      throw new NetworkConstructSelfEdgeError(
        `Constructed graph contains a self edge on node ${formatNodeIdentity(connection.from)} while self-edge validation is enabled.`,
      );
    }

    if (!options.validate.forbidDuplicateEdges) {
      continue;
    }

    const edgeKey = createEdgeKey(connection);
    const currentCount = edgeCountsByKey.get(edgeKey) ?? 0;
    edgeCountsByKey.set(edgeKey, currentCount + 1);

    if (currentCount > 0) {
      throw new NetworkConstructDuplicateEdgeError(
        `Constructed graph contains duplicate edges from ${formatNodeIdentity(connection.from)} to ${formatNodeIdentity(connection.to)}.`,
      );
    }
  }
}

function validateConnectionEndpoints(
  connection: Connection,
  includedNodeSet: ReadonlySet<Node>,
): void {
  const missingNodeMessages: string[] = [];

  if (!includedNodeSet.has(connection.from)) {
    missingNodeMessages.push(
      `source node ${formatNodeIdentity(connection.from)}`,
    );
  }

  if (!includedNodeSet.has(connection.to)) {
    missingNodeMessages.push(
      `target node ${formatNodeIdentity(connection.to)}`,
    );
  }

  if (connection.gater && !includedNodeSet.has(connection.gater)) {
    missingNodeMessages.push(
      `gater node ${formatNodeIdentity(connection.gater)}`,
    );
  }

  if (!missingNodeMessages.length) {
    return;
  }

  throw new NetworkConstructMissingReferencedNodeError(
    `Connection references ${missingNodeMessages.join(', ')} that were not included in parts.`,
  );
}

function validateHiddenConnectivity(
  orderedNodes: readonly Node[],
  inputNodes: readonly Node[],
  outputNodes: readonly Node[],
  referencedConnections: readonly Connection[],
  options: NormalizedConstructOptions,
): void {
  if (options.allowIsolatedHiddenNodes) {
    return;
  }

  const inputNodeSet = new Set(inputNodes);
  const outputNodeSet = new Set(outputNodes);
  const incidentEdgeCountByNode = new Map<Node, number>();

  for (const connection of referencedConnections) {
    incidentEdgeCountByNode.set(
      connection.from,
      (incidentEdgeCountByNode.get(connection.from) ?? 0) + 1,
    );

    if (connection.to !== connection.from) {
      incidentEdgeCountByNode.set(
        connection.to,
        (incidentEdgeCountByNode.get(connection.to) ?? 0) + 1,
      );
    }
  }

  const isolatedHiddenNodes = orderedNodes.filter((node) => {
    if (inputNodeSet.has(node) || outputNodeSet.has(node)) {
      return false;
    }

    return (incidentEdgeCountByNode.get(node) ?? 0) === 0;
  });

  if (!isolatedHiddenNodes.length) {
    return;
  }

  throw new NetworkConstructIsolatedHiddenNodeError(
    `Hidden nodes must participate in at least one edge unless allowIsolatedHiddenNodes is true. Isolated nodes: ${formatNodeIdentityList(isolatedHiddenNodes)}.`,
  );
}

function orderConnections(
  referencedConnections: readonly Connection[],
): Connection[] {
  return [...referencedConnections].toSorted(compareConnections);
}

function materializeNetwork(
  this: NetworkConstructor,
  orderedNodes: readonly Node[],
  orderedConnections: readonly Connection[],
  options: NormalizedConstructOptions,
): Network {
  // Step 1: Create one ordinary runtime network with the requested constructor-level flags.
  const network = new this(
    countNodesOfType(orderedNodes, INPUT_NODE_ROLE),
    countNodesOfType(orderedNodes, OUTPUT_NODE_ROLE),
    resolveRuntimeConstructorOptions(options),
  );
  const runtimeNetwork = network as unknown as RuntimeConstructNetwork;

  // Step 2: Replace the starter IO graph with the canonical node and edge lists.
  runtimeNetwork.nodes = [...orderedNodes];
  refreshNodeIndices(runtimeNetwork.nodes);
  runtimeNetwork.connections = orderedConnections.filter(notSelfConnection);
  runtimeNetwork.selfconns = orderedConnections.filter(isSelfConnection);
  runtimeNetwork.gates = orderedConnections.filter(
    (connection) => connection.gater !== null,
  );
  runtimeNetwork.layers = undefined;
  runtimeNetwork._activationSchedule = null;
  runtimeNetwork._activationSchedulingDiagnostics = null;
  runtimeNetwork._topoOrder = null;
  runtimeNetwork._topoDirty = true;
  runtimeNetwork._slabDirty = true;
  runtimeNetwork._adjDirty = true;
  runtimeNetwork._nodeIndexDirty = false;
  network.refreshExplicitIORoles();
  network.setTopologyIntent(resolveTopologyIntentForMode(options.mode));

  // Step 3: Reset transient activation state so the returned runtime starts clean.
  network.clear();

  return network;
}

function resolveRuntimeConstructorOptions(
  options: NormalizedConstructOptions,
): NetworkConstructorOptions {
  return {
    activationPrecision: options.activationPrecision,
    returnTypedActivations: options.returnTypedActivations,
    reuseActivationArrays: options.reuseActivationArrays,
    seed: options.seed,
    topologyIntent: resolveTopologyIntentForMode(options.mode),
  };
}

function resolveTopologyIntentForMode(
  mode: NormalizedConstructOptions['mode'],
): NetworkTopologyIntent {
  return mode === 'acyclic'
    ? FEED_FORWARD_TOPOLOGY_INTENT
    : UNCONSTRAINED_TOPOLOGY_INTENT;
}

function refreshNodeIndices(nodes: readonly Node[]): void {
  nodes.forEach((node, nodeIndex) => {
    node.index = nodeIndex;
  });
}

function countNodesOfType(
  nodes: readonly Node[],
  requestedRole: typeof INPUT_NODE_ROLE | typeof OUTPUT_NODE_ROLE,
): number {
  return nodes.filter((node) => node.type === requestedRole).length;
}

function validateCompiledMode(
  mode: NormalizedConstructOptions['mode'],
  schedulingDiagnostics: ActivationSchedulingDiagnostics,
  orderedConnections: readonly Connection[],
): void {
  if (mode !== 'acyclic' || schedulingDiagnostics.issue !== 'cycle-detected') {
    return;
  }

  const cyclePath = resolveCyclePath(
    schedulingDiagnostics.cycleNodeIds,
    orderedConnections,
  );
  const cycleNodeSuffix = cyclePath
    ? ` Cycle path: ${formatNodeIdentityPath(cyclePath)}.`
    : schedulingDiagnostics.cycleNodeIds.length
      ? ` Cycle nodes: ${schedulingDiagnostics.cycleNodeIds.join(', ')}.`
      : '';

  throw new NetworkConstructCycleModeError(
    `Constructed graph contains a cycle while mode is "acyclic". Use mode: "recurrent" or remove the reported back-connections.${cycleNodeSuffix}`,
  );
}

function resolveCyclePath(
  cycleNodeIds: readonly number[],
  orderedConnections: readonly Connection[],
): Node[] | null {
  const cycleNodeIdSet = new Set(cycleNodeIds);

  if (!cycleNodeIdSet.size) {
    return null;
  }

  const nodesByGeneId = new Map<number, Node>();
  const outgoingNodesByNode = new Map<Node, Node[]>();

  for (const connection of orderedConnections) {
    if (
      !cycleNodeIdSet.has(connection.from.geneId) ||
      !cycleNodeIdSet.has(connection.to.geneId)
    ) {
      continue;
    }

    nodesByGeneId.set(connection.from.geneId, connection.from);
    nodesByGeneId.set(connection.to.geneId, connection.to);

    const outgoingNodes = outgoingNodesByNode.get(connection.from) ?? [];
    outgoingNodes.push(connection.to);
    outgoingNodesByNode.set(connection.from, outgoingNodes);
  }

  const sortedNodes = [...nodesByGeneId.values()].toSorted(
    compareNodesByGeneId,
  );
  sortedNodes.forEach((node) => {
    const outgoingNodes = outgoingNodesByNode.get(node);

    if (!outgoingNodes?.length) {
      return;
    }

    outgoingNodesByNode.set(node, outgoingNodes.toSorted(compareNodesByGeneId));
  });

  const visitedNodes = new Set<Node>();
  const stackedNodes = new Set<Node>();
  const traversalPath: Node[] = [];

  for (const startNode of sortedNodes) {
    const cyclePath = visitNode(startNode);

    if (cyclePath) {
      return cyclePath;
    }
  }

  return null;

  function visitNode(node: Node): Node[] | null {
    if (stackedNodes.has(node)) {
      return resolveCycleFromTraversal(node, traversalPath);
    }

    if (visitedNodes.has(node)) {
      return null;
    }

    visitedNodes.add(node);
    stackedNodes.add(node);
    traversalPath.push(node);

    const outgoingNodes = outgoingNodesByNode.get(node) ?? [];

    for (const nextNode of outgoingNodes) {
      const cyclePath = visitNode(nextNode);

      if (cyclePath) {
        return cyclePath;
      }
    }

    traversalPath.pop();
    stackedNodes.delete(node);
    return null;
  }
}

function resolveCycleFromTraversal(
  repeatedNode: Node,
  traversalPath: readonly Node[],
): Node[] {
  const cycleStartIndex = traversalPath.findIndex(
    (candidateNode) => candidateNode === repeatedNode,
  );

  return [...traversalPath.slice(cycleStartIndex), repeatedNode];
}

function createConstructDiagnostics(
  network: Network,
  schedulingDiagnostics: ActivationSchedulingDiagnostics,
): ConstructDiagnostics {
  const runtimeNetwork = network as unknown as RuntimeConstructNetwork;

  return {
    nodeCount: network.nodes.length,
    edgeCount: network.connections.length + network.selfconns.length,
    detectedCycles:
      schedulingDiagnostics.issue === 'cycle-detected' ||
      schedulingDiagnostics.recurrentComponentCount > 0,
    activationOrder: resolveActivationOrderIndices(network, runtimeNetwork),
  };
}

function createConstructGraphSnapshot(
  network: Network,
  orderedConnections: readonly Connection[],
  requestedMode: NormalizedConstructOptions['mode'],
  diagnostics: ConstructDiagnostics,
): ConstructGraphSnapshot {
  const inputOrderByGeneId = createRoleOrderByGeneId(network.inputNodeIds);
  const outputOrderByGeneId = createRoleOrderByGeneId(network.outputNodeIds);

  return {
    requestedMode,
    topologyIntent: network.getTopologyIntent(),
    inputNodeIds: [...network.inputNodeIds],
    outputNodeIds: [...network.outputNodeIds],
    activationOrder: [...diagnostics.activationOrder],
    nodes: network.nodes.map((node, nodeIndex) => ({
      index: resolveConstructNodeIndex(node, nodeIndex),
      geneId: node.geneId,
      label: node.label ?? null,
      role: node.type,
      inputOrder: inputOrderByGeneId.get(node.geneId) ?? null,
      outputOrder: outputOrderByGeneId.get(node.geneId) ?? null,
    })),
    connections: orderedConnections.map((connection) => ({
      innovation: connection.innovation,
      fromIndex: resolveConstructNodeIndex(connection.from),
      toIndex: resolveConstructNodeIndex(connection.to),
      fromGeneId: connection.from.geneId,
      toGeneId: connection.to.geneId,
      gaterGeneId: connection.gater?.geneId ?? null,
      isSelfConnection: isSelfConnection(connection),
      enabled: connection.enabled,
      weight: connection.weight,
    })),
  };
}

function createRoleOrderByGeneId(
  nodeIds: readonly number[],
): Map<number, number> {
  return new Map(nodeIds.map((nodeId, roleIndex) => [nodeId, roleIndex]));
}

function resolveConstructNodeIndex(
  node: Node,
  fallbackNodeIndex?: number,
): number {
  if (typeof node.index === 'number') {
    return node.index;
  }

  if (typeof fallbackNodeIndex === 'number') {
    return fallbackNodeIndex;
  }

  throw new Error(
    `Construct graph snapshot could not resolve a runtime index for node ${formatNodeIdentity(node)}.`,
  );
}

function resolveActivationOrderIndices(
  network: Network,
  runtimeNetwork: RuntimeConstructNetwork,
): number[] {
  const nodeIndexByGeneId = new Map<number, number>();

  network.nodes.forEach((node, nodeIndex) => {
    nodeIndexByGeneId.set(node.geneId, nodeIndex);
  });

  const scheduledNodeIndices = flattenScheduledNodeIndices(
    runtimeNetwork._activationSchedule,
    nodeIndexByGeneId,
  );

  if (scheduledNodeIndices) {
    return scheduledNodeIndices;
  }

  if (runtimeNetwork._topoOrder?.length) {
    return runtimeNetwork._topoOrder.flatMap((node) => {
      return typeof node.index === 'number' ? [node.index] : [];
    });
  }

  return network.nodes.map((_node, nodeIndex) => nodeIndex);
}

function flattenScheduledNodeIndices(
  activationSchedule: ActivationSchedule | null,
  nodeIndexByGeneId: ReadonlyMap<number, number>,
): number[] | null {
  if (!activationSchedule) {
    return null;
  }

  const scheduledNodeIndices: number[] = [];

  for (const activationStep of activationSchedule.steps) {
    const iterationCount =
      activationStep.kind === 'recurrent-component'
        ? (activationStep.iterations ?? DEFAULT_SCHEDULE_ITERATIONS)
        : DEFAULT_SCHEDULE_ITERATIONS;

    for (
      let iterationIndex = 0;
      iterationIndex < iterationCount;
      iterationIndex++
    ) {
      for (const nodeId of activationStep.nodeIds) {
        const nodeIndex = nodeIndexByGeneId.get(nodeId);

        if (nodeIndex === undefined) {
          return null;
        }

        scheduledNodeIndices.push(nodeIndex);
      }
    }
  }

  return scheduledNodeIndices;
}

function compareNodesByGeneId(leftNode: Node, rightNode: Node): number {
  return leftNode.geneId - rightNode.geneId;
}

function compareConnections(
  leftConnection: Connection,
  rightConnection: Connection,
): number {
  return (
    leftConnection.from.geneId - rightConnection.from.geneId ||
    leftConnection.to.geneId - rightConnection.to.geneId ||
    leftConnection.innovation - rightConnection.innovation
  );
}

function isSelfConnection(connection: Connection): boolean {
  return connection.from === connection.to;
}

function notSelfConnection(connection: Connection): boolean {
  return !isSelfConnection(connection);
}

function createEdgeKey(connection: Connection): string {
  return `${connection.from.geneId}->${connection.to.geneId}`;
}

function formatRequestedNodeId(requestedNodeId: ConstructNodeId): string {
  return typeof requestedNodeId === 'string'
    ? `"${requestedNodeId}"`
    : `${requestedNodeId}`;
}

function formatNodeIdentity(node: Node): string {
  return node.label ? `"${node.label}"` : `geneId:${node.geneId}`;
}

function formatConnectionIdentity(connection: Connection): string {
  return `${formatNodeIdentity(connection.from)} -> ${formatNodeIdentity(connection.to)}`;
}

function formatGatedConnectionIdentity(connection: Connection): string {
  if (!connection.gater) {
    return formatConnectionIdentity(connection);
  }

  return `${formatNodeIdentity(connection.gater)} gates ${formatConnectionIdentity(connection)}`;
}

function formatConnectionIdentityList(
  connections: readonly Connection[],
): string {
  return connections
    .map((connection) => formatConnectionIdentity(connection))
    .join(', ');
}

function formatGatedConnectionIdentityList(
  connections: readonly Connection[],
): string {
  return connections
    .map((connection) => formatGatedConnectionIdentity(connection))
    .join(', ');
}

function formatNodeIdentityPath(nodes: readonly Node[]): string {
  return nodes.map((node) => formatNodeIdentity(node)).join(' -> ');
}

function formatNodeIdentityList(nodes: readonly Node[]): string {
  return nodes.map((node) => formatNodeIdentity(node)).join(', ');
}
