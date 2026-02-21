import type Network from '../../network';
import type Connection from '../../connection';
import Node from '../../node';
import mutation from '../../../methods/mutation';
import { config } from '../../../config';
import type {
  BackwardCandidateTraversalContext,
  ConnectionGroupReinitContext,
  ConnectionSplitResult,
  DeterministicChainMutationContext,
  DirectionalConnectionContext,
  DistinctNodePair,
  ForwardCandidateTraversalContext,
  InputOutputEndpoints,
  MutationHandler,
  MutationMethod,
  MutationMethodObject,
  NetworkMutationProps,
  NodePair,
  RecurrentLayerShape,
  SourcePeerConnectionCountContext,
  TargetLayerPeerContext,
  WeightSamplingRangeContext,
} from '../network.types';
export type { MutationMethod } from '../network.types';

/**
 * Canonical node-type literal for input nodes.
 */
const NODE_TYPE_INPUT: Node['type'] = 'input';

/**
 * Canonical node-type literal for output nodes.
 */
const NODE_TYPE_OUTPUT: Node['type'] = 'output';

/**
 * Canonical node-type literal for hidden nodes.
 */
const NODE_TYPE_HIDDEN: Node['type'] = 'hidden';

/**
 * Canonical recurrent block literal for LSTM expansion.
 */
const RECURRENT_BLOCK_LSTM = 'lstm' as const;

/**
 * Canonical recurrent block literal for GRU expansion.
 */
const RECURRENT_BLOCK_GRU = 'gru' as const;

/**
 * Width used when creating a minimal recurrent block.
 */
const SINGLE_UNIT_RECURRENT_BLOCK_WIDTH = 1;

/**
 * Weight delta used to keep mutation side effects numerically observable.
 */
const SUB_NODE_STABILITY_WEIGHT_DELTA = 1e-4;

/**
 * Threshold used for random 50/50 gating decisions.
 */
const GATE_REASSIGN_THRESHOLD = 0.5;

/**
 * Default minimum mutation value when no method override is provided.
 */
const DEFAULT_MUTATION_MIN = -1;

/**
 * Default maximum mutation value when no method override is provided.
 */
const DEFAULT_MUTATION_MAX = 1;

/**
 * Minimum redundant in/out degree required before removing a connection.
 */
const MIN_REDUNDANT_CONNECTION_COUNT = 1;

/**
 * Minimum node count required to perform swap-node mutation.
 */
const MIN_SWAPPABLE_NODE_COUNT = 2;

/**
 * Message emitted when no hidden node can be removed.
 */
const WARNING_NO_HIDDEN_NODES_TO_REMOVE = 'No hidden nodes left to remove!';

/**
 * Message emitted when activation mutation has no eligible nodes.
 */
const WARNING_NO_ACTIVATION_MUTATION_TARGETS =
  'No nodes available for activation function mutation based on config.';

/**
 * Message emitted when all self-connection candidates are already occupied.
 */
const WARNING_SELF_CONNECTIONS_ALREADY_PRESENT =
  'All eligible nodes already have self-connections.';

/**
 * Message emitted when no self-connections are available to remove.
 */
const WARNING_NO_SELF_CONNECTIONS_TO_REMOVE =
  'No self-connections exist to remove.';

/**
 * Message emitted when gating cannot be added because all are already gated.
 */
const WARNING_ALL_CONNECTIONS_GATED = 'All connections are already gated.';

/**
 * Message emitted when no gate exists to remove.
 */
const WARNING_NO_GATED_CONNECTIONS_TO_REMOVE =
  'No gated connections to ungate.';

/**
 * Prefix for unknown-mutation warning logs.
 */
const UNKNOWN_MUTATION_WARNING_PREFIX =
  '[mutate] Unknown mutation method ignored:';

/**
 * Error emitted when mutate is called without a valid method.
 */
const ERROR_NO_MUTATE_METHOD = 'No (correct) mutate method given!';

/**
 * Module path used to resolve Layer factory lazily.
 */
const LAYER_MODULE_PATH = '../../layer';

/**
 * Internal node field used to enable batch normalization.
 */
const BATCH_NORM_FLAG_KEY = '_batchNorm';

/**
 * Mutation dispatch table keyed by mutation identity.
 */
const MUTATION_DISPATCH: Record<string, MutationHandler> = {
  ADD_NODE: addNode,
  SUB_NODE: subNode,
  ADD_CONN: addConn,
  SUB_CONN: subConn,
  MOD_WEIGHT: modWeight,
  MOD_BIAS: modBias,
  MOD_ACTIVATION: modActivation,
  ADD_SELF_CONN: addSelfConn,
  SUB_SELF_CONN: subSelfConn,
  ADD_GATE: addGate,
  SUB_GATE: subGate,
  ADD_BACK_CONN: addBackConn,
  SUB_BACK_CONN: subBackConn,
  SWAP_NODES: swapNodes,
  ADD_LSTM_NODE: addLSTMNode,
  ADD_GRU_NODE: addGRUNode,
  REINIT_WEIGHT: reinitWeight,
  BATCH_NORM: batchNorm,
};

/**
 * Public entry point: apply a single mutation operator to the network.
 *
 * @param this - Network instance.
 * @param method - Mutation enum value or descriptor object.
 * @returns Nothing.
 */
export function mutateImpl(this: Network, method?: MutationMethod): void {
  if (method == null) {
    throw new Error(ERROR_NO_MUTATE_METHOD);
  }

  const mutationKey = resolveMutationKey(method);
  const mutationHandler = mutationKey
    ? MUTATION_DISPATCH[mutationKey]
    : undefined;

  if (!mutationHandler) {
    warnUnknownMutation(mutationKey);
    return;
  }

  mutationHandler.call(this, method);
  asMutationProps(this)._topoDirty = true;
}

/**
 * Converts a network to its internal mutation runtime shape.
 *
 * @param network - Network to convert.
 * @returns Runtime mutation props.
 */
function asMutationProps(network: Network): NetworkMutationProps {
  return network as unknown as NetworkMutationProps;
}

/**
 * Resolves a mutation dispatch key from user-provided method input.
 *
 * @param method - Mutation method input.
 * @returns Dispatch key or undefined.
 */
function resolveMutationKey(method: MutationMethod): string | undefined {
  if (isMutationMethodKeyString(method)) {
    return method;
  }

  return resolveMutationKeyFromObject(method);
}

/**
 * Checks whether mutation input is already a direct key string.
 *
 * @param method - Mutation method input.
 * @returns True when method is a key string.
 */
function isMutationMethodKeyString(method: MutationMethod): method is string {
  return typeof method === 'string';
}

/**
 * Resolves mutation key from object-form descriptor.
 *
 * @param methodObject - Mutation method object.
 * @returns Dispatch key or undefined.
 */
function resolveMutationKeyFromObject(
  methodObject: MutationMethodObject,
): string | undefined {
  const directKey = resolveDirectMutationKey(methodObject);
  if (directKey) {
    return directKey;
  }

  return findMutationKeyByIdentityReference(methodObject);
}

/**
 * Resolves direct object fields that can represent a mutation key.
 *
 * @param methodObject - Mutation method object.
 * @returns Direct key or undefined.
 */
function resolveDirectMutationKey(
  methodObject: MutationMethodObject,
): string | undefined {
  return methodObject.name ?? methodObject.type ?? methodObject.identity;
}

/**
 * Resolves a mutation key by direct identity-reference comparison.
 *
 * @param method - Mutation object reference.
 * @returns Matching mutation key or undefined.
 */
function findMutationKeyByIdentityReference(
  method: MutationMethod,
): string | undefined {
  const mutationMethods = mutation as Record<string, unknown>;
  const mutationKeys = Object.keys(mutationMethods);

  for (
    let mutationKeyIndex = 0;
    mutationKeyIndex < mutationKeys.length;
    mutationKeyIndex++
  ) {
    const mutationKey = mutationKeys[mutationKeyIndex];
    if (method === mutationMethods[mutationKey]) {
      return mutationKey;
    }
  }

  return undefined;
}

/**
 * Emits unknown-mutation warning when configured.
 *
 * @param mutationKey - Resolved mutation key.
 * @returns Nothing.
 */
function warnUnknownMutation(mutationKey?: string): void {
  if (!config.warnings) {
    return;
  }
  console.warn(UNKNOWN_MUTATION_WARNING_PREFIX, mutationKey);
}

/**
 * Marks topology caches dirty when acyclic mode is enforced.
 *
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function markTopoDirtyIfAcyclic(mutationProps: NetworkMutationProps): void {
  if (mutationProps._enforceAcyclic) {
    mutationProps._topoDirty = true;
  }
}

/**
 * Returns the first node by type.
 *
 * @param network - Target network.
 * @param nodeType - Node type to match.
 * @returns Matching node or undefined.
 */
function findFirstNodeByType(
  network: Network,
  nodeType: Node['type'],
): Node | undefined {
  for (let nodeIndex = 0; nodeIndex < network.nodes.length; nodeIndex++) {
    const currentNode = network.nodes[nodeIndex];
    if (currentNode.type === nodeType) {
      return currentNode;
    }
  }
  return undefined;
}

/**
 * Selects a random array element.
 *
 * @param entries - Source entries.
 * @param randomValue - Random generator.
 * @returns Random entry or undefined when empty.
 */
function pickRandomEntry<T>(
  entries: T[],
  randomValue: () => number,
): T | undefined {
  if (entries.length === 0) {
    return undefined;
  }
  const selectedIndex = Math.floor(randomValue() * entries.length);
  return entries[selectedIndex];
}

/**
 * Creates a hidden node with random activation mutation.
 *
 * @param randomValue - Random generator.
 * @returns Hidden node.
 */
function createHiddenNode(randomValue: () => number): Node {
  const hiddenNode = new Node(NODE_TYPE_HIDDEN, undefined, randomValue);
  hiddenNode.mutate(mutation.MOD_ACTIVATION);
  return hiddenNode;
}

/**
 * Inserts a node before output tail while preserving output block ordering.
 *
 * @param network - Target network.
 * @param nodeToInsert - Node to insert.
 * @param targetNode - Target node for insertion alignment.
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function insertNodeBeforeOutputTail(
  network: Network,
  nodeToInsert: Node,
  targetNode: Node,
  mutationProps: NetworkMutationProps,
): void {
  const targetIndex = network.nodes.indexOf(targetNode);
  const insertionIndex = Math.min(
    targetIndex,
    network.nodes.length - network.output,
  );
  network.nodes.splice(insertionIndex, 0, nodeToInsert);
  mutationProps._nodeIndexDirty = true;
}

/**
 * Gets a connection between two nodes when it exists.
 *
 * @param network - Target network.
 * @param fromNode - Source node.
 * @param toNode - Target node.
 * @returns Matching connection or undefined.
 */
function findConnection(
  network: Network,
  fromNode: Node,
  toNode: Node,
): Connection | undefined {
  for (
    let connectionIndex = 0;
    connectionIndex < network.connections.length;
    connectionIndex++
  ) {
    const currentConnection = network.connections[connectionIndex];
    if (
      currentConnection.from === fromNode &&
      currentConnection.to === toNode
    ) {
      return currentConnection;
    }
  }
  return undefined;
}

/**
 * Ensures a connection exists and returns it.
 *
 * @param network - Target network.
 * @param fromNode - Source node.
 * @param toNode - Target node.
 * @returns Existing or created connection.
 */
function ensureConnection(
  network: Network,
  fromNode: Node,
  toNode: Node,
): Connection | undefined {
  const existingConnection = findConnection(network, fromNode, toNode);
  if (existingConnection) {
    return existingConnection;
  }
  const createdConnections = network.connect(fromNode, toNode);
  return createdConnections.at(0);
}

/**
 * Disconnects a connection pair while suppressing errors.
 *
 * @param network - Target network.
 * @param connection - Connection to remove.
 * @returns Nothing.
 */
function tryDisconnectConnection(
  network: Network,
  connection: Connection,
): void {
  try {
    network.disconnect(connection.from, connection.to);
  } catch {
    return;
  }
}

/**
 * ADD_NODE mutation orchestrator.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addNode(this: Network): void {
  const mutationProps = asMutationProps(this);
  markTopoDirtyIfAcyclic(mutationProps);

  if (config.deterministicChainMode) {
    addNodeDeterministicChain(this, mutationProps);
    return;
  }

  addNodeRandomSplit(this, mutationProps);
}

/**
 * Applies deterministic chain-growth ADD_NODE mutation.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function addNodeDeterministicChain(
  network: Network,
  mutationProps: NetworkMutationProps,
): void {
  const deterministicContext = resolveDeterministicChainMutationContext(
    network,
    mutationProps,
  );
  if (!deterministicContext) {
    return;
  }

  const splitResult = splitConnectionThroughHiddenNode(
    network,
    mutationProps,
    deterministicContext.terminalConnection,
  );
  deterministicContext.deterministicChain.push(splitResult.hiddenNode);

  tryReassignGateAfterSplit(network, mutationProps._rand, splitResult);
  pruneDeterministicChainExtraEdges(
    network,
    deterministicContext.deterministicChain,
    deterministicContext.outputNode,
  );
}

/**
 * Resolves all deterministic add-node prerequisites into one context object.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @returns Deterministic context or undefined when any prerequisite fails.
 */
function resolveDeterministicChainMutationContext(
  network: Network,
  mutationProps: NetworkMutationProps,
): DeterministicChainMutationContext | undefined {
  const endpointNodes = resolveInputOutputEndpoints(network);
  if (!endpointNodes) {
    return undefined;
  }

  initializeDeterministicChain(
    network,
    mutationProps,
    endpointNodes.inputNode,
    endpointNodes.outputNode,
  );
  const deterministicChain = mutationProps._detChain;
  if (!deterministicChain || deterministicChain.length === 0) {
    return undefined;
  }

  const tailNode = deterministicChain.at(-1);
  if (!tailNode) {
    return undefined;
  }

  const terminalConnection = ensureConnection(
    network,
    tailNode,
    endpointNodes.outputNode,
  );
  if (!terminalConnection) {
    return undefined;
  }

  return {
    deterministicChain,
    outputNode: endpointNodes.outputNode,
    terminalConnection,
  };
}

/**
 * Resolves input/output endpoints required for seed and deterministic flows.
 *
 * @param network - Target network.
 * @returns Endpoint nodes or undefined when missing.
 */
function resolveInputOutputEndpoints(
  network: Network,
): InputOutputEndpoints | undefined {
  const inputNode = findFirstNodeByType(network, NODE_TYPE_INPUT);
  const outputNode = findFirstNodeByType(network, NODE_TYPE_OUTPUT);
  if (!inputNode || !outputNode) {
    return undefined;
  }
  return { inputNode, outputNode };
}

/**
 * Initializes deterministic chain storage and seed edge.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @param inputNode - Input node.
 * @param outputNode - Output node.
 * @returns Nothing.
 */
function initializeDeterministicChain(
  network: Network,
  mutationProps: NetworkMutationProps,
  inputNode: Node,
  outputNode: Node,
): void {
  if (mutationProps._detChain) {
    return;
  }

  ensureConnection(network, inputNode, outputNode);
  mutationProps._detChain = [inputNode];
}

/**
 * Prunes side edges from chain nodes to preserve linear deterministic depth.
 *
 * @param network - Target network.
 * @param deterministicChain - Chain node list.
 * @param outputNode - Output node.
 * @returns Nothing.
 */
function pruneDeterministicChainExtraEdges(
  network: Network,
  deterministicChain: Node[],
  outputNode: Node,
): void {
  deterministicChain.forEach((chainNode, chainNodeIndex) => {
    const expectedTargetNode = resolveExpectedChainTarget(
      deterministicChain,
      chainNodeIndex,
      outputNode,
    );
    disconnectUnexpectedOutgoingConnections(
      network,
      chainNode,
      expectedTargetNode,
    );
  });
}

/**
 * Resolves the expected outgoing target for a chain node position.
 *
 * @param deterministicChain - Chain node list.
 * @param chainNodeIndex - Current chain index.
 * @param outputNode - Terminal output node.
 * @returns Expected successor target.
 */
function resolveExpectedChainTarget(
  deterministicChain: Node[],
  chainNodeIndex: number,
  outputNode: Node,
): Node {
  return chainNodeIndex + 1 < deterministicChain.length
    ? deterministicChain[chainNodeIndex + 1]
    : outputNode;
}

/**
 * Removes outgoing connections that do not match the expected chain target.
 *
 * @param network - Target network.
 * @param chainNode - Node whose outgoing edges are validated.
 * @param expectedTargetNode - Allowed outgoing target.
 * @returns Nothing.
 */
function disconnectUnexpectedOutgoingConnections(
  network: Network,
  chainNode: Node,
  expectedTargetNode: Node,
): void {
  const outgoingConnections = [...chainNode.connections.out];
  outgoingConnections.forEach((candidateConnection) => {
    if (candidateConnection.to !== expectedTargetNode) {
      tryDisconnectConnection(network, candidateConnection);
    }
  });
}

/**
 * Applies non-deterministic ADD_NODE by splitting a random connection.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function addNodeRandomSplit(
  network: Network,
  mutationProps: NetworkMutationProps,
): void {
  if (!ensureSeedForwardConnectionWhenEmpty(network)) {
    return;
  }

  const selectedConnection = pickRandomEntry(
    network.connections,
    mutationProps._rand,
  );
  if (!selectedConnection) {
    return;
  }

  const splitResult = splitConnectionThroughHiddenNode(
    network,
    mutationProps,
    selectedConnection,
  );
  tryReassignGateAfterSplit(network, mutationProps._rand, splitResult);
}

/**
 * Replaces one connection by inserting a hidden node and reconnecting edges.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @param connectionToSplit - Connection to split.
 * @returns Split result values.
 */
function splitConnectionThroughHiddenNode(
  network: Network,
  mutationProps: NetworkMutationProps,
  connectionToSplit: Connection,
): ConnectionSplitResult {
  const previousGater = connectionToSplit.gater;
  network.disconnect(connectionToSplit.from, connectionToSplit.to);

  const hiddenNode = createHiddenNode(mutationProps._rand);
  insertNodeBeforeOutputTail(
    network,
    hiddenNode,
    connectionToSplit.to,
    mutationProps,
  );

  const sourceToHiddenConnection = ensureConnection(
    network,
    connectionToSplit.from,
    hiddenNode,
  );
  const hiddenToTargetConnection = ensureConnection(
    network,
    hiddenNode,
    connectionToSplit.to,
  );
  mutationProps._preferredChainEdge = hiddenToTargetConnection;

  return {
    hiddenNode,
    previousGater,
    sourceToHiddenConnection,
    hiddenToTargetConnection,
  };
}

/**
 * Reassigns prior gater to one of the new split connections when possible.
 *
 * @param network - Target network.
 * @param randomValue - Random generator.
 * @param splitResult - Split result values.
 * @returns Nothing.
 */
function tryReassignGateAfterSplit(
  network: Network,
  randomValue: () => number,
  splitResult: ConnectionSplitResult,
): void {
  if (
    !splitResult.previousGater ||
    !splitResult.sourceToHiddenConnection ||
    !splitResult.hiddenToTargetConnection
  ) {
    return;
  }

  const gatedConnection =
    randomValue() >= GATE_REASSIGN_THRESHOLD
      ? splitResult.sourceToHiddenConnection
      : splitResult.hiddenToTargetConnection;
  network.gate(splitResult.previousGater, gatedConnection);
}

/**
 * Ensures a seed input->output connection exists when connection list is empty.
 *
 * @param network - Target network.
 * @returns True when mutation may continue.
 */
function ensureSeedForwardConnectionWhenEmpty(network: Network): boolean {
  if (network.connections.length > 0) {
    return true;
  }

  const endpointNodes = resolveInputOutputEndpoints(network);
  if (!endpointNodes) {
    return false;
  }

  ensureConnection(network, endpointNodes.inputNode, endpointNodes.outputNode);
  return network.connections.length > 0;
}

/**
 * SUB_NODE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function subNode(this: Network): void {
  const hiddenNodes = collectNodesByType(this, NODE_TYPE_HIDDEN);
  if (hiddenNodes.length === 0) {
    warnWhenEnabled(WARNING_NO_HIDDEN_NODES_TO_REMOVE);
    return;
  }

  const selectedHiddenNode = selectHiddenNodeForRemoval(this, hiddenNodes);
  if (!selectedHiddenNode) {
    return;
  }

  removeHiddenNodeAndApplyStabilityNudge(this, selectedHiddenNode);
}

/**
 * Selects a hidden node candidate for SUB_NODE mutation.
 *
 * @param network - Target network.
 * @param hiddenNodes - Hidden nodes.
 * @returns Selected hidden node.
 */
function selectHiddenNodeForRemoval(
  network: Network,
  hiddenNodes: Node[],
): Node | undefined {
  return pickRandomEntry(hiddenNodes, asMutationProps(network)._rand);
}

/**
 * Removes selected hidden node and applies stability nudge.
 *
 * @param network - Target network.
 * @param hiddenNode - Hidden node to remove.
 * @returns Nothing.
 */
function removeHiddenNodeAndApplyStabilityNudge(
  network: Network,
  hiddenNode: Node,
): void {
  network.remove(hiddenNode);
  applyFirstConnectionStabilityNudge(network);
}

/**
 * Applies tiny stability nudge to the first remaining connection.
 *
 * @param network - Target network.
 * @returns Nothing.
 */
function applyFirstConnectionStabilityNudge(network: Network): void {
  const firstConnection = network.connections.at(0);
  if (!firstConnection) {
    return;
  }
  firstConnection.weight += SUB_NODE_STABILITY_WEIGHT_DELTA;
}

/**
 * Collects nodes by type.
 *
 * @param network - Source network.
 * @param nodeType - Desired node type.
 * @returns Matching nodes.
 */
function collectNodesByType(network: Network, nodeType: Node['type']): Node[] {
  const collectedNodes: Node[] = [];
  for (let nodeIndex = 0; nodeIndex < network.nodes.length; nodeIndex++) {
    const candidateNode = network.nodes[nodeIndex];
    if (candidateNode.type === nodeType) {
      collectedNodes.push(candidateNode);
    }
  }
  return collectedNodes;
}

/**
 * Emits a warning when warning mode is enabled.
 *
 * @param message - Warning message.
 * @returns Nothing.
 */
function warnWhenEnabled(message: string): void {
  if (config.warnings) {
    console.warn(message);
  }
}

/**
 * ADD_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addConn(this: Network): void {
  const mutationProps = asMutationProps(this);
  markTopoDirtyIfAcyclic(mutationProps);

  const selectedConnectionPair = resolveSelectedForwardConnectionPair(
    this,
    mutationProps._rand,
  );
  if (!selectedConnectionPair) {
    return;
  }

  connectPair(this, selectedConnectionPair);
}

/**
 * Resolves random selected forward connection pair.
 *
 * @param network - Target network.
 * @param randomValue - Random generator.
 * @returns Selected source/target pair.
 */
function resolveSelectedForwardConnectionPair(
  network: Network,
  randomValue: () => number,
): NodePair | undefined {
  const connectionCandidates = collectForwardConnectionCandidates(network);
  return pickRandomEntry(connectionCandidates, randomValue);
}

/**
 * Collects forward connection candidates.
 *
 * @param network - Target network.
 * @returns Candidate source/target pairs.
 */
function collectForwardConnectionCandidates(network: Network): NodePair[] {
  const traversalContexts = collectForwardTraversalContexts(network);
  return traversalContexts.reduce<NodePair[]>(
    collectForwardCandidatesFromContext,
    [],
  );
}

/**
 * Collects forward traversal contexts for all eligible source nodes.
 *
 * @param network - Target network.
 * @returns Forward traversal contexts.
 */
function collectForwardTraversalContexts(
  network: Network,
): ForwardCandidateTraversalContext[] {
  const forwardTraversalContexts: ForwardCandidateTraversalContext[] = [];

  for (
    let sourceNodeIndex = 0;
    sourceNodeIndex < network.nodes.length - network.output;
    sourceNodeIndex++
  ) {
    forwardTraversalContexts.push(
      createForwardCandidateTraversalContext(network, sourceNodeIndex),
    );
  }

  return forwardTraversalContexts;
}

/**
 * Reduces one forward traversal context into candidate connection pairs.
 *
 * @param forwardConnectionCandidates - Existing candidate pairs.
 * @param traversalContext - Source traversal context.
 * @returns Updated candidate pairs.
 */
function collectForwardCandidatesFromContext(
  forwardConnectionCandidates: NodePair[],
  traversalContext: ForwardCandidateTraversalContext,
): NodePair[] {
  const sourceCandidates = collectForwardCandidatesForSource(traversalContext);
  forwardConnectionCandidates.push(...sourceCandidates);
  return forwardConnectionCandidates;
}

/**
 * Creates context for one forward-candidate source traversal pass.
 *
 * @param network - Target network.
 * @param sourceNodeIndex - Current source index.
 * @returns Immutable traversal context.
 */
function createForwardCandidateTraversalContext(
  network: Network,
  sourceNodeIndex: number,
): ForwardCandidateTraversalContext {
  const sourceNode = network.nodes[sourceNodeIndex];
  const targetStartIndex = Math.max(sourceNodeIndex + 1, network.input);
  return {
    network,
    sourceNodeIndex,
    sourceNode,
    targetStartIndex,
  };
}

/**
 * Appends forward candidates for a single source-node traversal context.
 *
 * @param traversalContext - Source traversal context.
 * @param forwardConnectionCandidates - Collector array.
 * @returns Nothing.
 */
function appendForwardCandidatesForSource(
  traversalContext: ForwardCandidateTraversalContext,
  forwardConnectionCandidates: NodePair[],
): void {
  forwardConnectionCandidates.push(
    ...collectForwardCandidatesForSource(traversalContext),
  );
}

/**
 * Collects all forward candidates for one source traversal context.
 *
 * @param traversalContext - Source traversal context.
 * @returns Candidate source/target pairs.
 */
function collectForwardCandidatesForSource(
  traversalContext: ForwardCandidateTraversalContext,
): NodePair[] {
  const sourceCandidates: NodePair[] = [];

  for (
    let targetNodeIndex = traversalContext.targetStartIndex;
    targetNodeIndex < traversalContext.network.nodes.length;
    targetNodeIndex++
  ) {
    const targetNode = traversalContext.network.nodes[targetNodeIndex];
    if (
      !isForwardCandidateTargetAvailable(
        traversalContext.sourceNode,
        targetNode,
      )
    ) {
      continue;
    }
    sourceCandidates.push([traversalContext.sourceNode, targetNode]);
  }

  return sourceCandidates;
}

/**
 * Checks whether a forward candidate target is not already projected.
 *
 * @param sourceNode - Candidate source node.
 * @param targetNode - Candidate target node.
 * @returns True when connection may be added.
 */
function isForwardCandidateTargetAvailable(
  sourceNode: Node,
  targetNode: Node,
): boolean {
  return !sourceNode.isProjectingTo(targetNode);
}

/**
 * SUB_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function subConn(this: Network): void {
  const removableConnections = collectRemovableForwardConnections(this);
  const selectedConnection = pickRandomEntry(
    removableConnections,
    asMutationProps(this)._rand,
  );
  if (!selectedConnection) {
    return;
  }

  disconnectConnectionPair(this, selectedConnection);
}

/**
 * Disconnects selected connection by endpoints.
 *
 * @param network - Target network.
 * @param selectedConnection - Connection to disconnect.
 * @returns Nothing.
 */
function disconnectConnectionPair(
  network: Network,
  selectedConnection: Connection,
): void {
  network.disconnect(selectedConnection.from, selectedConnection.to);
}

/**
 * Collects removable forward connections using redundancy constraints.
 *
 * @param network - Target network.
 * @returns Removable forward connections.
 */
function collectRemovableForwardConnections(network: Network): Connection[] {
  return network.connections.filter((candidateConnection) =>
    isRemovableForwardConnection(network, candidateConnection),
  );
}

/**
 * Evaluates whether a forward connection is safe to remove.
 *
 * @param network - Target network.
 * @param candidateConnection - Connection under evaluation.
 * @returns True when removable.
 */
function isRemovableForwardConnection(
  network: Network,
  candidateConnection: Connection,
): boolean {
  if (!isForwardConnectionStructurallyRemovable(network, candidateConnection)) {
    return false;
  }

  return !wouldDisconnectTargetPeerLayerGroup(network, candidateConnection);
}

/**
 * Checks structural preconditions for removable forward connections.
 *
 * @param network - Target network.
 * @param candidateConnection - Candidate connection.
 * @returns True when the connection is a forward edge with redundant endpoints.
 */
function isForwardConnectionStructurallyRemovable(
  network: Network,
  candidateConnection: Connection,
): boolean {
  const directionContext = createDirectionalConnectionContext(
    network,
    candidateConnection,
  );
  if (!hasRedundantEndpoints(directionContext.candidateConnection)) {
    return false;
  }
  return isForwardDirectionalContext(directionContext);
}

/**
 * Creates indexed directional context for a connection candidate.
 *
 * @param network - Target network.
 * @param candidateConnection - Candidate connection.
 * @returns Directional context.
 */
function createDirectionalConnectionContext(
  network: Network,
  candidateConnection: Connection,
): DirectionalConnectionContext {
  const fromNodeIndex = network.nodes.indexOf(candidateConnection.from);
  const toNodeIndex = network.nodes.indexOf(candidateConnection.to);
  return { network, candidateConnection, fromNodeIndex, toNodeIndex };
}

/**
 * Checks whether both endpoints maintain at least one redundant edge.
 *
 * @param candidateConnection - Candidate connection.
 * @returns True when endpoint redundancy exists.
 */
function hasRedundantEndpoints(candidateConnection: Connection): boolean {
  const sourceHasMultipleOutgoing =
    candidateConnection.from.connections.out.length >
    MIN_REDUNDANT_CONNECTION_COUNT;
  const targetHasMultipleIncoming =
    candidateConnection.to.connections.in.length >
    MIN_REDUNDANT_CONNECTION_COUNT;
  return sourceHasMultipleOutgoing && targetHasMultipleIncoming;
}

/**
 * Checks whether a directional context represents a forward edge.
 *
 * @param directionContext - Directional context.
 * @returns True when forward.
 */
function isForwardDirectionalContext(
  directionContext: DirectionalConnectionContext,
): boolean {
  return directionContext.toNodeIndex > directionContext.fromNodeIndex;
}

/**
 * Determines whether removal would disconnect a target peer-layer group.
 *
 * @param network - Target network.
 * @param candidateConnection - Connection under evaluation.
 * @returns True when peer group would be disconnected.
 */
function wouldDisconnectTargetPeerLayerGroup(
  network: Network,
  candidateConnection: Connection,
): boolean {
  const targetLayerPeers = collectTargetLayerPeers(
    network,
    candidateConnection.to,
  );
  if (targetLayerPeers.length === 0) {
    return false;
  }

  const peerConnectionsFromSource = countSourceConnectionsIntoPeerSet(
    network,
    candidateConnection,
    targetLayerPeers,
  );
  return peerConnectionsFromSource <= MIN_REDUNDANT_CONNECTION_COUNT;
}

/**
 * Counts source-originated connections that end inside the target peer set.
 *
 * @param network - Target network.
 * @param candidateConnection - Candidate connection.
 * @param targetLayerPeers - Peer-set nodes.
 * @returns Number of source-to-peer connections.
 */
function countSourceConnectionsIntoPeerSet(
  network: Network,
  candidateConnection: Connection,
  targetLayerPeers: Node[],
): number {
  const countContext = createSourcePeerConnectionCountContext(
    candidateConnection,
    targetLayerPeers,
  );
  return network.connections.reduce(
    (peerConnectionsFromSource, existingConnection) =>
      countConnectionWhenSourceTargetsPeer(
        peerConnectionsFromSource,
        existingConnection,
        countContext,
      ),
    0,
  );
}

/**
 * Creates immutable context for source-to-peer connection counting.
 *
 * @param candidateConnection - Candidate connection.
 * @param targetLayerPeers - Peer-set nodes.
 * @returns Count context.
 */
function createSourcePeerConnectionCountContext(
  candidateConnection: Connection,
  targetLayerPeers: Node[],
): SourcePeerConnectionCountContext {
  return {
    sourceNode: candidateConnection.from,
    targetLayerPeers,
  };
}

/**
 * Counts one connection when it originates from source and targets a peer.
 *
 * @param peerConnectionsFromSource - Current count.
 * @param existingConnection - Existing network connection.
 * @param countContext - Count context.
 * @returns Updated count.
 */
function countConnectionWhenSourceTargetsPeer(
  peerConnectionsFromSource: number,
  existingConnection: Connection,
  countContext: SourcePeerConnectionCountContext,
): number {
  if (existingConnection.from !== countContext.sourceNode) {
    return peerConnectionsFromSource;
  }
  if (!containsNode(countContext.targetLayerPeers, existingConnection.to)) {
    return peerConnectionsFromSource;
  }
  return peerConnectionsFromSource + 1;
}

/**
 * Collects peers around a target node in the same type/layer neighborhood.
 *
 * @param network - Target network.
 * @param targetNode - Node whose peers are collected.
 * @returns Peer nodes.
 */
function collectTargetLayerPeers(network: Network, targetNode: Node): Node[] {
  const peerContext = createTargetLayerPeerContext(network, targetNode);
  return network.nodes.reduce<Node[]>(
    (peers, candidateNode, candidateNodeIndex) => {
      if (isTargetLayerPeer(candidateNode, candidateNodeIndex, peerContext)) {
        peers.push(candidateNode);
      }
      return peers;
    },
    [],
  );
}

/**
 * Checks whether candidate node belongs to the target peer-layer set.
 *
 * @param candidateNode - Candidate node.
 * @param candidateNodeIndex - Candidate node index.
 * @param peerContext - Peer traversal context.
 * @returns True when candidate is an eligible peer.
 */
function isTargetLayerPeer(
  candidateNode: Node,
  candidateNodeIndex: number,
  peerContext: TargetLayerPeerContext,
): boolean {
  if (!isTargetLayerPeerTypeMatch(candidateNode, peerContext)) {
    return false;
  }
  return isTargetLayerPeerWithinDistance(candidateNodeIndex, peerContext);
}

/**
 * Creates immutable context for peer-layer collection.
 *
 * @param network - Target network.
 * @param targetNode - Peer anchor node.
 * @returns Peer traversal context.
 */
function createTargetLayerPeerContext(
  network: Network,
  targetNode: Node,
): TargetLayerPeerContext {
  return {
    targetNodeType: targetNode.type,
    targetIndex: network.nodes.indexOf(targetNode),
    maxDistance: Math.max(network.input, network.output),
  };
}

/**
 * Checks whether node type matches the target-layer peer type.
 *
 * @param candidateNode - Candidate node.
 * @param peerContext - Peer traversal context.
 * @returns True when type matches.
 */
function isTargetLayerPeerTypeMatch(
  candidateNode: Node,
  peerContext: TargetLayerPeerContext,
): boolean {
  return candidateNode.type === peerContext.targetNodeType;
}

/**
 * Checks whether candidate index lies within allowed peer distance.
 *
 * @param candidateNodeIndex - Candidate node index.
 * @param peerContext - Peer traversal context.
 * @returns True when within distance.
 */
function isTargetLayerPeerWithinDistance(
  candidateNodeIndex: number,
  peerContext: TargetLayerPeerContext,
): boolean {
  return (
    Math.abs(candidateNodeIndex - peerContext.targetIndex) <
    peerContext.maxDistance
  );
}

/**
 * Checks whether a node list contains a node reference.
 *
 * @param nodes - Node list.
 * @param node - Node reference.
 * @returns True when contained.
 */
function containsNode(nodes: Node[], node: Node): boolean {
  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    if (nodes[nodeIndex] === node) {
      return true;
    }
  }
  return false;
}

/**
 * MOD_WEIGHT mutation.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
function modWeight(this: Network, method?: MutationMethod): void {
  const allConnections = collectAllConnections(this);
  const mutationProps = asMutationProps(this);
  const targetConnection = pickRandomEntry(allConnections, mutationProps._rand);
  if (!targetConnection) {
    return;
  }

  const methodObject = resolveMethodObject(method);
  const minDelta = methodObject.min ?? DEFAULT_MUTATION_MIN;
  const maxDelta = methodObject.max ?? DEFAULT_MUTATION_MAX;
  const sampledDelta = sampleUniform(mutationProps._rand, minDelta, maxDelta);
  targetConnection.weight += sampledDelta;
}

/**
 * Collects normal and self connections.
 *
 * @param network - Target network.
 * @returns Combined connections.
 */
function collectAllConnections(network: Network): Connection[] {
  return [...network.connections, ...network.selfconns];
}

/**
 * Extracts method-object form when provided.
 *
 * @param method - Optional mutation method.
 * @returns Method object view.
 */
function resolveMethodObject(
  method?: MutationMethod,
): Exclude<MutationMethod, string> {
  if (method && typeof method === 'object') {
    return method;
  }
  return {};
}

/**
 * Samples a uniform value from [minValue, maxValue].
 *
 * @param randomValue - Random generator.
 * @param minValue - Minimum value.
 * @param maxValue - Maximum value.
 * @returns Sampled value.
 */
function sampleUniform(
  randomValue: () => number,
  minValue: number,
  maxValue: number,
): number {
  return randomValue() * (maxValue - minValue) + minValue;
}

/**
 * MOD_BIAS mutation.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
function modBias(this: Network, method?: MutationMethod): void {
  const targetNode = pickRandomNonInputNode(
    this,
    false,
    asMutationProps(this)._rand,
  );
  if (!targetNode) {
    return;
  }
  targetNode.mutate(method);
}

/**
 * Selects a random mutable non-input node.
 *
 * @param network - Target network.
 * @param excludeOutputNodes - True to exclude outputs.
 * @param randomValue - Random generator.
 * @returns Selected mutable node.
 */
function pickRandomNonInputNode(
  network: Network,
  excludeOutputNodes: boolean,
  randomValue: () => number,
): Node | undefined {
  const mutableNodes = collectMutableNonInputNodes(network, excludeOutputNodes);
  return pickRandomEntry(mutableNodes, randomValue);
}

/**
 * Collects mutable non-input nodes.
 *
 * @param network - Target network.
 * @param excludeOutputNodes - True to exclude output nodes.
 * @returns Mutable nodes.
 */
function collectMutableNonInputNodes(
  network: Network,
  excludeOutputNodes: boolean,
): Node[] {
  const mutableNodes: Node[] = [];
  const lastMutableIndex = excludeOutputNodes
    ? network.nodes.length - network.output
    : network.nodes.length;

  for (
    let nodeIndex = network.input;
    nodeIndex < lastMutableIndex;
    nodeIndex++
  ) {
    mutableNodes.push(network.nodes[nodeIndex]);
  }

  return mutableNodes;
}

/**
 * MOD_ACTIVATION mutation.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
function modActivation(this: Network, method?: MutationMethod): void {
  const methodObject = resolveMethodObject(method);
  const canMutateOutput = methodObject.mutateOutput ?? true;
  const targetNode = pickRandomNonInputNode(
    this,
    !canMutateOutput,
    asMutationProps(this)._rand,
  );

  if (!targetNode) {
    warnWhenEnabled(WARNING_NO_ACTIVATION_MUTATION_TARGETS);
    return;
  }

  targetNode.mutate(method);
}

/**
 * ADD_SELF_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addSelfConn(this: Network): void {
  const mutationProps = asMutationProps(this);
  if (mutationProps._enforceAcyclic) {
    return;
  }

  const targetNode = resolveSelfConnectionTargetNode(this, mutationProps._rand);
  if (!targetNode) {
    warnWhenEnabled(WARNING_SELF_CONNECTIONS_ALREADY_PRESENT);
    return;
  }

  connectPair(this, [targetNode, targetNode]);
}

/**
 * Resolves random node eligible for self-connection creation.
 *
 * @param network - Target network.
 * @param randomValue - Random generator.
 * @returns Selected node.
 */
function resolveSelfConnectionTargetNode(
  network: Network,
  randomValue: () => number,
): Node | undefined {
  const candidates = collectNodesWithoutSelfLoop(network);
  return pickRandomEntry(candidates, randomValue);
}

/**
 * Collects non-input nodes that do not have self loops.
 *
 * @param network - Target network.
 * @returns Eligible nodes.
 */
function collectNodesWithoutSelfLoop(network: Network): Node[] {
  return network.nodes
    .slice(network.input)
    .filter((candidateNode) => isNodeWithoutSelfLoop(candidateNode));
}

/**
 * Checks whether a node currently has no self-loop connections.
 *
 * @param candidateNode - Node under evaluation.
 * @returns True when the node has no self-loop.
 */
function isNodeWithoutSelfLoop(candidateNode: Node): boolean {
  return candidateNode.connections.self.length === 0;
}

/**
 * SUB_SELF_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function subSelfConn(this: Network): void {
  const mutationProps = asMutationProps(this);
  const selectedSelfConnection = pickRandomEntry(
    this.selfconns,
    mutationProps._rand,
  );
  if (!selectedSelfConnection) {
    warnWhenEnabled(WARNING_NO_SELF_CONNECTIONS_TO_REMOVE);
    return;
  }

  this.disconnect(selectedSelfConnection.from, selectedSelfConnection.to);
}

/**
 * ADD_GATE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addGate(this: Network): void {
  const mutationProps = asMutationProps(this);
  const ungatedConnectionCandidates = collectUngatedConnections(this);
  const gatingNode = pickRandomNonInputNode(this, false, mutationProps._rand);
  const connectionToGate = pickRandomEntry(
    ungatedConnectionCandidates,
    mutationProps._rand,
  );

  if (!gatingNode || !connectionToGate) {
    warnWhenEnabled(WARNING_ALL_CONNECTIONS_GATED);
    return;
  }

  this.gate(gatingNode, connectionToGate);
}

/**
 * Collects ungated connections including self-connections.
 *
 * @param network - Target network.
 * @returns Ungated connections.
 */
function collectUngatedConnections(network: Network): Connection[] {
  const allConnections = collectAllConnections(network);
  return allConnections.filter((candidateConnection) =>
    isUngatedConnection(candidateConnection),
  );
}

/**
 * Checks whether a connection has no gater attached.
 *
 * @param candidateConnection - Connection under evaluation.
 * @returns True when ungated.
 */
function isUngatedConnection(candidateConnection: Connection): boolean {
  return candidateConnection.gater === null;
}

/**
 * SUB_GATE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function subGate(this: Network): void {
  const mutationProps = asMutationProps(this);
  const gatedConnection = pickRandomEntry(this.gates, mutationProps._rand);
  if (!gatedConnection) {
    warnWhenEnabled(WARNING_NO_GATED_CONNECTIONS_TO_REMOVE);
    return;
  }

  this.ungate(gatedConnection);
}

/**
 * ADD_BACK_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addBackConn(this: Network): void {
  const mutationProps = asMutationProps(this);
  if (mutationProps._enforceAcyclic) {
    return;
  }

  const selectedConnectionPair = resolveSelectedBackwardConnectionPair(
    this,
    mutationProps._rand,
  );

  if (!selectedConnectionPair) {
    return;
  }

  connectPair(this, selectedConnectionPair);
}

/**
 * Resolves random selected backward connection pair.
 *
 * @param network - Target network.
 * @param randomValue - Random generator.
 * @returns Selected source/target pair.
 */
function resolveSelectedBackwardConnectionPair(
  network: Network,
  randomValue: () => number,
): NodePair | undefined {
  const backwardConnectionCandidates =
    collectBackwardConnectionCandidates(network);
  return pickRandomEntry(backwardConnectionCandidates, randomValue);
}

/**
 * Collects backward (recurrent) connection candidates.
 *
 * @param network - Target network.
 * @returns Candidate source/target pairs.
 */
function collectBackwardConnectionCandidates(network: Network): NodePair[] {
  const traversalContexts = collectBackwardTraversalContexts(network);
  return traversalContexts.reduce<NodePair[]>(
    collectBackwardCandidatesFromContext,
    [],
  );
}

/**
 * Collects backward traversal contexts for all eligible later nodes.
 *
 * @param network - Target network.
 * @returns Backward traversal contexts.
 */
function collectBackwardTraversalContexts(
  network: Network,
): BackwardCandidateTraversalContext[] {
  const backwardTraversalContexts: BackwardCandidateTraversalContext[] = [];

  for (
    let laterNodeIndex = network.input;
    laterNodeIndex < network.nodes.length;
    laterNodeIndex++
  ) {
    backwardTraversalContexts.push(
      createBackwardCandidateTraversalContext(network, laterNodeIndex),
    );
  }

  return backwardTraversalContexts;
}

/**
 * Reduces one backward traversal context into candidate connection pairs.
 *
 * @param backwardConnectionCandidates - Existing candidate pairs.
 * @param traversalContext - Later-node traversal context.
 * @returns Updated candidate pairs.
 */
function collectBackwardCandidatesFromContext(
  backwardConnectionCandidates: NodePair[],
  traversalContext: BackwardCandidateTraversalContext,
): NodePair[] {
  const laterNodeCandidates =
    collectBackwardCandidatesForLaterNode(traversalContext);
  backwardConnectionCandidates.push(...laterNodeCandidates);
  return backwardConnectionCandidates;
}

/**
 * Creates context for one backward-candidate traversal pass.
 *
 * @param network - Target network.
 * @param laterNodeIndex - Current later-node index.
 * @returns Immutable traversal context.
 */
function createBackwardCandidateTraversalContext(
  network: Network,
  laterNodeIndex: number,
): BackwardCandidateTraversalContext {
  return {
    network,
    laterNodeIndex,
    laterNode: network.nodes[laterNodeIndex],
  };
}

/**
 * Appends backward candidates for one later-node traversal context.
 *
 * @param traversalContext - Later-node traversal context.
 * @param backwardConnectionCandidates - Collector array.
 * @returns Nothing.
 */
function appendBackwardCandidatesForLaterNode(
  traversalContext: BackwardCandidateTraversalContext,
  backwardConnectionCandidates: NodePair[],
): void {
  backwardConnectionCandidates.push(
    ...collectBackwardCandidatesForLaterNode(traversalContext),
  );
}

/**
 * Collects all backward candidates for one later-node traversal context.
 *
 * @param traversalContext - Later-node traversal context.
 * @returns Candidate source/target pairs.
 */
function collectBackwardCandidatesForLaterNode(
  traversalContext: BackwardCandidateTraversalContext,
): NodePair[] {
  const laterNodeCandidates: NodePair[] = [];

  for (
    let earlierNodeIndex = traversalContext.network.input;
    earlierNodeIndex < traversalContext.laterNodeIndex;
    earlierNodeIndex++
  ) {
    const earlierNode = traversalContext.network.nodes[earlierNodeIndex];
    if (
      isBackwardCandidateTargetAvailable(
        traversalContext.laterNode,
        earlierNode,
      )
    ) {
      laterNodeCandidates.push([traversalContext.laterNode, earlierNode]);
    }
  }

  return laterNodeCandidates;
}

/**
 * Checks whether a backward candidate target is not already projected.
 *
 * @param laterNode - Candidate source node.
 * @param earlierNode - Candidate target node.
 * @returns True when connection may be added.
 */
function isBackwardCandidateTargetAvailable(
  laterNode: Node,
  earlierNode: Node,
): boolean {
  return !laterNode.isProjectingTo(earlierNode);
}

/**
 * SUB_BACK_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function subBackConn(this: Network): void {
  const removableBackwardConnections =
    collectRemovableBackwardConnections(this);
  const selectedConnection = pickRandomEntry(
    removableBackwardConnections,
    asMutationProps(this)._rand,
  );

  if (!selectedConnection) {
    return;
  }

  disconnectConnectionPair(this, selectedConnection);
}

/**
 * Connects source/target node pair.
 *
 * @param network - Target network.
 * @param selectedConnectionPair - Source/target pair.
 * @returns Nothing.
 */
function connectPair(network: Network, selectedConnectionPair: NodePair): void {
  network.connect(selectedConnectionPair[0], selectedConnectionPair[1]);
}

/**
 * Collects removable backward connections using redundancy constraints.
 *
 * @param network - Target network.
 * @returns Removable backward connections.
 */
function collectRemovableBackwardConnections(network: Network): Connection[] {
  return network.connections.filter((candidateConnection) =>
    isRemovableBackwardConnection(network, candidateConnection),
  );
}

/**
 * Evaluates whether a backward connection is safe to remove.
 *
 * @param network - Target network.
 * @param candidateConnection - Connection under evaluation.
 * @returns True when removable.
 */
function isRemovableBackwardConnection(
  network: Network,
  candidateConnection: Connection,
): boolean {
  const directionContext = createDirectionalConnectionContext(
    network,
    candidateConnection,
  );
  if (!hasRedundantEndpoints(directionContext.candidateConnection)) {
    return false;
  }
  return isBackwardDirectionalContext(directionContext);
}

/**
 * Checks whether a directional context represents a backward edge.
 *
 * @param directionContext - Directional context.
 * @returns True when backward.
 */
function isBackwardDirectionalContext(
  directionContext: DirectionalConnectionContext,
): boolean {
  return directionContext.fromNodeIndex > directionContext.toNodeIndex;
}

/**
 * SWAP_NODES mutation.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
function swapNodes(this: Network, method?: MutationMethod): void {
  const mutationProps = asMutationProps(this);
  const swappableNodes = collectSwappableNodesForMutation(this, method);
  const distinctNodePair = pickDistinctNodePair(
    swappableNodes,
    mutationProps._rand,
  );
  if (!distinctNodePair) {
    return;
  }

  swapNodeBiasAndSquash(
    distinctNodePair.firstNode,
    distinctNodePair.secondNode,
  );
}

/**
 * Collects swap-eligible nodes based on mutation configuration.
 *
 * @param network - Target network.
 * @param method - Optional method descriptor.
 * @returns Swap-eligible nodes.
 */
function collectSwappableNodesForMutation(
  network: Network,
  method?: MutationMethod,
): Node[] {
  const methodObject = resolveMethodObject(method);
  const canSwapOutput = methodObject.mutateOutput ?? true;
  return collectMutableNonInputNodes(network, !canSwapOutput);
}

/**
 * Picks two distinct nodes from a candidate set.
 *
 * @param swappableNodes - Swap candidate nodes.
 * @param randomValue - Random generator.
 * @returns Distinct pair or undefined.
 */
function pickDistinctNodePair(
  swappableNodes: Node[],
  randomValue: () => number,
): DistinctNodePair | undefined {
  if (swappableNodes.length < MIN_SWAPPABLE_NODE_COUNT) {
    return undefined;
  }

  const firstNode = pickRandomEntry(swappableNodes, randomValue);
  if (!firstNode) {
    return undefined;
  }

  return resolveDistinctPairWithKnownFirstNode(
    swappableNodes,
    firstNode,
    randomValue,
  );
}

/**
 * Resolves a distinct pair when first node is already known.
 *
 * @param swappableNodes - Swap candidate nodes.
 * @param firstNode - Chosen first node.
 * @param randomValue - Random generator.
 * @returns Distinct pair or undefined.
 */
function resolveDistinctPairWithKnownFirstNode(
  swappableNodes: Node[],
  firstNode: Node,
  randomValue: () => number,
): DistinctNodePair | undefined {
  const secondNode = pickDistinctRandomNode(
    swappableNodes,
    firstNode,
    randomValue,
  );
  if (!secondNode) {
    return undefined;
  }
  return { firstNode, secondNode };
}

/**
 * Picks a random node distinct from a given reference.
 *
 * @param nodeCandidates - Candidate nodes.
 * @param excludedNode - Node to exclude.
 * @param randomValue - Random generator.
 * @returns Distinct node or undefined.
 */
function pickDistinctRandomNode(
  nodeCandidates: Node[],
  excludedNode: Node,
  randomValue: () => number,
): Node | undefined {
  const distinctNodeCandidates = collectDistinctNodeCandidates(
    nodeCandidates,
    excludedNode,
  );
  if (distinctNodeCandidates.length === 0) {
    return undefined;
  }

  return pickRandomEntry(distinctNodeCandidates, randomValue);
}

/**
 * Collects candidates that are distinct from an excluded node.
 *
 * @param nodeCandidates - Candidate nodes.
 * @param excludedNode - Node to exclude.
 * @returns Distinct candidates.
 */
function collectDistinctNodeCandidates(
  nodeCandidates: Node[],
  excludedNode: Node,
): Node[] {
  return nodeCandidates.filter(
    (candidateNode) => candidateNode !== excludedNode,
  );
}

/**
 * Swaps bias and squash values between two nodes.
 *
 * @param firstNode - First node.
 * @param secondNode - Second node.
 * @returns Nothing.
 */
function swapNodeBiasAndSquash(firstNode: Node, secondNode: Node): void {
  const firstBias = firstNode.bias;
  const firstSquash = firstNode.squash;

  firstNode.bias = secondNode.bias;
  firstNode.squash = secondNode.squash;

  secondNode.bias = firstBias;
  secondNode.squash = firstSquash;
}

/**
 * ADD_LSTM_NODE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addLSTMNode(this: Network): void {
  addRecurrentNode(this, RECURRENT_BLOCK_LSTM);
}

/**
 * ADD_GRU_NODE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function addGRUNode(this: Network): void {
  addRecurrentNode(this, RECURRENT_BLOCK_GRU);
}

/**
 * Shared orchestrator for recurrent-node mutation variants.
 *
 * @param network - Target network.
 * @param blockType - Recurrent block type.
 * @returns Nothing.
 */
function addRecurrentNode(
  network: Network,
  blockType: typeof RECURRENT_BLOCK_LSTM | typeof RECURRENT_BLOCK_GRU,
): void {
  const mutationProps = asMutationProps(network);
  if (mutationProps._enforceAcyclic || network.connections.length === 0) {
    return;
  }

  const selectedConnection = pickRandomEntry(
    network.connections,
    mutationProps._rand,
  );
  if (!selectedConnection) {
    return;
  }

  expandConnectionWithRecurrentBlock(network, selectedConnection, blockType);
}

/**
 * Replaces one connection with a minimal recurrent block.
 *
 * @param network - Target network.
 * @param connectionToExpand - Connection to replace.
 * @param blockType - Recurrent block type.
 * @returns Nothing.
 */
function expandConnectionWithRecurrentBlock(
  network: Network,
  connectionToExpand: Connection,
  blockType: typeof RECURRENT_BLOCK_LSTM | typeof RECURRENT_BLOCK_GRU,
): void {
  const previousGater = disconnectConnectionAndGetGater(
    network,
    connectionToExpand,
  );
  const recurrentLayer = createRecurrentLayer(blockType);
  const latestConnection = reconnectThroughRecurrentLayer(
    network,
    connectionToExpand,
    recurrentLayer,
  );
  tryGateLatestConnection(network, previousGater, latestConnection);
}

/**
 * Disconnects a connection and returns its previous gater.
 *
 * @param network - Target network.
 * @param connectionToExpand - Connection being expanded.
 * @returns Previous gater reference.
 */
function disconnectConnectionAndGetGater(
  network: Network,
  connectionToExpand: Connection,
): Connection['gater'] {
  const previousGater = connectionToExpand.gater;
  network.disconnect(connectionToExpand.from, connectionToExpand.to);
  return previousGater;
}

/**
 * Reconnects a source/target pair through a recurrent layer.
 *
 * @param network - Target network.
 * @param connectionToExpand - Original connection.
 * @param recurrentLayer - Recurrent-layer shape.
 * @returns Latest newly created connection or undefined.
 */
function reconnectThroughRecurrentLayer(
  network: Network,
  connectionToExpand: Connection,
  recurrentLayer: RecurrentLayerShape,
): Connection | undefined {
  appendRecurrentLayerNodes(network, recurrentLayer.nodes);
  network.connect(connectionToExpand.from, recurrentLayer.nodes[0]);
  network.connect(recurrentLayer.output.nodes[0], connectionToExpand.to);
  return network.connections.at(-1);
}

/**
 * Gates the latest connection when both previous gater and target exist.
 *
 * @param network - Target network.
 * @param previousGater - Previously assigned gater.
 * @param latestConnection - Connection to receive the gater.
 * @returns Nothing.
 */
function tryGateLatestConnection(
  network: Network,
  previousGater: Connection['gater'],
  latestConnection: Connection | undefined,
): void {
  if (!previousGater || !latestConnection) {
    return;
  }
  network.gate(previousGater, latestConnection);
}

/**
 * Creates recurrent layer by type.
 *
 * @param blockType - Recurrent block type.
 * @returns Created recurrent layer.
 */
function createRecurrentLayer(
  blockType: typeof RECURRENT_BLOCK_LSTM | typeof RECURRENT_BLOCK_GRU,
): RecurrentLayerShape {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const Layer = require(LAYER_MODULE_PATH).default;
  if (blockType === RECURRENT_BLOCK_LSTM) {
    return Layer.lstm(SINGLE_UNIT_RECURRENT_BLOCK_WIDTH);
  }
  return Layer.gru(SINGLE_UNIT_RECURRENT_BLOCK_WIDTH);
}

/**
 * Appends recurrent layer nodes as hidden nodes.
 *
 * @param network - Target network.
 * @param layerNodes - Layer nodes.
 * @returns Nothing.
 */
function appendRecurrentLayerNodes(network: Network, layerNodes: Node[]): void {
  for (let nodeIndex = 0; nodeIndex < layerNodes.length; nodeIndex++) {
    const layerNode = layerNodes[nodeIndex];
    layerNode.type = NODE_TYPE_HIDDEN;
    network.nodes.push(layerNode);
  }
}

/**
 * REINIT_WEIGHT mutation.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
function reinitWeight(this: Network, method?: MutationMethod): void {
  const mutationProps = asMutationProps(this);
  const targetNode = pickRandomNonInputNode(this, false, mutationProps._rand);
  if (!targetNode) {
    return;
  }

  const methodObject = resolveMethodObject(method);
  const reinitContext = createConnectionGroupReinitContext(
    mutationProps._rand,
    methodObject,
  );
  const connectionGroups = collectConnectionGroupsForReinit(targetNode);
  connectionGroups.forEach((connectionGroup) => {
    reinitializeConnectionGroupWeights(connectionGroup, reinitContext);
  });
}

/**
 * Creates immutable context for connection-group reinitialization.
 *
 * @param randomValue - Random generator.
 * @param methodObject - Method override object.
 * @returns Reinitialization context.
 */
function createConnectionGroupReinitContext(
  randomValue: () => number,
  methodObject: Exclude<MutationMethod, string>,
): ConnectionGroupReinitContext {
  return {
    randomValue,
    minWeight: methodObject.min ?? DEFAULT_MUTATION_MIN,
    maxWeight: methodObject.max ?? DEFAULT_MUTATION_MAX,
  };
}

/**
 * Collects all connection groups affected by REINIT_WEIGHT.
 *
 * @param targetNode - Node receiving the reinitialization.
 * @returns Mutable connection groups.
 */
function collectConnectionGroupsForReinit(targetNode: Node): Connection[][] {
  return [
    targetNode.connections.in,
    targetNode.connections.out,
    targetNode.connections.self,
  ];
}

/**
 * Reinitializes all weights in a connection group.
 *
 * @param connections - Connection group.
 * @param randomValue - Random generator.
 * @param minWeight - Minimum sampled weight.
 * @param maxWeight - Maximum sampled weight.
 * @returns Nothing.
 */
function reinitializeConnectionGroupWeights(
  connections: Connection[],
  reinitContext: ConnectionGroupReinitContext,
): void {
  const samplingContext = createWeightSamplingRangeContext(reinitContext);

  for (
    let connectionIndex = 0;
    connectionIndex < connections.length;
    connectionIndex++
  ) {
    const connection = connections[connectionIndex];
    connection.weight = sampleUniformFromContext(samplingContext);
  }
}

/**
 * Creates immutable sampling range context.
 *
 * @param reinitContext - Reinitialization context.
 * @returns Sampling range context.
 */
function createWeightSamplingRangeContext(
  reinitContext: ConnectionGroupReinitContext,
): WeightSamplingRangeContext {
  return {
    randomValue: reinitContext.randomValue,
    minValue: reinitContext.minWeight,
    maxValue: reinitContext.maxWeight,
  };
}

/**
 * Samples one weight using a prebuilt range context.
 *
 * @param samplingContext - Sampling range context.
 * @returns Sampled weight.
 */
function sampleUniformFromContext(
  samplingContext: WeightSamplingRangeContext,
): number {
  return sampleUniform(
    samplingContext.randomValue,
    samplingContext.minValue,
    samplingContext.maxValue,
  );
}

/**
 * BATCH_NORM mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function batchNorm(this: Network): void {
  const hiddenNodes = collectNodesByType(this, NODE_TYPE_HIDDEN);
  const selectedHiddenNode = pickRandomEntry(
    hiddenNodes,
    asMutationProps(this)._rand,
  );
  if (!selectedHiddenNode) {
    return;
  }

  enableNodeBatchNorm(selectedHiddenNode);
}

/**
 * Enables internal batch-norm flag on a node.
 *
 * @param node - Node to flag.
 * @returns Nothing.
 */
function enableNodeBatchNorm(node: Node): void {
  const nodeWithBatchNorm = node as unknown as Record<string, unknown>;
  nodeWithBatchNorm[BATCH_NORM_FLAG_KEY] = true;
}
