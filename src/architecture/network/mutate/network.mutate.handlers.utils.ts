import type Network from '../../network/network';
import type Connection from '../../connection';
import Layer from '../../layer/layer';
import Node from '../../node';
import mutation from '../../../methods/mutation/mutation';
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
  MutationMethod,
  NetworkMutationProps,
  NodePair,
  RecurrentLayerShape,
  SourcePeerConnectionCountContext,
  TargetLayerPeerContext,
  WeightSamplingRangeContext,
} from '../network.types';
import {
  BATCH_NORM_FLAG_KEY,
  DEFAULT_MUTATION_MAX,
  DEFAULT_MUTATION_MIN,
  GATE_REASSIGN_THRESHOLD,
  MIN_REDUNDANT_CONNECTION_COUNT,
  MIN_SWAPPABLE_NODE_COUNT,
  NODE_TYPE_HIDDEN,
  NODE_TYPE_INPUT,
  NODE_TYPE_OUTPUT,
  RECURRENT_BLOCK_GRU,
  RECURRENT_BLOCK_LSTM,
  SINGLE_UNIT_RECURRENT_BLOCK_WIDTH,
  SUB_NODE_STABILITY_WEIGHT_DELTA,
  WARNING_ALL_CONNECTIONS_GATED,
  WARNING_NO_ACTIVATION_MUTATION_TARGETS,
  WARNING_NO_GATED_CONNECTIONS_TO_REMOVE,
  WARNING_NO_HIDDEN_NODES_TO_REMOVE,
  WARNING_NO_SELF_CONNECTIONS_TO_REMOVE,
  WARNING_SELF_CONNECTIONS_ALREADY_PRESENT,
} from './network.mutate.utils.types';
import { NetworkMutateRecurrentLayerOutputInitializationError } from './network.mutate.errors';
import {
  appendTemporalDescriptorSet,
  buildGruTemporalDescriptorSet,
  buildLstmTemporalDescriptorSet,
  splitGruLayerNodes,
  splitLstmLayerNodes,
} from '../network.temporal.extensions.utils';
import { ensureGrowthBudget } from '../network.utils';

/** Net new connections created when replacing one edge with a minimal LSTM block. */
const LSTM_RECURRENT_BLOCK_ADDITIONAL_CONNECTION_COUNT = 5;

/** Net new connections created when replacing one edge with a minimal GRU block. */
const GRU_RECURRENT_BLOCK_ADDITIONAL_CONNECTION_COUNT = 8;

/**
 * Concrete mutation handler implementations used by the network mutate orchestrator.
 *
 * Organization:
 * - Exported functions represent public mutation operations mapped by dispatch key.
 * - Internal helpers encapsulate candidate collection, validation, and graph rewiring.
 * - Shared constants and warning strings are imported from `network.mutate.utils.types.ts`
 *   to keep cross-file contracts explicit and avoid circular dependencies.
 *
 * Behavioral notes:
 * - Handlers preserve fail-soft semantics where possible (return early when no candidate exists).
 * - Acyclic mode checks are enforced in handlers that could introduce recurrence.
 * - Randomness is sourced from network mutation internals for reproducible deterministic flows.
 *
 * @module network.mutate.handlers
 */

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
 * Adds one hidden node by splitting an existing connection.
 *
 * Execution modes:
 * - Deterministic chain mode grows a linear input→...→output chain.
 * - Standard mode splits a randomly selected forward connection.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addNode(this: Network): void {
  const mutationProps = asMutationProps(this);
  markTopoDirtyIfAcyclic(mutationProps);

  const requiredAdditionalConnections =
    this.connections.length === 0 ? 2 : 1;
  if (!ensureGrowthBudget(this, requiredAdditionalConnections)) {
    return;
  }

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
 * @returns Deterministic context or undefined when one or more prerequisites fail.
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
 * Removes one hidden node and applies a tiny weight nudge for numerical continuity.
 *
 * The stability nudge helps keep downstream mutation effects observable in edge cases
 * where node removal substantially changes effective signal flow.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function subNode(this: Network): void {
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
 * Adds one forward connection between currently unconnected eligible node pairs.
 *
 * Candidate generation respects node ordering so the added edge is feed-forward.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addConn(this: Network): void {
  const mutationProps = asMutationProps(this);
  markTopoDirtyIfAcyclic(mutationProps);

  const selectedConnectionPair = resolveSelectedForwardConnectionPair(
    this,
    mutationProps._rand,
  );
  if (!selectedConnectionPair) {
    return;
  }

  if (!ensureGrowthBudget(this, 1)) {
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
 * Removes one forward connection when structural redundancy constraints are satisfied.
 *
 * Constraints require endpoint redundancy and avoid disconnecting peer-layer groups.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function subConn(this: Network): void {
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
 * Perturbs one connection weight using a uniform delta sampled from configured bounds.
 *
 * The candidate pool includes standard and self-connections.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
export function modWeight(this: Network, method?: MutationMethod): void {
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
 * Mutates bias parameters on one random non-input node.
 *
 * Output nodes remain eligible for this operator.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
export function modBias(this: Network, method?: MutationMethod): void {
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
 * Mutates activation function on one random non-input node.
 *
 * Output-node eligibility is controlled by `method.mutateOutput` when provided.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
export function modActivation(this: Network, method?: MutationMethod): void {
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
 * Adds one self-connection on an eligible node that does not already have one.
 *
 * This operation is skipped in acyclic mode.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addSelfConn(this: Network): void {
  const mutationProps = asMutationProps(this);
  if (mutationProps._enforceAcyclic) {
    return;
  }

  const targetNode = resolveSelfConnectionTargetNode(this, mutationProps._rand);
  if (!targetNode) {
    warnWhenEnabled(WARNING_SELF_CONNECTIONS_ALREADY_PRESENT);
    return;
  }

  if (!ensureGrowthBudget(this, 1)) {
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
 * Removes one existing self-connection chosen at random.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function subSelfConn(this: Network): void {
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
 * Assigns a random eligible node as gater for a random ungated connection.
 *
 * Candidate pool includes normal and self-connections.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addGate(this: Network): void {
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
 * Removes gating from one randomly selected gated connection.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function subGate(this: Network): void {
  const mutationProps = asMutationProps(this);
  const gatedConnection = pickRandomEntry(this.gates, mutationProps._rand);
  if (!gatedConnection) {
    warnWhenEnabled(WARNING_NO_GATED_CONNECTIONS_TO_REMOVE);
    return;
  }

  this.ungate(gatedConnection);
}

/**
 * Adds one backward (recurrent) connection between eligible node pairs.
 *
 * This operation is skipped in acyclic mode.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addBackConn(this: Network): void {
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

  if (!ensureGrowthBudget(this, 1)) {
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
 * Removes one backward connection that satisfies redundancy constraints.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function subBackConn(this: Network): void {
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
 * Swaps bias and activation squash functions between two distinct mutable nodes.
 *
 * This provides a lightweight structural-parameter recombination without changing
 * graph connectivity.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
export function swapNodes(this: Network, method?: MutationMethod): void {
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
 * Replaces one connection by inserting a minimal LSTM recurrent block.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addLSTMNode(this: Network): void {
  addRecurrentNode(this, RECURRENT_BLOCK_LSTM);
}

/**
 * Replaces one connection by inserting a minimal GRU recurrent block.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function addGRUNode(this: Network): void {
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

  if (
    !ensureGrowthBudget(
      network,
      resolveRecurrentGrowthBudgetRequirement(blockType),
    )
  ) {
    return;
  }

  expandConnectionWithRecurrentBlock(network, selectedConnection, blockType);
}

/**
 * Resolve the net connection growth required by a minimal recurrent block.
 *
 * @param blockType - Recurrent block type being inserted.
 * @returns Net additional connections created by the mutation.
 */
function resolveRecurrentGrowthBudgetRequirement(
  blockType: typeof RECURRENT_BLOCK_LSTM | typeof RECURRENT_BLOCK_GRU,
): number {
  if (blockType === RECURRENT_BLOCK_LSTM) {
    return LSTM_RECURRENT_BLOCK_ADDITIONAL_CONNECTION_COUNT;
  }

  return GRU_RECURRENT_BLOCK_ADDITIONAL_CONNECTION_COUNT;
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
  appendTemporalDescriptorSet(
    network,
    buildRecurrentMutationDescriptorSet(network, recurrentLayer, blockType),
  );
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
  registerRecurrentLayerConnections(network, recurrentLayer.nodes);
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
  const recurrentLayer =
    blockType === RECURRENT_BLOCK_LSTM
      ? Layer.lstm(SINGLE_UNIT_RECURRENT_BLOCK_WIDTH)
      : Layer.gru(SINGLE_UNIT_RECURRENT_BLOCK_WIDTH);

  if (!recurrentLayer.output || !Array.isArray(recurrentLayer.output.nodes)) {
    throw new NetworkMutateRecurrentLayerOutputInitializationError(
      'Recurrent layer output was not initialized.',
    );
  }

  if (blockType === RECURRENT_BLOCK_LSTM) {
    return recurrentLayer as RecurrentLayerShape;
  }
  return recurrentLayer as RecurrentLayerShape;
}

function buildRecurrentMutationDescriptorSet(
  network: Network,
  recurrentLayer: RecurrentLayerShape,
  blockType: typeof RECURRENT_BLOCK_LSTM | typeof RECURRENT_BLOCK_GRU,
) {
  if (blockType === RECURRENT_BLOCK_LSTM) {
    const roleNodes = splitLstmLayerNodes(
      recurrentLayer.nodes,
      SINGLE_UNIT_RECURRENT_BLOCK_WIDTH,
    );

    return roleNodes
      ? buildLstmTemporalDescriptorSet(network, roleNodes)
      : undefined;
  }

  const roleNodes = splitGruLayerNodes(
    recurrentLayer.nodes,
    SINGLE_UNIT_RECURRENT_BLOCK_WIDTH,
  );

  return roleNodes
    ? buildGruTemporalDescriptorSet(network, roleNodes)
    : undefined;
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
 * Registers a recurrent layer's prebuilt internal connections on the network.
 *
 * Recurrent layer factories wire their own internal node graph before the layer
 * is attached to a `Network`. Mutation must therefore register those existing
 * connection objects onto the network's canonical forward, self, and gated
 * shelves before later serialization, validation, and compatibility paths read
 * the graph.
 *
 * @param network - Target network.
 * @param layerNodes - Recurrent layer nodes whose internal edges should be registered.
 * @returns Nothing.
 */
function registerRecurrentLayerConnections(
  network: Network,
  layerNodes: Node[],
): void {
  const seenConnections = new Set<Connection>();

  for (let nodeIndex = 0; nodeIndex < layerNodes.length; nodeIndex++) {
    const layerNode = layerNodes[nodeIndex];
    const candidateConnections = [
      ...layerNode.connections.out,
      ...layerNode.connections.self,
    ];

    for (
      let connectionIndex = 0;
      connectionIndex < candidateConnections.length;
      connectionIndex++
    ) {
      const candidateConnection = candidateConnections[connectionIndex];
      if (seenConnections.has(candidateConnection)) {
        continue;
      }

      seenConnections.add(candidateConnection);
      registerRecurrentLayerConnection(network, candidateConnection);
    }
  }
}

/**
 * Registers one recurrent-layer connection on the canonical runtime shelves.
 *
 * @param network - Target network.
 * @param connection - Recurrent-layer connection to register.
 * @returns Nothing.
 */
function registerRecurrentLayerConnection(
  network: Network,
  connection: Connection,
): void {
  const targetCollection =
    connection.from === connection.to ? network.selfconns : network.connections;

  if (!targetCollection.includes(connection)) {
    targetCollection.push(connection);
  }

  if (connection.gater && !network.gates.includes(connection)) {
    network.gates.push(connection);
  }
}

/**
 * Reinitializes incoming, outgoing, and self-connection weights for one target node.
 *
 * Weight sampling bounds come from method overrides or default mutation bounds.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
export function reinitWeight(this: Network, method?: MutationMethod): void {
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
 * Enables the internal batch-normalization flag on one random hidden node.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
export function batchNorm(this: Network): void {
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
