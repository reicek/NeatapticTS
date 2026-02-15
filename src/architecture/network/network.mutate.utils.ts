import type Network from '../network';
import type Connection from '../connection';
import Node from '../node';
import mutation from '../../methods/mutation';
import { config } from '../../config';

/**
 * Mutation method descriptor (can be string enum value or object with identity).
 */
export type MutationMethod =
  | string
  | {
      name?: string;
      type?: string;
      identity?: string;
      max?: number;
      min?: number;
      mutateOutput?: boolean;
      [key: string]: unknown;
    };

/**
 * Internal Network properties accessed during mutations.
 */
interface NetworkMutationProps {
  _enforceAcyclic?: boolean;
  _topoDirty?: boolean;
  _detChain?: Node[];
  _rand: () => number;
  _nodeIndexDirty?: boolean;
  _preferredChainEdge?: unknown;
}

/**
 * Handler contract for mutation dispatch entries.
 */
interface MutationHandler {
  /**
   * Applies a mutation to the bound network.
   *
   * @param this - Bound network instance.
   * @param method - Optional mutation descriptor.
   * @returns Nothing.
   */
  (this: Network, method?: MutationMethod): void;
}

/**
 * Mutation dispatch table keyed by mutation identity.
 */
const MUTATION_DISPATCH: Record<string, MutationHandler> = {
  ADD_NODE: _addNode,
  SUB_NODE: _subNode,
  ADD_CONN: _addConn,
  SUB_CONN: _subConn,
  MOD_WEIGHT: _modWeight,
  MOD_BIAS: _modBias,
  MOD_ACTIVATION: _modActivation,
  ADD_SELF_CONN: _addSelfConn,
  SUB_SELF_CONN: _subSelfConn,
  ADD_GATE: _addGate,
  SUB_GATE: _subGate,
  ADD_BACK_CONN: _addBackConn,
  SUB_BACK_CONN: _subBackConn,
  SWAP_NODES: _swapNodes,
  ADD_LSTM_NODE: _addLSTMNode,
  ADD_GRU_NODE: _addGRUNode,
  REINIT_WEIGHT: _reinitWeight,
  BATCH_NORM: _batchNorm,
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
    throw new Error('No (correct) mutate method given!');
  }

  const mutationKey = _resolveMutationKey(method);
  const mutationHandler = mutationKey
    ? MUTATION_DISPATCH[mutationKey]
    : undefined;

  if (!mutationHandler) {
    _warnUnknownMutation(mutationKey);
    return;
  }

  mutationHandler.call(this, method);
  _asMutationProps(this)._topoDirty = true;
}

/**
 * Converts a network to its internal mutation runtime shape.
 *
 * @param network - Network to convert.
 * @returns Runtime mutation props.
 */
function _asMutationProps(network: Network): NetworkMutationProps {
  return network as unknown as NetworkMutationProps;
}

/**
 * Resolves a mutation dispatch key from user-provided method input.
 *
 * @param method - Mutation method input.
 * @returns Dispatch key or undefined.
 */
function _resolveMutationKey(method: MutationMethod): string | undefined {
  if (typeof method === 'string') {
    return method;
  }

  const directKey = method.name ?? method.type ?? method.identity;
  if (directKey) {
    return directKey;
  }

  return _findMutationKeyByIdentityReference(method);
}

/**
 * Resolves a mutation key by direct identity-reference comparison.
 *
 * @param method - Mutation object reference.
 * @returns Matching mutation key or undefined.
 */
function _findMutationKeyByIdentityReference(
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
function _warnUnknownMutation(mutationKey?: string): void {
  if (!config.warnings) {
    return;
  }
  console.warn('[mutate] Unknown mutation method ignored:', mutationKey);
}

/**
 * Marks topology caches dirty when acyclic mode is enforced.
 *
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function _markTopoDirtyIfAcyclic(mutationProps: NetworkMutationProps): void {
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
function _findFirstNodeByType(
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
function _pickRandomEntry<T>(
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
function _createHiddenNode(randomValue: () => number): Node {
  const hiddenNode = new Node('hidden', undefined, randomValue);
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
function _insertNodeBeforeOutputTail(
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
function _findConnection(
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
function _ensureConnection(
  network: Network,
  fromNode: Node,
  toNode: Node,
): Connection | undefined {
  const existingConnection = _findConnection(network, fromNode, toNode);
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
function _tryDisconnectConnection(
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
function _addNode(this: Network): void {
  const mutationProps = _asMutationProps(this);
  _markTopoDirtyIfAcyclic(mutationProps);

  if (config.deterministicChainMode) {
    _addNodeDeterministicChain(this, mutationProps);
    return;
  }

  _addNodeRandomSplit(this, mutationProps);
}

/**
 * Applies deterministic chain-growth ADD_NODE mutation.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function _addNodeDeterministicChain(
  network: Network,
  mutationProps: NetworkMutationProps,
): void {
  const inputNode = _findFirstNodeByType(network, 'input');
  const outputNode = _findFirstNodeByType(network, 'output');
  if (!inputNode || !outputNode) {
    return;
  }

  _initializeDeterministicChain(network, mutationProps, inputNode, outputNode);
  const deterministicChain = mutationProps._detChain;
  if (!deterministicChain || deterministicChain.length === 0) {
    return;
  }

  const tailNode = deterministicChain.at(-1);
  if (!tailNode) {
    return;
  }

  const terminalConnection = _ensureConnection(network, tailNode, outputNode);
  if (!terminalConnection) {
    return;
  }

  const previousGater = terminalConnection.gater;
  network.disconnect(terminalConnection.from, terminalConnection.to);

  const hiddenNode = _createHiddenNode(mutationProps._rand);
  _insertNodeBeforeOutputTail(network, hiddenNode, outputNode, mutationProps);

  const sourceToHiddenConnection = _ensureConnection(
    network,
    tailNode,
    hiddenNode,
  );
  const hiddenToOutputConnection = _ensureConnection(
    network,
    hiddenNode,
    outputNode,
  );
  deterministicChain.push(hiddenNode);
  mutationProps._preferredChainEdge = hiddenToOutputConnection;

  if (previousGater && sourceToHiddenConnection && hiddenToOutputConnection) {
    const gatedConnection =
      mutationProps._rand() >= 0.5
        ? sourceToHiddenConnection
        : hiddenToOutputConnection;
    network.gate(previousGater, gatedConnection);
  }

  _pruneDeterministicChainExtraEdges(network, deterministicChain, outputNode);
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
function _initializeDeterministicChain(
  network: Network,
  mutationProps: NetworkMutationProps,
  inputNode: Node,
  outputNode: Node,
): void {
  if (mutationProps._detChain) {
    return;
  }

  _ensureConnection(network, inputNode, outputNode);
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
function _pruneDeterministicChainExtraEdges(
  network: Network,
  deterministicChain: Node[],
  outputNode: Node,
): void {
  for (
    let chainNodeIndex = 0;
    chainNodeIndex < deterministicChain.length;
    chainNodeIndex++
  ) {
    const chainNode = deterministicChain[chainNodeIndex];
    const expectedTargetNode =
      chainNodeIndex + 1 < deterministicChain.length
        ? deterministicChain[chainNodeIndex + 1]
        : outputNode;

    const outgoingConnections = [...chainNode.connections.out];
    for (
      let outgoingConnectionIndex = 0;
      outgoingConnectionIndex < outgoingConnections.length;
      outgoingConnectionIndex++
    ) {
      const candidateConnection = outgoingConnections[outgoingConnectionIndex];
      if (candidateConnection.to !== expectedTargetNode) {
        _tryDisconnectConnection(network, candidateConnection);
      }
    }
  }
}

/**
 * Applies non-deterministic ADD_NODE by splitting a random connection.
 *
 * @param network - Target network.
 * @param mutationProps - Runtime mutation props.
 * @returns Nothing.
 */
function _addNodeRandomSplit(
  network: Network,
  mutationProps: NetworkMutationProps,
): void {
  if (!_ensureSeedForwardConnectionWhenEmpty(network)) {
    return;
  }

  const selectedConnection = _pickRandomEntry(
    network.connections,
    mutationProps._rand,
  );
  if (!selectedConnection) {
    return;
  }

  const previousGater = selectedConnection.gater;
  network.disconnect(selectedConnection.from, selectedConnection.to);

  const hiddenNode = _createHiddenNode(mutationProps._rand);
  _insertNodeBeforeOutputTail(
    network,
    hiddenNode,
    selectedConnection.to,
    mutationProps,
  );

  const sourceToHiddenConnection = _ensureConnection(
    network,
    selectedConnection.from,
    hiddenNode,
  );
  const hiddenToTargetConnection = _ensureConnection(
    network,
    hiddenNode,
    selectedConnection.to,
  );

  mutationProps._preferredChainEdge = hiddenToTargetConnection;

  if (previousGater && sourceToHiddenConnection && hiddenToTargetConnection) {
    const gatedConnection =
      mutationProps._rand() >= 0.5
        ? sourceToHiddenConnection
        : hiddenToTargetConnection;
    network.gate(previousGater, gatedConnection);
  }
}

/**
 * Ensures a seed input->output connection exists when connection list is empty.
 *
 * @param network - Target network.
 * @returns True when mutation may continue.
 */
function _ensureSeedForwardConnectionWhenEmpty(network: Network): boolean {
  if (network.connections.length > 0) {
    return true;
  }

  const inputNode = _findFirstNodeByType(network, 'input');
  const outputNode = _findFirstNodeByType(network, 'output');
  if (!inputNode || !outputNode) {
    return false;
  }

  _ensureConnection(network, inputNode, outputNode);
  return network.connections.length > 0;
}

/**
 * SUB_NODE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _subNode(this: Network): void {
  const hiddenNodes = _collectNodesByType(this, 'hidden');
  if (hiddenNodes.length === 0) {
    _warnWhenEnabled('No hidden nodes left to remove!');
    return;
  }

  const selectedHiddenNode = _pickRandomEntry(
    hiddenNodes,
    _asMutationProps(this)._rand,
  );
  if (!selectedHiddenNode) {
    return;
  }

  this.remove(selectedHiddenNode);
  const firstConnection = this.connections.at(0);
  if (firstConnection) {
    firstConnection.weight += 1e-4;
  }
}

/**
 * Collects nodes by type.
 *
 * @param network - Source network.
 * @param nodeType - Desired node type.
 * @returns Matching nodes.
 */
function _collectNodesByType(network: Network, nodeType: Node['type']): Node[] {
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
function _warnWhenEnabled(message: string): void {
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
function _addConn(this: Network): void {
  const mutationProps = _asMutationProps(this);
  _markTopoDirtyIfAcyclic(mutationProps);

  const connectionCandidates = _collectForwardConnectionCandidates(this);
  const selectedConnectionPair = _pickRandomEntry(
    connectionCandidates,
    mutationProps._rand,
  );
  if (!selectedConnectionPair) {
    return;
  }

  this.connect(selectedConnectionPair[0], selectedConnectionPair[1]);
}

/**
 * Collects forward connection candidates.
 *
 * @param network - Target network.
 * @returns Candidate source/target pairs.
 */
function _collectForwardConnectionCandidates(
  network: Network,
): Array<[Node, Node]> {
  const forwardConnectionCandidates: Array<[Node, Node]> = [];

  for (
    let sourceNodeIndex = 0;
    sourceNodeIndex < network.nodes.length - network.output;
    sourceNodeIndex++
  ) {
    const sourceNode = network.nodes[sourceNodeIndex];
    const targetStartIndex = Math.max(sourceNodeIndex + 1, network.input);

    for (
      let targetNodeIndex = targetStartIndex;
      targetNodeIndex < network.nodes.length;
      targetNodeIndex++
    ) {
      const targetNode = network.nodes[targetNodeIndex];
      if (!sourceNode.isProjectingTo(targetNode)) {
        forwardConnectionCandidates.push([sourceNode, targetNode]);
      }
    }
  }

  return forwardConnectionCandidates;
}

/**
 * SUB_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _subConn(this: Network): void {
  const removableConnections = _collectRemovableForwardConnections(this);
  const selectedConnection = _pickRandomEntry(
    removableConnections,
    _asMutationProps(this)._rand,
  );
  if (!selectedConnection) {
    return;
  }

  this.disconnect(selectedConnection.from, selectedConnection.to);
}

/**
 * Collects removable forward connections using redundancy constraints.
 *
 * @param network - Target network.
 * @returns Removable forward connections.
 */
function _collectRemovableForwardConnections(network: Network): Connection[] {
  const removableConnections: Connection[] = [];

  for (
    let connectionIndex = 0;
    connectionIndex < network.connections.length;
    connectionIndex++
  ) {
    const candidateConnection = network.connections[connectionIndex];
    if (_isRemovableForwardConnection(network, candidateConnection)) {
      removableConnections.push(candidateConnection);
    }
  }

  return removableConnections;
}

/**
 * Evaluates whether a forward connection is safe to remove.
 *
 * @param network - Target network.
 * @param candidateConnection - Connection under evaluation.
 * @returns True when removable.
 */
function _isRemovableForwardConnection(
  network: Network,
  candidateConnection: Connection,
): boolean {
  const sourceHasMultipleOutgoing =
    candidateConnection.from.connections.out.length > 1;
  const targetHasMultipleIncoming =
    candidateConnection.to.connections.in.length > 1;
  const isForwardDirection =
    network.nodes.indexOf(candidateConnection.to) >
    network.nodes.indexOf(candidateConnection.from);

  if (
    !sourceHasMultipleOutgoing ||
    !targetHasMultipleIncoming ||
    !isForwardDirection
  ) {
    return false;
  }

  return !_wouldDisconnectTargetPeerLayerGroup(network, candidateConnection);
}

/**
 * Determines whether removal would disconnect a target peer-layer group.
 *
 * @param network - Target network.
 * @param candidateConnection - Connection under evaluation.
 * @returns True when peer group would be disconnected.
 */
function _wouldDisconnectTargetPeerLayerGroup(
  network: Network,
  candidateConnection: Connection,
): boolean {
  const targetLayerPeers = _collectTargetLayerPeers(
    network,
    candidateConnection.to,
  );
  if (targetLayerPeers.length === 0) {
    return false;
  }

  let peerConnectionsFromSource = 0;
  for (
    let connectionIndex = 0;
    connectionIndex < network.connections.length;
    connectionIndex++
  ) {
    const existingConnection = network.connections[connectionIndex];
    if (existingConnection.from !== candidateConnection.from) {
      continue;
    }
    if (_containsNode(targetLayerPeers, existingConnection.to)) {
      peerConnectionsFromSource++;
    }
  }

  return peerConnectionsFromSource <= 1;
}

/**
 * Collects peers around a target node in the same type/layer neighborhood.
 *
 * @param network - Target network.
 * @param targetNode - Node whose peers are collected.
 * @returns Peer nodes.
 */
function _collectTargetLayerPeers(network: Network, targetNode: Node): Node[] {
  const peers: Node[] = [];
  const targetIndex = network.nodes.indexOf(targetNode);
  const maxDistance = Math.max(network.input, network.output);

  for (let nodeIndex = 0; nodeIndex < network.nodes.length; nodeIndex++) {
    const candidateNode = network.nodes[nodeIndex];
    if (candidateNode.type !== targetNode.type) {
      continue;
    }
    if (Math.abs(nodeIndex - targetIndex) < maxDistance) {
      peers.push(candidateNode);
    }
  }

  return peers;
}

/**
 * Checks whether a node list contains a node reference.
 *
 * @param nodes - Node list.
 * @param node - Node reference.
 * @returns True when contained.
 */
function _containsNode(nodes: Node[], node: Node): boolean {
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
function _modWeight(this: Network, method?: MutationMethod): void {
  const allConnections = _collectAllConnections(this);
  const mutationProps = _asMutationProps(this);
  const targetConnection = _pickRandomEntry(
    allConnections,
    mutationProps._rand,
  );
  if (!targetConnection) {
    return;
  }

  const methodObject = _resolveMethodObject(method);
  const minDelta = methodObject.min ?? -1;
  const maxDelta = methodObject.max ?? 1;
  const sampledDelta = _sampleUniform(mutationProps._rand, minDelta, maxDelta);
  targetConnection.weight += sampledDelta;
}

/**
 * Collects normal and self connections.
 *
 * @param network - Target network.
 * @returns Combined connections.
 */
function _collectAllConnections(network: Network): Connection[] {
  return [...network.connections, ...network.selfconns];
}

/**
 * Extracts method-object form when provided.
 *
 * @param method - Optional mutation method.
 * @returns Method object view.
 */
function _resolveMethodObject(
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
function _sampleUniform(
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
function _modBias(this: Network, method?: MutationMethod): void {
  const targetNode = _pickRandomNonInputNode(
    this,
    false,
    _asMutationProps(this)._rand,
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
function _pickRandomNonInputNode(
  network: Network,
  excludeOutputNodes: boolean,
  randomValue: () => number,
): Node | undefined {
  const mutableNodes = _collectMutableNonInputNodes(
    network,
    excludeOutputNodes,
  );
  return _pickRandomEntry(mutableNodes, randomValue);
}

/**
 * Collects mutable non-input nodes.
 *
 * @param network - Target network.
 * @param excludeOutputNodes - True to exclude output nodes.
 * @returns Mutable nodes.
 */
function _collectMutableNonInputNodes(
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
function _modActivation(this: Network, method?: MutationMethod): void {
  const methodObject = _resolveMethodObject(method);
  const canMutateOutput = methodObject.mutateOutput ?? true;
  const targetNode = _pickRandomNonInputNode(
    this,
    !canMutateOutput,
    _asMutationProps(this)._rand,
  );

  if (!targetNode) {
    _warnWhenEnabled(
      'No nodes available for activation function mutation based on config.',
    );
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
function _addSelfConn(this: Network): void {
  const mutationProps = _asMutationProps(this);
  if (mutationProps._enforceAcyclic) {
    return;
  }

  const candidates = _collectNodesWithoutSelfLoop(this);
  const targetNode = _pickRandomEntry(candidates, mutationProps._rand);
  if (!targetNode) {
    _warnWhenEnabled('All eligible nodes already have self-connections.');
    return;
  }

  this.connect(targetNode, targetNode);
}

/**
 * Collects non-input nodes that do not have self loops.
 *
 * @param network - Target network.
 * @returns Eligible nodes.
 */
function _collectNodesWithoutSelfLoop(network: Network): Node[] {
  const nodesWithoutSelfLoop: Node[] = [];

  for (
    let nodeIndex = network.input;
    nodeIndex < network.nodes.length;
    nodeIndex++
  ) {
    const candidateNode = network.nodes[nodeIndex];
    if (candidateNode.connections.self.length === 0) {
      nodesWithoutSelfLoop.push(candidateNode);
    }
  }

  return nodesWithoutSelfLoop;
}

/**
 * SUB_SELF_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _subSelfConn(this: Network): void {
  const mutationProps = _asMutationProps(this);
  const selectedSelfConnection = _pickRandomEntry(
    this.selfconns,
    mutationProps._rand,
  );
  if (!selectedSelfConnection) {
    _warnWhenEnabled('No self-connections exist to remove.');
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
function _addGate(this: Network): void {
  const mutationProps = _asMutationProps(this);
  const ungatedConnectionCandidates = _collectUngatedConnections(this);
  const gatingNode = _pickRandomNonInputNode(this, false, mutationProps._rand);
  const connectionToGate = _pickRandomEntry(
    ungatedConnectionCandidates,
    mutationProps._rand,
  );

  if (!gatingNode || !connectionToGate) {
    _warnWhenEnabled('All connections are already gated.');
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
function _collectUngatedConnections(network: Network): Connection[] {
  const allConnections = _collectAllConnections(network);
  const ungatedConnections: Connection[] = [];

  for (
    let connectionIndex = 0;
    connectionIndex < allConnections.length;
    connectionIndex++
  ) {
    const candidateConnection = allConnections[connectionIndex];
    if (candidateConnection.gater === null) {
      ungatedConnections.push(candidateConnection);
    }
  }

  return ungatedConnections;
}

/**
 * SUB_GATE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _subGate(this: Network): void {
  const mutationProps = _asMutationProps(this);
  const gatedConnection = _pickRandomEntry(this.gates, mutationProps._rand);
  if (!gatedConnection) {
    _warnWhenEnabled('No gated connections to ungate.');
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
function _addBackConn(this: Network): void {
  const mutationProps = _asMutationProps(this);
  if (mutationProps._enforceAcyclic) {
    return;
  }

  const backwardConnectionCandidates =
    _collectBackwardConnectionCandidates(this);
  const selectedConnectionPair = _pickRandomEntry(
    backwardConnectionCandidates,
    mutationProps._rand,
  );

  if (!selectedConnectionPair) {
    return;
  }

  this.connect(selectedConnectionPair[0], selectedConnectionPair[1]);
}

/**
 * Collects backward (recurrent) connection candidates.
 *
 * @param network - Target network.
 * @returns Candidate source/target pairs.
 */
function _collectBackwardConnectionCandidates(
  network: Network,
): Array<[Node, Node]> {
  const backwardConnectionCandidates: Array<[Node, Node]> = [];

  for (
    let laterNodeIndex = network.input;
    laterNodeIndex < network.nodes.length;
    laterNodeIndex++
  ) {
    const laterNode = network.nodes[laterNodeIndex];

    for (
      let earlierNodeIndex = network.input;
      earlierNodeIndex < laterNodeIndex;
      earlierNodeIndex++
    ) {
      const earlierNode = network.nodes[earlierNodeIndex];
      if (!laterNode.isProjectingTo(earlierNode)) {
        backwardConnectionCandidates.push([laterNode, earlierNode]);
      }
    }
  }

  return backwardConnectionCandidates;
}

/**
 * SUB_BACK_CONN mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _subBackConn(this: Network): void {
  const removableBackwardConnections =
    _collectRemovableBackwardConnections(this);
  const selectedConnection = _pickRandomEntry(
    removableBackwardConnections,
    _asMutationProps(this)._rand,
  );

  if (!selectedConnection) {
    return;
  }

  this.disconnect(selectedConnection.from, selectedConnection.to);
}

/**
 * Collects removable backward connections using redundancy constraints.
 *
 * @param network - Target network.
 * @returns Removable backward connections.
 */
function _collectRemovableBackwardConnections(network: Network): Connection[] {
  const removableConnections: Connection[] = [];

  for (
    let connectionIndex = 0;
    connectionIndex < network.connections.length;
    connectionIndex++
  ) {
    const candidateConnection = network.connections[connectionIndex];
    const sourceHasMultipleOutgoing =
      candidateConnection.from.connections.out.length > 1;
    const targetHasMultipleIncoming =
      candidateConnection.to.connections.in.length > 1;
    const isBackwardDirection =
      network.nodes.indexOf(candidateConnection.from) >
      network.nodes.indexOf(candidateConnection.to);

    if (
      sourceHasMultipleOutgoing &&
      targetHasMultipleIncoming &&
      isBackwardDirection
    ) {
      removableConnections.push(candidateConnection);
    }
  }

  return removableConnections;
}

/**
 * SWAP_NODES mutation.
 *
 * @param this - Bound network.
 * @param method - Optional method descriptor.
 * @returns Nothing.
 */
function _swapNodes(this: Network, method?: MutationMethod): void {
  const methodObject = _resolveMethodObject(method);
  const canSwapOutput = methodObject.mutateOutput ?? true;
  const mutationProps = _asMutationProps(this);
  const swappableNodes = _collectMutableNonInputNodes(this, !canSwapOutput);

  if (swappableNodes.length < 2) {
    return;
  }

  const firstNode = _pickRandomEntry(swappableNodes, mutationProps._rand);
  if (!firstNode) {
    return;
  }

  const secondNode = _pickDistinctRandomNode(
    swappableNodes,
    firstNode,
    mutationProps._rand,
  );

  if (!secondNode) {
    return;
  }

  _swapNodeBiasAndSquash(firstNode, secondNode);
}

/**
 * Picks a random node distinct from a given reference.
 *
 * @param nodeCandidates - Candidate nodes.
 * @param excludedNode - Node to exclude.
 * @param randomValue - Random generator.
 * @returns Distinct node or undefined.
 */
function _pickDistinctRandomNode(
  nodeCandidates: Node[],
  excludedNode: Node,
  randomValue: () => number,
): Node | undefined {
  if (nodeCandidates.length < 2) {
    return undefined;
  }

  for (let attemptIndex = 0; attemptIndex < 16; attemptIndex++) {
    const candidateNode = _pickRandomEntry(nodeCandidates, randomValue);
    if (candidateNode && candidateNode !== excludedNode) {
      return candidateNode;
    }
  }

  for (let nodeIndex = 0; nodeIndex < nodeCandidates.length; nodeIndex++) {
    const candidateNode = nodeCandidates[nodeIndex];
    if (candidateNode !== excludedNode) {
      return candidateNode;
    }
  }

  return undefined;
}

/**
 * Swaps bias and squash values between two nodes.
 *
 * @param firstNode - First node.
 * @param secondNode - Second node.
 * @returns Nothing.
 */
function _swapNodeBiasAndSquash(firstNode: Node, secondNode: Node): void {
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
function _addLSTMNode(this: Network): void {
  const mutationProps = _asMutationProps(this);
  if (mutationProps._enforceAcyclic || this.connections.length === 0) {
    return;
  }

  const selectedConnection = _pickRandomEntry(
    this.connections,
    mutationProps._rand,
  );
  if (!selectedConnection) {
    return;
  }

  _expandConnectionWithRecurrentBlock(this, selectedConnection, 'lstm');
}

/**
 * ADD_GRU_NODE mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _addGRUNode(this: Network): void {
  const mutationProps = _asMutationProps(this);
  if (mutationProps._enforceAcyclic || this.connections.length === 0) {
    return;
  }

  const selectedConnection = _pickRandomEntry(
    this.connections,
    mutationProps._rand,
  );
  if (!selectedConnection) {
    return;
  }

  _expandConnectionWithRecurrentBlock(this, selectedConnection, 'gru');
}

/**
 * Replaces one connection with a minimal recurrent block.
 *
 * @param network - Target network.
 * @param connectionToExpand - Connection to replace.
 * @param blockType - Recurrent block type.
 * @returns Nothing.
 */
function _expandConnectionWithRecurrentBlock(
  network: Network,
  connectionToExpand: Connection,
  blockType: 'lstm' | 'gru',
): void {
  const previousGater = connectionToExpand.gater;
  network.disconnect(connectionToExpand.from, connectionToExpand.to);

  const recurrentLayer = _createRecurrentLayer(blockType);
  _appendRecurrentLayerNodes(network, recurrentLayer.nodes);

  network.connect(connectionToExpand.from, recurrentLayer.nodes[0]);
  network.connect(recurrentLayer.output.nodes[0], connectionToExpand.to);

  if (!previousGater) {
    return;
  }

  const latestConnection = network.connections.at(-1);
  if (latestConnection) {
    network.gate(previousGater, latestConnection);
  }
}

/**
 * Creates recurrent layer by type.
 *
 * @param blockType - Recurrent block type.
 * @returns Created recurrent layer.
 */
function _createRecurrentLayer(blockType: 'lstm' | 'gru'): {
  nodes: Node[];
  output: { nodes: Node[] };
} {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const Layer = require('../layer').default;
  if (blockType === 'lstm') {
    return Layer.lstm(1);
  }
  return Layer.gru(1);
}

/**
 * Appends recurrent layer nodes as hidden nodes.
 *
 * @param network - Target network.
 * @param layerNodes - Layer nodes.
 * @returns Nothing.
 */
function _appendRecurrentLayerNodes(
  network: Network,
  layerNodes: Node[],
): void {
  for (let nodeIndex = 0; nodeIndex < layerNodes.length; nodeIndex++) {
    const layerNode = layerNodes[nodeIndex];
    layerNode.type = 'hidden';
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
function _reinitWeight(this: Network, method?: MutationMethod): void {
  const mutationProps = _asMutationProps(this);
  const targetNode = _pickRandomNonInputNode(this, false, mutationProps._rand);
  if (!targetNode) {
    return;
  }

  const methodObject = _resolveMethodObject(method);
  const minWeight = methodObject.min ?? -1;
  const maxWeight = methodObject.max ?? 1;

  _reinitializeConnectionGroupWeights(
    targetNode.connections.in,
    mutationProps._rand,
    minWeight,
    maxWeight,
  );
  _reinitializeConnectionGroupWeights(
    targetNode.connections.out,
    mutationProps._rand,
    minWeight,
    maxWeight,
  );
  _reinitializeConnectionGroupWeights(
    targetNode.connections.self,
    mutationProps._rand,
    minWeight,
    maxWeight,
  );
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
function _reinitializeConnectionGroupWeights(
  connections: Connection[],
  randomValue: () => number,
  minWeight: number,
  maxWeight: number,
): void {
  for (
    let connectionIndex = 0;
    connectionIndex < connections.length;
    connectionIndex++
  ) {
    const connection = connections[connectionIndex];
    connection.weight = _sampleUniform(randomValue, minWeight, maxWeight);
  }
}

/**
 * BATCH_NORM mutation.
 *
 * @param this - Bound network.
 * @returns Nothing.
 */
function _batchNorm(this: Network): void {
  const hiddenNodes = _collectNodesByType(this, 'hidden');
  const selectedHiddenNode = _pickRandomEntry(
    hiddenNodes,
    _asMutationProps(this)._rand,
  );
  if (!selectedHiddenNode) {
    return;
  }

  (selectedHiddenNode as unknown as { _batchNorm: boolean })._batchNorm = true;
}
