import type {
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

/**
 * Dead-end repair helpers.
 *
 * This chapter owns the connectivity repair pass that makes sure input,
 * output, and hidden nodes are not stranded without the minimum in/out edges
 * expected by the surrounding mutation and evaluation flows.
 */

/**
 * Collect categorized node arrays for dead-end repair.
 *
 * @param networkToInspect - network to inspect
 * @returns grouped node arrays
 */
export function collectNodeGroupsForDeadEnds(
  networkToInspect: GenomeWithMetadata,
): {
  inputNodes: NodeWithMetadata[];
  outputNodes: NodeWithMetadata[];
  hiddenNodes: NodeWithMetadata[];
} {
  // Step 1: split nodes by type.
  return {
    inputNodes: networkToInspect.nodes.filter(
      (node: NodeWithMetadata) => node.type === 'input',
    ),
    outputNodes: networkToInspect.nodes.filter(
      (node: NodeWithMetadata) => node.type === 'output',
    ),
    hiddenNodes: networkToInspect.nodes.filter(
      (node: NodeWithMetadata) => node.type === 'hidden',
    ),
  };
}

/**
 * Ensure all input nodes have at least one outgoing connection.
 *
 * @param networkToEdit - network to edit
 * @param nodeGroupsToUse - grouped node arrays
 * @param internal - neat controller context
 * @returns void
 */
export function ensureInputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: {
    inputNodes: NodeWithMetadata[];
    outputNodes: NodeWithMetadata[];
    hiddenNodes: NodeWithMetadata[];
  },
  internal: NeatControllerForMutation,
): void {
  // Step 1: connect any input node lacking outgoing edges.
  for (const inputNode of nodeGroupsToUse.inputNodes) {
    if (hasOutgoingForDeadEnds(inputNode)) continue;

    const candidates = nodeGroupsToUse.hiddenNodes.length
      ? nodeGroupsToUse.hiddenNodes
      : nodeGroupsToUse.outputNodes;
    connectIfCandidatesExistForDeadEnds(
      networkToEdit,
      inputNode,
      candidates,
      false,
      internal,
    );
  }
}

/**
 * Ensure all output nodes have at least one incoming connection.
 *
 * @param networkToEdit - network to edit
 * @param nodeGroupsToUse - grouped node arrays
 * @param internal - neat controller context
 * @returns void
 */
export function ensureOutputConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: {
    inputNodes: NodeWithMetadata[];
    outputNodes: NodeWithMetadata[];
    hiddenNodes: NodeWithMetadata[];
  },
  internal: NeatControllerForMutation,
): void {
  // Step 1: connect any output node lacking incoming edges.
  for (const outputNode of nodeGroupsToUse.outputNodes) {
    if (hasIncomingForDeadEnds(outputNode)) continue;

    const candidates = nodeGroupsToUse.hiddenNodes.length
      ? nodeGroupsToUse.hiddenNodes
      : nodeGroupsToUse.inputNodes;
    connectIfCandidatesExistForDeadEnds(
      networkToEdit,
      outputNode,
      candidates,
      true,
      internal,
    );
  }
}

/**
 * Ensure hidden nodes have both incoming and outgoing connections.
 *
 * @param networkToEdit - network to edit
 * @param nodeGroupsToUse - grouped node arrays
 * @param internal - neat controller context
 * @returns void
 */
export function ensureHiddenConnectivityForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: {
    inputNodes: NodeWithMetadata[];
    outputNodes: NodeWithMetadata[];
    hiddenNodes: NodeWithMetadata[];
  },
  internal: NeatControllerForMutation,
): void {
  // Step 1: fix missing inbound and outbound edges for each hidden node.
  for (const hiddenNode of nodeGroupsToUse.hiddenNodes) {
    if (!hasIncomingForDeadEnds(hiddenNode)) {
      const incomingCandidates = nodeGroupsToUse.inputNodes.concat(
        nodeGroupsToUse.hiddenNodes.filter((node) => node !== hiddenNode),
      );
      connectIfCandidatesExistForDeadEnds(
        networkToEdit,
        hiddenNode,
        incomingCandidates,
        true,
        internal,
      );
    }

    if (!hasOutgoingForDeadEnds(hiddenNode)) {
      const outgoingCandidates = nodeGroupsToUse.outputNodes.concat(
        nodeGroupsToUse.hiddenNodes.filter((node) => node !== hiddenNode),
      );
      connectIfCandidatesExistForDeadEnds(
        networkToEdit,
        hiddenNode,
        outgoingCandidates,
        false,
        internal,
      );
    }
  }
}

/**
 * Connect a node to a random candidate if candidates exist.
 *
 * @param networkToEdit - network to edit
 * @param anchorNode - node to connect from/to
 * @param candidates - candidate nodes for connection
 * @param reverse - whether to connect candidate -> anchor
 * @param internal - neat controller context
 * @returns void
 */
export function connectIfCandidatesExistForDeadEnds(
  networkToEdit: GenomeWithMetadata,
  anchorNode: NodeWithMetadata,
  candidates: NodeWithMetadata[],
  reverse: boolean,
  internal: NeatControllerForMutation,
): void {
  // Step 1: return when no candidates exist.
  if (!candidates.length) return;

  // Step 2: choose a random candidate.
  const chosenNode = chooseRandomNodeForDeadEnds(candidates, internal);
  if (!chosenNode) return;

  // Step 3: connect nodes with correct orientation.
  try {
    if (reverse) {
      networkToEdit.connect?.(chosenNode, anchorNode);
    } else {
      networkToEdit.connect?.(anchorNode, chosenNode);
    }
  } catch {
    // Intentionally ignore: connection may fail if nodes are incompatible.
  }
}

/**
 * Check whether a node has any outgoing connections.
 *
 * @param node - node to inspect
 * @returns true when outgoing connections exist
 */
export function hasOutgoingForDeadEnds(node: NodeWithMetadata): boolean {
  // Step 1: confirm outgoing connection list is non-empty.
  return node.connections.out.length > 0;
}

/**
 * Check whether a node has any incoming connections.
 *
 * @param node - node to inspect
 * @returns true when incoming connections exist
 */
export function hasIncomingForDeadEnds(node: NodeWithMetadata): boolean {
  // Step 1: confirm incoming connection list is non-empty.
  return node.connections.in.length > 0;
}

/**
 * Choose a random node from candidates for dead-end repair.
 *
 * @param candidates - candidate nodes
 * @param internal - neat controller context
 * @returns selected node or null
 */
export function chooseRandomNodeForDeadEnds(
  candidates: NodeWithMetadata[],
  internal: NeatControllerForMutation,
): NodeWithMetadata | null {
  // Step 1: return null when no candidates exist.
  if (!candidates.length) return null;

  // Step 2: sample with the controller RNG.
  const randomValue = internal._getRNG()();
  const chosenIndex = Math.floor(randomValue * candidates.length);
  return candidates[chosenIndex] ?? null;
}
