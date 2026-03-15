import type {
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

/**
 * Minimum-hidden repair helpers.
 *
 * This chapter owns the small maintenance pass that enforces a minimum hidden
 * node budget and rewires newly created hidden nodes so they remain connected
 * enough for later mutation and evaluation passes.
 */

/**
 * Collect categorized node arrays for the network.
 *
 * @param networkToInspect - network to inspect
 * @returns grouped node arrays
 */
export function collectNodeGroupsForMinHidden(
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
 * Resolve the maximum node limit for the network.
 *
 * @param internal - neat controller context
 * @returns maximum node limit
 */
export function resolveMaxNodesForMinHidden(
  internal: NeatControllerForMutation,
): number {
  // Step 1: fall back to Infinity when not configured.
  return internal.options.maxNodes || Infinity;
}

/**
 * Resolve the minimum hidden node requirement for the network.
 *
 * @param networkToInspect - network to inspect
 * @param maxNodesLimit - maximum allowed nodes
 * @param multiplier - optional size multiplier
 * @param internal - neat controller context
 * @returns minimum hidden node count
 */
export function resolveMinHiddenForMinHidden(
  networkToInspect: GenomeWithMetadata,
  maxNodesLimit: number,
  multiplier: number | undefined,
  internal: NeatControllerForMutation,
): number {
  // Step 1: determine how many non-hidden nodes exist.
  const nonHiddenCount = networkToInspect.nodes.filter(
    (node: NodeWithMetadata) => node.type !== 'hidden',
  ).length;

  // Step 2: cap minimum hidden size based on maximum nodes.
  const minimumHidden = internal.getMinimumHiddenSize?.(multiplier) ?? 0;
  return Math.min(minimumHidden, maxNodesLimit - nonHiddenCount);
}

/**
 * Check whether the network has at least one input and output node.
 *
 * @param nodeGroupsToCheck - grouped node arrays
 * @returns true when inputs and outputs are present
 */
export function hasRequiredEndpointsForMinHidden(nodeGroupsToCheck: {
  inputNodes: NodeWithMetadata[];
  outputNodes: NodeWithMetadata[];
}): boolean {
  // Step 1: ensure both input and output lists are non-empty.
  return (
    nodeGroupsToCheck.inputNodes.length > 0 &&
    nodeGroupsToCheck.outputNodes.length > 0
  );
}

/**
 * Emit a warning when the network lacks input or output nodes.
 *
 * @returns void
 */
export function warnMissingEndpointsForMinHidden(): void {
  // Step 1: attempt to log the warning safely.
  try {
    console.warn(
      'Network is missing input or output nodes — skipping minHidden enforcement',
    );
  } catch {
    // Intentionally ignore: console may not be available in all environments.
  }
}

/**
 * Ensure the network has at least the minimum number of hidden nodes.
 *
 * @param networkToEdit - network to edit
 * @param nodeGroupsToEdit - grouped node arrays
 * @param minimumHidden - minimum hidden nodes required
 * @param maxNodesLimit - maximum allowed nodes
 * @returns Promise resolving when nodes are created
 */
export async function ensureHiddenNodeCountForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToEdit: {
    hiddenNodes: NodeWithMetadata[];
  },
  minimumHidden: number,
  maxNodesLimit: number,
): Promise<void> {
  // Step 1: return early when minimum is already satisfied.
  if (nodeGroupsToEdit.hiddenNodes.length >= minimumHidden) return;

  // Step 2: create hidden nodes until the minimum is satisfied.
  const { default: NodeClass } = await import('../../../architecture/node');
  while (
    nodeGroupsToEdit.hiddenNodes.length < minimumHidden &&
    networkToEdit.nodes.length < maxNodesLimit
  ) {
    const newNode = new NodeClass('hidden') as unknown as NodeWithMetadata;
    networkToEdit.nodes.push(newNode);
    nodeGroupsToEdit.hiddenNodes.push(newNode);
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
export function ensureHiddenConnectivityForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: {
    inputNodes: NodeWithMetadata[];
    outputNodes: NodeWithMetadata[];
    hiddenNodes: NodeWithMetadata[];
  },
  internal: NeatControllerForMutation,
): void {
  // Step 1: ensure incoming and outgoing connections for each hidden node.
  for (const hiddenNode of nodeGroupsToUse.hiddenNodes) {
    ensureIncomingConnectionForMinHidden(
      networkToEdit,
      nodeGroupsToUse,
      hiddenNode,
      internal,
    );
    ensureOutgoingConnectionForMinHidden(
      networkToEdit,
      nodeGroupsToUse,
      hiddenNode,
      internal,
    );
  }
}

/** Baseline minimum hidden nodes when no configuration is provided. */
export const MINIMUM_HIDDEN_BASELINE = 0;

/**
 * Compute the minimum hidden node count using explicit or multiplier-based settings.
 *
 * @param inputCount - Number of input nodes in the network.
 * @param outputCount - Number of output nodes in the network.
 * @param explicitMinimumHidden - Optional explicit minimum hidden count.
 * @param hiddenMultiplier - Optional multiplier used when explicit minimum is absent.
 * @returns Minimum hidden node requirement.
 */
export function computeMinimumHiddenSize(
  inputCount: number,
  outputCount: number,
  explicitMinimumHidden?: number,
  hiddenMultiplier?: number,
): number {
  if (typeof explicitMinimumHidden === 'number') return explicitMinimumHidden;
  if (
    typeof hiddenMultiplier === 'number' &&
    Number.isFinite(hiddenMultiplier)
  ) {
    const weightedNodeTotal = hiddenMultiplier * (inputCount + outputCount);
    return Math.max(MINIMUM_HIDDEN_BASELINE, Math.round(weightedNodeTotal));
  }
  return MINIMUM_HIDDEN_BASELINE;
}

/**
 * Ensure a hidden node has at least one incoming connection.
 *
 * @param networkToEdit - network to edit
 * @param nodeGroupsToUse - grouped node arrays
 * @param hiddenNode - hidden node to connect
 * @param internal - neat controller context
 * @returns void
 */
export function ensureIncomingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: {
    inputNodes: NodeWithMetadata[];
    hiddenNodes: NodeWithMetadata[];
  },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void {
  // Step 1: skip when an incoming connection already exists.
  if (hiddenNode.connections.in.length > 0) return;

  // Step 2: build candidate sources.
  const candidates = nodeGroupsToUse.inputNodes.concat(
    nodeGroupsToUse.hiddenNodes.filter((node) => node !== hiddenNode),
  );
  if (!candidates.length) return;

  // Step 3: connect a random candidate to the hidden node.
  const sourceNode = chooseRandomNodeForMinHidden(candidates, internal);
  if (!sourceNode) return;
  try {
    networkToEdit.connect?.(sourceNode, hiddenNode);
  } catch {
    // Intentionally ignore: connection may fail if nodes are incompatible.
  }
}

/**
 * Ensure a hidden node has at least one outgoing connection.
 *
 * @param networkToEdit - network to edit
 * @param nodeGroupsToUse - grouped node arrays
 * @param hiddenNode - hidden node to connect
 * @param internal - neat controller context
 * @returns void
 */
export function ensureOutgoingConnectionForMinHidden(
  networkToEdit: GenomeWithMetadata,
  nodeGroupsToUse: {
    outputNodes: NodeWithMetadata[];
    hiddenNodes: NodeWithMetadata[];
  },
  hiddenNode: NodeWithMetadata,
  internal: NeatControllerForMutation,
): void {
  // Step 1: skip when an outgoing connection already exists.
  if (hiddenNode.connections.out.length > 0) return;

  // Step 2: build candidate targets.
  const candidates = nodeGroupsToUse.outputNodes.concat(
    nodeGroupsToUse.hiddenNodes.filter((node) => node !== hiddenNode),
  );
  if (!candidates.length) return;

  // Step 3: connect the hidden node to a random candidate.
  const targetNode = chooseRandomNodeForMinHidden(candidates, internal);
  if (!targetNode) return;
  try {
    networkToEdit.connect?.(hiddenNode, targetNode);
  } catch {
    // Intentionally ignore: connection may fail if nodes are incompatible.
  }
}

/**
 * Choose a random node from a candidate list.
 *
 * @param candidates - candidate nodes
 * @param internal - neat controller context
 * @returns selected node or null
 */
export function chooseRandomNodeForMinHidden(
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

/**
 * Rebuild connection caches after structural edits.
 *
 * @param networkToEdit - network to rebuild
 * @returns Promise resolving after rebuild completes
 */
export async function rebuildNetworkConnectionsForMinHidden(
  networkToEdit: GenomeWithMetadata,
): Promise<void> {
  // Step 1: rebuild using the Network class helper.
  const { default: NetworkClass } =
    await import('../../../architecture/network');
  NetworkClass.rebuildConnections(networkToEdit as never);
}
