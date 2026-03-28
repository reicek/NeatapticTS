import type {
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

/**
 * Dead-end repair helpers.
 *
 * This file owns the connectivity half of the mutation-repair chapter.
 *
 * The surrounding `repair/` folder preserves one structural-viability policy
 * with two complementary concerns:
 * - this file repairs nodes that have become stranded with no legal inbound or
 *   outbound path,
 * - `mutation.min-hidden.ts` enforces a minimum hidden-node floor when the
 *   controller expects richer internal structure.
 *
 * Read this file when the network already has the right rough size but may have
 * broken local connectivity after import, mutation, crossover, or pruning.
 * The goal is not to redesign topology. The goal is to restore the smallest
 * practical set of connections that makes inputs, outputs, and hidden nodes
 * usable again.
 *
 * The repair flow is easiest to retain in four steps:
 * 1. split the network into input, output, and hidden node groups,
 * 2. repair stranded inputs,
 * 3. repair stranded outputs,
 * 4. repair hidden nodes that lost either their inbound or outbound side.
 *
 * ```mermaid
 * flowchart TD
 *   Inspect[Inspect current network] --> Groups[Group input output and hidden nodes]
 *   Groups --> Inputs[Repair stranded inputs]
 *   Inputs --> Outputs[Repair stranded outputs]
 *   Outputs --> Hidden[Repair hidden inbound and outbound gaps]
 *   Hidden --> Ready[Network regains basic connectivity]
 * ```
 */

/**
 * Collect categorized node arrays for dead-end repair.
 *
 * This is the chapter's shared preparation step. The later helpers all ask the
 * same structural question from different angles, so grouping the nodes once
 * keeps the repair flow declarative and avoids repeating node-type scans.
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
 * Inputs are the first dead-end family because an input with no outgoing edge
 * cannot influence the rest of the network at all. The helper prefers routing
 * into hidden nodes when they exist and falls back to direct output links when
 * the network has no hidden layer yet.
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
  // Step 1: connect each input node lacking outgoing edges.
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
 * Outputs are the second dead-end family. An output with no inbound edge can be
 * observed by later scoring code, but it carries no meaningful signal. The
 * helper therefore reconnects it from hidden nodes first and from inputs when
 * no hidden layer exists.
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
  // Step 1: connect each output node lacking incoming edges.
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
 * Hidden nodes are the most delicate repair family because they must stay on a
 * usable path through the network. A hidden node with only one side connected
 * is structural dead weight, so this helper repairs the missing side without
 * disturbing hidden nodes that are already participating in a path.
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
 * This is the chapter's small best-effort wiring primitive. It does not decide
 * whether repair should happen; it only applies one candidate connection in the
 * requested direction and tolerates incompatible node pairs without turning a
 * maintenance pass into a fatal error.
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
 * Check whether a node has outgoing connections.
 *
 * Dead-end repair uses this as the smallest possible structural predicate: if
 * the outgoing list is empty, the node cannot currently send signal forward.
 *
 * @param node - node to inspect
 * @returns true when outgoing connections exist
 */
export function hasOutgoingForDeadEnds(node: NodeWithMetadata): boolean {
  // Step 1: confirm outgoing connection list is non-empty.
  return node.connections.out.length > 0;
}

/**
 * Check whether a node has incoming connections.
 *
 * This is the inbound twin of {@link hasOutgoingForDeadEnds}. It answers the
 * local question "can an upstream node currently reach this one?" before the
 * higher-level repair helpers decide whether to reconnect it.
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
 * Repair deliberately stays lightweight and non-optimizing. Once a helper has
 * found a legal candidate pool, this selector uses the controller RNG to pick
 * one reconnection target or source without imposing another ranking policy on
 * the maintenance path.
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
