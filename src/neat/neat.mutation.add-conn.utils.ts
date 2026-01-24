import type {
  GenomeWithMetadata,
  NodeWithMetadata,
  ConnectionWithMetadata,
  NeatControllerForMutation,
} from './neat.mutation.types';

/** Default gene id placeholder when missing. */
const DEFAULT_GENE_ID = 0;

// ============================================================================
// Helpers for mutateAddConnReuse() function
// ============================================================================

/**
 * Collect legal (from,to) node pairs not already connected.
 *
 * @param genomeToInspect - genome to scan
 * @returns candidate node pairs
 */
export function collectCandidatePairsForConn(
  genomeToInspect: GenomeWithMetadata,
): Array<[NodeWithMetadata, NodeWithMetadata]> {
  // Step 1: allocate the candidate pair list.
  const pairs: Array<[NodeWithMetadata, NodeWithMetadata]> = [];

  // Step 2: build candidate pairs respecting node ordering.
  for (
    let sourceIndex = 0;
    sourceIndex < genomeToInspect.nodes.length - genomeToInspect.output;
    sourceIndex++
  ) {
    const sourceNode = genomeToInspect.nodes[sourceIndex];
    for (
      let targetIndex = Math.max(sourceIndex + 1, genomeToInspect.input);
      targetIndex < genomeToInspect.nodes.length;
      targetIndex++
    ) {
      const targetNode = genomeToInspect.nodes[targetIndex];
      if (!sourceNode.isProjectingTo?.(targetNode)) {
        pairs.push([sourceNode, targetNode]);
      }
    }
  }
  return pairs;
}

/**
 * Filter candidate pairs that already have innovation reuse keys.
 *
 * @param pairs - candidate node pairs
 * @param internal - neat controller context
 * @returns reuse candidates
 */
export function filterPairsWithInnovations(
  pairs: Array<[NodeWithMetadata, NodeWithMetadata]>,
  internal: NeatControllerForMutation,
): Array<[NodeWithMetadata, NodeWithMetadata]> {
  // Step 1: include only pairs with known innovation keys.
  return pairs.filter((pair) => {
    const symmetricKey = buildSymmetricKeyForConn(pair[0], pair[1]);
    return internal._connInnovations.has(symmetricKey);
  });
}

/**
 * Build the final selection pool based on reuse and hidden-node preference.
 *
 * @param allPairs - all candidate pairs
 * @param reusePairs - pairs with historical innovations
 * @returns selection pool
 */
export function selectPairPool(
  allPairs: Array<[NodeWithMetadata, NodeWithMetadata]>,
  reusePairs: Array<[NodeWithMetadata, NodeWithMetadata]>,
): Array<[NodeWithMetadata, NodeWithMetadata]> {
  // Step 1: honor reuse candidates when available.
  if (reusePairs.length) return reusePairs;

  // Step 2: prefer hidden-hidden pairs when present.
  const hiddenPairs = allPairs.filter(
    (pair) => pair[0].type === 'hidden' && pair[1].type === 'hidden',
  );
  if (hiddenPairs.length) return hiddenPairs;

  // Step 3: fall back to all candidates.
  return allPairs;
}

/**
 * Choose a pair deterministically when only one candidate exists.
 *
 * @param pairs - selection pool
 * @param internal - neat controller context
 * @returns chosen pair or null
 */
export function choosePairForConn(
  pairs: Array<[NodeWithMetadata, NodeWithMetadata]>,
  internal: NeatControllerForMutation,
): [NodeWithMetadata, NodeWithMetadata] | null {
  // Step 1: return null when no pairs exist.
  if (!pairs.length) return null;

  // Step 2: select deterministically when only one pair exists.
  if (pairs.length === 1) return pairs[0];

  // Step 3: sample using the controller RNG.
  const randomValue = internal._getRNG()();
  const chosenIndex = Math.floor(randomValue * pairs.length);
  return pairs[chosenIndex] ?? null;
}

/**
 * Resolve nodes and innovation key details for a chosen pair.
 *
 * @param chosenPair - pair to connect
 * @returns resolved pair metadata
 */
export function resolvePairNodes(
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
): {
  sourceNode: NodeWithMetadata;
  targetNode: NodeWithMetadata;
  symmetricKey: string;
  legacyForwardKey: string;
  legacyReverseKey: string;
} {
  // Step 1: pull source and target nodes.
  const sourceNode = chosenPair[0];
  const targetNode = chosenPair[1];

  // Step 2: compute innovation keys.
  const symmetricKey = buildSymmetricKeyForConn(sourceNode, targetNode);
  const legacyForwardKey = buildLegacyKeyForConn(sourceNode, targetNode);
  const legacyReverseKey = buildLegacyKeyForConn(targetNode, sourceNode);

  return {
    sourceNode,
    targetNode,
    symmetricKey,
    legacyForwardKey,
    legacyReverseKey,
  };
}

/**
 * Determine whether adding the connection would create a cycle.
 *
 * @param genomeToInspect - genome to inspect
 * @param pairNodes - resolved pair nodes
 * @returns true if the connection should be aborted
 */
export function shouldAbortForCycle(
  genomeToInspect: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata },
): boolean {
  // Step 1: skip when acyclic enforcement is disabled.
  if (!genomeToInspect._enforceAcyclic) return false;

  // Step 2: detect cycles with a DFS from the target node.
  return createsCycle(pairNodes.sourceNode, pairNodes.targetNode);
}

/**
 * Create the connection for the chosen pair.
 *
 * @param genomeToEdit - genome to edit
 * @param pairNodes - resolved pair nodes
 * @returns created connection or undefined
 */
export function connectChosenPair(
  genomeToEdit: GenomeWithMetadata,
  pairNodes: { sourceNode: NodeWithMetadata; targetNode: NodeWithMetadata },
): ConnectionWithMetadata | undefined {
  // Step 1: attempt to connect the chosen nodes.
  return genomeToEdit.connect?.(
    pairNodes.sourceNode,
    pairNodes.targetNode,
  )?.[0];
}

/**
 * Assign an innovation id for a new connection, reusing when possible.
 *
 * @param connection - newly created connection
 * @param pairNodes - resolved pair metadata
 * @param internal - neat controller context
 * @returns void
 */
export function assignInnovationForConnection(
  connection: ConnectionWithMetadata,
  pairNodes: {
    symmetricKey: string;
    legacyForwardKey: string;
    legacyReverseKey: string;
  },
  internal: NeatControllerForMutation,
): void {
  // Step 1: reuse existing innovation when present.
  if (internal._connInnovations.has(pairNodes.symmetricKey)) {
    connection.innovation = internal._connInnovations.get(
      pairNodes.symmetricKey,
    )!;
    return;
  }

  // Step 2: allocate and store a new innovation id.
  const newInnovationId = internal._nextGlobalInnovation++;
  connection.innovation = newInnovationId;
  internal._connInnovations.set(pairNodes.symmetricKey, newInnovationId);
  internal._connInnovations.set(pairNodes.legacyForwardKey, newInnovationId);
  internal._connInnovations.set(pairNodes.legacyReverseKey, newInnovationId);
}

/**
 * Build a symmetric innovation key for an unordered node pair.
 *
 * @param sourceNode - source node
 * @param targetNode - target node
 * @returns symmetric innovation key
 */
export function buildSymmetricKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string {
  // Step 1: normalize gene ids for ordering.
  const sourceGeneId = sourceNode.geneId ?? DEFAULT_GENE_ID;
  const targetGeneId = targetNode.geneId ?? DEFAULT_GENE_ID;
  if (sourceGeneId < targetGeneId) {
    return `${sourceGeneId}::${targetGeneId}`;
  }
  return `${targetGeneId}::${sourceGeneId}`;
}

/**
 * Build a legacy directional innovation key.
 *
 * @param sourceNode - source node
 * @param targetNode - target node
 * @returns directional innovation key
 */
export function buildLegacyKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string {
  // Step 1: normalize gene ids for the directional key.
  const sourceGeneId = sourceNode.geneId ?? DEFAULT_GENE_ID;
  const targetGeneId = targetNode.geneId ?? DEFAULT_GENE_ID;
  return `${sourceGeneId}::${targetGeneId}`;
}

/**
 * Detect whether adding a connection would create a cycle.
 *
 * @param sourceNode - source node of the new connection
 * @param targetNode - target node of the new connection
 * @returns true when a cycle is detected
 */
export function createsCycle(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): boolean {
  // Step 1: initialize DFS structures from the target node.
  const stack = [targetNode];
  const visitedNodes = new Set<NodeWithMetadata>();

  // Step 2: walk outward and detect a path back to the source.
  while (stack.length) {
    const currentNode = stack.pop()!;
    if (currentNode === sourceNode) return true;
    if (visitedNodes.has(currentNode)) continue;
    visitedNodes.add(currentNode);
    for (const connection of currentNode.connections.out) {
      stack.push(connection.to);
    }
  }
  return false;
}
