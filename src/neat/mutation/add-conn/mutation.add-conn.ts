import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

/** Default gene id placeholder when missing. */
const DEFAULT_GENE_ID = 0;

/**
 * Add-connection mutation helpers.
 *
 * This chapter owns candidate-pair discovery, cycle guarding, and innovation-id
 * reuse for newly added structural connections.
 *
 * Where `add-node/` grows structure by splitting one existing edge, this
 * chapter grows structure by discovering a legal pair of nodes that are not yet
 * connected. That sounds simpler, but it still has several responsibilities:
 * preserve innovation identity when the same node pair has been connected
 * before, prefer historically meaningful or structurally useful candidates, and
 * avoid illegal recurrent edges when acyclic topology is required.
 *
 * The flow is easiest to read as a pipeline:
 *
 * 1. enumerate legal candidate pairs,
 * 2. narrow the pool toward reusable or hidden-hidden pairs when possible,
 * 3. choose one pair,
 * 4. resolve the innovation keys for that pair,
 * 5. abort if the new edge would violate cycle policy,
 * 6. connect the pair and assign a reused or fresh innovation id.
 *
 * Read this chapter in that order when debugging connection growth.
 *
 * ```mermaid
 * flowchart TD
 *   Genome[Genome enters add-connection path] --> Candidates[Collect legal node pairs]
 *   Candidates --> Reuse[Filter pairs with known innovations]
 *   Reuse --> Pool[Choose reuse pool or structural fallback pool]
 *   Pool --> Pair[Choose one node pair]
 *   Pair --> Keys[Resolve symmetric and legacy keys]
 *   Keys --> Cycle{Would this edge create a cycle?}
 *   Cycle -->|yes| Abort[Skip structural edit]
 *   Cycle -->|no| Connect[Create new connection]
 *   Connect --> Innovation[Reuse or assign innovation id]
 * ```
 */

/**
 * Collect legal (from,to) node pairs not already connected.
 *
 * This helper defines the search space for connection growth. It respects the
 * node ordering conventions used by the genome representation so mutation does
 * not propose obviously invalid source-target directions before later cycle
 * checks even run.
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
 * Reuse candidates are especially valuable because they let independently
 * discovered structure share the same innovation identity. This helper pulls out
 * those historically known pairs so the selection path can favor them when such pairs
 * exist.
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
 * Pool selection is opinionated but still simple: prefer pairs with known
 * innovation history, otherwise prefer hidden-to-hidden growth, otherwise fall
 * back to the full candidate set. That keeps the chapter's structural bias
 * readable in one place.
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
 * The deterministic single-pair fast path avoids wasting randomness when the
 * structural search has already collapsed to one legal option.
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
 * Once selection has picked a pair, the mutation path needs more than the raw
 * nodes. It also needs the symmetric key used for modern innovation reuse and
 * the directional legacy keys kept for backward-compatible lookups.
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
 * The add-connection path only enforces cycle checks when the genome requests
 * acyclic topology. That keeps recurrent-capable runs permissive while still
 * giving feed-forward-style runs one clear abort seam.
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
 * This helper is intentionally thin: by the time the flow reaches it, pair
 * discovery, policy filtering, and cycle guards should already be complete.
 * The remaining job is just to materialize the chosen edge.
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
 * Innovation assignment is the historical memory for connection growth. If the
 * unordered node pair has been seen before, this helper reuses that innovation
 * id. Otherwise it allocates a new global id and stores it under both the
 * symmetric key and the legacy directional aliases.
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
 * The symmetric key is the preferred reuse identity because connection growth
 * is treated as one structural relationship between two genes, not as a
 * direction-specific novelty record.
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
 * Legacy directional keys are still stored so older code paths or preserved
 * historical records can resolve to the same innovation id as the modern
 * symmetric key.
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
 * The cycle check walks forward from the proposed target node and looks for a
 * path back to the proposed source. If one exists, adding the new edge would
 * close a loop and the caller can abort the structural edit for acyclic runs.
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
