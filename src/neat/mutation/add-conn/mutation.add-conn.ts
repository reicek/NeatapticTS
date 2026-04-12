import {
  getConnectionInnovation,
  recordConnectionInnovation,
  takeNextInnovationId,
} from '../../innovation-tracker/innovation-tracker';
import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

/** Default gene id placeholder when missing. */
const DEFAULT_GENE_ID = 0;
/** Default lower bound for freshly added connection weights. */
const DEFAULT_NEW_CONNECTION_WEIGHT_MIN = -0.1;
/** Default upper bound for freshly added connection weights. */
const DEFAULT_NEW_CONNECTION_WEIGHT_MAX = 0.1;

/**
 * Add-connection mutation helpers.
 *
 * This chapter owns candidate-pair discovery, cycle guarding, and innovation-id
 * reuse for newly added structural connections.
 *
 * Where `add-node/` grows structure by splitting one existing edge, this
 * chapter grows structure by discovering a legal pair of nodes that are not yet
 * connected. That sounds simpler, but it still has several responsibilities:
 * preserve innovation identity when the same directed edge has been connected
 * before, prefer historically meaningful or structurally useful candidates,
 * and avoid illegal recurrent edges when acyclic topology is required.
 *
 * The flow is easiest to read as a pipeline:
 *
 * 1. enumerate legal candidate pairs,
 * 2. narrow the pool toward reusable or hidden-hidden pairs when possible,
 * 3. choose one pair,
 * 4. resolve the directional innovation key for that pair,
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
 *   Pair --> Keys[Resolve directional key]
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
 * active topology policy instead of assuming that add-connection is always a
 * forward-only operator. Feed-forward runs keep the traditional forward source
 * ordering, while unconstrained recurrent runs also include self and backward
 * candidates so the mutation shelf follows the same heredity contract as
 * crossover.
 *
 * Only truly absent directed edges enter this candidate pool. If a genome
 * already carries the exact edge in a disabled state, add-connection does not
 * duplicate it; revival belongs to an explicit re-enable path, while this
 * chapter only recreates historically known edges when the exact direction is
 * absent from the genome.
 *
 * @param genomeToInspect - genome to scan
 * @param allowRecurrentConnections - whether recurrent and self candidates may be proposed
 * @returns candidate node pairs
 */
export function collectCandidatePairsForConn(
  genomeToInspect: GenomeWithMetadata,
  allowRecurrentConnections: boolean = false,
): Array<[NodeWithMetadata, NodeWithMetadata]> {
  // Step 1: follow the explicit topology policy for candidate generation.
  if (allowRecurrentConnections) {
    return collectUnconstrainedCandidatePairs(genomeToInspect);
  }

  return collectFeedForwardCandidatePairs(genomeToInspect);
}

/**
 * Collect forward-only connection candidates for feed-forward runs.
 *
 * @param genomeToInspect - genome to scan
 * @returns candidate node pairs
 */
function collectFeedForwardCandidatePairs(
  genomeToInspect: GenomeWithMetadata,
): Array<[NodeWithMetadata, NodeWithMetadata]> {
  const pairs: Array<[NodeWithMetadata, NodeWithMetadata]> = [];

  // Step 1: build candidate pairs respecting forward node ordering.
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
      if (isAbsentDirectedEdge(sourceNode, targetNode)) {
        pairs.push([sourceNode, targetNode]);
      }
    }
  }

  return pairs;
}

/**
 * Collect forward, backward, and self candidates for unconstrained runs.
 *
 * Input nodes remain invalid targets, but any non-input node may receive a new
 * connection, including self loops or edges from later nodes.
 *
 * @param genomeToInspect - genome to scan
 * @returns candidate node pairs
 */
function collectUnconstrainedCandidatePairs(
  genomeToInspect: GenomeWithMetadata,
): Array<[NodeWithMetadata, NodeWithMetadata]> {
  const pairs: Array<[NodeWithMetadata, NodeWithMetadata]> = [];

  // Step 1: allow every non-input target under the unconstrained contract.
  for (
    let sourceIndex = 0;
    sourceIndex < genomeToInspect.nodes.length;
    sourceIndex++
  ) {
    const sourceNode = genomeToInspect.nodes[sourceIndex];
    for (
      let targetIndex = genomeToInspect.input;
      targetIndex < genomeToInspect.nodes.length;
      targetIndex++
    ) {
      const targetNode = genomeToInspect.nodes[targetIndex];
      if (isAbsentDirectedEdge(sourceNode, targetNode)) {
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
    const connectionKey = buildDirectionalKeyForConn(pair[0], pair[1]);
    return (
      getConnectionInnovation(internal._innovationTracker, connectionKey) !==
      undefined
    );
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
 * Check whether one exact pair is legal under the active connection policy.
 *
 * Repair helpers often know the exact endpoint pair they want to reconnect.
 * This helper lets them ask the same legality question as the generic
 * add-connection chapter instead of recreating a second pair-validation policy.
 *
 * @param genomeToInspect - genome to inspect
 * @param chosenPair - exact pair being considered
 * @param allowRecurrentConnections - whether recurrent and self candidates may be proposed
 * @returns true when the pair is legal and cycle-safe for creation
 */
export function canApplyChosenPairForConn(
  genomeToInspect: GenomeWithMetadata,
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
  allowRecurrentConnections: boolean = false,
): boolean {
  // Step 1: reject pairs that do not fit the active candidate policy.
  if (
    !isCandidatePairAllowedForConn(
      genomeToInspect,
      chosenPair[0],
      chosenPair[1],
      allowRecurrentConnections,
    )
  ) {
    return false;
  }

  // Step 2: reject pairs that would violate the cycle contract.
  return !shouldAbortForCycle(
    genomeToInspect,
    resolvePairNodes(chosenPair),
  );
}

/**
 * Create one exact connection using canonical innovation reuse.
 *
 * The generic add-connection operator chooses the pair for itself, but repair
 * helpers often need to reconnect a known source and target. This service keeps
 * those maintenance paths on the same innovation and topology-policy contract
 * as the main mutation operator instead of letting them materialize edges with
 * direct runtime connects.
 *
 * @param genomeToEdit - genome to edit
 * @param chosenPair - exact source-target pair to connect
 * @param internal - neat controller context
 * @param allowRecurrentConnections - whether recurrent and self candidates may be proposed
 * @returns created connection or undefined when the pair is not legal
 */
export function connectChosenPairWithInnovationReuse(
  genomeToEdit: GenomeWithMetadata,
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
  internal: NeatControllerForMutation,
  allowRecurrentConnections: boolean = false,
): ConnectionWithMetadata | undefined {
  // Step 1: reject illegal or cycle-forming pairs up front.
  if (
    !canApplyChosenPairForConn(
      genomeToEdit,
      chosenPair,
      allowRecurrentConnections,
    )
  ) {
    return undefined;
  }

  // Step 2: connect the exact pair.
  const pairNodes = resolvePairNodes(chosenPair);
  const connection = connectChosenPair(genomeToEdit, pairNodes, internal);
  if (!connection) {
    return undefined;
  }

  // Step 3: assign canonical innovation reuse metadata.
  assignInnovationForConnection(connection, pairNodes, internal);
  return connection;
}

/**
 * Resolve nodes and innovation key details for a chosen pair.
 *
 * Once selection has picked a pair, the mutation path needs more than the raw
 * nodes. It also needs the exact directional key used for generation-scoped
 * innovation reuse.
 *
 * @param chosenPair - pair to connect
 * @returns resolved pair metadata
 */
export function resolvePairNodes(
  chosenPair: [NodeWithMetadata, NodeWithMetadata],
): {
  sourceNode: NodeWithMetadata;
  targetNode: NodeWithMetadata;
  connectionKey: string;
} {
  // Step 1: pull source and target nodes.
  const sourceNode = chosenPair[0];
  const targetNode = chosenPair[1];

  // Step 2: compute innovation keys.
  const connectionKey = buildDirectionalKeyForConn(sourceNode, targetNode);
  return {
    sourceNode,
    targetNode,
    connectionKey,
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
  internal: NeatControllerForMutation,
): ConnectionWithMetadata | undefined {
  // Step 1: derive a deterministic initial weight for the new edge.
  const connectionWeight =
    internal._getRNG()() *
      (DEFAULT_NEW_CONNECTION_WEIGHT_MAX -
        DEFAULT_NEW_CONNECTION_WEIGHT_MIN) +
    DEFAULT_NEW_CONNECTION_WEIGHT_MIN;

  // Step 2: attempt to connect the chosen nodes.
  return genomeToEdit.connect?.(
    pairNodes.sourceNode,
    pairNodes.targetNode,
    connectionWeight,
  )?.[0];
}

/**
 * Assign an innovation id for a new connection, reusing when possible.
 *
 * Innovation assignment is the historical memory for connection growth. If the
 * exact directed edge has already been recorded for the active generation,
 * this helper reuses that innovation id. Otherwise it allocates a new global
 * id and stores it under the pair's directional key.
 *
 * This keeps the revive-vs-recreate contract explicit: currently absent edges
 * may be recreated with their historical innovation, but currently present
 * disabled genes are not duplicated here because they never reach the absent
 * candidate pool.
 *
 * @param connection - newly created connection
 * @param pairNodes - resolved pair metadata
 * @param internal - neat controller context
 * @returns void
 */
export function assignInnovationForConnection(
  connection: ConnectionWithMetadata,
  pairNodes: {
    connectionKey: string;
  },
  internal: NeatControllerForMutation,
): void {
  // Step 1: reuse existing innovation when present.
  const knownInnovationId = getConnectionInnovation(
    internal._innovationTracker,
    pairNodes.connectionKey,
  );
  if (knownInnovationId !== undefined) {
    connection.innovation = knownInnovationId;
    return;
  }

  // Step 2: allocate and store a new innovation id.
  const newInnovationId = takeNextInnovationId(internal._innovationTracker);
  connection.innovation = newInnovationId;
  recordConnectionInnovation(
    internal._innovationTracker,
    pairNodes.connectionKey,
    newInnovationId,
  );
}

/**
 * Build a directional innovation key for an exact source-target edge.
 *
 * Direction matters once recurrent growth is allowed. Forward, backward, and
 * self edges between the same endpoint gene ids are distinct structural events
 * and must not alias through one recurrence-blind key.
 *
 * @param sourceNode - source node
 * @param targetNode - target node
 * @returns directional innovation key
 */
export function buildDirectionalKeyForConn(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): string {
  // Step 1: resolve stable endpoint identities.
  const sourceGeneId = sourceNode.geneId ?? DEFAULT_GENE_ID;
  const targetGeneId = targetNode.geneId ?? DEFAULT_GENE_ID;
  return `${sourceGeneId}->${targetGeneId}`;
}

/**
 * Determine whether the directed edge is absent from the genome.
 *
 * Disabled connections still count as structurally present genes. This helper
 * makes the mutation contract explicit: add-connection recreates historically
 * known edges only when the exact direction is absent, while dormant genes stay
 * reserved for explicit re-enable flows.
 *
 * @param sourceNode - source node of the candidate edge
 * @param targetNode - target node of the candidate edge
 * @returns true when the directed edge is absent from the genome
 */
function isAbsentDirectedEdge(
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
): boolean {
  // Step 1: treat enabled and disabled edges alike because both occupy the pair.
  return !sourceNode.isProjectingTo?.(targetNode);
}

/**
 * Check whether one exact pair is eligible for connection creation.
 *
 * This helper keeps the pair-shape rules for feed-forward and unconstrained
 * runs in one place so direct repair paths can stay aligned with the generic
 * candidate-generation chapter.
 *
 * @param genomeToInspect - genome to inspect
 * @param sourceNode - proposed source node
 * @param targetNode - proposed target node
 * @param allowRecurrentConnections - whether recurrent and self candidates may be proposed
 * @returns true when the pair fits the active candidate policy
 */
function isCandidatePairAllowedForConn(
  genomeToInspect: GenomeWithMetadata,
  sourceNode: NodeWithMetadata,
  targetNode: NodeWithMetadata,
  allowRecurrentConnections: boolean,
): boolean {
  // Step 1: reject nodes that do not belong to this genome or already connect.
  const sourceIndex = genomeToInspect.nodes.indexOf(sourceNode);
  const targetIndex = genomeToInspect.nodes.indexOf(targetNode);
  if (
    sourceIndex === -1 ||
    targetIndex === -1 ||
    !isAbsentDirectedEdge(sourceNode, targetNode)
  ) {
    return false;
  }

  // Step 2: allow unconstrained runs to target any non-input node.
  if (allowRecurrentConnections) {
    return targetIndex >= genomeToInspect.input;
  }

  // Step 3: require feed-forward ordering and keep outputs out of the source shelf.
  if (sourceIndex >= genomeToInspect.nodes.length - genomeToInspect.output) {
    return false;
  }

  return targetIndex >= Math.max(sourceIndex + 1, genomeToInspect.input);
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
