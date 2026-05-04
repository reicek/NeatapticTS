import {
  recordNodeSplitRecord,
  takeNextInnovationId,
} from '../../innovation-tracker/innovation-tracker';
import type { NodeSplitRecord } from '../../innovation-tracker/innovation-tracker.types';
import type {
  ConnectionWithMetadata,
  GenomeWithMetadata,
  NeatControllerForMutation,
  NodeWithMetadata,
} from '../shared/mutation.types';

/** Default weight for newly created connections. */
const DEFAULT_CONNECTION_WEIGHT = 1;
/** Default gene id placeholder when missing. */
const DEFAULT_GENE_ID = 0;
/** Default innovation id placeholder when missing. */
const DEFAULT_INNOVATION_ID = 0;
/** Prefix for canonical split identities keyed by connection innovation. */
const SPLIT_CONNECTION_INNOVATION_PREFIX = 'splitConnectionInnovation:';
/** Prefix for legacy split identities keyed by connection endpoints. */
const LEGACY_ENDPOINT_SPLIT_PREFIX = 'legacyEndpoints:';

/**
 * Add-node mutation helpers.
 *
 * This chapter owns the "split one connection into two" mechanics used by the
 * root mutation flow when preserving node-split innovation reuse.
 *
 * Add-node growth is the more identity-sensitive of the two structural mutation
 * paths. Adding a node is not just "insert one hidden unit." The controller is
 * trying to remember whether the exact same source-to-target split has already
 * happened elsewhere so that later crossover and speciation can recognize the
 * resulting structure as the same historical innovation rather than an
 * unrelated accident.
 *
 * The lifecycle in this chapter is therefore deliberate:
 *
 * 1. ensure there is at least one connection worth splitting,
 * 2. choose one enabled connection,
 * 3. derive a stable split-event key from the connection innovation when
 *    present, with a legacy endpoint fallback only when historical metadata
 *    is missing,
 * 4. either reuse an existing split record or assign a brand-new one,
 * 5. insert the new hidden node while preserving output ordering.
 *
 * Read this chapter from top to bottom when debugging structural growth by
 * connection splitting. The early helpers prepare a valid split target. The
 * middle helpers explain how one split becomes a reusable innovation record.
 * The final helpers explain where the new node and edges land in the genome.
 *
 * ```mermaid
 * flowchart TD
 *   Seed[Genome enters add-node path] --> Bootstrap{Connection available to split?}
 *   Bootstrap -->|no| Connect[Seed one input to output edge]
 *   Bootstrap -->|yes| Enabled[Collect enabled connections]
 *   Connect --> Enabled
 *   Enabled --> Choose[Choose one connection to split]
 *   Choose --> SplitKey[Build stable split descriptor]
 *   SplitKey --> Record{Existing split record?}
 *   Record -->|yes| Reuse[Reuse stored node and innovation ids]
 *   Record -->|no| NewRecord[Create node and assign fresh innovations]
 *   Reuse --> Insert[Insert hidden node and replacement edges]
 *   NewRecord --> Insert
 * ```
 */

/**
 * Ensure the genome has at least one connection by linking input to output.
 *
 * A connection split only makes sense when a genome already has an edge to cut.
 * This helper is the bootstrap escape hatch for extremely sparse genomes. It
 * seeds the smallest possible forward connection so the add-node path can keep
 * behaving like a split-based structural mutation instead of bailing out
 * immediately.
 *
 * @param genomeToSeed - genome that may need a bootstrap connection
 * @param internal - neat controller context retained for compatibility with existing callers
 * @returns void
 */
export function ensureBootstrapConnection(
  genomeToSeed: GenomeWithMetadata,
  internal: NeatControllerForMutation,
): void {
  void internal;

  // Step 1: return when connections already exist.
  if (genomeToSeed.connections.length > 0) return;

  // Step 2: find the first input and output nodes.
  const inputNode = findFirstNodeByType(genomeToSeed, 'input');
  const outputNode = findFirstNodeByType(genomeToSeed, 'output');
  if (!inputNode || !outputNode) return;

  // Step 3: attempt to connect input to output.
  try {
    genomeToSeed.connect?.(inputNode, outputNode, DEFAULT_CONNECTION_WEIGHT);
  } catch {
    // Intentionally ignore: connection may fail if nodes are incompatible.
  }
}

/**
 * Find the first node of a given type.
 *
 * The add-node bootstrap path only needs a minimal node lookup strategy, so
 * this helper stays intentionally simple and deterministic.
 *
 * @param genomeToSearch - genome whose nodes are searched
 * @param nodeType - node type to match
 * @returns the first matching node or undefined
 */
export function findFirstNodeByType(
  genomeToSearch: GenomeWithMetadata,
  nodeType: NodeWithMetadata['type'],
): NodeWithMetadata | undefined {
  // Step 1: scan nodes in order and return the first match.
  return genomeToSearch.nodes.find(
    (node: NodeWithMetadata) => node.type === nodeType,
  );
}

/**
 * Collect all enabled connections from a genome.
 *
 * Split mutations only operate on live structural edges. Disabled connections
 * remain historical artifacts and should not become split candidates because
 * doing so would grow new structure from topology the runtime is not currently
 * using.
 *
 * @param genomeToInspect - genome to inspect
 * @returns enabled connections list
 */
export function collectEnabledConnections(
  genomeToInspect: GenomeWithMetadata,
): ConnectionWithMetadata[] {
  // Step 1: filter connections that are not disabled.
  return genomeToInspect.connections.filter(
    (connection: ConnectionWithMetadata) => connection.enabled !== false,
  );
}

/**
 * Choose a random enabled connection to split.
 *
 * Once the candidate shelf is built, the split path keeps selection light: one
 * RNG draw chooses the connection whose history may now branch into a hidden
 * node insertion.
 *
 * @param enabledConnectionsList - candidate connections
 * @param internal - neat controller context
 * @returns selected connection or null
 */
export function chooseConnectionForSplit(
  enabledConnectionsList: ConnectionWithMetadata[],
  internal: NeatControllerForMutation,
): ConnectionWithMetadata | null {
  // Step 1: return null when no connections are available.
  if (!enabledConnectionsList.length) return null;

  // Step 2: sample a connection using the controller RNG.
  const randomValue = internal._getRNG()();
  const chosenIndex = Math.floor(randomValue * enabledConnectionsList.length);
  return enabledConnectionsList[chosenIndex];
}

/**
 * Build the split descriptor used for innovation lookup and connection creation.
 *
 * The descriptor is the compact identity packet for a split. Its key prefers
 * the historical identity of the split connection itself so homologous splits
 * follow the structural event rather than only the current endpoint pair.
 * When the connection lacks historical metadata, the descriptor falls back to a
 * legacy endpoint key so bootstrap and imported edge cases remain stable.
 *
 * @param connectionToSplit - connection being split
 * @returns split descriptor
 */
export function buildSplitDescriptor(
  connectionToSplit: ConnectionWithMetadata,
): {
  splitKey: string;
  originalWeight: number;
} {
  // Step 1: build a stable split-event key for innovation reuse.
  const splitKey = buildSplitKeyForConnection(connectionToSplit);

  // Step 2: capture original connection weight.
  return {
    splitKey,
    originalWeight: connectionToSplit.weight,
  };
}

/**
 * Build the canonical split identity for one connection.
 *
 * The proper-NEAT path keys split reuse by the historical marking of the edge
 * being split, which is the actual structural event. The endpoint fallback is
 * retained only so older or freshly bootstrapped connections without recorded
 * innovations still behave deterministically.
 *
 * @param connectionToSplit - connection whose split identity is being resolved
 * @returns canonical split key for tracker lookup
 */
function buildSplitKeyForConnection(
  connectionToSplit: ConnectionWithMetadata,
): string {
  // Step 1: prefer the split connection's historical innovation when present.
  if (Number.isInteger(connectionToSplit.innovation)) {
    return `${SPLIT_CONNECTION_INNOVATION_PREFIX}${connectionToSplit.innovation}`;
  }

  // Step 2: fall back to endpoint identity for legacy or bootstrap edges.
  const sourceGeneId = connectionToSplit.from.geneId ?? DEFAULT_GENE_ID;
  const targetGeneId = connectionToSplit.to.geneId ?? DEFAULT_GENE_ID;
  return `${LEGACY_ENDPOINT_SPLIT_PREFIX}${sourceGeneId}->${targetGeneId}`;
}

/**
 * Disconnect the original connection before inserting the split node.
 *
 * The add-node mutation is modeled as a real split, not as a parallel bypass.
 * Removing the original edge first preserves the intended NEAT-style topology
 * change: the signal must now pass through the new hidden node.
 *
 * @param genomeToEdit - genome to edit
 * @param connectionToRemove - original connection to remove
 * @returns void
 */
export function disconnectOriginalConnection(
  genomeToEdit: GenomeWithMetadata,
  connectionToRemove: ConnectionWithMetadata,
): void {
  // Step 1: remove the connection if possible.
  genomeToEdit.disconnect?.(connectionToRemove.from, connectionToRemove.to);
}

/**
 * Apply a split using an existing innovation record.
 *
 * This is the preferred path when the same structural split has already been
 * observed elsewhere in the population history. Reusing the stored node gene id
 * and edge innovation ids preserves historical identity, which makes later
 * alignment-based operations treat equivalent splits as equivalent structure.
 *
 * @param genomeToEdit - genome being modified
 * @param connectionToSplit - connection being split
 * @param splitDescriptor - metadata for the split
 * @param splitRecord - existing innovation record
 * @param NodeClass - node constructor
 * @returns void
 */
export function applySplitWithExistingRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number },
  splitRecord: NodeSplitRecord,
  NodeClass: new (
    type: NodeWithMetadata['type'],
    customActivation?: (x: number, derivate?: boolean) => number,
    rng?: () => number,
  ) => unknown,
  randomValue: () => number,
): void {
  // Step 1: create a new node instance with the historical gene id.
  const newNode = createSplitNode(NodeClass, randomValue);
  newNode.geneId = splitRecord.newNodeGeneId;

  // Step 2: insert the node before the original target node.
  const insertIndex = resolveInsertIndex(genomeToEdit, connectionToSplit.to);
  genomeToEdit.nodes.splice(insertIndex, 0, newNode);

  // Step 3: create split connections and apply historical innovations.
  const splitConnections = connectSplitEdges(
    genomeToEdit,
    connectionToSplit,
    newNode,
    splitDescriptor.originalWeight,
  );
  if (splitConnections.incomingConnection) {
    splitConnections.incomingConnection.innovation = splitRecord.inInnov;
  }
  if (splitConnections.outgoingConnection) {
    splitConnections.outgoingConnection.innovation = splitRecord.outInnov;
  }
}

/**
 * Apply a split and create a new innovation record.
 *
 * This path handles genuinely novel structural growth. It inserts a fresh
 * hidden node, assigns new innovations to the replacement edges, and records
 * the resulting identity under the split key so future genomes can reuse it.
 *
 * @param genomeToEdit - genome being modified
 * @param connectionToSplit - connection being split
 * @param splitDescriptor - metadata for the split
 * @param NodeClass - node constructor
 * @param internal - neat controller context
 * @returns void
 */
export function applySplitWithNewRecord(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  splitDescriptor: { splitKey: string; originalWeight: number },
  NodeClass: new (
    type: NodeWithMetadata['type'],
    customActivation?: (x: number, derivate?: boolean) => number,
    rng?: () => number,
  ) => unknown,
  internal: NeatControllerForMutation,
): void {
  // Step 1: create a new hidden node.
  const newNode = createSplitNode(NodeClass, internal._getRNG());

  // Step 2: insert the new node before the original target node so that the
  // acyclic connectivity check (which uses nodes.indexOf) can locate the new
  // node when connectSplitEdges is called.
  const insertIndex = resolveInsertIndex(genomeToEdit, connectionToSplit.to);
  genomeToEdit.nodes.splice(insertIndex, 0, newNode);

  // Step 3: connect the split edges and assign new innovations.
  const splitConnections = connectSplitEdges(
    genomeToEdit,
    connectionToSplit,
    newNode,
    splitDescriptor.originalWeight,
  );
  const splitRecord = assignInnovationsForNewSplit(
    newNode,
    splitConnections,
    internal,
  );
  recordNodeSplitRecord(
    internal._innovationTracker,
    splitDescriptor.splitKey,
    splitRecord,
  );
}

/**
 * Create a split-inserted hidden node using the controller RNG.
 *
 * The add-node replay contract depends on the inserted node receiving the same
 * bias initialization every time the same checkpointed mutation path resumes.
 * Passing the controller RNG through the node constructor keeps that
 * initialization deterministic instead of falling back to `Math.random()`.
 *
 * @param NodeClass - node constructor used by the mutation path.
 * @param randomValue - deterministic controller RNG.
 * @returns Newly created hidden node.
 */
function createSplitNode(
  NodeClass: new (
    type: NodeWithMetadata['type'],
    customActivation?: (x: number, derivate?: boolean) => number,
    rng?: () => number,
  ) => unknown,
  randomValue: () => number,
): NodeWithMetadata {
  return new NodeClass(
    'hidden',
    undefined,
    randomValue,
  ) as unknown as NodeWithMetadata;
}

/**
 * Resolve the insertion index for a new node, keeping outputs at the end.
 *
 * Node order matters in this codebase because output nodes are expected to stay
 * grouped at the tail of the genome node list. This helper preserves that local
 * invariant while still placing the new hidden node near the split target.
 *
 * @param genomeToEdit - genome whose node list is updated
 * @param targetNode - original target node of the split connection
 * @returns insertion index
 */
export function resolveInsertIndex(
  genomeToEdit: GenomeWithMetadata,
  targetNode: NodeWithMetadata,
): number {
  // Step 1: compute the target index.
  const targetIndex = genomeToEdit.nodes.indexOf(targetNode);

  // Step 2: ensure output nodes remain at the end of the list.
  return Math.min(targetIndex, genomeToEdit.nodes.length - genomeToEdit.output);
}

/**
 * Create the incoming and outgoing split connections.
 *
 * A split replaces one edge with two edges. The incoming edge starts with the
 * chapter's default bootstrap weight, while the outgoing edge preserves the
 * original connection weight so the pre-split signal can still pass forward in
 * a comparable way.
 *
 * @param genomeToEdit - genome being modified
 * @param connectionToSplit - connection being split
 * @param newNode - newly created hidden node
 * @param originalWeight - weight to preserve on the outgoing connection
 * @returns incoming/outgoing connection handles
 */
export function connectSplitEdges(
  genomeToEdit: GenomeWithMetadata,
  connectionToSplit: ConnectionWithMetadata,
  newNode: NodeWithMetadata,
  originalWeight: number,
): {
  incomingConnection?: ConnectionWithMetadata;
  outgoingConnection?: ConnectionWithMetadata;
} {
  // Step 1: connect source to the new node.
  const incomingConnection = genomeToEdit.connect?.(
    connectionToSplit.from,
    newNode,
    DEFAULT_CONNECTION_WEIGHT,
  )?.[0];

  // Step 2: connect new node to the original target with preserved weight.
  const outgoingConnection = genomeToEdit.connect?.(
    newNode,
    connectionToSplit.to,
    originalWeight,
  )?.[0];

  return { incomingConnection, outgoingConnection };
}

/**
 * Assign new innovations for a split and build the innovation record.
 *
 * New split records are the durable memory that turns a one-off structural edit
 * into reusable innovation history. This helper assigns the next global
 * innovation ids to the replacement edges and packages those ids together with
 * the new node gene id so later equivalent splits can be recognized quickly.
 *
 * @param newNode - newly created hidden node
 * @param splitConnections - incoming/outgoing connections
 * @param internal - neat controller context
 * @returns innovation record for the split
 */
export function assignInnovationsForNewSplit(
  newNode: NodeWithMetadata,
  splitConnections: {
    incomingConnection?: ConnectionWithMetadata;
    outgoingConnection?: ConnectionWithMetadata;
  },
  internal: NeatControllerForMutation,
): NodeSplitRecord {
  // Step 1: assign innovations to new connections.
  if (splitConnections.incomingConnection) {
    splitConnections.incomingConnection.innovation = takeNextInnovationId(
      internal._innovationTracker,
    );
  }
  if (splitConnections.outgoingConnection) {
    splitConnections.outgoingConnection.innovation = takeNextInnovationId(
      internal._innovationTracker,
    );
  }

  // Step 2: build and return the innovation record.
  return {
    newNodeGeneId: newNode.geneId ?? DEFAULT_GENE_ID,
    inInnov:
      splitConnections.incomingConnection?.innovation ?? DEFAULT_INNOVATION_ID,
    outInnov:
      splitConnections.outgoingConnection?.innovation ?? DEFAULT_INNOVATION_ID,
  };
}
