import type {
  GenomeWithMetadata,
  NodeWithMetadata,
  ConnectionWithMetadata,
  NeatControllerForMutation,
} from './neat.mutation.types';

/** Default weight for newly created connections. */
const DEFAULT_CONNECTION_WEIGHT = 1;
/** Default gene id placeholder when missing. */
const DEFAULT_GENE_ID = 0;
/** Default innovation id placeholder when missing. */
const DEFAULT_INNOVATION_ID = 0;

// ============================================================================
// Helpers for mutateAddNodeReuse() function
// ============================================================================

/**
 * Ensure the genome has at least one connection by linking input to output.
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

  // Step 1: return when any connections exist.
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
  return enabledConnectionsList[chosenIndex] ?? null;
}

/**
 * Build the split descriptor used for innovation lookup and connection creation.
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
  // Step 1: capture source/target gene ids for the key.
  const sourceGeneId = connectionToSplit.from.geneId;
  const targetGeneId = connectionToSplit.to.geneId;

  // Step 2: build a stable split key for innovation reuse.
  const splitKey = `${sourceGeneId}->${targetGeneId}`;

  // Step 3: capture original connection weight.
  return {
    splitKey,
    originalWeight: connectionToSplit.weight,
  };
}

/**
 * Disconnect the original connection before inserting the split node.
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
  splitRecord: { newNodeGeneId: number; inInnov: number; outInnov: number },
  NodeClass: new (type: NodeWithMetadata['type']) => unknown,
): void {
  // Step 1: create a new node instance with the historical gene id.
  const newNode = new NodeClass('hidden') as unknown as NodeWithMetadata;
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
  NodeClass: new (type: NodeWithMetadata['type']) => unknown,
  internal: NeatControllerForMutation,
): void {
  // Step 1: create a new hidden node.
  const newNode = new NodeClass('hidden') as unknown as NodeWithMetadata;

  // Step 2: connect the split edges and assign new innovations.
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
  internal._nodeSplitInnovations.set(splitDescriptor.splitKey, splitRecord);

  // Step 3: insert the new node before the original target node.
  const insertIndex = resolveInsertIndex(genomeToEdit, connectionToSplit.to);
  genomeToEdit.nodes.splice(insertIndex, 0, newNode);
}

/**
 * Resolve the insertion index for a new node, keeping outputs at the end.
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
): {
  newNodeGeneId: number;
  inInnov: number;
  outInnov: number;
} {
  // Step 1: assign innovations to new connections.
  if (splitConnections.incomingConnection) {
    splitConnections.incomingConnection.innovation =
      internal._nextGlobalInnovation++;
  }
  if (splitConnections.outgoingConnection) {
    splitConnections.outgoingConnection.innovation =
      internal._nextGlobalInnovation++;
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
