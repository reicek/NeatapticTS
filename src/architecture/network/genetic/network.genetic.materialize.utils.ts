import Node from '../../node';
import Connection from '../../connection';
import type {
  ConnectionGene,
  ConnectionGeneticProps,
  GeneEndpointsContext,
  GeneTraversalContext,
  GeneticNetwork,
  OffspringMaterializationContext,
} from '../network.types';
import { FIRST_INDEX, NO_GATER_INDEX } from './network.genetic.utils.types';

const NEUTRAL_NODE_CONSTRUCTOR_RANDOM = () => 0.5;

interface MaterializationNodeSource {
  node: Node;
  preferredOrder: number;
  sourcePriority: number;
  interfaceOrdinal?: number;
}

interface MaterializedNodeCandidate {
  node: Node;
  geneId: number;
  typeRank: number;
  preferredOrder: number;
  sourcePriority: number;
}

/**
 * Materializes selected connection genes in the offspring network.
 *
 * The offspring node array is reindexed after node selection, so this runtime
 * seam intentionally avoids parent-time node indexes altogether. This
 * materialization pass first rebuilds the offspring node set from required IO
 * nodes plus any hidden nodes referenced by the chosen genes, then resolves
 * endpoints and gaters through a stable `geneId` lookup map. Feed-forward
 * pruning is now derived from the offspring topology contract instead of any
 * incidental parent ordering hint.
 *
 * @param offspring Offspring network.
 * @param chosenGenes Chosen connection materialization descriptors.
 * @param sourceNetworks Parent networks used as fallback node-gene sources.
 * @returns Nothing.
 */
export function materializeOffspringConnections(
  offspring: GeneticNetwork,
  chosenGenes: ConnectionGene[],
  sourceNetworks: readonly GeneticNetwork[] = [],
): void {
  const materializationContext = createMaterializationContext(
    offspring,
    chosenGenes,
    sourceNetworks,
  );
  const traversalContexts = createTraversalContexts(
    materializationContext,
    chosenGenes,
  );
  materializeTraversalContexts(traversalContexts);
}

/**
 * Creates the immutable top-level context used during materialization.
 *
 * @param targetOffspring Offspring receiving concrete edges.
 * @param chosenGenes Chosen inherited genes.
 * @param sourceNetworks Parent networks used as fallback node-gene sources.
 * @returns Materialization context.
 */
function createMaterializationContext(
  targetOffspring: GeneticNetwork,
  chosenGenes: ConnectionGene[],
  sourceNetworks: readonly GeneticNetwork[],
): OffspringMaterializationContext {
  const sourceNodeEntriesByGeneId = createSourceNodeLookupByGeneId(
    targetOffspring.nodes,
    sourceNetworks,
  );
  targetOffspring.nodes = buildMaterializedOffspringNodes(
    targetOffspring.nodes,
    chosenGenes,
    sourceNodeEntriesByGeneId,
  );
  assignOffspringNodeIndexes(targetOffspring.nodes);
  targetOffspring.refreshExplicitIORoles();

  return {
    offspring: targetOffspring,
    topologyIntent: targetOffspring.getTopologyIntent(),
    offspringNodesByGeneId: createOffspringNodeLookupByGeneId(
      targetOffspring.nodes,
    ),
    sourceNodesByGeneId: createSourceNodeMap(sourceNodeEntriesByGeneId),
    sourceNodeInterfaceOrdinalsByGeneId: createSourceInterfaceOrdinalMap(
      sourceNodeEntriesByGeneId,
    ),
  };
}

/**
 * Builds the fallback node-gene lookup used when chosen genes reference nodes
 * not present in the provisional offspring scaffold.
 *
 * @param currentOffspringNodes Provisional offspring node genes.
 * @param sourceNetworks Parent networks that can contribute missing nodes.
 * @returns Lookup of source node genes by stable gene id.
 */
function createSourceNodeLookupByGeneId(
  currentOffspringNodes: Node[],
  sourceNetworks: readonly GeneticNetwork[],
): Map<number, MaterializationNodeSource> {
  const sourceNodesByGeneId = new Map<number, MaterializationNodeSource>();

  registerNodeSources(sourceNodesByGeneId, currentOffspringNodes, 0);
  for (
    let sourceNetworkIndex = 0;
    sourceNetworkIndex < sourceNetworks.length;
    sourceNetworkIndex++
  ) {
    registerNodeSources(
      sourceNodesByGeneId,
      sourceNetworks[sourceNetworkIndex].nodes,
      sourceNetworkIndex + 1,
    );
  }

  return sourceNodesByGeneId;
}

/**
 * Registers one ordered node list as a fallback source for gene-id lookup.
 *
 * @param sourceNodesByGeneId Mutable source-node lookup.
 * @param nodes Ordered nodes from one source network.
 * @param sourcePriority Stable priority for this source list.
 * @returns Nothing.
 */
function registerNodeSources(
  sourceNodesByGeneId: Map<number, MaterializationNodeSource>,
  nodes: Node[],
  sourcePriority: number,
): void {
  let inputOrdinal = 0;
  let outputOrdinal = 0;

  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const node = nodes[nodeIndex];
    if (
      typeof node.geneId !== 'number' ||
      sourceNodesByGeneId.has(node.geneId)
    ) {
      continue;
    }

    sourceNodesByGeneId.set(node.geneId, {
      node,
      preferredOrder: nodeIndex,
      sourcePriority,
      interfaceOrdinal:
        node.type === 'input'
          ? inputOrdinal++
          : node.type === 'output'
            ? outputOrdinal++
            : undefined,
    });
  }
}

/**
 * Flattens source-node entries into a plain node lookup keyed by gene id.
 *
 * @param sourceNodeEntriesByGeneId Source-node entries keyed by gene id.
 * @returns Plain source node lookup.
 */
function createSourceNodeMap(
  sourceNodeEntriesByGeneId: Map<number, MaterializationNodeSource>,
): Map<number, Node> {
  return new Map(
    Array.from(sourceNodeEntriesByGeneId.entries()).map(
      ([geneId, sourceNodeEntry]) => [geneId, sourceNodeEntry.node],
    ),
  );
}

/**
 * Flattens source interface ordinals into a plain lookup keyed by gene id.
 *
 * @param sourceNodeEntriesByGeneId Source-node entries keyed by gene id.
 * @returns Plain source interface-ordinal lookup.
 */
function createSourceInterfaceOrdinalMap(
  sourceNodeEntriesByGeneId: Map<number, MaterializationNodeSource>,
): Map<number, number> {
  return new Map(
    Array.from(sourceNodeEntriesByGeneId.entries())
      .filter(
        ([, sourceNodeEntry]) =>
          typeof sourceNodeEntry.interfaceOrdinal === 'number',
      )
      .map(([geneId, sourceNodeEntry]) => [
        geneId,
        sourceNodeEntry.interfaceOrdinal as number,
      ]),
  );
}

/**
 * Rebuilds the offspring node set from required IO nodes plus inherited genes.
 *
 * Step 2.3 moves node survival away from the provisional slot count chosen
 * during setup. If an inherited connection references a hidden node that the
 * provisional scaffold omitted, this pass rehydrates that node by `geneId`
 * before any connection materialization begins.
 *
 * @param currentOffspringNodes Provisional offspring nodes from setup.
 * @param chosenGenes Chosen inherited connection genes.
 * @param sourceNodesByGeneId Fallback node-gene lookup.
 * @returns Rebuilt offspring node set.
 */
function buildMaterializedOffspringNodes(
  currentOffspringNodes: Node[],
  chosenGenes: ConnectionGene[],
  sourceNodesByGeneId: Map<number, MaterializationNodeSource>,
): Node[] {
  const requiredGeneIds = collectRequiredNodeGeneIds(
    currentOffspringNodes,
    chosenGenes,
    sourceNodesByGeneId,
  );
  const currentNodeOrderByGeneId = createNodeOrderLookup(currentOffspringNodes);

  return Array.from(requiredGeneIds)
    .map((geneId) =>
      createMaterializedNodeCandidate(
        geneId,
        currentOffspringNodes,
        currentNodeOrderByGeneId,
        sourceNodesByGeneId,
      ),
    )
    .filter(
      (nodeCandidate): nodeCandidate is MaterializedNodeCandidate =>
        nodeCandidate !== undefined,
    )
    .toSorted(compareMaterializedNodeCandidates)
    .map((nodeCandidate) => nodeCandidate.node);
}

/**
 * Collects the node gene ids that must survive materialization.
 *
 * The offspring always preserves its IO interface, then adds any nodes named by
 * inherited connection endpoints or gaters.
 *
 * @param currentOffspringNodes Provisional offspring nodes from setup.
 * @param chosenGenes Chosen inherited connection genes.
 * @param sourceNodesByGeneId Fallback node-gene lookup.
 * @returns Required node gene ids.
 */
function collectRequiredNodeGeneIds(
  currentOffspringNodes: Node[],
  chosenGenes: ConnectionGene[],
  sourceNodesByGeneId: Map<number, MaterializationNodeSource>,
): Set<number> {
  const requiredGeneIds = new Set<number>();
  const currentNodeOrderByGeneId = createNodeOrderLookup(currentOffspringNodes);

  for (
    let nodeIndex = 0;
    nodeIndex < currentOffspringNodes.length;
    nodeIndex++
  ) {
    const node = currentOffspringNodes[nodeIndex];
    if (typeof node.geneId === 'number' && isRequiredInterfaceNode(node)) {
      requiredGeneIds.add(node.geneId);
    }
  }

  for (let geneIndex = 0; geneIndex < chosenGenes.length; geneIndex++) {
    const chosenGene = chosenGenes[geneIndex];
    addInheritedGeneIdWhenRequired(
      requiredGeneIds,
      currentNodeOrderByGeneId,
      sourceNodesByGeneId,
      chosenGene.fromGeneId,
    );
    addInheritedGeneIdWhenRequired(
      requiredGeneIds,
      currentNodeOrderByGeneId,
      sourceNodesByGeneId,
      chosenGene.toGeneId,
    );
    addInheritedGeneIdWhenRequired(
      requiredGeneIds,
      currentNodeOrderByGeneId,
      sourceNodesByGeneId,
      chosenGene.gaterGeneId,
    );
  }

  return requiredGeneIds;
}

/**
 * Creates a node-order lookup keyed by stable gene id.
 *
 * @param nodes Ordered node list.
 * @returns Current node order by gene id.
 */
function createNodeOrderLookup(nodes: Node[]): Map<number, number> {
  const nodeOrderByGeneId = new Map<number, number>();

  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const node = nodes[nodeIndex];
    if (typeof node.geneId === 'number') {
      nodeOrderByGeneId.set(node.geneId, nodeIndex);
    }
  }

  return nodeOrderByGeneId;
}

/**
 * Creates one ordered node candidate for the rebuilt offspring node set.
 *
 * Existing provisional offspring nodes keep their selected structural traits.
 * Missing inherited nodes are cloned from the first parent/source that still
 * exposes the required `geneId`.
 *
 * @param geneId Required stable node gene id.
 * @param currentOffspringNodes Provisional offspring nodes from setup.
 * @param currentNodeOrderByGeneId Current offspring node order lookup.
 * @param sourceNodesByGeneId Fallback node-gene lookup.
 * @returns Ordered node candidate when the gene id can be resolved.
 */
function createMaterializedNodeCandidate(
  geneId: number,
  currentOffspringNodes: Node[],
  currentNodeOrderByGeneId: Map<number, number>,
  sourceNodesByGeneId: Map<number, MaterializationNodeSource>,
): MaterializedNodeCandidate | undefined {
  const existingNode = currentOffspringNodes.find(
    (node) => node.geneId === geneId,
  );
  if (existingNode) {
    return {
      node: existingNode,
      geneId,
      typeRank: resolveNodeTypeRank(existingNode),
      preferredOrder: currentNodeOrderByGeneId.get(geneId)!,
      sourcePriority: 0,
    };
  }

  const sourceNode = sourceNodesByGeneId.get(geneId)!;

  const clonedSourceNode = cloneSourceNodeGene(sourceNode.node);
  return {
    node: clonedSourceNode,
    geneId,
    typeRank: resolveNodeTypeRank(clonedSourceNode),
    preferredOrder: sourceNode.preferredOrder,
    sourcePriority: sourceNode.sourcePriority,
  };
}

/**
 * Resolves whether one node is part of the required public IO contract.
 *
 * @param node Candidate node.
 * @returns True when the node is input or output.
 */
function isRequiredInterfaceNode(node: Node): boolean {
  return node.type === 'input' || node.type === 'output';
}

/**
 * Adds one gene id to the required-node set when it is defined.
 *
 * @param requiredGeneIds Mutable required-node set.
 * @param currentNodeOrderByGeneId Current offspring node order lookup.
 * @param sourceNodesByGeneId Fallback node-gene lookup.
 * @param geneId Candidate stable node gene id.
 * @returns Nothing.
 */
function addInheritedGeneIdWhenRequired(
  requiredGeneIds: Set<number>,
  currentNodeOrderByGeneId: Map<number, number>,
  sourceNodesByGeneId: Map<number, MaterializationNodeSource>,
  geneId: number | null,
): void {
  if (typeof geneId !== 'number') {
    return;
  }

  const sourceNode = sourceNodesByGeneId.get(geneId)?.node;
  if (!sourceNode) {
    return;
  }

  if (
    isRequiredInterfaceNode(sourceNode) &&
    !currentNodeOrderByGeneId.has(geneId)
  ) {
    return;
  }

  requiredGeneIds.add(geneId);
}

/**
 * Compares rebuilt node candidates using IO/hidden ordering first.
 *
 * @param leftNodeCandidate Left node candidate.
 * @param rightNodeCandidate Right node candidate.
 * @returns Relative sort order.
 */
function compareMaterializedNodeCandidates(
  leftNodeCandidate: MaterializedNodeCandidate,
  rightNodeCandidate: MaterializedNodeCandidate,
): number {
  const typeRankDelta =
    leftNodeCandidate.typeRank - rightNodeCandidate.typeRank;
  const preferredOrderDelta =
    leftNodeCandidate.preferredOrder - rightNodeCandidate.preferredOrder;
  const sourcePriorityDelta =
    leftNodeCandidate.sourcePriority - rightNodeCandidate.sourcePriority;

  return typeRankDelta || preferredOrderDelta || sourcePriorityDelta;
}

/**
 * Resolves a stable materialization rank for one node type.
 *
 * @param node Candidate node.
 * @returns Type rank used for rebuilt offspring ordering.
 */
function resolveNodeTypeRank(node: Node): number {
  if (node.type === 'input') {
    return 0;
  }

  if (node.type === 'output') {
    return 2;
  }

  return 1;
}

/**
 * Clones one fallback source node for materialization-time insertion.
 *
 * @param sourceNode Source node gene.
 * @returns Structural clone for the rebuilt offspring node set.
 */
function cloneSourceNodeGene(sourceNode: Node): Node {
  const clonedNode = new Node(
    sourceNode.type,
    undefined,
    NEUTRAL_NODE_CONSTRUCTOR_RANDOM,
  );
  clonedNode.geneId = sourceNode.geneId;
  clonedNode.bias = sourceNode.bias;
  clonedNode.squash = sourceNode.squash;
  return clonedNode;
}

/**
 * Assigns contiguous node indices after rebuilt node ordering is finalized.
 *
 * @param nodes Rebuilt offspring nodes.
 * @returns Nothing.
 */
function assignOffspringNodeIndexes(nodes: Node[]): void {
  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    nodes[nodeIndex].index = nodeIndex;
  }
}

/**
 * Builds a stable node lookup keyed by gene id for one offspring scaffold.
 *
 * Connection genes preserve historical endpoint identity even when the runtime
 * node array is rebuilt in a different slot order. This map lets the
 * materialization pass reattach inherited genes to the right runtime nodes.
 *
 * @param nodes Offspring nodes after node selection and reindexing.
 * @returns Gene-id lookup map.
 */
function createOffspringNodeLookupByGeneId(nodes: Node[]): Map<number, Node> {
  const nodesByGeneId = new Map<number, Node>();

  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const node = nodes[nodeIndex];
    nodesByGeneId.set(node.geneId!, node);
  }

  return nodesByGeneId;
}

/**
 * Builds traversal contexts for each candidate gene.
 *
 * @param context Top-level materialization context.
 * @param genes Candidate genes.
 * @returns Traversal contexts.
 */
function createTraversalContexts(
  context: OffspringMaterializationContext,
  genes: ConnectionGene[],
): GeneTraversalContext[] {
  return genes.map((connectionGene) => ({
    materializationContext: context,
    connectionGene,
  }));
}

/**
 * Materializes each eligible traversal context independently.
 *
 * @param traversalContexts Eligible traversal contexts.
 * @returns Nothing.
 */
function materializeTraversalContexts(
  traversalContexts: GeneTraversalContext[],
): void {
  for (
    let traversalIndex = 0;
    traversalIndex < traversalContexts.length;
    traversalIndex++
  ) {
    materializeSingleTraversalContext(traversalContexts[traversalIndex]);
  }
}

/**
 * Materializes one eligible traversal context when no duplicate projection exists.
 *
 * @param traversalContext Traversal context.
 * @returns Nothing.
 */
function materializeSingleTraversalContext(
  traversalContext: GeneTraversalContext,
): void {
  const endpointsContext = resolveGeneEndpointsContext(traversalContext);
  if (
    !endpointsContext ||
    !isEndpointsContextAllowedByTopologyPolicy(endpointsContext) ||
    hasExistingProjection(endpointsContext)
  ) {
    return;
  }

  const createdConnection = createConnectionForEndpoints(endpointsContext);
  if (!createdConnection) {
    return;
  }

  applyConnectionGeneToConnection(
    createdConnection,
    traversalContext.connectionGene,
  );
  attachGaterIfAvailable(
    traversalContext.materializationContext,
    createdConnection,
    traversalContext.connectionGene.gaterGeneId,
  );
}

/**
 * Resolves concrete endpoint nodes for a traversal context.
 *
 * @param traversalContext Traversal context.
 * @returns Endpoint context or undefined.
 */
function resolveGeneEndpointsContext(
  traversalContext: GeneTraversalContext,
): GeneEndpointsContext | undefined {
  const fromNode = resolveNodeByGeneId(
    traversalContext.materializationContext,
    traversalContext.connectionGene.fromGeneId,
  );
  const toNode = resolveNodeByGeneId(
    traversalContext.materializationContext,
    traversalContext.connectionGene.toGeneId,
  );

  if (!fromNode || !toNode) {
    return undefined;
  }

  return {
    traversalContext,
    fromNode,
    toNode,
  };
}

/**
 * Resolves one offspring node by its stable gene id.
 *
 * @param context Materialization context holding the lookup map.
 * @param geneId Stable node gene id referenced by a connection gene.
 * @returns Matching offspring node, when present.
 */
function resolveNodeByGeneId(
  context: OffspringMaterializationContext,
  geneId: number,
): Node | undefined {
  const resolvedNode = context.offspringNodesByGeneId.get(geneId);
  if (resolvedNode) {
    return resolvedNode;
  }

  const sourceNode = context.sourceNodesByGeneId.get(geneId);
  if (!sourceNode || !isRequiredInterfaceNode(sourceNode)) {
    return undefined;
  }

  return resolveInterfaceNodeByOrdinal(
    context,
    sourceNode.type,
    context.sourceNodeInterfaceOrdinalsByGeneId.get(geneId)!,
  );
}

/**
 * Resolves one offspring IO node by its stable interface ordinal.
 *
 * @param context Materialization context.
 * @param nodeType Interface node type to resolve.
 * @param interfaceOrdinal Ordinal within the source interface partition.
 * @returns Matching offspring IO node, when present.
 */
function resolveInterfaceNodeByOrdinal(
  context: OffspringMaterializationContext,
  nodeType: string,
  interfaceOrdinal: number,
): Node | undefined {
  return context.offspring.nodes
    .filter((node) => node.type === nodeType)
    .at(interfaceOrdinal);
}

/**
 * Creates a runtime connection for endpoint nodes.
 *
 * @param endpointsContext Endpoint context.
 * @returns Created connection or undefined.
 */
function createConnectionForEndpoints(
  endpointsContext: GeneEndpointsContext,
): Connection | undefined {
  return createOffspringConnection(
    endpointsContext.traversalContext.materializationContext.offspring,
    endpointsContext.fromNode,
    endpointsContext.toNode,
    endpointsContext.traversalContext.connectionGene.weight,
  );
}

/**
 * Checks whether the source endpoint already projects to the target endpoint.
 *
 * @param endpointsContext Endpoint context.
 * @returns True when projection already exists.
 */
function hasExistingProjection(
  endpointsContext: GeneEndpointsContext,
): boolean {
  return endpointsContext.fromNode.isProjectingTo(endpointsContext.toNode);
}

/**
 * Validates one resolved gene against the offspring topology policy.
 *
 * Feed-forward offspring prune self and backward genes explicitly. Unconstrained
 * offspring keep those genes and let `connect()` register them in the runtime
 * self/recurrent collections.
 *
 * @param endpointsContext Resolved endpoints for one inherited gene.
 * @returns True when the gene is legal for the offspring topology intent.
 */
function isEndpointsContextAllowedByTopologyPolicy(
  endpointsContext: GeneEndpointsContext,
): boolean {
  if (
    endpointsContext.traversalContext.materializationContext.topologyIntent !==
    'feed-forward'
  ) {
    return true;
  }

  const sourceNodeIndex = endpointsContext.fromNode.index!;
  const targetNodeIndex = endpointsContext.toNode.index!;
  return sourceNodeIndex < targetNodeIndex;
}

/**
 * Creates a single offspring connection edge.
 *
 * @param offspring Offspring network.
 * @param fromNode Source node.
 * @param toNode Destination node.
 * @param connectionWeight Connection weight.
 * @returns Created connection or undefined.
 */
function createOffspringConnection(
  offspring: GeneticNetwork,
  fromNode: Node,
  toNode: Node,
  connectionWeight: number,
): Connection | undefined {
  const createdConnections = offspring.connect(
    fromNode,
    toNode,
    connectionWeight,
  );
  return createdConnections.at(FIRST_INDEX);
}

/**
 * Applies gene properties to a runtime connection.
 *
 * @param connection Runtime connection.
 * @param connectionGene Gene source.
 * @returns Nothing.
 */
function applyConnectionGeneToConnection(
  connection: Connection,
  connectionGene: ConnectionGene,
): void {
  connection.weight = connectionGene.weight;
  connection.innovation = connectionGene.innovation;
  (connection as Connection & ConnectionGeneticProps).enabled =
    connectionGene.enabled !== false;
}

/**
 * Attaches a gater node when the inherited gater gene exists in the offspring.
 *
 * @param context Materialization context.
 * @param connection Connection to gate.
 * @param gaterGeneId Candidate gater node gene id.
 * @returns Nothing.
 */
function attachGaterIfAvailable(
  context: OffspringMaterializationContext,
  connection: Connection,
  gaterGeneId: number | null,
): void {
  if (gaterGeneId == null || gaterGeneId === NO_GATER_INDEX) {
    return;
  }

  const gaterNode = resolveNodeByGeneId(context, gaterGeneId);
  if (!gaterNode) {
    return;
  }

  context.offspring.gate(gaterNode, connection);
}
