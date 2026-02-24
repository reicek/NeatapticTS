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

/**
 * Materializes selected connection genes in the offspring network.
 *
 * @param offspring - Offspring network.
 * @param chosenGenes - Chosen connection genes.
 * @returns Nothing.
 */
export function materializeOffspringConnections(
  offspring: GeneticNetwork,
  chosenGenes: ConnectionGene[],
): void {
  const materializationContext = createMaterializationContext(offspring);
  const eligibleTraversalContexts = collectEligibleTraversalContexts(
    materializationContext,
    chosenGenes,
  );
  materializeTraversalContexts(eligibleTraversalContexts);
}

/**
 * Creates the immutable top-level context used during materialization.
 *
 * @param targetOffspring - Offspring receiving concrete edges.
 * @returns Materialization context.
 */
function createMaterializationContext(
  targetOffspring: GeneticNetwork,
): OffspringMaterializationContext {
  return {
    offspring: targetOffspring,
    offspringNodeCount: targetOffspring.nodes.length,
  };
}

/**
 * Collects traversal contexts that satisfy all structural eligibility checks.
 *
 * @param context - Top-level materialization context.
 * @param genes - Candidate genes.
 * @returns Eligible traversal contexts.
 */
function collectEligibleTraversalContexts(
  context: OffspringMaterializationContext,
  genes: ConnectionGene[],
): GeneTraversalContext[] {
  const allTraversalContexts = createTraversalContexts(context, genes);
  const boundedTraversalContexts =
    keepTraversalContextsWithinNodeBounds(allTraversalContexts);
  return keepFeedForwardTraversalContexts(boundedTraversalContexts);
}

/**
 * Builds traversal contexts for each candidate gene.
 *
 * @param context - Top-level materialization context.
 * @param genes - Candidate genes.
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
 * Keeps traversal contexts whose endpoints are inside offspring bounds.
 *
 * @param traversalContexts - Candidate traversal contexts.
 * @returns Node-bounded contexts.
 */
function keepTraversalContextsWithinNodeBounds(
  traversalContexts: GeneTraversalContext[],
): GeneTraversalContext[] {
  return traversalContexts.filter(isTraversalContextWithinNodeBounds);
}

/**
 * Keeps traversal contexts that preserve feed-forward edge direction.
 *
 * @param traversalContexts - Node-bounded traversal contexts.
 * @returns Feed-forward contexts.
 */
function keepFeedForwardTraversalContexts(
  traversalContexts: GeneTraversalContext[],
): GeneTraversalContext[] {
  return traversalContexts.filter(isTraversalContextFeedForward);
}

/**
 * Materializes each eligible traversal context independently.
 *
 * @param traversalContexts - Eligible traversal contexts.
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
 * @param traversalContext - Traversal context.
 * @returns Nothing.
 */
function materializeSingleTraversalContext(
  traversalContext: GeneTraversalContext,
): void {
  const endpointsContext = resolveGeneEndpointsContext(traversalContext);
  if (!endpointsContext || hasExistingProjection(endpointsContext)) {
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
    traversalContext.materializationContext.offspring,
    createdConnection,
    traversalContext.connectionGene.gater,
  );
}

/**
 * Resolves concrete endpoint nodes for a traversal context.
 *
 * @param traversalContext - Traversal context.
 * @returns Endpoint context or undefined.
 */
function resolveGeneEndpointsContext(
  traversalContext: GeneTraversalContext,
): GeneEndpointsContext | undefined {
  const { offspring } = traversalContext.materializationContext;
  const { from, to } = traversalContext.connectionGene;
  const fromNode = offspring.nodes[from];
  const toNode = offspring.nodes[to];

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
 * Creates a runtime connection for endpoint nodes.
 *
 * @param endpointsContext - Endpoint context.
 * @returns Created connection or undefined.
 */
function createConnectionForEndpoints(
  endpointsContext: GeneEndpointsContext,
): Connection | undefined {
  return createOffspringConnection(
    endpointsContext.traversalContext.materializationContext.offspring,
    endpointsContext.fromNode,
    endpointsContext.toNode,
  );
}

/**
 * Checks whether the source endpoint already projects to the target endpoint.
 *
 * @param endpointsContext - Endpoint context.
 * @returns True when projection already exists.
 */
function hasExistingProjection(
  endpointsContext: GeneEndpointsContext,
): boolean {
  return endpointsContext.fromNode.isProjectingTo(endpointsContext.toNode);
}

/**
 * Validates that a traversal context endpoints are inside offspring bounds.
 *
 * @param traversalContext - Traversal context.
 * @returns True when both indices are bounded.
 */
function isTraversalContextWithinNodeBounds(
  traversalContext: GeneTraversalContext,
): boolean {
  const { connectionGene } = traversalContext;
  const { offspringNodeCount } = traversalContext.materializationContext;
  return (
    connectionGene.from < offspringNodeCount &&
    connectionGene.to < offspringNodeCount
  );
}

/**
 * Validates that a traversal context follows feed-forward ordering.
 *
 * @param traversalContext - Traversal context.
 * @returns True when the gene is strictly forward.
 */
function isTraversalContextFeedForward(
  traversalContext: GeneTraversalContext,
): boolean {
  const { from, to } = traversalContext.connectionGene;
  return from < to;
}

/**
 * Creates a single offspring connection edge.
 *
 * @param offspring - Offspring network.
 * @param fromNode - Source node.
 * @param toNode - Destination node.
 * @returns Created connection or undefined.
 */
function createOffspringConnection(
  offspring: GeneticNetwork,
  fromNode: Node,
  toNode: Node,
): Connection | undefined {
  const createdConnections = offspring.connect(fromNode, toNode);
  return createdConnections.at(FIRST_INDEX);
}

/**
 * Applies gene properties to a runtime connection.
 *
 * @param connection - Runtime connection.
 * @param connectionGene - Gene source.
 * @returns Nothing.
 */
function applyConnectionGeneToConnection(
  connection: Connection,
  connectionGene: ConnectionGene,
): void {
  connection.weight = connectionGene.weight;
  (connection as Connection & ConnectionGeneticProps).enabled =
    connectionGene.enabled !== false;
}

/**
 * Attaches a gater node when the target index is valid.
 *
 * @param offspring - Offspring network.
 * @param connection - Connection to gate.
 * @param gaterIndex - Candidate gater node index.
 * @returns Nothing.
 */
function attachGaterIfAvailable(
  offspring: GeneticNetwork,
  connection: Connection,
  gaterIndex: number,
): void {
  if (gaterIndex === NO_GATER_INDEX || gaterIndex >= offspring.nodes.length) {
    return;
  }
  offspring.gate(offspring.nodes[gaterIndex], connection);
}
