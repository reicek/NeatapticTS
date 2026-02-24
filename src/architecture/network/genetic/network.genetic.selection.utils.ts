import Connection from '../../connection';
import type {
  ConnectionGene,
  ConnectionGeneSelectionContext,
  ConnectionGeneticProps,
  GeneticNetwork,
  Parent1GeneTraversalContext,
  Parent1TraversalSelectionResult,
  ParentMetrics,
} from '../network.types';
import {
  DEFAULT_REENABLE_PROBABILITY,
  NO_GATER_INDEX,
  RANDOM_BINARY_SELECTION_THRESHOLD,
  type RandomGenerator,
} from './network.genetic.utils.types';

/**
 * Collects all connection genes (standard + self) keyed by innovation ID.
 *
 * @param parent - Parent network.
 * @returns Innovation-keyed connection gene map.
 */
export function collectConnectionGenes(
  parent: GeneticNetwork,
): Record<string, ConnectionGene> {
  const genesByInnovationId: Record<string, ConnectionGene> = {};
  const allConnections = [...parent.connections, ...parent.selfconns];

  for (
    let connectionIndex = 0;
    connectionIndex < allConnections.length;
    connectionIndex++
  ) {
    const connectionGene = buildConnectionGene(allConnections[connectionIndex]);
    if (!connectionGene) {
      continue;
    }
    const innovationId = Connection.innovationID(
      connectionGene.from,
      connectionGene.to,
    );
    genesByInnovationId[innovationId] = connectionGene;
  }

  return genesByInnovationId;
}

/**
 * Selects connection genes for offspring inheritance.
 *
 * @param parent1 - First parent.
 * @param parent2 - Second parent.
 * @param parentMetrics - Parent metrics.
 * @param parent1Genes - Parent 1 genes by innovation.
 * @param parent2Genes - Parent 2 genes by innovation.
 * @param equal - Equal-treatment mode.
 * @param randomGenerator - Random generator.
 * @returns Chosen genes for offspring materialization.
 */
export function chooseConnectionGenes(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  parent1Genes: Record<string, ConnectionGene>,
  parent2Genes: Record<string, ConnectionGene>,
  equal: boolean,
  randomGenerator: RandomGenerator,
): ConnectionGene[] {
  const selectionContext = createSelectionContext(
    parent1,
    parent2,
    parentMetrics,
    parent1Genes,
    parent2Genes,
    equal,
    randomGenerator,
  );
  const parent1TraversalResult = selectParent1TraversalGenes(selectionContext);
  const remainingParent2Genes = selectRemainingParent2Genes(
    selectionContext,
    parent1TraversalResult.consumedParent2InnovationIds,
  );
  const parent2OnlyGenes = selectParent2OnlyGenes(
    selectionContext,
    remainingParent2Genes,
  );
  return combineChosenGenes(
    parent1TraversalResult.selectedGenes,
    parent2OnlyGenes,
  );
}

/**
 * Builds a connection gene from a concrete connection instance.
 *
 * @param connection - Runtime connection.
 * @returns Gene descriptor, or undefined when endpoints lack valid indices.
 */
function buildConnectionGene(
  connection: Connection,
): ConnectionGene | undefined {
  if (
    typeof connection.from.index !== 'number' ||
    typeof connection.to.index !== 'number'
  ) {
    return undefined;
  }

  return {
    weight: connection.weight,
    from: connection.from.index,
    to: connection.to.index,
    gater:
      connection.gater && typeof connection.gater.index === 'number'
        ? connection.gater.index
        : NO_GATER_INDEX,
    enabled:
      (connection as Connection & ConnectionGeneticProps).enabled !== false,
  };
}

/**
 * Creates the immutable context for this selection pass.
 *
 * @param sourceParent1 - First parent.
 * @param sourceParent2 - Second parent.
 * @param sourceParentMetrics - Shared parent metrics.
 * @param sourceParent1Genes - Parent-1 genes.
 * @param sourceParent2Genes - Parent-2 genes.
 * @param sourceEqual - Equal-treatment flag.
 * @param sourceRandomGenerator - Random source.
 * @returns Selection context.
 */
function createSelectionContext(
  sourceParent1: GeneticNetwork,
  sourceParent2: GeneticNetwork,
  sourceParentMetrics: ParentMetrics,
  sourceParent1Genes: Record<string, ConnectionGene>,
  sourceParent2Genes: Record<string, ConnectionGene>,
  sourceEqual: boolean,
  sourceRandomGenerator: RandomGenerator,
): ConnectionGeneSelectionContext {
  return {
    parent1: sourceParent1,
    parent2: sourceParent2,
    parentMetrics: sourceParentMetrics,
    equal: sourceEqual,
    randomGenerator: sourceRandomGenerator,
    parent1Genes: sourceParent1Genes,
    parent2Genes: sourceParent2Genes,
  };
}

/**
 * Selects genes reachable from parent-1 innovation traversal.
 *
 * @param context - Selection context.
 * @returns Parent-1 traversal result.
 */
function selectParent1TraversalGenes(
  context: ConnectionGeneSelectionContext,
): Parent1TraversalSelectionResult {
  const parent1TraversalContexts = createParent1TraversalContexts(context);
  return foldParent1TraversalContexts(parent1TraversalContexts);
}

/**
 * Builds parent-1 traversal contexts keyed by innovation IDs.
 *
 * @param context - Selection context.
 * @returns Parent-1 traversal contexts.
 */
function createParent1TraversalContexts(
  context: ConnectionGeneSelectionContext,
): Parent1GeneTraversalContext[] {
  return Object.keys(context.parent1Genes).map((innovationId) => ({
    selectionContext: context,
    innovationId,
    parent1Gene: context.parent1Genes[innovationId],
    parent2Gene: context.parent2Genes[innovationId],
  }));
}

/**
 * Folds parent-1 traversal contexts into selected genes and consumed IDs.
 *
 * @param traversalContexts - Parent-1 traversal contexts.
 * @returns Parent-1 selection result.
 */
function foldParent1TraversalContexts(
  traversalContexts: Parent1GeneTraversalContext[],
): Parent1TraversalSelectionResult {
  const selectedGenes: ConnectionGene[] = [];
  const consumedParent2InnovationIds: string[] = [];

  for (
    let traversalIndex = 0;
    traversalIndex < traversalContexts.length;
    traversalIndex++
  ) {
    const traversalContext = traversalContexts[traversalIndex];
    const selectedGene = selectGeneForParent1TraversalContext(traversalContext);
    if (!selectedGene) {
      continue;
    }

    selectedGenes.push(selectedGene);
    if (traversalContext.parent2Gene) {
      consumedParent2InnovationIds.push(traversalContext.innovationId);
    }
  }

  return {
    selectedGenes,
    consumedParent2InnovationIds,
  };
}

/**
 * Selects one inheritable gene for a parent-1 traversal context.
 *
 * @param traversalContext - Parent-1 traversal context.
 * @returns Selected gene or undefined.
 */
function selectGeneForParent1TraversalContext(
  traversalContext: Parent1GeneTraversalContext,
): ConnectionGene | undefined {
  if (traversalContext.parent2Gene) {
    return chooseMatchingGene(
      traversalContext.selectionContext.parent1,
      traversalContext.selectionContext.parent2,
      traversalContext.parent1Gene,
      traversalContext.parent2Gene,
      traversalContext.selectionContext.randomGenerator,
    );
  }

  if (!canInheritParent1DisjointGenes(traversalContext.selectionContext)) {
    return undefined;
  }

  return chooseDisjointGeneFromParent(
    traversalContext.selectionContext.parent1,
    traversalContext.parent1Gene,
  );
}

/**
 * Builds parent-2 gene map after removing consumed matching innovations.
 *
 * @param context - Selection context.
 * @param consumedInnovationIds - Innovation IDs already consumed via matching genes.
 * @returns Remaining parent-2 genes.
 */
function selectRemainingParent2Genes(
  context: ConnectionGeneSelectionContext,
  consumedInnovationIds: string[],
): Record<string, ConnectionGene> {
  const consumedInnovationIdSet = new Set(consumedInnovationIds);
  return Object.fromEntries(
    Object.entries(context.parent2Genes).filter(
      ([innovationId]) => !consumedInnovationIdSet.has(innovationId),
    ),
  );
}

/**
 * Selects inheritable parent-2-only disjoint/excess genes.
 *
 * @param context - Selection context.
 * @param remainingParent2Genes - Parent-2-only gene map.
 * @returns Selected parent-2-only genes.
 */
function selectParent2OnlyGenes(
  context: ConnectionGeneSelectionContext,
  remainingParent2Genes: Record<string, ConnectionGene>,
): ConnectionGene[] {
  if (!canInheritParent2DisjointGenes(context)) {
    return [];
  }

  return Object.values(remainingParent2Genes).map((parent2OnlyGene) =>
    chooseDisjointGeneFromParent(context.parent2, parent2OnlyGene),
  );
}

/**
 * Determines if parent-1 disjoint/excess genes are inheritable.
 *
 * @param context - Selection context.
 * @returns True when parent-1 disjoint genes can be selected.
 */
function canInheritParent1DisjointGenes(
  context: ConnectionGeneSelectionContext,
): boolean {
  return (
    context.parentMetrics.score1 >= context.parentMetrics.score2 ||
    context.equal
  );
}

/**
 * Determines if parent-2 disjoint/excess genes are inheritable.
 *
 * @param context - Selection context.
 * @returns True when parent-2 disjoint genes can be selected.
 */
function canInheritParent2DisjointGenes(
  context: ConnectionGeneSelectionContext,
): boolean {
  return (
    context.parentMetrics.score2 >= context.parentMetrics.score1 ||
    context.equal
  );
}

/**
 * Combines selected gene partitions into one ordered list.
 *
 * @param parent1TraversalGenes - Genes selected from parent-1 traversal.
 * @param parent2OnlyGenesToAppend - Parent-2-only genes.
 * @returns Combined chosen genes.
 */
function combineChosenGenes(
  parent1TraversalGenes: ConnectionGene[],
  parent2OnlyGenesToAppend: ConnectionGene[],
): ConnectionGene[] {
  return [...parent1TraversalGenes, ...parent2OnlyGenesToAppend];
}

/**
 * Chooses a gene for matching innovation IDs.
 *
 * @param parent1 - First parent.
 * @param parent2 - Second parent.
 * @param parent1Gene - Parent 1 matching gene.
 * @param parent2Gene - Parent 2 matching gene.
 * @param randomGenerator - Random generator.
 * @returns Selected gene.
 */
function chooseMatchingGene(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parent1Gene: ConnectionGene,
  parent2Gene: ConnectionGene,
  randomGenerator: RandomGenerator,
): ConnectionGene {
  const selectedGene =
    randomGenerator() >= RANDOM_BINARY_SELECTION_THRESHOLD
      ? parent1Gene
      : parent2Gene;
  const clonedSelectedGene = cloneConnectionGene(selectedGene);

  if (parent1Gene.enabled === false || parent2Gene.enabled === false) {
    const reenableProbability = resolveReenableProbability(
      parent1._reenableProb,
      parent2._reenableProb,
    );
    clonedSelectedGene.enabled = Math.random() < reenableProbability;
  }

  return clonedSelectedGene;
}

/**
 * Chooses a disjoint/excess gene from a single parent.
 *
 * @param parent - Source parent.
 * @param sourceGene - Source gene.
 * @returns Selected disjoint gene.
 */
function chooseDisjointGeneFromParent(
  parent: GeneticNetwork,
  sourceGene: ConnectionGene,
): ConnectionGene {
  const clonedGene = cloneConnectionGene(sourceGene);
  if (clonedGene.enabled === false) {
    const reenableProbability = resolveReenableProbability(
      parent._reenableProb,
    );
    clonedGene.enabled = Math.random() < reenableProbability;
  }
  return clonedGene;
}

/**
 * Clones a connection gene.
 *
 * @param sourceGene - Source gene.
 * @returns Independent clone.
 */
function cloneConnectionGene(sourceGene: ConnectionGene): ConnectionGene {
  return {
    weight: sourceGene.weight,
    from: sourceGene.from,
    to: sourceGene.to,
    gater: sourceGene.gater,
    enabled: sourceGene.enabled,
  };
}

/**
 * Resolves re-enable probability with fallback to default value.
 *
 * @param preferredProbability - Preferred parent probability.
 * @param fallbackProbability - Secondary parent probability.
 * @returns Probability in [0, 1].
 */
function resolveReenableProbability(
  preferredProbability?: number,
  fallbackProbability?: number,
): number {
  return (
    preferredProbability ?? fallbackProbability ?? DEFAULT_REENABLE_PROBABILITY
  );
}
