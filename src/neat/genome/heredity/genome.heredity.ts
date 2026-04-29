import type { NeatGenomeConnectionGene } from '../genome.types';
import type {
  GenomeHereditySelectionContext,
  GenomeHereditySourceParent,
  SelectedGenomeConnectionGene,
} from './genome.heredity.types';

const DEFAULT_HEREDITY_REENABLE_PROBABILITY = 0.25;
const RANDOM_BINARY_SELECTION_THRESHOLD = 0.5;

interface GenomeConnectionGeneSelectionState extends GenomeHereditySelectionContext {
  parent1GenesByInnovation: Record<string, NeatGenomeConnectionGene>;
  parent2GenesByInnovation: Record<string, NeatGenomeConnectionGene>;
}

interface GenomeParent1TraversalContext {
  selectionState: GenomeConnectionGeneSelectionState;
  innovationId: string;
  parent1Gene: NeatGenomeConnectionGene;
  parent2Gene?: NeatGenomeConnectionGene;
}

interface GenomeParent1TraversalResult {
  selectedGenes: SelectedGenomeConnectionGene[];
  consumedParent2InnovationIds: string[];
}

/**
 * Select inherited connection genes using only the strict genome contract.
 *
 * Step 7.2a moves innovation-aligned heredity selection behind the genome
 * boundary without widening the runtime crossover facade. The runtime shelf
 * still owns node scaffolding and phenotype materialization, while this helper
 * owns three structural decisions:
 *
 * 1. collect parent connection genes by preserved innovation number,
 * 2. resolve matching, disjoint, and excess inheritance from scores plus
 *    equal-mode policy,
 * 3. apply the explicit disabled-gene re-enable rule through the inherited RNG.
 *
 * @param context - Pure genome heredity context.
 * @returns Ordered inherited connection genes plus their source-parent labels.
 *
 * @example
 * ```ts
 * const selectedGenes = selectGenomeHeredityConnectionGenes({
 *   parent1Genome,
 *   parent2Genome,
 *   parent1Score: 2,
 *   parent2Score: 1,
 *   equal: false,
 *   randomGenerator: () => 0.25,
 * });
 * ```
 */
export function selectGenomeHeredityConnectionGenes(
  context: GenomeHereditySelectionContext,
): SelectedGenomeConnectionGene[] {
  // Step 1: Normalize both parents into innovation-keyed structural maps.
  const selectionState = createSelectionState(context);

  // Step 2: Traverse parent 1 first so matching and fitter-parent genes share one path.
  const parent1TraversalResult = selectParent1TraversalGenes(selectionState);

  // Step 3: Append the remaining parent-2-only genes when policy allows it.
  const remainingParent2Genes = selectRemainingParent2Genes(
    selectionState,
    parent1TraversalResult.consumedParent2InnovationIds,
  );
  const parent2OnlyGenes = selectParent2OnlyGenes(
    selectionState,
    remainingParent2Genes,
  );

  return combineSelectedGenomeGenes(
    parent1TraversalResult.selectedGenes,
    parent2OnlyGenes,
  );
}

function createSelectionState(
  context: GenomeHereditySelectionContext,
): GenomeConnectionGeneSelectionState {
  return {
    ...context,
    parent1GenesByInnovation: collectGenomeConnectionGenesByInnovation(
      context.parent1Genome.connectionGenes,
    ),
    parent2GenesByInnovation: collectGenomeConnectionGenesByInnovation(
      context.parent2Genome.connectionGenes,
    ),
  };
}

function collectGenomeConnectionGenesByInnovation(
  connectionGenes: readonly NeatGenomeConnectionGene[],
): Record<string, NeatGenomeConnectionGene> {
  return Object.fromEntries(
    connectionGenes.map((connectionGene) => [
      String(connectionGene.innovation),
      connectionGene,
    ]),
  );
}

function selectParent1TraversalGenes(
  selectionState: GenomeConnectionGeneSelectionState,
): GenomeParent1TraversalResult {
  const traversalContexts = createParent1TraversalContexts(selectionState);
  return foldParent1TraversalContexts(traversalContexts);
}

function createParent1TraversalContexts(
  selectionState: GenomeConnectionGeneSelectionState,
): GenomeParent1TraversalContext[] {
  return Object.keys(selectionState.parent1GenesByInnovation)
    .toSorted(
      (leftInnovationId, rightInnovationId) =>
        Number(leftInnovationId) - Number(rightInnovationId),
    )
    .map((innovationId) => ({
      selectionState,
      innovationId,
      parent1Gene: selectionState.parent1GenesByInnovation[innovationId],
      parent2Gene: selectionState.parent2GenesByInnovation[innovationId],
    }));
}

function foldParent1TraversalContexts(
  traversalContexts: GenomeParent1TraversalContext[],
): GenomeParent1TraversalResult {
  const selectedGenes: SelectedGenomeConnectionGene[] = [];
  const consumedParent2InnovationIds: string[] = [];

  for (
    let traversalContextIndex = 0;
    traversalContextIndex < traversalContexts.length;
    traversalContextIndex++
  ) {
    const traversalContext = traversalContexts[traversalContextIndex];
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

function selectGeneForParent1TraversalContext(
  traversalContext: GenomeParent1TraversalContext,
): SelectedGenomeConnectionGene | undefined {
  if (traversalContext.parent2Gene) {
    return chooseMatchingGene(
      traversalContext.selectionState,
      traversalContext.parent1Gene,
      traversalContext.parent2Gene,
    );
  }

  if (!canInheritParent1DisjointGenes(traversalContext.selectionState)) {
    return undefined;
  }

  return chooseDisjointGeneFromParent(
    traversalContext.parent1Gene,
    'parent1',
    traversalContext.selectionState.parent1ReenableProbability,
    traversalContext.selectionState.randomGenerator,
  );
}

function selectRemainingParent2Genes(
  selectionState: GenomeConnectionGeneSelectionState,
  consumedParent2InnovationIds: string[],
): Record<string, NeatGenomeConnectionGene> {
  const consumedInnovationIdSet = new Set(consumedParent2InnovationIds);

  return Object.fromEntries(
    Object.entries(selectionState.parent2GenesByInnovation).filter(
      ([innovationId]) => !consumedInnovationIdSet.has(innovationId),
    ),
  );
}

function selectParent2OnlyGenes(
  selectionState: GenomeConnectionGeneSelectionState,
  remainingParent2Genes: Record<string, NeatGenomeConnectionGene>,
): SelectedGenomeConnectionGene[] {
  if (!canInheritParent2DisjointGenes(selectionState)) {
    return [];
  }

  return Object.entries(remainingParent2Genes)
    .toSorted(
      ([leftInnovationId], [rightInnovationId]) =>
        Number(leftInnovationId) - Number(rightInnovationId),
    )
    .map(([, parent2OnlyGene]) =>
      chooseDisjointGeneFromParent(
        parent2OnlyGene,
        'parent2',
        selectionState.parent2ReenableProbability,
        selectionState.randomGenerator,
      ),
    );
}

function canInheritParent1DisjointGenes(
  selectionState: GenomeConnectionGeneSelectionState,
): boolean {
  return (
    selectionState.parent1Score >= selectionState.parent2Score ||
    selectionState.equal
  );
}

function canInheritParent2DisjointGenes(
  selectionState: GenomeConnectionGeneSelectionState,
): boolean {
  return (
    selectionState.parent2Score >= selectionState.parent1Score ||
    selectionState.equal
  );
}

function combineSelectedGenomeGenes(
  parent1TraversalGenes: SelectedGenomeConnectionGene[],
  parent2OnlyGenesToAppend: SelectedGenomeConnectionGene[],
): SelectedGenomeConnectionGene[] {
  return [...parent1TraversalGenes, ...parent2OnlyGenesToAppend].toSorted(
    (leftGene, rightGene) =>
      leftGene.connectionGene.innovation - rightGene.connectionGene.innovation,
  );
}

function chooseMatchingGene(
  selectionState: GenomeConnectionGeneSelectionState,
  parent1Gene: NeatGenomeConnectionGene,
  parent2Gene: NeatGenomeConnectionGene,
): SelectedGenomeConnectionGene {
  const sourceParent: GenomeHereditySourceParent =
    selectionState.randomGenerator() >= RANDOM_BINARY_SELECTION_THRESHOLD
      ? 'parent1'
      : 'parent2';
  const selectedSourceGene =
    sourceParent === 'parent1' ? parent1Gene : parent2Gene;
  const clonedSelectedGene = cloneConnectionGene(selectedSourceGene);

  if (parent1Gene.enabled === false || parent2Gene.enabled === false) {
    const reenableProbability = resolveReenableProbability(
      selectionState.parent1ReenableProbability,
      selectionState.parent2ReenableProbability,
    );
    clonedSelectedGene.enabled = shouldReenableDisabledGene(
      reenableProbability,
      selectionState.randomGenerator,
    );
  }

  return {
    connectionGene: clonedSelectedGene,
    sourceParent,
  };
}

function chooseDisjointGeneFromParent(
  sourceGene: NeatGenomeConnectionGene,
  sourceParent: GenomeHereditySourceParent,
  reenableProbability: number | undefined,
  randomGenerator: () => number,
): SelectedGenomeConnectionGene {
  const clonedGene = cloneConnectionGene(sourceGene);
  if (clonedGene.enabled === false) {
    clonedGene.enabled = shouldReenableDisabledGene(
      resolveReenableProbability(reenableProbability),
      randomGenerator,
    );
  }

  return {
    connectionGene: clonedGene,
    sourceParent,
  };
}

function cloneConnectionGene(
  sourceGene: NeatGenomeConnectionGene,
): NeatGenomeConnectionGene {
  return {
    innovation: sourceGene.innovation,
    fromGeneId: sourceGene.fromGeneId,
    toGeneId: sourceGene.toGeneId,
    weight: sourceGene.weight,
    enabled: sourceGene.enabled,
    gaterGeneId: sourceGene.gaterGeneId,
  };
}

function resolveReenableProbability(
  preferredProbability?: number,
  fallbackProbability?: number,
): number {
  return (
    preferredProbability ??
    fallbackProbability ??
    DEFAULT_HEREDITY_REENABLE_PROBABILITY
  );
}

function shouldReenableDisabledGene(
  reenableProbability: number,
  randomGenerator: () => number,
): boolean {
  return randomGenerator() < reenableProbability;
}
