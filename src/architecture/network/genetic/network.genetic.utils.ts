import type Network from '../../network';
import Node from '../../node';
import Connection from '../../connection';
import type {
  ConnectionGene,
  ConnectionGeneSelectionContext,
  ConnectionGeneticProps,
  CrossoverContext,
  CrossoverNodeBuildContext,
  GeneEndpointsContext,
  GeneTraversalContext,
  GeneticNetwork,
  NetworkConstructor,
  NetworkGeneticProps,
  OffspringMaterializationContext,
  Parent1GeneTraversalContext,
  Parent1TraversalSelectionResult,
  ParentMetrics,
} from '../network.types';

/**
 * Canonical threshold used for random binary parent/gene choice.
 */
const RANDOM_BINARY_SELECTION_THRESHOLD = 0.5;

/**
 * Default probability for re-enabling disabled genes during crossover.
 */
const DEFAULT_REENABLE_PROBABILITY = 0.25;

/**
 * Sentinel index representing that no gater node is assigned.
 */
const NO_GATER_INDEX = -1;

/**
 * First element index used when reading newly created connections.
 */
const FIRST_INDEX = 0;

/**
 * Shared compatibility error message for crossover parent validation.
 */
const PARENT_COMPATIBILITY_ERROR_MESSAGE =
  'Parent networks must have the same input and output sizes for crossover.';

/**
 * Genetic operator: NEAT‑style crossover (legacy merge operator removed).
 *
 * This module now focuses solely on producing recombinant offspring via {@link crossOver}.
 * The previous experimental Network.merge has been removed to reduce maintenance surface area
 * and avoid implying a misleading “sequential composition” guarantee.
 *
 * @module network.genetic
 */

/**
 * NEAT-inspired crossover between two parent networks producing a single offspring.
 *
 * Simplifications relative to canonical NEAT:
 *  - Innovation ID is synthesized from (from.index, to.index) via Connection.innovationID instead of
 *    maintaining a global innovation number per mutation event.
 *  - Node alignment relies on current index ordering. This is weaker than historical innovation
 *    tracking, but adequate for many lightweight evolutionary experiments.
 *
 * High-level algorithm:
 *  1. Validate that parents have identical I/O dimensionality (required for compatibility).
 *  2. Decide offspring node array length:
 *       - If equal flag set or scores tied: random length in [minNodes, maxNodes].
 *       - Else: length of fitter parent.
 *  3. For each index up to chosen size, pick a node gene from parents per rules:
 *       - Input indices: always from parent1 (assumes identical input interface).
 *       - Output indices (aligned from end): randomly choose if both present else take existing.
 *       - Hidden indices: if both present pick randomly; else inherit from fitter (or either if equal).
 *  4. Reindex offspring nodes.
 *  5. Collect connections (standard + self) from each parent into maps keyed by innovationID capturing
 *     weight, enabled flag, and gater index.
 *  6. For overlapping genes (present in both), randomly choose one; if either disabled apply optional
 *     re-enable probability (reenableProb) to possibly re-activate.
 *  7. For disjoint/excess genes, inherit only from fitter parent (or both if equal flag set / scores tied).
 *  8. Materialize selected connection genes if their endpoints both exist in offspring; set weight & enabled state.
 *  9. Reattach gating if gater node exists in offspring.
 *
 * Enabled reactivation probability:
 *  - Parents may carry disabled connections; offspring may re-enable them with a probability derived
 *    from parent-specific _reenableProb (or default 0.25). This allows dormant structures to resurface.
 *
 * @param parentNetwork1 - First parent (ties resolved in its favor when scores equal and equal=false for some cases).
 * @param parentNetwork2 - Second parent.
 * @param equal - Force symmetric treatment regardless of fitness (true => node count random between sizes and both parents equally contribute disjoint genes).
 * @returns Offspring network instance.
 * @throws If input/output sizes differ.
 */
export function crossOver(
  parentNetwork1: Network,
  parentNetwork2: Network,
  equal = false,
): Network {
  const crossoverContext = createCrossoverContext(
    parentNetwork1,
    parentNetwork2,
    equal,
  );
  const nodeBuildContext = createNodeBuildContext(crossoverContext);
  assignOffspringNodes(nodeBuildContext);
  const chosenGenes = chooseOffspringConnectionGenes(crossoverContext);
  materializeOffspringConnections(crossoverContext.offspring, chosenGenes);
  return crossoverContext.offspring;

  /**
   * Creates the immutable crossover baseline context.
   *
   * @param sourceParentNetwork1 - First parent network.
   * @param sourceParentNetwork2 - Second parent network.
   * @param sourceEqual - Equal-treatment mode flag.
   * @returns Initialized crossover context.
   */
  function createCrossoverContext(
    sourceParentNetwork1: Network,
    sourceParentNetwork2: Network,
    sourceEqual: boolean,
  ): CrossoverContext {
    // Step 1: Validate compatibility.
    validateParentCompatibility(sourceParentNetwork1, sourceParentNetwork2);

    // Step 2: Normalize runtime parent shapes.
    const sourceParent1 = asGeneticNetwork(sourceParentNetwork1);
    const sourceParent2 = asGeneticNetwork(sourceParentNetwork2);

    // Step 3: Create the empty offspring scaffold.
    const sourceOffspring = createOffspringScaffold(
      sourceParentNetwork1.input,
      sourceParentNetwork1.output,
    );

    // Step 4: Resolve metrics and deterministic parent indices.
    const sourceParentMetrics = resolveParentMetrics(
      sourceParent1,
      sourceParent2,
      sourceParentNetwork1.output,
    );
    assignNodeIndexes(sourceParent1.nodes);
    assignNodeIndexes(sourceParent2.nodes);

    // Step 5: Resolve random source.
    const sourceRandomGenerator = getRandomGenerator(sourceParent1);

    return {
      parentNetwork1: sourceParentNetwork1,
      parentNetwork2: sourceParentNetwork2,
      equal: sourceEqual,
      parent1: sourceParent1,
      parent2: sourceParent2,
      offspring: sourceOffspring,
      parentMetrics: sourceParentMetrics,
      randomGenerator: sourceRandomGenerator,
    };
  }

  /**
   * Creates the node-build context for offspring node selection.
   *
   * @param context - Crossover baseline context.
   * @returns Node-build context.
   */
  function createNodeBuildContext(
    context: CrossoverContext,
  ): CrossoverNodeBuildContext {
    const offspringNodeCount = determineOffspringNodeCount(
      context.equal,
      context.parentMetrics,
      context.randomGenerator,
    );
    return {
      crossoverContext: context,
      offspringNodeCount,
    };
  }

  /**
   * Builds and reindexes offspring nodes.
   *
   * @param nodeContext - Node-build context.
   * @returns Nothing.
   */
  function assignOffspringNodes(nodeContext: CrossoverNodeBuildContext): void {
    const { crossoverContext, offspringNodeCount } = nodeContext;
    crossoverContext.offspring.nodes = buildOffspringNodes(
      crossoverContext.parent1,
      crossoverContext.parent2,
      crossoverContext.parentMetrics,
      offspringNodeCount,
      crossoverContext.equal,
      crossoverContext.randomGenerator,
    );
    assignNodeIndexes(crossoverContext.offspring.nodes);
  }

  /**
   * Chooses all offspring connection genes from both parents.
   *
   * @param context - Crossover baseline context.
   * @returns Chosen connection genes.
   */
  function chooseOffspringConnectionGenes(
    context: CrossoverContext,
  ): ConnectionGene[] {
    const parent1Genes = collectConnectionGenes(context.parent1);
    const parent2Genes = collectConnectionGenes(context.parent2);
    return chooseConnectionGenes(
      context.parent1,
      context.parent2,
      context.parentMetrics,
      parent1Genes,
      parent2Genes,
      context.equal,
      context.randomGenerator,
    );
  }
}

/**
 * Validates parent compatibility for crossover.
 *
 * @param parentNetwork1 - First parent candidate.
 * @param parentNetwork2 - Second parent candidate.
 * @returns Nothing.
 * @throws If input/output dimensions differ.
 */
function validateParentCompatibility(
  parentNetwork1: Network,
  parentNetwork2: Network,
): void {
  if (
    parentNetwork1.input !== parentNetwork2.input ||
    parentNetwork1.output !== parentNetwork2.output
  ) {
    throw new Error(PARENT_COMPATIBILITY_ERROR_MESSAGE);
  }
}

/**
 * Coerces a network to the internal genetic runtime shape.
 *
 * @param network - Source network.
 * @returns Network with runtime genetic properties.
 */
function asGeneticNetwork(network: Network): GeneticNetwork {
  return network as GeneticNetwork;
}

/**
 * Dynamically resolves the Network constructor to avoid circular import issues.
 *
 * @returns Network constructor.
 */
function getNetworkConstructor(): NetworkConstructor {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require('../../network').default as NetworkConstructor;
}

/**
 * Creates an empty offspring scaffold with reset runtime arrays.
 *
 * @param inputSize - Input count.
 * @param outputSize - Output count.
 * @returns Initialized offspring runtime object.
 */
function createOffspringScaffold(
  inputSize: number,
  outputSize: number,
): GeneticNetwork {
  // Step 1: Build a fresh Network instance.
  const DynamicNetworkConstructor = getNetworkConstructor();
  const offspring = new DynamicNetworkConstructor(
    inputSize,
    outputSize,
  ) as GeneticNetwork;

  // Step 2: Reset mutable runtime arrays used by genetic operators.
  offspring.connections = [];
  offspring.nodes = [];
  offspring.selfconns = [];
  offspring.gates = [];
  return offspring;
}

/**
 * Computes common parent metrics reused across helper functions.
 *
 * @param parent1 - First parent network.
 * @param parent2 - Second parent network.
 * @param outputSize - Shared output size.
 * @returns Parent metrics.
 */
function resolveParentMetrics(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  outputSize: number,
): ParentMetrics {
  return {
    score1: parent1.score ?? 0,
    score2: parent2.score ?? 0,
    nodeCount1: parent1.nodes.length,
    nodeCount2: parent2.nodes.length,
    outputSize,
  };
}

/**
 * Resolves the random generator used by crossover decisions.
 *
 * @param parentNetwork - Parent network that may provide a deterministic `_rand` source.
 * @returns Random function.
 */
function getRandomGenerator(parentNetwork: Network): () => number {
  const networkWithDynamicFields = parentNetwork as Record<
    string,
    (() => number) | number | undefined
  >;
  const dynamicRandomCandidate = networkWithDynamicFields._rand;
  return typeof dynamicRandomCandidate === 'function'
    ? dynamicRandomCandidate
    : Math.random;
}

/**
 * Determines offspring node count from fitness/equality policy.
 *
 * @param equal - Whether equal treatment mode is enabled.
 * @param parentMetrics - Parent metrics.
 * @param randomGenerator - Random generator.
 * @returns Offspring node count.
 */
function determineOffspringNodeCount(
  equal: boolean,
  parentMetrics: ParentMetrics,
  randomGenerator: () => number,
): number {
  if (equal || parentMetrics.score1 === parentMetrics.score2) {
    const minNodes = Math.min(
      parentMetrics.nodeCount1,
      parentMetrics.nodeCount2,
    );
    const maxNodes = Math.max(
      parentMetrics.nodeCount1,
      parentMetrics.nodeCount2,
    );
    return Math.floor(randomGenerator() * (maxNodes - minNodes + 1) + minNodes);
  }
  return parentMetrics.score1 > parentMetrics.score2
    ? parentMetrics.nodeCount1
    : parentMetrics.nodeCount2;
}

/**
 * Assigns contiguous indices to a node list.
 *
 * @param nodes - Nodes to reindex.
 * @returns Nothing.
 */
function assignNodeIndexes(nodes: Node[]): void {
  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    nodes[nodeIndex].index = nodeIndex;
  }
}

/**
 * Builds the offspring node list by selecting genes per slot.
 *
 * @param parent1 - First parent.
 * @param parent2 - Second parent.
 * @param parentMetrics - Parent metrics.
 * @param offspringNodeCount - Target offspring size.
 * @param equal - Equal-treatment mode.
 * @param randomGenerator - Random generator.
 * @returns Cloned offspring node genes.
 */
function buildOffspringNodes(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  offspringNodeCount: number,
  equal: boolean,
  randomGenerator: () => number,
): Node[] {
  const offspringNodes: Node[] = [];

  // Step 1: Select parent node genes by slot and clone structural properties.
  for (let nodeIndex = 0; nodeIndex < offspringNodeCount; nodeIndex++) {
    const selectedNode = selectNodeGeneAtIndex(
      nodeIndex,
      offspringNodeCount,
      parent1,
      parent2,
      parentMetrics,
      equal,
      randomGenerator,
    );
    if (selectedNode) {
      offspringNodes.push(cloneNodeGene(selectedNode));
    }
  }

  return offspringNodes;
}

/**
 * Selects a node gene for a specific offspring slot.
 *
 * @param nodeIndex - Slot index.
 * @param offspringNodeCount - Total offspring slots.
 * @param parent1 - First parent.
 * @param parent2 - Second parent.
 * @param parentMetrics - Parent metrics.
 * @param equal - Equal-treatment mode.
 * @param randomGenerator - Random generator.
 * @returns Selected parent node gene, if any.
 */
function selectNodeGeneAtIndex(
  nodeIndex: number,
  offspringNodeCount: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: () => number,
): Node | undefined {
  if (nodeIndex < parent1.input) {
    return selectInputNodeGene(nodeIndex, parent1);
  }

  if (nodeIndex >= offspringNodeCount - parentMetrics.outputSize) {
    return selectOutputNodeGene(
      nodeIndex,
      offspringNodeCount,
      parent1,
      parent2,
      parentMetrics,
      randomGenerator,
    );
  }

  return selectHiddenNodeGene(
    nodeIndex,
    parent1,
    parent2,
    parentMetrics,
    equal,
    randomGenerator,
  );
}

/**
 * Selects an input-region node gene.
 *
 * @param nodeIndex - Slot index.
 * @param parent1 - First parent.
 * @returns Parent 1 input node gene.
 */
function selectInputNodeGene(
  nodeIndex: number,
  parent1: GeneticNetwork,
): Node | undefined {
  return nodeIndex < parent1.nodes.length
    ? parent1.nodes[nodeIndex]
    : undefined;
}

/**
 * Selects an output-region node gene using tail alignment.
 *
 * @param nodeIndex - Slot index.
 * @param offspringNodeCount - Target offspring size.
 * @param parent1 - First parent.
 * @param parent2 - Second parent.
 * @param parentMetrics - Parent metrics.
 * @param randomGenerator - Random generator.
 * @returns Selected output node gene.
 */
function selectOutputNodeGene(
  nodeIndex: number,
  offspringNodeCount: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  randomGenerator: () => number,
): Node | undefined {
  const alignedParent1Index =
    parentMetrics.nodeCount1 - (offspringNodeCount - nodeIndex);
  const alignedParent2Index =
    parentMetrics.nodeCount2 - (offspringNodeCount - nodeIndex);

  const parent1Node = getAlignedOutputNode(parent1, alignedParent1Index);
  const parent2Node = getAlignedOutputNode(parent2, alignedParent2Index);

  if (parent1Node && parent2Node) {
    return randomGenerator() >= RANDOM_BINARY_SELECTION_THRESHOLD
      ? parent1Node
      : parent2Node;
  }

  return parent1Node ?? parent2Node;
}

/**
 * Reads an aligned output candidate node if index is in the valid non-input range.
 *
 * @param parent - Parent network.
 * @param alignedIndex - Tail-aligned index.
 * @returns Output candidate node.
 */
function getAlignedOutputNode(
  parent: GeneticNetwork,
  alignedIndex: number,
): Node | undefined {
  if (alignedIndex < parent.input || alignedIndex >= parent.nodes.length) {
    return undefined;
  }
  return parent.nodes[alignedIndex];
}

/**
 * Selects a hidden-region node gene.
 *
 * @param nodeIndex - Slot index.
 * @param parent1 - First parent.
 * @param parent2 - Second parent.
 * @param parentMetrics - Parent metrics.
 * @param equal - Equal-treatment mode.
 * @param randomGenerator - Random generator.
 * @returns Selected hidden node gene.
 */
function selectHiddenNodeGene(
  nodeIndex: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: () => number,
): Node | undefined {
  const parent1Node =
    nodeIndex < parentMetrics.nodeCount1 ? parent1.nodes[nodeIndex] : undefined;
  const parent2Node =
    nodeIndex < parentMetrics.nodeCount2 ? parent2.nodes[nodeIndex] : undefined;

  if (parent1Node && parent2Node) {
    return randomGenerator() >= RANDOM_BINARY_SELECTION_THRESHOLD
      ? parent1Node
      : parent2Node;
  }

  if (parent1Node && (parentMetrics.score1 >= parentMetrics.score2 || equal)) {
    return parent1Node;
  }

  if (parent2Node && (parentMetrics.score2 >= parentMetrics.score1 || equal)) {
    return parent2Node;
  }

  return undefined;
}

/**
 * Clones node structural gene attributes.
 *
 * @param sourceNode - Source node gene.
 * @returns Cloned node.
 */
function cloneNodeGene(sourceNode: Node): Node {
  const clonedNode = new Node(sourceNode.type);
  clonedNode.bias = sourceNode.bias;
  clonedNode.squash = sourceNode.squash;
  return clonedNode;
}

/**
 * Collects all connection genes (standard + self) keyed by innovation ID.
 *
 * @param parent - Parent network.
 * @returns Innovation-keyed connection gene map.
 */
function collectConnectionGenes(
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
function chooseConnectionGenes(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  parent1Genes: Record<string, ConnectionGene>,
  parent2Genes: Record<string, ConnectionGene>,
  equal: boolean,
  randomGenerator: () => number,
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
    sourceRandomGenerator: () => number,
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
      const selectedGene =
        selectGeneForParent1TraversalContext(traversalContext);
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
  randomGenerator: () => number,
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

/**
 * Materializes selected connection genes in the offspring network.
 *
 * @param offspring - Offspring network.
 * @param chosenGenes - Chosen connection genes.
 * @returns Nothing.
 */
function materializeOffspringConnections(
  offspring: GeneticNetwork,
  chosenGenes: ConnectionGene[],
): void {
  const materializationContext = createMaterializationContext(offspring);
  const eligibleTraversalContexts = collectEligibleTraversalContexts(
    materializationContext,
    chosenGenes,
  );
  materializeTraversalContexts(eligibleTraversalContexts);
  return;

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

export default { crossOver };
