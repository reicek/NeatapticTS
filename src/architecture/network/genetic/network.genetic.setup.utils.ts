import Network from '../../network/network';
import Node from '../../node';
import type {
  ConnectionGene,
  CrossoverContext,
  CrossoverNodeBuildContext,
  GeneticNetwork,
  ParentMetrics,
} from '../network.types';
import {
  PARENT_COMPATIBILITY_ERROR_MESSAGE,
  RANDOM_BINARY_SELECTION_THRESHOLD,
  type RandomGenerator,
} from './network.genetic.utils.types';
import { NetworkGeneticParentCompatibilityError } from './network.genetic.errors';
import {
  chooseConnectionGenes,
  collectConnectionGenes,
} from './network.genetic.selection.utils';

/**
 * Creates the immutable crossover baseline context.
 *
 * @param parentNetwork1 - First parent network.
 * @param parentNetwork2 - Second parent network.
 * @param equal - Equal-treatment mode flag.
 * @returns Initialized crossover context.
 */
export function createCrossoverContext(
  parentNetwork1: Network,
  parentNetwork2: Network,
  equal: boolean,
): CrossoverContext {
  // Step 1: Validate compatibility.
  validateParentCompatibility(parentNetwork1, parentNetwork2);

  // Step 2: Normalize runtime parent shapes.
  const parent1 = asGeneticNetwork(parentNetwork1);
  const parent2 = asGeneticNetwork(parentNetwork2);

  // Step 3: Create the empty offspring scaffold.
  const offspring = createOffspringScaffold(
    parentNetwork1.input,
    parentNetwork1.output,
  );

  // Step 4: Resolve metrics and deterministic parent indices.
  const parentMetrics = resolveParentMetrics(
    parent1,
    parent2,
    parentNetwork1.output,
  );
  assignNodeIndexes(parent1.nodes);
  assignNodeIndexes(parent2.nodes);

  // Step 5: Resolve random source.
  const randomGenerator = getRandomGenerator(parent1);

  return {
    parentNetwork1,
    parentNetwork2,
    equal,
    parent1,
    parent2,
    offspring,
    parentMetrics,
    randomGenerator,
  };
}

/**
 * Creates the node-build context for offspring node selection.
 *
 * @param context - Crossover baseline context.
 * @returns Node-build context.
 */
export function createNodeBuildContext(
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
export function assignOffspringNodes(
  nodeContext: CrossoverNodeBuildContext,
): void {
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
export function chooseOffspringConnectionGenes(
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
    throw new NetworkGeneticParentCompatibilityError(
      PARENT_COMPATIBILITY_ERROR_MESSAGE,
    );
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
  const offspring = new Network(inputSize, outputSize) as GeneticNetwork;

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
function getRandomGenerator(parentNetwork: Network): RandomGenerator {
  const networkWithDynamicFields = parentNetwork as Record<
    string,
    RandomGenerator | number | undefined
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
  randomGenerator: RandomGenerator,
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
  randomGenerator: RandomGenerator,
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
 * @returns Selected parent node gene, when present.
 */
function selectNodeGeneAtIndex(
  nodeIndex: number,
  offspringNodeCount: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
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
  randomGenerator: RandomGenerator,
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
  randomGenerator: RandomGenerator,
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
