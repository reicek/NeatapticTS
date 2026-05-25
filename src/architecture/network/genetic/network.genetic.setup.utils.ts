import Network from '../../network/network';
import Node from '../../node';
import type {
  ConnectionGene,
  CrossoverContext,
  CrossoverNodeBuildContext,
  GeneticNetwork,
  NetworkTopologyIntent,
  ParentMetrics,
} from '../network.types';
import { hasFeedForwardTopologyContract } from '../topology/network.topology.contract.utils';
import {
  PARENT_COMPATIBILITY_ERROR_MESSAGE,
  RANDOM_BINARY_SELECTION_THRESHOLD,
  type RandomGenerator,
} from './network.genetic.utils.types';
import { NetworkGeneticParentCompatibilityError } from './network.genetic.errors';
import { chooseConnectionGenes } from './network.genetic.selection.utils';

const OFFSPRING_SCAFFOLD_SEED = 0;
const NEUTRAL_NODE_CONSTRUCTOR_RANDOM = () => 0.5;

interface ParentNodeRegions {
  inputNodes: Node[];
  hiddenNodes: Node[];
  outputNodes: Node[];
}

/**
 * Crossover setup helpers for the network genetic boundary.
 *
 * This file prepares the immutable context used by the rest of the crossover
 * pipeline.
 *
 * Key responsibilities:
 *
 * - Validate that parents are compatible (same input/output sizes).
 * - Normalize parent runtime shapes and establish deterministic node indexing.
 * - Resolve the offspring topology intent conservatively.
 * - Resolve the crossover RNG source:
 *   - prefer an explicitly injected RNG (controller-owned determinism),
 *   - otherwise fall back to the parent runtime `_rand` hook,
 *   - finally fall back to `Math.random` for standalone use.
 *
 * This separation keeps the public crossover surface compact while giving the
 * evolve controller one clear seam for deterministic replay.
 */

/**
 * Creates the immutable crossover baseline context.
 *
 * @param parentNetwork1 First parent network.
 * @param parentNetwork2 Second parent network.
 * @param equal Equal-treatment mode flag.
 * @param injectedRandomGenerator Optional explicit crossover RNG supplied by the evolve controller.
 * @returns Initialized crossover context.
 */
export function createCrossoverContext(
  parentNetwork1: Network,
  parentNetwork2: Network,
  equal: boolean,
  injectedRandomGenerator?: RandomGenerator,
): CrossoverContext {
  // Step 1: Validate compatibility.
  validateParentCompatibility(parentNetwork1, parentNetwork2);

  // Step 2: Normalize runtime parent shapes.
  const parent1 = asGeneticNetwork(parentNetwork1);
  const parent2 = asGeneticNetwork(parentNetwork2);

  // Step 3: Create the empty offspring scaffold.
  const offspringTopologyIntent = resolveOffspringTopologyIntent(
    parentNetwork1,
    parentNetwork2,
  );
  const offspring = createOffspringScaffold(
    parentNetwork1.input,
    parentNetwork1.output,
    offspringTopologyIntent,
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
  const randomGenerator =
    injectedRandomGenerator ?? getRandomGenerator(parent1);

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
 * @param context Crossover baseline context.
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
 * @param nodeContext Node-build context.
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
  crossoverContext.offspring.refreshExplicitIORoles();
}

/**
 * Chooses all offspring connection genes from both parents.
 *
 * Innovation-aligned heredity selection now lives behind the strict genome
 * boundary. This runtime seam now returns only the stable materialization
 * descriptor consumed by the phenotype builder, without carrying parent-local
 * node-index hints across the adapter.
 *
 * @param context Crossover baseline context.
 * @returns Chosen connection genes.
 */
export function chooseOffspringConnectionGenes(
  context: CrossoverContext,
): ConnectionGene[] {
  return chooseConnectionGenes(
    context.parent1,
    context.parent2,
    context.parentMetrics,
    context.equal,
    context.randomGenerator,
  );
}

/**
 * Validates parent compatibility for crossover.
 *
 * @param parentNetwork1 First parent candidate.
 * @param parentNetwork2 Second parent candidate.
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
 * @param network Source network.
 * @returns Network with runtime genetic properties.
 */
function asGeneticNetwork(network: Network): GeneticNetwork {
  return network as GeneticNetwork;
}

/**
 * Creates an empty offspring scaffold with reset runtime arrays.
 *
 * @param inputSize Input count.
 * @param outputSize Output count.
 * @param topologyIntent Topology policy inherited by the offspring scaffold.
 * @returns Initialized offspring runtime object.
 */
function createOffspringScaffold(
  inputSize: number,
  outputSize: number,
  topologyIntent: NetworkTopologyIntent,
): GeneticNetwork {
  // Step 1: Build a fresh Network instance.
  const offspring = new Network(inputSize, outputSize, {
    seed: OFFSPRING_SCAFFOLD_SEED,
    topologyIntent,
  }) as GeneticNetwork;

  // Step 2: Reset mutable runtime arrays used by genetic operators.
  offspring.connections = [];
  offspring.nodes = [];
  offspring.selfconns = [];
  offspring.gates = [];
  offspring.refreshExplicitIORoles();
  return offspring;
}

/**
 * Resolves the topology policy to preserve on the offspring scaffold.
 *
 * A mixed-parent crossover should not silently downgrade recurrent-capable
 * parents back to feed-forward mode, because that would prune valid inherited
 * genes during materialization. The offspring therefore stays feed-forward only
 * when both parents advertise the feed-forward contract.
 *
 * @param parentNetwork1 First parent network.
 * @param parentNetwork2 Second parent network.
 * @returns Offspring topology intent.
 */
function resolveOffspringTopologyIntent(
  parentNetwork1: Network,
  parentNetwork2: Network,
): NetworkTopologyIntent {
  return hasFeedForwardTopologyContract(parentNetwork1) &&
    hasFeedForwardTopologyContract(parentNetwork2)
    ? 'feed-forward'
    : 'unconstrained';
}

/**
 * Computes common parent metrics reused across helper functions.
 *
 * @param parent1 First parent network.
 * @param parent2 Second parent network.
 * @param outputSize Shared output size.
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
 * @param parentNetwork Parent network that may provide a deterministic `_rand` source.
 * @returns Random function.
 */
function getRandomGenerator(parentNetwork: Network): RandomGenerator {
  return (parentNetwork as unknown as { _rand: RandomGenerator })._rand;
}

/**
 * Determines offspring node count from fitness/equality policy.
 *
 * @param equal Whether equal treatment mode is enabled.
 * @param parentMetrics Parent metrics.
 * @param randomGenerator Random generator.
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
 * @param nodes Nodes to reindex.
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
 * @param parent1 First parent.
 * @param parent2 Second parent.
 * @param parentMetrics Parent metrics.
 * @param offspringNodeCount Target offspring size.
 * @param equal Equal-treatment mode.
 * @param randomGenerator Random generator.
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
  const parent1NodeRegions = createParentNodeRegions(parent1.nodes);
  const parent2NodeRegions = createParentNodeRegions(parent2.nodes);
  const offspringNodes: Node[] = [];

  // Step 1: Select parent node genes by slot and clone structural properties.
  for (let nodeIndex = 0; nodeIndex < offspringNodeCount; nodeIndex++) {
    const selectedNode = selectNodeGeneAtIndex(
      nodeIndex,
      offspringNodeCount,
      parent1NodeRegions,
      parent2NodeRegions,
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
 * @param nodeIndex Slot index.
 * @param offspringNodeCount Total offspring slots.
 * @param parent1NodeRegions First parent node partitions.
 * @param parent2NodeRegions Second parent node partitions.
 * @param parentMetrics Parent metrics.
 * @param equal Equal-treatment mode.
 * @param randomGenerator Random generator.
 * @returns Selected parent node gene, when present.
 */
function selectNodeGeneAtIndex(
  nodeIndex: number,
  offspringNodeCount: number,
  parent1NodeRegions: ParentNodeRegions,
  parent2NodeRegions: ParentNodeRegions,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
): Node | undefined {
  if (nodeIndex < parent1NodeRegions.inputNodes.length) {
    return selectInputNodeGene(nodeIndex, parent1NodeRegions);
  }

  if (nodeIndex >= offspringNodeCount - parentMetrics.outputSize) {
    return selectOutputNodeGene(
      nodeIndex - (offspringNodeCount - parentMetrics.outputSize),
      parent1NodeRegions,
      parent2NodeRegions,
      randomGenerator,
    );
  }

  return selectHiddenNodeGene(
    nodeIndex - parent1NodeRegions.inputNodes.length,
    parent1NodeRegions,
    parent2NodeRegions,
    parentMetrics,
    equal,
    randomGenerator,
  );
}

/**
 * Selects an input-region node gene.
 *
 * @param nodeIndex Slot index.
 * @param parent1 First parent.
 * @returns Parent 1 input node gene.
 */
function selectInputNodeGene(
  nodeIndex: number,
  parent1NodeRegions: ParentNodeRegions,
): Node | undefined {
  return parent1NodeRegions.inputNodes.at(nodeIndex);
}

/**
 * Selects an output-region node gene by interface ordinal.
 *
 * Runtime parents can drift away from strict input-hidden-output ordering after
 * structural edits. This runtime shelf therefore reads outputs from canonical
 * per-type partitions rather than trusting the raw tail slots on `nodes[]`.
 *
 * @param outputOrdinal Output-slot ordinal.
 * @param parent1NodeRegions First parent node partitions.
 * @param parent2NodeRegions Second parent node partitions.
 * @param randomGenerator Random generator.
 * @returns Selected output node gene.
 */
function selectOutputNodeGene(
  outputOrdinal: number,
  parent1NodeRegions: ParentNodeRegions,
  parent2NodeRegions: ParentNodeRegions,
  randomGenerator: RandomGenerator,
): Node | undefined {
  const parent1Node = parent1NodeRegions.outputNodes.at(outputOrdinal);
  const parent2Node = parent2NodeRegions.outputNodes.at(outputOrdinal);

  if (parent1Node && parent2Node) {
    return randomGenerator() >= RANDOM_BINARY_SELECTION_THRESHOLD
      ? parent1Node
      : parent2Node;
  }

  return parent1Node ?? parent2Node;
}

/**
 * Selects a hidden-region node gene.
 *
 * @param hiddenOrdinal Hidden-slot ordinal.
 * @param parent1NodeRegions First parent node partitions.
 * @param parent2NodeRegions Second parent node partitions.
 * @param parentMetrics Parent metrics.
 * @param equal Equal-treatment mode.
 * @param randomGenerator Random generator.
 * @returns Selected hidden node gene.
 */
function selectHiddenNodeGene(
  hiddenOrdinal: number,
  parent1NodeRegions: ParentNodeRegions,
  parent2NodeRegions: ParentNodeRegions,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
): Node | undefined {
  const parent1Node = parent1NodeRegions.hiddenNodes.at(hiddenOrdinal);
  const parent2Node = parent2NodeRegions.hiddenNodes.at(hiddenOrdinal);

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
 * Partitions one runtime node list into canonical IO and hidden shelves.
 *
 * Crossover setup still owns runtime node selection, but it must not assume the
 * raw `nodes[]` array already keeps outputs at the tail. Structural edits can
 * preserve a valid phenotype while drifting away from that canonical order.
 * Reading through per-type partitions keeps output-slot inheritance stable
 * without mutating the parent runtime graph.
 *
 * @param nodes Ordered runtime node list.
 * @returns Canonical node partitions.
 */
function createParentNodeRegions(nodes: Node[]): ParentNodeRegions {
  const inputNodes: Node[] = [];
  const hiddenNodes: Node[] = [];
  const outputNodes: Node[] = [];

  for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
    const node = nodes[nodeIndex];

    if (node.type === 'input') {
      inputNodes.push(node);
      continue;
    }

    if (node.type === 'output') {
      outputNodes.push(node);
      continue;
    }

    hiddenNodes.push(node);
  }

  return {
    inputNodes,
    hiddenNodes,
    outputNodes,
  };
}

/**
 * Clones node structural gene attributes.
 *
 * Historical node identity must survive crossover even though the offspring is
 * rebuilt as a fresh runtime graph. Preserving `geneId` here lets later
 * materialization resolve inherited connection endpoints by stable gene identity
 * instead of by whatever transient slot the node lands in after reindexing.
 *
 * @param sourceNode Source node gene.
 * @returns Cloned node.
 */
function cloneNodeGene(sourceNode: Node): Node {
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
