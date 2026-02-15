import type Network from '../network';
import Node from '../node';
import Connection from '../connection';

/**
 * Internal runtime properties used during genetic operations.
 * These properties are accessed dynamically and not part of the formal Network interface.
 */
interface NetworkGeneticProps {
  connections: Connection[];
  nodes: Node[];
  selfconns: Connection[];
  gates: Connection[];
  score?: number;
  _reenableProb?: number;
}

/**
 * Connection gene descriptor used during crossover.
 */
interface ConnectionGene {
  weight: number;
  from: number;
  to: number;
  gater: number;
  enabled: boolean;
}

/**
 * Extended Connection properties for genetic operations.
 */
interface ConnectionGeneticProps {
  enabled?: boolean;
}

/**
 * Concrete runtime shape used by crossover internals.
 */
type GeneticNetwork = Network & NetworkGeneticProps;

/**
 * Compact metrics derived once from both parents.
 */
interface ParentMetrics {
  score1: number;
  score2: number;
  nodeCount1: number;
  nodeCount2: number;
  outputSize: number;
}

/**
 * Constructor signature for dynamic Network import.
 */
interface NetworkConstructor {
  new (input: number, output: number): Network;
}

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
  // Step 1: Validate structural compatibility and normalize parent runtime shapes.
  validateParentCompatibility(parentNetwork1, parentNetwork2);
  const parent1 = asGeneticNetwork(parentNetwork1);
  const parent2 = asGeneticNetwork(parentNetwork2);

  // Step 2: Create an empty offspring network scaffold.
  const offspring = createOffspringScaffold(
    parentNetwork1.input,
    parentNetwork1.output,
  );

  // Step 3: Resolve parent metrics and deterministic index assignment.
  const parentMetrics = resolveParentMetrics(
    parent1,
    parent2,
    parentNetwork1.output,
  );
  assignNodeIndexes(parent1.nodes);
  assignNodeIndexes(parent2.nodes);

  // Step 4: Build offspring node genes.
  const randomGenerator = getRandomGenerator(parent1);
  const offspringNodeCount = determineOffspringNodeCount(
    equal,
    parentMetrics,
    randomGenerator,
  );
  offspring.nodes = buildOffspringNodes(
    parent1,
    parent2,
    parentMetrics,
    offspringNodeCount,
    equal,
    randomGenerator,
  );
  assignNodeIndexes(offspring.nodes);

  // Step 5: Collect and choose connection genes.
  const parent1Genes = collectConnectionGenes(parent1);
  const parent2Genes = collectConnectionGenes(parent2);
  const chosenGenes = chooseConnectionGenes(
    parent1,
    parent2,
    parentMetrics,
    parent1Genes,
    parent2Genes,
    equal,
    randomGenerator,
  );

  // Step 6: Materialize selected genes into concrete offspring connections.
  materializeOffspringConnections(offspring, chosenGenes);
  return offspring;
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
    throw new Error(
      'Parent networks must have the same input and output sizes for crossover.',
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
 * Dynamically resolves the Network constructor to avoid circular import issues.
 *
 * @returns Network constructor.
 */
function getNetworkConstructor(): NetworkConstructor {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  return require('../network').default as NetworkConstructor;
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
    return randomGenerator() >= 0.5 ? parent1Node : parent2Node;
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
    return randomGenerator() >= 0.5 ? parent1Node : parent2Node;
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
        : -1,
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
  const chosenGenes: ConnectionGene[] = [];
  const parent1InnovationIds = Object.keys(parent1Genes);

  // Step 1: Resolve matching and parent1-only genes.
  for (
    let innovationIndex = 0;
    innovationIndex < parent1InnovationIds.length;
    innovationIndex++
  ) {
    const innovationId = parent1InnovationIds[innovationIndex];
    const parent1Gene = parent1Genes[innovationId];
    const parent2Gene = parent2Genes[innovationId];

    if (parent2Gene) {
      chosenGenes.push(
        chooseMatchingGene(
          parent1,
          parent2,
          parent1Gene,
          parent2Gene,
          randomGenerator,
        ),
      );
      delete parent2Genes[innovationId];
      continue;
    }

    if (parentMetrics.score1 >= parentMetrics.score2 || equal) {
      chosenGenes.push(chooseDisjointGeneFromParent(parent1, parent1Gene));
    }
  }

  // Step 2: Resolve parent2-only genes when allowed.
  if (parentMetrics.score2 >= parentMetrics.score1 || equal) {
    const remainingParent2InnovationIds = Object.keys(parent2Genes);
    for (
      let innovationIndex = 0;
      innovationIndex < remainingParent2InnovationIds.length;
      innovationIndex++
    ) {
      const innovationId = remainingParent2InnovationIds[innovationIndex];
      const parent2Gene = parent2Genes[innovationId];
      chosenGenes.push(chooseDisjointGeneFromParent(parent2, parent2Gene));
    }
  }

  return chosenGenes;
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
  const selectedGene = randomGenerator() >= 0.5 ? parent1Gene : parent2Gene;
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
  return preferredProbability ?? fallbackProbability ?? 0.25;
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
  const offspringNodeCount = offspring.nodes.length;

  for (let geneIndex = 0; geneIndex < chosenGenes.length; geneIndex++) {
    const connectionGene = chosenGenes[geneIndex];

    if (!isMaterializableGene(connectionGene, offspringNodeCount)) {
      continue;
    }

    const fromNode = offspring.nodes[connectionGene.from];
    const toNode = offspring.nodes[connectionGene.to];
    if (fromNode.isProjectingTo(toNode)) {
      continue;
    }

    const createdConnection = createOffspringConnection(
      offspring,
      fromNode,
      toNode,
    );
    if (!createdConnection) {
      continue;
    }

    applyConnectionGeneToConnection(createdConnection, connectionGene);
    attachGaterIfAvailable(offspring, createdConnection, connectionGene.gater);
  }
}

/**
 * Determines whether a connection gene can be materialized.
 *
 * @param connectionGene - Gene candidate.
 * @param offspringNodeCount - Offspring node count.
 * @returns True when the gene should be materialized.
 */
function isMaterializableGene(
  connectionGene: ConnectionGene,
  offspringNodeCount: number,
): boolean {
  if (
    connectionGene.from >= offspringNodeCount ||
    connectionGene.to >= offspringNodeCount
  ) {
    return false;
  }

  // Enforce feed-forward ordering for crossover offspring.
  return connectionGene.from < connectionGene.to;
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
  return createdConnections.at(0);
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
  if (gaterIndex === -1 || gaterIndex >= offspring.nodes.length) {
    return;
  }
  offspring.gate(offspring.nodes[gaterIndex], connection);
}

export default { crossOver };
