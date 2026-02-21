# architecture/network/genetic

## architecture/network/genetic/network.genetic.utils.ts

### applyConnectionGeneToConnection

`(connection: import("C:/NeatapticTS/src/architecture/connection").default, connectionGene: import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene) => void`

Applies gene properties to a runtime connection.

Parameters:
- `connection` - - Runtime connection.
- `connectionGene` - - Gene source.

Returns: Nothing.

### asGeneticNetwork

`(network: import("C:/NeatapticTS/src/architecture/network").default) => import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork`

Coerces a network to the internal genetic runtime shape.

Parameters:
- `network` - - Source network.

Returns: Network with runtime genetic properties.

### assignNodeIndexes

`(nodes: import("C:/NeatapticTS/src/architecture/node").default[]) => void`

Assigns contiguous indices to a node list.

Parameters:
- `nodes` - - Nodes to reindex.

Returns: Nothing.

### attachGaterIfAvailable

`(offspring: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, connection: import("C:/NeatapticTS/src/architecture/connection").default, gaterIndex: number) => void`

Attaches a gater node when the target index is valid.

Parameters:
- `offspring` - - Offspring network.
- `connection` - - Connection to gate.
- `gaterIndex` - - Candidate gater node index.

Returns: Nothing.

### buildConnectionGene

`(connection: import("C:/NeatapticTS/src/architecture/connection").default) => import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene | undefined`

Builds a connection gene from a concrete connection instance.

Parameters:
- `connection` - - Runtime connection.

Returns: Gene descriptor, or undefined when endpoints lack valid indices.

### buildOffspringNodes

`(parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics, offspringNodeCount: number, equal: boolean, randomGenerator: () => number) => import("C:/NeatapticTS/src/architecture/node").default[]`

Builds the offspring node list by selecting genes per slot.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `offspringNodeCount` - - Target offspring size.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Cloned offspring node genes.

### chooseConnectionGenes

`(parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics, parent1Genes: Record<string, import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene>, parent2Genes: Record<string, import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene>, equal: boolean, randomGenerator: () => number) => import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene[]`

Selects connection genes for offspring inheritance.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `parent1Genes` - - Parent 1 genes by innovation.
- `parent2Genes` - - Parent 2 genes by innovation.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Chosen genes for offspring materialization.

### chooseDisjointGeneFromParent

`(parent: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, sourceGene: import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene) => import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene`

Chooses a disjoint/excess gene from a single parent.

Parameters:
- `parent` - - Source parent.
- `sourceGene` - - Source gene.

Returns: Selected disjoint gene.

### chooseMatchingGene

`(parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent1Gene: import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene, parent2Gene: import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene, randomGenerator: () => number) => import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene`

Chooses a gene for matching innovation IDs.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parent1Gene` - - Parent 1 matching gene.
- `parent2Gene` - - Parent 2 matching gene.
- `randomGenerator` - - Random generator.

Returns: Selected gene.

### cloneConnectionGene

`(sourceGene: import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene) => import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene`

Clones a connection gene.

Parameters:
- `sourceGene` - - Source gene.

Returns: Independent clone.

### cloneNodeGene

`(sourceNode: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/node").default`

Clones node structural gene attributes.

Parameters:
- `sourceNode` - - Source node gene.

Returns: Cloned node.

### collectConnectionGenes

`(parent: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork) => Record<string, import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene>`

Collects all connection genes (standard + self) keyed by innovation ID.

Parameters:
- `parent` - - Parent network.

Returns: Innovation-keyed connection gene map.

### createOffspringConnection

`(offspring: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, fromNode: import("C:/NeatapticTS/src/architecture/node").default, toNode: import("C:/NeatapticTS/src/architecture/node").default) => import("C:/NeatapticTS/src/architecture/connection").default | undefined`

Creates a single offspring connection edge.

Parameters:
- `offspring` - - Offspring network.
- `fromNode` - - Source node.
- `toNode` - - Destination node.

Returns: Created connection or undefined.

### createOffspringScaffold

`(inputSize: number, outputSize: number) => import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork`

Creates an empty offspring scaffold with reset runtime arrays.

Parameters:
- `inputSize` - - Input count.
- `outputSize` - - Output count.

Returns: Initialized offspring runtime object.

### crossOver

`(parentNetwork1: import("C:/NeatapticTS/src/architecture/network").default, parentNetwork2: import("C:/NeatapticTS/src/architecture/network").default, equal: boolean) => import("C:/NeatapticTS/src/architecture/network").default`

Genetic operator: NEAT‑style crossover (legacy merge operator removed).

This module now focuses solely on producing recombinant offspring via {@link crossOver}.
The previous experimental Network.merge has been removed to reduce maintenance surface area
and avoid implying a misleading “sequential composition” guarantee.

### determineOffspringNodeCount

`(equal: boolean, parentMetrics: import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics, randomGenerator: () => number) => number`

Determines offspring node count from fitness/equality policy.

Parameters:
- `equal` - - Whether equal treatment mode is enabled.
- `parentMetrics` - - Parent metrics.
- `randomGenerator` - - Random generator.

Returns: Offspring node count.

### getAlignedOutputNode

`(parent: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, alignedIndex: number) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Reads an aligned output candidate node if index is in the valid non-input range.

Parameters:
- `parent` - - Parent network.
- `alignedIndex` - - Tail-aligned index.

Returns: Output candidate node.

### getNetworkConstructor

`() => import("C:/NeatapticTS/src/architecture/network/network.types").NetworkConstructor`

Dynamically resolves the Network constructor to avoid circular import issues.

Returns: Network constructor.

### getRandomGenerator

`(parentNetwork: import("C:/NeatapticTS/src/architecture/network").default) => () => number`

Resolves the random generator used by crossover decisions.

Parameters:
- `parentNetwork` - - Parent network that may provide a deterministic `_rand` source.

Returns: Random function.

### materializeOffspringConnections

`(offspring: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, chosenGenes: import("C:/NeatapticTS/src/architecture/network/network.types").ConnectionGene[]) => void`

Materializes selected connection genes in the offspring network.

Parameters:
- `offspring` - - Offspring network.
- `chosenGenes` - - Chosen connection genes.

Returns: Nothing.

### resolveParentMetrics

`(parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, outputSize: number) => import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics`

Computes common parent metrics reused across helper functions.

Parameters:
- `parent1` - - First parent network.
- `parent2` - - Second parent network.
- `outputSize` - - Shared output size.

Returns: Parent metrics.

### resolveReenableProbability

`(preferredProbability: number | undefined, fallbackProbability: number | undefined) => number`

Resolves re-enable probability with fallback to default value.

Parameters:
- `preferredProbability` - - Preferred parent probability.
- `fallbackProbability` - - Secondary parent probability.

Returns: Probability in [0, 1].

### selectHiddenNodeGene

`(nodeIndex: number, parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics, equal: boolean, randomGenerator: () => number) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Selects a hidden-region node gene.

Parameters:
- `nodeIndex` - - Slot index.
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Selected hidden node gene.

### selectInputNodeGene

`(nodeIndex: number, parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Selects an input-region node gene.

Parameters:
- `nodeIndex` - - Slot index.
- `parent1` - - First parent.

Returns: Parent 1 input node gene.

### selectNodeGeneAtIndex

`(nodeIndex: number, offspringNodeCount: number, parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics, equal: boolean, randomGenerator: () => number) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Selects a node gene for a specific offspring slot.

Parameters:
- `nodeIndex` - - Slot index.
- `offspringNodeCount` - - Total offspring slots.
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Selected parent node gene, if any.

### selectOutputNodeGene

`(nodeIndex: number, offspringNodeCount: number, parent1: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parent2: import("C:/NeatapticTS/src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("C:/NeatapticTS/src/architecture/network/network.types").ParentMetrics, randomGenerator: () => number) => import("C:/NeatapticTS/src/architecture/node").default | undefined`

Selects an output-region node gene using tail alignment.

Parameters:
- `nodeIndex` - - Slot index.
- `offspringNodeCount` - - Target offspring size.
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `randomGenerator` - - Random generator.

Returns: Selected output node gene.

### validateParentCompatibility

`(parentNetwork1: import("C:/NeatapticTS/src/architecture/network").default, parentNetwork2: import("C:/NeatapticTS/src/architecture/network").default) => void`

Validates parent compatibility for crossover.

Parameters:
- `parentNetwork1` - - First parent candidate.
- `parentNetwork2` - - Second parent candidate.

Returns: Nothing.
