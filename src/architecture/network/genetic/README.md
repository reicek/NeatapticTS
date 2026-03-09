# architecture/network/genetic

## architecture/network/genetic/network.genetic.utils.types.ts

### network.genetic.utils.types

Canonical threshold used for random binary parent/gene choice.

### DEFAULT_REENABLE_PROBABILITY

### FIRST_INDEX

### NO_GATER_INDEX

### PARENT_COMPATIBILITY_ERROR_MESSAGE

### RANDOM_BINARY_SELECTION_THRESHOLD

### RandomGenerator

`() => number`

Shared random generator signature for genetic operators.

## architecture/network/genetic/network.genetic.utils.ts

### crossOver

`(parentNetwork1: import("src/architecture/network").default, parentNetwork2: import("src/architecture/network").default, equal: boolean) => import("src/architecture/network").default`

Genetic operator: NEAT‑style crossover (legacy merge operator removed).

This module now focuses solely on producing recombinant offspring via {@link crossOver}.
The previous experimental `Network.merge` flow has been removed to reduce maintenance
surface area and avoid implying a misleading sequential-composition guarantee.

Design notes:
- The implementation favors deterministic, inspectable orchestration at the top level.
- Gene-selection details are delegated to setup/materialization helpers so the public
  crossover API stays compact and predictable.
- The resulting offspring preserves the same input/output interface as both parents,
  which keeps downstream evaluation and training pipelines compatible.

## architecture/network/genetic/network.genetic.setup.utils.ts

### asGeneticNetwork

`(network: import("src/architecture/network").default) => import("src/architecture/network/network.types").GeneticNetwork`

Coerces a network to the internal genetic runtime shape.

Parameters:
- `network` - - Source network.

Returns: Network with runtime genetic properties.

### assignNodeIndexes

`(nodes: import("src/architecture/node").default[]) => void`

Assigns contiguous indices to a node list.

Parameters:
- `nodes` - - Nodes to reindex.

Returns: Nothing.

### assignOffspringNodes

`(nodeContext: import("src/architecture/network/network.types").CrossoverNodeBuildContext) => void`

Builds and reindexes offspring nodes.

Parameters:
- `nodeContext` - - Node-build context.

Returns: Nothing.

### buildOffspringNodes

`(parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("src/architecture/network/network.types").ParentMetrics, offspringNodeCount: number, equal: boolean, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/node").default[]`

Builds the offspring node list by selecting genes per slot.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `offspringNodeCount` - - Target offspring size.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Cloned offspring node genes.

### chooseOffspringConnectionGenes

`(context: import("src/architecture/network/network.types").CrossoverContext) => import("src/architecture/network/network.types").ConnectionGene[]`

Chooses all offspring connection genes from both parents.

Parameters:
- `context` - - Crossover baseline context.

Returns: Chosen connection genes.

### cloneNodeGene

`(sourceNode: import("src/architecture/node").default) => import("src/architecture/node").default`

Clones node structural gene attributes.

Parameters:
- `sourceNode` - - Source node gene.

Returns: Cloned node.

### createCrossoverContext

`(parentNetwork1: import("src/architecture/network").default, parentNetwork2: import("src/architecture/network").default, equal: boolean) => import("src/architecture/network/network.types").CrossoverContext`

Creates the immutable crossover baseline context.

Parameters:
- `parentNetwork1` - - First parent network.
- `parentNetwork2` - - Second parent network.
- `equal` - - Equal-treatment mode flag.

Returns: Initialized crossover context.

### createNodeBuildContext

`(context: import("src/architecture/network/network.types").CrossoverContext) => import("src/architecture/network/network.types").CrossoverNodeBuildContext`

Creates the node-build context for offspring node selection.

Parameters:
- `context` - - Crossover baseline context.

Returns: Node-build context.

### createOffspringScaffold

`(inputSize: number, outputSize: number) => import("src/architecture/network/network.types").GeneticNetwork`

Creates an empty offspring scaffold with reset runtime arrays.

Parameters:
- `inputSize` - - Input count.
- `outputSize` - - Output count.

Returns: Initialized offspring runtime object.

### determineOffspringNodeCount

`(equal: boolean, parentMetrics: import("src/architecture/network/network.types").ParentMetrics, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => number`

Determines offspring node count from fitness/equality policy.

Parameters:
- `equal` - - Whether equal treatment mode is enabled.
- `parentMetrics` - - Parent metrics.
- `randomGenerator` - - Random generator.

Returns: Offspring node count.

### getAlignedOutputNode

`(parent: import("src/architecture/network/network.types").GeneticNetwork, alignedIndex: number) => import("src/architecture/node").default | undefined`

Reads an aligned output candidate node if index is in the valid non-input range.

Parameters:
- `parent` - - Parent network.
- `alignedIndex` - - Tail-aligned index.

Returns: Output candidate node.

### getRandomGenerator

`(parentNetwork: import("src/architecture/network").default) => import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator`

Resolves the random generator used by crossover decisions.

Parameters:
- `parentNetwork` - - Parent network that may provide a deterministic `_rand` source.

Returns: Random function.

### resolveParentMetrics

`(parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, outputSize: number) => import("src/architecture/network/network.types").ParentMetrics`

Computes common parent metrics reused across helper functions.

Parameters:
- `parent1` - - First parent network.
- `parent2` - - Second parent network.
- `outputSize` - - Shared output size.

Returns: Parent metrics.

### selectHiddenNodeGene

`(nodeIndex: number, parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("src/architecture/network/network.types").ParentMetrics, equal: boolean, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/node").default | undefined`

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

`(nodeIndex: number, parent1: import("src/architecture/network/network.types").GeneticNetwork) => import("src/architecture/node").default | undefined`

Selects an input-region node gene.

Parameters:
- `nodeIndex` - - Slot index.
- `parent1` - - First parent.

Returns: Parent 1 input node gene.

### selectNodeGeneAtIndex

`(nodeIndex: number, offspringNodeCount: number, parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("src/architecture/network/network.types").ParentMetrics, equal: boolean, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/node").default | undefined`

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

`(nodeIndex: number, offspringNodeCount: number, parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("src/architecture/network/network.types").ParentMetrics, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/node").default | undefined`

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

`(parentNetwork1: import("src/architecture/network").default, parentNetwork2: import("src/architecture/network").default) => void`

Validates parent compatibility for crossover.

Parameters:
- `parentNetwork1` - - First parent candidate.
- `parentNetwork2` - - Second parent candidate.

Returns: Nothing.

## architecture/network/genetic/network.genetic.selection.utils.ts

### buildConnectionGene

`(connection: import("src/architecture/connection").default) => import("src/architecture/network/network.types").ConnectionGene | undefined`

Builds a connection gene from a concrete connection instance.

Parameters:
- `connection` - - Runtime connection.

Returns: Gene descriptor, or undefined when endpoints lack valid indices.

### canInheritParent1DisjointGenes

`(context: import("src/architecture/network/network.types").ConnectionGeneSelectionContext) => boolean`

Determines if parent-1 disjoint/excess genes are inheritable.

Parameters:
- `context` - - Selection context.

Returns: True when parent-1 disjoint genes can be selected.

### canInheritParent2DisjointGenes

`(context: import("src/architecture/network/network.types").ConnectionGeneSelectionContext) => boolean`

Determines if parent-2 disjoint/excess genes are inheritable.

Parameters:
- `context` - - Selection context.

Returns: True when parent-2 disjoint genes can be selected.

### chooseConnectionGenes

`(parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, parentMetrics: import("src/architecture/network/network.types").ParentMetrics, parent1Genes: Record<string, import("src/architecture/network/network.types").ConnectionGene>, parent2Genes: Record<string, import("src/architecture/network/network.types").ConnectionGene>, equal: boolean, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/network/network.types").ConnectionGene[]`

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

`(parent: import("src/architecture/network/network.types").GeneticNetwork, sourceGene: import("src/architecture/network/network.types").ConnectionGene) => import("src/architecture/network/network.types").ConnectionGene`

Chooses a disjoint/excess gene from a single parent.

Parameters:
- `parent` - - Source parent.
- `sourceGene` - - Source gene.

Returns: Selected disjoint gene.

### chooseMatchingGene

`(parent1: import("src/architecture/network/network.types").GeneticNetwork, parent2: import("src/architecture/network/network.types").GeneticNetwork, parent1Gene: import("src/architecture/network/network.types").ConnectionGene, parent2Gene: import("src/architecture/network/network.types").ConnectionGene, randomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/network/network.types").ConnectionGene`

Chooses a gene for matching innovation IDs.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parent1Gene` - - Parent 1 matching gene.
- `parent2Gene` - - Parent 2 matching gene.
- `randomGenerator` - - Random generator.

Returns: Selected gene.

### cloneConnectionGene

`(sourceGene: import("src/architecture/network/network.types").ConnectionGene) => import("src/architecture/network/network.types").ConnectionGene`

Clones a connection gene.

Parameters:
- `sourceGene` - - Source gene.

Returns: Independent clone.

### collectConnectionGenes

`(parent: import("src/architecture/network/network.types").GeneticNetwork) => Record<string, import("src/architecture/network/network.types").ConnectionGene>`

Collects all connection genes (standard + self) keyed by innovation ID.

Parameters:
- `parent` - - Parent network.

Returns: Innovation-keyed connection gene map.

### combineChosenGenes

`(parent1TraversalGenes: import("src/architecture/network/network.types").ConnectionGene[], parent2OnlyGenesToAppend: import("src/architecture/network/network.types").ConnectionGene[]) => import("src/architecture/network/network.types").ConnectionGene[]`

Combines selected gene partitions into one ordered list.

Parameters:
- `parent1TraversalGenes` - - Genes selected from parent-1 traversal.
- `parent2OnlyGenesToAppend` - - Parent-2-only genes.

Returns: Combined chosen genes.

### createParent1TraversalContexts

`(context: import("src/architecture/network/network.types").ConnectionGeneSelectionContext) => import("src/architecture/network/network.types").Parent1GeneTraversalContext[]`

Builds parent-1 traversal contexts keyed by innovation IDs.

Parameters:
- `context` - - Selection context.

Returns: Parent-1 traversal contexts.

### createSelectionContext

`(sourceParent1: import("src/architecture/network/network.types").GeneticNetwork, sourceParent2: import("src/architecture/network/network.types").GeneticNetwork, sourceParentMetrics: import("src/architecture/network/network.types").ParentMetrics, sourceParent1Genes: Record<string, import("src/architecture/network/network.types").ConnectionGene>, sourceParent2Genes: Record<string, import("src/architecture/network/network.types").ConnectionGene>, sourceEqual: boolean, sourceRandomGenerator: import("src/architecture/network/genetic/network.genetic.utils.types").RandomGenerator) => import("src/architecture/network/network.types").ConnectionGeneSelectionContext`

Creates the immutable context for this selection pass.

Parameters:
- `sourceParent1` - - First parent.
- `sourceParent2` - - Second parent.
- `sourceParentMetrics` - - Shared parent metrics.
- `sourceParent1Genes` - - Parent-1 genes.
- `sourceParent2Genes` - - Parent-2 genes.
- `sourceEqual` - - Equal-treatment flag.
- `sourceRandomGenerator` - - Random source.

Returns: Selection context.

### foldParent1TraversalContexts

`(traversalContexts: import("src/architecture/network/network.types").Parent1GeneTraversalContext[]) => import("src/architecture/network/network.types").Parent1TraversalSelectionResult`

Folds parent-1 traversal contexts into selected genes and consumed IDs.

Parameters:
- `traversalContexts` - - Parent-1 traversal contexts.

Returns: Parent-1 selection result.

### resolveReenableProbability

`(preferredProbability: number | undefined, fallbackProbability: number | undefined) => number`

Resolves re-enable probability with fallback to default value.

Parameters:
- `preferredProbability` - - Preferred parent probability.
- `fallbackProbability` - - Secondary parent probability.

Returns: Probability in [0, 1].

### selectGeneForParent1TraversalContext

`(traversalContext: import("src/architecture/network/network.types").Parent1GeneTraversalContext) => import("src/architecture/network/network.types").ConnectionGene | undefined`

Selects one inheritable gene for a parent-1 traversal context.

Parameters:
- `traversalContext` - - Parent-1 traversal context.

Returns: Selected gene or undefined.

### selectParent1TraversalGenes

`(context: import("src/architecture/network/network.types").ConnectionGeneSelectionContext) => import("src/architecture/network/network.types").Parent1TraversalSelectionResult`

Selects genes reachable from parent-1 innovation traversal.

Parameters:
- `context` - - Selection context.

Returns: Parent-1 traversal result.

### selectParent2OnlyGenes

`(context: import("src/architecture/network/network.types").ConnectionGeneSelectionContext, remainingParent2Genes: Record<string, import("src/architecture/network/network.types").ConnectionGene>) => import("src/architecture/network/network.types").ConnectionGene[]`

Selects inheritable parent-2-only disjoint/excess genes.

Parameters:
- `context` - - Selection context.
- `remainingParent2Genes` - - Parent-2-only gene map.

Returns: Selected parent-2-only genes.

### selectRemainingParent2Genes

`(context: import("src/architecture/network/network.types").ConnectionGeneSelectionContext, consumedInnovationIds: string[]) => Record<string, import("src/architecture/network/network.types").ConnectionGene>`

Builds parent-2 gene map after removing consumed matching innovations.

Parameters:
- `context` - - Selection context.
- `consumedInnovationIds` - - Innovation IDs already consumed via matching genes.

Returns: Remaining parent-2 genes.

## architecture/network/genetic/network.genetic.materialize.utils.ts

### applyConnectionGeneToConnection

`(connection: import("src/architecture/connection").default, connectionGene: import("src/architecture/network/network.types").ConnectionGene) => void`

Applies gene properties to a runtime connection.

Parameters:
- `connection` - - Runtime connection.
- `connectionGene` - - Gene source.

Returns: Nothing.

### attachGaterIfAvailable

`(offspring: import("src/architecture/network/network.types").GeneticNetwork, connection: import("src/architecture/connection").default, gaterIndex: number) => void`

Attaches a gater node when the target index is valid.

Parameters:
- `offspring` - - Offspring network.
- `connection` - - Connection to gate.
- `gaterIndex` - - Candidate gater node index.

Returns: Nothing.

### collectEligibleTraversalContexts

`(context: import("src/architecture/network/network.types").OffspringMaterializationContext, genes: import("src/architecture/network/network.types").ConnectionGene[]) => import("src/architecture/network/network.types").GeneTraversalContext[]`

Collects traversal contexts that satisfy all structural eligibility checks.

Parameters:
- `context` - - Top-level materialization context.
- `genes` - - Candidate genes.

Returns: Eligible traversal contexts.

### createConnectionForEndpoints

`(endpointsContext: import("src/architecture/network/network.types").GeneEndpointsContext) => import("src/architecture/connection").default | undefined`

Creates a runtime connection for endpoint nodes.

Parameters:
- `endpointsContext` - - Endpoint context.

Returns: Created connection or undefined.

### createMaterializationContext

`(targetOffspring: import("src/architecture/network/network.types").GeneticNetwork) => import("src/architecture/network/network.types").OffspringMaterializationContext`

Creates the immutable top-level context used during materialization.

Parameters:
- `targetOffspring` - - Offspring receiving concrete edges.

Returns: Materialization context.

### createOffspringConnection

`(offspring: import("src/architecture/network/network.types").GeneticNetwork, fromNode: import("src/architecture/node").default, toNode: import("src/architecture/node").default) => import("src/architecture/connection").default | undefined`

Creates a single offspring connection edge.

Parameters:
- `offspring` - - Offspring network.
- `fromNode` - - Source node.
- `toNode` - - Destination node.

Returns: Created connection or undefined.

### createTraversalContexts

`(context: import("src/architecture/network/network.types").OffspringMaterializationContext, genes: import("src/architecture/network/network.types").ConnectionGene[]) => import("src/architecture/network/network.types").GeneTraversalContext[]`

Builds traversal contexts for each candidate gene.

Parameters:
- `context` - - Top-level materialization context.
- `genes` - - Candidate genes.

Returns: Traversal contexts.

### hasExistingProjection

`(endpointsContext: import("src/architecture/network/network.types").GeneEndpointsContext) => boolean`

Checks whether the source endpoint already projects to the target endpoint.

Parameters:
- `endpointsContext` - - Endpoint context.

Returns: True when projection already exists.

### isTraversalContextFeedForward

`(traversalContext: import("src/architecture/network/network.types").GeneTraversalContext) => boolean`

Validates that a traversal context follows feed-forward ordering.

Parameters:
- `traversalContext` - - Traversal context.

Returns: True when the gene is strictly forward.

### isTraversalContextWithinNodeBounds

`(traversalContext: import("src/architecture/network/network.types").GeneTraversalContext) => boolean`

Validates that a traversal context endpoints are inside offspring bounds.

Parameters:
- `traversalContext` - - Traversal context.

Returns: True when both indices are bounded.

### keepFeedForwardTraversalContexts

`(traversalContexts: import("src/architecture/network/network.types").GeneTraversalContext[]) => import("src/architecture/network/network.types").GeneTraversalContext[]`

Keeps traversal contexts that preserve feed-forward edge direction.

Parameters:
- `traversalContexts` - - Node-bounded traversal contexts.

Returns: Feed-forward contexts.

### keepTraversalContextsWithinNodeBounds

`(traversalContexts: import("src/architecture/network/network.types").GeneTraversalContext[]) => import("src/architecture/network/network.types").GeneTraversalContext[]`

Keeps traversal contexts whose endpoints are inside offspring bounds.

Parameters:
- `traversalContexts` - - Candidate traversal contexts.

Returns: Node-bounded contexts.

### materializeOffspringConnections

`(offspring: import("src/architecture/network/network.types").GeneticNetwork, chosenGenes: import("src/architecture/network/network.types").ConnectionGene[]) => void`

Materializes selected connection genes in the offspring network.

Parameters:
- `offspring` - - Offspring network.
- `chosenGenes` - - Chosen connection genes.

Returns: Nothing.

### materializeSingleTraversalContext

`(traversalContext: import("src/architecture/network/network.types").GeneTraversalContext) => void`

Materializes one eligible traversal context when no duplicate projection exists.

Parameters:
- `traversalContext` - - Traversal context.

Returns: Nothing.

### materializeTraversalContexts

`(traversalContexts: import("src/architecture/network/network.types").GeneTraversalContext[]) => void`

Materializes each eligible traversal context independently.

Parameters:
- `traversalContexts` - - Eligible traversal contexts.

Returns: Nothing.

### resolveGeneEndpointsContext

`(traversalContext: import("src/architecture/network/network.types").GeneTraversalContext) => import("src/architecture/network/network.types").GeneEndpointsContext | undefined`

Resolves concrete endpoint nodes for a traversal context.

Parameters:
- `traversalContext` - - Traversal context.

Returns: Endpoint context or undefined.
