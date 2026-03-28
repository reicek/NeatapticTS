# architecture/network/genetic

Canonical threshold used for random binary parent/gene choice.

## architecture/network/genetic/network.genetic.utils.types.ts

### RANDOM_BINARY_SELECTION_THRESHOLD

Canonical threshold used for random binary parent/gene choice.

### DEFAULT_REENABLE_PROBABILITY

Default probability for re-enabling disabled genes during crossover.

### NO_GATER_INDEX

Sentinel index representing that no gater node is assigned.

### FIRST_INDEX

First element index used when reading newly created connections.

### PARENT_COMPATIBILITY_ERROR_MESSAGE

Shared compatibility error message for crossover parent validation.

### RandomGenerator

```ts
RandomGenerator(): number
```

Shared random generator signature for genetic operators.

## architecture/network/genetic/network.genetic.utils.ts

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

### crossOver

```ts
crossOver(
  parentNetwork1: default,
  parentNetwork2: default,
  equal: boolean,
): default
```

NEAT-inspired crossover between two parent networks producing a single offspring.

Conceptual model:
- A "gene" corresponds to either a node choice at a structural index or a connection
  keyed by innovation identity.
- The offspring is assembled in two phases: node assignment first, then connection
  materialization constrained by available offspring endpoints.
- Fitness controls inheritance pressure unless `equal` is enabled, in which case both
  parents contribute symmetrically where possible.

Simplifications relative to canonical NEAT:
 - Innovation ID is synthesized from (from.index, to.index) via Connection.innovationID instead of
   maintaining a global innovation number per mutation event.
 - Node alignment relies on current index ordering. This is weaker than historical innovation
   tracking, but adequate for many lightweight evolutionary experiments.

Compatibility assumptions:
- Both parents must expose identical input/output counts.
- Parent node index ordering should represent comparable structural positions.
- Parent fitness scores are interpreted by setup helpers when deciding fitter-parent inheritance.

High-level algorithm:
 1. Validate that parents have identical I/O dimensionality (required for compatibility).
 2. Decide offspring node array length:
      - If equal flag set or scores tied: random length in [minNodes, maxNodes].
      - Else: length of fitter parent.
 3. For each index up to chosen size, pick a node gene from parents per rules:
      - Input indices: always from parent1 (assumes identical input interface).
      - Output indices (aligned from end): randomly choose if both present else take existing.
      - Hidden indices: if both present pick randomly; else inherit from fitter (or either if equal).
 4. Reindex offspring nodes.
 5. Collect connections (standard + self) from each parent into maps keyed by innovationID capturing
    weight, enabled flag, and gater index.
 6. For overlapping genes (present in both), randomly choose one; if either disabled apply optional
    re-enable probability (reenableProb) to possibly re-activate.
 7. For disjoint/excess genes, inherit only from fitter parent (or both if equal flag set / scores tied).
 8. Materialize selected connection genes if their endpoints both exist in offspring; set weight & enabled state.
 9. Reattach gating if gater node exists in offspring.

Enabled reactivation probability:
 - Parents may carry disabled connections; offspring may re-enable them with a probability derived
   from parent-specific _reenableProb (or default 0.25). This allows dormant structures to resurface.

Parameters:
- `parentNetwork1` - - First parent (ties resolved in its favor when scores equal and equal=false for some cases).
- `parentNetwork2` - - Second parent.
- `equal` - - Force symmetric treatment regardless of fitness (true => node count random between sizes and both parents equally contribute disjoint genes).

Returns: Offspring network instance.

Example:

```ts
const offspring = crossOver(parentA, parentB);
offspring.mutate();
```

## architecture/network/genetic/network.genetic.setup.utils.ts

### createCrossoverContext

```ts
createCrossoverContext(
  parentNetwork1: default,
  parentNetwork2: default,
  equal: boolean,
): CrossoverContext
```

Creates the immutable crossover baseline context.

Parameters:
- `parentNetwork1` - - First parent network.
- `parentNetwork2` - - Second parent network.
- `equal` - - Equal-treatment mode flag.

Returns: Initialized crossover context.

### createNodeBuildContext

```ts
createNodeBuildContext(
  context: CrossoverContext,
): CrossoverNodeBuildContext
```

Creates the node-build context for offspring node selection.

Parameters:
- `context` - - Crossover baseline context.

Returns: Node-build context.

### assignOffspringNodes

```ts
assignOffspringNodes(
  nodeContext: CrossoverNodeBuildContext,
): void
```

Builds and reindexes offspring nodes.

Parameters:
- `nodeContext` - - Node-build context.

Returns: Nothing.

### chooseOffspringConnectionGenes

```ts
chooseOffspringConnectionGenes(
  context: CrossoverContext,
): ConnectionGene[]
```

Chooses all offspring connection genes from both parents.

Parameters:
- `context` - - Crossover baseline context.

Returns: Chosen connection genes.

### validateParentCompatibility

```ts
validateParentCompatibility(
  parentNetwork1: default,
  parentNetwork2: default,
): void
```

Validates parent compatibility for crossover.

Parameters:
- `parentNetwork1` - - First parent candidate.
- `parentNetwork2` - - Second parent candidate.

Returns: Nothing.

### asGeneticNetwork

```ts
asGeneticNetwork(
  network: default,
): GeneticNetwork
```

Coerces a network to the internal genetic runtime shape.

Parameters:
- `network` - - Source network.

Returns: Network with runtime genetic properties.

### createOffspringScaffold

```ts
createOffspringScaffold(
  inputSize: number,
  outputSize: number,
): GeneticNetwork
```

Creates an empty offspring scaffold with reset runtime arrays.

Parameters:
- `inputSize` - - Input count.
- `outputSize` - - Output count.

Returns: Initialized offspring runtime object.

### resolveParentMetrics

```ts
resolveParentMetrics(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  outputSize: number,
): ParentMetrics
```

Computes common parent metrics reused across helper functions.

Parameters:
- `parent1` - - First parent network.
- `parent2` - - Second parent network.
- `outputSize` - - Shared output size.

Returns: Parent metrics.

### getRandomGenerator

```ts
getRandomGenerator(
  parentNetwork: default,
): RandomGenerator
```

Resolves the random generator used by crossover decisions.

Parameters:
- `parentNetwork` - - Parent network that may provide a deterministic `_rand` source.

Returns: Random function.

### determineOffspringNodeCount

```ts
determineOffspringNodeCount(
  equal: boolean,
  parentMetrics: ParentMetrics,
  randomGenerator: RandomGenerator,
): number
```

Determines offspring node count from fitness/equality policy.

Parameters:
- `equal` - - Whether equal treatment mode is enabled.
- `parentMetrics` - - Parent metrics.
- `randomGenerator` - - Random generator.

Returns: Offspring node count.

### assignNodeIndexes

```ts
assignNodeIndexes(
  nodes: default[],
): void
```

Assigns contiguous indices to a node list.

Parameters:
- `nodes` - - Nodes to reindex.

Returns: Nothing.

### buildOffspringNodes

```ts
buildOffspringNodes(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  offspringNodeCount: number,
  equal: boolean,
  randomGenerator: RandomGenerator,
): default[]
```

Builds the offspring node list by selecting genes per slot.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `offspringNodeCount` - - Target offspring size.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Cloned offspring node genes.

### selectNodeGeneAtIndex

```ts
selectNodeGeneAtIndex(
  nodeIndex: number,
  offspringNodeCount: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
): default | undefined
```

Selects a node gene for a specific offspring slot.

Parameters:
- `nodeIndex` - - Slot index.
- `offspringNodeCount` - - Total offspring slots.
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Selected parent node gene, when present.

### selectInputNodeGene

```ts
selectInputNodeGene(
  nodeIndex: number,
  parent1: GeneticNetwork,
): default | undefined
```

Selects an input-region node gene.

Parameters:
- `nodeIndex` - - Slot index.
- `parent1` - - First parent.

Returns: Parent 1 input node gene.

### selectOutputNodeGene

```ts
selectOutputNodeGene(
  nodeIndex: number,
  offspringNodeCount: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  randomGenerator: RandomGenerator,
): default | undefined
```

Selects an output-region node gene using tail alignment.

Parameters:
- `nodeIndex` - - Slot index.
- `offspringNodeCount` - - Target offspring size.
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `randomGenerator` - - Random generator.

Returns: Selected output node gene.

### getAlignedOutputNode

```ts
getAlignedOutputNode(
  parent: GeneticNetwork,
  alignedIndex: number,
): default | undefined
```

Reads an aligned output candidate node if index is in the valid non-input range.

Parameters:
- `parent` - - Parent network.
- `alignedIndex` - - Tail-aligned index.

Returns: Output candidate node.

### selectHiddenNodeGene

```ts
selectHiddenNodeGene(
  nodeIndex: number,
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  equal: boolean,
  randomGenerator: RandomGenerator,
): default | undefined
```

Selects a hidden-region node gene.

Parameters:
- `nodeIndex` - - Slot index.
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parentMetrics` - - Parent metrics.
- `equal` - - Equal-treatment mode.
- `randomGenerator` - - Random generator.

Returns: Selected hidden node gene.

### cloneNodeGene

```ts
cloneNodeGene(
  sourceNode: default,
): default
```

Clones node structural gene attributes.

Parameters:
- `sourceNode` - - Source node gene.

Returns: Cloned node.

## architecture/network/genetic/network.genetic.selection.utils.ts

### collectConnectionGenes

```ts
collectConnectionGenes(
  parent: GeneticNetwork,
): Record<string, ConnectionGene>
```

Collects all connection genes (standard + self) keyed by innovation ID.

Parameters:
- `parent` - - Parent network.

Returns: Innovation-keyed connection gene map.

### chooseConnectionGenes

```ts
chooseConnectionGenes(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parentMetrics: ParentMetrics,
  parent1Genes: Record<string, ConnectionGene>,
  parent2Genes: Record<string, ConnectionGene>,
  equal: boolean,
  randomGenerator: RandomGenerator,
): ConnectionGene[]
```

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

### buildConnectionGene

```ts
buildConnectionGene(
  connection: default,
): ConnectionGene | undefined
```

Builds a connection gene from a concrete connection instance.

Parameters:
- `connection` - - Runtime connection.

Returns: Gene descriptor, or undefined when endpoints lack valid indices.

### createSelectionContext

```ts
createSelectionContext(
  sourceParent1: GeneticNetwork,
  sourceParent2: GeneticNetwork,
  sourceParentMetrics: ParentMetrics,
  sourceParent1Genes: Record<string, ConnectionGene>,
  sourceParent2Genes: Record<string, ConnectionGene>,
  sourceEqual: boolean,
  sourceRandomGenerator: RandomGenerator,
): ConnectionGeneSelectionContext
```

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

### selectParent1TraversalGenes

```ts
selectParent1TraversalGenes(
  context: ConnectionGeneSelectionContext,
): Parent1TraversalSelectionResult
```

Selects genes reachable from parent-1 innovation traversal.

Parameters:
- `context` - - Selection context.

Returns: Parent-1 traversal result.

### createParent1TraversalContexts

```ts
createParent1TraversalContexts(
  context: ConnectionGeneSelectionContext,
): Parent1GeneTraversalContext[]
```

Builds parent-1 traversal contexts keyed by innovation IDs.

Parameters:
- `context` - - Selection context.

Returns: Parent-1 traversal contexts.

### foldParent1TraversalContexts

```ts
foldParent1TraversalContexts(
  traversalContexts: Parent1GeneTraversalContext[],
): Parent1TraversalSelectionResult
```

Folds parent-1 traversal contexts into selected genes and consumed IDs.

Parameters:
- `traversalContexts` - - Parent-1 traversal contexts.

Returns: Parent-1 selection result.

### selectGeneForParent1TraversalContext

```ts
selectGeneForParent1TraversalContext(
  traversalContext: Parent1GeneTraversalContext,
): ConnectionGene | undefined
```

Selects one inheritable gene for a parent-1 traversal context.

Parameters:
- `traversalContext` - - Parent-1 traversal context.

Returns: Selected gene or undefined.

### selectRemainingParent2Genes

```ts
selectRemainingParent2Genes(
  context: ConnectionGeneSelectionContext,
  consumedInnovationIds: string[],
): Record<string, ConnectionGene>
```

Builds parent-2 gene map after removing consumed matching innovations.

Parameters:
- `context` - - Selection context.
- `consumedInnovationIds` - - Innovation IDs already consumed via matching genes.

Returns: Remaining parent-2 genes.

### selectParent2OnlyGenes

```ts
selectParent2OnlyGenes(
  context: ConnectionGeneSelectionContext,
  remainingParent2Genes: Record<string, ConnectionGene>,
): ConnectionGene[]
```

Selects inheritable parent-2-only disjoint/excess genes.

Parameters:
- `context` - - Selection context.
- `remainingParent2Genes` - - Parent-2-only gene map.

Returns: Selected parent-2-only genes.

### canInheritParent1DisjointGenes

```ts
canInheritParent1DisjointGenes(
  context: ConnectionGeneSelectionContext,
): boolean
```

Determines if parent-1 disjoint/excess genes are inheritable.

Parameters:
- `context` - - Selection context.

Returns: True when parent-1 disjoint genes can be selected.

### canInheritParent2DisjointGenes

```ts
canInheritParent2DisjointGenes(
  context: ConnectionGeneSelectionContext,
): boolean
```

Determines if parent-2 disjoint/excess genes are inheritable.

Parameters:
- `context` - - Selection context.

Returns: True when parent-2 disjoint genes can be selected.

### combineChosenGenes

```ts
combineChosenGenes(
  parent1TraversalGenes: ConnectionGene[],
  parent2OnlyGenesToAppend: ConnectionGene[],
): ConnectionGene[]
```

Combines selected gene partitions into one ordered list.

Parameters:
- `parent1TraversalGenes` - - Genes selected from parent-1 traversal.
- `parent2OnlyGenesToAppend` - - Parent-2-only genes.

Returns: Combined chosen genes.

### chooseMatchingGene

```ts
chooseMatchingGene(
  parent1: GeneticNetwork,
  parent2: GeneticNetwork,
  parent1Gene: ConnectionGene,
  parent2Gene: ConnectionGene,
  randomGenerator: RandomGenerator,
): ConnectionGene
```

Chooses a gene for matching innovation IDs.

Parameters:
- `parent1` - - First parent.
- `parent2` - - Second parent.
- `parent1Gene` - - Parent 1 matching gene.
- `parent2Gene` - - Parent 2 matching gene.
- `randomGenerator` - - Random generator.

Returns: Selected gene.

### chooseDisjointGeneFromParent

```ts
chooseDisjointGeneFromParent(
  parent: GeneticNetwork,
  sourceGene: ConnectionGene,
): ConnectionGene
```

Chooses a disjoint/excess gene from a single parent.

Parameters:
- `parent` - - Source parent.
- `sourceGene` - - Source gene.

Returns: Selected disjoint gene.

### cloneConnectionGene

```ts
cloneConnectionGene(
  sourceGene: ConnectionGene,
): ConnectionGene
```

Clones a connection gene.

Parameters:
- `sourceGene` - - Source gene.

Returns: Independent clone.

### resolveReenableProbability

```ts
resolveReenableProbability(
  preferredProbability: number | undefined,
  fallbackProbability: number | undefined,
): number
```

Resolves re-enable probability with fallback to default value.

Parameters:
- `preferredProbability` - - Preferred parent probability.
- `fallbackProbability` - - Secondary parent probability.

Returns: Probability in [0, 1].

## architecture/network/genetic/network.genetic.materialize.utils.ts

### materializeOffspringConnections

```ts
materializeOffspringConnections(
  offspring: GeneticNetwork,
  chosenGenes: ConnectionGene[],
): void
```

Materializes selected connection genes in the offspring network.

Parameters:
- `offspring` - - Offspring network.
- `chosenGenes` - - Chosen connection genes.

Returns: Nothing.

### createMaterializationContext

```ts
createMaterializationContext(
  targetOffspring: GeneticNetwork,
): OffspringMaterializationContext
```

Creates the immutable top-level context used during materialization.

Parameters:
- `targetOffspring` - - Offspring receiving concrete edges.

Returns: Materialization context.

### collectEligibleTraversalContexts

```ts
collectEligibleTraversalContexts(
  context: OffspringMaterializationContext,
  genes: ConnectionGene[],
): GeneTraversalContext[]
```

Collects traversal contexts that satisfy all structural eligibility checks.

Parameters:
- `context` - - Top-level materialization context.
- `genes` - - Candidate genes.

Returns: Eligible traversal contexts.

### createTraversalContexts

```ts
createTraversalContexts(
  context: OffspringMaterializationContext,
  genes: ConnectionGene[],
): GeneTraversalContext[]
```

Builds traversal contexts for each candidate gene.

Parameters:
- `context` - - Top-level materialization context.
- `genes` - - Candidate genes.

Returns: Traversal contexts.

### keepTraversalContextsWithinNodeBounds

```ts
keepTraversalContextsWithinNodeBounds(
  traversalContexts: GeneTraversalContext[],
): GeneTraversalContext[]
```

Keeps traversal contexts whose endpoints are inside offspring bounds.

Parameters:
- `traversalContexts` - - Candidate traversal contexts.

Returns: Node-bounded contexts.

### keepFeedForwardTraversalContexts

```ts
keepFeedForwardTraversalContexts(
  traversalContexts: GeneTraversalContext[],
): GeneTraversalContext[]
```

Keeps traversal contexts that preserve feed-forward edge direction.

Parameters:
- `traversalContexts` - - Node-bounded traversal contexts.

Returns: Feed-forward contexts.

### materializeTraversalContexts

```ts
materializeTraversalContexts(
  traversalContexts: GeneTraversalContext[],
): void
```

Materializes each eligible traversal context independently.

Parameters:
- `traversalContexts` - - Eligible traversal contexts.

Returns: Nothing.

### materializeSingleTraversalContext

```ts
materializeSingleTraversalContext(
  traversalContext: GeneTraversalContext,
): void
```

Materializes one eligible traversal context when no duplicate projection exists.

Parameters:
- `traversalContext` - - Traversal context.

Returns: Nothing.

### resolveGeneEndpointsContext

```ts
resolveGeneEndpointsContext(
  traversalContext: GeneTraversalContext,
): GeneEndpointsContext | undefined
```

Resolves concrete endpoint nodes for a traversal context.

Parameters:
- `traversalContext` - - Traversal context.

Returns: Endpoint context or undefined.

### createConnectionForEndpoints

```ts
createConnectionForEndpoints(
  endpointsContext: GeneEndpointsContext,
): default | undefined
```

Creates a runtime connection for endpoint nodes.

Parameters:
- `endpointsContext` - - Endpoint context.

Returns: Created connection or undefined.

### hasExistingProjection

```ts
hasExistingProjection(
  endpointsContext: GeneEndpointsContext,
): boolean
```

Checks whether the source endpoint already projects to the target endpoint.

Parameters:
- `endpointsContext` - - Endpoint context.

Returns: True when projection already exists.

### isTraversalContextWithinNodeBounds

```ts
isTraversalContextWithinNodeBounds(
  traversalContext: GeneTraversalContext,
): boolean
```

Validates that a traversal context endpoints are inside offspring bounds.

Parameters:
- `traversalContext` - - Traversal context.

Returns: True when both indices are bounded.

### isTraversalContextFeedForward

```ts
isTraversalContextFeedForward(
  traversalContext: GeneTraversalContext,
): boolean
```

Validates that a traversal context follows feed-forward ordering.

Parameters:
- `traversalContext` - - Traversal context.

Returns: True when the gene is strictly forward.

### createOffspringConnection

```ts
createOffspringConnection(
  offspring: GeneticNetwork,
  fromNode: default,
  toNode: default,
): default | undefined
```

Creates a single offspring connection edge.

Parameters:
- `offspring` - - Offspring network.
- `fromNode` - - Source node.
- `toNode` - - Destination node.

Returns: Created connection or undefined.

### applyConnectionGeneToConnection

```ts
applyConnectionGeneToConnection(
  connection: default,
  connectionGene: ConnectionGene,
): void
```

Applies gene properties to a runtime connection.

Parameters:
- `connection` - - Runtime connection.
- `connectionGene` - - Gene source.

Returns: Nothing.

### attachGaterIfAvailable

```ts
attachGaterIfAvailable(
  offspring: GeneticNetwork,
  connection: default,
  gaterIndex: number,
): void
```

Attaches a gater node when the target index is valid.

Parameters:
- `offspring` - - Offspring network.
- `connection` - - Connection to gate.
- `gaterIndex` - - Candidate gater node index.

Returns: Nothing.
