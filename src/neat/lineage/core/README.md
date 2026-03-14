# neat/lineage/core

Minimal genome shape used by lineage helpers.

Lineage analysis only needs the genome id and optional parent ids, so this
boundary intentionally leaves the rest of the genome open-ended.

## neat/lineage/core/lineage.types.ts

### AncestorQueueEntry

Queue entry used during breadth-first ancestor traversal.

### GenomeIndexPair

Index pair representing a sampled genome pair.

### GenomeLike

Minimal genome shape used by lineage helpers.

Lineage analysis only needs the genome id and optional parent ids, so this
boundary intentionally leaves the rest of the genome open-ended.

### NeatLineageContext

Minimal NEAT context required by lineage helpers.

The lineage boundary only needs the current population and the RNG provider
used for sampled ancestor uniqueness.

## neat/lineage/core/lineage.core.ts

Lineage-analysis mechanics used by NEAT ancestry helpers.

This chapter holds the breadth-first ancestor traversal, sampled pair
generation, and Jaccard-distance aggregation logic behind the public lineage
metrics.

### calculateMaxSamplePairs

```ts
calculateMaxSamplePairs(
  size: number,
): number
```

Compute the upper bound on sampled genome pairs.

Parameters:
- `size` - - Population size.

Returns: Sample cap respecting both the combinatorial count and the global limit.

### collectAncestorIds

```ts
collectAncestorIds(
  queueEntries: AncestorQueueEntry[],
  population: GenomeLike[],
): Set<number>
```

Collect ancestor IDs encountered within the configured depth window.

Parameters:
- `queueEntries` - - Breadth-first queue seeded with direct parents.
- `population` - - Current population for ID lookups.

Returns: Unique ancestor IDs encountered within the depth window.

### computeAverageDistance

```ts
computeAverageDistance(
  distances: number[],
): number
```

Compute the mean ancestor distance across sampled pairs.

Parameters:
- `distances` - - Pairwise Jaccard distances.

Returns: Mean distance rounded to the configured decimal precision.

### computePairDistances

```ts
computePairDistances(
  pairs: GenomeIndexPair[],
  population: GenomeLike[],
  buildAncestorSet: (genome: GenomeLike) => Set<number>,
): number[]
```

Compute Jaccard distances for sampled genome pairs.

Parameters:
- `pairs` - - Sampled index pairs.
- `population` - - Current population.
- `buildAncestorSet` - - Helper that builds ancestor sets for genomes.

Returns: Jaccard distances for valid pairs.

### createInitialQueue

```ts
createInitialQueue(
  parentIds: number[],
  population: GenomeLike[],
): AncestorQueueEntry[]
```

Create the initial breadth-first queue from direct parent IDs.

Parameters:
- `parentIds` - - Direct parent IDs to seed the queue.
- `population` - - Current population for ID lookups.

Returns: Queue entries at depth 1.

### hasMinimumPopulation

```ts
hasMinimumPopulation(
  size: number,
): boolean
```

Check whether the population is large enough to form a sampled pair.

Parameters:
- `size` - - Population size.

Returns: `true` when at least two genomes exist.

### normalizeParentIds

```ts
normalizeParentIds(
  value: GenomeLike,
): number[]
```

Normalize the parent ID list for a genome.

Parameters:
- `value` - - Genome to read parents from.

Returns: Parent ID list, or an empty array when absent.

### sampleGenomePairs

```ts
sampleGenomePairs(
  sampleCount: number,
  size: number,
  rngFactory: () => () => number,
): GenomeIndexPair[]
```

Sample genome index pairs for ancestor uniqueness.

Parameters:
- `sampleCount` - - Number of pairs to sample.
- `size` - - Population size for index bounds.
- `rngFactory` - - RNG provider used to obtain a random function.

Returns: Array of sampled index pairs.
