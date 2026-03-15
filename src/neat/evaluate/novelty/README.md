# neat/evaluate/novelty

Novelty-search helpers for the NEAT evaluate chapter.

This chapter keeps the behavior-descriptor path together: collect
descriptors, compute pairwise distances, score novelty against nearby
neighbors, blend novelty into the current fitness score, and append archive
entries when they beat the configured threshold.

## neat/evaluate/novelty/evaluate.novelty.ts

### addGenomeToNoveltyArchive

```ts
addGenomeToNoveltyArchive(
  controller: NeatControllerForEval,
  descriptor: number[],
  novelty: number,
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): void
```

Add a descriptor to the novelty archive when it exceeds the threshold.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `descriptor` - - Behavior descriptor for the current genome.
- `novelty` - - Computed novelty score.
- `noveltyOptions` - - Novelty configuration.

### applyNoveltyToPopulation

```ts
applyNoveltyToPopulation(
  controller: NeatControllerForEval,
  descriptors: number[][],
  distanceMatrix: number[][],
  kNeighbors: number,
  blendFactor: number,
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): void
```

Apply novelty scores and archive writes across the population.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `descriptors` - - Descriptor vectors for each genome.
- `distanceMatrix` - - Dense distance matrix.
- `kNeighbors` - - Number of nearest neighbors to average.
- `blendFactor` - - Novelty-vs-fitness blend factor.
- `noveltyOptions` - - Novelty configuration.

### blendNoveltyIntoScore

```ts
blendNoveltyIntoScore(
  genome: GenomeForEvaluation,
  novelty: number,
  blendFactor: number,
): void
```

Blend novelty into the genome score when the score already exists.

Parameters:
- `genome` - - Genome to update.
- `novelty` - - Computed novelty value.
- `blendFactor` - - Blend factor for novelty versus fitness.

### buildDistanceMatrix

```ts
buildDistanceMatrix(
  descriptors: number[][],
): number[][]
```

Build the full pairwise distance matrix for the descriptor set.

Parameters:
- `descriptors` - - Descriptor vectors for the current population.

Returns: Dense distance matrix aligned with population order.

### buildNoveltyDescriptors

```ts
buildNoveltyDescriptors(
  controller: NeatControllerForEval,
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): number[][]
```

Build behavior descriptors for the current population.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `noveltyOptions` - - Novelty configuration.

Returns: Descriptor vectors aligned with population order.

### computeDescriptorDistance

```ts
computeDescriptorDistance(
  leftDescriptor: number[],
  rightDescriptor: number[],
  isSame: boolean,
): number
```

Compute the Euclidean distance between two descriptors.

Parameters:
- `leftDescriptor` - - Left descriptor vector.
- `rightDescriptor` - - Right descriptor vector.
- `isSame` - - Whether both descriptors belong to the same genome index.

Returns: Euclidean distance across the shared prefix.

### computeNoveltyScore

```ts
computeNoveltyScore(
  distanceRow: number[],
  kNeighbors: number,
): number
```

Compute a novelty score from one row of the distance matrix.

Parameters:
- `distanceRow` - - Distance values for a single genome.
- `kNeighbors` - - Number of nearest neighbors to average.

Returns: Novelty score for the genome.

### getNoveltyBlendFactor

```ts
getNoveltyBlendFactor(
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): number
```

Resolve the novelty-vs-fitness blend factor.

Parameters:
- `noveltyOptions` - - Novelty configuration.

Returns: Blend factor used when a genome already has a numeric score.

### getNoveltyNeighborCount

```ts
getNoveltyNeighborCount(
  noveltyOptions: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; },
): number
```

Resolve the number of nearest neighbors used for novelty scoring.

Parameters:
- `noveltyOptions` - - Novelty configuration.

Returns: Neighbor count clamped to at least one.

### runNoveltyBlendAndArchive

```ts
runNoveltyBlendAndArchive(
  controller: NeatControllerForEval,
  evaluationOptions: { [key: string]: unknown; fitnessPopulation?: boolean | undefined; clear?: boolean | undefined; novelty?: { enabled?: boolean | undefined; descriptor?: ((genome: GenomeForEvaluation) => number[]) | undefined; k?: number | undefined; blendFactor?: number | undefined; archiveAddThreshold?: number | undefined; } | undefined; entropySharingTuning?: { enabled?: boolean | undefined; targetEntropyVar?: number | undefined; adjustRate?: number | undefined; minSigma?: number | undefined; maxSigma?: number | undefined; } | undefined; entropyCompatTuning?: { enabled?: boolean | undefined; targetEntropy?: number | undefined; deadband?: number | undefined; adjustRate?: number | undefined; minThreshold?: number | undefined; maxThreshold?: number | undefined; } | undefined; autoDistanceCoeffTuning?: { enabled?: boolean | undefined; adjustRate?: number | undefined; minCoeff?: number | undefined; maxCoeff?: number | undefined; } | undefined; multiObjective?: { enabled?: boolean | undefined; autoEntropy?: boolean | undefined; dynamic?: { enabled?: boolean | undefined; } | undefined; } | undefined; speciation?: boolean | undefined; targetSpecies?: number | undefined; compatAdjust?: boolean | undefined; speciesAllocation?: { extendedHistory?: boolean | undefined; } | undefined; sharingSigma?: number | undefined; compatibilityThreshold?: number | undefined; excessCoeff?: number | undefined; disjointCoeff?: number | undefined; },
): void
```

Compute novelty, blend it into scores, and update the novelty archive.

Parameters:
- `controller` - - NEAT controller instance for evaluation.
- `evaluationOptions` - - Options object for the current evaluation pass.
