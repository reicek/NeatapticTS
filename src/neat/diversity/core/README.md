# neat/diversity/core

Minimal node interface used by diversity computations.

Diversity helpers only need the outgoing connection count, so this type keeps
the telemetry boundary narrower than the full runtime node model.

## neat/diversity/core/diversity.types.ts

### CompatComputer

Minimal interface for computing compatibility distance between genomes.

The narrow contract keeps diversity telemetry independent from the broader
NEAT controller implementation.

### DiversityStats

Diversity statistics returned by sampled population analysis.

Each field captures one aggregate lens on the current population: lineage
spread, structural size, compatibility separation, or entropy.

### GenomeWithMetrics

Minimal genome shape used by diversity computations.

The diversity chapter focuses on structural size, lineage depth, and sampled
compatibility, so only those fields are modeled here.

### NodeWithConnections

Minimal node interface used by diversity computations.

Diversity helpers only need the outgoing connection count, so this type keeps
the telemetry boundary narrower than the full runtime node model.

## neat/diversity/core/diversity.core.ts

Diversity-statistics mechanics used by telemetry and diagnostics.

This chapter holds the reusable folds behind population diversity reports:
structural entropy, sampled lineage distance, sampled compatibility, and the
small numeric helpers used to aggregate those signals.

### calculateDiversityStats

```ts
calculateDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined
```

Compute diversity statistics for a NEAT population.

Parameters:
- `population` - - Population genomes exposing nodes, connections, and optional `_depth`.
- `compatibilityComputer` - - Object exposing `_compatibilityDistance(a, b)`.

Returns: Diversity report or `undefined` when the population is empty.

### calculateStructuralEntropy

```ts
calculateStructuralEntropy(
  graph: default,
): number
```

Compute the Shannon-style entropy of a network's out-degree distribution.

Parameters:
- `graph` - - Network instance to evaluate.

Returns: Shannon-style entropy value.

### computeMeanAbsolutePairDistance

```ts
computeMeanAbsolutePairDistance(
  values: number[],
  sampleLimit: number,
): number
```

Compute the mean absolute pairwise distance across a sampled value list.

Parameters:
- `values` - - Values to compare.
- `sampleLimit` - - Maximum number of sampled values to include.

Returns: Mean absolute pair distance across the sampled values.

### computeMeanCompatibilityDistance

```ts
computeMeanCompatibilityDistance(
  genomes: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
  sampleLimit: number,
): number
```

Compute the mean compatibility distance across sampled genome pairs.

Parameters:
- `genomes` - - Population genomes to compare.
- `compatibilityComputer` - - Compatibility-distance provider.
- `sampleLimit` - - Maximum number of genomes to include.

Returns: Mean compatibility distance across the sampled pairs.

### MAX_COMPATIBILITY_SAMPLE

Maximum population sample size for compatibility comparisons.

### MAX_LINEAGE_PAIR_SAMPLE

Maximum lineage sample size for pairwise depth comparisons.

### mean

```ts
mean(
  values: number[],
): number
```

Compute the arithmetic mean of a numeric array.

Parameters:
- `values` - - Values to average.

Returns: Arithmetic mean, or `0` when the array is empty.

### variance

```ts
variance(
  values: number[],
): number
```

Compute the population variance of a numeric array.

Parameters:
- `values` - - Values to evaluate.

Returns: Population variance, or `0` when the array is empty.
