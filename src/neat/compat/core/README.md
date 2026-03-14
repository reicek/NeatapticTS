# neat/compat/core

Shape of a connection entry used during compatibility checks.

The compatibility helpers only need endpoint indices, an optional
innovation number, and the connection weight. Keeping this shape small makes
the distance chapter easier to reuse in tests and controller helpers.

## neat/compat/core/compat.types.ts

### ComparisonMetrics

Aggregated comparison metrics for compatibility calculations.

### ConnectionLike

Shape of a connection entry used during compatibility checks.

The compatibility helpers only need endpoint indices, an optional
innovation number, and the connection weight. Keeping this shape small makes
the distance chapter easier to reuse in tests and controller helpers.

### GenomeLike

Minimal genome shape used for compatibility distance calculations.

The `_compatCache` stores sorted `[innovation, weight]` pairs so repeated
comparisons within a generation can reuse the same derived list.

### NeatLikeForCompat

Minimal NEAT context required by compatibility helpers.

This keeps the boundary tightly focused on generation-scoped caches,
compatibility coefficients, and the fallback innovation resolver.

## neat/compat/core/compat.core.ts

Compatibility-distance mechanics used by NEAT speciation.

This chapter holds the reusable inner loop: generation cache setup,
innovation-list caching, linear list comparison, and the final distance
calculation.

### buildPairKey

```ts
buildPairKey(
  firstGenome: GenomeLike,
  secondGenome: GenomeLike,
): string
```

Build a stable cache key for a genome pair.

Parameters:
- `firstGenome` - - First genome in the pair.
- `secondGenome` - - Second genome in the pair.

Returns: Stable cache key in the form `minId|maxId`.

### compareInnovationLists

```ts
compareInnovationLists(
  firstList: [number, number][],
  secondList: [number, number][],
): ComparisonMetrics
```

Compare two sorted innovation lists and derive compatibility metrics.

Parameters:
- `firstList` - - Sorted innovation list for the first genome.
- `secondList` - - Sorted innovation list for the second genome.

Returns: Aggregated comparison metrics for distance computation.

### computeCompatibilityDistance

```ts
computeCompatibilityDistance(
  neatContext: NeatLikeForCompat,
  metrics: ComparisonMetrics,
): number
```

Compute the compatibility distance from precomputed metrics.

Parameters:
- `neatContext` - - NEAT context providing compatibility coefficients.
- `metrics` - - Aggregated comparison metrics.

Returns: Final compatibility distance for the genome pair.

### ensureGenerationCache

```ts
ensureGenerationCache(
  neatContext: NeatLikeForCompat,
): void
```

Ensure generation-scoped compatibility caches exist.

Parameters:
- `neatContext` - - Current NEAT context holding generation and caches.

Returns: Nothing. The helper resets caches when the generation changes.

### getDistanceCacheMap

```ts
getDistanceCacheMap(
  neatContext: NeatLikeForCompat,
): Map<string, number>
```

Retrieve the generation-scoped cache map for pairwise distances.

Parameters:
- `neatContext` - - Current NEAT context with the cache map.

Returns: Map storing cached distances for genome pairs this generation.

### getSortedInnovationCache

```ts
getSortedInnovationCache(
  neatContext: NeatLikeForCompat,
  genome: GenomeLike,
): [number, number][]
```

Retrieve or build a sorted innovation list for a genome.

Parameters:
- `neatContext` - - NEAT context used for fallback innovation numbers.
- `genome` - - Genome to derive a sorted innovation list for.

Returns: Array of `[innovationNumber, weight]` sorted by innovation number.

### resolveMaxInnovation

```ts
resolveMaxInnovation(
  list: [number, number][],
): number
```

Resolve the highest innovation id from a sorted list.

Parameters:
- `list` - - Sorted innovation list for a genome.

Returns: Highest innovation id or `0` when the list is empty.
