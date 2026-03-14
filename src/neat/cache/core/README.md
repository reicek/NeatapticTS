# neat/cache/core

Genome-owned cache fields that should be cleared when a mutation changes
structure or outputs.

These cache keys are grouped here so the cache chapter documents the
invalidation surface in one place.

## neat/cache/core/cache.core.ts

### invalidateGenomeCaches

```ts
invalidateGenomeCaches(
  genomeCandidate: unknown,
): void
```

Invalidate the derived caches attached to a genome candidate.

Mutation and crossover helpers attach memoized compatibility, activation,
and trace data directly onto genome objects for speed. Once the genome
changes, those values become stale. This helper centralizes that cleanup so
every mutation path clears the same cache fields.

Parameters:
- `genomeCandidate` - - Genome-shaped value whose attached caches should be cleared.

Returns: Nothing. The helper mutates the candidate in place when it is an object.

Example:

```ts
const genome = { _compatCache: {}, _outputCache: [1, 0] };
invalidateGenomeCaches(genome);
console.log('_compatCache' in genome, '_outputCache' in genome);
```

## neat/cache/core/cache.constants.ts

### GENOME_CACHE_FIELD_KEYS

Genome-owned cache fields that should be cleared when a mutation changes
structure or outputs.

These cache keys are grouped here so the cache chapter documents the
invalidation surface in one place.
