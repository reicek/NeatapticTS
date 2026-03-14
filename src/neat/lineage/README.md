# neat/lineage

Lineage and ancestry analysis helpers for NEAT populations.

The root lineage chapter keeps the two public ancestry metrics together while
moving the queue mechanics, sampling helpers, and narrow runtime types into
`core/`.

- `core/` explains ancestor traversal, sampled pair generation, and Jaccard-distance aggregation.

## neat/lineage/lineage.ts

### buildAnc

```ts
buildAnc(
  genome: GenomeLike,
): Set<number>
```

Build the shallow ancestor ID set for a genome using breadth-first traversal.

Parameters:
- `this` - - NEAT lineage context providing the current population.
- `genome` - - Genome whose shallow ancestor set should be computed.

Returns: Set of ancestor IDs within the configured depth window.

### computeAncestorUniqueness

```ts
computeAncestorUniqueness(): number
```

Compute the ancestor uniqueness metric for the current population.

Parameters:
- `this` - - NEAT lineage context exposing the population and RNG provider.

Returns: Mean sampled Jaccard distance across shallow ancestor sets.

### GenomeLike

Minimal genome shape used by lineage helpers.

Lineage analysis only needs the genome id and optional parent ids, so this
boundary intentionally leaves the rest of the genome open-ended.

### NeatLineageContext

Minimal NEAT context required by lineage helpers.

The lineage boundary only needs the current population and the RNG provider
used for sampled ancestor uniqueness.
