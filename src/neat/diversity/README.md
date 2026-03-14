# neat/diversity

Diversity-reporting helpers for NEAT populations.

This root diversity chapter stays compact on purpose: it surfaces the two
public read models first, then points readers to `core/` for the sampled
aggregation mechanics and narrow telemetry types.

- `core/` explains the sampling limits, structural entropy helpers, and the
  population metrics used by telemetry and diagnostics.

## neat/diversity/diversity.ts

### computeDiversityStats

```ts
computeDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined
```

Compute sampled diversity statistics for a NEAT population.

The helper intentionally samples pairwise lineage and compatibility work so
large populations can still produce telemetry without quadratic blowups.

Parameters:
- `population` - - Population genomes exposing nodes, connections, and optional lineage depth.
- `compatibilityComputer` - - Compatibility-distance provider used for pair sampling.

Returns: Aggregate diversity statistics or `undefined` when the population is empty.

### DiversityStats

Diversity statistics returned by sampled population analysis.

Each field captures one aggregate lens on the current population: lineage
spread, structural size, compatibility separation, or entropy.

### MAX_COMPATIBILITY_SAMPLE

Maximum population sample size for compatibility comparisons.

### MAX_LINEAGE_PAIR_SAMPLE

Maximum lineage sample size for pairwise depth comparisons.

### structuralEntropy

```ts
structuralEntropy(
  graph: default,
): number
```

Compute the Shannon-style entropy of a network's out-degree distribution.

Structural entropy here is a lightweight topology fingerprint: it measures
how evenly outgoing connections are distributed across nodes. It does not
inspect weights or recurrent dynamics, so it works well as a cheap structural
diversity signal.

Parameters:
- `graph` - - Network to summarize structurally.

Returns: Shannon-style entropy of the out-degree distribution.
