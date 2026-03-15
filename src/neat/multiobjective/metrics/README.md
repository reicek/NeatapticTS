# neat/multiobjective/metrics

Read-heavy Pareto metrics and archive access helpers.

This chapter owns the small helpers that summarize multi-objective state
after ranking has already happened: compact per-genome metrics,
reconstructed Pareto front views, bounded archive slices, and JSONL export
of archived objective vectors.

## neat/multiobjective/metrics/multiobjective.metrics.ts

### buildMultiObjectiveMetrics

```ts
buildMultiObjectiveMetrics(
  population: default[],
): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Build lightweight multi-objective metrics for each genome in the population.

### DEFAULT_MAX_PARETO_FRONTS

Default number of Pareto fronts returned by accessors.

### DEFAULT_PARETO_ARCHIVE_JSONL_MAX

Default slice size when exporting Pareto archive as JSONL.

### DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES

Default slice size when reading Pareto archive entries.

### exportParetoArchiveJsonl

```ts
exportParetoArchiveJsonl(
  archive: unknown[],
  maxEntries: number,
): string
```

Export a Pareto archive slice as JSON Lines.

### reconstructParetoFronts

```ts
reconstructParetoFronts(
  population: default[],
  maxFronts: number,
  isMultiObjectiveEnabled: boolean,
): default[][]
```

Reconstruct Pareto fronts from stored rank annotations.

### sliceParetoArchive

```ts
sliceParetoArchive(
  archive: T[],
  maxEntries: number,
): T[]
```

Return the most recent Pareto archive entries up to the provided limit.
