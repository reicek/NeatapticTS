# neat/multiobjective/metrics

Read-heavy Pareto metrics and archive access helpers.

This chapter owns the small helpers that summarize multi-objective state
after ranking has already happened: compact per-genome metrics,
reconstructed Pareto front views, bounded archive slices, and JSONL export
of archived objective vectors.

The boundary is intentionally read-side only. The ranking chapters decide
`_moRank`, `_moCrowd`, frontier membership, and archive contents earlier in
the evolve loop. This file exists for the next question: once that evidence
has been written onto genomes and archive arrays, how should inspection code
read it back in a compact, stable way?

The helpers fall into four small families:
- per-genome metrics for dashboards and quick diagnostics,
- front reconstruction from stored rank annotations,
- bounded archive slicing for recent-history views,
- JSONL export for downstream tooling.

Read this chapter when the missing question is "how do I inspect or export
the current multi-objective state without re-running ranking?" Read
`multiobjective/` or `archive/` first if the missing context is how that
state was produced.

## neat/multiobjective/metrics/multiobjective.metrics.ts

### buildMultiObjectiveMetrics

```ts
buildMultiObjectiveMetrics(
  population: default[],
): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Build lightweight multi-objective metrics for each genome in the population.

This helper turns the transient `_moRank` and `_moCrowd` annotations into a
compact read model that can be rendered directly in telemetry, debug tables,
or quick assertions. It deliberately mixes multi-objective evidence with a
few structural summary fields so callers can inspect competitive position and
genome size in one pass.

Parameters:
- `population` - - Ranked population with optional multi-objective
annotations.

Returns: Compact metrics aligned with the current population order.

### DEFAULT_MAX_PARETO_FRONTS

Default number of Pareto fronts returned by accessors.

Read helpers stay deliberately bounded by default so inspection callers get a
useful frontier summary without accidentally materializing every tail front.

### DEFAULT_PARETO_ARCHIVE_JSONL_MAX

Default slice size when exporting Pareto archive as JSONL.

Export uses a slightly larger default window than in-memory reads so offline
tooling can inspect a broader recent history without requiring the full
archive.

### DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES

Default slice size when reading Pareto archive entries.

This favors recent history, which is usually the most relevant window for
telemetry panels or interactive inspection.

### exportParetoArchiveJsonl

```ts
exportParetoArchiveJsonl(
  archive: unknown[],
  maxEntries: number,
): string
```

Export a Pareto archive slice as JSON Lines.

JSONL keeps each archived snapshot on its own line, which makes the output
easy to stream, diff, or feed into external tooling without inventing another
archive-specific export format.

Parameters:
- `archive` - - Archive collection ordered from oldest to newest.
- `maxEntries` - - Maximum number of recent entries to export.

Returns: Newline-delimited JSON for the selected archive window.

### reconstructParetoFronts

```ts
reconstructParetoFronts(
  population: default[],
  maxFronts: number,
  isMultiObjectiveEnabled: boolean,
): default[][]
```

Reconstruct Pareto fronts from stored rank annotations.

This helper is for read-time reconstruction, not ranking-time discovery.
Instead of rerunning dominance and crowding, it groups genomes by their
stored `_moRank` values and returns the leading fronts up to `maxFronts`.

When multi-objective mode is disabled, the function falls back to one front
containing the whole population so callers can keep a uniform read path.

Parameters:
- `population` - - Current population with optional `_moRank` annotations.
- `maxFronts` - - Maximum number of fronts to reconstruct.
- `isMultiObjectiveEnabled` - - Whether the controller is currently using
multi-objective ranking.

Returns: Reconstructed fronts in ascending rank order.

### sliceParetoArchive

```ts
sliceParetoArchive(
  archive: T[],
  maxEntries: number,
): T[]
```

Return the most recent Pareto archive entries up to the provided limit.

Archive reads are intentionally recent-first in spirit: callers usually want
the freshest frontier history rather than the earliest snapshots from a long
run. This helper therefore keeps the slicing rule explicit and reusable.

Parameters:
- `archive` - - Archive collection ordered from oldest to newest.
- `maxEntries` - - Maximum number of recent entries to keep.

Returns: A trailing slice containing at most `maxEntries` items.
