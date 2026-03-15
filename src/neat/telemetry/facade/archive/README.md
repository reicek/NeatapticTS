# neat/telemetry/facade/archive

## neat/telemetry/facade/archive/telemetry.facade.archive.ts

### clearParetoArchive

```ts
clearParetoArchive(
  host: TelemetryFacadeArchiveHost,
): void
```

Clear the Pareto archive metadata stored on the host.

Parameters:
- `host` - - `Neat` instance whose Pareto archive should be emptied.

Returns: Nothing. The archive buffer is reset in place.

### exportParetoFrontJSONL

```ts
exportParetoFrontJSONL(
  host: TelemetryFacadeArchiveHost,
  maxEntries: number,
): string
```

Export recent Pareto archive entries as JSON Lines.

Parameters:
- `host` - - `Neat` instance storing Pareto objective snapshots.
- `maxEntries` - - Maximum number of entries to serialize.

Returns: JSONL payload for recent Pareto archive entries.

### getMultiObjectiveMetrics

```ts
getMultiObjectiveMetrics(
  host: TelemetryFacadeArchiveHost,
): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Build compact multi-objective metrics for the current population snapshot.

This chapter keeps the highest-level Pareto inspection helpers together so a
caller can move from per-genome rank summaries to reconstructed fronts and
archived vectors without leaving the same conceptual boundary.

Parameters:
- `host` - - `Neat` instance whose population should be summarized.

Returns: Rank, crowding, score, and size metrics per genome.

### getParetoArchive

```ts
getParetoArchive(
  host: TelemetryFacadeArchiveHost,
  maxEntries: number,
): ParetoArchiveEntry[]
```

Return the most recent Pareto archive entries.

Parameters:
- `host` - - `Neat` instance storing archived Pareto metadata.
- `maxEntries` - - Maximum number of archive entries to return.

Returns: Slice of the recent Pareto archive.

### getParetoFronts

```ts
getParetoFronts(
  host: TelemetryFacadeArchiveHost,
  maxFronts: number,
): default[][]
```

Reconstruct Pareto fronts from current rank annotations.

Parameters:
- `host` - - `Neat` instance whose population should be partitioned.
- `maxFronts` - - Maximum number of fronts to reconstruct.

Returns: Pareto fronts ordered from best to worst.

### TelemetryFacadeArchiveHost

Narrow telemetry-facade host surface required by the archive chapter.

This chapter groups the public multi-objective inspection helpers so the
root telemetry facade can treat Pareto fronts, archive snapshots, and their
compact derived summaries as one concept cluster.
