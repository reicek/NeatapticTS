# neat/telemetry/facade/archive

Pareto-front and archive inspection helpers inside the public telemetry facade.

This chapter is the multi-objective read-side companion to the broader
telemetry facade. Once a run starts ranking genomes by Pareto dominance,
callers usually need one of three views:

- a compact per-genome metrics table for quick inspection,
- reconstructed live fronts from the current population,
- archived objective snapshots that can be exported or reviewed later.

The helpers stay together because those views answer the same practical
question from different distances: what tradeoff structure is the controller
currently seeing, and what evidence has it kept around from previous steps?

Read this chapter after the root telemetry facade when the remaining question
is specifically about multi-objective ranking. The root surface shows where
objective inspection lives overall; this file narrows that map to the
archive- and Pareto-oriented read helpers.

```mermaid
flowchart TD
  Population[Current population] --> Metrics[getMultiObjectiveMetrics()<br/>compact per-genome view]
  Population --> Fronts[getParetoFronts()<br/>live reconstructed fronts]
  Archive[Stored Pareto archive] --> Slice[getParetoArchive()<br/>recent snapshots]
  Slice --> Export[exportParetoFrontJSONL()<br/>portable archive export]
  Archive --> Clear[clearParetoArchive()<br/>reset archive state]
```

## neat/telemetry/facade/archive/telemetry.facade.archive.ts

### clearParetoArchive

```ts
clearParetoArchive(
  host: TelemetryFacadeArchiveHost,
): void
```

Clear the Pareto archive metadata stored on the host.

Reach for this when a caller wants a fresh archive observation window
without resetting the rest of the telemetry system.

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

Prefer this when archive inspection is leaving the process boundary. JSONL is
easy to append to files, load into notebooks, or post-process with simple
scripts while preserving one archived snapshot per line.

Parameters:
- `host` - - `Neat` instance storing Pareto objective snapshots.
- `maxEntries` - - Maximum number of entries to serialize.

Returns: JSONL payload for recent Pareto archive entries.

Example:

```ts
const archiveJsonl = exportParetoFrontJSONL(neat, 100);
console.log(archiveJsonl.split('\n').at(0));
```

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

Example:

```ts
const metrics = getMultiObjectiveMetrics(neat);
console.table(metrics.slice(0, 5));
```

### getParetoArchive

```ts
getParetoArchive(
  host: TelemetryFacadeArchiveHost,
  maxEntries: number,
): ParetoArchiveEntry[]
```

Return the most recent Pareto archive entries.

This is the historical companion to {@link getParetoFronts}. Instead of
reconstructing the current live fronts, it slices the archive the controller
has already decided to retain for later inspection or export.

Parameters:
- `host` - - `Neat` instance storing archived Pareto metadata.
- `maxEntries` - - Maximum number of archive entries to return.

Returns: Slice of the recent Pareto archive.

Example:

```ts
const recentArchive = getParetoArchive(neat, 25);
console.log(recentArchive.length);
```

### getParetoFronts

```ts
getParetoFronts(
  host: TelemetryFacadeArchiveHost,
  maxFronts: number,
): default[][]
```

Reconstruct Pareto fronts from current rank annotations.

Use this when you want the live frontier grouping itself rather than a flat
metrics table. The helper rebuilds the front structure from the population's
current multi-objective annotations, which makes it useful for dashboards,
tests, or teaching material that needs to show how genomes separate into
dominance layers.

Parameters:
- `host` - - `Neat` instance whose population should be partitioned.
- `maxFronts` - - Maximum number of fronts to reconstruct.

Returns: Pareto fronts ordered from best to worst.

Example:

```ts
const fronts = getParetoFronts(neat, 3);
console.log(fronts.map((front) => front.length));
```

### TelemetryFacadeArchiveHost

Narrow telemetry-facade host surface required by the archive chapter.

This chapter groups the public multi-objective inspection helpers so the
root telemetry facade can treat Pareto fronts, archive snapshots, and their
compact derived summaries as one concept cluster.

The host contract combines the live population with the stored Pareto
archive because callers often need to compare current fronts against the
snapshots that survived earlier generations.
