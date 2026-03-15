# neat/telemetry/facade

Public read-heavy facade helpers for Neat telemetry, objectives, and archive inspection.

This module groups the parts of the Neat surface that mainly expose existing
state rather than drive evolution. Keeping the public telemetry root beside
its chapter folders lets `src/neat.ts` stay orchestration-first while the
telemetry split now reads as one discoverable subtree.

## neat/telemetry/facade/telemetry.facade.ts

### clearParetoArchive

```ts
clearParetoArchive(
  host: NeatTelemetryFacadeHost,
): void
```

Clear the Pareto archive metadata stored on the host.

Parameters:
- `host` - - `Neat` instance whose Pareto archive should be emptied.

Returns: Nothing. The archive buffer is reset in place.

### clearTelemetry

```ts
clearTelemetry(
  host: NeatTelemetryFacadeHost,
): void
```

Clear cached telemetry entries.

Parameters:
- `host` - - `Neat` instance whose telemetry buffer should be reset.

Returns: Nothing. The helper mutates the host buffer in place.

### clearTelemetryObjectives

```ts
clearTelemetryObjectives(
  host: NeatTelemetryFacadeHost,
): void
```

Remove all registered custom objectives so only the default objective path remains.

Parameters:
- `host` - - `Neat` instance whose objective registry should be cleared.

Returns: Nothing. The helper mutates the objective registry in place.

### exportParetoFrontJSONL

```ts
exportParetoFrontJSONL(
  host: NeatTelemetryFacadeHost,
  maxEntries: number | undefined,
): string
```

Export recent Pareto archive entries as JSON Lines.

Parameters:
- `host` - - `Neat` instance storing Pareto objective snapshots.
- `maxEntries` - - Maximum number of entries to serialize.

Returns: JSONL payload for recent Pareto archive entries.

### exportSpeciesHistoryCSV

```ts
exportSpeciesHistoryCSV(
  host: NeatTelemetryFacadeHost,
  maxEntries: number,
): string
```

Export species history as CSV rows.

Parameters:
- `host` - - `Neat` instance whose species history should be exported.
- `maxEntries` - - Maximum number of recent history entries to include.

Returns: CSV payload for offline species analysis.

### exportSpeciesHistoryJSONL

```ts
exportSpeciesHistoryJSONL(
  host: NeatTelemetryFacadeHost,
  maxEntries: number,
): string
```

Export species history as JSON Lines.

Parameters:
- `host` - - `Neat` instance whose species history should be serialized.
- `maxEntries` - - Maximum number of recent history entries to include.

Returns: JSONL payload describing recent species history snapshots.

### exportTelemetryCSV

```ts
exportTelemetryCSV(
  host: NeatTelemetryFacadeHost,
  maxEntries: number,
): string
```

Export recent telemetry entries as CSV for quick spreadsheet inspection.

Parameters:
- `host` - - `Neat` instance whose telemetry buffer should be exported.
- `maxEntries` - - Maximum number of recent entries to include.

Returns: CSV string containing the requested telemetry window.

### exportTelemetryJSONL

```ts
exportTelemetryJSONL(
  host: NeatTelemetryFacadeHost,
): string
```

Export telemetry as JSON Lines so logs can stream into files or post-processors.

Parameters:
- `host` - - `Neat` instance whose telemetry buffer should be serialized.

Returns: JSONL payload with one telemetry object per line.

### getDiversityStats

```ts
getDiversityStats(
  host: NeatTelemetryFacadeHost,
): DiversityStats
```

Return cached diversity metrics, computing a fallback snapshot when needed.

This keeps the public facade resilient: callers can always ask for diversity
stats even before a full metrics pass has run.

Parameters:
- `host` - - `Neat` instance exposing cached diversity state.

Returns: Diversity metrics for the current population.

### getLineageSnapshot

```ts
getLineageSnapshot(
  host: NeatTelemetryFacadeHost,
  limit: number,
): { id: number; parents: number[]; }[]
```

Return a compact lineage sample for the first genomes in the current population.

This is meant for inspection and teaching, not for full genealogy export. It
keeps payloads small by clipping the returned slice while still showing which
genomes share parents.

Parameters:
- `host` - - `Neat` instance whose population lineage should be sampled.
- `limit` - - Maximum number of genomes to include in the snapshot.

Returns: Array of `{ id, parents }` lineage entries.

### getMultiObjectiveMetrics

```ts
getMultiObjectiveMetrics(
  host: NeatTelemetryFacadeHost,
): { rank: number; crowding: number; score: number; nodes: number; connections: number; }[]
```

Build compact multi-objective metrics for the current population snapshot.

Parameters:
- `host` - - `Neat` instance whose population should be summarized.

Returns: Rank, crowding, score, and size metrics per genome.

### getNoveltyArchiveSize

```ts
getNoveltyArchiveSize(
  host: NeatTelemetryFacadeHost,
): number
```

Return the current novelty archive size.

Parameters:
- `host` - - `Neat` instance tracking novelty behavior descriptors.

Returns: Number of archived novelty descriptors.

### getObjectiveEvents

```ts
getObjectiveEvents(
  host: NeatTelemetryFacadeHost,
): { gen: number; type: "add" | "remove"; key: string; }[]
```

Snapshot recent objective add/remove events for telemetry consumers.

Parameters:
- `host` - - `Neat` instance storing objective lifecycle events.

Returns: Shallow copy of the recorded objective events.

### getObjectiveKeys

```ts
getObjectiveKeys(
  host: NeatTelemetryFacadeHost,
): string[]
```

Return just the registered objective keys in stable order.

This is the shortest inspection surface for tests and quick diagnostics that
only need to confirm which objectives are active, not the full descriptor
payload.

Parameters:
- `host` - - `Neat` instance exposing objective descriptors.

Returns: Ordered list of active objective keys.

### getObjectives

```ts
getObjectives(
  host: NeatTelemetryFacadeHost,
): { key: string; direction: "max" | "min"; }[]
```

Return a compact view of active objective descriptors.

The full objective descriptor includes accessors and internal metadata. This
read model trims that down to the pieces most useful in UI surfaces and
debugging output: the key and whether the objective is minimized or
maximized.

Parameters:
- `host` - - `Neat` instance exposing objective descriptors.

Returns: Compact objective summaries in evaluation order.

### getOperatorStats

```ts
getOperatorStats(
  host: NeatTelemetryFacadeHost,
): { name: string; success: number; attempts: number; }[]
```

Return aggregated mutation/operator statistics.

Parameters:
- `host` - - `Neat` instance recording operator attempts and successes.

Returns: Operator summaries suitable for dashboards and debugging.

### getParetoArchive

```ts
getParetoArchive(
  host: NeatTelemetryFacadeHost,
  maxEntries: number | undefined,
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
  host: NeatTelemetryFacadeHost,
  maxFronts: number | undefined,
): default[][]
```

Reconstruct Pareto fronts from current rank annotations.

Parameters:
- `host` - - `Neat` instance whose population should be partitioned.
- `maxFronts` - - Maximum number of fronts to reconstruct.

Returns: Pareto fronts ordered from best to worst.

### getPerformanceStats

```ts
getPerformanceStats(
  host: NeatTelemetryFacadeHost,
): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Return coarse timing metrics for the last evaluation and evolution passes.

Parameters:
- `host` - - `Neat` instance tracking performance timings.

Returns: Snapshot of the last evaluation and evolution durations.

### getSpeciesHistory

```ts
getSpeciesHistory(
  host: NeatTelemetryFacadeHost,
): SpeciesHistoryEntry[]
```

Return recorded species history, lazily backfilling extended metrics when enabled.

Parameters:
- `host` - - `Neat` instance storing species history snapshots.

Returns: Historical species entries for each recorded generation.

### getSpeciesStats

```ts
getSpeciesStats(
  host: NeatTelemetryFacadeHost,
): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Return a concise summary for each current species.

Parameters:
- `host` - - `Neat` instance whose live species registry should be summarized.

Returns: Array of current species summaries.

### getTelemetry

```ts
getTelemetry(
  host: NeatTelemetryFacadeHost,
): TelemetryEntry[]
```

Return the in-memory telemetry buffer.

Parameters:
- `host` - - `Neat` instance storing generation telemetry snapshots.

Returns: Telemetry entries captured so far, or an empty array when telemetry
is not initialized.

### NeatTelemetryFacadeHost

Narrow `Neat` surface needed by the public telemetry, objective, and archive
facade methods.

This interface exists so the public `src/neat.ts` facade can delegate
read-heavy diagnostics and export helpers into one focused module without
exposing the entire controller implementation. The host shape keeps the
contract small: population snapshots, telemetry caches, objective accessors,
and archive buffers.

### registerTelemetryObjective

```ts
registerTelemetryObjective(
  host: NeatTelemetryFacadeHost,
  key: string,
  direction: "max" | "min",
  accessor: (genome: GenomeLike) => number,
): void
```

Register or replace a custom objective.

Parameters:
- `host` - - `Neat` instance whose multi-objective registry should change.
- `key` - - Unique objective key.
- `direction` - - Whether lower or higher values are considered better.
- `accessor` - - Function that reads the objective value from a genome.

Returns: Nothing. The objective registry on `host` is updated in place.

### resetNoveltyArchive

```ts
resetNoveltyArchive(
  host: NeatTelemetryFacadeHost,
): void
```

Clear the novelty archive.

Parameters:
- `host` - - `Neat` instance whose novelty archive should be reset.

Returns: Nothing. The archive is mutated in place.
