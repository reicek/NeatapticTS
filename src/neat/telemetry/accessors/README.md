# neat/telemetry/accessors

Shared telemetry accessors used by both the internal telemetry chapters and
the already-folderized public telemetry facade.

This chapter keeps the smallest read-only host helpers together: buffer
reads, objective-event snapshots, lineage sampling defaults, cached diversity
reads, and coarse performance snapshots. Those helpers are intentionally
lower-level than the facade, but they are stable enough to deserve a direct
telemetry chapter of their own instead of staying as a flat root utility
file.

## neat/telemetry/accessors/telemetry.accessors.ts

### buildLineageSnapshot

```ts
buildLineageSnapshot(
  population: { _id?: number | undefined; _parents?: number[] | undefined; }[],
  limit: number,
): { id: number; parents: number[]; }[]
```

Snapshot lineage metadata for the first `limit` genomes.

### clearTelemetryBuffer

```ts
clearTelemetryBuffer(
  host: TelemetryAccessorHost,
): void
```

Clear the telemetry buffer in place.

### getCachedDiversityStats

```ts
getCachedDiversityStats(
  host: TelemetryAccessorHost,
): DiversityStats | undefined
```

Read cached diversity statistics.

### getObjectiveEventsSnapshot

```ts
getObjectiveEventsSnapshot(
  host: TelemetryAccessorHost,
): { gen: number; type: "add" | "remove"; key: string; }[]
```

Return a shallow copy of recent objective events.

### getPerformanceStatsSnapshot

```ts
getPerformanceStatsSnapshot(
  host: TelemetryAccessorHost,
): { lastEvalMs: number | undefined; lastEvolveMs: number | undefined; }
```

Snapshot performance timings for evaluation and evolution steps.

### getTelemetryBuffer

```ts
getTelemetryBuffer(
  host: TelemetryAccessorHost,
): TelemetryEntry[]
```

Return the telemetry buffer, defaulting to an empty array when missing.

### LINEAGE_SNAPSHOT_DEFAULT_LIMIT

Default limit for lineage snapshots to avoid large payloads.

### TelemetryAccessorHost

Minimal host surface needed by telemetry accessors.
