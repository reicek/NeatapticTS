# neat/telemetry/facade/lineage

## neat/telemetry/facade/lineage/telemetry.facade.lineage.ts

### getLineageSnapshot

```ts
getLineageSnapshot(
  host: TelemetryFacadeLineageHost,
  limit: number,
): TelemetryLineageSnapshotEntry[]
```

Return a compact lineage sample for the first genomes in the current population.

This chapter keeps the public telemetry facade focused on orchestration while
lineage inspection lives beside the narrow host contract and default limit it
depends on.

Parameters:
- `host` - - `Neat` instance whose population lineage should be sampled.
- `limit` - - Maximum number of genomes to include in the snapshot.

Returns: Array of `{ id, parents }` lineage entries.

Example:

```ts
const lineageSnapshot = getLineageSnapshot(neat);
console.log(lineageSnapshot.at(-1)?.parents);
```

### LINEAGE_SNAPSHOT_DEFAULT_LIMIT

Default limit for lineage snapshots to avoid large payloads.

### TelemetryFacadeLineageHost

Narrow telemetry-facade host surface required by the lineage chapter.

The lineage snapshot path only needs the current population and the
lightweight ancestry markers stored on each genome.

### TelemetryLineageSnapshotEntry

Compact lineage entry exposed by the telemetry facade.

This read model stays intentionally small so inspection helpers can show
immediate ancestry without exporting full genealogy trees.
