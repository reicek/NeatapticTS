# neat/telemetry/facade/species

## neat/telemetry/facade/species/telemetry.facade.species.ts

### exportSpeciesHistoryCSV

```ts
exportSpeciesHistoryCSV(
  host: TelemetryFacadeSpeciesHost,
  maxEntries: number,
): string
```

Export species history as CSV rows.

This chapter keeps the species-oriented forwarding logic out of the broader
telemetry facade so the root surface can stay organized by concept instead of
accumulating unrelated read helpers in one file.

Parameters:
- `host` - - `Neat` instance whose species history should be exported.
- `maxEntries` - - Maximum number of recent history entries to include.

Returns: CSV payload for offline species analysis.

### exportSpeciesHistoryJSONL

```ts
exportSpeciesHistoryJSONL(
  host: TelemetryFacadeSpeciesHost,
  maxEntries: number,
): string
```

Export species history as JSON Lines.

Parameters:
- `host` - - `Neat` instance whose species history should be serialized.
- `maxEntries` - - Maximum number of recent history entries to include.

Returns: JSONL payload describing recent species history snapshots.

### getSpeciesHistory

```ts
getSpeciesHistory(
  host: TelemetryFacadeSpeciesHost,
): SpeciesHistoryEntry[]
```

Return recorded species history, lazily backfilling extended metrics when enabled.

Parameters:
- `host` - - `Neat` instance storing species history snapshots.

Returns: Historical species entries for each recorded generation.

### getSpeciesStats

```ts
getSpeciesStats(
  host: TelemetryFacadeSpeciesHost,
): { id: number; size: number; bestScore: number; lastImproved: number; }[]
```

Return a concise summary for each current species.

Parameters:
- `host` - - `Neat` instance whose live species registry should be summarized.

Returns: Array of current species summaries.

### TelemetryFacadeSpeciesHost

Narrow telemetry-facade host surface required by the species chapter.

These helpers sit between the public telemetry facade and the dedicated
species modules, so they only depend on the pieces needed to forward species
reads and exports safely.
