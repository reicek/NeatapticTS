# neat/telemetry/exports

Telemetry export helpers for stream-friendly logs and spreadsheet-oriented analysis.

This chapter keeps the public serialization story together after the telemetry
facade split: JSONL helpers preserve full-fidelity event payloads for log
pipelines, while the CSV helpers flatten the most useful telemetry and
species-history metrics into a deterministic tabular shape for notebooks,
spreadsheets, and quick audits.

The neighboring `telemetry/facade/*` chapters own public `Neat` wrappers.
This exports chapter owns the actual serialization mechanics those wrappers
delegate to.

## neat/telemetry/exports/telemetry.exports.ts

### buildSpeciesHistoryCsv

```ts
buildSpeciesHistoryCsv(
  recentHistory: SpeciesHistoryEntry[],
  headers: string[],
): string
```

Build the full CSV string for species history entries.

Parameters:
- `recentHistory` - - History entries to export.
- `headers` - - Ordered headers for the export window.

Returns: Complete CSV payload.

### buildTelemetryHeaders

```ts
buildTelemetryHeaders(
  info: TelemetryHeaderInfo,
): string[]
```

Build the ordered header list for telemetry CSV output.

Parameters:
- `info` - - Collected header discovery state.

Returns: Ordered header names used for serialization.

### collectTelemetryHeaderInfo

```ts
collectTelemetryHeaderInfo(
  entries: TelemetryEntry[],
): TelemetryHeaderInfo
```

Collect header metadata from the sampled telemetry entries.

Parameters:
- `entries` - - Telemetry entries included in the export window.

Returns: Flattened header discovery state.

### DEFAULT_SPECIES_BEST_SCORE

Default fallback best score when missing.

### DEFAULT_SPECIES_HISTORY_GENERATION

Default fallback generation when missing.

### DEFAULT_SPECIES_HISTORY_MAX_ENTRIES

Default max entries for species history CSV exports.

### DEFAULT_SPECIES_ID

Default fallback species id when missing.

### DEFAULT_SPECIES_LAST_IMPROVED

Default fallback last improved when missing.

### DEFAULT_SPECIES_SIZE

Default fallback species size when missing.

### exportSpeciesHistoryCSV

```ts
exportSpeciesHistoryCSV(
  maxEntries: number,
): string
```

Export species history snapshots to CSV.

The exporter preserves runtime-added species stat fields by discovering the
union of keys across the requested history window. When history has not been
recorded yet but live species data exists, the helper synthesizes a minimal
snapshot so early-run CSV exports remain deterministic.

Parameters:
- `this` - - Neat instance exposing species history and optional live species.
- `maxEntries` - - Maximum number of recent history snapshots to include.

Returns: CSV payload describing one species row per generation snapshot.

### exportTelemetryCSV

```ts
exportTelemetryCSV(
  maxEntries: number,
): string
```

Export recent telemetry entries to a CSV string.

Flattening rules:
- nested `complexity`, `perf`, `lineage`, and selected `diversity` fields
  become `group.key` columns,
- optional arrays and maps are serialized as JSON cells only when present in
  the sampled window,
- the most recent `maxEntries` records are exported to keep output bounded.

Parameters:
- `this` - - Neat instance exposing the internal telemetry buffer.
- `maxEntries` - - Maximum number of recent telemetry rows to include.

Returns: CSV string containing headers plus one row per exported entry.

### exportTelemetryJSONL

```ts
exportTelemetryJSONL(): string
```

Serialize telemetry as JSON Lines for stream-friendly log exports.

Parameters:
- `this` - - Neat instance exposing the internal telemetry buffer.

Returns: JSONL payload with one telemetry object per line.

Example:

```ts
const jsonl = exportTelemetryJSONL.call(neat);
console.log(jsonl.split('\n').length);
```

### serializeTelemetryEntry

```ts
serializeTelemetryEntry(
  entry: TelemetryEntry,
  headers: string[],
): string
```

Serialize one telemetry entry into a CSV row using the ordered headers.

Parameters:
- `entry` - - Telemetry entry being serialized.
- `headers` - - Ordered headers for the whole export window.

Returns: CSV row string.

### TelemetryHeaderInfo

Shape describing collected telemetry header discovery info.

## neat/telemetry/exports/telemetry.exports.utils.ts

Shared header-discovery and species-row helpers for the telemetry exports chapter.

The main exports file owns the user-facing JSONL and CSV entrypoints, while
this companion utility file keeps the lower-level bookkeeping focused on two
jobs: discover a stable flattened column set for telemetry CSV windows, and
normalize species-history rows so early-run exports remain deterministic even
when the live controller state is still sparse.

### buildSpeciesHistoryStats

```ts
buildSpeciesHistoryStats(
  speciesList: SpeciesHistoryStat[],
  defaultSpeciesId: number,
  defaultSpeciesSize: number,
  defaultBestScore: number,
  defaultLastImproved: number,
): SpeciesHistoryStat[]
```

Normalize raw species records into exportable history stats.

Parameters:
- `speciesList` - - Raw species records to normalize.
- `defaultSpeciesId` - - Default species id when missing.
- `defaultSpeciesSize` - - Default species size when missing.
- `defaultBestScore` - - Default best score when missing.
- `defaultLastImproved` - - Default last improved when missing.

Returns: Normalized stats for CSV export.

### collectBaseKeys

```ts
collectBaseKeys(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
  frontsHeader: string,
): void
```

Collect base (top-level) telemetry keys for a single entry.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.
- `frontsHeader` - - Header label for fronts column.

Returns: void. Mutates `state.baseKeys`.

### collectDiversityLineageMetrics

```ts
collectDiversityLineageMetrics(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void
```

Collect curated diversity lineage metrics for stable CSV exports.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates diversity lineage key set.

### collectGroupedMetricKeys

```ts
collectGroupedMetricKeys(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void
```

Collect nested metric keys for grouped telemetry fields.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates complexity/perf/lineage key sets.

### collectOptionalColumnPresence

```ts
collectOptionalColumnPresence(
  entry: TelemetryEntry,
  state: TelemetryHeaderCollectionState,
): void
```

Collect presence flags for optional telemetry columns.

Parameters:
- `entry` - - Telemetry entry to inspect.
- `state` - - Mutable header collection state.

Returns: void. Mutates optional-column flags.

### collectSpeciesHistoryHeaders

```ts
collectSpeciesHistoryHeaders(
  history: SpeciesHistoryEntry[],
  generationHeader: string,
): string[]
```

Collect ordered header keys for species history CSV export.

Parameters:
- `history` - - Recent species history entries.
- `generationHeader` - - Header label for generation column.

Returns: Ordered header list for CSV output.

### ensureMinimalSpeciesSnapshot

```ts
ensureMinimalSpeciesSnapshot(
  neatInstance: NeatLike & { _speciesHistory?: SpeciesHistoryEntry[] | undefined; _species?: SpeciesHistoryStat[] | undefined; generation?: number | undefined; },
  history: SpeciesHistoryEntry[],
  fallbackGeneration: number,
  defaultSpeciesId: number,
  defaultSpeciesSize: number,
  defaultBestScore: number,
  defaultLastImproved: number,
): void
```

Ensure a minimal species snapshot exists for deterministic CSV headers.

Parameters:
- `neatInstance` - - Neat instance with optional species history and species.
- `history` - - Species history backing array.
- `fallbackGeneration` - - Generation fallback when missing.
- `defaultSpeciesId` - - Default species id when missing.
- `defaultSpeciesSize` - - Default species size when missing.
- `defaultBestScore` - - Default best score when missing.
- `defaultLastImproved` - - Default last improved when missing.

Returns: void. Mutates history when a minimal snapshot is needed.

### ensureSpeciesHistoryArray

```ts
ensureSpeciesHistoryArray(
  neatInstance: NeatLike & { _speciesHistory?: SpeciesHistoryEntry[] | undefined; },
): SpeciesHistoryEntry[]
```

Ensure the species history array exists on the Neat instance.

Parameters:
- `neatInstance` - - Neat instance holding species history.

Returns: Species history backing array (ensured on instance).

### resolveSpeciesHistoryCellValue

```ts
resolveSpeciesHistoryCellValue(
  historyEntry: SpeciesHistoryEntry,
  speciesStat: SpeciesHistoryStat,
  headerName: string,
  generationHeader: string,
): string
```

Resolve a single species history cell value for the provided header.

Parameters:
- `historyEntry` - - A single generation snapshot.
- `speciesStat` - - A single species stat record.
- `headerName` - - Column header name.
- `generationHeader` - - Column header name for generation.

Returns: Serialized cell (JSON) or empty string for missing values.

### safeStringifyCell

```ts
safeStringifyCell(
  value: unknown,
): string
```

Serialize a CSV cell with JSON.stringify safeguards.

Parameters:
- `value` - - Any value to stringify.

Returns: JSON string or empty string when JSON.stringify returns undefined.

### serializeSpeciesHistoryRow

```ts
serializeSpeciesHistoryRow(
  historyEntry: SpeciesHistoryEntry,
  speciesStat: SpeciesHistoryStat,
  orderedHeaders: string[],
  generationHeader: string,
): string
```

Serialize one species history row using the provided headers.

Parameters:
- `historyEntry` - - A single generation snapshot.
- `speciesStat` - - A single species stat record for that generation.
- `orderedHeaders` - - Ordered header list for stable CSV.
- `generationHeader` - - Column header name for generation.

Returns: CSV row string matching the provided header order.

### TelemetryHeaderCollectionState

Mutable state container used while collecting telemetry header metadata.
