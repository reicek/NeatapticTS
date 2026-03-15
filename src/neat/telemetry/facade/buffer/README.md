# neat/telemetry/facade/buffer

## neat/telemetry/facade/buffer/telemetry.facade.buffer.ts

### clearTelemetry

```ts
clearTelemetry(
  host: TelemetryFacadeBufferHost,
): void
```

Clear cached telemetry entries.

Parameters:
- `host` - - `Neat` instance whose telemetry buffer should be reset.

Returns: Nothing. The helper mutates the host buffer in place.

### exportTelemetryCSV

```ts
exportTelemetryCSV(
  host: TelemetryFacadeBufferHost,
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
  host: TelemetryFacadeBufferHost,
): string
```

Export telemetry as JSON Lines so logs can stream into files or post-processors.

Parameters:
- `host` - - `Neat` instance whose telemetry buffer should be serialized.

Returns: JSONL payload with one telemetry object per line.

### getTelemetry

```ts
getTelemetry(
  host: TelemetryFacadeBufferHost,
): TelemetryEntry[]
```

Return the in-memory telemetry buffer.

This chapter groups the lowest-level telemetry reads with the export helpers
so callers can treat "inspect the buffer" and "serialize the buffer" as one
concept cluster inside the broader telemetry facade.

Parameters:
- `host` - - `Neat` instance storing generation telemetry snapshots.

Returns: Telemetry entries captured so far, or an empty array when telemetry
is not initialized.

Example:

```ts
const recentTelemetry = getTelemetry(neat);
console.log(recentTelemetry.at(-1)?.gen);
```

### TelemetryFacadeBufferHost

Narrow telemetry-facade host surface required by the buffer/export chapter.

This chapter exists to keep the root telemetry facade focused on public
orchestration while the mechanics for reading, clearing, and serializing the
telemetry buffer live together in one small boundary.
