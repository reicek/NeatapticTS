# neat/telemetry/facade/novelty

Narrow telemetry-facade host surface required by the novelty chapter.

This chapter owns the tiny novelty-archive maintenance surface directly so
the public telemetry facade can delegate both read and reset operations into
the same small novelty boundary without depending on a leftover root helper.

## neat/telemetry/facade/novelty/telemetry.facade.novelty.ts

### getNoveltyArchiveSize

```ts
getNoveltyArchiveSize(
  host: TelemetryFacadeNoveltyHost,
): number
```

Return the current novelty archive size.

This helper gives diagnostics and tests a small read path for novelty search
state without pulling the rest of the telemetry facade into view.

Parameters:
- `host` - - `Neat` instance tracking novelty behavior descriptors.

Returns: Number of archived novelty descriptors.

### resetNoveltyArchive

```ts
resetNoveltyArchive(
  host: TelemetryFacadeNoveltyHost,
): void
```

Clear the novelty archive.

Parameters:
- `host` - - `Neat` instance whose novelty archive should be reset.

Returns: Nothing. The archive is mutated in place.

### TelemetryFacadeNoveltyHost

Narrow telemetry-facade host surface required by the novelty chapter.

This chapter owns the tiny novelty-archive maintenance surface directly so
the public telemetry facade can delegate both read and reset operations into
the same small novelty boundary without depending on a leftover root helper.
