# neat/telemetry/runtime

Write-side telemetry runtime helpers shared by the recorder path.

This chapter owns the tiny mechanics that make telemetry recording safe at
runtime: initialize the buffer on demand, stream entries without breaking the
evolution loop, and keep the in-memory history bounded. Keeping those helpers
together gives the telemetry subtree a direct internal chapter for the
"record and stream" path instead of leaving that write-side behavior in a
flat root utility file.

## neat/telemetry/runtime/telemetry.runtime.ts

### ensureTelemetryBuffer

```ts
ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[]
```

Ensure the telemetry buffer is initialized before a generation snapshot is recorded.

Parameters:
- `telemetryContext` - - Neat-like context holding the mutable telemetry buffer.

Returns: Mutable telemetry buffer used for in-memory history.

Example:

```ts
const telemetryBuffer = ensureTelemetryBuffer(neat);
telemetryBuffer.push(entry);
```

### safelyStreamTelemetryEntry

```ts
safelyStreamTelemetryEntry(
  telemetryContext: { options?: TelemetryStreamOptions | undefined; },
  telemetryEntry: TelemetryEntry,
): void
```

Stream a telemetry entry when the host enables runtime callbacks.

Callback failures are intentionally swallowed so diagnostics hooks cannot
destabilize an evolution run.

Parameters:
- `telemetryContext` - - Neat-like context with optional telemetry stream settings.
- `telemetryEntry` - - Entry to forward to the configured stream callback.

Returns: Nothing. The helper only invokes the callback when the runtime opts in.

Example:

```ts
safelyStreamTelemetryEntry(
  { options: { telemetryStream: { enabled: true, onEntry: console.log } } },
  entry,
);
```

### trimTelemetryBuffer

```ts
trimTelemetryBuffer(
  telemetryBufferRef: TelemetryEntry[],
  maxEntries: number,
): void
```

Trim the telemetry buffer to its configured maximum size.

Parameters:
- `telemetryBufferRef` - - Buffer to trim in place.
- `maxEntries` - - Maximum number of recent entries to keep.

Returns: Nothing. Older entries are dropped from the front of the buffer.

Example:

```ts
trimTelemetryBuffer(neat._telemetry ?? [], 500);
```
