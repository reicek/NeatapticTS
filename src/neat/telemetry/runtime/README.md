# neat/telemetry/runtime

Write-side telemetry runtime helpers shared by the recorder path.

This chapter owns the tiny mechanics that make telemetry recording safe at
runtime: initialize the buffer on demand, stream entries without breaking the
evolution loop, and keep the in-memory history bounded. Keeping those helpers
together gives the telemetry subtree a direct internal chapter for the
"record and stream" path instead of leaving that write-side behavior in a
flat root utility file.

The key idea is containment. The recorder chapter decides what a telemetry
entry means; this runtime layer decides how that entry can be stored and
observed safely while evolution is still in progress.
By isolating buffer creation, callback delivery, and trimming here, the
telemetry pipeline can remain useful to dashboards and experiments without
letting diagnostics logic leak back into the evolutionary control flow.

## neat/telemetry/runtime/telemetry.runtime.ts

### ensureTelemetryBuffer

```ts
ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[]
```

Ensure the telemetry buffer is initialized before a generation snapshot is recorded.

This helper keeps telemetry opt-in and cheap. Runs that never inspect or
export telemetry do not need an eager buffer allocation, while runs that do
record telemetry can ask for a stable mutable array at the moment they need
it.

Parameters:
- `telemetryContext` - Neat-like context holding the mutable telemetry buffer.

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
This is the main runtime safety boundary between telemetry observers and the
search loop: visibility is allowed, but observer failures must never become
control-flow failures for evolution itself.

Parameters:
- `telemetryContext` - Neat-like context with optional telemetry stream settings.
- `telemetryEntry` - Entry to forward to the configured stream callback.

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

Telemetry is useful precisely because it can span many generations, but an
unbounded in-memory history would turn that visibility into a memory leak.
This helper enforces the retention contract by keeping only the most recent
slice of the run.

Parameters:
- `telemetryBufferRef` - Buffer to trim in place.
- `maxEntries` - Maximum number of recent entries to keep.

Returns: Nothing. Older entries are dropped from the front of the buffer.

Example:

```ts
trimTelemetryBuffer(neat._telemetry ?? [], 500);
```
