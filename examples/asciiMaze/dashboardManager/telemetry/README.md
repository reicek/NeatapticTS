# dashboardManager/telemetry

Telemetry-export boundary for the ASCII Maze dashboard.

This file is where the dashboard stops being only a terminal or browser view
and becomes a source of structured data. It builds the bounded histories,
rich detail snapshots, and export payloads that let other hosts inspect the
same run through charts, hooks, or serialized snapshots instead of only live
text output.

The telemetry boundary exists so "what should be observable" stays separate
from "how should the dashboard be painted." That distinction keeps browser
integrations, debug tooling, and offline inspection from leaking presentation
assumptions back into the redraw path.

A practical reading order is:

1. `getDashboardLastTelemetry()` for the public snapshot surface,
2. `updateTelemetryHistory()` for bounded-history upkeep,
3. the detail builders below when you want the richer analytical shelf.

## dashboardManager/telemetry/dashboardManager.telemetry.services.ts

### createDetailedStatsSnapshot

```ts
createDetailedStatsSnapshot(
  state: DashboardManagerState,
  neat: unknown,
): AsciiMazeDetailedStats | null
```

Build the rich telemetry detail snapshot shown in browser hooks and exported snapshots.

Parameters:
- `state` - Mutable dashboard state with histories and current best candidate.
- `neat` - Optional NEAT instance used for population-level telemetry.

Returns: Detailed telemetry snapshot or `null` when no data is available.

### emitTelemetryPayload

```ts
emitTelemetryPayload(
  state: DashboardManagerState,
  generation: number,
  telemetryHook: ((payload: DashboardTelemetryPayload) => void) | undefined,
): void
```

Emit the structured telemetry payload used by browser hosts and runtime hooks.

Parameters:
- `state` - Mutable dashboard state used to assemble the payload.
- `generation` - Current generation number.
- `telemetryHook` - Optional runtime hook installed by the browser host.

### getDashboardLastTelemetry

```ts
getDashboardLastTelemetry(
  state: DashboardManagerState,
): AsciiMazeTelemetrySnapshot
```

Produce the latest public telemetry snapshot from current dashboard state.

Parameters:
- `state` - Mutable dashboard state.

Returns: Public telemetry snapshot used by browser hosts.

### resolveActivationSchedulingDetails

```ts
resolveActivationSchedulingDetails(
  network: INetwork | null | undefined,
): AsciiMazeActivationSchedulingStats | null
```

Resolve compact activation-scheduling details for telemetry export.

Parameters:
- `network` - Current best network instance.

Returns: Compact scheduling detail snapshot or null when unavailable.

### updateTelemetryHistory

```ts
updateTelemetryHistory(
  state: DashboardManagerState,
  neatInstance: { getTelemetry?: (() => unknown[]) | undefined; } | undefined,
): void
```

Pull the latest NEAT telemetry snapshot and update bounded dashboard histories.

Parameters:
- `state` - Mutable dashboard state that owns bounded histories.
- `neatInstance` - Optional NEAT-like runtime exposing `getTelemetry()`.
