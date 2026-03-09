# dashboardManager/telemetry

## dashboardManager/telemetry/dashboardManager.telemetry.services.ts

### createDetailedStatsSnapshot

`(state: import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardManagerState, neat: unknown) => import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").AsciiMazeDetailedStats | null`

Build the rich telemetry detail snapshot shown in browser hooks and exported snapshots.

Parameters:
- `state` - - Mutable dashboard state with histories and current best candidate.
- `neat` - - Optional NEAT instance used for population-level telemetry.

Returns: Detailed telemetry snapshot or `null` when no data is available.

### emitTelemetryPayload

`(state: import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardManagerState, generation: number, telemetryHook: ((payload: import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardTelemetryPayload) => void) | undefined) => void`

Emit the structured telemetry payload used by browser hosts and runtime hooks.

Parameters:
- `state` - - Mutable dashboard state used to assemble the payload.
- `generation` - - Current generation number.
- `telemetryHook` - - Optional runtime hook installed by the browser host.

### getDashboardLastTelemetry

`(state: import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardManagerState) => import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").AsciiMazeTelemetrySnapshot`

Produce the latest public telemetry snapshot from current dashboard state.

Parameters:
- `state` - - Mutable dashboard state.

Returns: Public telemetry snapshot used by browser hosts.

### updateTelemetryHistory

`(state: import("test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardManagerState, neatInstance: { getTelemetry?: (() => unknown[]) | undefined; } | undefined) => void`

Pull the latest NEAT telemetry snapshot and update bounded dashboard histories.

Parameters:
- `state` - - Mutable dashboard state that owns bounded histories.
- `neatInstance` - - Optional NEAT-like runtime exposing `getTelemetry()`.
