# dashboardManager/live

Live redraw boundary for the ASCII Maze dashboard.

This file is the runtime-facing chapter opening for the dashboard's live
surface. It owns the part of the dashboard that needs to feel immediate to a
human watching evolution unfold: the frame, the current champion snapshot,
and the refreshed detailed stats that make each redraw more than a raw text
repaint.

The live boundary exists so rendering cadence can stay separate from archive
persistence and telemetry export. That split matters because redraw work is
frequent and user-facing, while archive and telemetry helpers answer slower,
more analytical questions.

A useful reading order is:

1. start here for the visible dashboard refresh path,
2. continue with `dashboardManager.telemetry.services.ts` to see how redraws
   pick up richer detail snapshots,
3. finish with `dashboardManager.archive.services.ts` for the long-lived
   record of especially important runs.

## dashboardManager/live/dashboardManager.live.services.ts

### redrawDashboard

```ts
redrawDashboard(
  context: DashboardManagerContext,
  currentMaze: string[],
  neat: unknown,
): void
```

Repaint the live dashboard from current state and refresh the detailed snapshot.

Parameters:
- `context` - Dashboard runtime context for state and output callbacks.
- `currentMaze` - Maze currently being evolved.
- `neat` - Optional NEAT runtime instance used for detailed stats.

### resolveActivationSchedulingValue

```ts
resolveActivationSchedulingValue(
  detailedStats: AsciiMazeDetailedStats | null,
): string | null
```

Resolve the compact activation-scheduling label shown in the live dashboard.

Parameters:
- `detailedStats` - Latest detailed dashboard snapshot.

Returns: Human-readable scheduling summary or null when unavailable.
