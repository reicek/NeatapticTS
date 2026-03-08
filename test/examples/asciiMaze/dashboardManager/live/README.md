# dashboardManager/live

## dashboardManager/live/dashboardManager.live.services.ts

### redrawDashboard

`(context: import("C:/NeatapticTS/test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardManagerContext, currentMaze: string[], neat: unknown) => void`

Repaint the live dashboard from current state and refresh the detailed snapshot.

Parameters:

- `context` - - Dashboard runtime context for state and output callbacks.
- `currentMaze` - - Maze currently being evolved.
- `neat` - - Optional NEAT runtime instance used for detailed stats.
