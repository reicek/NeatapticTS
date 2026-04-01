# dashboardManager/archive

Solved-run archive boundary for the ASCII Maze dashboard.

This file answers a different question than the live dashboard: not "what is
happening right now?" but "which runs were important enough to keep?" It
owns the durable storytelling surface for solved mazes, including the
formatted block that preserves path metrics, compact sparklines, and network
structure details once a layout has been solved.

Keeping archive formatting separate from live redraws prevents the hot path
from carrying long-form reporting concerns. The result is easier to read in
code and easier to reason about when a future reader wants to change archive
policy without touching the redraw cadence.

Read this file after the live redraw boundary if you want to follow how the
dashboard turns one notable run into a stable historical record.

## dashboardManager/archive/dashboardManager.archive.services.ts

### recordSolvedMaze

```ts
recordSolvedMaze(
  context: DashboardManagerContext,
  maze: string[],
  result: IMazeRunResult,
  network: INetwork,
  generation: number,
): void
```

Record and emit a newly solved maze archive block when the layout has not been seen before.

Parameters:
- `context` - - Dashboard runtime context containing archive state and callbacks.
- `maze` - - Solved maze layout.
- `result` - - Successful run result used for archive stats.
- `network` - - Network that solved the maze.
- `generation` - - Generation number at solve time.
