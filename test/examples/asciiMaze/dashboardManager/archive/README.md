# dashboardManager/archive

## dashboardManager/archive/dashboardManager.archive.services.ts

### recordSolvedMaze

`(context: import("C:/NeatapticTS/test/examples/asciiMaze/dashboardManager/dashboardManager.types").DashboardManagerContext, maze: string[], result: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").IMazeRunResult, network: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork, generation: number) => void`

Record and emit a newly solved maze archive block when the layout has not been seen before.

Parameters:

- `context` - - Dashboard runtime context containing archive state and callbacks.
- `maze` - - Solved maze layout.
- `result` - - Successful run result used for archive stats.
- `network` - - Network that solved the maze.
- `generation` - - Generation number at solve time.
