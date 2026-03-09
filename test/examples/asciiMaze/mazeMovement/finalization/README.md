# mazeMovement/finalization

## mazeMovement/finalization/mazeMovement.finalization.ts

### mazeMovement.finalization

Result-finalization helpers for the dedicated mazeMovement module.

These helpers assemble the final simulation payload once the orchestration
facade has finished stepping the run. Keeping this logic here lets the main
facade stay focused on the episode loop rather than on score shaping math.

### computeMazeMovementActionEntropy

`(directionCounts: number[]) => number`

Compute the normalized action-entropy summary for a finished run.

Parameters:
- `directionCounts` - - Per-direction action counts recorded during the run.

Returns: Normalized entropy in the range `[0, 1]`.

### finalizeFailedMazeMovementRun

`(state: import("test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, encodedMaze: number[][], startPos: readonly [number, number], exitPos: readonly [number, number], distanceMap: number[][] | undefined) => import("test/examples/asciiMaze/mazeMovement/mazeMovement.types").MazeMovementSimulationResult`

Build the finalized payload for a failed maze run.

Parameters:
- `state` - - Completed simulation state for the failed run.
- `encodedMaze` - - Maze grid used to compute fallback geometric progress.
- `startPos` - - Start coordinate for the current episode.
- `exitPos` - - Exit coordinate for the current episode.
- `distanceMap` - - Optional precomputed distance map aligned to the maze.

Returns: Failure result with shaped fitness, path, and diagnostic summaries.

### finalizeSuccessfulMazeMovementRun

`(state: import("test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, maxSteps: number) => import("test/examples/asciiMaze/mazeMovement/mazeMovement.types").MazeMovementSimulationResult`

Build the finalized payload for a successful maze run.

Parameters:
- `state` - - Completed simulation state for the successful run.
- `maxSteps` - - Maximum allowed step budget for the run.

Returns: Success result with fitness, path, and diagnostic summaries.
