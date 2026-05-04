# mazeMovement/finalization

Result-finalization helpers for the dedicated mazeMovement module.

This file is the fold step for the mazeMovement chapter. By the time control
reaches this boundary, runtime updates, policy choices, and shaping logic are
already finished. The remaining job is to turn that rich mutable run state
into one stable result payload that callers, tests, and visualizers can rely
on.

These helpers assemble the final simulation payload once the orchestration
facade has finished stepping the run. Keeping this logic here lets the main
facade stay focused on the episode loop rather than on score shaping math,
entropy summaries, path materialization, and terminal success/failure record
assembly.

Read this after runtime, policy, and shaping if you want to see how the
module turns one completed episode into a reportable result instead of just a
pile of counters.

## mazeMovement/finalization/mazeMovement.finalization.ts

### computeMazeMovementActionEntropy

```ts
computeMazeMovementActionEntropy(
  directionCounts: number[],
): number
```

Compute the normalized action-entropy summary for a finished run.

Parameters:
- `directionCounts` - Per-direction action counts recorded during the run.

Returns: Normalized entropy in the range `[0, 1]`.

### finalizeFailedMazeMovementRun

```ts
finalizeFailedMazeMovementRun(
  state: SimulationState,
  encodedMaze: number[][],
  startPos: readonly [number, number],
  exitPos: readonly [number, number],
  distanceMap: number[][] | undefined,
): MazeMovementSimulationResult
```

Build the finalized payload for a failed maze run.

Parameters:
- `state` - Completed simulation state for the failed run.
- `encodedMaze` - Maze grid used to compute fallback geometric progress.
- `startPos` - Start coordinate for the current episode.
- `exitPos` - Exit coordinate for the current episode.
- `distanceMap` - Optional precomputed distance map aligned to the maze.

Returns: Failure result with shaped fitness, path, and diagnostic summaries.

### finalizeSuccessfulMazeMovementRun

```ts
finalizeSuccessfulMazeMovementRun(
  state: SimulationState,
  maxSteps: number,
): MazeMovementSimulationResult
```

Build the finalized payload for a successful maze run.

Parameters:
- `state` - Completed simulation state for the successful run.
- `maxSteps` - Maximum allowed step budget for the run.

Returns: Success result with fitness, path, and diagnostic summaries.
