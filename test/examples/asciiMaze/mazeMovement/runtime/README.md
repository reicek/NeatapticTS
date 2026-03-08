# mazeMovement/runtime

## mazeMovement/runtime/mazeMovement.runtime.ts

### mazeMovement.runtime

Runtime/environment helpers for the dedicated mazeMovement module.

This file owns the low-level simulation primitives that do not define maze
policy: cell-open checks, distance lookup, run-state creation, visit-ring
bookkeeping, and perception-state updates.

### buildMazeMovementVisionAndDistance

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, encodedMaze: number[][], exitPos: readonly [number, number], distanceMap: number[][] | undefined) => void`

Build the current perception vector and update distance-tracking state.

Parameters:
- `state` - - Mutable simulation state for the active run.
- `encodedMaze` - - Maze grid used for perception and distance lookup.
- `exitPos` - - Goal coordinate for the current run.
- `distanceMap` - - Optional precomputed distance map.

### createMazeMovementRunState

`(encodedMaze: number[][], startPos: readonly [number, number], distanceMap: number[][] | undefined, maxSteps: number) => import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState`

Create the initial run-state object for one simulation episode.

Parameters:
- `encodedMaze` - - Maze grid used by the run.
- `startPos` - - Starting coordinate.
- `distanceMap` - - Optional precomputed distance map.
- `maxSteps` - - Maximum number of allowed simulation steps.

Returns: Fresh simulation state backed by the shared buffer pools.

### getMazeMovementDistance

`(encodedMaze: readonly (readonly number[])[], __1: readonly [number, number], distanceMap: number[][] | undefined) => number`

Resolve the current distance value for a maze coordinate.

Parameters:
- `encodedMaze` - - Maze grid aligned with the optional distance map.
- `coordinates` - - Zero-based `[x, y]` coordinate tuple.
- `distanceMap` - - Optional precomputed distance map.

Returns: Finite distance when present, otherwise `Infinity`.

### getMazeMovementHistoryFromEnd

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, nth: number) => number | undefined`

Return the `nth` most recent cell index from the visit-history ring.

Parameters:
- `state` - - Mutable simulation state containing the ring buffer.
- `nth` - - One-based index from the history tail.

Returns: The requested cell index or `undefined` when out of range.

### isMazeMovementCellOpen

`(encodedMaze: readonly (readonly number[])[], x: number, y: number, coordinateScratch: Int32Array<ArrayBufferLike>) => boolean`

Determine whether a maze cell is inside bounds and not a wall.

Parameters:
- `encodedMaze` - - Maze grid to inspect.
- `x` - - Zero-based maze column.
- `y` - - Zero-based maze row.
- `coordinateScratch` - - Reused integer scratch buffer for coordinate coercion.

Returns: True when the target cell is within bounds and open.

### pushMazeMovementHistory

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, cellIndex: number) => void`

Push a cell index into the circular visit-history ring.

Parameters:
- `state` - - Mutable simulation state containing the ring buffer.
- `cellIndex` - - Linearized cell index to append.

### recordMazeMovementVisitAndPenalties

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState) => void`

Record the current cell visit and update visit-driven penalties.

Parameters:
- `state` - - Mutable simulation state for the active run.
