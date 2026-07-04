# mazeMovement/runtime

Runtime/environment helpers for the dedicated mazeMovement module.

This file is the environment-facing foundation beneath the rest of the
mazeMovement stack. When the higher-level policy and shaping helpers need to
ask what world the agent is actually inside, they depend on this boundary for
cell-open checks, distance lookup, run-state creation, visit-ring
bookkeeping, and perception-state updates.

The point of this split is to keep "what the maze world currently is" apart
from "what the policy wants to do next" and "how that action should be
rewarded." That separation makes the public facade easier to read because the
hot path can be understood as environment refresh first, decision second,
shaping third, and result folding last.

Read this file as the substrate for the rest of the module: runtime helpers
define the world state that policy, shaping, and finalization build on top of.

## mazeMovement/runtime/mazeMovement.runtime.ts

### buildMazeMovementVisionAndDistance

```ts
buildMazeMovementVisionAndDistance(
  state: SimulationState,
  encodedMaze: number[][],
  exitPos: readonly [number, number],
  distanceMap: number[][] | undefined,
): void
```

Build the current perception vector and update distance-tracking state.

Parameters:

- `state` - Mutable simulation state for the active run.
- `encodedMaze` - Maze grid used for perception and distance lookup.
- `exitPos` - Goal coordinate for the current run.
- `distanceMap` - Optional precomputed distance map.

### createMazeMovementRunState

```ts
createMazeMovementRunState(
  encodedMaze: number[][],
  startPos: readonly [number, number],
  distanceMap: number[][] | undefined,
  maxSteps: number,
): SimulationState
```

Create the initial run-state object for one simulation episode.

Parameters:

- `encodedMaze` - Maze grid used by the run.
- `startPos` - Starting coordinate.
- `distanceMap` - Optional precomputed distance map.
- `maxSteps` - Maximum number of allowed simulation steps.

Returns: Fresh simulation state backed by the shared buffer pools.

### getMazeMovementDistance

```ts
getMazeMovementDistance(
  encodedMaze: readonly (readonly number[])[],
  __1: readonly [number, number],
  distanceMap: number[][] | undefined,
): number
```

Resolve the current distance value for a maze coordinate.

Parameters:

- `encodedMaze` - Maze grid aligned with the optional distance map.
- `coordinates` - Zero-based `[x, y]` coordinate tuple.
- `distanceMap` - Optional precomputed distance map.

Returns: Finite distance when present, otherwise `Infinity`.

### getMazeMovementHistoryFromEnd

```ts
getMazeMovementHistoryFromEnd(
  state: SimulationState,
  nth: number,
): number | undefined
```

Return the `nth` most recent cell index from the visit-history ring.

Parameters:

- `state` - Mutable simulation state containing the ring buffer.
- `nth` - One-based index from the history tail.

Returns: The requested cell index or `undefined` when out of range.

### isMazeMovementCellOpen

```ts
isMazeMovementCellOpen(
  encodedMaze: readonly (readonly number[])[],
  x: number,
  y: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): boolean
```

Determine whether a maze cell is inside bounds and not a wall.

Parameters:

- `encodedMaze` - Maze grid to inspect.
- `x` - Zero-based maze column.
- `y` - Zero-based maze row.
- `coordinateScratch` - Reused integer scratch buffer for coordinate coercion.

Returns: True when the target cell is within bounds and open.

### pushMazeMovementHistory

```ts
pushMazeMovementHistory(
  state: SimulationState,
  cellIndex: number,
): void
```

Push a cell index into the circular visit-history ring.

Parameters:

- `state` - Mutable simulation state containing the ring buffer.
- `cellIndex` - Linearized cell index to append.

### recordMazeMovementVisitAndPenalties

```ts
recordMazeMovementVisitAndPenalties(
  state: SimulationState,
): void
```

Record the current cell visit and update visit-driven penalties.

Parameters:

- `state` - Mutable simulation state for the active run.
