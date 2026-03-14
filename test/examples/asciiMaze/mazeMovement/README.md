# mazeMovement

Public MazeMovement facade for the dedicated mazeMovement module boundary.

The folder now owns runtime helpers, policy helpers, shaping helpers, and
finalization logic. This file keeps the user-facing API in one place while
the implementation stays split into focused helpers.

## mazeMovement/mazeMovement.types.ts

Shared type surface for the dedicated mazeMovement module.

Step 2 moves internal simulation contracts here first so later helper files
can depend on one narrow typed surface.

### DirectionSelectionStats

Diagnostic telemetry produced when selecting a direction from network logits.

Encapsulates the chosen direction along with entropy and probability data so
downstream helpers can apply shaping rewards and penalties without
rederiving softmax statistics on hot paths.

### MazeMovementBufferPools

Initialized pooled buffers shared across maze movement simulations.

These pools are reused between runs to keep the hot path allocation-light
while preserving a narrow typed seam for service helpers.

### MazeMovementRunServiceState

Mutable run-scoped service state shared by the mazeMovement facade.

The dedicated services module owns these counters so later runtime, policy,
and shaping helpers can depend on one explicit mutable surface instead of
directly reaching into class-private state.

### MazeMovementSimulationResult

Result shape returned by `MazeMovement.simulateAgent`.

This contract matches the legacy inline return annotation so callers can
keep depending on the current fields while the dedicated module boundary is
being extracted.

### SimulationState

Internal aggregate state used during a single agent simulation run.

Purpose:
- Hold all derived runtime values, counters and diagnostic stats used by the
  MazeMovement simulation helpers. This shape is intentionally rich so tests
  and visualisers can inspect intermediate state when debugging.

Notes:
- This interface remains internal to the mazeMovement module boundary.
- Property descriptions are explicit to surface helpful tooltips in editors.

## mazeMovement/mazeMovement.ts

### MazeMovement

Maze movement entry surface used by fitness evaluation and evolution runs.

The public API intentionally remains class-based so existing example imports
do not change while the implementation lives under the dedicated module.

#### #COORDINATE_SCRATCH

Reused integer coordinate scratch for hot-path movement helpers.

#### #hasReachedExit

```ts
#hasReachedExit(
  simulationState: SimulationState,
  exitPos: readonly [number, number],
): boolean
```

Determine whether the current state has reached the maze exit.

Parameters:
- `simulationState` - - Mutable run state for the active episode.
- `exitPos` - - Exit coordinate for the run.

Returns: True when the agent position matches the exit coordinate.

#### #processMovementAndShaping

```ts
#processMovementAndShaping(
  simulationState: SimulationState,
  encodedMaze: number[][],
  distanceMap: number[][] | undefined,
): boolean
```

Execute the selected move, apply post-action shaping, and evaluate stop rules.

Parameters:
- `simulationState` - - Mutable run state for the active episode.
- `encodedMaze` - - Maze grid used for movement and distance lookup.
- `distanceMap` - - Optional precomputed distance map.

Returns: True when the episode should stop after this step.

#### #processPerceptionAndPolicy

```ts
#processPerceptionAndPolicy(
  simulationState: SimulationState,
  network: INetwork,
  encodedMaze: number[][],
  exitPos: readonly [number, number],
  distanceMap: number[][] | undefined,
): void
```

Refresh visit bookkeeping, perception state, and direction policy.

Parameters:
- `simulationState` - - Mutable run state for the active episode.
- `network` - - Policy network used for action selection.
- `encodedMaze` - - Maze grid used for the run.
- `exitPos` - - Exit coordinate for the run.
- `distanceMap` - - Optional precomputed distance map.

#### moveAgent

```ts
moveAgent(
  encodedMaze: readonly (readonly number[])[],
  position: readonly [number, number],
  direction: number,
): [number, number]
```

Move the agent one step in the requested direction when the target cell is open.

Parameters:
- `encodedMaze` - - Maze grid used for collision checks.
- `position` - - Current agent position.
- `direction` - - Direction index in the action space.

Returns: New position when the move is valid, otherwise the original position.

Example:

const moved = MazeMovement.moveAgent(encodedMaze, [3, 2], 1);

#### selectDirection

```ts
selectDirection(
  outputs: number[],
): DirectionSelectionStats
```

Convert raw network outputs into a chosen action plus diagnostics.

Parameters:
- `outputs` - - Raw action logits for the four maze directions.

Returns: Chosen direction plus softmax and entropy diagnostics.

Example:

const stats = MazeMovement.selectDirection([0.2, 1.4, -0.1, 0]);

#### simulateAgent

```ts
simulateAgent(
  network: INetwork,
  encodedMaze: number[][],
  startPos: readonly [number, number],
  exitPos: readonly [number, number],
  distanceMap: number[][] | undefined,
  maxSteps: number,
): MazeMovementSimulationResult
```

Simulate one full maze episode for a network-controlled agent.

Parameters:
- `network` - - Policy network used to choose actions.
- `encodedMaze` - - Maze grid for the active episode.
- `startPos` - - Start coordinate.
- `exitPos` - - Exit coordinate.
- `distanceMap` - - Optional precomputed distance map.
- `maxSteps` - - Maximum allowed step count before termination.

Returns: Final simulation result including path, fitness, and progress.

## mazeMovement/mazeMovement.services.ts

Shared mutable services for the dedicated mazeMovement module.

This module owns the pooled buffers, PRNG state, output-history plumbing,
and shared run-scoped counters used by the legacy MazeMovement facade while
Step 2 incrementally moves helper categories into the dedicated boundary.

### getMazeMovementBufferMetadata

```ts
getMazeMovementBufferMetadata(): { cachedWidth: number; cachedHeight: number; }
```

Read the currently cached maze dimensions for bounds and index helpers.

Returns: Cached width and height for the active pooled buffers.

### getMazeMovementRunServiceState

```ts
getMazeMovementRunServiceState(): MazeMovementRunServiceState
```

Expose the shared mutable run-scoped state used across helper categories.

Returns: The singleton mutable run-state object for the current process.

### indexMazeMovementCell

```ts
indexMazeMovementCell(
  x: number,
  y: number,
): number
```

Convert a cell coordinate into the pooled linear grid index.

Parameters:
- `x` - - Zero-based maze column.
- `y` - - Zero-based maze row.

Returns: Linearized index used by pooled grid buffers.

### initializeMazeMovementBufferPools

```ts
initializeMazeMovementBufferPools(
  width: number,
  height: number,
  maxSteps: number,
): MazeMovementBufferPools
```

Ensure the pooled grid and path buffers are initialized for a run.

Parameters:
- `width` - - Maze width in cells.
- `height` - - Maze height in cells.
- `maxSteps` - - Maximum path length expected for the run.

Returns: The initialized pooled buffer surface.

### materializeMazeMovementPath

```ts
materializeMazeMovementPath(
  length: number,
): [number, number][]
```

Materialize the active pooled path buffers into a fresh tuple array.

Parameters:
- `length` - - Number of active path entries to copy.

Returns: A newly allocated materialized path snapshot.

### randomMazeMovementUnit

```ts
randomMazeMovementUnit(): number
```

Generate a pseudo-random number in the range `[0, 1)`.

Returns: A deterministic or host-random unit float for exploration logic.

### readMazeMovementOutputHistory

```ts
readMazeMovementOutputHistory(
  network: INetwork,
): number[][] | undefined
```

Read the reflected `_lastStepOutputs` network history when present.

Parameters:
- `network` - - Network that may carry the reflected output history.

Returns: Sanitized output history or `undefined` when absent or invalid.

### requireMazeMovementBufferPools

```ts
requireMazeMovementBufferPools(): MazeMovementBufferPools
```

Return the initialized pooled buffer surface for the current run.

Returns: The shared buffer pools.

### resetMazeMovementRunServiceState

```ts
resetMazeMovementRunServiceState(): MazeMovementRunServiceState
```

Reset the shared mutable run-scoped state before a new simulation begins.

Returns: The reused singleton state after reset.

### writeMazeMovementOutputHistory

```ts
writeMazeMovementOutputHistory(
  network: INetwork,
  history: number[][],
): void
```

Persist the reflected `_lastStepOutputs` network history.

Parameters:
- `network` - - Network receiving the reflected output history.
- `history` - - Bounded output-history payload to persist.

## mazeMovement/mazeMovement.constants.ts

Frozen tuning surface for the dedicated mazeMovement module.

This constant table keeps simulation policy, shaping thresholds, and lookup
tables in one place so the public facade can stay focused on orchestration.

### MAZE_MOVEMENT_CONSTANTS

Frozen tuning surface for the dedicated mazeMovement module.

This constant table keeps simulation policy, shaping thresholds, and lookup
tables in one place so the public facade can stay focused on orchestration.

## mazeMovement/mazeMovement.utils.ts

### computeActionEntropyFromCounts

```ts
computeActionEntropyFromCounts(
  directionCounts: number[],
  logActions: number,
  scratch: Float64Array<ArrayBufferLike>,
): number
```

Compute normalized action entropy from direction counts.

Parameters:
- `directionCounts` - - Number of moves taken in each direction.
- `logActions` - - Precomputed normalization factor for the action space.
- `scratch` - - Single-value floating-point scratch buffer reused by the caller.

Returns: Normalized entropy in the range `[0, 1]`.

### isFiniteNumberArray

```ts
isFiniteNumberArray(
  candidate: unknown,
): boolean
```

Determine whether the provided value is a finite-number array.

Parameters:
- `candidate` - - Value to inspect.

Returns: True when the input is an array of finite numbers.

### materializePath

```ts
materializePath(
  length: number,
  pathX: Int32Array<ArrayBufferLike>,
  pathY: Int32Array<ArrayBufferLike>,
): [number, number][]
```

Materialize the active prefix of pooled path buffers into a fresh array.

Parameters:
- `length` - - Number of path entries to materialize.
- `pathX` - - Pooled X-coordinate buffer.
- `pathY` - - Pooled Y-coordinate buffer.

Returns: A newly allocated array of path tuples.

### nextPowerOfTwo

```ts
nextPowerOfTwo(
  n: number,
): number
```

Return the smallest power-of-two integer greater than or equal to `n`.

Parameters:
- `n` - - Target minimum integer capacity.

Returns: The smallest power of two greater than or equal to `n`.

### readOutputHistory

```ts
readOutputHistory(
  network: INetwork,
): number[][] | undefined
```

Read the optional `_lastStepOutputs` history stored on a network.

Parameters:
- `network` - - Network instance that may expose a reflected outputs history.

Returns: Sanitized history buffer or `undefined` when absent or invalid.

### sumVisionGroup

```ts
sumVisionGroup(
  vision: number[],
  start: number,
  groupLength: number,
  scratch: Float64Array<ArrayBufferLike>,
): number
```

Sum a contiguous group of entries from a vision vector into a reusable scratch buffer.

Parameters:
- `vision` - - Flat perception vector.
- `start` - - Start index of the group to sum.
- `groupLength` - - Number of entries in the group.
- `scratch` - - Reusable scratch buffer populated with copied values.

Returns: Numeric sum of the selected group.

### writeOutputHistory

```ts
writeOutputHistory(
  network: INetwork,
  history: number[][],
): void
```

Persist a bounded outputs history on the network via reflection.

Parameters:
- `network` - - Target network to mutate.
- `history` - - Updated history buffer.
