# mazeMovement/shaping

Reward-shaping helpers for the dedicated mazeMovement module.

This file is the judgment layer of the mazeMovement pipeline. After the
policy boundary chooses a move, the shaping boundary decides how that move
should affect learning pressure: which behaviors deserve reward, which should
be penalized, and which patterns signal stagnation or useful exploration.

It owns movement reward shaping, stagnation penalties, entropy-guided
bonuses, and other post-action fitness adjustments so the main facade does
not have to mix motion updates with reward math in one long control-flow
block.

Read this file as the bridge between "the agent moved" and "the run's score
changed." Runtime supplies the world state, policy chooses the action, and
this boundary turns the aftermath into training signal.

## mazeMovement/shaping/mazeMovement.shaping.ts

### applyMazeMovementEntropyGuidanceShaping

```ts
applyMazeMovementEntropyGuidanceShaping(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply entropy-guided shaping based on confidence and perceptual guidance.

Parameters:

- `state` - Mutable simulation state for the active run.
- `rewardScale` - Global reward scale used by the penalties and bonuses.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementExplorationVisitAdjustment

```ts
applyMazeMovementExplorationVisitAdjustment(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply the per-cell exploration bonus or revisit penalty.

Parameters:

- `state` - Mutable simulation state for the active run.
- `rewardScale` - Global reward scale used by the adjustment.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementGlobalDistanceImprovementBonus

```ts
applyMazeMovementGlobalDistanceImprovementBonus(
  state: SimulationState,
  encodedMaze: number[][],
  rewardScale: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply the long-horizon global-distance improvement bonus.

Parameters:

- `state` - Mutable simulation state for the active run.
- `encodedMaze` - Maze grid used for distance lookup.
- `rewardScale` - Global reward scale used by the bonus magnitude.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementLocalAreaPenalty

```ts
applyMazeMovementLocalAreaPenalty(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply a local-area stagnation penalty when the run oscillates in a tight window.

Parameters:

- `state` - Mutable simulation state for the active run.
- `rewardScale` - Global reward scale used for the penalty magnitude.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementPostActionPenalties

```ts
applyMazeMovementPostActionPenalties(
  state: SimulationState,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply the post-action shaping and penalty aggregation phase.

Parameters:

- `state` - Mutable simulation state for the active run.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementProgressShaping

```ts
applyMazeMovementProgressShaping(
  state: SimulationState,
  distanceDelta: number,
  improved: boolean,
  worsened: boolean,
  rewardScale: number,
): void
```

Apply progress and away-from-goal shaping after a move.

Parameters:

- `state` - Mutable simulation state for the active run.
- `distanceDelta` - Positive when the agent moved closer to the goal.
- `improved` - True when the move improved distance to the goal.
- `worsened` - True when the move increased distance to the goal.
- `rewardScale` - Global reward scale used by the shaping terms.

### applyMazeMovementRepetitionAndBacktrackPenalties

```ts
applyMazeMovementRepetitionAndBacktrackPenalties(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply repetition and immediate-backtrack penalties.

Parameters:

- `state` - Mutable simulation state for the active run.
- `rewardScale` - Global reward scale used by the penalties.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementSaturationPenaltyCycle

```ts
applyMazeMovementSaturationPenaltyCycle(
  state: SimulationState,
  rewardScale: number,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply the periodic saturation penalty cycle.

Parameters:

- `state` - Mutable simulation state for the active run.
- `rewardScale` - Global reward scale used by the penalties.
- `coordinateScratch` - Reused coordinate scratch buffer.

### executeMazeMovementAndRewards

```ts
executeMazeMovementAndRewards(
  state: SimulationState,
  encodedMaze: number[][],
  distanceMap: number[][] | undefined,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Execute the chosen move and apply the shaping terms tied to that move.

Parameters:

- `state` - Mutable simulation state for the active run.
- `encodedMaze` - Maze grid used for move validity and distance lookup.
- `distanceMap` - Optional precomputed distance map.
- `coordinateScratch` - Reused coordinate scratch buffer.

### maybeTerminateMazeMovementDeepStagnation

```ts
maybeTerminateMazeMovementDeepStagnation(
  state: SimulationState,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): boolean
```

Apply the deep-stagnation termination penalty when appropriate.

Parameters:

- `state` - Mutable simulation state for the active run.
- `coordinateScratch` - Reused coordinate scratch buffer.

Returns: True when the run should terminate early.
