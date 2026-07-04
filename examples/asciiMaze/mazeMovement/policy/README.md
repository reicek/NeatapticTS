# mazeMovement/policy

Action policy helpers for the dedicated mazeMovement module.

This file is the decision-making shelf inside the mazeMovement module. Once
runtime helpers have refreshed the agent's current world view, the policy
boundary decides how logits, exploration pressure, and short-horizon
heuristics combine into one movement choice.

It owns direction selection, epsilon handling, short-horizon policy
overrides, and saturation-driven bias control because those concerns answer a
shared question: how should the next action be chosen before reward shaping
reacts to the result?

Read this after the runtime helpers if you want the clean handoff from
perception into action choice. Then continue into shaping to see how the
trainer judges the consequences of that choice.

## mazeMovement/policy/mazeMovement.policy.ts

### applyMazeMovementEpsilonExploration

```ts
applyMazeMovementEpsilonExploration(
  state: SimulationState,
  encodedMaze: number[][],
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply epsilon-greedy exploration to the current action choice.

Parameters:
- `state` - Mutable simulation state for the active run.
- `encodedMaze` - Maze grid used for move validity checks.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementForcedExploration

```ts
applyMazeMovementForcedExploration(
  state: SimulationState,
  encodedMaze: number[][],
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Force a random valid move when the policy has stalled with repeated no-move outputs.

Parameters:
- `state` - Mutable simulation state for the active run.
- `encodedMaze` - Maze grid used for move validity checks.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementProximityGreedy

```ts
applyMazeMovementProximityGreedy(
  state: SimulationState,
  encodedMaze: number[][],
  distanceMap: number[][] | undefined,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Apply the short-horizon proximity-greedy override near the maze exit.

Parameters:
- `state` - Mutable simulation state for the active run.
- `encodedMaze` - Maze grid used for move validity checks.
- `distanceMap` - Optional precomputed distance map.
- `coordinateScratch` - Reused coordinate scratch buffer.

### applyMazeMovementSaturationAndBiasAdjust

```ts
applyMazeMovementSaturationAndBiasAdjust(
  state: SimulationState,
  outputs: number[],
  network: INetwork,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Detect saturation and optionally damp output-node biases.

Parameters:
- `state` - Mutable simulation state for the active run.
- `outputs` - Raw network logits for the current step.
- `network` - Policy network that produced the logits.
- `coordinateScratch` - Reused scratch buffer for temporary penalties.

### computeMazeMovementEpsilon

```ts
computeMazeMovementEpsilon(
  stepNumber: number,
  stepsSinceImprovement: number,
  distHere: number,
  saturations: number,
): number
```

Compute the adaptive epsilon used for policy exploration.

Parameters:
- `stepNumber` - Global step number inside the active simulation.
- `stepsSinceImprovement` - Number of steps without improvement.
- `distHere` - Current distance to goal for the active position.
- `saturations` - Rolling saturation count from the shared run state.

Returns: Exploration epsilon in the range `[0, 1]`.

### decideMazeMovementDirection

```ts
decideMazeMovementDirection(
  state: SimulationState,
  network: INetwork,
  coordinateScratch: Int32Array<ArrayBufferLike>,
): void
```

Activate the network, record output history, and choose the next direction.

Parameters:
- `state` - Mutable simulation state for the active run.
- `network` - Policy network used for the current step.

### selectMazeMovementDirection

```ts
selectMazeMovementDirection(
  outputs: number[],
): DirectionSelectionStats
```

Convert raw network outputs into a chosen direction plus diagnostics.

Parameters:
- `outputs` - Raw action logits for the four maze directions.

Returns: Chosen direction plus softmax and entropy diagnostics.
