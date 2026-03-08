# mazeMovement/policy

## mazeMovement/policy/mazeMovement.policy.ts

### mazeMovement.policy

Action policy helpers for the dedicated mazeMovement module.

This file owns direction selection, epsilon handling, short-horizon policy
overrides, and saturation-driven bias control.

### applyMazeMovementEpsilonExploration

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, encodedMaze: number[][], coordinateScratch: Int32Array<ArrayBufferLike>) => void`

Apply epsilon-greedy exploration to the current action choice.

Parameters:
- `state` - - Mutable simulation state for the active run.
- `encodedMaze` - - Maze grid used for move validity checks.
- `coordinateScratch` - - Reused coordinate scratch buffer.

### applyMazeMovementForcedExploration

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, encodedMaze: number[][], coordinateScratch: Int32Array<ArrayBufferLike>) => void`

Force a random valid move when the policy has stalled with repeated no-move outputs.

Parameters:
- `state` - - Mutable simulation state for the active run.
- `encodedMaze` - - Maze grid used for move validity checks.
- `coordinateScratch` - - Reused coordinate scratch buffer.

### applyMazeMovementProximityGreedy

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, encodedMaze: number[][], distanceMap: number[][] | undefined, coordinateScratch: Int32Array<ArrayBufferLike>) => void`

Apply the short-horizon proximity-greedy override near the maze exit.

Parameters:
- `state` - - Mutable simulation state for the active run.
- `encodedMaze` - - Maze grid used for move validity checks.
- `distanceMap` - - Optional precomputed distance map.
- `coordinateScratch` - - Reused coordinate scratch buffer.

### applyMazeMovementSaturationAndBiasAdjust

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, outputs: number[], network: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork, coordinateScratch: Int32Array<ArrayBufferLike>) => void`

Detect saturation and optionally damp output-node biases.

Parameters:
- `state` - - Mutable simulation state for the active run.
- `outputs` - - Raw network logits for the current step.
- `network` - - Policy network that produced the logits.
- `coordinateScratch` - - Reused scratch buffer for temporary penalties.

### computeMazeMovementEpsilon

`(stepNumber: number, stepsSinceImprovement: number, distHere: number, saturations: number) => number`

Compute the adaptive epsilon used for policy exploration.

Parameters:
- `stepNumber` - - Global step number inside the active simulation.
- `stepsSinceImprovement` - - Number of steps without improvement.
- `distHere` - - Current distance to goal for the active position.
- `saturations` - - Rolling saturation count from the shared run state.

Returns: Exploration epsilon in the range `[0, 1]`.

### decideMazeMovementDirection

`(state: import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").SimulationState, network: import("C:/NeatapticTS/test/examples/asciiMaze/interfaces").INetwork, coordinateScratch: Int32Array<ArrayBufferLike>) => void`

Activate the network, record output history, and choose the next direction.

Parameters:
- `state` - - Mutable simulation state for the active run.
- `network` - - Policy network used for the current step.

### selectMazeMovementDirection

`(outputs: number[]) => import("C:/NeatapticTS/test/examples/asciiMaze/mazeMovement/mazeMovement.types").DirectionSelectionStats`

Convert raw network outputs into a chosen direction plus diagnostics.

Parameters:
- `outputs` - - Raw action logits for the four maze directions.

Returns: Chosen direction plus softmax and entropy diagnostics.
