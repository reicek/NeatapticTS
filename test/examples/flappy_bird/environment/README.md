# environment

## environment/environment.types.ts

### FlappyBird

Bird kinematic state for one simulation frame.

### FlappyDifficultyScale

Difficulty scale used by the curriculum scheduler.

- `0` means easiest profile (wide gaps, slower pipes).
- `1` means fully adaptive profile based on passed pipes.

### FlappyGameState

Full simulation state for one Flappy episode.

### FlappyObservationFeatures

Structured observation features used to build the neural-network input vector.

Re-exported from shared simulation utilities so trainer and browser paths
stay synchronized as the observation schema evolves.

### FlappyPipe

Pipe obstacle definition.

Pipes move from right to left. The bird scores once per pipe when the pipe
completely crosses the bird x-position.

## environment/environment.constants.ts

### FLAPPY_ENVIRONMENT_DEFAULT_CONTROL_SUBSTEPS_PER_FRAME

### FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE

### FLAPPY_ENVIRONMENT_MAX_FRAMES_PER_EPISODE

## environment/environment.step.service.ts

### stepFlappyState

`(state: import("C:/NeatapticTS/test/examples/flappy_bird/environment/environment.types").FlappyGameState, rng: import("C:/NeatapticTS/test/examples/flappy_bird/rng").FlappyRng, flap: boolean, difficultyScale: number) => void`

Advance the simulation by one frame.

Parameters:
- `state` - - Mutable state object to update in-place.
- `rng` - - Random source used to spawn pipes.
- `flap` - - If true, applies an upward velocity impulse.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].

### stepFlappyStateWithControlSubsteps

`(state: import("C:/NeatapticTS/test/examples/flappy_bird/environment/environment.types").FlappyGameState, rng: import("C:/NeatapticTS/test/examples/flappy_bird/rng").FlappyRng, shouldFlapForSubstep: () => boolean, difficultyScale: number, controlSubstepsPerFrame: number) => void`

Advance one logical frame using multiple control/physics substeps.

This allows policies to react multiple times before `frameIndex` advances,
improving responsiveness in high-difficulty scenarios.

Parameters:
- `state` - - Mutable state object to update in-place.
- `rng` - - Random source used to spawn pipes.
- `shouldFlapForSubstep` - - Callback deciding flap action per substep.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].
- `controlSubstepsPerFrame` - - Number of substeps to run this frame.

Returns: Nothing.

## environment/environment.state.service.ts

### createInitialFlappyState

`(rng: import("C:/NeatapticTS/test/examples/flappy_bird/rng").FlappyRng) => import("C:/NeatapticTS/test/examples/flappy_bird/environment/environment.types").FlappyGameState`

Create a fresh Flappy Bird episode state.

Parameters:
- `rng` - - Random source used to generate initial pipe configuration.

Returns: Initial state for one deterministic rollout.

## environment/environment.collision.utils.ts

### updateCollisionAndProgressState

`(state: import("C:/NeatapticTS/test/examples/flappy_bird/environment/environment.types").FlappyGameState) => void`

Apply out-of-bounds, pipe-collision, and pass-credit rules for one substep.

Parameters:
- `state` - - Mutable simulation state to update in-place.

## environment/environment.observation.utils.ts

### getFlappyObservation

`(state: import("C:/NeatapticTS/test/examples/flappy_bird/environment/environment.types").FlappyGameState, difficultyScale: number) => number[]`

Generate the network observation vector for the current state.

Observation (12 numbers):
 1) bird y position normalized to [0, 1]
 2) bird vertical velocity normalized to [-1, 1]
 3) distance to next pipe normalized to [0, 1]
 4) delta (bird y - gap center y) normalized to [-1, 1]
 5) next pipe gap top normalized to [0, 1]
 6) next pipe gap bottom normalized to [0, 1]
 7) distance to second pipe normalized to [0, 1]
 8) delta to second gap center normalized to [-1, 1]
 9) time-to-next-pipe closeness normalized to [0, 1]
10) signed clearance relative to next gap normalized to [-1, 1]
11) required vertical velocity toward next gap center normalized to [-1, 1]
12) gap-center transition (next to second) normalized to [-1, 1]

Parameters:
- `state` - - Current state.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].

Returns: Input vector for the neural network.

### getFlappyObservationFeatures

`(state: import("C:/NeatapticTS/test/examples/flappy_bird/environment/environment.types").FlappyGameState, difficultyScale: number) => import("C:/NeatapticTS/test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures`

Resolve structured observation features for policy input and reward shaping.

Parameters:
- `state` - - Current state.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].

Returns: Named feature object.
