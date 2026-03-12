# environment

## environment/environment.types.ts

### FlappyBird

Bird kinematic state for one simulation frame.

The environment keeps only the minimum physics state needed to advance the
episode: vertical position and vertical velocity.

### FlappyDifficultyScale

Difficulty scale used by the curriculum scheduler.

- `0` means easiest profile (wide gaps, slower pipes).
- `1` means fully adaptive profile based on passed pipes.

Values between `0` and `1` interpolate between those extremes, which lets the
trainer or environment caller dial curriculum strength continuously.

### FlappyGameState

Full simulation state for one Flappy episode.

Educational note:
This is the canonical single-episode world state used by evaluation and some
trainer-facing helpers. It is intentionally compact so stepping the world is
deterministic and easy to inspect.

### FlappyObservationFeatures

Structured observation features used to build the neural-network input vector.

Re-exported from shared simulation utilities so trainer and browser paths
stay synchronized as the observation schema evolves.

### FlappyPipe

Pipe obstacle definition.

Pipes move from right to left. The bird scores once per pipe when the pipe
completely crosses the bird x-position.

This is the environment-owned pipe state, distinct from the packed snapshot
transport shapes used by the browser worker.

## environment/environment.constants.ts

### FLAPPY_ENVIRONMENT_DEFAULT_CONTROL_SUBSTEPS_PER_FRAME

### FLAPPY_ENVIRONMENT_DEFAULT_DIFFICULTY_SCALE

### FLAPPY_ENVIRONMENT_MAX_FRAMES_PER_EPISODE

## environment/environment.step.service.ts

### stepFlappyState

`(state: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState, rng: import("test/examples/flappy_bird/rng").FlappyRng, flap: boolean, difficultyScale: number) => void`

Advance the simulation by one frame.

This is the simplest stepping surface: one logical frame and one flap choice.
More advanced callers can use the control-substep variant below.

Parameters:
- `state` - - Mutable state object to update in-place.
- `rng` - - Random source used to spawn pipes.
- `flap` - - If true, applies an upward velocity impulse.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].

Returns: Nothing.

### stepFlappyStateWithControlSubsteps

`(state: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState, rng: import("test/examples/flappy_bird/rng").FlappyRng, shouldFlapForSubstep: () => boolean, difficultyScale: number, controlSubstepsPerFrame: number) => void`

Advance one logical frame using multiple control/physics substeps.

This allows policies to react multiple times before `frameIndex` advances,
improving responsiveness in high-difficulty scenarios.

Educational note:
Splitting a logical frame into smaller control steps is a simple numerical
stability trick. It reduces the chance that fast pipes or large velocity
updates make the environment feel artificially coarse.

For background reading, the Wikipedia article on "numerical integration"
provides the general idea behind updating continuous motion in small steps.

Parameters:
- `state` - - Mutable state object to update in-place.
- `rng` - - Random source used to spawn pipes.
- `shouldFlapForSubstep` - - Callback deciding flap action per substep.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].
- `controlSubstepsPerFrame` - - Number of substeps to run this frame.

Returns: Nothing.

## environment/environment.state.service.ts

### createInitialFlappyState

`(rng: import("test/examples/flappy_bird/rng").FlappyRng) => import("test/examples/flappy_bird/environment/environment.types").FlappyGameState`

Create a fresh Flappy Bird episode state.

Educational note:
A new episode starts with one initial pipe already materialized so the first
observation is meaningful immediately. That avoids a cold-start phase where a
policy would receive mostly empty-space inputs.

Parameters:
- `rng` - - Random source used to generate initial pipe configuration.

Returns: Initial state for one deterministic rollout.

## environment/environment.collision.utils.ts

### updateCollisionAndProgressState

`(state: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState) => void`

Apply out-of-bounds, pipe-collision, and pass-credit rules for one substep.

Educational note:
Collision resolution and progress credit live together because both depend on
the same bird-vs-pipe geometry for the current substep. Keeping them in one
place helps the environment avoid inconsistent "passed but also collided"
edge cases.

Parameters:
- `state` - - Mutable simulation state to update in-place.

Returns: Nothing.

## environment/environment.observation.utils.ts

### getFlappyObservation

`(state: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState, difficultyScale: number) => number[]`

Generate the network observation vector for the current state.

Educational note:
This helper is the environment-facing bridge into the shared observation
system. It keeps the environment API simple while ensuring evaluation,
training, and browser playback all derive their policy inputs from the same
feature definitions.

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

`(state: import("test/examples/flappy_bird/environment/environment.types").FlappyGameState, difficultyScale: number) => import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures`

Resolve structured observation features for policy input and reward shaping.

Parameters:
- `state` - - Current state.
- `difficultyScale` - - Curriculum difficulty scale in [0, 1].

Returns: Named feature object.
