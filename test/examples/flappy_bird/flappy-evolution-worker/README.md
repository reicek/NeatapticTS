# flappy-evolution-worker

## flappy-evolution-worker/flappy-evolution-worker.types.ts

### SerializedNetwork

Loose JSON-compatible network payload used by worker messages.

### WorkerErrorMessage

Worker error response message.

### WorkerFrameBirdSnapshot

Render-only bird snapshot DTO posted to host.

### WorkerFramePipeSnapshot

Render-only pipe snapshot DTO posted to host.

### WorkerGenerationReadyMessage

Worker generation-ready response message.

### WorkerHeuristicObservationFeatures

Structured features used by heuristic generation-0 teacher policy.

### WorkerInitMessage

Worker init request message.

### WorkerPlaybackFrameSnapshot

Full frame snapshot payload posted to host.

### WorkerPlaybackState

Mutable simulation state stored between worker playback requests.

### WorkerPlaybackStepMessage

Worker playback-step response message.

### WorkerPopulationBird

Mutable bird state tracked by the worker playback simulation.

### WorkerPopulationPipe

Mutable pipe state tracked by the worker playback simulation.

### WorkerRequestGenerationMessage

Worker request asking to evolve one generation.

### WorkerRequestMessage

Union of inbound worker request messages.

### WorkerRequestPlaybackStepMessage

Worker request asking to advance playback by N simulation steps.

### WorkerResponseMessage

Union of outbound worker response messages.

### WorkerStartPlaybackMessage

Worker request asking to initialize playback state.

### WorkerStopMessage

Worker stop request message.

## flappy-evolution-worker/flappy-evolution-worker.simulation.types.ts

### WorkerPlaybackFrameContext

Shared mutable inputs for one worker playback frame simulation pass.

## flappy-evolution-worker/flappy-evolution-worker.ts

### beginWorkerGenerationRequest

`(workerMutableRuntimeState: WorkerMutableRuntimeState) => void`

Begins one asynchronous generation request and captures failures.

Parameters:

- `workerMutableRuntimeState` - - Mutable worker runtime state.

Returns: Nothing.

### beginWorkerInitialization

`(workerMutableRuntimeState: WorkerMutableRuntimeState, initPayload: { populationSize: number; elitismCount: number; rngSeed: number; }) => void`

Begins worker initialization and captures asynchronous failures.

Parameters:

- `workerMutableRuntimeState` - - Mutable worker runtime state.
- `initPayload` - - Initialization payload.

Returns: Nothing.

### beginWorkerPlayback

`(workerMutableRuntimeState: WorkerMutableRuntimeState, payload: { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }) => void`

Begins a new playback session from the current evolved population.

Parameters:

- `workerMutableRuntimeState` - - Mutable worker runtime state.
- `payload` - - Playback start payload.

Returns: Nothing.

### createWorkerMessageHandler

`(workerMutableRuntimeState: WorkerMutableRuntimeState) => (event: MessageEvent<import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerRequestMessage>) => void`

Creates the top-level worker message handler.

Parameters:

- `workerMutableRuntimeState` - - Mutable worker runtime state.

Returns: Worker message handler.

### createWorkerMutableRuntimeState

`() => WorkerMutableRuntimeState`

Creates the mutable worker runtime state container.

Returns: Mutable worker runtime state.

### createWorkerProtocolHandlers

`(workerMutableRuntimeState: WorkerMutableRuntimeState) => { markStopped: () => void; beginInitialization: (payload: { populationSize: number; elitismCount: number; rngSeed: number; }) => void; beginGenerationRequest: () => void; hasPopulation: () => boolean; startPlayback: (payload: { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }) => void; hasPlaybackState: () => boolean; processPlaybackStep: (payload: { simulationSteps: number; visibleWorldWidthPx: number; visibleWorldHeightPx: number; }) => void; postWorkerMessage: typeof postWorkerMessage; }`

Creates protocol handlers bound to the mutable worker runtime state.

Parameters:

- `workerMutableRuntimeState` - - Mutable worker runtime state.

Returns: Protocol handler bundle.

### evolveAndPublishGeneration

`(workerMutableRuntimeState: WorkerMutableRuntimeState) => Promise<void>`

Evolves one generation and publishes the best-network summary message.

Educational note:
This method is the orchestration seam between evolutionary search and
browser rendering: it runs evolution, snapshots the population, and emits
a compact payload for UI state updates.

Returns: Promise resolved after generation payload is posted.

### initializeRuntime

`(workerMutableRuntimeState: WorkerMutableRuntimeState, initPayload: { populationSize: number; elitismCount: number; rngSeed: number; }) => Promise<void>`

Initializes the worker-local NEAT runtime used by browser evolution playback.

Educational note:
The runtime is configured once with deterministic RNG state and a lightweight
early-termination fitness rollout. Keeping this setup centralized helps ensure
reproducibility between runs and keeps host<->worker contracts simple.

Parameters:

- `initPayload` - - Initialization values from the browser host.

Returns: Promise resolved when runtime setup is complete.

### postWorkerMessage

`(workerMessage: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerResponseMessage) => void`

Posts a typed message from worker to host.

Parameters:

- `workerMessage` - - Outbound worker response payload.

Returns: Nothing.

### processWorkerPlaybackStepRequest

`(workerMutableRuntimeState: WorkerMutableRuntimeState, playbackStepPayload: { simulationSteps: number; visibleWorldWidthPx: number; visibleWorldHeightPx: number; }) => void`

Advances playback by a host-requested number of simulation steps.

Educational note:
The browser host can request multiple simulation steps per RAF to trade visual
smoothness against throughput. This function keeps that loop deterministic and
emits one compact snapshot payload per request.

Parameters:

- `playbackStepPayload` - - Host-selected simulation-step budget and viewport.

Returns: Nothing.

## flappy-evolution-worker/flappy-evolution-worker.errors.ts

### createWorkerErrorMessage

`(message: string) => import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerErrorMessage`

Creates a typed worker error response payload from a message string.

Parameters:

- `message` - - Error message text.

Returns: Worker error response message.

### createWorkerErrorMessageFromUnknown

`(error: unknown) => import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerErrorMessage`

Creates a typed worker error response payload from an unknown thrown value.

Parameters:

- `error` - - Unknown thrown value.

Returns: Worker error response message.

### FLAPPY_WORKER_INIT_FAILED_ERROR_MESSAGE

### FLAPPY_WORKER_PLAYBACK_START_REQUIRES_GENERATION_ERROR_MESSAGE

### FLAPPY_WORKER_PLAYBACK_STEP_REQUIRES_START_ERROR_MESSAGE

### resolveWorkerUnknownErrorMessage

`(error: unknown) => string`

Resolves unknown error-like values into display-safe worker error messages.

Parameters:

- `error` - - Unknown error value thrown by worker logic.

Returns: Normalized error message string.

## flappy-evolution-worker/flappy-evolution-worker.constants.ts

### flappy-evolution-worker.constants

Synthetic sample count used for generation-0 warm-start pretraining.

### FLAPPY_WORKER_GEN0_PRETRAIN_BATCH_SIZE

### FLAPPY_WORKER_GEN0_PRETRAIN_BIAS_NOISE_STDDEV

### FLAPPY_WORKER_GEN0_PRETRAIN_ITERATIONS

### FLAPPY_WORKER_GEN0_PRETRAIN_RATE

### FLAPPY_WORKER_GEN0_PRETRAIN_SAMPLE_COUNT

### FLAPPY_WORKER_GEN0_PRETRAIN_VISIBLE_WORLD_WIDTH_PX

### FLAPPY_WORKER_GEN0_PRETRAIN_WEIGHT_NOISE_STDDEV

## flappy-evolution-worker/flappy-evolution-worker.runtime.service.ts

### createInitializedWorkerRuntime

`(initPayload: { populationSize: number; elitismCount: number; rngSeed: number; }) => import("src/neat").default`

Creates and configures the worker-local NEAT runtime used by browser evolution playback.

Parameters:

- `initPayload` - - Initialization values from the browser host.

Returns: Initialized NEAT runtime.

## flappy-evolution-worker/flappy-evolution-worker.playback.service.ts

### beginWorkerPlaybackSession

`(options: { currentPopulation: import("src/architecture/network").default[]; payload: { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }; createPopulationRenderState: (networks: import("src/architecture/network").default[], rng: import("test/examples/flappy_bird/rng").FlappyRng, initialVisibleWorldWidthPx: number, initialVisibleWorldHeightPx: number) => import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState; }) => { currentPlaybackState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState; currentPlaybackRng: import("test/examples/flappy_bird/rng").FlappyRng; playbackWinnerIndex: number; }`

Creates a fresh worker playback session state from the current evolved population.

Parameters:

- `currentPopulation` - - Current evolved population.
- `payload` - - Playback start viewport payload.
- `createPopulationRenderState` - - Callback that builds initial simulation state.

Returns: Playback runtime state and deterministic RNG.

### processWorkerPlaybackStep

`(options: { playbackStepPayload: { simulationSteps: number; visibleWorldWidthPx: number; visibleWorldHeightPx: number; }; currentPlaybackState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState; currentPlaybackRng: import("test/examples/flappy_bird/rng").FlappyRng; currentPopulation: import("src/architecture/network").default[]; neatRuntime: import("src/neat").default | undefined; stepPopulationFrame: (renderState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState, rng: import("test/examples/flappy_bird/rng").FlappyRng, difficultyProfile: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedDifficultyProfile) => number; createPlaybackSnapshot: (playbackState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState) => import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackFrameSnapshot; postWorkerMessage: (workerMessage: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerResponseMessage) => void; }) => { currentPlaybackState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState | undefined; currentPlaybackRng: import("test/examples/flappy_bird/rng").FlappyRng | undefined; currentPopulation: import("src/architecture/network").default[]; playbackWinnerIndex: number; }`

Processes one worker playback-step request including completion/finalization logic.

Parameters:

- `options` - - Playback step dependencies and mutable runtime state.

Returns: Updated playback runtime state after processing this step.

## flappy-evolution-worker/flappy-evolution-worker.protocol.service.ts

### routeWorkerProtocolMessage

`(workerMessage: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerRequestMessage, handlers: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.protocol.service").WorkerProtocolHandlers) => void`

Routes one inbound worker request message to the corresponding runtime action.

Parameters:

- `workerMessage` - - Inbound worker request payload.
- `handlers` - - Runtime action callbacks and state probes.

Returns: Nothing.

### WorkerProtocolHandlers

Callback bundle used by worker protocol routing.

## flappy-evolution-worker/flappy-evolution-worker.evolution.service.ts

### evolveAndBuildGenerationReadyMessage

`(options: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.evolution.service").WorkerEvolutionServiceOptions) => Promise<import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerGenerationReadyMessage>`

Evolves one generation and creates the compact generation-ready response payload.

Parameters:

- `options` - - Evolution dependencies and runtime state accessors.

Returns: Generation-ready worker response payload.

### WorkerEvolutionServiceOptions

Dependencies required to evolve one generation and prepare host payload output.

## flappy-evolution-worker/flappy-evolution-worker.warm-start.service.ts

### applyTemplateWeightsWithNoise

`(genome: import("src/architecture/network").default, template: import("src/architecture/network").default, rng: import("test/examples/flappy_bird/rng").FlappyRng, noise: { weightStdDev: number; biasStdDev: number; }) => void`

Copies template parameters into a genome and injects small Gaussian noise.

Parameters:

- `genome` - - Target genome to mutate in-place.
- `template` - - Trained template source network.
- `rng` - - Deterministic random source for noise sampling.
- `noise` - - Standard deviations for weight and bias perturbations.

Returns: Nothing.

### buildHeuristicPretrainSet

`(rng: import("test/examples/flappy_bird/rng").FlappyRng, sampleCount: number) => { input: number[]; output: number[]; }[]`

Builds synthetic supervised samples for generation-0 behavior cloning.

Parameters:

- `rng` - - Deterministic random source.
- `sampleCount` - - Requested number of synthetic samples.

Returns: Supervised dataset of input/output pairs.

### resolveHeuristicTeacherFlapDecision

`(features: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures) => boolean`

Heuristic teacher policy used to label synthetic pretraining samples.

Parameters:

- `features` - - Structured observation features for one synthetic state.

Returns: True when the teacher says to flap.

### sampleGaussian

`(rng: import("test/examples/flappy_bird/rng").FlappyRng) => number`

Samples one standard-normal value using the Box-Muller transform.

Parameters:

- `rng` - - Deterministic random source.

Returns: One approximately standard-normal random value.

### warmStartWorkerGenerationZeroIfNeeded

`(neatController: import("src/neat").default, warmStartState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.warm-start.service").WorkerWarmStartState) => Promise<void>`

Applies a one-time generation-0 warm-start to improve initial demo quality.

Educational note:
The worker entry should stay protocol-first. This service owns the short
supervised bootstrap pass that nudges generation 0 away from pure noise while
preserving the later NEAT-driven search loop.

Parameters:

- `neatController` - - Initialized NEAT runtime.
- `warmStartState` - - Mutable warm-start lifecycle state.

Returns: Promise resolved when warm-start evaluation finishes.

### WorkerWarmStartState

State carried between generation requests for one worker runtime.

## flappy-evolution-worker/flappy-evolution-worker.simulation.frame.service.ts

### advanceBirdPhysics

`(frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => void`

Integrates bird velocity and vertical motion for one control substep.

Parameters:

- `frameContext` - - Shared frame context for this logical frame.

Returns: Nothing.

### advancePipes

`(frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => void`

Advances all visible pipes and culls those that have left the camera window.

Parameters:

- `frameContext` - - Shared frame context for this logical frame.

Returns: Nothing.

### commitPassedPipeProgress

`(bird: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPopulationBird, pipe: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPopulationPipe) => void`

Commits one passed-pipe progress increment for a bird when eligible.

Parameters:

- `bird` - - Mutable bird state.
- `pipe` - - Pipe candidate to mark as passed.

Returns: Nothing.

### incrementLivingBirdFrameCounters

`(renderState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState) => void`

Increments survival counters for birds that remain active at frame start.

Parameters:

- `renderState` - - Mutable playback state.

Returns: Nothing.

### resolveBirdCollisionAgainstPipe

`(bird: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPopulationBird, pipe: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPopulationPipe, frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => boolean`

Resolves whether a bird collides with one pipe corridor during this substep.

Parameters:

- `bird` - - Mutable bird state.
- `pipe` - - Pipe candidate to test.
- `frameContext` - - Shared frame context for this logical frame.

Returns: `true` when the bird overlaps the pipe body instead of the gap.

### resolveBirdControlActions

`(frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => number`

Runs policy evaluation and commits the resulting observation memory updates.

Parameters:

- `frameContext` - - Shared frame context for this logical frame.

Returns: Number of activation calls performed in the substep.

### resolveBirdOutOfBounds

`(bird: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPopulationBird, visibleWorldHeightPx: number) => boolean`

Resolves whether a bird has exceeded the vertical play area.

Parameters:

- `bird` - - Mutable bird state.
- `visibleWorldHeightPx` - - Current visible world height.

Returns: `true` when the bird is outside the vertical bounds.

### resolveBirdTerminationAndProgress

`(frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => void`

Resolves bird deaths and passed-pipe progress after motion is applied.

Parameters:

- `frameContext` - - Shared frame context for this logical frame.

Returns: Nothing.

### resolveCameraLeftXPx

`(visibleWorldWidthPx: number) => number`

Resolves the current left-edge of the visible world in world-space pixels.

Parameters:

- `visibleWorldWidthPx` - - Current visible world width.

Returns: Left edge x-position in world coordinates.

### runWorkerPopulationControlSubstep

`(frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => number`

Advances one control substep of the worker playback simulation.

Parameters:

- `frameContext` - - Shared frame context for this logical frame.

Returns: Number of activation calls performed in the substep.

### spawnPipeIfNeeded

`(frameContext: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.simulation.types").WorkerPlaybackFrameContext) => void`

Spawns a new pipe when the substep budget crosses the spawn boundary.

Parameters:

- `frameContext` - - Shared frame context for this logical frame.

Returns: Nothing.

### stepWorkerPopulationFrame

`(renderState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState, rng: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike, difficultyProfile: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedDifficultyProfile) => number`

Advances the whole population simulation by one logical frame.

Parameters:

- `renderState` - - Mutable simulation state.
- `rng` - - Deterministic random source for spawn variation.
- `difficultyProfile` - - Active dynamic difficulty profile.

Returns: Number of policy activation calls made in this frame.

## flappy-evolution-worker/flappy-evolution-worker.snapshot.utils.ts

### createWorkerPlaybackSnapshot

`(playbackState: import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState) => import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackFrameSnapshot`

Creates a serializable snapshot of current playback state.

Workers should send only structured-clone-safe payloads. This helper strips
runtime-only references (e.g., network instances, sets) and keeps only
renderer-relevant fields.

Parameters:

- `playbackState` - - Current mutable playback state.

Returns: Immutable frame snapshot for the host.

## flappy-evolution-worker/flappy-evolution-worker.simulation.utils.ts

### createWorkerPopulationRenderState

`(networks: import("src/architecture/network").default[], rng: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike, initialVisibleWorldWidthPx: number, initialVisibleWorldHeightPx: number) => import("test/examples/flappy_bird/flappy-evolution-worker/flappy-evolution-worker.types").WorkerPlaybackState`

Creates initial playback state for a population of networks.

Parameters:

- `networks` - - Population to visualize.
- `rng` - - Deterministic random source.
- `initialVisibleWorldWidthPx` - - Initial viewport width from host.
- `initialVisibleWorldHeightPx` - - Initial viewport height from host.

Returns: Fresh mutable playback state.
