# browser-entry/worker-channel

## browser-entry/worker-channel/worker-channel.types.ts

### WorkerChannelGenerationPayload

Aliases the generation payload contract returned by the worker.

### WorkerChannelMessage

Aliases the shared worker message union for worker-channel modules.

### WorkerChannelPlaybackStepPayload

Aliases the playback-step payload contract returned by the worker.

### WorkerChannelPlaybackStepRequest

Request payload sent when asking the worker to advance playback simulation.

## browser-entry/worker-channel/worker-channel.ts

### createEvolutionWorker

`() => Worker`

Creates the evolution worker used to keep heavy NEAT compute off the UI thread.

Returns: Initialized worker instance.

### requestWorkerGeneration

`(evolutionWorker: Worker) => Promise<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionGenerationPayload>`

Waits for the next generation payload emitted by the evolution worker.

Parameters:
- `evolutionWorker` - - Worker emitting generation-ready messages.

Returns: Next generation payload.

### requestWorkerPlaybackStep

`(evolutionWorker: Worker, playbackStepRequest: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/worker-channel/worker-channel.types").WorkerChannelPlaybackStepRequest) => Promise<{ snapshot: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }>`

Requests one playback batch step from the worker.

Parameters:
- `evolutionWorker` - - Worker that owns playback simulation state.
- `playbackStepRequest` - - Requested simulation budget and viewport width.

Returns: Playback-step payload including snapshot and completion marker.

## browser-entry/worker-channel/worker-channel.errors.ts

### worker-channel.errors

Error raised when the evolution worker responds with an explicit protocol error payload.

### createWorkerChannelResponseError

`(message: string) => Error`

Converts worker protocol error payloads into typed worker-channel errors.

Parameters:
- `message` - - Message supplied by the worker error payload.

Returns: Typed worker-channel protocol error.

### resolveWorkerChannelRuntimeError

`(errorLike: unknown, fallbackMessage: string) => Error`

Resolves a worker `ErrorEvent` into a normalized `Error` instance.

Parameters:
- `errorLike` - - Optional `event.error` payload.
- `fallbackMessage` - - Fallback message from `event.message`.

Returns: Normalized runtime error.

### WorkerChannelResponseError

Error raised when the evolution worker responds with an explicit protocol error payload.

## browser-entry/worker-channel/worker-channel.url.service.ts

### worker-channel.url.service

Resolves the evolution worker bundle URL relative to the active browser-entry bundle.

@returns Absolute URL string for `flappy-evolution.worker.bundle.js`.

### resolveEvolutionWorkerBundleUrl

`() => string`

Resolves the evolution worker bundle URL relative to the active browser-entry bundle.

Returns: Absolute URL string for `flappy-evolution.worker.bundle.js`.

## browser-entry/worker-channel/worker-channel.request.service.ts

### requestWorkerResponse

`(options: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/worker-channel/worker-channel.request.service").WorkerChannelRequestOptions<ResponsePayload>) => Promise<ResponsePayload>`

Sends one request to the worker and resolves with the first matching response payload.

Parameters:
- `options` - - Worker request options and response resolver callback.

Returns: Promise resolving with the matched worker response payload.

### WorkerChannelRequestMessage

Message shape sent to the worker request channel.

### WorkerChannelRequestOptions

Configuration used for one worker request/response lifecycle.

## browser-entry/worker-channel/worker-channel.playback.service.ts

### requestWorkerPlaybackStep

`(evolutionWorker: Worker, playbackStepRequest: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/worker-channel/worker-channel.types").WorkerChannelPlaybackStepRequest) => Promise<{ snapshot: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }>`

Requests one playback batch step from the worker channel.

Parameters:
- `evolutionWorker` - - Worker that owns playback simulation state.
- `playbackStepRequest` - - Requested simulation budget and viewport size.

Returns: Playback-step payload including snapshot and completion marker.

## browser-entry/worker-channel/worker-channel.generation.service.ts

### requestWorkerGeneration

`(evolutionWorker: Worker) => Promise<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionGenerationPayload>`

Requests the next evolved generation payload from the worker channel.

Parameters:
- `evolutionWorker` - - Worker emitting generation-ready messages.

Returns: Next generation payload.
