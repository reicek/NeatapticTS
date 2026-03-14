# browser-entry/worker-channel

Public worker-channel facade for the Flappy Bird browser runtime.

This boundary wraps the lower-level message protocol in a smaller browser API:
create the worker, request a generation result, or request the next playback
step. The point is to keep the rest of the UI code thinking in terms of
intent rather than raw `postMessage` plumbing.

Minimal usage sketch:
```ts
const evolutionWorker = createEvolutionWorker();
const generation = await requestWorkerGeneration(evolutionWorker);
```

## browser-entry/worker-channel/worker-channel.types.ts

Shared protocol contracts for the browser-entry worker channel.

These types define the thin translation layer between generic worker messages
and the specific request/response flows used by the Flappy Bird browser UI.

If you want background reading, the Wikipedia article on "message passing"
gives the right mental model for this boundary.

### WorkerChannelGenerationPayload

Aliases the generation payload contract returned by the worker.

This payload arrives when one NEAT generation has finished evolving and the
browser is ready to update its "best so far" view.

### WorkerChannelMessage

Aliases the shared worker message union for worker-channel modules.

Keeping the alias local makes submodules read as protocol-focused code rather
than browser-entry plumbing.

### WorkerChannelPlaybackStepPayload

Aliases the playback-step payload contract returned by the worker.

This is the browser-side shape of one streamed playback frame plus its
accompanying aggregate telemetry.

### WorkerChannelPlaybackStepRequest

Request payload sent when asking the worker to advance playback simulation.

The request declares both simulation budget and viewport dimensions so the
worker can package a frame that already matches the browser's current canvas
world.

## browser-entry/worker-channel/worker-channel.ts

### createEvolutionWorker

`() => Worker`

Creates the evolution worker used to keep heavy NEAT compute off the UI thread.

Returns: Initialized worker instance.

### requestWorkerGeneration

`(evolutionWorker: Worker) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionGenerationPayload>`

Waits for the next generation payload emitted by the evolution worker.

Parameters:
- `evolutionWorker` - - Worker emitting generation-ready messages.

Returns: Next generation payload.

### requestWorkerPlaybackStep

`(evolutionWorker: Worker, playbackStepRequest: import("test/examples/flappy_bird/browser-entry/worker-channel/worker-channel.types").WorkerChannelPlaybackStepRequest) => Promise<{ requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }>`

Requests one playback batch step from the worker.

Parameters:
- `evolutionWorker` - - Worker that owns playback simulation state.
- `playbackStepRequest` - - Requested simulation budget and viewport width.

Returns: Playback-step payload including snapshot and completion marker.

## browser-entry/worker-channel/worker-channel.errors.ts

Error raised when the evolution worker responds with an explicit protocol error payload.

Protocol errors are different from runtime worker crashes: the worker is
alive, but it is explicitly telling the browser that the requested operation
could not be completed.

### createWorkerChannelResponseError

`(message: string) => Error`

Converts worker protocol error payloads into typed worker-channel errors.

Using a dedicated error class makes it easier for browser code to distinguish
"worker rejected my request" from "the worker crashed".

Parameters:
- `message` - - Message supplied by the worker error payload.

Returns: Typed worker-channel protocol error.

### resolveWorkerChannelRuntimeError

`(errorLike: unknown, fallbackMessage: string) => Error`

Resolves a worker `ErrorEvent` into a normalized `Error` instance.

Browser worker errors are not always surfaced as proper `Error` objects, so
this helper converts the event payload into a predictable error shape before
it escapes the channel layer.

Parameters:
- `errorLike` - - Optional `event.error` payload.
- `fallbackMessage` - - Fallback message from `event.message`.

Returns: Normalized runtime error.

### WorkerChannelResponseError

Error raised when the evolution worker responds with an explicit protocol error payload.

Protocol errors are different from runtime worker crashes: the worker is
alive, but it is explicitly telling the browser that the requested operation
could not be completed.

## browser-entry/worker-channel/worker-channel.url.service.ts

Resolves the evolution worker bundle URL relative to the active browser-entry bundle.

The browser bundle and worker bundle are emitted side-by-side by the docs/demo
build. Resolving the worker URL relative to the currently loaded browser
bundle keeps the demo portable across local files, static hosting, and docs
builds without hard-coding absolute paths.

### resolveEvolutionWorkerBundleUrl

`() => string`

Resolves the evolution worker bundle URL relative to the active browser-entry bundle.

The browser bundle and worker bundle are emitted side-by-side by the docs/demo
build. Resolving the worker URL relative to the currently loaded browser
bundle keeps the demo portable across local files, static hosting, and docs
builds without hard-coding absolute paths.

Returns: Absolute URL string for `flappy-evolution.worker.bundle.js`.

## browser-entry/worker-channel/worker-channel.request.service.ts

Generic request/response helper for worker-channel interactions.

This module implements a small RPC-like pattern on top of browser worker
events. The browser sends one message, listens for the first matching reply,
and normalizes protocol failures into ordinary `Error` instances.

### requestWorkerResponse

`(options: import("test/examples/flappy_bird/browser-entry/worker-channel/worker-channel.request.service").WorkerChannelRequestOptions<ResponsePayload>) => Promise<ResponsePayload>`

Sends one request to the worker and resolves with the first matching response payload.

This keeps generation requests simple: the caller describes the response it is
waiting for, and this helper handles transient listeners, protocol errors,
and runtime worker failures.

Parameters:
- `options` - - Worker request options and response resolver callback.

Returns: Promise resolving with the matched worker response payload.

Example:

```ts
const generation = await requestWorkerResponse({
  evolutionWorker,
  requestMessage: { type: 'request-generation' },
  resolveResponsePayload: (message) =>
    message.type === 'generation-ready' ? message.payload : undefined,
});
```

### WorkerChannelRequestMessage

Message shape sent to the worker request channel.

The worker protocol stays stringly-typed at the transport edge on purpose so
the message stream is easy to inspect during debugging.

### WorkerChannelRequestOptions

Configuration used for one worker request/response lifecycle.

Callers provide the worker, the outbound message, and the predicate that says
which inbound worker message should satisfy the request.

## browser-entry/worker-channel/worker-channel.playback.service.ts

### PendingPlaybackRequest

Stateful playback request channel for one evolution worker.

Playback is intentionally handled differently from generation requests. The
browser asks for a sequence of incremental frames, and the channel keeps a
small amount of per-worker state so each request can be matched to the
correct reply.

### requestWorkerPlaybackStep

`(evolutionWorker: Worker, playbackStepRequest: import("test/examples/flappy_bird/browser-entry/worker-channel/worker-channel.types").WorkerChannelPlaybackStepRequest) => Promise<{ requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }>`

Requests one playback batch step from the worker channel.

A playback-step request is effectively "advance the simulation by this many
internal steps, then send me a packed frame I can render". The request is
tagged with a monotonically increasing request id so stale or out-of-order
replies can be ignored safely.

Parameters:
- `evolutionWorker` - - Worker that owns playback simulation state.
- `playbackStepRequest` - - Requested simulation budget and viewport size.

Returns: Playback-step payload including snapshot and completion marker.

Example:

```ts
const playbackPayload = await requestWorkerPlaybackStep(evolutionWorker, {
  simulationSteps: 2,
  visibleWorldWidthPx: 640,
  visibleWorldHeightPx: 480,
});
```

### resolvePlaybackWorkerChannelState

`(evolutionWorker: Worker) => PlaybackWorkerChannelState`

Resolves persistent playback worker-channel state for one worker instance.

The state tracks request ids and the one allowed in-flight playback request.
That single-flight rule keeps the protocol simple and avoids ambiguous frame
ordering on the browser side.

Parameters:
- `evolutionWorker` - - Worker that owns playback simulation state.

Returns: Persistent playback worker-channel state for the worker.

## browser-entry/worker-channel/worker-channel.generation.service.ts

Generation request helper for the browser-entry worker channel.

This module asks the worker to evolve until the next generation boundary and
then returns the summary payload the browser needs for HUD updates and best
network visualization.

### requestWorkerGeneration

`(evolutionWorker: Worker) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionGenerationPayload>`

Requests the next evolved generation payload from the worker channel.

Unlike playback streaming, generation evolution is a simple single-response
exchange: ask for the next generation and wait for the next
`generation-ready` message.

Parameters:
- `evolutionWorker` - - Worker emitting generation-ready messages.

Returns: Next generation payload.
