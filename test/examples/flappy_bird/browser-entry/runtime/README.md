# browser-entry/runtime

## browser-entry/runtime/runtime.types.ts

### RuntimeContainerTarget

Container argument accepted by the browser runtime start function.

Callers can either pass a host element directly or provide an element id for
late resolution inside the runtime startup path.

### RuntimeGlobalWindow

Browser `window` extension shape used for global runtime wiring.

This keeps the auto-start and compatibility globals typed without coupling
the runtime modules directly to ad hoc window-property access.

### RuntimeHostViewContext

Browser host view handles used by the runtime entry orchestration.

This is the typed bundle of canvas, HUD, and visualization handles returned
by the host layer once the browser DOM has been prepared.

### RuntimeMutableLifecycleState

Mutable lifecycle state used to coordinate stop semantics and completion.

The runtime loop is asynchronous and long-lived, so the browser keeps a small
shared lifecycle object for idempotent shutdown and completion signaling.

### RuntimeRunHandle

Core runtime contracts for the Flappy Bird browser demo.

The runtime boundary is where the browser-side application comes together:
DOM host setup, worker creation, telemetry wiring, lifecycle control, and the
configuration passed into the evolution loop.

### RuntimeStartConfig

Static runtime configuration resolved before the browser loop starts.

These values define the high-level NEAT and network shape used for the whole
browser session.

### RuntimeStartContext

Shared runtime startup dependencies created before evolution begins.

Once this context exists, the browser has everything it needs to launch the
actual evolution/playback loop.

## browser-entry/runtime/runtime.ts

### RuntimeRunHandle

Core runtime contracts for the Flappy Bird browser demo.

The runtime boundary is where the browser-side application comes together:
DOM host setup, worker creation, telemetry wiring, lifecycle control, and the
configuration passed into the evolution loop.

### start

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle>`

## browser-entry/runtime/runtime.errors.ts

### resolveRequiredRuntimeHostElement

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => HTMLElement`

Resolves and validates the browser runtime host element.

The runtime accepts either a string id or a concrete element so this helper
folds that loose input into one validated host node.

Parameters:
- `container` - - Element id or HTMLElement provided to runtime start.

Returns: Resolved host element.

### resolveRuntimeHudErrorStatus

`(error: unknown) => string`

Formats unknown runtime failures into a stable HUD status string.

The HUD should not need to understand arbitrary thrown values, so this helper
normalizes anything throwable into one readable status line.

Parameters:
- `error` - - Unknown runtime exception value.

Returns: Normalized status string for HUD output.

### RuntimeContainerNotFoundError

Runtime-specific error helpers for the browser entrypoint.

These errors normalize two user-facing failure modes: the browser cannot find
the requested host container, or the runtime needs to report an unexpected
failure back into the HUD.

## browser-entry/runtime/runtime.startup.service.ts

### createRuntimeStartConfig

`() => import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartConfig`

Resolves the static runtime configuration used during browser startup.

Centralizing the configuration fold here makes the runtime entry read as
orchestration instead of constant plumbing.

Returns: Runtime configuration derived from shared constants.

### createRuntimeStartContext

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext`

Runtime startup helpers for the Flappy Bird browser demo.

These functions cover the pre-loop phase: resolve the host container, build a
typed browser view, derive static config, create telemetry state, spawn the
worker, and paint the initial HUD before evolution begins.

### initializeRuntimeHud

`(runtimeStartContext: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext) => void`

Paints the initial runtime HUD values before the evolution loop starts.

The HUD is seeded immediately so the page communicates that startup is in
progress rather than appearing blank while the worker and loop are booting.

Parameters:
- `runtimeStartContext` - - Shared runtime start context.

Returns: Nothing.

## browser-entry/runtime/runtime.lifecycle.service.ts

### createRuntimeLifecycleState

`() => import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeMutableLifecycleState`

Lifecycle and teardown helpers for the browser runtime.

The runtime behaves like a small application process. These helpers create the
mutable state and public handle needed to stop it cleanly, terminate the
worker, and resolve the completion promise exactly once.

### createRuntimeRunHandle

`(runtimeStartContext: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext, runtimeLifecycleState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeMutableLifecycleState) => import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle`

Builds the public run handle and binds it to runtime teardown behavior.

The handle is the user-facing control surface for the demo. Internally it is
just a thin closure layer over the mutable lifecycle state and startup
context.

Parameters:
- `runtimeStartContext` - - Shared runtime start context.
- `runtimeLifecycleState` - - Mutable lifecycle state for stop semantics.

Returns: Public run handle exposed to callers.

## browser-entry/runtime/runtime.telemetry.service.ts

### createRuntimeTelemetryState

`() => import("test/examples/flappy_bird/browser-entry/runtime/runtime.telemetry.service").RuntimeTelemetryState`

Creates telemetry state and attaches optional minor-GC observer.

Instrumentation is feature-gated so the demo can run in a low-noise mode when
telemetry is not desired.

Returns: Initialized telemetry state.

### disconnectRuntimeTelemetry

`(telemetryState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.telemetry.service").RuntimeTelemetryState) => void`

Disconnects runtime telemetry observers.

This is part of runtime teardown and prevents instrumentation observers from
lingering after the demo has stopped.

Parameters:
- `telemetryState` - - Runtime telemetry state.

Returns: Nothing.

### resolveInitialRuntimeTelemetryHudValues

`() => { telemetryHeader: string; telemetryActivationsPerFrame: string; telemetrySimulationStepsPerRaf: string; telemetryHudUpdatesPerSecond: string; telemetryMinorGcPerMinute: string; }`

Resolves default telemetry HUD values used before first playback updates.

The initial values make the instrumentation section self-describing even
before the first playback frame arrives.

Returns: Initial telemetry field values.

### resolveRuntimeTelemetryHudValues

`(frameStats: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats, telemetryState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.telemetry.service").RuntimeTelemetryState) => { telemetryActivationsPerFrame: string; telemetrySimulationStepsPerRaf: string; telemetryHudUpdatesPerSecond: string; telemetryMinorGcPerMinute: string; }`

Resolves per-frame telemetry HUD values and updates rolling windows.

On each published playback frame, the runtime folds the new telemetry sample
into rolling windows and emits human-readable HUD strings.

Parameters:
- `frameStats` - - Playback frame stats for the current frame.
- `telemetryState` - - Runtime telemetry mutable state.

Returns: Formatted telemetry HUD values for this frame.

### RuntimeTelemetryState

Runtime telemetry helpers for live browser HUD updates.

The runtime tracks a small rolling window of operational signals such as HUD
update frequency and minor GC activity. These are not part of the simulation
itself; they are observability features for understanding how expensive the
browser playback loop is.

## browser-entry/runtime/runtime.evolution-loop.service.ts

### runRuntimeEvolutionLoop

`(options: import("test/examples/flappy_bird/browser-entry/runtime/runtime.evolution-loop.service").RuntimeEvolutionLoopOptions) => Promise<void>`

Runs generation orchestration and playback until a stop signal is observed.

The loop alternates between two phases:
1. Evolve off-thread until the worker emits the next best-generation summary.
2. Play that generation back on the main thread while streaming HUD updates.

This rhythm makes the demo feel like a live training dashboard instead of a
one-shot batch job.

Parameters:
- `options` - - Runtime evolution dependencies and mutable state accessors.

Returns: Nothing.

### RuntimeEvolutionLoopOptions

Long-running evolution/playback orchestration for the browser runtime.

This loop is the heart of the interactive demo. It repeatedly asks the worker
for the next evolved generation, updates the HUD and network view, plays back
that generation on the canvas, then folds the outcome into best-so-far
browser state.

## browser-entry/runtime/runtime.browser-globals.service.ts

### installRuntimeBrowserGlobals

`(startRuntime: import("test/examples/flappy_bird/browser-entry/runtime/runtime.browser-globals.service").RuntimeStartFunction) => void`

Publishes browser globals for demo auto-start and host-driven control.

This keeps the runtime friendly to static docs pages where the bundle may be
loaded by script tag rather than imported programmatically.

This keeps parity with the asciiMaze entry style:
- `window.flappyBird.start(...)` for explicit invocation,
- `window.flappyBirdStart(...)` for compatibility,
- one guarded auto-start for standalone HTML usage.

Parameters:
- `startRuntime` - - Runtime entry function.

Returns: Nothing.

### RuntimeStartFunction

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget | undefined) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle>`

Browser-global wiring for the Flappy Bird runtime entrypoint.

The demo supports both module-style startup and traditional global-page usage.
This module publishes the small global surface used by standalone docs pages
and compatibility integrations.

## browser-entry/runtime/runtime.evolution-launch.service.ts

### launchRuntimeEvolution

`(runtimeStartContext: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext, runtimeLifecycleState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeMutableLifecycleState, stop: () => void) => void`

Launch wrapper for the long-running browser runtime loop.

The evolution loop itself is asynchronous and may surface unexpected errors.
This launcher keeps the entrypoint clean by centralizing the catch path that
routes failures into the HUD before shutting the runtime down.
