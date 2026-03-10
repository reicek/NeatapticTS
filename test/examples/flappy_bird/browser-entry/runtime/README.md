# browser-entry/runtime

## browser-entry/runtime/runtime.types.ts

### RuntimeContainerTarget

Container argument accepted by the browser runtime start function.

### RuntimeGlobalWindow

Browser `window` extension shape used for global runtime wiring.

### RuntimeHostViewContext

Browser host view handles used by the runtime entry orchestration.

### RuntimeMutableLifecycleState

Mutable lifecycle state used to coordinate stop semantics and completion.

### RuntimeRunHandle

Public run handle returned by the browser runtime entrypoint.

### RuntimeStartConfig

Static runtime configuration resolved before the browser loop starts.

### RuntimeStartContext

Shared runtime startup dependencies created before evolution begins.

## browser-entry/runtime/runtime.ts

### RuntimeRunHandle

Public run handle returned by the browser runtime entrypoint.

### start

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle>`

## browser-entry/runtime/runtime.errors.ts

### resolveRequiredRuntimeHostElement

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => HTMLElement`

Resolves and validates the browser runtime host element.

Parameters:
- `container` - - Element id or HTMLElement provided to runtime start.

Returns: Resolved host element.

### resolveRuntimeHudErrorStatus

`(error: unknown) => string`

Formats unknown runtime failures into a stable HUD status string.

Parameters:
- `error` - - Unknown runtime exception value.

Returns: Normalized status string for HUD output.

### RuntimeContainerNotFoundError

Error raised when the browser runtime host container cannot be resolved.

## browser-entry/runtime/runtime.startup.service.ts

### createRuntimeStartConfig

`() => import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartConfig`

Resolves the static runtime configuration used during browser startup.

Returns: Runtime configuration derived from shared constants.

### createRuntimeStartContext

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext`

Creates the shared runtime startup dependencies used by the entry orchestration.

Parameters:
- `container` - - Element id or HTMLElement to host the demo.

Returns: Shared runtime start context for setup and loop launch.

### initializeRuntimeHud

`(runtimeStartContext: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext) => void`

Paints the initial runtime HUD values before the evolution loop starts.

Parameters:
- `runtimeStartContext` - - Shared runtime start context.

Returns: Nothing.

## browser-entry/runtime/runtime.lifecycle.service.ts

### createRuntimeLifecycleState

`() => import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeMutableLifecycleState`

Creates mutable lifecycle state for stop semantics and completion signaling.

Returns: Mutable lifecycle state used by the run handle.

### createRuntimeRunHandle

`(runtimeStartContext: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext, runtimeLifecycleState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeMutableLifecycleState) => import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle`

Builds the public run handle and binds it to runtime teardown behavior.

Parameters:
- `runtimeStartContext` - - Shared runtime start context.
- `runtimeLifecycleState` - - Mutable lifecycle state for stop semantics.

Returns: Public run handle exposed to callers.

## browser-entry/runtime/runtime.telemetry.service.ts

### createRuntimeTelemetryState

`() => import("test/examples/flappy_bird/browser-entry/runtime/runtime.telemetry.service").RuntimeTelemetryState`

Creates telemetry state and attaches optional minor-GC observer.

Returns: Initialized telemetry state.

### disconnectRuntimeTelemetry

`(telemetryState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.telemetry.service").RuntimeTelemetryState) => void`

Disconnects runtime telemetry observers.

Parameters:
- `telemetryState` - - Runtime telemetry state.

Returns: Nothing.

### resolveInitialRuntimeTelemetryHudValues

`() => { telemetryHeader: string; telemetryActivationsPerFrame: string; telemetrySimulationStepsPerRaf: string; telemetryHudUpdatesPerSecond: string; telemetryMinorGcPerMinute: string; }`

Resolves default telemetry HUD values used before first playback updates.

Returns: Initial telemetry field values.

### resolveRuntimeTelemetryHudValues

`(frameStats: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats, telemetryState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.telemetry.service").RuntimeTelemetryState) => { telemetryActivationsPerFrame: string; telemetrySimulationStepsPerRaf: string; telemetryHudUpdatesPerSecond: string; telemetryMinorGcPerMinute: string; }`

Resolves per-frame telemetry HUD values and updates rolling windows.

Parameters:
- `frameStats` - - Playback frame stats for the current frame.
- `telemetryState` - - Runtime telemetry mutable state.

Returns: Formatted telemetry HUD values for this frame.

### RuntimeTelemetryState

Runtime telemetry mutable state used for rolling HUD metrics.

## browser-entry/runtime/runtime.evolution-loop.service.ts

### runRuntimeEvolutionLoop

`(options: import("test/examples/flappy_bird/browser-entry/runtime/runtime.evolution-loop.service").RuntimeEvolutionLoopOptions) => Promise<void>`

Runs generation orchestration and playback until a stop signal is observed.

Parameters:
- `options` - - Runtime evolution dependencies and mutable state accessors.

Returns: Nothing.

### RuntimeEvolutionLoopOptions

Dependencies required to run the browser runtime evolution loop.

## browser-entry/runtime/runtime.browser-globals.service.ts

### installRuntimeBrowserGlobals

`(startRuntime: import("test/examples/flappy_bird/browser-entry/runtime/runtime.browser-globals.service").RuntimeStartFunction) => void`

Publishes browser globals for demo auto-start and host-driven control.

This keeps parity with the asciiMaze entry style:
- `window.flappyBird.start(...)` for explicit invocation,
- `window.flappyBirdStart(...)` for compatibility,
- one guarded auto-start for standalone HTML usage.

Parameters:
- `startRuntime` - - Runtime entry function.

Returns: Nothing.

### RuntimeStartFunction

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget | undefined) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle>`

Runtime start function signature used by browser-global wiring.

## browser-entry/runtime/runtime.evolution-launch.service.ts

### launchRuntimeEvolution

`(runtimeStartContext: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeStartContext, runtimeLifecycleState: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeMutableLifecycleState, stop: () => void) => void`

Starts the runtime evolution loop and routes unexpected failures to the HUD.

Parameters:
- `runtimeStartContext` - - Shared runtime start context.
- `runtimeLifecycleState` - - Mutable lifecycle state used for stop checks.
- `stop` - - Idempotent stop function bound to the current runtime handle.

Returns: Nothing.
