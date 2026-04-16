# browser-entry/runtime

Top-level browser runtime facade for the Flappy Bird example.

This is the narrowest public browser entry with real behavior behind it.
Calling `start` resolves the DOM host, installs the worker-backed runtime,
initializes the HUD, and launches the evolve-to-playback loop that powers the
demo. The surrounding services keep those responsibilities split, but this
file is the place where they are folded back into one lifecycle.

The boundary matters because the browser demo is more than a canvas render:
it is host setup, worker orchestration, telemetry, HUD updates, and shutdown
control presented as one small API.

## browser-entry/runtime/runtime.ts

### RuntimeRunHandle

Public run handle returned by the browser runtime entrypoint.

The handle is intentionally minimal so callers can treat the demo like a
long-running process with start/stop/status semantics.

### start

```ts
start(
  container: RuntimeContainerTarget,
): Promise<FlappyBirdRunHandle>
```

Starts the Flappy Bird NeatapticTS browser demo and returns lifecycle controls.

This function is intentionally orchestration-focused:
1) resolve runtime dependencies (DOM host, worker, host UI),
2) initialize worker and telemetry plumbing,
3) run the evolve -> playback -> HUD fold loop until stopped,
4) expose a small stop/isRunning/done handle for callers.

Parameters:
- `container` - Element id or HTMLElement to host the demo.

Returns: Run handle for stop/state control.

Example:

```ts
const runHandle = await start('flappy-bird-output');
// later
runHandle.stop();
await runHandle.done;
```

### startRuntimeSession

```ts
startRuntimeSession(
  container: RuntimeContainerTarget,
  runtimeStartOptions: RuntimeStartOptions,
): Promise<FlappyBirdRunHandle>
```

Starts one Flappy browser runtime session with optional internal profile selection state.

The public `start(...)` API always enters through this helper. Internal
browser-UI restarts reuse it so architecture button clicks can launch a fresh
worker-backed run without widening the public API surface.

Parameters:
- `container` - Element id or HTMLElement to host the demo.
- `runtimeStartOptions` - Internal per-session runtime inputs.

Returns: Run handle for the started browser session.

## browser-entry/runtime/runtime.types.ts

Core runtime contracts for the Flappy Bird browser demo.

The runtime boundary is where the browser-side application comes together:
DOM host setup, worker creation, telemetry wiring, lifecycle control, and the
configuration passed into the evolution loop.

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

Public run handle returned by the browser runtime entrypoint.

The handle is intentionally minimal so callers can treat the demo like a
long-running process with start/stop/status semantics.

### RuntimeStartConfig

Static runtime configuration resolved before the browser loop starts.

These values define the high-level NEAT and network shape used for the whole
browser session.

### RuntimeStartContext

Shared runtime startup dependencies created before evolution begins.

Once this context exists, the browser has everything it needs to launch the
actual evolution/playback loop.

### RuntimeStartOptions

Optional runtime session inputs used by internal browser startup flows.

### RuntimeStartupPreviewHandle

Internal legend-preview handle used while a pre-playback canvas card is active.

## browser-entry/runtime/runtime.startup.service.ts

Runtime startup helpers for the Flappy Bird browser demo.

These functions cover the pre-loop phase: resolve the host container, build a
typed browser view, derive static config, create telemetry state, spawn the
worker, and paint the initial HUD before evolution begins.

### createRuntimeStartConfig

```ts
createRuntimeStartConfig(
  runtimeStartOptions: RuntimeStartOptions,
): RuntimeStartConfig
```

Resolves the static runtime configuration used during browser startup.

Centralizing the configuration fold here makes the runtime entry read as
orchestration instead of constant plumbing.

Returns: Runtime configuration derived from shared constants.

### createRuntimeStartContext

```ts
createRuntimeStartContext(
  container: RuntimeContainerTarget,
  runtimeStartOptions: RuntimeStartOptions,
): RuntimeStartContext
```

Creates the shared runtime startup dependencies used by the entry orchestration.

This is the main bootstrap fold for browser startup. After it returns, the
runtime has a host view, a worker channel, telemetry state, and a resolved
configuration object.

Parameters:
- `container` - Element id or HTMLElement to host the demo.

Returns: Shared runtime start context for setup and loop launch.

### initializeRuntimeHud

```ts
initializeRuntimeHud(
  runtimeStartContext: RuntimeStartContext,
): void
```

Paints the initial runtime HUD values before the evolution loop starts.

The HUD is seeded immediately so the page communicates that startup is in
progress rather than appearing blank while the worker and loop are booting.

Parameters:
- `runtimeStartContext` - Shared runtime start context.

Returns: Nothing.

### resolveRuntimePopulationBudget

```ts
resolveRuntimePopulationBudget(
  architectureProfileId: ExampleArchitectureProfileId,
): Pick<RuntimeStartConfig, "populationSize" | "elitismCount">
```

Resolves the browser evolution budget for one architecture profile.

Sparse and NARX keep wider browser budgets than the dense MLP baseline so
the interactive demo still has room to discover pipe-clearing behavior in a
small number of generations. GRU and LSTM now stay materially smaller than
NARX because their gated recurrent blocks still cause visible main-thread
stutter at broader browser flock sizes.

Parameters:
- `architectureProfileId` - Selected shared Flappy profile id.

Returns: Browser-local population and elitism settings.

## browser-entry/runtime/runtime.startup-preview.service.ts

### finalizeRuntimeLegendPreview

```ts
finalizeRuntimeLegendPreview(
  legendPreviewHandle: RuntimeStartupPreviewHandle,
): Promise<void>
```

Waits for a runtime legend preview to finish and then stops its animation loop.

Parameters:
- `legendPreviewHandle` - Active preview handle.

Returns: Nothing.

### paintRuntimeStartupPreviewFrame

```ts
paintRuntimeStartupPreviewFrame(
  context: CanvasRenderingContext2D,
  input: { canvasWidthPx: number; canvasHeightPx: number; legendText: string; opacity: number; frameIndex: number; scrollBasePx: number; legendFontSizePx: number; },
): void
```

Paints one full startup-preview frame.

Parameters:
- `context` - Target canvas 2D context.
- `input` - Render-ready startup-preview frame values.

Returns: Nothing.

### presentRuntimeGenerationPreview

```ts
presentRuntimeGenerationPreview(
  options: RuntimeGenerationPreviewOptions,
): Promise<void>
```

Presents a short generation title card before playback begins.

Every generation-ready event can reuse the same background-plus-neon style as
the startup loading screen, but with a bounded fade-in, hold, and fade-out
lifecycle that automatically completes before playback starts.

Parameters:
- `options` - Canvas, context, stop-state, and generation legend inputs.

Returns: Nothing.

### renderRuntimeStartupPreviewLegend

```ts
renderRuntimeStartupPreviewLegend(
  context: CanvasRenderingContext2D,
  input: { canvasWidthPx: number; canvasHeightPx: number; legendText: string; opacity: number; legendFontSizePx: number; },
): void
```

Draws the centered neon loading legend over the startup preview canvas.

Parameters:
- `context` - Target canvas 2D context.
- `input` - Canvas geometry and resolved opacity for the legend.

Returns: Nothing.

### resolveRuntimeStartupPreviewNowMs

```ts
resolveRuntimeStartupPreviewNowMs(): number
```

Resolves the best available startup-preview clock source.

Returns: Current timestamp in milliseconds.

### startRuntimeEvolvingPreview

```ts
startRuntimeEvolvingPreview(
  options: RuntimeGenerationPreviewOptions,
): RuntimeStartupPreviewHandle
```

Starts the centered evolving-preview shown between completed generations.

This preview keeps the simulation canvas visibly alive while the worker is
evolving the next generation and there are intentionally no birds to render.
Unlike the first-load startup preview, this one is driven by the request
lifecycle of the next generation rather than by initial boot.

Parameters:
- `options` - Canvas, context, stop-state, and evolving legend inputs.

Returns: Handle used to fade the overlay out once the next generation is ready.

### startRuntimeLegendPreview

```ts
startRuntimeLegendPreview(
  options: RuntimeLegendPreviewOptions,
): RuntimeStartupPreviewHandle
```

Starts a legend-driven runtime preview loop on the main playback canvas.

This shared helper powers both the startup loading screen and the bounded
generation title cards. The difference is whether the fade-out is scheduled
up front or started later by calling `complete()`.

Parameters:
- `options` - Canvas, legend text, and preview lifecycle inputs.

Returns: Handle used to complete or stop the preview.

### startRuntimeStartupPreview

```ts
startRuntimeStartupPreview(
  options: RuntimeStartupPreviewOptions,
): RuntimeStartupPreviewHandle
```

Starts the animated browser startup preview shown before the first birds appear.

This preview owns only the initial loading window. It reuses the existing
background renderer so the canvas stays alive during worker initialization,
then fades a centered loading legend out before the first playback session is
allowed to begin.

Parameters:
- `options` - Canvas, context, and stop-state inputs for the startup preview.

Returns: Handle used to begin the exit transition or stop the preview outright.

## browser-entry/runtime/runtime.startup-preview.utils.ts

### resolveRuntimeStartupPreviewLegendFontSizePx

```ts
resolveRuntimeStartupPreviewLegendFontSizePx(
  canvasWidthPx: number,
  canvasHeightPx: number,
): number
```

Resolves a responsive legend font size for the startup preview.

Parameters:
- `canvasWidthPx` - Current preview canvas width.
- `canvasHeightPx` - Current preview canvas height.

Returns: Responsive legend font size in pixels.

### resolveRuntimeStartupPreviewVisualState

```ts
resolveRuntimeStartupPreviewVisualState(
  input: { nowMs: number; previewStartTimeMs: number; previewExitStartTimeMs?: number | undefined; canvasWidthPx: number; canvasHeightPx: number; pipeScrollSpeedPxPerFrame: number; },
): RuntimeStartupPreviewVisualState
```

Resolves the visual state needed to draw one startup-preview frame.

The preview needs only a few derived values: the current opacity for the
fade-in/fade-out lifecycle, a deterministic frame index and scroll offset for
the animated background, and a responsive font size that stays readable on
both compact and wide canvases.

Parameters:
- `input` - Timing, canvas size, and scroll-speed inputs for the preview frame.

Returns: Resolved visual state for the current startup-preview frame.

### RuntimeStartupPreviewVisualState

Resolved visual state for one startup-preview render frame.

## browser-entry/runtime/runtime.lifecycle.service.ts

Lifecycle and teardown helpers for the browser runtime.

The runtime behaves like a small application process. These helpers create the
mutable state and public handle needed to stop it cleanly, terminate the
worker, and resolve the completion promise exactly once.

### createRuntimeLifecycleState

```ts
createRuntimeLifecycleState(): RuntimeMutableLifecycleState
```

Creates mutable lifecycle state for stop semantics and completion signaling.

The returned object is shared across runtime orchestration paths so both
expected shutdown and failure-driven shutdown follow the same completion flow.

Returns: Mutable lifecycle state used by the run handle.

### createRuntimeRunHandle

```ts
createRuntimeRunHandle(
  runtimeStartContext: RuntimeStartContext,
  runtimeLifecycleState: RuntimeMutableLifecycleState,
): FlappyBirdRunHandle
```

Builds the public run handle and binds it to runtime teardown behavior.

The handle is the user-facing control surface for the demo. Internally it is
just a thin closure layer over the mutable lifecycle state and startup
context.

Parameters:
- `runtimeStartContext` - Shared runtime start context.
- `runtimeLifecycleState` - Mutable lifecycle state for stop semantics.

Returns: Public run handle exposed to callers.

## browser-entry/runtime/runtime.evolution-launch.service.ts

Launch wrapper for the long-running browser runtime loop.

The evolution loop itself is asynchronous and may surface unexpected errors.
This launcher keeps the entrypoint clean by centralizing the catch path that
routes failures into the HUD before shutting the runtime down.

### launchRuntimeEvolution

```ts
launchRuntimeEvolution(
  runtimeStartContext: RuntimeStartContext,
  runtimeLifecycleState: RuntimeMutableLifecycleState,
  stop: () => void,
): void
```

Starts the runtime evolution loop and routes unexpected failures to the HUD.

This is the boundary between normal browser startup and the long-running async
loop that drives evolution plus playback.

Parameters:
- `runtimeStartContext` - Shared runtime start context.
- `runtimeLifecycleState` - Mutable lifecycle state used for stop checks.
- `stop` - Idempotent stop function bound to the current runtime handle.

Returns: Nothing.

## browser-entry/runtime/runtime.evolution-loop.service.ts

Long-running evolution/playback orchestration for the browser runtime.

This loop is the heart of the interactive demo. It repeatedly asks the worker
for the next evolved generation, updates the HUD and network view, plays back
that generation on the canvas, then folds the outcome into the generation
summary section and the cross-generation history used by the architecture
selector.

### finalizeStartupPreview

```ts
finalizeStartupPreview(
  startupPreviewHandle: RuntimeStartupPreviewHandle | undefined,
): Promise<void>
```

Completes and tears down the startup preview when one is active.

Parameters:
- `startupPreviewHandle` - Optional preview handle created for first-load boot.

Returns: Nothing.

### requestGenerationWithOptionalStartupPreview

```ts
requestGenerationWithOptionalStartupPreview(
  options: { evolutionWorker: Worker; canvas: HTMLCanvasElement; context: CanvasRenderingContext2D; isStopped: () => boolean; showStartupPreview: boolean; waitLegendText: string; },
): Promise<EvolutionGenerationPayload>
```

Requests the next generation and optionally shows the first-load startup preview.

The browser should only show the animated loading preview while the very
first generation is still booting. Later generations can reuse the same
canvas style through the separate generation-presentation path.

Parameters:
- `options` - Generation request inputs and preview-gating state.

Returns: The next generation payload from the worker.

### resolveEvolutionWaitLegendText

```ts
resolveEvolutionWaitLegendText(
  generation: number,
): string
```

Resolves the centered legend text shown while the worker evolves the next generation.

Parameters:
- `generation` - Next generation number expected from the worker.

Returns: Evolving overlay text.

### resolveGenerationPopulationNetworks

```ts
resolveGenerationPopulationNetworks(
  generationPayload: { populationNetworksJson?: Record<string, unknown>[] | undefined; },
  bestNetwork: default | undefined,
): default[]
```

Resolves the browser-side network cache for the current playback generation.

Playback birds are created from the generation population in stable array
order, so the browser can reuse this ordered cache to redraw the network
panel when the red-bird champion changes.

Parameters:
- `generationPayload` - Worker generation-ready payload.
- `bestNetwork` - Current generation best-network fallback.

Returns: Ordered population networks for the upcoming playback session.

### resolveGenerationPopulationSize

```ts
resolveGenerationPopulationSize(
  generationPopulationNetworks: default[],
  fallbackPopulationSize: number,
): number
```

Resolves which population size the browser HUD should display for the current generation.

Browser sessions may start with one population budget and later downshift to a
smaller one after a successful run. The worker generation payload already
carries the actual serialized population, so the HUD should prefer that real
size over the startup budget whenever it is available.

Parameters:
- `generationPopulationNetworks` - Browser-side cache of the current generation population.
- `fallbackPopulationSize` - Startup budget used before a generation payload is available.

Returns: Population size that should be displayed in the HUD for this generation.

### resolveGenerationPresentationLegendText

```ts
resolveGenerationPresentationLegendText(
  generation: number,
): string
```

Resolves the centered legend text used to present one ready generation.

Parameters:
- `generation` - Ready generation number from the worker payload.

Returns: Generation presentation legend text.

### resolveGenerationSummaryHudValues

```ts
resolveGenerationSummaryHudValues(
  input: { architectureLabel: string; bestFitness: number; playbackSummary?: PlaybackEpisodeSummary | undefined; },
): { summaryHeader: string; summaryFitness: string; summaryWinnerFrames: string; summaryWinnerPipes: string; summaryAveragePipes: string; summaryP90Frames: string; summaryArchitecture: string; }
```

Resolves the generation-summary HUD values shown beside the live run counters.

The summary intentionally shows metrics the worker already computes for the
whole population so the browser can present higher-signal data without doing
extra aggregation on the main thread.

Parameters:
- `input` - Summary source values for the active generation.

Returns: HUD-ready summary values with placeholders until playback completes.

### runRuntimeEvolutionLoop

```ts
runRuntimeEvolutionLoop(
  options: RuntimeEvolutionLoopOptions,
): Promise<void>
```

Runs generation orchestration and playback until a stop signal is observed.

The loop alternates between two phases:
1. Evolve off-thread until the worker emits the next best-generation summary.
2. Play that generation back on the main thread while streaming HUD updates.

This rhythm makes the demo feel like a live training dashboard instead of a
one-shot batch job.

Parameters:
- `options` - Runtime evolution dependencies and mutable state accessors.

Returns: Nothing.

### RuntimeEvolutionLoopOptions

Dependencies required to run the browser runtime evolution loop.

Grouping these dependencies into one object keeps the public loop entry more
declarative and avoids a long positional parameter list.

## browser-entry/runtime/runtime.telemetry.service.ts

Runtime telemetry helpers for live browser HUD updates.

The runtime tracks a small rolling window of operational signals such as HUD
update frequency and minor GC activity. These are not part of the simulation
itself; they are observability features for understanding how expensive the
browser playback loop is.

### createRuntimeTelemetryState

```ts
createRuntimeTelemetryState(): RuntimeTelemetryState
```

Creates telemetry state and attaches optional minor-GC observer.

Instrumentation is feature-gated so the demo can run in a low-noise mode when
telemetry is not desired.

Returns: Initialized telemetry state.

### disconnectRuntimeTelemetry

```ts
disconnectRuntimeTelemetry(
  telemetryState: RuntimeTelemetryState,
): void
```

Disconnects runtime telemetry observers.

This is part of runtime teardown and prevents instrumentation observers from
lingering after the demo has stopped.

Parameters:
- `telemetryState` - Runtime telemetry state.

Returns: Nothing.

### resolveInitialRuntimeTelemetryHudValues

```ts
resolveInitialRuntimeTelemetryHudValues(): { telemetryHeader: string; telemetryActivationsPerFrame: string; telemetrySimulationStepsPerRaf: string; telemetryHudUpdatesPerSecond: string; telemetryMinorGcPerMinute: string; }
```

Resolves default telemetry HUD values used before first playback updates.

The initial values make the instrumentation section self-describing even
before the first playback frame arrives.

Returns: Initial telemetry field values.

### resolveRuntimeTelemetryHudValues

```ts
resolveRuntimeTelemetryHudValues(
  frameStats: PlaybackFrameStats,
  telemetryState: RuntimeTelemetryState,
): { telemetryActivationsPerFrame: string; telemetrySimulationStepsPerRaf: string; telemetryHudUpdatesPerSecond: string; telemetryMinorGcPerMinute: string; }
```

Resolves per-frame telemetry HUD values and updates rolling windows.

On each published playback frame, the runtime folds the new telemetry sample
into rolling windows and emits human-readable HUD strings.

Parameters:
- `frameStats` - Playback frame stats for the current frame.
- `telemetryState` - Runtime telemetry mutable state.

Returns: Formatted telemetry HUD values for this frame.

### RuntimeTelemetryState

Runtime telemetry mutable state used for rolling HUD metrics.

The state keeps timestamp windows rather than pre-aggregated counters so the
HUD can report smoothed recent rates instead of lifetime totals.

## browser-entry/runtime/runtime.browser-globals.service.ts

Browser-global wiring for the Flappy Bird runtime entrypoint.

The demo supports both module-style startup and traditional global-page usage.
This module publishes the small global surface used by standalone docs pages
and compatibility integrations.

### installRuntimeBrowserGlobals

```ts
installRuntimeBrowserGlobals(
  startRuntime: RuntimeStartFunction,
): void
```

Publishes browser globals for demo auto-start and host-driven control.

This keeps the runtime friendly to static docs pages where the bundle may be
loaded by script tag rather than imported programmatically.

This keeps parity with the asciiMaze entry style:
- `window.flappyBird.start(...)` for explicit invocation,
- `window.flappyBirdStart(...)` for compatibility,
- one guarded auto-start for standalone HTML usage.

Parameters:
- `startRuntime` - Runtime entry function.

Returns: Nothing.

### RuntimeStartFunction

```ts
RuntimeStartFunction(
  container: RuntimeContainerTarget | undefined,
): Promise<FlappyBirdRunHandle>
```

Runtime start function signature used by browser-global wiring.

The signature accepts an optional host target so global callers can either
rely on the default container or explicitly direct the runtime to another
element.

## browser-entry/runtime/runtime.errors.ts

Runtime-specific error helpers for the browser entrypoint.

These errors normalize two user-facing failure modes: the browser cannot find
the requested host container, or the runtime needs to report an unexpected
failure back into the HUD.

### resolveRequiredRuntimeHostElement

```ts
resolveRequiredRuntimeHostElement(
  container: RuntimeContainerTarget,
): HTMLElement
```

Resolves and validates the browser runtime host element.

The runtime accepts either a string id or a concrete element so this helper
folds that loose input into one validated host node.

Parameters:
- `container` - Element id or HTMLElement provided to runtime start.

Returns: Resolved host element.

### resolveRuntimeHudErrorStatus

```ts
resolveRuntimeHudErrorStatus(
  error: unknown,
): string
```

Formats unknown runtime failures into a stable HUD status string.

The HUD should not need to understand arbitrary thrown values, so this helper
normalizes anything throwable into one readable status line.

Parameters:
- `error` - Unknown runtime exception value.

Returns: Normalized status string for HUD output.

### RuntimeContainerNotFoundError

Error raised when the browser runtime host container cannot be resolved.

This usually means the caller passed the wrong element id or attempted to
start the demo before the target container existed in the DOM.

## browser-entry/runtime/runtime.architecture-profile.service.ts

### isRuntimeArchitectureScoreBetter

```ts
isRuntimeArchitectureScoreBetter(
  candidateBestScore: RuntimeArchitectureBestScore,
  currentBestScore: RuntimeArchitectureBestScore | undefined,
): boolean
```

Resolves whether a candidate browser score should replace the current record.

Parameters:
- `candidateBestScore` - Candidate score being considered.
- `currentBestScore` - Current stored record for the profile.

Returns: True when the candidate is strictly better.

### persistRuntimeArchitectureHistory

```ts
persistRuntimeArchitectureHistory(
  historyByProfileId: Partial<Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>>,
  storage: RuntimeArchitectureHistoryStorage | undefined,
): void
```

Persists the current browser-local architecture record table when storage exists.

Parameters:
- `historyByProfileId` - Local history table to persist.
- `storage` - Optional storage override for tests.

Returns: Nothing.

### resolveAvailableRuntimeArchitectureProfiles

```ts
resolveAvailableRuntimeArchitectureProfiles(): ExampleArchitectureProfile[]
```

Resolves the currently approved shared Flappy architecture profiles.

Returns: Approved shared profiles in the curated Flappy selector order.

### resolveRuntimeArchitectureHistory

```ts
resolveRuntimeArchitectureHistory(
  storage: RuntimeArchitectureHistoryStorage | undefined,
): Partial<Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>>
```

Reads persisted local browser architecture records when storage is available.

Parameters:
- `storage` - Optional storage override for tests.

Returns: Previously stored local records or an empty table.

### resolveRuntimeArchitectureHistoryLeaderProfileId

```ts
resolveRuntimeArchitectureHistoryLeaderProfileId(
  historyByProfileId: Partial<Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>>,
): ExampleArchitectureProfileId | undefined
```

Resolves the highest-scoring architecture profile from the local history table.

Parameters:
- `historyByProfileId` - Browser-local history table.

Returns: Leading profile id when at least one stored record exists.

### resolveRuntimeArchitectureHistoryStorage

```ts
resolveRuntimeArchitectureHistoryStorage(): RuntimeArchitectureHistoryStorage | undefined
```

Resolves browser storage for Flappy local architecture history when available.

Returns: Browser storage implementation or `undefined` outside the browser.

### resolveRuntimeArchitectureSelectorItems

```ts
resolveRuntimeArchitectureSelectorItems(
  options: { availableProfiles: ExampleArchitectureProfile[]; selectedProfileId: ExampleArchitectureProfileId; historyByProfileId: Partial<Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>>; },
): HostArchitectureSelectorItem[]
```

Resolves render-ready selector items from approved profiles and local history.

Parameters:
- `options` - Approved profile set, selected profile id, and local history table.

Returns: Render-ready selector items for the Flappy host UI.

### resolveRuntimeArchitectureTooltipBodyLines

```ts
resolveRuntimeArchitectureTooltipBodyLines(
  profile: ExampleArchitectureProfile,
): string[]
```

Resolves the educational tooltip copy shown for one Flappy architecture profile.

Parameters:
- `profile` - Shared Flappy architecture profile.

Returns: Short, punchy tooltip lines for the selector button.

### resolveRuntimeArchitectureTooltipHeading

```ts
resolveRuntimeArchitectureTooltipHeading(
  profile: ExampleArchitectureProfile,
): string
```

Resolves the punchy tooltip heading used by the Flappy architecture selector.

Parameters:
- `profile` - Shared Flappy architecture profile.

Returns: Tooltip heading shown above the selector button.

### resolveSelectedRuntimeArchitectureProfile

```ts
resolveSelectedRuntimeArchitectureProfile(
  profileId: ExampleArchitectureProfileId | undefined,
): ExampleArchitectureProfile
```

Resolves the selected shared Flappy profile for the next browser session.

Parameters:
- `profileId` - Optional requested profile id.

Returns: Resolved Flappy-ready shared profile.

### RuntimeArchitectureBestScore

Best-known local browser record for one Flappy architecture profile.

### RuntimeArchitectureHistoryByProfileId

Local-browser record table keyed by the shared Flappy architecture profile id.

### updateRuntimeArchitectureHistory

```ts
updateRuntimeArchitectureHistory(
  historyByProfileId: Partial<Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>>,
  profileId: ExampleArchitectureProfileId,
  candidateBestScore: RuntimeArchitectureBestScore,
): Partial<Record<ExampleArchitectureProfileId, RuntimeArchitectureBestScore>>
```

Folds one session-best Flappy record into the persisted architecture history table.

Records compare by pipes passed first and frames survived as a stable
tiebreaker so the selector caption reflects the most meaningful browser score.

Parameters:
- `historyByProfileId` - Existing browser-local history table.
- `profileId` - Shared architecture profile receiving the score update.
- `candidateBestScore` - Session-best browser score for the profile.

Returns: Updated history table when the candidate improves the stored record.
