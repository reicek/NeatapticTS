# browser-entry

## browser-entry/browser-entry.types.ts

### browser-entry.types

Aggregated public type surface for the Flappy Bird browser runtime.

The browser demo spans several concerns at once: worker messaging, playback
rendering, telemetry, viewport math, and network visualization. Re-exporting
the public contracts from one place gives readers a compact map of that
runtime without forcing them to know the internal folder layout first.

### BrowserDifficultyProfile

Difficulty profile consumed by simulation observation helpers.

This bundles the three variables that define how demanding a stretch of the
course is: corridor width, pipe speed, and spawn cadence.

### BrowserPopulationBirdLike

Simulation-facing browser contracts shared by playback helpers.

These types describe the minimum world state the browser needs while it is
reconstructing, rendering, or summarizing worker-produced frames.

### BrowserPopulationPipeLike

Pipe shape used by utility observation-vector helpers.

### ColorLegendRow

Legend row model for network visualization color legends.

Each row labels a numeric interval and the color used to render it.

### ColorTier

Connection or bias tier used for color mapping ramps.

Visualization buckets continuous weights into legible color bands so humans
can scan sign and magnitude at a glance.

### CreateFlappyStatsTableRowsInput

Input contract for declarative runtime stats table row builder.

The builder needs both the target table and a small policy surface that says
whether instrumentation rows should appear and how rows should be colored.

### EvolutionGenerationPayload

Worker payload describing evolved generation summary values.

This is the browser-facing summary of one completed NEAT generation: what
generation finished, how fit the best genome was, and optionally the best
network for visualization or playback.

### EvolutionGenerationReadyMessage

Worker message emitted when a generation has completed evolving.

### EvolutionPlaybackStepMessage

Worker message carrying one playback step and aggregate markers.

Besides the frame snapshot itself, this message also carries summary values
used by the HUD so the browser can show performance and progress without
recomputing population-wide statistics on the main thread.

### EvolutionPlaybackStepSnapshot

Per-frame snapshot received from the worker playback channel.

A snapshot combines geometry, packed population state, and lightweight world
metadata so the browser can render a deterministic frame without rerunning
the simulation locally.

### EvolutionWorkerErrorMessage

Worker message emitted for simulation/playback errors.

### EvolutionWorkerMessage

Union of all supported worker messages consumed by browser entry.

A closed union keeps the main-thread message handler explicit and easy to
audit when the protocol evolves.

### FlappyBirdRunHandle

Public lifecycle contracts for the Flappy Bird browser runtime.

These types describe how the demo is started, stopped, and exposed on the
browser `window` object. They are intentionally small because callers should
control the demo at a high level without depending on private implementation
details.

### FlappyStatsCategoryColors

Color pair used for stats category key/value styling.

The HUD uses paired colors so labels and values stay visually grouped while
still separating categories such as current run, telemetry, and best-so-far.

### FlappyStatsKey

HUD and runtime stats contracts for the Flappy Bird browser demo.

The browser HUD is intentionally declarative: keys describe what should be
shown, and helper utilities map those keys to DOM rows and live values. That
keeps the status panel readable even as telemetry grows.

### FlappyStatsRowDescriptor

Declarative row descriptor for the runtime stats table.

Each row is described as data first so the HUD can be assembled in a stable,
testable order instead of being hand-written imperatively.

### FlappyStatsTableCells

Runtime lookup map of stat keys to writable value cells.

This acts like a small DOM index so the update loop can mutate the correct
cells directly without repeatedly querying the document.

### NetworkLegendLayout

Precomputed legend panel layout used by visualization renderer.

Layout is resolved up front so the draw path can stay focused on painting,
not recomputing geometry every frame.

### NetworkNodeDimensionsLike

Pixel dimensions used for network-node rectangle rendering.

Keeping node box dimensions explicit makes legend and topology layout easier
to tune without hidden drawing constants.

### NetworkVisualizationHandle

Network-visualization contracts for the Flappy Bird browser demo.

One of the educational goals of the example is to let people watch evolved
controllers as structures, not just as scores. These types describe the
lightweight shapes used by the architecture panel so rendering logic can stay
decoupled from the full internal network implementation.

### PackedPlaybackBirdSnapshot

Packed typed-array payload for playback bird snapshot transport.

This mirrors the pipe packing strategy so playback can move large population
snapshots with less allocation pressure than object-per-bird messages.

### PackedPlaybackPipeSnapshot

Packed typed-array payload for playback pipe snapshot transport.

Typed arrays keep frame payloads compact and predictable, which matters when
the worker is streaming many birds and pipes across animation frames.

### PlaybackFrameStats

Lightweight per-frame telemetry emitted to HUD update callback.

These values are the browser-friendly metrics shown in the live status panel:
how many birds remain, how far the leader has progressed, and how expensive
the current playback cadence is.

### PopulationBird

Renderable bird state snapshot emitted by the playback worker.

The browser does not receive full neural state here. It only gets the fields
needed for presentation and HUD summaries, which keeps per-frame transport
light.

### PopulationPipe

Renderable pipe state snapshot emitted by the playback worker.

This is the smallest pipe shape the browser renderer needs for one frame:
horizontal position plus the vertical corridor geometry.

### PopulationRenderState

Mutable render-state model consumed by the population frame renderer.

The playback layer incrementally updates this state as worker snapshots
arrive, which lets rendering stay deterministic without re-deriving world
history from scratch each frame.

### PositionedNetworkNodeLike

Positioned node instance used by network visualization drawing.

Layout and rendering are split: first a node is assigned screen coordinates,
then the renderer paints it.

### RenderClosedOuterBoxInput

Input contract for outer frame rendering helper.

The outer frame is the decorative shell that visually separates the playable
world and telemetry panels from the rest of the page.

### RenderStandaloneTitleBoxInput

Canvas frame-layout contracts for the Flappy Bird browser demo.

These types support the demo's deliberately stylized text-frame chrome: title
boxes, outer borders, and viewport transforms that make the example feel more
like an instrument panel than a plain canvas game.

### RngLike

Minimal random source contract used by utility random helpers.

The narrow contract keeps deterministic spawn utilities portable across
browser and test contexts.

### RuntimeWindow

Runtime window contract for the Flappy Bird browser demo.

The demo exposes a small debug-friendly surface on `window` so manual browser
experiments and docs examples can start the simulation without importing the
bundle as a module.

### SerializedNetwork

Worker transport contracts for the Flappy Bird browser runtime.

The browser UI and the evolution worker communicate through a deliberately
explicit message protocol. The goal is educational as well as practical: it
makes it obvious which values are computed off-thread, which snapshots are
transferred frame-by-frame, and which events advance the demo state.

If you want background reading, the Wikipedia article on "message passing"
provides a useful conceptual frame for this boundary.

### TextFrameMetrics

Canvas text-grid metrics used for frame rendering layout helpers.

The frame renderer measures glyph and row geometry once, then uses that grid
to place ASCII-style UI elements consistently.

### TrailPoint

Trail point used by playback trail rendering cache.

A trail point stores where one bird was at one frame so the UI can draw a
short motion history behind active agents.

### TrailState

Mutable trail cache keyed by bird index for frame rendering.

This cache exists purely for visualization ergonomics; it is not part of the
worker simulation state.

### ViewportInfo

Viewport transform values for world-to-canvas rendering.

These numbers answer the classic graphics question: how does one unit in the
simulated world map into the current canvas rectangle?

### VisualNetworkConnectionLike

Lightweight connection shape used by network visualization drawing.

The renderer only needs connectivity, weight, and enabled state, not the full
training-time behavior of a connection object.

### VisualNetworkNodeLike

Lightweight node shape used by network visualization drawing.

This shape keeps the renderer independent from the concrete Network class
while still exposing the semantic fields that matter visually.

## browser-entry/browser-entry.stats.types.ts

### browser-entry.stats.types

HUD and runtime stats contracts for the Flappy Bird browser demo.

The browser HUD is intentionally declarative: keys describe what should be
shown, and helper utilities map those keys to DOM rows and live values. That
keeps the status panel readable even as telemetry grows.

### CreateFlappyStatsTableRowsInput

Input contract for declarative runtime stats table row builder.

The builder needs both the target table and a small policy surface that says
whether instrumentation rows should appear and how rows should be colored.

### FlappyStatsCategoryColors

Color pair used for stats category key/value styling.

The HUD uses paired colors so labels and values stay visually grouped while
still separating categories such as current run, telemetry, and best-so-far.

### FlappyStatsKey

HUD and runtime stats contracts for the Flappy Bird browser demo.

The browser HUD is intentionally declarative: keys describe what should be
shown, and helper utilities map those keys to DOM rows and live values. That
keeps the status panel readable even as telemetry grows.

### FlappyStatsRowDescriptor

Declarative row descriptor for the runtime stats table.

Each row is described as data first so the HUD can be assembled in a stable,
testable order instead of being hand-written imperatively.

### FlappyStatsTableCells

Runtime lookup map of stat keys to writable value cells.

This acts like a small DOM index so the update loop can mutate the correct
cells directly without repeatedly querying the document.

## browser-entry/browser-entry.render.types.ts

### browser-entry.render.types

Canvas frame-layout contracts for the Flappy Bird browser demo.

These types support the demo's deliberately stylized text-frame chrome: title
boxes, outer borders, and viewport transforms that make the example feel more
like an instrument panel than a plain canvas game.

### RenderClosedOuterBoxInput

Input contract for outer frame rendering helper.

The outer frame is the decorative shell that visually separates the playable
world and telemetry panels from the rest of the page.

### RenderStandaloneTitleBoxInput

Canvas frame-layout contracts for the Flappy Bird browser demo.

These types support the demo's deliberately stylized text-frame chrome: title
boxes, outer borders, and viewport transforms that make the example feel more
like an instrument panel than a plain canvas game.

### TextFrameMetrics

Canvas text-grid metrics used for frame rendering layout helpers.

The frame renderer measures glyph and row geometry once, then uses that grid
to place ASCII-style UI elements consistently.

### ViewportInfo

Viewport transform values for world-to-canvas rendering.

These numbers answer the classic graphics question: how does one unit in the
simulated world map into the current canvas rectangle?

## browser-entry/browser-entry.worker.types.ts

### browser-entry.worker.types

Worker transport contracts for the Flappy Bird browser runtime.

The browser UI and the evolution worker communicate through a deliberately
explicit message protocol. The goal is educational as well as practical: it
makes it obvious which values are computed off-thread, which snapshots are
transferred frame-by-frame, and which events advance the demo state.

If you want background reading, the Wikipedia article on "message passing"
provides a useful conceptual frame for this boundary.

### EvolutionGenerationPayload

Worker payload describing evolved generation summary values.

This is the browser-facing summary of one completed NEAT generation: what
generation finished, how fit the best genome was, and optionally the best
network for visualization or playback.

### EvolutionGenerationReadyMessage

Worker message emitted when a generation has completed evolving.

### EvolutionPlaybackStepMessage

Worker message carrying one playback step and aggregate markers.

Besides the frame snapshot itself, this message also carries summary values
used by the HUD so the browser can show performance and progress without
recomputing population-wide statistics on the main thread.

### EvolutionPlaybackStepSnapshot

Per-frame snapshot received from the worker playback channel.

A snapshot combines geometry, packed population state, and lightweight world
metadata so the browser can render a deterministic frame without rerunning
the simulation locally.

### EvolutionWorkerErrorMessage

Worker message emitted for simulation/playback errors.

### EvolutionWorkerMessage

Union of all supported worker messages consumed by browser entry.

A closed union keeps the main-thread message handler explicit and easy to
audit when the protocol evolves.

### PackedPlaybackBirdSnapshot

Packed typed-array payload for playback bird snapshot transport.

This mirrors the pipe packing strategy so playback can move large population
snapshots with less allocation pressure than object-per-bird messages.

### PackedPlaybackPipeSnapshot

Packed typed-array payload for playback pipe snapshot transport.

Typed arrays keep frame payloads compact and predictable, which matters when
the worker is streaming many birds and pipes across animation frames.

### PlaybackFrameStats

Lightweight per-frame telemetry emitted to HUD update callback.

These values are the browser-friendly metrics shown in the live status panel:
how many birds remain, how far the leader has progressed, and how expensive
the current playback cadence is.

### PopulationBird

Renderable bird state snapshot emitted by the playback worker.

The browser does not receive full neural state here. It only gets the fields
needed for presentation and HUD summaries, which keeps per-frame transport
light.

### PopulationPipe

Renderable pipe state snapshot emitted by the playback worker.

This is the smallest pipe shape the browser renderer needs for one frame:
horizontal position plus the vertical corridor geometry.

### SerializedNetwork

Worker transport contracts for the Flappy Bird browser runtime.

The browser UI and the evolution worker communicate through a deliberately
explicit message protocol. The goal is educational as well as practical: it
makes it obvious which values are computed off-thread, which snapshots are
transferred frame-by-frame, and which events advance the demo state.

If you want background reading, the Wikipedia article on "message passing"
provides a useful conceptual frame for this boundary.

## browser-entry/browser-entry.runtime.types.ts

### browser-entry.runtime.types

Public lifecycle contracts for the Flappy Bird browser runtime.

These types describe how the demo is started, stopped, and exposed on the
browser `window` object. They are intentionally small because callers should
control the demo at a high level without depending on private implementation
details.

### FlappyBirdRunHandle

Public lifecycle contracts for the Flappy Bird browser runtime.

These types describe how the demo is started, stopped, and exposed on the
browser `window` object. They are intentionally small because callers should
control the demo at a high level without depending on private implementation
details.

### RuntimeWindow

Runtime window contract for the Flappy Bird browser demo.

The demo exposes a small debug-friendly surface on `window` so manual browser
experiments and docs examples can start the simulation without importing the
bundle as a module.

## browser-entry/browser-entry.simulation.types.ts

### BrowserDifficultyProfile

Difficulty profile consumed by simulation observation helpers.

This bundles the three variables that define how demanding a stretch of the
course is: corridor width, pipe speed, and spawn cadence.

### BrowserPopulationBirdLike

Simulation-facing browser contracts shared by playback helpers.

These types describe the minimum world state the browser needs while it is
reconstructing, rendering, or summarizing worker-produced frames.

### BrowserPopulationPipeLike

Pipe shape used by utility observation-vector helpers.

### PopulationRenderState

Mutable render-state model consumed by the population frame renderer.

The playback layer incrementally updates this state as worker snapshots
arrive, which lets rendering stay deterministic without re-deriving world
history from scratch each frame.

### RngLike

Minimal random source contract used by utility random helpers.

The narrow contract keeps deterministic spawn utilities portable across
browser and test contexts.

### TrailPoint

Trail point used by playback trail rendering cache.

A trail point stores where one bird was at one frame so the UI can draw a
short motion history behind active agents.

### TrailState

Mutable trail cache keyed by bird index for frame rendering.

This cache exists purely for visualization ergonomics; it is not part of the
worker simulation state.

## browser-entry/browser-entry.visualization.types.ts

### ColorLegendRow

Legend row model for network visualization color legends.

Each row labels a numeric interval and the color used to render it.

### ColorTier

Connection or bias tier used for color mapping ramps.

Visualization buckets continuous weights into legible color bands so humans
can scan sign and magnitude at a glance.

### NetworkLegendLayout

Precomputed legend panel layout used by visualization renderer.

Layout is resolved up front so the draw path can stay focused on painting,
not recomputing geometry every frame.

### NetworkNodeDimensionsLike

Pixel dimensions used for network-node rectangle rendering.

Keeping node box dimensions explicit makes legend and topology layout easier
to tune without hidden drawing constants.

### NetworkVisualizationHandle

Network-visualization contracts for the Flappy Bird browser demo.

One of the educational goals of the example is to let people watch evolved
controllers as structures, not just as scores. These types describe the
lightweight shapes used by the architecture panel so rendering logic can stay
decoupled from the full internal network implementation.

### PositionedNetworkNodeLike

Positioned node instance used by network visualization drawing.

Layout and rendering are split: first a node is assigned screen coordinates,
then the renderer paints it.

### VisualNetworkConnectionLike

Lightweight connection shape used by network visualization drawing.

The renderer only needs connectivity, weight, and enabled state, not the full
training-time behavior of a connection object.

### VisualNetworkNodeLike

Lightweight node shape used by network visualization drawing.

This shape keeps the renderer independent from the concrete Network class
while still exposing the semantic fields that matter visually.

## browser-entry/browser-entry.ts

### browser-entry

Browser demo public entry for the Flappy Bird example.

This boundary is the hand-off point between library consumers and the demo's
browser runtime. Calling `start` boots the canvas UI, worker channel, HUD,
and playback loop without exposing all of that internal orchestration as part
of the public API.

In other words, this file is intentionally tiny because it acts like a clean
facade over a much larger interactive system.

### FlappyBirdRunHandle

Public lifecycle contracts for the Flappy Bird browser runtime.

These types describe how the demo is started, stopped, and exposed on the
browser `window` object. They are intentionally small because callers should
control the demo at a high level without depending on private implementation
details.

### start

`(container: import("test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => Promise<import("test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle>`

## browser-entry/browser-entry.host.utils.ts

### createCanvasHostInternal

`(containerElement: HTMLElement) => import("test/examples/flappy_bird/browser-entry/host/host.types").CanvasHostResult`

Builds the browser demo host tree and returns rendering handles.

The orchestration is deliberately step-shaped: clear old DOM, build layout,
create canvases, wire resize behavior, render placeholders, then return the
handles the runtime will mutate during execution.

Parameters:
- `containerElement` - - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### updateStatsTableValues

`(statsValueByKey: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>, partialValues: Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, string>>) => void`

Applies partial stat updates to the rendered stats table.

The runtime writes HUD values incrementally, so the host exposes a narrow
partial-update helper rather than requiring full table redraws.

Parameters:
- `statsValueByKey` - - Lookup of stat keys to value cells.
- `partialValues` - - Subset of values to write this tick.

Returns: Nothing.

## browser-entry/browser-entry.math.utils.ts

### applyAlphaToHexColor

`(hexColor: string, alphaValue: number) => string`

Converts a six-digit hex color to rgba with the requested alpha.

Parameters:
- `hexColor` - - Color in `#RRGGBB` form.
- `alphaValue` - - Alpha value to apply.

Returns: rgba color string, or original value when not 6-digit hex.

### clamp

`(value: number, min: number, max: number) => number`

Clamps a numeric value to the inclusive `[min, max]` interval.

Parameters:
- `value` - - Candidate value.
- `min` - - Inclusive lower bound.
- `max` - - Inclusive upper bound.

Returns: Clamped value.

### clamp01

`(value: number) => number`

Clamps a numeric value to the inclusive `[0, 1]` interval.

Parameters:
- `value` - - Candidate value.

Returns: Value clamped between 0 and 1.

### interpolateValue

`(startValue: number, endValue: number, progress: number) => number`

Linear interpolation helper.

Parameters:
- `startValue` - - Start value at progress `0`.
- `endValue` - - End value at progress `1`.
- `progress` - - Normalized interpolation progress.

Returns: Interpolated value.

## browser-entry/browser-entry.spawn.utils.ts

### createBirdColor

`(birdIndex: number, totalBirds: number) => string`

Resolves deterministic bird color from palette index.

Parameters:
- `birdIndex` - - Bird index in current population.
- `totalBirds` - - Population size.

Returns: Hex color string.

### resolveGapCenterUpperBoundYPx

`(worldHeightPx: number) => number`

Resolves the exclusive upper bound used for gap-center sampling.

Educational note:
We cap dynamic viewport-derived bounds at the shared simulation maximum to
keep browser playback distribution aligned with trainer/evaluation defaults,
while still supporting smaller world heights.

Parameters:
- `worldHeightPx` - - Current world height.

Returns: Exclusive upper bound for `nextInt(minInclusive, maxExclusive)`.

### resolveNextSpawnGapCenterY

`(previousGapCenterYPx: number, rng: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike, worldHeightPx: number) => number`

Resolves next gap center with bounded per-pipe delta.

Parameters:
- `previousGapCenterYPx` - - Previous spawn gap center.
- `rng` - - Deterministic RNG.
- `worldHeightPx` - - World height used to clamp candidate gap centers.

Returns: Next gap center y-position.

### resolveNextSpawnGapSize

`(previousSpawnGapPx: number | undefined, difficultyProfile: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserDifficultyProfile, rng: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike) => number`

Resolves next spawn gap size using progressive shrink and jitter.

Parameters:
- `previousSpawnGapPx` - - Previous spawn gap size.
- `difficultyProfile` - - Active difficulty profile.
- `rng` - - Deterministic RNG.

Returns: Next spawn gap size.

### resolveNextSpawnIntervalFrames

`(previousSpawnIntervalFrames: number | undefined, difficultyProfile: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserDifficultyProfile) => number`

Resolves next spawn interval using progressive shrink.

Parameters:
- `previousSpawnIntervalFrames` - - Previous spawn interval.
- `difficultyProfile` - - Active difficulty profile.

Returns: Next spawn interval in frames.

### sampleGapCenterY

`(rng: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike, worldHeightPx: number) => number`

Samples a random gap center y-position.

Parameters:
- `rng` - - Deterministic RNG.
- `worldHeightPx` - - World height used to derive valid gap-center bounds.

Returns: Sampled y-position.

## browser-entry/browser-entry.stats.utils.ts

### createFlappyStatsTableRows

`(input: import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").CreateFlappyStatsTableRowsInput) => Partial<Record<import("test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>`

Builds stats table rows and returns value-cell lookup by key.

Parameters:
- `input` - - Table construction inputs.

Returns: Mapping from stat key to value cell.

### formatArchitectureStatsValue

`(architectureValue: string) => string`

Splits architecture suffix onto a second line for readability in the stats table.

Parameters:
- `architectureValue` - - Full architecture label.

Returns: Line-broken label value.

## browser-entry/browser-entry.playback.utils.ts

### browser-entry.playback.utils

Compatibility facade for the browser-entry playback boundary.

Older imports still reach playback through this file, while the real
implementation now lives in the dedicated playback folder. Keeping the facade
explicit preserves stable imports while letting the playback subsystem grow
into a clearer module boundary.

### animatePopulationEpisodeInternal

`(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, evolutionWorker: Worker, onFrameStats: (stats: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats) => void) => Promise<import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackEpisodeSummary>`

Internal playback orchestration entry retained for compatibility re-exports.

The implementation is shared with the public entry so legacy imports and the
newer folderized surface behave identically.

Parameters:
- `canvas` - - Target playback canvas.
- `context` - - Canvas 2D context.
- `evolutionWorker` - - Worker owning playback simulation state.
- `onFrameStats` - - Callback receiving per-frame playback telemetry.

Returns: Aggregate playback summary for the current episode.

## browser-entry/browser-entry.viewport.utils.ts

### resolvePipeSpawnXPx

`(visibleWorldWidthPx: number) => number`

Resolves the world-space x spawn position for new pipes.

Parameters:
- `visibleWorldWidthPx` - - Current visible world width.

Returns: Spawn x-position.

### resolveVisibleWorldHeightPx

`(canvas: HTMLCanvasElement) => number`

Resolves visible world height represented by the current canvas.

Educational note:
The current viewport model uses a 1:1 mapping between canvas pixels and
world-space pixels, so visible height is the canvas height directly.

Parameters:
- `canvas` - - Playback canvas.

Returns: Visible height in world-space pixels.

### resolveVisibleWorldWidthPx

`(canvas: HTMLCanvasElement) => number`

Resolves visible world width represented by the current canvas.

Educational note:
The current viewport model uses a 1:1 mapping between canvas pixels and
world-space pixels, so visible width is the canvas width directly.

Parameters:
- `canvas` - - Playback canvas.

Returns: Visible width in world-space pixels.

### resolveWorldViewport

`(canvas: HTMLCanvasElement) => import("test/examples/flappy_bird/browser-entry/browser-entry.render.types").ViewportInfo`

Resolves world viewport transformation based on canvas size.

Parameters:
- `canvas` - - Playback canvas.

Returns: Viewport scale and offsets.

## browser-entry/browser-entry.telemetry.utils.ts

### createMinorGcObserver

`(minorGcTimestampsMs: number[]) => PerformanceObserver | undefined`

Creates a PerformanceObserver that tracks minor GC events when supported.

Parameters:
- `minorGcTimestampsMs` - - Mutable minor-GC timestamp buffer.

Returns: Observer when supported; otherwise `undefined`.

### resolveEventsPerMinute

`(samples: number[]) => number`

Resolves events per minute from the latest sample window.

Parameters:
- `samples` - - Event timestamps.

Returns: Events-per-minute estimate.

### resolveHudUpdatesPerSecond

`(samples: number[]) => number`

Resolves HUD updates per second from the latest sample window.

Parameters:
- `samples` - - HUD update timestamps.

Returns: Updates-per-second estimate.

### trimSamplesToWindow

`(samples: number[], windowMs: number, nowMs: number) => void`

Trims timestamp samples to a sliding time window.

Parameters:
- `samples` - - Mutable timestamp buffer.
- `windowMs` - - Window width in milliseconds.
- `nowMs` - - Current timestamp.

Returns: Nothing.

## browser-entry/browser-entry.text-frame.utils.ts

### buildCenteredTitleBoxLines

`(centeredColumns: number, titleText: string) => string[]`

Builds an ASCII centered title box.

Parameters:
- `centeredColumns` - - Available centered column count.
- `titleText` - - Title text.

Returns: Three-row title box.

### buildOuterBoxLines

`(centeredColumns: number, totalRows: number) => string[]`

Builds an ASCII outer frame with closed borders.

Parameters:
- `centeredColumns` - - Centered column count.
- `totalRows` - - Total row count.

Returns: Frame lines.

### renderClosedOuterBox

`(input: import("test/examples/flappy_bird/browser-entry/browser-entry.render.types").RenderClosedOuterBoxInput) => void`

Renders a complete closed outer glyph box.

Parameters:
- `input` - - Rendering input object.

Returns: Nothing.

### renderStandaloneTitleBox

`(input: import("test/examples/flappy_bird/browser-entry/browser-entry.render.types").RenderStandaloneTitleBoxInput) => void`

Renders only the centered title box.

Parameters:
- `input` - - Rendering input object.

Returns: Nothing.

### resolveGlyphWidthPx

`(context: CanvasRenderingContext2D) => number`

Resolves a stable glyph width used for frame-column math.

Parameters:
- `context` - - Rendering context used to measure text.

Returns: Floored glyph width clamped to a minimum pixel value.

### resolveTextFrameMetrics

`(frameWidthPx: number, frameHeightPx: number, glyphWidthPx: number, rowHeightPx: number, minimumColumns: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.render.types").TextFrameMetrics`

Resolves core text-frame metrics for glyph box rendering.

Parameters:
- `frameWidthPx` - - Frame width.
- `frameHeightPx` - - Frame height.
- `glyphWidthPx` - - Measured glyph width.
- `rowHeightPx` - - Glyph row height.
- `minimumColumns` - - Minimum column count.

Returns: Text frame metrics.

## browser-entry/browser-entry.observation.utils.ts

### commitObservationMemoryStep

`(observationMemoryState: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState, observationFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, shouldFlap: boolean) => void`

Commits one browser decision step into temporal memory.

Parameters:
- `observationMemoryState` - - Mutable memory state for one bird.
- `observationFeatures` - - Structured features used for this decision.
- `shouldFlap` - - Action selected by the policy.

Returns: Nothing.

### hasAliveBirds

`(birds: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[]) => boolean`

Checks whether at least one bird remains alive.

Parameters:
- `birds` - - Population birds.

Returns: True when any bird is alive.

### resolveAliveBirdCount

`(birds: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[]) => number`

Counts birds that are still alive.

Parameters:
- `birds` - - Population birds.

Returns: Alive bird count.

### resolveFlapDecision

`(rawOutputs: unknown) => boolean`

Resolves flap/no-flap decision from network outputs.

Parameters:
- `rawOutputs` - - Activation output payload.

Returns: True when flap should trigger.

### resolveFramePrimaryWinnerIndex

`(birds: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[], includeAliveOnly: boolean) => number`

Resolves winner index for current frame.

Parameters:
- `birds` - - Population birds.
- `includeAliveOnly` - - When true, ignores dead birds.

Returns: Winner index, or `-1` when unavailable.

### resolveLeaderPipesPassed

`(birds: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[]) => number`

Resolves leading pipes-passed score in the population.

Parameters:
- `birds` - - Population birds.

Returns: Maximum pipes passed.

### resolveObservationVector

`(birdYPx: number, velocityYPxPerFrame: number, pipes: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike[], visibleWorldWidthPx: number, worldHeightPx: number, difficultyProfile: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserDifficultyProfile, activeSpawnIntervalFrames: number, observationMemoryState: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState) => { observationVector: number[]; observationFeatures: import("test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures; }`

Builds the normalized observation vector consumed by bird networks.

Parameters:
- `birdYPx` - - Bird y position.
- `velocityYPxPerFrame` - - Bird vertical velocity.
- `pipes` - - Current pipe list.
- `visibleWorldWidthPx` - - Current visible world width.
- `worldHeightPx` - - Current world height used for normalization and bounds.
- `difficultyProfile` - - Active difficulty profile.
- `activeSpawnIntervalFrames` - - Current spawn interval.
- `observationMemoryState` - - Temporal memory state for recurrent observation features.

Returns: Ordered normalized observation vector.

### resolveUpcomingPipes

`(pipes: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike[]) => [import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike | undefined, import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike | undefined]`

Resolves the next two upcoming pipes in front of the bird.

Parameters:
- `pipes` - - Current pipe list.

Returns: Tuple of first and second upcoming pipes.

## browser-entry/browser-entry.network-view.utils.ts

### browser-entry.network-view.utils

Compatibility facade for browser-entry network-view helpers.

Legacy imports still use this file while the network-view subsystem is split
into smaller topology, layout, label, and drawing modules.

### drawNetworkVisualization

`(context: CanvasRenderingContext2D, network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => void`

Draws a complete, layer-based visualization of the active network.

Conceptually, this is the main fold from network object to finished panel:
resolve scene state, compute layout, paint the graph, then paint overlays.

Parameters:
- `context` - - Canvas 2D drawing context.
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Nothing.

### resolveNetworkArchitectureLabel

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => string`

Resolves compact architecture label text for headers and HUD rows.

The label compresses the active network into a short human-readable summary:
input size, hidden-layer structure, output size, and graph size metadata.

Parameters:
- `network` - - Network to describe.
- `inputSize` - - Configured input size.
- `outputSize` - - Configured output size.

Returns: Readable architecture label.

### resolveNetworkVisualizationHeightPx

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => number`

Resolves responsive visualization canvas height from network shape.

Dense or deeper networks need more vertical room to stay readable, so panel
height is driven by topology rather than fixed to a single constant.

Parameters:
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Recommended height in pixels.

### resolveNetworkVisualizationLayers

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Topology resolution helpers for the browser network view.

These helpers answer a key visualization question: how should the current
network be partitioned into ordered layers so layout and architecture labels
stay meaningful even when some metadata is missing?

## browser-entry/browser-entry.visualization.utils.ts

### browser-entry.visualization.utils

Compatibility facade for browser-entry network visualization helpers.

Legacy imports still flow through this file while the visualization subsystem
is organized into smaller, clearer modules under the dedicated folder.

### createColorLegendRows

`(scale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale, symbol: "w" | "b") => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]`

Legend-layout helpers for network visualization.

The legend explains how colors map back to numeric weights and biases. These
helpers turn color scales into labeled rows and place the legend so it stays
readable across different canvas sizes.

### createLogDivergingColorTiers

`(input: { maxAbsValue: number; centerBlueThreshold: number; negativePalette: readonly string[]; centerBluePalette: readonly string[]; positivePalette: readonly string[]; logarithmicSteepness: number; edgeStartAbsValue?: number | undefined; edgeTierCount?: number | undefined; }) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[]`

Color-scale synthesis helpers for network visualization.

These utilities convert raw connection weights and node biases into tiered
neon color scales. The goal is not photorealism; it is interpretability. A
reader should be able to glance at the network panel and see where strong
positive, strong negative, and near-zero values live.

### drawBiasNodesLayer

`(context: CanvasRenderingContext2D, positionedNodes: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], nodeDimensions: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, biasScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws all network nodes with bias labels.

The node layer pairs each rectangle with a compact bias label so the panel can
show both topology and a lightweight hint of parameter state.

Parameters:
- `context` - - Render context.
- `positionedNodes` - - Positioned nodes.
- `nodeDimensions` - - Node dimensions.
- `biasScale` - - Dynamic bias color scale.

Returns: Nothing.

### drawNetworkColorLegend

`(context: CanvasRenderingContext2D, architectureLabel: string, colorScales: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales) => void`

Draws the color legend for connections and node bias values.

This legend is what turns the panel from "colorful art" into an interpretable
instrument: it tells the viewer what each weight and bias color actually
means numerically.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Compact architecture description.
- `colorScales` - - Connection and bias color scales.

Returns: Nothing.

### drawNetworkVisualizationHeader

`(context: CanvasRenderingContext2D, architectureLabel: string) => void`

Draws network architecture header text.

The header gives viewers a compact architecture summary before they inspect
individual nodes and edges.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Header label.

Returns: Nothing.

### drawWeightedConnectionsLayer

`(context: CanvasRenderingContext2D, runtimeConnections: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike[], positionByNodeIndex: Map<number, import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>, connectionScale: import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws weighted connection lines.

Connection styling carries semantic meaning: color encodes magnitude and sign,
while dash patterns and auxiliary marks help distinguish disabled or negative
edges in a way that still reads quickly on a dense graph.

Parameters:
- `context` - - Render context.
- `runtimeConnections` - - Runtime connection list.
- `positionByNodeIndex` - - Node layout map.
- `connectionScale` - - Dynamic connection color scale.

Returns: Nothing.

### formatNodeBiasLabel

`(nodeBias: number) => string`

Formats node bias labels with fixed sign and precision.

Consistent sign and precision make dense node labels easier to scan quickly in
the rendered network panel.

Parameters:
- `nodeBias` - - Node bias value.

Returns: Label text.

### resolveBiasRangeColor

`(nodeBias: number) => string`

Resolves bias color for a raw node bias.

Bias colors follow the same diverging logic as connection colors so the legend
remains conceptually consistent across channels.

Parameters:
- `nodeBias` - - Node bias.

Returns: Tier color.

### resolveConnectionRangeColor

`(connectionWeight: number) => string`

Resolves connection color for a raw weight.

This small helper is useful when one-off drawing code wants the same color
semantics as the full dynamic scale machinery.

Parameters:
- `connectionWeight` - - Connection weight.

Returns: Tier color.

### resolveDefaultNetworkLegendLayout

`(context: CanvasRenderingContext2D, network: import("src/architecture/network").default | undefined) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves default legend layout from internal tier definitions.

This convenience helper is used when the caller wants a layout driven by the
currently active network and does not need to assemble the intermediate rows
manually.

Parameters:
- `context` - - Render context.
- `network` - - Active network instance.

Returns: Legend layout.

### resolveNetworkLegendLayout

`(context: CanvasRenderingContext2D, connectionLegendRows: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[], biasLegendRows: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves network legend layout from canvas constraints.

The legend layout adapts between regular and compact modes so the network
panel can stay informative on smaller viewports without swallowing the whole
canvas.

Parameters:
- `context` - - Render context.
- `connectionLegendRows` - - Connection legend rows.
- `biasLegendRows` - - Bias legend rows.

Returns: Computed legend layout.

### resolveNetworkVisualizationColorScales

`(network: import("src/architecture/network").default | undefined) => import("test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales`

Resolves dynamic connection/bias color scales from the active network range.

The active network may contain only a narrow slice of the full theoretical
value range, so the legend adapts to what is currently present instead of
always rendering a fixed generic scale.

Parameters:
- `network` - - Active network.

Returns: Dynamic scales used by graph drawing and legend rows.

### resolveNetworkVisualizationLayers

`(network: import("src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Topology resolution helpers for the browser network view.

These helpers answer a key visualization question: how should the current
network be partitioned into ordered layers so layout and architecture labels
stay meaningful even when some metadata is missing?

### resolveTierColor

`(value: number, tiers: import("test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[], aboveTierColor: string) => string`

Resolves a color from ordered tier definitions.

This is the final classification step that maps one numeric weight or bias to
the swatch color the renderer should paint.

Parameters:
- `value` - - Numeric value to classify.
- `tiers` - - Ordered tier list.
- `aboveTierColor` - - Fallback color for values above the last tier.

Returns: Resolved color string.

## browser-entry/browser-entry.worker-channel.utils.ts

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
