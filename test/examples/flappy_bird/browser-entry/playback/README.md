# browser-entry/playback

## browser-entry/playback/playback.types.ts

### PlaybackEdgeBounds

Shared playback utility types for the browser-entry subsystem.

These types support rendering concerns that cut across multiple playback
helpers, such as edge-aware trail fading and cached parallax backgrounds.

### PlaybackStarfieldLayerSpec

Declarative recipe for building one cached starfield parallax layer.

Each layer spec describes how dense, bright, blurred, and fast one visual
depth plane should feel.

### StarTile

Shared type contract for starfield tile rendering layers.

A tile is pre-rendered and repeated horizontally to draw efficient
parallax backgrounds during playback.

Separating the image type from the tile record lets the same starfield logic
work with ordinary canvases and `OffscreenCanvas` when available.

## browser-entry/playback/playback.starfield.types.ts

### playback.starfield.types

Starfield and parallax rendering contracts for playback backgrounds.

The playback view uses a cached layered starfield to add depth without paying
a large per-frame rendering cost. These types define the tile, layer, and
deterministic placement data needed for that effect.

### CreateStarTileCanvasOptions

Input contract for pre-rendering one deterministic starfield tile.

The generated tile is cached and repeated horizontally during playback,
so every field here affects both the visual look and the parallax cost.

### PlaybackStarfieldLayerSpec

Declarative recipe for building one cached starfield parallax layer.

Each layer spec describes how dense, bright, blurred, and fast one visual
depth plane should feel.

### StarfieldCanvasDimensions

Normalized canvas dimensions used by browser and offscreen tile creation.

The creation path works with both `HTMLCanvasElement` and `OffscreenCanvas`,
so dimensions are stored in a narrow shared shape rather than tied to one DOM
type.

### StarPlacement

Deterministic placement and appearance for one rendered star sprite.

Determinism matters here because cached starfield tiles should remain stable
across redraws instead of sparkling randomly every frame.

### StarTile

Shared type contract for starfield tile rendering layers.

A tile is pre-rendered and repeated horizontally to draw efficient
parallax backgrounds during playback.

Separating the image type from the tile record lets the same starfield logic
work with ordinary canvases and `OffscreenCanvas` when available.

### StarTileImage

Starfield and parallax rendering contracts for playback backgrounds.

The playback view uses a cached layered starfield to add depth without paying
a large per-frame rendering cost. These types define the tile, layer, and
deterministic placement data needed for that effect.

## browser-entry/playback/playback.orchestration.types.ts

### PlaybackEpisodeSummary

Playback orchestration contracts for the Flappy Bird browser demo.

These types describe the moving pieces of one playback episode: the public
summary returned at the end, the mutable loop bookkeeping used while frames
are streaming, and the session context mirrored locally in the browser.

### PlaybackIterationContext

Shared dependencies and mutable state used by one playback iteration.

Grouping these fields into one context object keeps the iteration services
declarative and avoids long parameter lists across the playback loop.

### PlaybackLoopState

Mutable loop bookkeeping shared across playback iterations.

This is the browser-side state machine for the playback loop: how much
simulation budget is being requested, whether the episode has finished, and
what aggregate summary has been observed so far.

### PlaybackMutableSummary

Mutable playback summary extended with latest leader telemetry fallbacks.

During playback the browser may need temporary "latest known" values before
the worker emits final aggregate statistics, so the mutable form carries both
final fields and rolling fallbacks.

### PlaybackSessionContext

Shared mutable playback state mirrored locally while worker playback runs.

The worker remains the source of truth for simulation, but the browser keeps
lightweight mirrored state for rendering, trail accumulation, and loop
orchestration.

## browser-entry/playback/playback.ts

### animatePopulationEpisode

`(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, evolutionWorker: Worker, onFrameStats: (stats: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats) => void) => Promise<import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackEpisodeSummary>`

Public playback orchestration for the Flappy Bird browser demo.

Playback is the bridge between off-thread simulation and on-screen
visualization. The worker advances the world and streams packed snapshots;
this layer mirrors enough state locally to animate those snapshots at browser
frame cadence, render trails and backgrounds, and emit HUD telemetry.

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

### PlaybackEpisodeSummary

Playback orchestration contracts for the Flappy Bird browser demo.

These types describe the moving pieces of one playback episode: the public
summary returned at the end, the mutable loop bookkeeping used while frames
are streaming, and the session context mirrored locally in the browser.

## browser-entry/playback/playback.errors.ts

### playback.errors

Error message emitted when playback requires RAF but it is unavailable.

### PLAYBACK_ANIMATION_FRAME_UNAVAILABLE_ERROR_MESSAGE

### PlaybackAnimationFrameUnavailableError

Error thrown when a playback frame wait is requested without RAF support.

## browser-entry/playback/playback.constants.ts

### cachedStarfieldTilesByHeight

## browser-entry/playback/playback.loop.service.ts

### nextAnimationFrame

`() => Promise<void>`

Browser frame-pacing helpers for playback animation.

The playback loop advances worker simulation in batches but still presents
frames at browser animation cadence. This module isolates the
`requestAnimationFrame` dependency so playback orchestration can read more
clearly and fail with a targeted error when RAF is unavailable.

## browser-entry/playback/playback.render.service.ts

### drawPipeNeonOutline

`(context: CanvasRenderingContext2D, rectangleLeftPx: number, rectangleTopPx: number, rectangleWidthPx: number, rectangleHeightPx: number) => void`

Draws a simplified neon outline around a pipe rectangle.

Parameters:
- `context` - - Canvas 2D context.
- `rectangleLeftPx` - - Rectangle left position.
- `rectangleTopPx` - - Rectangle top position.
- `rectangleWidthPx` - - Rectangle width.
- `rectangleHeightPx` - - Rectangle height.

Returns: Nothing.

## browser-entry/playback/playback.session.services.ts

### createInitialPlaybackLoopState

`() => import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackLoopState`

Creates the mutable loop state used while processing playback steps.

This is the browser's running notebook for one episode: budget, completion
flag, and the latest known aggregate outcome metrics.

Returns: Initialized loop state and aggregate summary values.

### createInitialRenderState

`(viewportDimensions: { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }) => import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState`

Creates the initial render state used before the first worker snapshot.

The browser starts from an empty-but-shaped render state so rendering helpers
can assume the object graph exists even before the worker has emitted any
population geometry.

Parameters:
- `viewportDimensions` - - Current visible world dimensions.

Returns: Initialized population render state.

### createInitialTrailState

`() => import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState`

Creates the initial trail state used before any snapshots have been applied.

Trails are purely visual history, so they begin empty and accumulate only as
playback frames are observed.

Returns: Empty trail state for all birds.

### initializePlaybackSessionContext

`(canvas: HTMLCanvasElement, evolutionWorker: Worker) => import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackSessionContext`

Initializes worker playback and local state mirrors for one episode.

This is the point where the browser and worker agree on a fresh episode. The
browser sends the initial viewport dimensions to the worker, then builds the
local render and summary mirrors that will be updated as snapshots arrive.

Parameters:
- `canvas` - - Target playback canvas.
- `evolutionWorker` - - Worker owning playback simulation state.

Returns: Session context shared across the playback loop.

### resolvePlaybackEpisodeSummary

`(summary: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackMutableSummary) => import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackEpisodeSummary`

Folds the mutable loop summary into the public playback summary shape.

The public summary is intentionally smaller than the internal loop state. It
exposes the outcome, not the browser's intermediate bookkeeping.

Parameters:
- `summary` - - Mutable loop summary accumulated during playback.

Returns: Public playback episode summary.

### resolvePlaybackViewportDimensions

`(canvas: HTMLCanvasElement) => { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }`

Session initialization and summary-folding helpers for playback.

These services answer three orchestration questions:
1. What viewport is the browser currently showing?
2. What local mirror state should exist before the first worker snapshot?
3. How should mutable loop state be folded back into a public summary?

### syncPlaybackViewportDimensions

`(canvas: HTMLCanvasElement, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => void`

Synchronizes the render state viewport fields with the current canvas size.

Playback can continue while the canvas size changes, so the browser refreshes
its local viewport mirror rather than assuming dimensions stay fixed.

Parameters:
- `canvas` - - Target playback canvas.
- `renderState` - - Mutable render state updated in place.

Returns: Nothing.

## browser-entry/playback/playback.starfield.service.ts

### resolveStarfieldTiles

`(visibleWorldHeightPx: number) => readonly import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarTile[]`

Resolves (and lazily creates) cached starfield tile layers for the viewport.

Parameters:
- `visibleWorldHeightPx` - - Viewport height in world pixels.

Returns: Ordered far/mid/near starfield tiles.

## browser-entry/playback/playback.iteration.services.ts

### applyPlaybackStepSnapshot

`(sessionContext: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackSessionContext, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Applies the latest worker snapshot to render state and trail caches.

Parameters:
- `sessionContext` - - Shared mutable playback session state.
- `snapshot` - - Worker snapshot for the current playback batch.

Returns: Nothing.

### emitPlaybackFrameStats

`(iterationContext: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackIterationContext, playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }) => void`

Resolves leader telemetry and emits the public frame-stats callback.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.
- `playbackStepPayload` - - Worker playback result for the current iteration.

Returns: Nothing.

### requestPlaybackStepPayload

`(iterationContext: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackIterationContext) => Promise<{ requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }>`

Requests one playback step batch from the evolution worker.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.

Returns: Worker playback step payload for the current iteration.

### runPlaybackIteration

`(iterationContext: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackIterationContext) => Promise<void>`

Executes one playback iteration from viewport sync through render pacing.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.

Returns: Nothing.

### runPlaybackLoop

`(iterationContext: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackIterationContext) => Promise<void>`

Runs playback iterations until the worker reports that the episode is done.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.

Returns: Nothing.

### updatePlaybackLoopCompletion

`(loopState: import("test/examples/flappy_bird/browser-entry/playback/playback.orchestration.types").PlaybackLoopState, playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }) => void`

Updates the loop summary when the worker reports playback completion.

Parameters:
- `loopState` - - Mutable playback loop state.
- `playbackStepPayload` - - Worker playback result for the current iteration.

Returns: Nothing.

## browser-entry/playback/playback.starfield.services.ts

### createCanvasSizeFallback

`(canvasDimensions: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarfieldCanvasDimensions) => HTMLCanvasElement`

Creates a size-only fallback so non-browser tests can skip rendering safely.

Parameters:
- `canvasDimensions` - - Already-normalized pixel dimensions.

Returns: Minimal canvas-shaped object cast to the compatible return type.

### createCompatibleCanvas

`(widthPx: number, heightPx: number) => HTMLCanvasElement | OffscreenCanvas`

Creates a browser-compatible canvas with clamped integer dimensions.

Parameters:
- `widthPx` - - Requested tile width in pixels.
- `heightPx` - - Requested tile height in pixels.

Returns: Offscreen canvas when supported, otherwise a DOM canvas fallback.

### createDocumentCanvasIfSupported

`(canvasDimensions: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarfieldCanvasDimensions) => HTMLCanvasElement | null`

Creates a DOM canvas when document APIs are available.

Parameters:
- `canvasDimensions` - - Already-normalized pixel dimensions.

Returns: DOM canvas instance or `null` when unavailable.

### createOffscreenCanvasIfSupported

`(canvasDimensions: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarfieldCanvasDimensions) => OffscreenCanvas | null`

Creates an offscreen canvas when the current runtime supports it.

Parameters:
- `canvasDimensions` - - Already-normalized pixel dimensions.

Returns: Offscreen canvas instance or `null` when unavailable.

### createStarTileCanvas

`(options: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").CreateStarTileCanvasOptions) => import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarTileImage`

Pre-renders a deterministic tile that can be reused across animation frames.

Parameters:
- `options` - - Declarative drawing recipe for one parallax layer.

Returns: Canvas image source containing the rendered star strip.

### initializeStarTileContext

`(options: { tileContext: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D; canvas: HTMLCanvasElement | OffscreenCanvas; }) => void`

Clears the canvas and applies the neutral settings shared by all rendered stars.

Parameters:
- `options` - - Context initialization dependencies.

Returns: Nothing. The provided context is mutated in place.

### normalizeCanvasDimensions

`(widthPx: number, heightPx: number) => import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarfieldCanvasDimensions`

Normalizes requested canvas dimensions into positive integer pixel sizes.

Parameters:
- `widthPx` - - Requested width in pixels.
- `heightPx` - - Requested height in pixels.

Returns: Clamped integer dimensions safe for canvas allocation.

### renderSeededStars

`(options: { tileContext: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D; seededRandom: () => number; canvasOptions: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").CreateStarTileCanvasOptions; }) => void`

Draws all stars for one tile using a seeded random source.

Parameters:
- `options` - - Drawing context, seed source, and tile recipe.

Returns: Nothing. The provided context is mutated in place.

### resetStarTileContext

`(tileContext: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D) => void`

Restores neutral drawing state so later canvas consumers start from defaults.

Parameters:
- `tileContext` - - 2D context used to render the star tile.

Returns: Nothing. The provided context is mutated in place.

### resolveStarPlacement

`(options: { seededRandom: () => number; tileWidthPx: number; tileHeightPx: number; minSizePx: number; maxSizePx: number; minAlpha: number; maxAlpha: number; }) => import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarPlacement`

Resolves one deterministic star placement and appearance from the seeded RNG.

Parameters:
- `options` - - Random source and star placement bounds.

Returns: Pixel location, square size, and alpha for one rendered star.

### resolveStarTileContext

`(canvas: HTMLCanvasElement | OffscreenCanvas) => CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null`

Resolves the rendering context used for star tile pre-rendering.

Parameters:
- `canvas` - - Compatible canvas returned by the runtime-specific factory.

Returns: A 2D drawing context when rendering is supported.

## browser-entry/playback/playback.frame-render.service.ts

### playback.frame-render.service

Compatibility facade for playback frame rendering.

Older imports still reach the frame renderer through this file while the
implementation now lives in the dedicated frame-render folder.

### renderPopulationFrame

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState) => void`

High-level frame-render orchestration for playback.

This boundary coordinates one visual frame of the playback experience. It
resolves the scene, prepares the canvas, paints the background and entities,
and maintains the short champion trail used for motion emphasis.

### updateTrailState

`(trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => void`

Updates the trail cache from the latest frame snapshot.

The renderer intentionally keeps only a short champion trail instead of full
history for every bird, which keeps the visual emphasis clear and the per-frame
work small.

Parameters:
- `trailState` - - Mutable trail state.
- `renderState` - - Current render state.

Returns: Nothing.

## browser-entry/playback/playback.starfield.layer.services.ts

### createStarTile

`(layerSpec: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").PlaybackStarfieldLayerSpec, tileHeightPx: number) => import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarTile`

Creates one cached tile layer from a declarative layer specification.

Parameters:
- `layerSpec` - - Density and motion contract for a starfield layer.
- `tileHeightPx` - - Height of the visible sky band in pixels.

Returns: Cached tile metadata for parallax drawing.

## browser-entry/playback/playback.render.pipe-outline.service.ts

### drawPipeNeonOutline

`(context: CanvasRenderingContext2D, rectangleLeftPx: number, rectangleTopPx: number, rectangleWidthPx: number, rectangleHeightPx: number) => void`

Draws a simplified neon outline around a pipe rectangle.

Parameters:
- `context` - - Canvas 2D context.
- `rectangleLeftPx` - - Rectangle left position.
- `rectangleTopPx` - - Rectangle top position.
- `rectangleWidthPx` - - Rectangle width.
- `rectangleHeightPx` - - Rectangle height.

Returns: Nothing.

### resolveAlignedPipeOutlineRectangle

`(input: { rectangleLeftPx: number; rectangleTopPx: number; rectangleWidthPx: number; rectangleHeightPx: number; }) => { alignedLeftPx: number; alignedTopPx: number; alignedWidthPx: number; alignedHeightPx: number; }`

Resolves a pixel-aligned rectangle used by the pipe outline renderer.

Parameters:
- `input` - - Raw pipe rectangle values.

Returns: Aligned rectangle ready for outline rendering.

### resolvePipeOutlinePath

`(alignedRectangle: { alignedLeftPx: number; alignedTopPx: number; alignedWidthPx: number; alignedHeightPx: number; }) => Path2D`

Resolves the reusable outline path for one pipe body and its entrance rim.

Parameters:
- `alignedRectangle` - - Pixel-aligned rectangle used by the outline renderer.

Returns: Path containing the outer pipe outline and optional entrance rim.

## browser-entry/playback/playback.trail.utils.ts

### clamp01

`(value: number) => number`

Clamps a number to the inclusive [0, 1] range.

The trail renderer combines several normalized fade factors, so keeping this
utility local to the module makes the intent obvious: every opacity channel
must remain safe for direct canvas alpha use.

Parameters:
- `value` - - Candidate value.

Returns: Clamped value.

### pushChampionTrailPoint

`(trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], frameIndex: number, yPosition: number) => void`

Appends one point to the champion-only short trail history.

The browser highlights the current leader with a shorter, denser trail than
the rest of the flock. Using a dedicated helper keeps that policy explicit in
the call site instead of scattering champion-specific retention numbers
through the playback renderer.

Parameters:
- `trailPoints` - - Mutable champion trail collection.
- `frameIndex` - - Source frame index.
- `yPosition` - - Bird y position.

Returns: Nothing.

### pushTrailPoint

`(trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], frameIndex: number, yPosition: number, maxRetainedPoints: number) => void`

Appends one trail point while enforcing the maximum retained history length.

Playback trails are intentionally modeled as short rolling histories rather
than unbounded path logs. That keeps the neon afterimage readable, prevents
old turns from dominating the current frame, and avoids per-frame growth in a
long-running browser session.

Parameters:
- `trailPoints` - - Mutable trail collection.
- `frameIndex` - - Source frame index.
- `yPosition` - - Bird y position.
- `maxRetainedPoints` - - Optional maximum retained trail history length.

Returns: Nothing.

### resolveEdgeOpacityFactor

`(pointXPx: number, pointYPx: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds) => number`

Converts distance-to-edge into a normalized opacity factor.

Trail points fade as they approach the viewport border so the rendered path
feels cropped by the camera instead of abruptly chopped off. This mirrors the
common animation principle of easing visual intensity near a frame boundary.

Returns 0 exactly on or beyond an edge and rises to 1 once distance exceeds
the configured fade band.

Parameters:
- `pointXPx` - - Point x position.
- `pointYPx` - - Point y position.
- `edgeBounds` - - Visible world bounds used for edge distance checks.

Returns: Opacity multiplier in [0, 1].

### resolveTrailLifetimeOpacityFactor

`(frameOffset: number, maxTrailFrameOffset: number) => number`

Converts trail age into a normalized opacity factor.

This helper implements the other half of the afterimage effect: recent trail
samples should read as energetic and bright, while older samples should fade
away smoothly so the viewer's eye stays anchored to the current flock motion.

Oldest retained history approaches 0 opacity; newest approaches 1.

Parameters:
- `frameOffset` - - Frames between this point and newest trail point.
- `maxTrailFrameOffset` - - Oldest age offset currently retained by trail.

Returns: Opacity multiplier in [0, 1].

## browser-entry/playback/playback.render.utils.ts

### PlaybackBirdRenderStyle

Pure render-style result for one bird body draw pass.

### resolveBirdRenderStyle

`(birdIndex: number, championBirdIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle`

Resolves opacity, body color, and champion marker for one bird.

Parameters:
- `birdIndex` - - Index of the bird currently being rendered.
- `championBirdIndex` - - Resolved champion index for the frame.

Returns: Pure style payload used by the render service.

### resolveChampionBirdIndex

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => number`

Resolves the champion bird index for the current render frame.

Champion selection first prefers the primary winner resolver and then
falls back to the first alive bird when no winner index is available.

Parameters:
- `renderState` - - Current frame render snapshot.

Returns: Champion index or `-1` when no bird is alive.

## browser-entry/playback/playback.snapshot.utils.ts

### playback.snapshot.utils

Compatibility facade for playback snapshot helpers.

Legacy imports still reach snapshot synchronization through this file while
the implementation now lives in the dedicated snapshot folder.

### applyPlaybackSnapshot

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Snapshot synchronization helpers for playback.

The worker streams packed typed-array snapshots, while the browser renderer
wants stable mutable arrays of pipes and birds. This module performs that
translation in place so playback can stay fast and allocation-light.

### resolveLeaderFramesSurvived

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => number`

Small summary helpers derived from hydrated playback render state.

Once a packed snapshot has been synchronized into the browser render state,
these helpers compute simple aggregate values needed by the HUD and playback
reporting flow.

## browser-entry/playback/playback.starfield.utils.ts

### createSeededRandom

`(seed: number) => () => number`

Creates a deterministic pseudo-random generator for starfield tile layouts.

Parameters:
- `seed` - - Unsigned integer seed.

Returns: Function that yields values in the range [0, 1).

### positiveModulo

`(value: number, modulo: number) => number`

Resolves positive modulo suitable for horizontal tiling offsets.

Parameters:
- `value` - - Input value to wrap.
- `modulo` - - Modulus base.

Returns: Wrapped value in [0, modulo).

## browser-entry/playback/playback.worker-channel.utils.ts

### PlaybackStepPayload

Shared alias for the worker playback-step payload.

This keeps the playback worker-channel modules focused on playback semantics
instead of long imported protocol names.

### PlaybackStepRequest

Playback-specific worker-channel contracts.

These types sit above the lower-level browser worker protocol and describe the
request budgeting plus summary data flow used by the playback loop.

### resolvePlaybackCompletionSummary

`(playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, latestLeaderPipesPassed: number, latestLeaderFramesSurvived: number) => { averagePipesPassed: number; p90FramesSurvived: number; winnerPipesPassed: number; winnerFramesSurvived: number; }`

Resolves final playback summary values when the worker reports completion.

Some end-of-episode aggregates may be omitted from the worker payload, so the
browser falls back to the latest leader values it has already observed during
playback.

Parameters:
- `playbackStepPayload` - - Playback payload returned by worker.
- `latestLeaderPipesPassed` - - Last observed leader pipes passed fallback.
- `latestLeaderFramesSurvived` - - Last observed leader frames fallback.

Returns: Final aggregate playback summary.

### resolvePlaybackFrameStats

`(playbackStepPayload: { requestId: number; snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, frameIndex: number, activeBirdCount: number, leaderPipesPassed: number, leaderFramesSurvived: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats`

Summary and HUD helpers for playback worker-channel results.

Once the worker replies with a playback-step payload, these helpers turn that
raw protocol data into the browser-facing telemetry and end-of-episode summary
values used elsewhere in the playback loop.

### resolvePlaybackStepRequest

`(input: import("test/examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.types").ResolvePlaybackStepRequestInput) => import("test/examples/flappy_bird/browser-entry/playback/worker-channel/playback.worker-channel.types").ResolvePlaybackStepRequestResult`

Playback batch-request helpers for the browser worker channel.

The playback loop accumulates simulation budget in fractional units, then
converts that budget into integer worker step requests on each render tick.

### ResolvePlaybackStepRequestInput

Input used to resolve the next playback-step request and budget remainder.

Playback uses a fractional frame budget so browser render cadence and worker
simulation cadence can be smoothed together over time.

### ResolvePlaybackStepRequestResult

Output for the resolved playback-step request and frame-budget remainder.

The resolved request records both the integer step batch to send now and the
leftover fractional budget to carry into the next render tick.
