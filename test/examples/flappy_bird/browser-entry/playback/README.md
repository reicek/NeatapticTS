# browser-entry/playback

## browser-entry/playback/playback.types.ts

### PlaybackEdgeBounds

Axis-aligned visible world bounds used for edge-aware trail fading.

### PlaybackStarfieldLayerSpec

Declarative recipe for building one cached starfield parallax layer.

### StarTile

Shared type contract for starfield tile rendering layers.

A tile is pre-rendered and repeated horizontally to draw efficient
parallax backgrounds during playback.

## browser-entry/playback/playback.starfield.types.ts

### playback.starfield.types

Shared type contract for starfield tile rendering layers.

A tile is pre-rendered and repeated horizontally to draw efficient
parallax backgrounds during playback.

### CreateStarTileCanvasOptions

Input contract for pre-rendering one deterministic starfield tile.

The generated tile is cached and repeated horizontally during playback,
so every field here affects both the visual look and the parallax cost.

### PlaybackStarfieldLayerSpec

Declarative recipe for building one cached starfield parallax layer.

### StarfieldCanvasDimensions

Normalized canvas dimensions used by browser and offscreen tile creation.

### StarPlacement

Deterministic placement and appearance for one rendered star sprite.

### StarTile

Shared type contract for starfield tile rendering layers.

A tile is pre-rendered and repeated horizontally to draw efficient
parallax backgrounds during playback.

## browser-entry/playback/playback.ts

### animatePopulationEpisode

`(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, evolutionWorker: Worker, onFrameStats: (stats: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats) => void) => Promise<import("test/examples/flappy_bird/browser-entry/playback/playback").PlaybackEpisodeSummary>`

Public playback entry point used by browser runtime orchestration.

Parameters:
- `canvas` - - Target playback canvas.
- `context` - - Canvas 2D context.
- `evolutionWorker` - - Worker owning playback simulation state.
- `onFrameStats` - - Callback receiving per-frame playback telemetry.

Returns: Aggregate playback summary for the current episode.

### animatePopulationEpisodeInternal

`(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, evolutionWorker: Worker, onFrameStats: (stats: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats) => void) => Promise<import("test/examples/flappy_bird/browser-entry/playback/playback").PlaybackEpisodeSummary>`

Internal playback orchestration entry retained for compatibility re-exports.

Parameters:
- `canvas` - - Target playback canvas.
- `context` - - Canvas 2D context.
- `evolutionWorker` - - Worker owning playback simulation state.
- `onFrameStats` - - Callback receiving per-frame playback telemetry.

Returns: Aggregate playback summary for the current episode.

### applyPlaybackStepSnapshot

`(sessionContext: PlaybackSessionContext, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Applies the latest worker snapshot to render state and trail caches.

Parameters:
- `sessionContext` - - Shared mutable playback session state.
- `snapshot` - - Worker snapshot for the current playback batch.

Returns: Nothing.

### createInitialPlaybackLoopState

`() => PlaybackLoopState`

Creates the mutable loop state used while processing playback steps.

Returns: Initialized loop state and aggregate summary values.

### createInitialRenderState

`(viewportDimensions: { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }) => import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState`

Creates the initial render state used before the first worker snapshot.

Parameters:
- `viewportDimensions` - - Current visible world dimensions.

Returns: Initialized population render state.

### createInitialTrailState

`() => import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState`

Creates the initial trail state used before any snapshots have been applied.

Returns: Empty trail state for all birds.

### emitPlaybackFrameStats

`(iterationContext: PlaybackIterationContext, playbackStepPayload: { snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }) => void`

Resolves leader telemetry and emits the public frame-stats callback.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.
- `playbackStepPayload` - - Worker playback result for the current iteration.

Returns: Nothing.

### initializePlaybackSessionContext

`(canvas: HTMLCanvasElement, evolutionWorker: Worker) => PlaybackSessionContext`

Initializes worker playback and local state mirrors for one episode.

Parameters:
- `canvas` - - Target playback canvas.
- `evolutionWorker` - - Worker owning playback simulation state.

Returns: Session context shared across the playback loop.

### PlaybackEpisodeSummary

### requestPlaybackStepPayload

`(iterationContext: PlaybackIterationContext) => Promise<{ snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }>`

Requests one playback step batch from the evolution worker.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.

Returns: Worker playback step payload for the current iteration.

### resolvePlaybackEpisodeSummary

`(summary: PlaybackMutableSummary) => import("test/examples/flappy_bird/browser-entry/playback/playback").PlaybackEpisodeSummary`

Folds the mutable loop summary into the public playback summary shape.

Parameters:
- `summary` - - Mutable loop summary accumulated during playback.

Returns: Public playback episode summary.

### resolvePlaybackViewportDimensions

`(canvas: HTMLCanvasElement) => { visibleWorldWidthPx: number; visibleWorldHeightPx: number; }`

Resolves the current visible playback viewport dimensions from the canvas.

Parameters:
- `canvas` - - Target playback canvas.

Returns: Visible world width and height in pixels.

### runPlaybackIteration

`(iterationContext: PlaybackIterationContext) => Promise<void>`

Executes one playback iteration from viewport sync through render pacing.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.

Returns: Nothing.

### runPlaybackLoop

`(iterationContext: PlaybackIterationContext) => Promise<void>`

Runs playback iterations until the worker reports that the episode is done.

Parameters:
- `iterationContext` - - Shared loop dependencies and mutable playback state.

Returns: Nothing.

### syncPlaybackViewportDimensions

`(canvas: HTMLCanvasElement, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => void`

Synchronizes the render state viewport fields with the current canvas size.

Parameters:
- `canvas` - - Target playback canvas.
- `renderState` - - Mutable render state updated in place.

Returns: Nothing.

### updatePlaybackLoopCompletion

`(loopState: PlaybackLoopState, playbackStepPayload: { snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }) => void`

Updates the loop summary when the worker reports playback completion.

Parameters:
- `loopState` - - Mutable playback loop state.
- `playbackStepPayload` - - Worker playback result for the current iteration.

Returns: Nothing.

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

Yields until the next browser animation frame.

Returns: Promise resolved on next animation frame.

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

## browser-entry/playback/playback.starfield.service.ts

### createStarTile

`(layerSpec: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").PlaybackStarfieldLayerSpec, tileHeightPx: number) => import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarTile`

Creates one cached tile layer from a declarative layer specification.

Parameters:
- `layerSpec` - - Density and motion contract for a starfield layer.
- `tileHeightPx` - - Height of the visible sky band in pixels.

Returns: Cached tile metadata for parallax drawing.

### resolveStarfieldTiles

`(visibleWorldHeightPx: number) => readonly import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarTile[]`

Resolves (and lazily creates) cached starfield tile layers for the viewport.

Parameters:
- `visibleWorldHeightPx` - - Viewport height in world pixels.

Returns: Ordered far/mid/near starfield tiles.

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

`(options: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").CreateStarTileCanvasOptions) => CanvasImageSource`

Pre-renders a deterministic tile that can be reused across animation frames.

Parameters:
- `options` - - Declarative drawing recipe for one parallax layer.

Returns: Canvas image source containing the rendered star strip.

### initializeStarTileContext

`(options: { tileContext: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D; canvas: HTMLCanvasElement | OffscreenCanvas; blurPx: number; }) => void`

Clears the canvas and applies the glow settings shared by all rendered stars.

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

### renderPopulationFrame

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState) => void`

Draws one simulation frame for the current population state.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `trailState` - - Leader trail render cache.

Returns: Nothing.

### updateTrailState

`(trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => void`

Updates the trail cache from the latest frame snapshot.

Parameters:
- `trailState` - - Mutable trail state.
- `renderState` - - Current render state.

Returns: Nothing.

## browser-entry/playback/playback.trail.utils.ts

### clamp01

`(value: number) => number`

Clamps a number to the inclusive [0, 1] range.

Parameters:
- `value` - - Candidate value.

Returns: Clamped value.

### pushTrailPoint

`(trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], frameIndex: number, yPosition: number) => void`

Appends one trail point while enforcing max retained history length.

Parameters:
- `trailPoints` - - Mutable trail collection.
- `frameIndex` - - Source frame index.
- `yPosition` - - Bird y position.

Returns: Nothing.

### resolveEdgeOpacityFactor

`(pointXPx: number, pointYPx: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds) => number`

Converts distance-to-edge into a normalized opacity factor.

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

### applyPlaybackSnapshot

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot) => void`

Applies worker snapshot data to the mutable playback render state.

Parameters:
- `renderState` - - Mutable render state mirror used by the browser.
- `snapshot` - - Worker playback snapshot for the current render tick.

Returns: Nothing.

### resolveLeaderFramesSurvived

`(renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => number`

Resolves the maximum survived-frame count in the current render state.

Parameters:
- `renderState` - - Current render state.

Returns: Maximum frames survived by any bird.

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

Shared alias for worker playback-step payload.

### PlaybackStepRequest

Request payload for one playback-step worker call.

### resolvePlaybackCompletionSummary

`(playbackStepPayload: { snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, latestLeaderPipesPassed: number, latestLeaderFramesSurvived: number) => { averagePipesPassed: number; p90FramesSurvived: number; winnerPipesPassed: number; winnerFramesSurvived: number; }`

Resolves final playback summary values when worker reports completion.

Parameters:
- `playbackStepPayload` - - Playback payload returned by worker.
- `latestLeaderPipesPassed` - - Last observed leader pipes passed fallback.
- `latestLeaderFramesSurvived` - - Last observed leader frames fallback.

Returns: Final aggregate playback summary.

### resolvePlaybackFrameStats

`(playbackStepPayload: { snapshot: import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").EvolutionPlaybackStepSnapshot; instrumentation?: { activationCallsPerFrame: number; simulationStepsPerRaf: number; } | undefined; done: boolean; averagePipesPassed?: number | undefined; p90FramesSurvived?: number | undefined; winnerPipesPassed?: number | undefined; winnerFramesSurvived?: number | undefined; }, frameIndex: number, activeBirdCount: number, leaderPipesPassed: number, leaderFramesSurvived: number) => import("test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats`

Resolves HUD playback frame stats from worker payload and leader metrics.

Parameters:
- `playbackStepPayload` - - Playback payload returned by worker.
- `frameIndex` - - Current render frame index.
- `activeBirdCount` - - Number of alive birds in current frame.
- `leaderPipesPassed` - - Current frame leader pipes passed.
- `leaderFramesSurvived` - - Current frame leader survived frames.

Returns: Normalized per-frame HUD telemetry payload.

### resolvePlaybackStepRequest

`(input: import("test/examples/flappy_bird/browser-entry/playback/playback.worker-channel.utils").ResolvePlaybackStepRequestInput) => import("test/examples/flappy_bird/browser-entry/playback/playback.worker-channel.utils").ResolvePlaybackStepRequestResult`

Resolves step count and request payload for the next worker playback batch.

Parameters:
- `input` - - Current frame budget and viewport dimensions.

Returns: Request payload plus carried-over fractional frame budget.

### ResolvePlaybackStepRequestInput

Input used to resolve next playback-step request and budget remainder.

### ResolvePlaybackStepRequestResult

Output for resolved playback-step request and frame-budget remainder.
