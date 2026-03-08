# browser-entry

## browser-entry/browser-entry.types.ts

### BrowserDifficultyProfile

Difficulty profile consumed by simulation observation helpers.

### BrowserPopulationBirdLike

Bird shape used by utility winner/leader resolver helpers.

### BrowserPopulationPipeLike

Pipe shape used by utility observation-vector helpers.

### ColorLegendRow

Legend row model for network visualization color legends.

### ColorTier

Connection or bias tier used for color mapping ramps.

### CreateFlappyStatsTableRowsInput

Input contract for declarative runtime stats table row builder.

### EvolutionGenerationPayload

Worker payload describing evolved generation summary values.

### EvolutionGenerationReadyMessage

Worker message emitted when a generation has completed evolving.

### EvolutionPlaybackStepMessage

Worker message carrying one playback step and aggregate markers.

### EvolutionPlaybackStepSnapshot

Per-frame snapshot received from the worker playback channel.

### EvolutionWorkerErrorMessage

Worker message emitted for simulation/playback errors.

### EvolutionWorkerMessage

Union of all supported worker messages consumed by browser entry.

### FlappyBirdRunHandle

Handle returned by `start` for controlling demo execution lifecycle.

### FlappyStatsCategoryColors

Color pair used for stats category key/value styling.

### FlappyStatsKey

Runtime stats table key union used across browser-entry helpers.

### FlappyStatsRowDescriptor

Declarative row descriptor for the runtime stats table.

### FlappyStatsTableCells

Runtime lookup map of stat keys to writable value cells.

### NetworkLegendLayout

Precomputed legend panel layout used by visualization renderer.

### NetworkNodeDimensionsLike

Pixel dimensions used for network-node rectangle rendering.

### NetworkVisualizationHandle

Draw callback contract for network architecture panel updates.

### PlaybackFrameStats

Lightweight per-frame telemetry emitted to HUD update callback.

### PopulationBird

Renderable bird state snapshot emitted by the playback worker.

### PopulationPipe

Renderable pipe state snapshot emitted by the playback worker.

### PopulationRenderState

Mutable render-state model consumed by the population frame renderer.

### PositionedNetworkNodeLike

Positioned node instance used by network visualization drawing.

### RenderClosedOuterBoxInput

Input contract for outer frame rendering helper.

### RenderStandaloneTitleBoxInput

Input contract for standalone title frame rendering helper.

### RngLike

Minimal random source contract used by utility random helpers.

### RuntimeWindow

Runtime window contract for the Flappy Bird browser demo.

### SerializedNetwork

Loose JSON-compatible network payload used by worker messages.

### TextFrameMetrics

Canvas text-grid metrics used for frame rendering layout helpers.

### TrailPoint

Trail point used by playback trail rendering cache.

### TrailState

Mutable trail cache keyed by bird index for frame rendering.

### ViewportInfo

Viewport transform values for world-to-canvas rendering.

### VisualNetworkConnectionLike

Lightweight connection shape used by network visualization drawing.

### VisualNetworkNodeLike

Lightweight node shape used by network visualization drawing.

## browser-entry/browser-entry.stats.types.ts

### browser-entry.stats.types

Runtime stats table key union used across browser-entry helpers.

### CreateFlappyStatsTableRowsInput

Input contract for declarative runtime stats table row builder.

### FlappyStatsCategoryColors

Color pair used for stats category key/value styling.

### FlappyStatsKey

Runtime stats table key union used across browser-entry helpers.

### FlappyStatsRowDescriptor

Declarative row descriptor for the runtime stats table.

### FlappyStatsTableCells

Runtime lookup map of stat keys to writable value cells.

## browser-entry/browser-entry.render.types.ts

### browser-entry.render.types

Input contract for standalone title frame rendering helper.

### RenderClosedOuterBoxInput

Input contract for outer frame rendering helper.

### RenderStandaloneTitleBoxInput

Input contract for standalone title frame rendering helper.

### TextFrameMetrics

Canvas text-grid metrics used for frame rendering layout helpers.

### ViewportInfo

Viewport transform values for world-to-canvas rendering.

## browser-entry/browser-entry.worker.types.ts

### browser-entry.worker.types

Loose JSON-compatible network payload used by worker messages.

### EvolutionGenerationPayload

Worker payload describing evolved generation summary values.

### EvolutionGenerationReadyMessage

Worker message emitted when a generation has completed evolving.

### EvolutionPlaybackStepMessage

Worker message carrying one playback step and aggregate markers.

### EvolutionPlaybackStepSnapshot

Per-frame snapshot received from the worker playback channel.

### EvolutionWorkerErrorMessage

Worker message emitted for simulation/playback errors.

### EvolutionWorkerMessage

Union of all supported worker messages consumed by browser entry.

### PlaybackFrameStats

Lightweight per-frame telemetry emitted to HUD update callback.

### PopulationBird

Renderable bird state snapshot emitted by the playback worker.

### PopulationPipe

Renderable pipe state snapshot emitted by the playback worker.

### SerializedNetwork

Loose JSON-compatible network payload used by worker messages.

## browser-entry/browser-entry.runtime.types.ts

### browser-entry.runtime.types

Handle returned by `start` for controlling demo execution lifecycle.

### FlappyBirdRunHandle

Handle returned by `start` for controlling demo execution lifecycle.

### RuntimeWindow

Runtime window contract for the Flappy Bird browser demo.

## browser-entry/browser-entry.simulation.types.ts

### BrowserDifficultyProfile

Difficulty profile consumed by simulation observation helpers.

### BrowserPopulationBirdLike

Bird shape used by utility winner/leader resolver helpers.

### BrowserPopulationPipeLike

Pipe shape used by utility observation-vector helpers.

### PopulationRenderState

Mutable render-state model consumed by the population frame renderer.

### RngLike

Minimal random source contract used by utility random helpers.

### TrailPoint

Trail point used by playback trail rendering cache.

### TrailState

Mutable trail cache keyed by bird index for frame rendering.

## browser-entry/browser-entry.visualization.types.ts

### ColorLegendRow

Legend row model for network visualization color legends.

### ColorTier

Connection or bias tier used for color mapping ramps.

### NetworkLegendLayout

Precomputed legend panel layout used by visualization renderer.

### NetworkNodeDimensionsLike

Pixel dimensions used for network-node rectangle rendering.

### NetworkVisualizationHandle

Draw callback contract for network architecture panel updates.

### PositionedNetworkNodeLike

Positioned node instance used by network visualization drawing.

### VisualNetworkConnectionLike

Lightweight connection shape used by network visualization drawing.

### VisualNetworkNodeLike

Lightweight node shape used by network visualization drawing.

## browser-entry/browser-entry.ts

### FlappyBirdRunHandle

Handle returned by `start` for controlling demo execution lifecycle.

### start

`(container: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/runtime/runtime.types").RuntimeContainerTarget) => Promise<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.runtime.types").FlappyBirdRunHandle>`

## browser-entry/browser-entry.host.utils.ts

### createCanvasHostInternal

`(containerElement: HTMLElement) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/host/host.types").CanvasHostResult`

Builds the browser demo host tree and returns rendering handles.

Parameters:
- `containerElement` - - Root host container.

Returns: Canvas handles, stats cells and network render callback.

### updateStatsTableValues

`(statsValueByKey: Partial<Record<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>, partialValues: Partial<Record<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, string>>) => void`

Applies partial stat updates to the rendered stats table.

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

`(previousGapCenterYPx: number, rng: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike, worldHeightPx: number) => number`

Resolves next gap center with bounded per-pipe delta.

Parameters:
- `previousGapCenterYPx` - - Previous spawn gap center.
- `rng` - - Deterministic RNG.
- `worldHeightPx` - - World height used to clamp candidate gap centers.

Returns: Next gap center y-position.

### resolveNextSpawnGapSize

`(previousSpawnGapPx: number | undefined, difficultyProfile: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserDifficultyProfile, rng: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike) => number`

Resolves next spawn gap size using progressive shrink and jitter.

Parameters:
- `previousSpawnGapPx` - - Previous spawn gap size.
- `difficultyProfile` - - Active difficulty profile.
- `rng` - - Deterministic RNG.

Returns: Next spawn gap size.

### resolveNextSpawnIntervalFrames

`(previousSpawnIntervalFrames: number | undefined, difficultyProfile: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserDifficultyProfile) => number`

Resolves next spawn interval using progressive shrink.

Parameters:
- `previousSpawnIntervalFrames` - - Previous spawn interval.
- `difficultyProfile` - - Active difficulty profile.

Returns: Next spawn interval in frames.

### sampleGapCenterY

`(rng: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").RngLike, worldHeightPx: number) => number`

Samples a random gap center y-position.

Parameters:
- `rng` - - Deterministic RNG.
- `worldHeightPx` - - World height used to derive valid gap-center bounds.

Returns: Sampled y-position.

## browser-entry/browser-entry.stats.utils.ts

### createFlappyStatsTableRows

`(input: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.stats.types").CreateFlappyStatsTableRowsInput) => Partial<Record<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.stats.types").FlappyStatsKey, HTMLTableCellElement>>`

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

### animatePopulationEpisodeInternal

`(canvas: HTMLCanvasElement, context: CanvasRenderingContext2D, evolutionWorker: Worker, onFrameStats: (stats: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.worker.types").PlaybackFrameStats) => void) => Promise<import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/playback").PlaybackEpisodeSummary>`

Internal playback orchestration entry retained for compatibility re-exports.

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

`(canvas: HTMLCanvasElement) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.render.types").ViewportInfo`

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

`(input: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.render.types").RenderClosedOuterBoxInput) => void`

Renders a complete closed outer glyph box.

Parameters:
- `input` - - Rendering input object.

Returns: Nothing.

### renderStandaloneTitleBox

`(input: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.render.types").RenderStandaloneTitleBoxInput) => void`

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

`(frameWidthPx: number, frameHeightPx: number, glyphWidthPx: number, rowHeightPx: number, minimumColumns: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.render.types").TextFrameMetrics`

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

`(observationMemoryState: import("C:/NeatapticTS/test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState, observationFeatures: import("C:/NeatapticTS/test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures, shouldFlap: boolean) => void`

Commits one browser decision step into temporal memory.

Parameters:
- `observationMemoryState` - - Mutable memory state for one bird.
- `observationFeatures` - - Structured features used for this decision.
- `shouldFlap` - - Action selected by the policy.

Returns: Nothing.

### hasAliveBirds

`(birds: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[]) => boolean`

Checks whether at least one bird remains alive.

Parameters:
- `birds` - - Population birds.

Returns: True when any bird is alive.

### resolveAliveBirdCount

`(birds: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[]) => number`

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

`(birds: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[], includeAliveOnly: boolean) => number`

Resolves winner index for current frame.

Parameters:
- `birds` - - Population birds.
- `includeAliveOnly` - - When true, ignores dead birds.

Returns: Winner index, or `-1` when unavailable.

### resolveLeaderPipesPassed

`(birds: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationBirdLike[]) => number`

Resolves leading pipes-passed score in the population.

Parameters:
- `birds` - - Population birds.

Returns: Maximum pipes passed.

### resolveObservationVector

`(birdYPx: number, velocityYPxPerFrame: number, pipes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike[], visibleWorldWidthPx: number, worldHeightPx: number, difficultyProfile: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserDifficultyProfile, activeSpawnIntervalFrames: number, observationMemoryState: import("C:/NeatapticTS/test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationMemoryState) => { observationVector: number[]; observationFeatures: import("C:/NeatapticTS/test/examples/flappy_bird/simulation-shared/simulation-shared.types").SharedObservationFeatures; }`

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

`(pipes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike[]) => [import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike | undefined, import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").BrowserPopulationPipeLike | undefined]`

Resolves the next two upcoming pipes in front of the bird.

Parameters:
- `pipes` - - Current pipe list.

Returns: Tuple of first and second upcoming pipes.

## browser-entry/browser-entry.network-view.utils.ts

### drawNetworkVisualization

`(context: CanvasRenderingContext2D, network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => void`

Draws a complete, layer-based visualization of the active network.

Parameters:
- `context` - - Canvas 2D drawing context.
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Nothing.

### resolveNetworkArchitectureLabel

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => string`

Resolves compact architecture label text for headers and HUD rows.

Parameters:
- `network` - - Network to describe.
- `inputSize` - - Configured input size.
- `outputSize` - - Configured output size.

Returns: Readable architecture label.

### resolveNetworkVisualizationHeightPx

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => number`

Resolves responsive visualization canvas height from network shape.

Parameters:
- `network` - - Network to visualize.
- `inputSize` - - Input-layer size.
- `outputSize` - - Output-layer size.

Returns: Recommended height in pixels.

### resolveNetworkVisualizationLayers

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Resolves layered node groups for network-view layout and rendering.

Educational note:
Layer grouping is a network-view concern because it drives sizing, node
placement, and architecture presentation. Visualization code can still reuse
the result, but this helper now lives with the module that owns layout.

Parameters:
- `network` - - Runtime network instance.
- `inputSize` - - Input count fallback.
- `outputSize` - - Output count fallback.

Returns: Layered nodes for rendering.

## browser-entry/browser-entry.visualization.utils.ts

### createColorLegendRows

`(scale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale, symbol: "w" | "b") => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]`

Creates legend rows from ordered tiers.

Parameters:
- `scale` - - Dynamic color scale containing bounds, tiers, and overflow color.
- `symbol` - - Label symbol.

Returns: Legend rows.

### createLogDivergingColorTiers

`(input: { maxAbsValue: number; centerBlueThreshold: number; negativePalette: readonly string[]; centerBluePalette: readonly string[]; positivePalette: readonly string[]; logarithmicSteepness: number; edgeStartAbsValue?: number | undefined; edgeTierCount?: number | undefined; }) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[]`

Builds logarithmic diverging color tiers with a center band and edge extension.

Parameters:
- `input` - - Tier creation options.

Returns: Ordered tier list.

### drawBiasNodesLayer

`(context: CanvasRenderingContext2D, positionedNodes: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike[], nodeDimensions: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkNodeDimensionsLike, biasScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws all network nodes with bias labels.

Parameters:
- `context` - - Render context.
- `positionedNodes` - - Positioned nodes.
- `nodeDimensions` - - Node dimensions.
- `biasScale` - - Dynamic bias color scale.

Returns: Nothing.

### drawNetworkColorLegend

`(context: CanvasRenderingContext2D, architectureLabel: string, colorScales: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales) => void`

Draws the color legend for connections and node bias values.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Compact architecture description.
- `colorScales` - - Connection and bias color scales.

Returns: Nothing.

### drawNetworkVisualizationHeader

`(context: CanvasRenderingContext2D, architectureLabel: string) => void`

Draws network architecture header text.

Parameters:
- `context` - - Render context.
- `architectureLabel` - - Header label.

Returns: Nothing.

### drawWeightedConnectionsLayer

`(context: CanvasRenderingContext2D, runtimeConnections: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkConnectionLike[], positionByNodeIndex: Map<number, import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").PositionedNetworkNodeLike>, connectionScale: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").DynamicColorScale) => void`

Draws weighted connection lines.

Parameters:
- `context` - - Render context.
- `runtimeConnections` - - Runtime connection list.
- `positionByNodeIndex` - - Node layout map.
- `connectionScale` - - Dynamic connection color scale.

Returns: Nothing.

### formatNodeBiasLabel

`(nodeBias: number) => string`

Formats node bias labels with fixed sign and precision.

Parameters:
- `nodeBias` - - Node bias value.

Returns: Label text.

### resolveBiasRangeColor

`(nodeBias: number) => string`

Resolves bias color for a raw node bias.

Parameters:
- `nodeBias` - - Node bias.

Returns: Tier color.

### resolveConnectionRangeColor

`(connectionWeight: number) => string`

Resolves connection color for a raw weight.

Parameters:
- `connectionWeight` - - Connection weight.

Returns: Tier color.

### resolveDefaultNetworkLegendLayout

`(context: CanvasRenderingContext2D, network: import("C:/NeatapticTS/src/architecture/network").default | undefined) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves default legend layout from internal tier definitions.

Parameters:
- `context` - - Render context.
- `network` - - Active network instance.

Returns: Legend layout.

### resolveNetworkLegendLayout

`(context: CanvasRenderingContext2D, connectionLegendRows: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[], biasLegendRows: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorLegendRow[]) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").NetworkLegendLayout`

Resolves network legend layout from canvas constraints.

Parameters:
- `context` - - Render context.
- `connectionLegendRows` - - Connection legend rows.
- `biasLegendRows` - - Bias legend rows.

Returns: Computed legend layout.

### resolveNetworkVisualizationColorScales

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/visualization/visualization.types").NetworkVisualizationColorScales`

Resolves dynamic connection/bias color scales from the active network range.

Parameters:
- `network` - - Active network.

Returns: Dynamic scales used by graph drawing and legend rows.

### resolveNetworkVisualizationLayers

`(network: import("C:/NeatapticTS/src/architecture/network").default | undefined, inputSize: number, outputSize: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").VisualNetworkNodeLike[][]`

Resolves layered node groups for network-view layout and rendering.

Educational note:
Layer grouping is a network-view concern because it drives sizing, node
placement, and architecture presentation. Visualization code can still reuse
the result, but this helper now lives with the module that owns layout.

Parameters:
- `network` - - Runtime network instance.
- `inputSize` - - Input count fallback.
- `outputSize` - - Output count fallback.

Returns: Layered nodes for rendering.

### resolveTierColor

`(value: number, tiers: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/browser-entry.visualization.types").ColorTier[], aboveTierColor: string) => string`

Resolves a color from ordered tier definitions.

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
