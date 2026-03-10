# browser-entry/playback/background/ground-grid

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.types.ts

### PlaybackBackgroundGroundGridRequest

Narrow request required to render the playback ground grid.

### PlaybackBackgroundGroundGridResolvedScene

Geometry and style package resolved before drawing the ground grid.

### PlaybackBackgroundGroundGridSceneContext

Immutable scene context resolved for one lower-band ground-grid pass.

### PlaybackBackgroundGroundGridSourceScene

Helper alias used when adapting the shared background scene context.

### PlaybackBackgroundGroundGridStyle

Theme-owned style contract for the neon ground grid.

### PlaybackGroundGridAnchorBounds

Visible horizon bounds projected onto the bottom anchor line.

### PlaybackGroundGridAnchorBoundsInput

Input used when projecting a visible horizon span onto anchor space.

### PlaybackGroundGridAnchorProjectionInput

Input used when projecting one horizon x-position onto the floor anchor line.

### PlaybackGroundGridGeometry

Pure geometry bundle generated before canvas drawing begins.

### PlaybackGroundGridHorizontalGeometry

Cached horizontal geometry bundle reused across matching scene sizes.

### PlaybackGroundGridHorizontalGeometryFactory

`() => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridHorizontalGeometry`

Lazy builder used when one horizontal geometry cache entry is missing.

### PlaybackGroundGridLineSegment

Declarative line segment model used by the ground-grid renderer.

### PlaybackGroundGridPulse

One visible pulse square rendered above the grid lines.

### PlaybackGroundGridPulseInput

Input contract used while resolving one deterministic pulse event.

### PlaybackGroundGridPulseOrientation

Travel orientation used by lightweight pulse overlays.

### PlaybackGroundGridPulsePath

Simplified path used by one visible pulse event.

### PlaybackGroundGridPulseTrackThicknessInput

Input used when adapting a pulse position into a local track thickness.

### PlaybackGroundGridPulseTravelRatioInput

Direction and timing state used when resolving pulse travel progress.

### PlaybackGroundGridSegmentBatch

Ordered batch of line segments that share one render style.

### PlaybackGroundGridVerticalCycleContext

Wrapped vertical-cycle state derived from scroll for one frame.

### PlaybackGroundGridVerticalGeometry

Cached vertical geometry bundle reused across one wrapped scroll cycle.

### PlaybackGroundGridVerticalGeometryFactory

`() => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalGeometry`

Lazy builder used when one vertical geometry cache entry is missing.

### PlaybackGroundGridVerticalRayInput

Internal helper contract used while generating vertical-ray sub-segments.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.ts

### renderPlaybackBackgroundGroundGrid

`(context: CanvasRenderingContext2D, sourceScene: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSourceScene, request: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridRequest) => void`

Draws the neon lower-band ground grid beneath the horizon.

The grid is intentionally stylized rather than physically realistic: fixed
horizontal depth bands compress toward the horizon, while moving perspective
rays slide sideways but still converge to the centered vanishing point.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sourceScene` - - Shared lower-band geometry from the background module.
- `request` - - Shared parallax scroll input for the current frame.

Returns: Nothing.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.services.ts

### drawGroundGridFog

`(context: CanvasRenderingContext2D, resolvedScene: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridResolvedScene) => void`

Draws the lower-band atmospheric wash behind the neon line work.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.

Returns: Nothing.

### drawGroundGridPulse

`(context: CanvasRenderingContext2D, pulse: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulse | null, fillColor: string, glowColor: string) => void`

Draws one pulse square above the grid lines and below gameplay entities.

Parameters:
- `context` - - Canvas 2D drawing context.
- `pulse` - - Visible pulse square for the current frame.
- `fillColor` - - Core neon fill color.
- `glowColor` - - Outer glow color used behind the pulse.

Returns: Nothing.

### drawGroundGridSegmentBatch

`(context: CanvasRenderingContext2D, batch: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridSegmentBatch) => void`

Draws one batch of neon line segments that share one render style.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batch` - - Ordered line-segment batch that shares one render style.

Returns: Nothing.

### drawGroundGridSegmentBatches

`(context: CanvasRenderingContext2D, batches: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridSegmentBatch[], lineColor: string, glowColor: string) => void`

Draws one ordered collection of neon segment batches.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batches` - - Ordered line-segment batches to render.
- `lineColor` - - Core neon stroke color.
- `glowColor` - - Outer glow color used for bloom.

Returns: Nothing.

### drawPlaybackGroundGrid

`(context: CanvasRenderingContext2D, resolvedScene: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridResolvedScene, geometry: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridGeometry) => void`

Draws the resolved neon ground grid inside the lower background band.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.
- `geometry` - - Precomputed horizontal and vertical line segments.

Returns: Nothing.

### strokePlaybackGroundGridBatch

`(context: CanvasRenderingContext2D, batch: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridSegmentBatch) => void`

Strokes one ground-grid batch using a cached path when the environment supports it.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batch` - - Ordered line-segment batch that shares one render style.

Returns: Nothing.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.constants.ts

### FLAPPY_BACKGROUND_GROUND_GRID_STYLE

### FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS

### FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT

### FLAPPY_GROUND_GRID_FOG_ALPHA

### FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO

### FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT

### FLAPPY_GROUND_GRID_MAX_ALPHA

### FLAPPY_GROUND_GRID_MAX_BLUR_PX

### FLAPPY_GROUND_GRID_MAX_THICKNESS_PX

### FLAPPY_GROUND_GRID_MIN_ALPHA

### FLAPPY_GROUND_GRID_MIN_BLUR_PX

### FLAPPY_GROUND_GRID_MIN_THICKNESS_PX

### FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT

### FLAPPY_GROUND_GRID_PULSE_ALPHA

### FLAPPY_GROUND_GRID_PULSE_GLOW_ALPHA_RATIO

### FLAPPY_GROUND_GRID_PULSE_GLOW_BLUR_PX

### FLAPPY_GROUND_GRID_PULSE_GLOW_SIZE_MULTIPLIER

### FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS

### FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS

### FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX

### FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX

### FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX

### FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO

### FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX

### FLAPPY_GROUND_GRID_SCROLL_RATIO

### FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX

### FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX

### FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR

### FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT

### FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO

### FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.cache.services.ts

### ensureGroundGridViewportCacheValidity

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => string`

Ensures ground-grid caches only retain entries for the current viewport size.

The ground grid is derived from viewport width and total scene height, so a
page resize invalidates every cached geometry variant and fog gradient.

Parameters:
- `sceneContext` - - Current lower-band scene geometry.

Returns: Stable viewport-size cache key for the current frame.

### resolveCachedGroundGridFogGradient

`(context: CanvasRenderingContext2D, sceneCacheKey: string, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, fogColor: string) => CanvasGradient`

Resolves a cached fog gradient for one canvas and local scene.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `sceneContext` - - Current lower-band scene geometry.
- `fogColor` - - Theme-owned fog color token.

Returns: Cached fog gradient aligned to the lower-band scene.

### resolveCachedGroundGridHorizontalGeometry

`(sceneCacheKey: string, factory: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridHorizontalGeometryFactory) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridHorizontalGeometry`

Resolves cached horizontal geometry for one scene.

Parameters:
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `factory` - - Lazy geometry builder used when the cache misses.

Returns: Cached horizontal geometry bundle for the scene.

### resolveCachedGroundGridVerticalGeometry

`(cycleCacheKey: string, factory: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalGeometryFactory) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalGeometry`

Resolves cached vertical geometry for one scene and wrapped offset cycle.

Parameters:
- `cycleCacheKey` - - Scene-and-offset cache key for the active frame.
- `factory` - - Lazy geometry builder used when the cache misses.

Returns: Cached vertical geometry bundle for the cycle.

### resolveGroundGridSceneCacheKey

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => string`

Resolves the stable local-scene cache key for ground-grid geometry.

Parameters:
- `sceneContext` - - Current lower-band scene geometry.

Returns: Scene key suitable for static horizontal and vertical cache entries.

### resolveGroundGridVerticalCycleCacheKey

`(sceneCacheKey: string, wrappedOffsetPx: number) => string`

Resolves the cache key for one wrapped vertical-geometry cycle.

Parameters:
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `wrappedOffsetPx` - - Quantized wrapped offset within one lane cycle.

Returns: Cycle key used for vertical geometry reuse.

### resolveGroundGridViewportCacheKey

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => string`

Resolves the viewport-size cache key used by the ground-grid caches.

Parameters:
- `sceneContext` - - Current lower-band scene geometry.

Returns: Cache key that changes whenever the page size changes.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.utils.ts

### interpolatePlaybackGroundGridPoint

`(startXPx: number, startYPx: number, endXPx: number, endYPx: number, interpolationRatio: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.math.utils").PlaybackGroundGridPoint`

Interpolates one point along a perspective ray.

Parameters:
- `startXPx` - - Bottom anchor x-position.
- `startYPx` - - Bottom anchor y-position.
- `endXPx` - - Vanishing-point x-position.
- `endYPx` - - Vanishing-point y-position.
- `interpolationRatio` - - Normalized 0..1 position along the ray.

Returns: Interpolated point on the perspective ray.

### resolvePlaybackGroundGridDepthCurve

`(depthRatio: number) => number`

Maps a normalized depth ratio into a stronger synthwave spacing curve.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Curved depth ratio used for line placement and styling.

### resolvePlaybackGroundGridDepthFromHorizonDistance

`(distanceToHorizonPx: number, maximumDistanceToHorizonPx: number) => number`

Resolves normalized depth from a vertical distance away from the horizon.

Parameters:
- `distanceToHorizonPx` - - Vertical distance from the vanishing horizon.
- `maximumDistanceToHorizonPx` - - Largest visible vertical horizon distance.

Returns: Normalized 0..1 depth where 0 is at the horizon and 1 is nearest.

### resolvePlaybackGroundGridGeometry

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, frameIndex: number, scrollBasePx: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridGeometry`

Builds the line geometry for the neon ground grid.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `frameIndex` - - Current deterministic playback frame index.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Horizontal depth bands and perspective rays for the current frame.

### resolvePlaybackGroundGridLineAlpha

`(depthRatio: number) => number`

Resolves neon alpha for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Opacity for the rendered line.

### resolvePlaybackGroundGridLineBlur

`(depthRatio: number) => number`

Resolves glow blur for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Blur radius for the rendered line.

### resolvePlaybackGroundGridLineThickness

`(depthRatio: number) => number`

Resolves stroke thickness for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Stroke width in pixels.

### resolvePlaybackGroundGridSceneContext

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSourceScene) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext`

Resolves the shared scene context used by the ground-grid renderer.

Parameters:
- `sceneContext` - - Lower-band geometry provided by the background module.

Returns: Narrow scene contract consumed by grid-specific helpers.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.math.utils.ts

### interpolatePlaybackGroundGridPoint

`(startXPx: number, startYPx: number, endXPx: number, endYPx: number, interpolationRatio: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.math.utils").PlaybackGroundGridPoint`

Interpolates one point along a perspective ray.

Parameters:
- `startXPx` - - Bottom anchor x-position.
- `startYPx` - - Bottom anchor y-position.
- `endXPx` - - Vanishing-point x-position.
- `endYPx` - - Vanishing-point y-position.
- `interpolationRatio` - - Normalized 0..1 position along the ray.

Returns: Interpolated point on the perspective ray.

### PlaybackGroundGridPoint

Small point value used when interpolating positions along one grid ray.

### resolvePlaybackGroundGridDepthCurve

`(depthRatio: number) => number`

Maps a normalized depth ratio into a stronger synthwave spacing curve.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Curved depth ratio used for line placement and styling.

### resolvePlaybackGroundGridDepthFromHorizonDistance

`(distanceToHorizonPx: number, maximumDistanceToHorizonPx: number) => number`

Resolves normalized depth from a vertical distance away from the horizon.

Parameters:
- `distanceToHorizonPx` - - Vertical distance from the vanishing horizon.
- `maximumDistanceToHorizonPx` - - Largest visible vertical horizon distance.

Returns: Normalized 0..1 depth where 0 is at the horizon and 1 is nearest.

### resolvePlaybackGroundGridLineAlpha

`(depthRatio: number) => number`

Resolves neon alpha for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Opacity for the rendered line.

### resolvePlaybackGroundGridLineBlur

`(depthRatio: number) => number`

Resolves glow blur for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Blur radius for the rendered line.

### resolvePlaybackGroundGridLineThickness

`(depthRatio: number) => number`

Resolves stroke thickness for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Stroke width in pixels.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.pulse.utils.ts

### resolvePlaybackGroundGridHorizontalPulsePath

`(horizontalPulsePaths: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath[], pulseSlotIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath | null`

Selects one thick-enough horizontal band for the current pulse slot.

Parameters:
- `horizontalPulsePaths` - - Cached horizontal pulse paths eligible for travel.
- `pulseSlotIndex` - - Zero-based pulse slot index.

Returns: Horizontal pulse path, or null when none are suitable.

### resolvePlaybackGroundGridPulse

`(input: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulseInput) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulse | null`

Resolves one rare, deterministic pulse square for the current frame.

Parameters:
- `input` - - Current frame timing and visible pulse path candidates.

Returns: Visible pulse square, or null when the current slot is inactive.

### resolvePlaybackGroundGridPulseOrientation

`(pulseSlotIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulseOrientation`

Resolves pulse orientation for one deterministic pulse slot.

Parameters:
- `pulseSlotIndex` - - Zero-based pulse slot index.

Returns: Horizontal or vertical pulse travel orientation.

### resolvePlaybackGroundGridPulseTrackThickness

`(input: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulseTrackThicknessInput) => number`

Resolves the local track thickness at the pulse position.

Parameters:
- `input` - - Pulse position, path, and scene geometry.

Returns: Thickness of the current line under the pulse.

### resolvePlaybackGroundGridPulseTravelRatio

`(input: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulseTravelRatioInput) => number`

Resolves the pulse travel ratio along its chosen line.

Parameters:
- `input` - - Pulse timing direction and orientation.

Returns: Normalized 0..1 travel ratio along the chosen line.

### resolvePlaybackGroundGridUnitHash

`(seed: number, salt: number) => number`

Resolves a deterministic unit-interval hash from a slot index and salt.

Parameters:
- `seed` - - Slot-local seed value.
- `salt` - - Small integer salt used to pick a stable random stream.

Returns: Stable random value in the range 0..1.

### resolvePlaybackGroundGridVerticalPulsePath

`(verticalPulsePaths: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath[], visibleVerticalPulsePaths: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath[], pulseSlotIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath | null`

Selects one sparse vertical pulse path for the current pulse slot.

Parameters:
- `verticalPulsePaths` - - Full vertical ray paths.
- `visibleVerticalPulsePaths` - - Visible subset preferred for on-screen pulses.
- `pulseSlotIndex` - - Zero-based pulse slot index.

Returns: Vertical pulse path, or null when none are available.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.geometry.utils.ts

### appendPlaybackGroundGridVerticalLineSegments

`(targetSegments: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[], startIndex: number, input: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalRayInput) => number`

Appends tapered style segments for one perspective ray.

Parameters:
- `targetSegments` - - Target line-segment buffer.
- `startIndex` - - Current insertion index within the target buffer.
- `input` - - Geometry and depth context for one ray.

Returns: Next insertion index after all ray segments have been written.

### buildPlaybackGroundGridHorizontalGeometry

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridHorizontalGeometry`

Builds the screen-horizontal depth bands for the lower neon plane.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.

Returns: Ordered far-to-near line segments and pulse subsets.

### buildPlaybackGroundGridVerticalGeometry

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, safeLaneSpacingPx: number, quantizedWrappedOffsetPx: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalGeometry`

Builds the perspective rays that converge to the centered horizon point.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `safeLaneSpacingPx` - - Stable lane spacing used for ray anchors.
- `quantizedWrappedOffsetPx` - - Quantized wrapped offset used for cache reuse.

Returns: Wrapped left-to-right perspective rays and pulse subsets.

### groupPlaybackGroundGridSegmentsByStyle

`(segments: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[]) => readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridSegmentBatch[]`

Groups line segments into ordered style batches for lower-overhead drawing.

Parameters:
- `segments` - - Ordered line segments that should preserve draw grouping.

Returns: Ordered style batches that can be stroked with fewer state changes.

### isPlaybackGroundGridVerticalPulsePathVisible

`(pulsePath: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, midTravelRatio: number) => boolean`

Resolves whether one vertical pulse path is safely visible in the viewport.

Parameters:
- `pulsePath` - - Candidate vertical pulse path.
- `sceneContext` - - Current lower-band scene geometry.
- `midTravelRatio` - - Midpoint travel ratio used for visibility checks.

Returns: True when the pulse midpoint stays inside the visible ground band.

### projectPlaybackGroundGridHorizonXToAnchorX

`(input: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridAnchorProjectionInput) => number`

Projects one horizon x-position down to the required floor anchor x-position.

Parameters:
- `input` - - Horizon target and scene geometry.

Returns: Bottom anchor x-position whose ray reaches the target horizon x.

### resolvePlaybackGroundGridAnchorBounds

`(input: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridAnchorBoundsInput) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridAnchorBounds`

Projects the visible horizon span back onto the floor anchor line.

Parameters:
- `input` - - Visible horizon bounds and scene geometry.

Returns: Bottom-anchor bounds required to cover the full visible horizon.

### resolvePlaybackGroundGridBatchPath

`(segments: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[]) => Path2D | null`

Resolves one cached draw-ready path for a grouped segment batch.

Browsers can stroke a reused Path2D more cheaply than replaying dozens of
moveTo/lineTo calls every frame. Test environments may not provide Path2D,
so callers must tolerate a null fallback and replay raw segments instead.

Parameters:
- `segments` - - Ordered line segments that belong to one style batch.

Returns: Cached Path2D when available, otherwise null.

### resolvePlaybackGroundGridHorizontalGeometry

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridHorizontalGeometry`

Resolves cached screen-horizontal depth bands for the lower neon plane.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.

Returns: Ordered far-to-near line segments and pulse subsets.

### resolvePlaybackGroundGridPreferredHorizontalPulsePaths

`(horizontalLines: readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[]) => readonly import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridPulsePath[]`

Prefers the nearer, thicker horizontal tracks when picking a pulse lane.

Parameters:
- `horizontalLines` - - Visible horizontal grid bands.

Returns: Pulse-eligible horizontal paths biased toward the foreground.

### resolvePlaybackGroundGridVerticalCycleContext

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, scrollBasePx: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalCycleContext`

Resolves the wrapped vertical-geometry cycle for the current scroll value.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Quantized wrapped offset and safe lane spacing for cache lookups.

### resolvePlaybackGroundGridVerticalGeometry

`(sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, scrollBasePx: number) => import("test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridVerticalGeometry`

Resolves cached perspective rays that converge to the centered horizon point.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Wrapped left-to-right perspective rays and pulse subsets.
