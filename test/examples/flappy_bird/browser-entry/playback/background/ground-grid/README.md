# browser-entry/playback/background/ground-grid

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.types.ts

### PlaybackBackgroundGroundGridRequest

Narrow request required to render the playback ground grid.

The ground grid only needs time and world-scroll context. That separation is
deliberate: the lower-band renderer should respond to camera motion like a
piece of scenic lighting, not reach into gameplay entities or playback HUD
state.

### PlaybackGroundGridHorizontalGeometryFactory

```ts
PlaybackGroundGridHorizontalGeometryFactory(): PlaybackGroundGridHorizontalGeometry
```

Lazy builder used when one horizontal geometry cache entry is missing.

Horizontal depth bands depend only on stable scene dimensions, so callers can
defer their construction until a cache miss proves the work is actually
needed.

### PlaybackGroundGridVerticalGeometryFactory

```ts
PlaybackGroundGridVerticalGeometryFactory(): PlaybackGroundGridVerticalGeometry
```

Lazy builder used when one vertical geometry cache entry is missing.

Vertical rays additionally depend on wrapped scroll state, so the cache keeps
a compact builder hook rather than eagerly storing every possible cycle.

### PlaybackGroundGridVerticalSceneMetricsFactory

```ts
PlaybackGroundGridVerticalSceneMetricsFactory(): PlaybackGroundGridVerticalSceneMetrics
```

Lazy builder used when one vertical scene-metrics cache entry is missing.

Scene metrics such as projected anchor bounds are pure functions of the
viewport, which makes them ideal cache inputs for the performance-sensitive
lower-band renderer.

### PlaybackBackgroundGroundGridSceneContext

Immutable scene context resolved for one lower-band ground-grid pass.

This is the adapted subset of the shared background scene that the ground
grid cares about: horizon position, lower-band bounds, and the centered
vanishing point that gives the grid its forced-perspective look.

### PlaybackBackgroundGroundGridStyle

Theme-owned style contract for the neon ground grid.

The ground grid uses three coordinated colors: the structural line work, a
fog wash that softens the lower band, and small pulse markers that briefly
travel along eligible tracks.

### PlaybackGroundGridPulseOrientation

Travel orientation used by lightweight pulse overlays.

Horizontal pulses skim along depth bands, while vertical pulses ride the
perspective rays toward or away from the horizon.

### PlaybackBackgroundGroundGridResolvedScene

Geometry and style package resolved before drawing the ground grid.

Splitting scene resolution from drawing keeps canvas code simple: by the time
the renderer runs, every geometric and palette decision has already been made
and packed into one immutable object graph.

### PlaybackGroundGridLineSegment

Declarative line segment model used by the ground-grid renderer.

The grid is built as plain data first so batch builders can group segments by
shared style before any canvas path or glow work happens.

### PlaybackGroundGridSegmentBatch

Ordered batch of line segments that share one render style.

Grouping segments by alpha, blur, and thickness reduces canvas state churn
and gives the renderer a natural place to cache `Path2D` instances when the
environment supports them.

### PlaybackGroundGridVerticalRayInput

Internal helper contract used while generating vertical-ray sub-segments.

Each vertical ray is split into depth-aware pieces so stroke thickness and
glow can evolve as the ray approaches the viewer instead of staying uniform.

### PlaybackGroundGridPulsePath

Simplified path used by one visible pulse event.

Pulses do not need the full batch geometry; they only need a single lane to
travel along, represented here as a start-end segment plus local thickness.

### PlaybackGroundGridPulse

One visible pulse square rendered above the grid lines.

These pulses are tiny accent lights, not gameplay markers. Their job is to
give the lower band a hint of moving circuitry without competing with birds,
pipes, or the active network HUD.

### PlaybackGroundGridGeometry

Pure geometry bundle generated before canvas drawing begins.

Horizontal bands, vertical rays, and the optional pulse are resolved into one
package so the draw layer can stay strictly about paint order.

### PlaybackGroundGridHorizontalGeometry

Cached horizontal geometry bundle reused across matching scene sizes.

Horizontal depth lines are stable for a given viewport, which makes them the
cheapest part of the grid to cache aggressively.

### PlaybackGroundGridVerticalGeometry

Cached vertical geometry bundle reused across one wrapped scroll cycle.

Vertical rays move with scroll, but only within a repeating wrapped cycle.
Caching at that granularity captures most of the reuse without pretending the
rays are globally static.

### PlaybackGroundGridVerticalSceneMetrics

Cached scene metrics reused across matching vertical-grid frames.

These metrics answer the expensive geometric questions once per viewport,
such as how wide the visible anchor span is and how many perspective lanes
can fit while preserving the intended spacing.

### PlaybackGroundGridAnchorBounds

Visible horizon bounds projected onto the bottom anchor line.

Perspective rays begin conceptually at the horizon and terminate on the floor
anchor line, so this structure captures the visible lane span after the
horizon segment has been projected downward.

### PlaybackGroundGridAnchorBoundsInput

Input used when projecting a visible horizon span onto anchor space.

### PlaybackGroundGridAnchorProjectionInput

Input used when projecting one horizon x-position onto the floor anchor line.

### PlaybackGroundGridVerticalCycleContext

Wrapped vertical-cycle state derived from scroll for one frame.

The perspective rays repeat on a fixed cycle. Wrapping the scroll state lets
the renderer reuse cached geometry while still appearing to drift sideways.

### PlaybackGroundGridPulseInput

Input contract used while resolving one deterministic pulse event.

Pulses are deterministic decoration: given the same frame and scene, the same
lane should light up. This input bundle gathers the candidate paths needed to
make that choice without consulting external state.

### PlaybackGroundGridPulseTimingState

Timing state resolved for one deterministic ground-grid pulse slot.

The slot model keeps pulse timing legible: each pulse has a start bucket, an
elapsed time inside that bucket, and a normalized progress value used by the
position and fade helpers.

### PlaybackGroundGridPulseTravelRatioInput

Direction and timing state used when resolving pulse travel progress.

Some pulse lanes feel better moving away from the viewer and others toward
it, so travel resolution keeps orientation and forward/reverse intent paired
with the current lifetime progress.

### PlaybackGroundGridPulseTrackThicknessInput

Input used when adapting a pulse position into a local track thickness.

Because the grid uses perspective-weighted stroke widths, the pulse size must
be adjusted to the local lane thickness rather than using one fixed square.

### PlaybackGroundGridVerticalPulseContinuationState

Cached continuation state used to keep one vertical pulse on the same ray.

Without this continuity state a vertical pulse could jitter between adjacent
rays across frames, which looks like noise instead of a deliberate light.

### PlaybackBackgroundGroundGridSourceScene

Helper alias used when adapting the shared background scene context.

The parent background module owns the full sky-plus-ground scene contract;
the grid renderer narrows that shape to only the fields needed for the lower
band so the dependency direction stays clear.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.ts

### renderPlaybackBackgroundGroundGrid

```ts
renderPlaybackBackgroundGroundGrid(
  context: CanvasRenderingContext2D,
  sourceScene: PlaybackBackgroundGroundGridSourceScene,
  request: PlaybackBackgroundGroundGridRequest,
): void
```

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

### drawPlaybackGroundGrid

```ts
drawPlaybackGroundGrid(
  context: CanvasRenderingContext2D,
  resolvedScene: PlaybackBackgroundGroundGridResolvedScene,
  geometry: PlaybackGroundGridGeometry,
): void
```

Draws the resolved neon ground grid inside the lower background band.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.
- `geometry` - - Precomputed horizontal and vertical line segments.

Returns: Nothing.

### drawGroundGridSegmentBatch

```ts
drawGroundGridSegmentBatch(
  context: CanvasRenderingContext2D,
  batch: PlaybackGroundGridSegmentBatch,
): void
```

Draws one batch of neon line segments that share one render style.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batch` - - Ordered line-segment batch that shares one render style.

Returns: Nothing.

### drawGroundGridSegmentBatches

```ts
drawGroundGridSegmentBatches(
  context: CanvasRenderingContext2D,
  batches: readonly PlaybackGroundGridSegmentBatch[],
  lineColor: string,
): void
```

Draws one ordered collection of neon segment batches.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batches` - - Ordered line-segment batches to render.
- `lineColor` - - Core neon stroke color.

Returns: Nothing.

### drawGroundGridFog

```ts
drawGroundGridFog(
  context: CanvasRenderingContext2D,
  resolvedScene: PlaybackBackgroundGroundGridResolvedScene,
): void
```

Draws the lower-band atmospheric wash behind the neon line work.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.

Returns: Nothing.

### drawGroundGridPulse

```ts
drawGroundGridPulse(
  context: CanvasRenderingContext2D,
  pulse: PlaybackGroundGridPulse | null,
  fillColor: string,
): void
```

Draws one pulse square above the grid lines and below gameplay entities.

Parameters:
- `context` - - Canvas 2D drawing context.
- `pulse` - - Visible pulse square for the current frame.
- `fillColor` - - Core neon fill color.

Returns: Nothing.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.constants.ts

### FLAPPY_GROUND_GRID_HORIZONTAL_LINE_COUNT

Number of horizontal depth bands used by the neon ground grid.

More bands increase the sense of depth, but they also thicken the lower band
visually and add more line work to each frame.

### FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX

Target visible spacing between adjacent vertical rays at the pipe floor.

The lower pipes visually attach to a projected floor band in the grid. The
important invariant is phase repeat, not just raw gap width: each new pipe at
max difficulty should land on the same relative grid position as the previous
one. That requires matching the full pipe-to-pipe pitch, not only the open
edge-to-edge gap between pipe bodies.

### FLAPPY_GROUND_GRID_MIN_VERTICAL_LINE_COUNT

Minimum visible perspective-ray count used on very narrow viewports.

Even on a small canvas, the grid still needs at least a few rays to read as
perspective instead of a flat block of horizontal stripes.

### FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT

Extra off-screen perspective rays drawn for seamless wrap.

Overflow rays prevent the parallax cycle from exposing empty gaps when the
wrapped scroll offset lands near a lane boundary.

### FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX

Target screen-space height for one vertical-ray style segment (pixels).

Segmenting the rays lets the renderer vary alpha, thickness, and blur by
depth rather than drawing each ray with one flat style.

### FLAPPY_GROUND_GRID_APPROX_FRAME_DURATION_MS

Approximate playback frame duration used for deterministic pulse timing.

The pulse system is designed to feel stable at ordinary browser animation
cadence without requiring access to wall-clock time in every helper.

### FLAPPY_GROUND_GRID_PULSE_INTERVAL_MS

Interval between visible pulse events (milliseconds).

A relatively slow cadence keeps the pulses as occasional accent lights rather
than a constant distraction under the birds.

### FLAPPY_GROUND_GRID_PULSE_LIFETIME_MS

Lifetime of one pulse as it travels across its chosen line (milliseconds).

The lifetime is slightly shorter than the full interval so one pulse fades
out before the next slot becomes active.

### FLAPPY_GROUND_GRID_PULSE_ALPHA

Peak opacity used by visible pulse squares.

### FLAPPY_GROUND_GRID_PULSE_MIN_SIZE_PX

Smallest visible pulse square size (pixels).

### FLAPPY_GROUND_GRID_PULSE_MAX_SIZE_PX

Largest visible pulse square size (pixels).

### FLAPPY_GROUND_GRID_VERTICAL_PULSE_START_RATIO

Earliest progress ratio allowed for vertical pulse travel.

Vertical pulses start a little away from the horizon so they are visible as
distinct squares instead of immediately disappearing into compressed depth.

### FLAPPY_GROUND_GRID_VERTICAL_PULSE_END_RATIO

Latest progress ratio allowed for vertical pulse travel.

Ending early keeps the pulse out of the extreme foreground, where its square
would become too large and visually heavy.

### FLAPPY_GROUND_GRID_PULSE_MIN_ELIGIBLE_THICKNESS_PX

Minimum horizontal line thickness eligible for pulse travel.

### FLAPPY_GROUND_GRID_PULSE_PREFERRED_HORIZONTAL_START_RATIO

Earliest eligible slice of horizontal lines used for visible pulse picks.

This biases horizontal pulses toward the more legible near-midground instead
of the compressed lines nearest the horizon.

### FLAPPY_GROUND_GRID_PULSE_VISIBLE_VIEWPORT_INSET_PX

Horizontal inset that keeps vertical pulse picks away from clipped edges.

### FLAPPY_GROUND_GRID_UNSIGNED_NORMALIZATION_DIVISOR

Normalization divisor used for deterministic pulse hash generation.

The pulse selection helpers convert unsigned integer hashes into stable
floating-point picks in the unit interval.

### FLAPPY_GROUND_GRID_DEPTH_CURVE_EXPONENT

Non-linear exponent used to compress depth lines toward the horizon.

This is the main perspective stylization control: higher values bunch more of
the depth bands near the horizon and leave broader spacing near the viewer.

### FLAPPY_GROUND_GRID_PIPE_CONNECTION_LINE_OFFSET_FROM_BOTTOM

Near-edge horizontal line offset used for the lower-pipe floor illusion.

`1` targets the first usable grid band above the bottom edge rather than the
terminal line that coincides with the lower-band boundary itself.

### FLAPPY_GROUND_GRID_SCROLL_RATIO

Scroll ratio applied to the moving vertical perspective rays.

Keeping the rays slower than gameplay motion makes the grid feel like a deep
environmental layer rather than a surface glued to the pipes.

### FLAPPY_GROUND_GRID_SCROLL_OFFSET_QUANTIZATION_DECIMALS

Decimal precision used when quantizing the wrapped vertical-ray offset.

Quantization stabilizes cache reuse by preventing tiny floating-point drift
from generating effectively identical geometry variants.

### FLAPPY_GROUND_GRID_MIN_ALPHA

Minimum alpha used by the farthest horizontal depth lines.

### FLAPPY_GROUND_GRID_MAX_ALPHA

Maximum alpha used by the nearest depth and perspective lines.

### FLAPPY_GROUND_GRID_MAX_BLUR_PX

Blur radius used by the farthest depth lines near the horizon.

### FLAPPY_GROUND_GRID_MIN_BLUR_PX

Blur radius used by the nearest depth lines at the bottom edge.

### FLAPPY_GROUND_GRID_MIN_THICKNESS_PX

Minimum stroke width used by far depth lines.

### FLAPPY_GROUND_GRID_MAX_THICKNESS_PX

Maximum stroke width used by near depth lines.

### FLAPPY_GROUND_GRID_FOG_HEIGHT_RATIO

Height ratio reserved for the subtle lower-band neon fog wash.

The fog sits in the lower portion of the band so it enriches the foreground
without muting the crisp horizon seam.

### FLAPPY_GROUND_GRID_FOG_ALPHA

Peak opacity used by the lower-band neon fog wash.

The fog should tint the band, not obscure the line geometry, so the opacity
is intentionally modest.

### FLAPPY_BACKGROUND_GROUND_GRID_STYLE

Frozen neon style bundle reused by the playback ground-grid renderer.

The values stay theme-owned but are materialized once so the renderer does
not allocate a new style object during every frame.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.batch.services.ts

### drawGroundGridSegmentBatches

```ts
drawGroundGridSegmentBatches(
  context: CanvasRenderingContext2D,
  batches: readonly PlaybackGroundGridSegmentBatch[],
  lineColor: string,
): void
```

Draws one ordered collection of neon segment batches.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batches` - - Ordered line-segment batches to render.
- `lineColor` - - Core neon stroke color.

Returns: Nothing.

### drawGroundGridSegmentBatch

```ts
drawGroundGridSegmentBatch(
  context: CanvasRenderingContext2D,
  batch: PlaybackGroundGridSegmentBatch,
): void
```

Draws one batch of neon line segments that share one render style.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batch` - - Ordered line-segment batch that shares one render style.

Returns: Nothing.

### strokePlaybackGroundGridBatch

```ts
strokePlaybackGroundGridBatch(
  context: CanvasRenderingContext2D,
  batch: PlaybackGroundGridSegmentBatch,
): void
```

Strokes one ground-grid batch using a cached path when the environment supports it.

Parameters:
- `context` - - Canvas 2D drawing context.
- `batch` - - Ordered line-segment batch that shares one render style.

Returns: Nothing.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.cache.services.ts

### ensureGroundGridViewportCacheValidity

```ts
ensureGroundGridViewportCacheValidity(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): string
```

Ensures ground-grid caches only retain entries for the current viewport size.

The ground grid is derived from viewport width and total scene height, so a
page resize invalidates every cached geometry variant and fog gradient.

Parameters:
- `sceneContext` - - Current lower-band scene geometry.

Returns: Stable viewport-size cache key for the current frame.

### resolveGroundGridViewportCacheKey

```ts
resolveGroundGridViewportCacheKey(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): string
```

Resolves the viewport-size cache key used by the ground-grid caches.

Parameters:
- `sceneContext` - - Current lower-band scene geometry.

Returns: Cache key that changes whenever the page size changes.

### resolveGroundGridSceneCacheKey

```ts
resolveGroundGridSceneCacheKey(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): string
```

Resolves the stable local-scene cache key for ground-grid geometry.

Parameters:
- `sceneContext` - - Current lower-band scene geometry.

Returns: Scene key suitable for static horizontal and vertical cache entries.

### resolveGroundGridVerticalCycleCacheKey

```ts
resolveGroundGridVerticalCycleCacheKey(
  sceneCacheKey: string,
  wrappedOffsetPx: number,
): string
```

Resolves the cache key for one wrapped vertical-geometry cycle.

Parameters:
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `wrappedOffsetPx` - - Wrapped offset within one lane cycle.

Returns: Cycle key used for vertical geometry reuse.

### resolveCachedGroundGridHorizontalGeometry

```ts
resolveCachedGroundGridHorizontalGeometry(
  sceneCacheKey: string,
  factory: PlaybackGroundGridHorizontalGeometryFactory,
): PlaybackGroundGridHorizontalGeometry
```

Resolves cached horizontal geometry for one scene.

Parameters:
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `factory` - - Lazy geometry builder used when the cache misses.

Returns: Cached horizontal geometry bundle for the scene.

### resolveCachedGroundGridVerticalSceneMetrics

```ts
resolveCachedGroundGridVerticalSceneMetrics(
  sceneCacheKey: string,
  factory: PlaybackGroundGridVerticalSceneMetricsFactory,
): PlaybackGroundGridVerticalSceneMetrics
```

Resolves cached scene metrics for one vertical-grid layout.

Parameters:
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `factory` - - Lazy scene-metrics builder used when the cache misses.

Returns: Cached vertical scene metrics for the scene.

### resolveCachedGroundGridVerticalGeometry

```ts
resolveCachedGroundGridVerticalGeometry(
  cycleCacheKey: string,
  factory: PlaybackGroundGridVerticalGeometryFactory,
): PlaybackGroundGridVerticalGeometry
```

Resolves cached vertical geometry for one scene and wrapped offset cycle.

Parameters:
- `cycleCacheKey` - - Scene-and-offset cache key for the active frame.
- `factory` - - Lazy geometry builder used when the cache misses.

Returns: Cached vertical geometry bundle for the cycle.

### resolveCachedGroundGridFogGradient

```ts
resolveCachedGroundGridFogGradient(
  context: CanvasRenderingContext2D,
  sceneCacheKey: string,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  fogColor: string,
): CanvasGradient
```

Resolves a cached fog gradient for one canvas and local scene.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneCacheKey` - - Stable scene key for the active viewport.
- `sceneContext` - - Current lower-band scene geometry.
- `fogColor` - - Theme-owned fog color token.

Returns: Cached fog gradient aligned to the lower-band scene.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.layer.services.ts

### drawGroundGridFog

```ts
drawGroundGridFog(
  context: CanvasRenderingContext2D,
  resolvedScene: PlaybackBackgroundGroundGridResolvedScene,
): void
```

Draws the lower-band atmospheric wash behind the neon line work.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.

Returns: Nothing.

### drawGroundGridPulse

```ts
drawGroundGridPulse(
  context: CanvasRenderingContext2D,
  pulse: PlaybackGroundGridPulse | null,
  fillColor: string,
): void
```

Draws one pulse square above the grid lines and below gameplay entities.

Parameters:
- `context` - - Canvas 2D drawing context.
- `pulse` - - Visible pulse square for the current frame.
- `fillColor` - - Core neon fill color.

Returns: Nothing.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.scene.services.ts

### resolvePlaybackGroundGridSceneContext

```ts
resolvePlaybackGroundGridSceneContext(
  sceneContext: PlaybackBackgroundGroundGridSourceScene,
): PlaybackBackgroundGroundGridSceneContext
```

Resolves the shared scene context used by the ground-grid renderer.

Parameters:
- `sceneContext` - - Lower-band geometry provided by the background module.

Returns: Narrow scene contract consumed by grid-specific helpers.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.geometry.services.ts

### resolvePlaybackGroundGridGeometry

```ts
resolvePlaybackGroundGridGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  frameIndex: number,
  scrollBasePx: number,
): PlaybackGroundGridGeometry
```

Builds the line geometry for the neon ground grid.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `frameIndex` - - Current deterministic playback frame index.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Horizontal depth bands and perspective rays for the current frame.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.utils.ts

### interpolatePlaybackGroundGridPoint

```ts
interpolatePlaybackGroundGridPoint(
  startXPx: number,
  startYPx: number,
  endXPx: number,
  endYPx: number,
  interpolationRatio: number,
): PlaybackGroundGridPoint
```

Interpolates one point along a perspective ray.

Parameters:
- `startXPx` - - Bottom anchor x-position.
- `startYPx` - - Bottom anchor y-position.
- `endXPx` - - Vanishing-point x-position.
- `endYPx` - - Vanishing-point y-position.
- `interpolationRatio` - - Normalized 0..1 position along the ray.

Returns: Interpolated point on the perspective ray.

### resolvePlaybackGroundGridDepthCurve

```ts
resolvePlaybackGroundGridDepthCurve(
  depthRatio: number,
): number
```

Maps a normalized depth ratio into a stronger synthwave spacing curve.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Curved depth ratio used for line placement and styling.

### resolvePlaybackGroundGridDepthFromHorizonDistance

```ts
resolvePlaybackGroundGridDepthFromHorizonDistance(
  distanceToHorizonPx: number,
  maximumDistanceToHorizonPx: number,
): number
```

Resolves normalized depth from a vertical distance away from the horizon.

Parameters:
- `distanceToHorizonPx` - - Vertical distance from the vanishing horizon.
- `maximumDistanceToHorizonPx` - - Largest visible vertical horizon distance.

Returns: Normalized 0..1 depth where 0 is at the horizon and 1 is nearest.

### resolvePlaybackGroundGridLineAlpha

```ts
resolvePlaybackGroundGridLineAlpha(
  depthRatio: number,
): number
```

Resolves neon alpha for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Opacity for the rendered line.

### resolvePlaybackGroundGridLineBlur

```ts
resolvePlaybackGroundGridLineBlur(
  depthRatio: number,
): number
```

Resolves glow blur for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Blur radius for the rendered line.

### resolvePlaybackGroundGridLineThickness

```ts
resolvePlaybackGroundGridLineThickness(
  depthRatio: number,
): number
```

Resolves stroke thickness for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Stroke width in pixels.

### resolvePlaybackGroundGridGeometry

```ts
resolvePlaybackGroundGridGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  frameIndex: number,
  scrollBasePx: number,
): PlaybackGroundGridGeometry
```

Builds the line geometry for the neon ground grid.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `frameIndex` - - Current deterministic playback frame index.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Horizontal depth bands and perspective rays for the current frame.

### resolvePlaybackGroundGridSceneContext

```ts
resolvePlaybackGroundGridSceneContext(
  sceneContext: PlaybackBackgroundGroundGridSourceScene,
): PlaybackBackgroundGroundGridSceneContext
```

Resolves the shared scene context used by the ground-grid renderer.

Parameters:
- `sceneContext` - - Lower-band geometry provided by the background module.

Returns: Narrow scene contract consumed by grid-specific helpers.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.math.utils.ts

### resolvePlaybackGroundGridDepthCurve

```ts
resolvePlaybackGroundGridDepthCurve(
  depthRatio: number,
): number
```

Maps a normalized depth ratio into a stronger synthwave spacing curve.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Curved depth ratio used for line placement and styling.

### resolvePlaybackGroundGridLineAlpha

```ts
resolvePlaybackGroundGridLineAlpha(
  depthRatio: number,
): number
```

Resolves neon alpha for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Opacity for the rendered line.

### resolvePlaybackGroundGridLineBlur

```ts
resolvePlaybackGroundGridLineBlur(
  depthRatio: number,
): number
```

Resolves glow blur for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Blur radius for the rendered line.

### resolvePlaybackGroundGridLineThickness

```ts
resolvePlaybackGroundGridLineThickness(
  depthRatio: number,
): number
```

Resolves stroke thickness for one line based on its normalized depth.

Parameters:
- `depthRatio` - - Normalized 0..1 depth where 0 is far and 1 is near.

Returns: Stroke width in pixels.

### resolvePlaybackGroundGridDepthFromHorizonDistance

```ts
resolvePlaybackGroundGridDepthFromHorizonDistance(
  distanceToHorizonPx: number,
  maximumDistanceToHorizonPx: number,
): number
```

Resolves normalized depth from a vertical distance away from the horizon.

Parameters:
- `distanceToHorizonPx` - - Vertical distance from the vanishing horizon.
- `maximumDistanceToHorizonPx` - - Largest visible vertical horizon distance.

Returns: Normalized 0..1 depth where 0 is at the horizon and 1 is nearest.

### resolvePlaybackGroundGridPipeConnectionProfile

```ts
resolvePlaybackGroundGridPipeConnectionProfile(
  visibleWorldHeightPx: number,
): PlaybackGroundGridPipeConnectionProfile
```

Resolves the shared lower-pipe floor and matched grid-ray scroll ratio.

The lower pipe is visually clipped to the first usable horizontal grid band
above the bottom edge. The returned scroll ratio then speeds up the moving
perspective rays so their lateral motion matches the pipe speed exactly at
that same projected height.

Parameters:
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Pipe-floor y-position plus the matching vertical-ray scroll ratio.

### interpolatePlaybackGroundGridPoint

```ts
interpolatePlaybackGroundGridPoint(
  startXPx: number,
  startYPx: number,
  endXPx: number,
  endYPx: number,
  interpolationRatio: number,
): PlaybackGroundGridPoint
```

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

### PlaybackGroundGridPipeConnectionProfile

Shared pipe-floor projection resolved from the lower ground-grid geometry.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.pulse.utils.ts

### resolvePlaybackGroundGridPulse

```ts
resolvePlaybackGroundGridPulse(
  input: PlaybackGroundGridPulseInput,
): PlaybackGroundGridPulse | null
```

Resolves one rare, deterministic pulse square for the current frame.

Parameters:
- `input` - - Current frame timing and visible pulse path candidates.

Returns: Visible pulse square, or null when the current slot is inactive.

### resolvePlaybackGroundGridPulseTrackThickness

```ts
resolvePlaybackGroundGridPulseTrackThickness(
  input: PlaybackGroundGridPulseTrackThicknessInput,
): number
```

Resolves the local track thickness at the pulse position.

Parameters:
- `input` - - Pulse position, path, and scene geometry.

Returns: Thickness of the current line under the pulse.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.geometry.utils.ts

### resolvePlaybackGroundGridHorizontalGeometry

```ts
resolvePlaybackGroundGridHorizontalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridHorizontalGeometry
```

Resolves cached screen-horizontal depth bands for the lower neon plane.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.

Returns: Ordered far-to-near line segments and pulse subsets.

### resolvePlaybackGroundGridVerticalGeometry

```ts
resolvePlaybackGroundGridVerticalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  scrollBasePx: number,
): PlaybackGroundGridVerticalGeometry
```

Resolves cached perspective rays that converge to the centered horizon point.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Wrapped left-to-right perspective rays and pulse subsets.

### buildPlaybackGroundGridHorizontalGeometry

```ts
buildPlaybackGroundGridHorizontalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridHorizontalGeometry
```

Builds the screen-horizontal depth bands for the lower neon plane.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.

Returns: Ordered far-to-near line segments and pulse subsets.

### buildPlaybackGroundGridVerticalGeometry

```ts
buildPlaybackGroundGridVerticalGeometry(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
  verticalSceneMetrics: PlaybackGroundGridVerticalSceneMetrics,
  wrappedOffsetPx: number,
): PlaybackGroundGridVerticalGeometry
```

Builds the perspective rays that converge to the centered horizon point.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `safeLaneSpacingPx` - - Stable lane spacing used for ray anchors.
- `wrappedOffsetPx` - - Wrapped offset used for cache reuse and ray placement.

Returns: Wrapped left-to-right perspective rays and pulse subsets.

### appendPlaybackGroundGridVerticalLineSegments

```ts
appendPlaybackGroundGridVerticalLineSegments(
  targetSegments: PlaybackGroundGridLineSegment[],
  startIndex: number,
  input: PlaybackGroundGridVerticalRayInput,
): number
```

Appends tapered style segments for one perspective ray.

Parameters:
- `targetSegments` - - Target line-segment buffer.
- `startIndex` - - Current insertion index within the target buffer.
- `input` - - Geometry and depth context for one ray.

Returns: Next insertion index after all ray segments have been written.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.pulse.timing.utils.ts

### resolvePlaybackGroundGridPulseTiming

```ts
resolvePlaybackGroundGridPulseTiming(
  frameIndex: number,
): PlaybackGroundGridPulseTimingState | null
```

Resolves timing state for the currently active deterministic pulse slot.

Parameters:
- `frameIndex` - - Current deterministic playback frame index.

Returns: Pulse timing state, or null when no pulse is active in this frame.

### resolvePlaybackGroundGridPulseOrientation

```ts
resolvePlaybackGroundGridPulseOrientation(
  pulseSlotIndex: number,
): PlaybackGroundGridPulseOrientation
```

Resolves pulse orientation for one deterministic pulse slot.

Parameters:
- `pulseSlotIndex` - - Zero-based pulse slot index.

Returns: Horizontal or vertical pulse travel orientation.

### resolvePlaybackGroundGridHorizontalPulsePath

```ts
resolvePlaybackGroundGridHorizontalPulsePath(
  horizontalPulsePaths: readonly PlaybackGroundGridPulsePath[],
  pulseSlotIndex: number,
): PlaybackGroundGridPulsePath | null
```

Selects one thick-enough horizontal band for the current pulse slot.

Parameters:
- `horizontalPulsePaths` - - Cached horizontal pulse paths eligible for travel.
- `pulseSlotIndex` - - Zero-based pulse slot index.

Returns: Horizontal pulse path, or null when none are suitable.

### resolvePlaybackGroundGridPulseTravelRatio

```ts
resolvePlaybackGroundGridPulseTravelRatio(
  input: PlaybackGroundGridPulseTravelRatioInput,
): number
```

Resolves the pulse travel ratio along its chosen line.

Parameters:
- `input` - - Pulse timing direction and orientation.

Returns: Normalized 0..1 travel ratio along the chosen line.

### resolvePlaybackGroundGridUnitHash

```ts
resolvePlaybackGroundGridUnitHash(
  seed: number,
  salt: number,
): number
```

Resolves a deterministic unit-interval hash from a slot index and salt.

Parameters:
- `seed` - - Slot-local seed value.
- `salt` - - Small integer salt used to pick a stable random stream.

Returns: Stable random value in the range 0..1.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.geometry.batch.utils.ts

### groupPlaybackGroundGridSegmentsByStyle

```ts
groupPlaybackGroundGridSegmentsByStyle(
  segments: readonly PlaybackGroundGridLineSegment[],
): readonly PlaybackGroundGridSegmentBatch[]
```

Groups line segments into ordered style batches for lower-overhead drawing.

Parameters:
- `segments` - - Ordered line segments that should preserve draw grouping.

Returns: Ordered style batches that can be stroked with fewer state changes.

### resolvePlaybackGroundGridPreferredHorizontalPulsePaths

```ts
resolvePlaybackGroundGridPreferredHorizontalPulsePaths(
  horizontalLines: readonly PlaybackGroundGridLineSegment[],
): readonly PlaybackGroundGridPulsePath[]
```

Prefers the nearer, thicker horizontal tracks when picking a pulse lane.

Parameters:
- `horizontalLines` - - Visible horizontal grid bands.

Returns: Pulse-eligible horizontal paths biased toward the foreground.

### resolvePlaybackGroundGridBatchPath

```ts
resolvePlaybackGroundGridBatchPath(
  segments: readonly PlaybackGroundGridLineSegment[],
): Path2D | null
```

Resolves one cached draw-ready path for a grouped segment batch.

Parameters:
- `segments` - - Ordered line segments that belong to one style batch.

Returns: Cached Path2D when available, otherwise null.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.geometry.layout.utils.ts

### buildPlaybackGroundGridVerticalSceneMetrics

```ts
buildPlaybackGroundGridVerticalSceneMetrics(
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): PlaybackGroundGridVerticalSceneMetrics
```

Builds the static scene metrics reused across one viewport-sized grid layout.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.

Returns: Stable anchor bounds and lane spacing for vertical-ray reuse.

### resolvePlaybackGroundGridVerticalCycleContext

```ts
resolvePlaybackGroundGridVerticalCycleContext(
  safeLaneSpacingPx: number,
  lowerBandBottomYPx: number,
  scrollBasePx: number,
): PlaybackGroundGridVerticalCycleContext
```

Resolves the wrapped vertical-geometry cycle for the current scroll value.

Parameters:
- `safeLaneSpacingPx` - - Stable lane spacing used by current viewport metrics.
- `lowerBandBottomYPx` - - Lower edge of the visible ground-grid band.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Quantized wrapped offset and safe lane spacing for cache lookups.

### resolvePlaybackGroundGridAnchorBounds

```ts
resolvePlaybackGroundGridAnchorBounds(
  input: PlaybackGroundGridAnchorBoundsInput,
): PlaybackGroundGridAnchorBounds
```

Projects one visible horizontal span back onto the floor anchor line.

Parameters:
- `input` - - Visible span bounds and scene geometry.

Returns: Bottom-anchor bounds required to cover the chosen projected span.

### isPlaybackGroundGridVerticalPulsePathVisible

```ts
isPlaybackGroundGridVerticalPulsePathVisible(
  pulsePath: PlaybackGroundGridPulsePath,
  sceneContext: PlaybackBackgroundGroundGridSceneContext,
): boolean
```

Resolves whether one vertical pulse path is safely visible in the viewport.

Parameters:
- `pulsePath` - - Candidate vertical pulse path.
- `sceneContext` - - Current lower-band scene geometry.

Returns: True when the pulse midpoint stays inside the visible ground band.

### projectPlaybackGroundGridProjectedXToAnchorX

```ts
projectPlaybackGroundGridProjectedXToAnchorX(
  input: { projectedXPx: number; projectedYPx: number; sceneContext: PlaybackBackgroundGroundGridSceneContext; },
): number
```

Projects one visible x-position at an arbitrary y-level to the anchor line.

Parameters:
- `input` - - Projected target and scene geometry.

Returns: Bottom anchor x-position whose ray reaches the projected point.

### resolvePlaybackGroundGridRetainedWidthRatioAtYPx

```ts
resolvePlaybackGroundGridRetainedWidthRatioAtYPx(
  lowerBandBottomYPx: number,
  vanishingPointYPx: number,
  targetYPx: number,
): number
```

Resolves how much adjacent-ray spacing remains at one projected y-position.

Perspective rays linearly collapse toward the vanishing point, so the local
lane width at any y-position is just the bottom-anchor spacing multiplied by
the remaining width ratio between the bottom edge and the vanishing point.

Parameters:
- `lowerBandBottomYPx` - - Bottom edge of the visible ground band.
- `vanishingPointYPx` - - Shared vanishing-point y-position.
- `targetYPx` - - Projected y-position whose retained width should be measured.

Returns: Width-retention ratio in the inclusive `[0, 1]` range.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.pulse.selection.utils.ts

### resolvePlaybackGroundGridVerticalPulseSelection

```ts
resolvePlaybackGroundGridVerticalPulseSelection(
  input: { verticalPulsePaths: readonly PlaybackGroundGridPulsePath[]; visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[]; pulseSlotIndex: number; frameIndex: number; travelProgressRatio: number; resolveUnitHash: (seed: number, salt: number) => number; },
): PlaybackGroundGridPulsePath | null
```

Resolves the current vertical pulse path while preserving per-slot continuity.

A vertical pulse should stay attached to one moving ray for its whole
lifetime, even though the frame-local ray array is rebuilt as scroll wraps.
This helper first prefers the nearest continuation of the previous frame's
pulse position, then falls back to deterministic slot-based selection.

Parameters:
- `verticalPulsePaths` - - Full vertical ray paths for the current frame.
- `visibleVerticalPulsePaths` - - Visible subset preferred for on-screen pulses.
- `pulseSlotIndex` - - Zero-based pulse slot index.
- `frameIndex` - - Current deterministic frame index.
- `travelProgressRatio` - - Current travel ratio along the chosen line.
- `resolveUnitHash` - - Deterministic unit-hash helper used for fallback picks.

Returns: Vertical pulse path, or null when none are available.

### rememberPlaybackGroundGridVerticalPulseSelection

```ts
rememberPlaybackGroundGridVerticalPulseSelection(
  pulseSlotIndex: number,
  continuationState: PlaybackGroundGridVerticalPulseContinuationState,
): void
```

Stores the resolved pulse center for continuation on the next frame.

Parameters:
- `pulseSlotIndex` - - Zero-based pulse slot index.
- `continuationState` - - Latest visible pulse position for the slot.

Returns: Nothing.

### resolveContinuedVerticalPulsePath

```ts
resolveContinuedVerticalPulsePath(
  input: { candidatePulsePaths: readonly PlaybackGroundGridPulsePath[]; pulseSlotIndex: number; frameIndex: number; travelProgressRatio: number; },
): PlaybackGroundGridPulsePath | null
```

Resolves the nearest continued vertical pulse path for an active slot.

Parameters:
- `input` - - Continuation input for the current frame.

Returns: Continued pulse path when one can be matched, otherwise null.

### trimCachedVerticalPulseContinuationState

```ts
trimCachedVerticalPulseContinuationState(
  currentPulseSlotIndex: number,
): void
```

Trims cached continuation state so only the current or previous pulse slots remain.

Parameters:
- `currentPulseSlotIndex` - - Pulse slot currently being resolved.

Returns: Nothing.

### resolveStableVerticalPulsePathCandidates

```ts
resolveStableVerticalPulsePathCandidates(
  verticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
  visibleVerticalPulsePaths: readonly PlaybackGroundGridPulsePath[],
): readonly PlaybackGroundGridPulsePath[]
```

Resolves a stable vertical pulse-candidate set for one frame.

Parameters:
- `verticalPulsePaths` - - Full vertical ray paths for the current frame.
- `visibleVerticalPulsePaths` - - Midpoint-visible subset used as fallback.

Returns: Stable candidate set for deterministic vertical pulse selection.
