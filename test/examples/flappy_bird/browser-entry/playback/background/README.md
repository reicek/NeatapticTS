# browser-entry/playback/background

Layered playback background composition for the browser demo.

This boundary keeps atmosphere separate from gameplay entities. The frame
renderer can ask for one deterministic scenic backdrop while this module owns
the details of sky styling, ground-grid composition, and the glowing seam
that ties both halves together.

## browser-entry/playback/background/playback.background.ts

### renderPlaybackBackground

```ts
renderPlaybackBackground(
  context: CanvasRenderingContext2D,
  request: PlaybackBackgroundRequest,
): void
```

Draws the layered playback background.

This boundary exists to keep atmosphere separate from gameplay entities. The
frame renderer should be able to ask for a complete backdrop in one call
without also absorbing starfield policy, horizon styling, and ground-grid
composition details.

The composition is intentionally chapter-like: sky first, ground second,
horizon seam last. That ordering gives the playback scene a stable visual
identity while keeping the background deterministic and cheap to re-render.

Parameters:
- `context` - - Canvas 2D drawing context.
- `request` - - Narrow render input required for background composition.

Returns: Nothing.

Example:

```ts
renderPlaybackBackground(context, {
  viewportLeftXPx: cameraLeftPx,
  visibleWorldWidthPx: 288,
  visibleWorldHeightPx: 512,
  frameIndex,
  scrollBasePx: frameIndex * pipeSpeedPxPerFrame,
});
```

## browser-entry/playback/background/playback.background.types.ts

Minimal render input required to draw the playback background.

The background is intentionally treated as a deterministic camera effect
rather than a gameplay-aware renderer. By restricting the contract to
viewport geometry, frame index, and scroll position, the module can create a
stable neon sky-ground composition without coupling itself to bird state,
pipe arrays, or trail caches.

Example:

```ts
const request: PlaybackBackgroundRequest = {
  viewportLeftXPx: cameraLeftPx,
  visibleWorldWidthPx: 288,
  visibleWorldHeightPx: 512,
  frameIndex,
  scrollBasePx: frameIndex * pipeSpeedPxPerFrame,
};
```

### PlaybackBackgroundLayout

Resolved vertical scene split used by playback background composition.

The layout fixes the classic synthwave composition used by this demo: a tall
sky band for layered starfield parallax and a compressed lower strip for the
perspective grid. Caching this structure by viewport size keeps redraws cheap
when the scene is otherwise stable.

### PlaybackBackgroundLayoutFactory

```ts
PlaybackBackgroundLayoutFactory(): PlaybackBackgroundLayout
```

Zero-argument builder used to lazily construct one cached background layout.

The cache service accepts a factory instead of raw data so callers can defer
the slightly more expensive layout computation until a viewport-size cache
miss actually occurs.

### PlaybackBackgroundRequest

Minimal render input required to draw the playback background.

The background is intentionally treated as a deterministic camera effect
rather than a gameplay-aware renderer. By restricting the contract to
viewport geometry, frame index, and scroll position, the module can create a
stable neon sky-ground composition without coupling itself to bird state,
pipe arrays, or trail caches.

Example:

```ts
const request: PlaybackBackgroundRequest = {
  viewportLeftXPx: cameraLeftPx,
  visibleWorldWidthPx: 288,
  visibleWorldHeightPx: 512,
  frameIndex,
  scrollBasePx: frameIndex * pipeSpeedPxPerFrame,
};
```

### PlaybackBackgroundSceneContext

Derived scene contract shared by the playback background render passes.

This is the background module's precomputed staging area. The scene service
resolves the sky/lower-band split, vanishing point, and horizon styling once
so the draw passes can stay orchestration-first and avoid repeating geometry
math every frame.

### PlaybackHorizonLineRequest

Draw request for the horizon divider line.

This narrow contract is the final handoff from layout math to the canvas
stroke helper: world-space x extents, the pixel-snapped y position, and the
resolved glow style needed for both line passes.

### PlaybackHorizonStyle

Neon styling contract for the horizon divider line.

The horizon is rendered as both a crisp divider and a glow source, much like
the luminous skyline separator common in synthwave and TRON-inspired poster
art. Keeping those paint properties bundled makes it easier to reason about
the horizon as one semantic effect instead of a pile of canvas state.

## browser-entry/playback/background/playback.background.constants.ts

### FLAPPY_BACKGROUND_COMPOSITE_LIGHTER

Composite mode used when stacking glow-heavy starfield layers.

The background currently resets to ordinary compositing for main passes, but
this constant documents the additive blend mode used when glow layers need to
visually accumulate rather than overwrite one another.

### FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER

Composite mode used for standard opaque drawing passes.

Most background passes should replace pixels normally so the scene remains
predictable before selective glow passes are added on top.

### FLAPPY_BACKGROUND_HORIZON_GLOW_ALPHA

Soft glow opacity applied during the horizon glow pass.

The glow is intentionally strong enough to read as neon, but still shy of a
full opaque bloom so the crisp core line remains visible.

### FLAPPY_BACKGROUND_HORIZON_GLOW_BLUR_PX

Blur radius used to bloom the horizon divider glow (pixels).

This is the main control for how far the horizon's light appears to bleed
into the neighboring sky and ground bands.

### FLAPPY_BACKGROUND_HORIZON_HALF_THICKNESS_MULTIPLIER

Half-thickness multiplier used to center the horizon line on the split.

The layout computes the horizon around the sky/lower-band seam, so this
multiplier converts stroke thickness into the offset needed to center the
divider on that seam rather than placing it fully below it.

### FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX

Thickness of the neon horizon divider line (pixels).

A slightly heavier stroke helps the divider remain legible against both the
starfield and the bright grid below it.

### FLAPPY_BACKGROUND_HORIZON_STYLE

Frozen neon paint bundle reused by the playback horizon renderer.

Keeping this style object in the constants module prevents repeated
allocation during every background frame while still keeping the palette
centrally theme-owned.

### FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX

Minimum safe viewport dimension used by background render math (pixels).

Canvas helpers in this module assume positive dimensions. Clamping tiny or
temporarily zero-sized layouts to this floor prevents resize races from
producing invalid cache keys or negative geometry.

### FLAPPY_BACKGROUND_ODD_STROKE_ALIGNMENT_OFFSET_PX

Pixel offset used to align odd-width strokes to the device pixel grid.

Offsetting odd-width lines by half a pixel is a standard raster technique
for reducing blur in canvas line rendering.

### FLAPPY_BACKGROUND_ODD_STROKE_DIVISOR

Divisor used to detect odd stroke widths for pixel snapping.

Canvas 2D strokes look soft when odd-width lines are left on whole pixels.
This constant supports the classic half-pixel alignment check used to keep
the horizon divider visually crisp.

### FLAPPY_BACKGROUND_SKY_HEIGHT_RATIO

Background layout ratio reserved for the starfield sky band.

The top band intentionally occupies most of the scene so the future ground
layer can take over the lower strip without competing with the stars.

### FLAPPY_BACKGROUND_TILE_ROW_BUFFER_COUNT

Extra tile count rendered past the visible right edge for seamless wrap.

The starfield is drawn as repeated cached strips. One buffered strip beyond
the viewport prevents empty columns from appearing while the parallax offset
advances between frames.

### FLAPPY_BACKGROUND_TILE_ROW_START_INDEX

Index offset used to draw one extra tile before the visible left edge.

Starting one tile early hides wrap seams when the parallax offset lands near
a tile boundary and the camera reveals a sliver of content just off-screen.

### FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR

Transparent shadow color used to reset canvas glow state.

Canvas shadow state is sticky, so explicit transparent resets prevent one
glow-heavy pass from leaking blur into later solid fills or line work.

## browser-entry/playback/background/playback.background.services.ts

### drawPlaybackBackgroundHorizon

```ts
drawPlaybackBackgroundHorizon(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
): void
```

Draws the glowing horizon divider across the visible viewport.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.

Returns: Nothing.

### drawPlaybackBackgroundSky

```ts
drawPlaybackBackgroundSky(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
  request: PlaybackBackgroundRequest,
): void
```

Draws the starfield parallax clipped to the upper sky band.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.
- `request` - - Narrow render input required for background composition.

Returns: Nothing.

### paintPlaybackBackgroundBase

```ts
paintPlaybackBackgroundBase(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
): void
```

Paints the base background fill for the currently visible viewport.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.

Returns: Nothing.

### resolvePlaybackBackgroundSceneContext

```ts
resolvePlaybackBackgroundSceneContext(
  request: PlaybackBackgroundRequest,
): PlaybackBackgroundSceneContext
```

Resolves the derived scene contract required by the background passes.

Parameters:
- `request` - - Narrow render input required for background composition.

Returns: Immutable scene context shared by the private render helpers.

## browser-entry/playback/background/playback.background.scene.services.ts

### resolvePlaybackBackgroundSceneContext

```ts
resolvePlaybackBackgroundSceneContext(
  request: PlaybackBackgroundRequest,
): PlaybackBackgroundSceneContext
```

Resolves the derived scene contract required by the background passes.

Parameters:
- `request` - - Narrow render input required for background composition.

Returns: Immutable scene context shared by the private render helpers.

## browser-entry/playback/background/playback.background.draw.services.ts

### drawPlaybackBackgroundHorizon

```ts
drawPlaybackBackgroundHorizon(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
): void
```

Draws the glowing horizon divider across the visible viewport.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.

Returns: Nothing.

### drawPlaybackBackgroundSky

```ts
drawPlaybackBackgroundSky(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
  request: PlaybackBackgroundRequest,
): void
```

Draws the starfield parallax clipped to the upper sky band.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.
- `request` - - Narrow render input required for background composition.

Returns: Nothing.

### drawPlaybackHorizonLine

```ts
drawPlaybackHorizonLine(
  context: CanvasRenderingContext2D,
  request: PlaybackHorizonLineRequest,
): void
```

Draws the glowing horizon divider using the provided neon style.

Parameters:
- `context` - - Canvas 2D drawing context.
- `request` - - Width, aligned y-position, and style for the divider.

Returns: Nothing.

### drawPlaybackTiledImageRow

```ts
drawPlaybackTiledImageRow(
  context: CanvasRenderingContext2D,
  startXPx: number,
  tile: StarTileImage,
  tileWidthPx: number,
  visibleWidthPx: number,
  offsetPx: number,
): void
```

Draws a horizontally tiled image strip across the visible width.

Parameters:
- `context` - - Canvas 2D drawing context.
- `startXPx` - - Leftmost visible world x-position for the tiled strip.
- `tile` - - Pre-rendered tile image reused across the sky band.
- `tileWidthPx` - - Width of one repeated tile in pixels.
- `visibleWidthPx` - - Current visible width that must be fully covered.
- `offsetPx` - - Parallax scroll offset used to wrap tile placement.

Returns: Nothing.

### paintPlaybackBackgroundBase

```ts
paintPlaybackBackgroundBase(
  context: CanvasRenderingContext2D,
  sceneContext: PlaybackBackgroundSceneContext,
): void
```

Paints the base background fill for the currently visible viewport.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.

Returns: Nothing.

## browser-entry/playback/background/playback.background.cache.services.ts

### ensurePlaybackBackgroundViewportCacheValidity

```ts
ensurePlaybackBackgroundViewportCacheValidity(
  visibleWorldWidthPx: number,
  visibleWorldHeightPx: number,
): string
```

Ensures background caches only retain entries for the current viewport size.

When the page size changes, cached geometry and coverage counts become
obsolete because the background bands and tile coverage both depend on the
current viewport dimensions.

Parameters:
- `visibleWorldWidthPx` - - Current visible world width in pixels.
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Stable viewport-size cache key for the current frame.

### resolveCachedPlaybackBackgroundLayout

```ts
resolveCachedPlaybackBackgroundLayout(
  visibleWorldHeightPx: number,
  factory: PlaybackBackgroundLayoutFactory,
): PlaybackBackgroundLayout
```

Resolves cached background layout for the current viewport height.

Parameters:
- `visibleWorldHeightPx` - - Current visible world height in pixels.
- `factory` - - Lazy layout builder used when the cache misses.

Returns: Cached background layout for the current viewport size.

### resolveCachedPlaybackTileCoverageCount

```ts
resolveCachedPlaybackTileCoverageCount(
  tileWidthPx: number,
  factory: () => number,
): number
```

Resolves cached tile coverage count for one tile width.

Parameters:
- `tileWidthPx` - - Width of one repeated starfield tile in pixels.
- `factory` - - Lazy coverage builder used when the cache misses.

Returns: Cached tile coverage count for the active viewport width.

### resolvePlaybackBackgroundViewportCacheKey

```ts
resolvePlaybackBackgroundViewportCacheKey(
  visibleWorldWidthPx: number,
  visibleWorldHeightPx: number,
): string
```

Resolves the stable viewport-size cache key used by background caches.

Parameters:
- `visibleWorldWidthPx` - - Current visible world width in pixels.
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Cache key that changes whenever the page size changes.

## browser-entry/playback/background/playback.background.utils.ts

### resolveAlignedHorizonYPx

```ts
resolveAlignedHorizonYPx(
  horizonYPx: number,
  lineThicknessPx: number,
): number
```

Resolves pixel-snapped horizon positioning for crisp canvas strokes.

Parameters:
- `horizonYPx` - - Logical horizon centerline in pixels.
- `lineThicknessPx` - - Stroke thickness in pixels.

Returns: Pixel-snapped y-position for the stroke.

### resolvePlaybackBackgroundLayout

```ts
resolvePlaybackBackgroundLayout(
  visibleWorldHeightPx: number,
): PlaybackBackgroundLayout
```

Resolves the vertical split between the starfield sky and the future ground.

Parameters:
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Stable scene layout for the current frame.

### resolvePlaybackHorizonStyle

```ts
resolvePlaybackHorizonStyle(): PlaybackHorizonStyle
```

Resolves the neon paint settings for the horizon divider.

Returns: Reusable draw style for both the glow and crisp line passes.

### resolveSafeBackgroundDimension

```ts
resolveSafeBackgroundDimension(
  dimensionPx: number,
): number
```

Clamps a background dimension into a render-safe positive integer.

Parameters:
- `dimensionPx` - - Candidate viewport dimension in pixels.

Returns: Positive integer dimension suitable for canvas math.
