# browser-entry/playback/background

## browser-entry/playback/background/playback.background.types.ts

### playback.background.types

Minimal render input required to draw the playback background.

Keeping this contract narrow prevents the background renderer from reaching
into unrelated playback state such as birds, pipes, or trail caches.

### PlaybackBackgroundLayout

Resolved vertical scene split used by playback background composition.

### PlaybackBackgroundLayoutFactory

`() => import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundLayout`

Zero-argument builder used to lazily construct one cached background layout.

### PlaybackBackgroundRequest

Minimal render input required to draw the playback background.

Keeping this contract narrow prevents the background renderer from reaching
into unrelated playback state such as birds, pipes, or trail caches.

### PlaybackBackgroundSceneContext

Derived scene contract shared by the playback background render passes.

### PlaybackHorizonLineRequest

Draw request for the horizon divider line.

### PlaybackHorizonStyle

Neon styling contract for the horizon divider line.

## browser-entry/playback/background/playback.background.ts

### renderPlaybackBackground

`(context: CanvasRenderingContext2D, request: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundRequest) => void`

Draws the layered playback background.

The composition keeps the top two-thirds for the neon starfield, fills the
lower band with a TRON-like perspective ground grid, and separates both
regions with a glowing horizon divider.

Parameters:
- `context` - - Canvas 2D drawing context.
- `request` - - Narrow render input required for background composition.

Returns: Nothing.

## browser-entry/playback/background/playback.background.services.ts

### drawPlaybackBackgroundHorizon

`(context: CanvasRenderingContext2D, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundSceneContext) => void`

Draws the glowing horizon divider across the visible viewport.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.

Returns: Nothing.

### drawPlaybackBackgroundSky

`(context: CanvasRenderingContext2D, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundSceneContext, request: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundRequest) => void`

Draws the starfield parallax clipped to the upper sky band.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.
- `request` - - Narrow render input required for background composition.

Returns: Nothing.

### drawPlaybackHorizonLine

`(context: CanvasRenderingContext2D, request: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackHorizonLineRequest) => void`

Draws the glowing horizon divider using the provided neon style.

Parameters:
- `context` - - Canvas 2D drawing context.
- `request` - - Width, aligned y-position, and style for the divider.

Returns: Nothing.

### drawPlaybackTiledImageRow

`(context: CanvasRenderingContext2D, startXPx: number, tile: import("test/examples/flappy_bird/browser-entry/playback/playback.starfield.types").StarTileImage, tileWidthPx: number, visibleWidthPx: number, offsetPx: number) => void`

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

`(context: CanvasRenderingContext2D, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundSceneContext) => void`

Paints the base background fill for the currently visible viewport.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Derived scene geometry and style contract.

Returns: Nothing.

### resolvePlaybackBackgroundSceneContext

`(request: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundRequest) => import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundSceneContext`

Resolves the derived scene contract required by the background passes.

Parameters:
- `request` - - Narrow render input required for background composition.

Returns: Immutable scene context shared by the private render helpers.

## browser-entry/playback/background/playback.background.constants.ts

### FLAPPY_BACKGROUND_COMPOSITE_LIGHTER

### FLAPPY_BACKGROUND_COMPOSITE_SOURCE_OVER

### FLAPPY_BACKGROUND_HORIZON_GLOW_ALPHA

### FLAPPY_BACKGROUND_HORIZON_GLOW_BLUR_PX

### FLAPPY_BACKGROUND_HORIZON_HALF_THICKNESS_MULTIPLIER

### FLAPPY_BACKGROUND_HORIZON_LINE_THICKNESS_PX

### FLAPPY_BACKGROUND_HORIZON_STYLE

### FLAPPY_BACKGROUND_MIN_VIEWPORT_DIMENSION_PX

### FLAPPY_BACKGROUND_ODD_STROKE_ALIGNMENT_OFFSET_PX

### FLAPPY_BACKGROUND_ODD_STROKE_DIVISOR

### FLAPPY_BACKGROUND_SKY_HEIGHT_RATIO

### FLAPPY_BACKGROUND_TILE_ROW_BUFFER_COUNT

### FLAPPY_BACKGROUND_TILE_ROW_START_INDEX

### FLAPPY_BACKGROUND_TRANSPARENT_SHADOW_COLOR

## browser-entry/playback/background/playback.background.cache.services.ts

### ensurePlaybackBackgroundViewportCacheValidity

`(visibleWorldWidthPx: number, visibleWorldHeightPx: number) => string`

Ensures background caches only retain entries for the current viewport size.

When the page size changes, cached geometry and coverage counts become
obsolete because the background bands and tile coverage both depend on the
current viewport dimensions.

Parameters:
- `visibleWorldWidthPx` - - Current visible world width in pixels.
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Stable viewport-size cache key for the current frame.

### resolveCachedPlaybackBackgroundLayout

`(visibleWorldHeightPx: number, factory: import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundLayoutFactory) => import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundLayout`

Resolves cached background layout for the current viewport height.

Parameters:
- `visibleWorldHeightPx` - - Current visible world height in pixels.
- `factory` - - Lazy layout builder used when the cache misses.

Returns: Cached background layout for the current viewport size.

### resolveCachedPlaybackTileCoverageCount

`(tileWidthPx: number, factory: () => number) => number`

Resolves cached tile coverage count for one tile width.

Parameters:
- `tileWidthPx` - - Width of one repeated starfield tile in pixels.
- `factory` - - Lazy coverage builder used when the cache misses.

Returns: Cached tile coverage count for the active viewport width.

### resolvePlaybackBackgroundViewportCacheKey

`(visibleWorldWidthPx: number, visibleWorldHeightPx: number) => string`

Resolves the stable viewport-size cache key used by background caches.

Parameters:
- `visibleWorldWidthPx` - - Current visible world width in pixels.
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Cache key that changes whenever the page size changes.

## browser-entry/playback/background/playback.background.utils.ts

### resolveAlignedHorizonYPx

`(horizonYPx: number, lineThicknessPx: number) => number`

Resolves pixel-snapped horizon positioning for crisp canvas strokes.

Parameters:
- `horizonYPx` - - Logical horizon centerline in pixels.
- `lineThicknessPx` - - Stroke thickness in pixels.

Returns: Pixel-snapped y-position for the stroke.

### resolvePlaybackBackgroundLayout

`(visibleWorldHeightPx: number) => import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackBackgroundLayout`

Resolves the vertical split between the starfield sky and the future ground.

Parameters:
- `visibleWorldHeightPx` - - Current visible world height in pixels.

Returns: Stable scene layout for the current frame.

### resolvePlaybackHorizonStyle

`() => import("test/examples/flappy_bird/browser-entry/playback/background/playback.background.types").PlaybackHorizonStyle`

Resolves the neon paint settings for the horizon divider.

Returns: Reusable draw style for both the glow and crisp line passes.

### resolveSafeBackgroundDimension

`(dimensionPx: number) => number`

Clamps a background dimension into a render-safe positive integer.

Parameters:
- `dimensionPx` - - Candidate viewport dimension in pixels.

Returns: Positive integer dimension suitable for canvas math.
