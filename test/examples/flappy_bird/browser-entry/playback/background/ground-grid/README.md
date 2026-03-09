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

### PlaybackGroundGridGeometry

Pure geometry bundle generated before canvas drawing begins.

### PlaybackGroundGridLineSegment

Declarative line segment model used by the ground-grid renderer.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.ts

### renderPlaybackBackgroundGroundGrid

`(context: CanvasRenderingContext2D, sourceScene: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSourceScene, request: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridRequest) => void`

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

`(context: CanvasRenderingContext2D, resolvedScene: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridResolvedScene) => void`

Draws the lower-band atmospheric wash behind the neon line work.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.

Returns: Nothing.

### drawGroundGridSegment

`(context: CanvasRenderingContext2D, segment: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment, lineColor: string, glowColor: string) => void`

Draws one neon line segment with a glow pass and crisp core line.

Parameters:
- `context` - - Canvas 2D drawing context.
- `segment` - - One resolved line segment.
- `lineColor` - - Core neon stroke color.
- `glowColor` - - Outer glow color used for bloom.

Returns: Nothing.

### drawGroundGridSegments

`(context: CanvasRenderingContext2D, segments: readonly import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[], lineColor: string, glowColor: string) => void`

Draws one ordered collection of neon line segments.

Parameters:
- `context` - - Canvas 2D drawing context.
- `segments` - - Ordered line segments to render.
- `lineColor` - - Core neon stroke color.
- `glowColor` - - Outer glow color used for bloom.

Returns: Nothing.

### drawPlaybackGroundGrid

`(context: CanvasRenderingContext2D, resolvedScene: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridResolvedScene, geometry: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridGeometry) => void`

Draws the resolved neon ground grid inside the lower background band.

Parameters:
- `context` - - Canvas 2D drawing context.
- `resolvedScene` - - Geometry and style for the current viewport.
- `geometry` - - Precomputed horizontal and vertical line segments.

Returns: Nothing.

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.constants.ts

### playback.background.ground-grid.constants

Number of horizontal depth bands used by the neon ground grid.

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

### FLAPPY_GROUND_GRID_SCROLL_RATIO

### FLAPPY_GROUND_GRID_TARGET_VERTICAL_LINE_SPACING_PX

### FLAPPY_GROUND_GRID_TARGET_VERTICAL_SEGMENT_HEIGHT_PX

### FLAPPY_GROUND_GRID_VERTICAL_OVERFLOW_COUNT

## browser-entry/playback/background/ground-grid/playback.background.ground-grid.utils.ts

### interpolatePlaybackGroundGridPoint

`(startXPx: number, startYPx: number, endXPx: number, endYPx: number, interpolationRatio: number) => PlaybackGroundGridPoint`

Interpolates one point along a perspective ray.

Parameters:
- `startXPx` - - Bottom anchor x-position.
- `startYPx` - - Bottom anchor y-position.
- `endXPx` - - Vanishing-point x-position.
- `endYPx` - - Vanishing-point y-position.
- `interpolationRatio` - - Normalized 0..1 position along the ray.

Returns: Interpolated point on the perspective ray.

### projectPlaybackGroundGridHorizonXToAnchorX

`(input: PlaybackGroundGridAnchorProjectionInput) => number`

Projects one horizon x-position down to the required floor anchor x-position.

Parameters:
- `input` - - Horizon target and scene geometry.

Returns: Bottom anchor x-position whose ray reaches the target horizon x.

### resolvePlaybackGroundGridAnchorBounds

`(input: PlaybackGroundGridAnchorBoundsInput) => PlaybackGroundGridAnchorBounds`

Projects the visible horizon span back onto the floor anchor line.

Parameters:
- `input` - - Visible horizon bounds and scene geometry.

Returns: Bottom-anchor bounds required to cover the full visible horizon.

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

`(sceneContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, scrollBasePx: number) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridGeometry`

Builds the line geometry for the neon ground grid.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Horizontal depth bands and perspective rays for the current frame.

### resolvePlaybackGroundGridHorizontalLines

`(sceneContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => readonly import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[]`

Builds the screen-horizontal depth bands for the lower neon plane.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.

Returns: Ordered far-to-near line segments.

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

`(sceneContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext) => import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext`

Resolves the shared scene context used by the ground-grid renderer.

Parameters:
- `sceneContext` - - Lower-band geometry provided by the background module.

Returns: Narrow scene contract consumed by grid-specific helpers.

### resolvePlaybackGroundGridVerticalLines

`(sceneContext: import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackBackgroundGroundGridSceneContext, scrollBasePx: number) => readonly import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[]`

Builds the perspective rays that converge to the centered horizon point.

Parameters:
- `sceneContext` - - Lower-band geometry for the current viewport.
- `scrollBasePx` - - Shared world scroll used for parallax motion.

Returns: Wrapped left-to-right perspective rays.

### resolvePlaybackGroundGridVerticalLineSegments

`(input: PlaybackGroundGridVerticalSegmentsInput) => readonly import("C:/NeatapticTS/test/examples/flappy_bird/browser-entry/playback/background/ground-grid/playback.background.ground-grid.types").PlaybackGroundGridLineSegment[]`

Builds tapered style segments for one perspective ray.

Parameters:
- `input` - - Geometry and depth context for one ray.

Returns: Ordered near-to-far segments for one perspective ray.
