# browser-entry/playback/frame-render

## browser-entry/playback/frame-render/playback.frame-render.types.ts

### PlaybackBirdGeometry

Pixel-aligned square geometry used by bird paint helpers.

Bird geometry is resolved once so the detailed bird renderer can reuse a
stable square body box across glow and fill passes.

### PlaybackBirdPaintInput

Bird-paint input shared by the bird render helper module.

This combines geometry and style into one small input object for lower-level
drawing helpers.

### PlaybackBirdRenderer

`(context: CanvasRenderingContext2D, birdYPx: number, birdIndex: number, championBirdIndex: number) => void`

Bird body renderer contract used by frame-render orchestration helpers.

Keeping the renderer as an injected function makes the high-level frame pass
independent from the detailed bird-paint implementation.

### PlaybackFrameSceneContext

Local type contracts for playback frame rendering.

These types are extracted from the broader frame renderer so scene state,
bird geometry, and trail styling can evolve behind a dedicated module
boundary.

Together they describe the inputs and intermediate state needed to paint one
world-space frame on the browser canvas.

### PlaybackTrailRenderer

`(context: CanvasRenderingContext2D, trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], color: string, anchorX: number, baseOpacity: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds) => void`

Trail segment renderer used by frame-render orchestration helpers.

The detailed trail painter owns edge fading and stepped segments; the frame
orchestration only decides when to call it.

### PlaybackTrailRenderStyle

Resolved opacity and color for one bird trail render pass.

Trails use a lighter-weight style record than birds because they only need a
base opacity plus stroke color.

### PlaybackTrailStyleResolver

`(birdIndex: number, championBirdIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailRenderStyle`

Trail style resolver used by frame-render orchestration helpers.

This lets orchestration ask for champion-vs-non-champion trail styling
without embedding color policy directly in the frame pass.

## browser-entry/playback/frame-render/playback.frame-render.service.ts

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

## browser-entry/playback/frame-render/playback.frame-render.services.ts

### beginPlaybackFrameViewportTransform

`(context: CanvasRenderingContext2D, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Applies the viewport transform used for world-space frame rendering.

After this transform, draw calls can work in simulation coordinates instead of
raw canvas pixel coordinates.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### finalizePlaybackFrameCanvas

`(context: CanvasRenderingContext2D) => void`

Restores the caller canvas state after viewport-space frame drawing.

This ensures later canvas users do not inherit playback-specific transform or
alpha state.

Parameters:
- `context` - - Canvas 2D drawing context.

Returns: Nothing.

### preparePlaybackFrameCanvas

`(context: CanvasRenderingContext2D) => void`

Canvas-state helpers for playback frame rendering.

These functions isolate the mutable canvas setup and teardown needed for one
frame so the orchestration layer can read as declarative world rendering.

### renderPlaybackFrameBackground

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Entity-layer rendering helpers for playback frames.

These services paint the world-space contents of a frame once the scene and
canvas transforms have already been resolved.

### renderPlaybackFrameBirds

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext, renderBird: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdRenderer) => void`

Draws all active birds for the current frame.

Only live birds are painted so the playback frame reflects the active
population rather than leaving ghost bodies behind.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.
- `renderBird` - - Bird body renderer owned by the detailed utility layer.

Returns: Nothing.

### renderPlaybackFramePipes

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Draws all visible pipe segments and their neon outlines for the frame.

Pipes are rendered as upper and lower segments connected to the projected
floor profile used by the background grid.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### renderPlaybackFrameTrails

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext, resolveTrailStyle: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailStyleResolver, renderTrail: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailRenderer) => void`

Draws stepped trails for all active birds in the frame.

In practice the trail cache usually contains only the champion trail, but the
renderer stays generic and asks the trail-style policy how each active bird
should be painted.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `trailState` - - Leader trail render cache.
- `sceneContext` - - Shared scene geometry for the frame.
- `resolveTrailStyle` - - Trail style resolver owned by the detailed utility layer.
- `renderTrail` - - Trail segment renderer owned by the detailed utility layer.

Returns: Nothing.

### resolvePlaybackFrameSceneContext

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext`

Scene-resolution helpers for playback frame rendering.

Before anything can be painted, the renderer needs a camera-relative view of
the world: viewport scale, visible bounds, camera origin, and which bird is
currently the champion.

## browser-entry/playback/frame-render/playback.frame-render.scene.services.ts

### resolvePlaybackFrameSceneContext

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext`

Scene-resolution helpers for playback frame rendering.

Before anything can be painted, the renderer needs a camera-relative view of
the world: viewport scale, visible bounds, camera origin, and which bird is
currently the champion.

## browser-entry/playback/frame-render/playback.frame-render.canvas.services.ts

### beginPlaybackFrameViewportTransform

`(context: CanvasRenderingContext2D, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Applies the viewport transform used for world-space frame rendering.

After this transform, draw calls can work in simulation coordinates instead of
raw canvas pixel coordinates.

Parameters:
- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### finalizePlaybackFrameCanvas

`(context: CanvasRenderingContext2D) => void`

Restores the caller canvas state after viewport-space frame drawing.

This ensures later canvas users do not inherit playback-specific transform or
alpha state.

Parameters:
- `context` - - Canvas 2D drawing context.

Returns: Nothing.

### preparePlaybackFrameCanvas

`(context: CanvasRenderingContext2D) => void`

Canvas-state helpers for playback frame rendering.

These functions isolate the mutable canvas setup and teardown needed for one
frame so the orchestration layer can read as declarative world rendering.

## browser-entry/playback/frame-render/playback.frame-render.entity.services.ts

### renderPlaybackFrameBackground

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Entity-layer rendering helpers for playback frames.

These services paint the world-space contents of a frame once the scene and
canvas transforms have already been resolved.

### renderPlaybackFrameBirds

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext, renderBird: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdRenderer) => void`

Draws all active birds for the current frame.

Only live birds are painted so the playback frame reflects the active
population rather than leaving ghost bodies behind.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.
- `renderBird` - - Bird body renderer owned by the detailed utility layer.

Returns: Nothing.

### renderPlaybackFramePipes

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Draws all visible pipe segments and their neon outlines for the frame.

Pipes are rendered as upper and lower segments connected to the projected
floor profile used by the background grid.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### renderPlaybackFrameTrails

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext, resolveTrailStyle: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailStyleResolver, renderTrail: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailRenderer) => void`

Draws stepped trails for all active birds in the frame.

In practice the trail cache usually contains only the champion trail, but the
renderer stays generic and asks the trail-style policy how each active bird
should be painted.

Parameters:
- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `trailState` - - Leader trail render cache.
- `sceneContext` - - Shared scene geometry for the frame.
- `resolveTrailStyle` - - Trail style resolver owned by the detailed utility layer.
- `renderTrail` - - Trail segment renderer owned by the detailed utility layer.

Returns: Nothing.

## browser-entry/playback/frame-render/playback.frame-render.utils.ts

### drawChampionPlaybackBirdGlow

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the champion bird using the same two-pass additive outline glow method as pipes.

Reusing the pipe-style glow language helps the whole playback scene feel like
one visual system instead of unrelated rendering effects.

Parameters:
- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### drawPlaybackBirdBody

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the square bird body with its base neon glow.

This is the standard body pass shared by champion and non-champion birds.

Parameters:
- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### drawTrail

`(context: CanvasRenderingContext2D, trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], color: string, anchorX: number, baseOpacity: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds) => void`

Draws the stepped trail history for one active bird.

The stepped shape makes the trajectory feel more schematic and readable than a
perfectly smooth spline, which fits the overall instrument-panel aesthetic.

Parameters:
- `context` - - Canvas 2D drawing context.
- `trailPoints` - - Cached per-frame trail points for one bird.
- `color` - - Stroke color for the trail.
- `anchorX` - - Bird anchor x-position in world space.
- `baseOpacity` - - Base opacity before edge and lifetime fading.
- `edgeBounds` - - Visible world bounds used for edge fading.

Returns: Nothing.

### renderPlaybackBird

`(context: CanvasRenderingContext2D, birdYPx: number, birdIndex: number, championBirdIndex: number) => void`

Detailed bird-painting helpers for playback frames.

The bird renderer keeps the body intentionally simple and geometric: a neon
square with optional champion glow passes. That makes the population readable
at a glance even when many birds overlap.

### resolvePlaybackBirdBodyGlowBlur

`(birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => number`

Resolves the body glow blur for one bird render pass.

Champion birds receive a slightly stronger body glow than the rest of the
population.

Parameters:
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Blur radius used behind the square bird body.

### resolvePlaybackBirdGeometry

`(birdYPx: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry`

Resolves the fixed bird geometry used by all body rendering passes.

Geometry is snapped to integer pixels so the square body stays crisp instead
of blurring across subpixel boundaries.

Parameters:
- `birdYPx` - - Bird vertical position in world pixels.

Returns: Pixel-aligned square geometry for the bird body.

### resolvePlaybackTrailStyle

`(birdIndex: number, championBirdIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailRenderStyle`

Detailed trail-painting helpers for playback frames.

Trails are rendered as stepped segments with both lifetime fading and edge
fading so motion remains legible without overwhelming the scene.

## browser-entry/playback/frame-render/playback.frame-render.bird.utils.ts

### drawChampionPlaybackBirdGlow

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the champion bird using the same two-pass additive outline glow method as pipes.

Reusing the pipe-style glow language helps the whole playback scene feel like
one visual system instead of unrelated rendering effects.

Parameters:
- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### drawPlaybackBirdBody

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the square bird body with its base neon glow.

This is the standard body pass shared by champion and non-champion birds.

Parameters:
- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### renderPlaybackBird

`(context: CanvasRenderingContext2D, birdYPx: number, birdIndex: number, championBirdIndex: number) => void`

Detailed bird-painting helpers for playback frames.

The bird renderer keeps the body intentionally simple and geometric: a neon
square with optional champion glow passes. That makes the population readable
at a glance even when many birds overlap.

### resolvePlaybackBirdBodyGlowBlur

`(birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => number`

Resolves the body glow blur for one bird render pass.

Champion birds receive a slightly stronger body glow than the rest of the
population.

Parameters:
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Blur radius used behind the square bird body.

### resolvePlaybackBirdGeometry

`(birdYPx: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry`

Resolves the fixed bird geometry used by all body rendering passes.

Geometry is snapped to integer pixels so the square body stays crisp instead
of blurring across subpixel boundaries.

Parameters:
- `birdYPx` - - Bird vertical position in world pixels.

Returns: Pixel-aligned square geometry for the bird body.

## browser-entry/playback/frame-render/playback.frame-render.trail.utils.ts

### drawTrail

`(context: CanvasRenderingContext2D, trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], color: string, anchorX: number, baseOpacity: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds) => void`

Draws the stepped trail history for one active bird.

The stepped shape makes the trajectory feel more schematic and readable than a
perfectly smooth spline, which fits the overall instrument-panel aesthetic.

Parameters:
- `context` - - Canvas 2D drawing context.
- `trailPoints` - - Cached per-frame trail points for one bird.
- `color` - - Stroke color for the trail.
- `anchorX` - - Bird anchor x-position in world space.
- `baseOpacity` - - Base opacity before edge and lifetime fading.
- `edgeBounds` - - Visible world bounds used for edge fading.

Returns: Nothing.

### drawTrailSegmentWithEdgeFade

`(context: CanvasRenderingContext2D, startXPx: number, startYPx: number, endXPx: number, endYPx: number, baseOpacity: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds, startFrameOffset: number, endFrameOffset: number, maximumTrailFrameOffset: number) => void`

Draws one trail segment with combined edge and lifetime fading.

Two independent fade signals are combined here: old segments dim over time,
and segments near the viewport edge fade to avoid harsh clipping.

Parameters:
- `context` - - Canvas 2D drawing context.
- `startXPx` - - Segment start x-position.
- `startYPx` - - Segment start y-position.
- `endXPx` - - Segment end x-position.
- `endYPx` - - Segment end y-position.
- `baseOpacity` - - Base opacity before fade factors.
- `edgeBounds` - - Visible world bounds used for edge fading.
- `startFrameOffset` - - Relative age of the segment start.
- `endFrameOffset` - - Relative age of the segment end.
- `maximumTrailFrameOffset` - - Oldest visible trail age.

Returns: Nothing.

### resolvePlaybackTrailStyle

`(birdIndex: number, championBirdIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailRenderStyle`

Detailed trail-painting helpers for playback frames.

Trails are rendered as stepped segments with both lifetime fading and edge
fading so motion remains legible without overwhelming the scene.
