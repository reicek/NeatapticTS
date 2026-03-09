# browser-entry/playback/frame-render

## browser-entry/playback/frame-render/playback.frame-render.types.ts

### PlaybackBirdGeometry

Pixel-aligned square geometry used by bird paint helpers.

### PlaybackFrameSceneContext

Local type contracts for playback frame rendering.

These types are extracted from the broader frame renderer so scene state,
bird geometry, and trail styling can evolve behind a dedicated module
boundary.

### PlaybackTrailRenderStyle

Resolved opacity and color for one bird trail render pass.

## browser-entry/playback/frame-render/playback.frame-render.service.ts

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

## browser-entry/playback/frame-render/playback.frame-render.services.ts

### beginPlaybackFrameViewportTransform

`(context: CanvasRenderingContext2D, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Applies the viewport transform used for world-space frame rendering.

Parameters:

- `context` - - Canvas 2D drawing context.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### finalizePlaybackFrameCanvas

`(context: CanvasRenderingContext2D) => void`

Restores the caller canvas state after viewport-space frame drawing.

Parameters:

- `context` - - Canvas 2D drawing context.

Returns: Nothing.

### preparePlaybackFrameCanvas

`(context: CanvasRenderingContext2D) => void`

Resets the target canvas and base paint state before frame drawing begins.

Parameters:

- `context` - - Canvas 2D drawing context.

Returns: Nothing.

### renderPlaybackFrameBackground

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Draws the split playback background for the current world viewport.

Parameters:

- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### renderPlaybackFrameBirds

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext, renderBird: PlaybackBirdRenderer) => void`

Draws all active birds for the current frame.

Parameters:

- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.
- `renderBird` - - Bird body renderer owned by the detailed utility layer.

Returns: Nothing.

### renderPlaybackFramePipes

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext) => void`

Draws all visible pipe segments and their neon outlines for the frame.

Parameters:

- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.
- `sceneContext` - - Shared scene geometry for the frame.

Returns: Nothing.

### renderPlaybackFrameTrails

`(context: CanvasRenderingContext2D, renderState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").PopulationRenderState, trailState: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailState, sceneContext: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackFrameSceneContext, resolveTrailStyle: PlaybackTrailStyleResolver, renderTrail: PlaybackTrailRenderer) => void`

Draws stepped trails for all active birds in the frame.

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

Resolves the shared scene contract used by one frame render pass.

Parameters:

- `context` - - Canvas 2D drawing context.
- `renderState` - - Mutable simulation state snapshot.

Returns: Viewport, camera, and edge-bounds state for the frame.

## browser-entry/playback/frame-render/playback.frame-render.utils.ts

### drawPlaybackBirdBody

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the square bird body with its base neon glow.

Parameters:

- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### drawPlaybackBirdChampionAura

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the soft champion aura plate behind the bird body.

Parameters:

- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### drawPlaybackBirdChampionGlowPlate

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, birdRenderStyle: import("test/examples/flappy_bird/browser-entry/playback/playback.render.utils").PlaybackBirdRenderStyle) => void`

Draws the champion-only red glow plate beneath the bird body.

Parameters:

- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `birdRenderStyle` - - Resolved bird style payload.

Returns: Nothing.

### drawPlaybackBirdLeaderRing

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, isChampionBird: boolean) => void`

Draws the leader ring around the champion bird.

Parameters:

- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `isChampionBird` - - Whether the current bird is the champion.

Returns: Nothing.

### drawPlaybackBirdShine

`(context: CanvasRenderingContext2D, birdGeometry: import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry, isChampionBird: boolean) => void`

Draws the reflective shine highlight for one bird body.

Parameters:

- `context` - - Canvas 2D drawing context.
- `birdGeometry` - - Pixel-aligned bird geometry.
- `isChampionBird` - - Whether the current bird is the champion.

Returns: Nothing.

### drawTrail

`(context: CanvasRenderingContext2D, trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], color: string, anchorX: number, baseOpacity: number, edgeBounds: import("test/examples/flappy_bird/browser-entry/playback/playback.types").PlaybackEdgeBounds) => void`

Draws the stepped trail history for one active bird.

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

### renderPlaybackBird

`(context: CanvasRenderingContext2D, birdYPx: number, birdIndex: number, championBirdIndex: number) => void`

Draws one active bird body, glow, shine, and leader ring.

Parameters:

- `context` - - Canvas 2D drawing context.
- `birdYPx` - - Bird vertical position in world pixels.
- `birdIndex` - - Index of the bird being rendered.
- `championBirdIndex` - - Champion index for the current frame.

Returns: Nothing.

### resolvePlaybackBirdGeometry

`(birdYPx: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackBirdGeometry`

Resolves the fixed bird geometry used by all body rendering passes.

Parameters:

- `birdYPx` - - Bird vertical position in world pixels.

Returns: Pixel-aligned square geometry for the bird body.

### resolvePlaybackTrailStyle

`(birdIndex: number, championBirdIndex: number) => import("test/examples/flappy_bird/browser-entry/playback/frame-render/playback.frame-render.types").PlaybackTrailRenderStyle`

Resolves the trail style used for one bird's stepped trail.

Parameters:

- `birdIndex` - - Index of the bird being rendered.
- `championBirdIndex` - - Champion index for the current frame.

Returns: Base opacity and color for the bird trail.
