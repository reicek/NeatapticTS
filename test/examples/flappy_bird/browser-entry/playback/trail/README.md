# browser-entry/playback/trail

## browser-entry/playback/trail/playback.trail.history.services.ts

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

## browser-entry/playback/trail/playback.trail.opacity.utils.ts

### clamp01

`(value: number) => number`

Clamps a number to the inclusive [0, 1] range.

The trail renderer combines several normalized fade factors, so keeping this
utility local to the module makes the intent obvious: every opacity channel
must remain safe for direct canvas alpha use.

Parameters:
- `value` - - Candidate value.

Returns: Clamped value.

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
