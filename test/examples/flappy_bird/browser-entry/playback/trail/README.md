# browser-entry/playback/trail

## browser-entry/playback/trail/playback.trail.history.services.ts

### pushChampionTrailPoint

`(trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], frameIndex: number, yPosition: number) => void`

Appends one point to the champion-only short trail history.

Parameters:
- `trailPoints` - - Mutable champion trail collection.
- `frameIndex` - - Source frame index.
- `yPosition` - - Bird y position.

Returns: Nothing.

### pushTrailPoint

`(trailPoints: import("test/examples/flappy_bird/browser-entry/browser-entry.simulation.types").TrailPoint[], frameIndex: number, yPosition: number, maxRetainedPoints: number) => void`

Appends one trail point while enforcing the maximum retained history length.

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

Parameters:
- `value` - - Candidate value.

Returns: Clamped value.

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
