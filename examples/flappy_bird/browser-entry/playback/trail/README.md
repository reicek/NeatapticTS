# browser-entry/playback/trail

Trail-history helpers for the playback renderer.

This module owns the small rolling histories that create the flock's neon
afterimage. The policy is intentionally simple: keep only a short recent
window, and keep the champion trail even shorter and denser.

Minimal usage sketch:

```ts
const trailPoints = [{ frameIndex: 10, yPx: 140 }];
pushTrailPoint(trailPoints, 11, 136, 2);
```

## browser-entry/playback/trail/playback.trail.history.services.ts

### pushChampionTrailPoint

```ts
pushChampionTrailPoint(
  trailPoints: TrailPoint[],
  frameIndex: number,
  yPosition: number,
): void
```

Appends one point to the champion-only short trail history.

The browser highlights the current leader with a shorter, denser trail than
the rest of the flock. Using a dedicated helper keeps that policy explicit in
the call site instead of scattering champion-specific retention numbers
through the playback renderer.

Parameters:

- `trailPoints` - Mutable champion trail collection.
- `frameIndex` - Source frame index.
- `yPosition` - Bird y position.

Returns: Nothing.

### pushTrailPoint

```ts
pushTrailPoint(
  trailPoints: TrailPoint[],
  frameIndex: number,
  yPosition: number,
  maxRetainedPoints: number,
): void
```

Appends one trail point while enforcing the maximum retained history length.

Playback trails are intentionally modeled as short rolling histories rather
than unbounded path logs. That keeps the neon afterimage readable, prevents
old turns from dominating the current frame, and avoids per-frame growth in a
long-running browser session.

Parameters:

- `trailPoints` - Mutable trail collection.
- `frameIndex` - Source frame index.
- `yPosition` - Bird y position.
- `maxRetainedPoints` - Optional maximum retained trail history length.

Returns: Nothing.

Example:

```ts
const trailPoints = [{ frameIndex: 10, yPx: 140 }];
pushTrailPoint(trailPoints, 11, 136, 2);
```

## browser-entry/playback/trail/playback.trail.opacity.utils.ts

Opacity helpers for playback trail fading.

The trail renderer blends two independent fade stories: lifetime and
distance-to-edge. These helpers keep that math isolated so the draw path can
stay focused on painting rather than re-deriving normalization rules.

Minimal usage sketch:

```ts
const edgeOpacity = resolveEdgeOpacityFactor(120, 140, edgeBounds);
const ageOpacity = resolveTrailLifetimeOpacityFactor(3, 12);
const alpha = edgeOpacity * ageOpacity;
```

### clamp01

```ts
clamp01(
  value: number,
): number
```

Clamps a number to the inclusive [0, 1] range.

The trail renderer combines several normalized fade factors, so keeping this
utility local to the module makes the intent obvious: every opacity channel
must remain safe for direct canvas alpha use.

Parameters:

- `value` - Candidate value.

Returns: Clamped value.

### resolveEdgeOpacityFactor

```ts
resolveEdgeOpacityFactor(
  pointXPx: number,
  pointYPx: number,
  edgeBounds: PlaybackEdgeBounds,
): number
```

Converts distance-to-edge into a normalized opacity factor.

Trail points fade as they approach the viewport border so the rendered path
feels cropped by the camera instead of abruptly chopped off. This mirrors the
common animation principle of easing visual intensity near a frame boundary.

Returns 0 exactly on or beyond an edge and rises to 1 once distance exceeds
the configured fade band.

Parameters:

- `pointXPx` - Point x position.
- `pointYPx` - Point y position.
- `edgeBounds` - Visible world bounds used for edge distance checks.

Returns: Opacity multiplier in [0, 1].

Example:

```ts
const edgeOpacity = resolveEdgeOpacityFactor(120, 140, edgeBounds);
const ageOpacity = resolveTrailLifetimeOpacityFactor(3, 12);
const alpha = edgeOpacity * ageOpacity;
```

### resolveTrailLifetimeOpacityFactor

```ts
resolveTrailLifetimeOpacityFactor(
  frameOffset: number,
  maxTrailFrameOffset: number,
): number
```

Converts trail age into a normalized opacity factor.

This helper implements the other half of the afterimage effect: recent trail
samples should read as energetic and bright, while older samples should fade
away smoothly so the viewer's eye stays anchored to the current flock motion.

Oldest retained history approaches 0 opacity; newest approaches 1.

Parameters:

- `frameOffset` - Frames between this point and newest trail point.
- `maxTrailFrameOffset` - Oldest age offset currently retained by trail.

Returns: Opacity multiplier in [0, 1].
