# track

A single 2D line segment forming part of the track centerline.

Segments are ordered from start to finish of the closed loop.
The closed-loop invariant requires that `segments.at(-1).endX === segments[0].startX`
and `segments.at(-1).endY === segments[0].startY`.

## track/track.generator.types.ts

### SplineSample

One precomputed Catmull-Rom lane-center sample shared across racing systems.

The ownership metadata keeps each sample attached to its source chord while
preserving the stable closed-loop ordering of the sampled ribbon.

### TrackAabb

Axis-aligned pit geometry used for occupancy checks and renderer overlays.

### TrackGenerationViewport

Optional viewport guidance for shape-aware track generation.

The generator uses this to choose whether the loop should be closer to a
circle, a horizontal oval, or a vertical oval while preserving rounded
segment geometry.

### TrackGeneratorInput

Inputs consumed by the deterministic track generator.

Together they form the determinism key: the same values always produce the
same `TrackSpec`.

### TrackPitBox

Per-team pit metadata frozen into the track specification.

Tier 4+ may generate multiple pit descriptors per team. `entranceCorridor`
is the axis-aligned world-space box used by the environment and validators
to decide whether a car has legally reached pit entry, while `pitBox` is
the off-line stall rectangle drawn by the renderer.

### TrackSegment

A single 2D line segment forming part of the track centerline.

Segments are ordered from start to finish of the closed loop.
The closed-loop invariant requires that `segments.at(-1).endX === segments[0].startX`
and `segments.at(-1).endY === segments[0].startY`.

### TrackSpec

Frozen specification for one procedurally generated track layout.

A `TrackSpec` is frozen into the race pack at episode reset and never
regenerated during a live resize. The determinism key is
`seed + layoutVersion + sizeBucket + viewport`. Generated racing tracks may
also carry a deterministic `pitBoxes` list so each team gets dedicated
pit entrance corridors and rendered stalls.

## track/track.generator.ts

### buildPitBoxes

```ts
buildPitBoxes(
  splineSamples: readonly SplineSample[],
): readonly TrackPitBox[]
```

Builds deterministic alternating-team pit metadata from spline progress.

Parameters:
- `splineSamples` - Shared lane-center samples for the generated track.

Returns: Six frozen pit-box descriptors in `[0, 1, 0, 1, 0, 1]` ownership order.

### buildPitBoxForTeam

```ts
buildPitBoxForTeam(
  teamIndex: 0 | 1,
  anchorSample: SplineSample,
  splineSamples: readonly SplineSample[],
  normalDirection: 1 | -1,
): TrackPitBox
```

Builds one team's pit metadata from a spline anchor and normal direction.

The entrance corridor is offset half a track width away from the centerline
so cars on the normal racing line do not accidentally trigger pit stops.

Parameters:
- `teamIndex` - Owning team index.
- `anchorSample` - Spline sample anchoring the pit location.
- `normalDirection` - Signed side selector (`1` or `-1`).

Returns: Deterministic pit metadata for one team.

### clampNumber

```ts
clampNumber(
  value: number,
  minValue: number,
  maxValue: number,
): number
```

Clamps a number to the closed range `[minValue, maxValue]`.

Parameters:
- `value` - Input value.
- `minValue` - Lower bound.
- `maxValue` - Upper bound.

Returns: Clamped value.

### createAxisAlignedBox

```ts
createAxisAlignedBox(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
): TrackAabb
```

Creates an axis-aligned rectangle from center-point inputs.

Parameters:
- `centerX` - Rectangle center X coordinate.
- `centerY` - Rectangle center Y coordinate.
- `width` - Rectangle width.
- `height` - Rectangle height.

Returns: Rounded axis-aligned box descriptor.

### createDeterministicRandom

```ts
createDeterministicRandom(
  initialSeed: number,
): () => number
```

Creates a deterministic xorshift32 PRNG.

Parameters:
- `initialSeed` - Unsigned 32-bit seed.

Returns: Stable pseudo-random number generator in the range [0, 1).

### freezeTrackSpec

```ts
freezeTrackSpec(
  spec: TrackSpec,
): Readonly<TrackSpec>
```

Deeply freezes a `TrackSpec` so that any mutation attempt throws a
`TypeError` in strict mode.

Call this immediately after generation to satisfy the frozen-at-reset
contract: once the spec is locked into the race pack it must be immutable.

Parameters:
- `spec` - The `TrackSpec` to freeze.

Returns: A `Readonly<TrackSpec>` that rejects mutation.

Example:

```ts
const spec = freezeTrackSpec(generateTrack(input));
spec.segments.push(segment); // throws TypeError in strict mode
```

### generateTrack

```ts
generateTrack(
  input: TrackGeneratorInput,
): TrackSpec
```

Generates a deterministic closed-loop `TrackSpec` from the given seed,
layout version, and size bucket.

The same `(seed, layoutVersion, sizeBucket, viewport)` tuple always produces
the same `TrackSpec`. The spec is frozen into the race pack at episode reset;
no regeneration occurs during a live viewport resize. Tier 4+ also derives
three pit boxes per team (six total) at stable lap-progress anchors, with
on-ribbon entrance-corridor AABBs and off-line rendered stall rectangles.

Parameters:
- `input` - Determinism key for the generation algorithm.

Returns: A frozen `TrackSpec` whose segments form a valid closed loop.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
spec.segments; // ordered closed-loop centerline segments
spec.pitBoxes?.length; // 6
```

### resolveCombinedSeed

```ts
resolveCombinedSeed(
  input: TrackGeneratorInput,
): number
```

Combines the generator determinism tuple into one 32-bit seed.

Parameters:
- `input` - Track generator determinism key.

Returns: Unsigned 32-bit seed for the local PRNG.

### resolveOffsetPoint

```ts
resolveOffsetPoint(
  x: number,
  y: number,
  normalX: number,
  normalY: number,
  offsetDistance: number,
): { readonly x: number; readonly y: number; }
```

Resolves an offset point along the supplied local normal vector.

Parameters:
- `x` - Anchor X coordinate.
- `y` - Anchor Y coordinate.
- `normalX` - Unit normal X component.
- `normalY` - Unit normal Y component.
- `offsetDistance` - Signed offset distance.

Returns: Offset point in world coordinates.

### resolvePitAnchorSample

```ts
resolvePitAnchorSample(
  splineSamples: readonly SplineSample[],
  progress01: number,
): SplineSample
```

Resolves the spline sample nearest the requested lap progress.

Parameters:
- `splineSamples` - Shared lane-center samples for the generated track.
- `progress01` - Closed `[0, 1]` lap-progress target.

Returns: Stable sampled anchor for the requested progress bucket.

### resolveTrackRadiusProfile

```ts
resolveTrackRadiusProfile(
  input: TrackGeneratorInput,
  baseRadius: number,
  baseWidth: number,
): { readonly x: number; readonly y: number; }
```

Resolves radius scale factors that adapt the loop to viewport aspect ratio.

The profile scales each axis from the available viewport half-size after
edge padding and lane-width safety margins are reserved, so generated loops
fill the visible area while preserving rounded circle/oval geometry.

Parameters:
- `input` - Generator input possibly carrying viewport metadata.
- `baseRadius` - Size-bucket baseline radius before viewport scaling.
- `baseWidth` - Size-bucket baseline lane width before viewport scaling.

Returns: Radius multipliers for X and Y axes.

### roundTrackGeometry

```ts
roundTrackGeometry(
  value: number,
): number
```

Rounds geometry values so serialized specs stay byte-stable.

Parameters:
- `value` - Floating-point geometry value.

Returns: Rounded geometry value.

## track/track.validation.ts

### areCloseEnough

```ts
areCloseEnough(
  leftValue: number,
  rightValue: number,
): boolean
```

Compares two coordinates using the track closed-loop tolerance.

Parameters:
- `leftValue` - First coordinate.
- `rightValue` - Second coordinate.

Returns: True when the coordinates are effectively equal.

### doAxisAlignedBoxesOverlap

```ts
doAxisAlignedBoxesOverlap(
  firstBox: TrackAabb,
  secondBox: TrackAabb,
): boolean
```

Returns whether two axis-aligned rectangles overlap.

Parameters:
- `firstBox` - First rectangle.
- `secondBox` - Second rectangle.

Returns: True when the rectangles overlap with positive area.

### doLineSegmentsIntersect

```ts
doLineSegmentsIntersect(
  firstSegment: TrackSegment,
  secondSegment: TrackSegment,
): boolean
```

Returns true when two 2D line segments intersect or overlap.

Parameters:
- `firstSegment` - First line segment.
- `secondSegment` - Second line segment.

Returns: True when the two segments cross.

### hasNoSelfIntersection

```ts
hasNoSelfIntersection(
  spec: TrackSpec,
): boolean
```

Returns `true` when no two segments in the spec geometrically intersect.

Adjacent segments may share an endpoint; non-adjacent segments must not
cross.

Parameters:
- `spec` - The track geometry to inspect.

Returns: `true` when no self-intersection is found.

Example:

```ts
if (!hasNoSelfIntersection(spec)) throw new Error('track is self-intersecting');
```

### isAdjacentSegmentPair

```ts
isAdjacentSegmentPair(
  firstSegmentIndex: number,
  secondSegmentIndex: number,
  lastSegmentIndex: number,
): boolean
```

Returns true when a segment pair is adjacent in the closed-loop ordering.

Parameters:
- `firstSegmentIndex` - First segment index.
- `secondSegmentIndex` - Second segment index.
- `lastSegmentIndex` - Final segment index in the loop.

Returns: True when the segments share the loop adjacency exemption.

### isPitCorridorReachable

```ts
isPitCorridorReachable(
  spec: TrackSpec,
  corridor: TrackAabb,
): boolean
```

Returns whether a pit entrance corridor can be reached from the track ribbon.

When spline samples are present, reachability is checked against the smooth
sampled lane center (accurate for corridors placed via spline-normal offsets).
When no spline samples exist (e.g., unit-test fixtures), the check falls back
to the polygon segment approximation.

Parameters:
- `spec` - Track specification.
- `corridor` - Candidate pit entrance corridor.

Returns: True when the corridor lies within the reachable ribbon distance.

### isPointOnSegment

```ts
isPointOnSegment(
  startPoint: { x: number; y: number; },
  point: { x: number; y: number; },
  endPoint: { x: number; y: number; },
): boolean
```

Returns true when a collinear point falls within a segment's bounds.

Parameters:
- `startPoint` - Segment start.
- `point` - Candidate point.
- `endPoint` - Segment end.

Returns: True when the point lies on the segment.

### resolveOrientation

```ts
resolveOrientation(
  startPoint: { x: number; y: number; },
  middlePoint: { x: number; y: number; },
  endPoint: { x: number; y: number; },
): 0 | 1 | 2
```

Resolves the orientation of three points.

Parameters:
- `startPoint` - First point.
- `middlePoint` - Second point.
- `endPoint` - Third point.

Returns: 0 for collinear, 1 for clockwise, 2 for counterclockwise.

### resolvePointToSegmentDistance

```ts
resolvePointToSegmentDistance(
  point: { x: number; y: number; },
  segmentStart: { x: number; y: number; },
  segmentEnd: { x: number; y: number; },
): number
```

Resolves the shortest distance from a point to a line segment.

Parameters:
- `point` - Query point.
- `segmentStart` - Segment start point.
- `segmentEnd` - Segment end point.

Returns: Euclidean point-to-segment distance.

### validatePitCorridorNonOverlap

```ts
validatePitCorridorNonOverlap(
  spec: TrackSpec,
): true
```

Validates that generated pit corridors do not overlap.

Pit entry uses each corridor's axis-aligned bounding box directly, so overlap
would make ownership ambiguous and could invalidate multi-pit team layouts.

Parameters:
- `spec` - Track specification containing optional pit metadata.

Returns: `true` when the pit-corridor layout is valid.

### validatePitCorridorReachability

```ts
validatePitCorridorReachability(
  spec: TrackSpec,
): true
```

Validates that each Tier 4 pit corridor stays reachable from the track.

Reachability is approximated by checking whether the corridor center lies
within the segment half-width plus the corridor radius of any centerline
segment. This keeps the corridor AABB honest: cars must be able to enter it
from the driveable ribbon instead of teleporting into a detached pit zone.

Parameters:
- `spec` - Track specification containing optional pit metadata.

Returns: `true` when every pit corridor is reachable.

### validateTrackSpec

```ts
validateTrackSpec(
  spec: TrackSpec,
): true
```

Validates that a `TrackSpec` satisfies all generation invariants.

Invariants checked:
- `segments.length >= 3` (minimum closed-loop polygon)
- All segments have `width > 0`
- Closed-loop: last segment end equals first segment start (within tolerance)
- No self-intersecting segment pairs
- Tier 4 pit corridors do not overlap
- Tier 4 pit corridors remain reachable from the driveable ribbon

Parameters:
- `spec` - The generated track to validate.

Returns: `true` when all invariants pass.

Example:

```ts
validateTrackSpec(spec); // throws if spec has zero segments
```

## track/track.spline.utils.ts

### buildTrackSplineSamples

```ts
buildTrackSplineSamples(
  segments: readonly TrackSegment[],
  laneCount: number,
): readonly SplineSample[]
```

Builds the shared Catmull-Rom lane-center samples for a closed-loop track.

Parameters:
- `segments` - Ordered closed-loop control segments.
- `laneCount` - Number of drivable lanes; defaults to 2.

Returns: Stable spline samples shared by the renderer and controllers.

### resolveInnerLaneCenterlineOffsetWorld

```ts
resolveInnerLaneCenterlineOffsetWorld(
  splineSample: SplineSample,
): number
```

Resolves the lateral distance from the road centerline to the inner-lane
centerline for one sample.

Parameters:
- `splineSample` - Sample carrying lane geometry metadata.

Returns: Inner-lane centerline offset in world units.

### resolveInnerLaneCenterlinePoint

```ts
resolveInnerLaneCenterlinePoint(
  splineSample: SplineSample,
  splineSampleFrame: SplineSampleFrame,
): { readonly x: number; readonly y: number; }
```

Resolves the world-space point on the inner-lane centerline for one sample.

Parameters:
- `splineSample` - Sample whose centerline anchor is known.
- `splineSampleFrame` - Local tangent frame for the sample.

Returns: World-space point on the inner-lane centerline.

### resolveSplineSampleFrame

```ts
resolveSplineSampleFrame(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): SplineSampleFrame
```

Resolves the local tangent frame for one sampled spline point.

Parameters:
- `splineSamples` - Ordered closed-loop spline samples.
- `sampleIndex` - Global index of the focal sample.

Returns: Tangent heading plus the unit left normal.

### SplineSampleFrame

Tangent-space frame resolved for one sampled spline point.

### TRACK_SPLINE_SAMPLES_PER_SEGMENT

Number of Catmull-Rom samples emitted for each control segment.
