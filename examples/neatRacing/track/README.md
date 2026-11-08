# track

NeatRacing track grid draw pass (slice P1-grid-footprint, AC-045).

This pass wraps the same flat-horizon fullscreen shader already shipped in
`../engine/passes/grid.pass.ts` but feeds it the larger track-grid spacing
(5.0 m, giving a 25 m² cell footprint area versus the engine grid's
1 m² baseline). Everything else — the depth-disabled draw, the blending
mode, the alpha envelope, and the pulse pass — stays unchanged.

## track/grid.pass.ts

### buildTrackGridSpacing

```ts
buildTrackGridSpacing(): number
```

Returns the track grid line spacing in world meters.

This is a thin helper so tests can assert the same value the pass will
feed to the `uSpacing` uniform, without hard-coding `5.0` in multiple
places.

Returns: The track grid spacing in meters.

Example:

```ts
const spacing = buildTrackGridSpacing(); // 5.0
```

### compileTrackGridShader

```ts
compileTrackGridShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader
```

Compiles one shader stage or throws a descriptive error carrying the stage
label and the driver info log.

### createTrackGridPass

```ts
createTrackGridPass(
  gl: WebGL2RenderingContext,
): TrackGridPass
```

Creates the track grid pass: compiles and links the reused engine grid
program and resolves all uniform locations. The returned handle never
recompiles or re-uploads; `draw` only sets uniforms and issues the same
fullscreen-triangle draw the engine grid uses, but with the larger
track-grid spacing.

Parameters:
- `gl` - WebGL2 rendering context to bind the pass to.

Returns: The track grid pass handle.

### invertMatrix44

```ts
invertMatrix44(
  out: Float32Array<ArrayBufferLike>,
  m: Float32Array<ArrayBufferLike>,
): boolean
```

Inverts a 4x4 column-major matrix into `out` using the adjugate/determinant
method. The source and destination must not alias.

### NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2

Baseline cell footprint area from the engine ground grid, in m².

### NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2

Target cell footprint area for the track grid, in m².
Equal to the baseline area times the multiplier.

### NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER

Multiplier applied to the baseline footprint area for the track grid.

### NEATRACING_TRACK_GRID_PROGRAM_KEY

Warmup/draw registry key for the track grid pass (distinct from engine grid).

### NEATRACING_TRACK_GRID_SPACING_M

Linear spacing of the track grid lines, derived from the 25 m² cell
footprint area. For a square grid cell, `area = spacing²`, so
`spacing = sqrt(area)`.

### toTrackGridWarmupProgram

```ts
toTrackGridWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Exposes the track grid pass as a shader warmup program (AC-040 seam).

The returned entry keeps compilation lazy: `compileAndLink` builds the
pass, and `drawOffscreen` issues a zero-pose track-grid draw through the
shared pass path so the driver warms the exact render pipeline used on
screen. The context argument is accepted for interface parity; the pass is
bound to the context supplied at construction.

Parameters:
- `context` - WebGL2 context the pass resources will be built against.

Returns: Warmup entry keyed to the `track-grid` registry slot.

### trackGridLineCountForExtent

```ts
trackGridLineCountForExtent(
  extentM: number,
  spacingM: number,
): number
```

Counts how many grid lines fit into a one-axis world extent at the given
spacing. Used by tests to prove that the larger footprint area scales
the line density versus the 1.0 m baseline.

Parameters:
- `extentM` - World extent along one axis, in meters.
- `spacingM` - Distance between adjacent grid lines, in meters.

Returns: The number of whole grid intervals that fit in the extent.

Example:

```ts
const count = trackGridLineCountForExtent(100, buildTrackGridSpacing());
```

### TrackGridPass

The track grid pass handle: create once, draw every frame.

### TrackGridPassDrawState

Per-draw inputs for the track grid pass.

### TrackGridUniformLocations

Resolved uniform locations keyed by uniform name.

## track/road-edge.ts

Hexagonal neon road-edge flat-paint builder for the neatRacing example
(plan slice P1-road-edge-hex, Step 14C).

The roadway floor is paved edge-to-edge with a continuous honeycomb of
large 2 m diameter flat hexagonal tiles. Each tile is a bright neon edge
stroke around a one-shade darker same-hue fill; shared edges are stroked
exactly once, so neighbors join into one continuous hexagonal grid with no
corner gaps and no double borders. The default hue is the existing
cyan/teal palette; beyond the track edge, single-file border lines appear in
red, yellow, then green (~6 m past the edge total). All paint — darker fill
and bright stroke — shares one small positive y offset so it sits just above
the slab top face as a single flat plane. The darker fill ends at the inner
radius and the bright stroke covers the inner-to-outer radius band, so the
matte slab below never shows through as a black outline. There are no
vertical pillars or side faces.

The honeycomb lattice is derived from the same track geometry that builds
the 0.5 m slab ribbon, so the paint aligns with the ribbon and its edge
markers rather than being a separate approximation. The sampled lattice
window is measured from the RIBBON itself (the centerline AABB grown by
the widest published half width) plus the full 6 m border band, so the
tri-color band stays COMPLETE around the convex corners, where the ribbon
cap reaches past the slab raster bounds.

THE EDGE IS NOT THIS MODULE'S TO DECIDE: the band is measured from the
NAMED asymmetric edge published by the edge seam (`./track.edge`) - the
widened outer half-width on the outer side of a corner, the inner
half-width on the inner side - so the band starts where the road actually
ends. This module derives NO half-width of its own and keeps only its OWN
ratified policy: the ladder constants and their CUMULATIVE outer
semantics are untouched, only the base the bands are measured from moves.

Determinism: {@link buildRoadEdgeHexVoxels} is a pure function of the same
`{ seed, layoutVersion, sizeBucket }` tuple used by the slab builder. Two
calls with the same tuple produce byte-identical `JSON.stringify` output on
the same runtime.

### appendEdgeStrokeTriangles

```ts
appendEdgeStrokeTriangles(
  p1: readonly [number, number, number],
  p2: readonly [number, number, number],
  color: readonly [number, number, number],
  out: RoadEdgeHexTriangle[],
): void
```

Appends two triangles that form a thin stroke centered on the edge segment
`p1-p2`. The stroke is extended by a half-width perpendicular offset and
by half the stroke width along the edge direction so adjacent stroke pieces
meet cleanly at hex vertices without leaving corner gaps.

### appendTileFillTriangles

```ts
appendTileFillTriangles(
  tile: RoadEdgeHexLatticeTile,
  out: RoadEdgeHexTriangle[],
): void
```

Appends the six fill triangles for one tile to the output list.

### axialToWorld

```ts
axialToWorld(
  q: number,
  r: number,
): readonly [number, number]
```

Converts flat-topped axial coordinates `(q, r)` to world-space `(x, z)`.

### buildRoadEdgeHexLattice

```ts
buildRoadEdgeHexLattice(
  input: TrackSlabBuildInput,
): RoadEdgeHexLattice
```

Builds the complete honeycomb lattice for the deterministic track tuple.

Exported as a test seam so contract tests can verify edge-to-edge tiling,
single-stroke shared edges, and border-line ordering without
reverse-engineering geometry from overlapping triangle vertices.

Parameters:
- `input` - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.

Returns: Lattice with tiles, fill/edge triangles, and a canonical edge list.

### buildRoadEdgeHexVoxels

```ts
buildRoadEdgeHexVoxels(
  input: TrackSlabBuildInput,
): RoadEdgeHexMesh
```

Builds the flat hexagonal neon road-edge paint mesh for the deterministic
track tuple.

Parameters:
- `input` - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.

Returns: Colored triangle mesh of 2 m flat hex tiles paving the roadway
floor and the red/yellow/green border lines beyond the track edge.

### collectHoneycombTiles

```ts
collectHoneycombTiles(
  spec: TrackSpec,
  edgeProfile: TrackEdgeProfile,
): RoadEdgeHexLatticeTile[]
```

Builds a flat-topped axial honeycomb grid covering the track ribbon plus
the full ~6 m border band, classifies each cell center with the same
track-edge logic that paints the ribbon, and returns the cells that should
emit tiles.

The sampled window is measured from the RIBBON (the union of the
per-segment capsules), NOT from the slab raster bounds: the slab grid
spans the centerline AABB, so at a convex corner the ribbon cap - and with
it the whole border band - reaches past that grid. A window derived from
the slab bounds therefore cut the red shell off at the outer corners and
left them with an incomplete marker set.

Parameters:
- `spec` - Generated track spec.
- `edgeProfile` - The edge profile published by the edge seam for the
ribbon being painted.

Returns: The cells that should emit tiles.

### collectRoadEdgeTileCenters

```ts
collectRoadEdgeTileCenters(
  input: TrackSlabBuildInput,
): { readonly x: number; readonly z: number; readonly colorway: TileColorway; }[]
```

Builds a 2 m honeycomb grid aligned to the track ribbon geometry,
classifies each cell center with the same track-edge logic that paints the
ribbon, and returns every cell that should emit a roadway or border-line
tile.

Exported as a test seam so contract tests can verify lattice alignment
without reverse-engineering centers from overlapping triangle vertices.

### colorwayToColors

```ts
colorwayToColors(
  colorway: TileColorway,
): readonly [readonly [number, number, number], readonly [number, number, number]]
```

Resolves the fill+edge color pair for a tile colorway.

### hexVertices

```ts
hexVertices(
  centerX: number,
  centerZ: number,
  radius: number,
  y: number,
): readonly (readonly [number, number, number])[]
```

Computes the six flat-topped hex vertices at the given radius and height.

### measureTrackBorderBandBounds

```ts
measureTrackBorderBandBounds(
  spec: TrackSpec,
  edgeProfile: TrackEdgeProfile,
): { readonly minX: number; readonly maxX: number; readonly minZ: number; readonly maxZ: number; }
```

Measures the world-space rectangle that has to be sampled so the tri-color
border band is COMPLETE around the whole ribbon.

The ribbon is the union of the per-segment capsules (each chord grown by
its PUBLISHED half width - the widened outer half-width on the outer side
of a corner, the inner half-width on the inner side), so the ribbon AABB
is the centerline AABB grown by the WIDEST published half width. The
border band reaches 6 m past that ribbon, and one hex radius of margin is
added so the outermost tile geometry stays inside the sampled rectangle.

The world z axis mirrors the generator y axis (`worldZ = -y`), the same
handedness the slab builder uses.

A spec with NO segments has no ribbon to grow, so the helper returns an
inverted (empty) rectangle instead of an unbounded one: the caller then
samples no cell at all and emits no tile. This keeps the helper total for
degenerate specs - it never yields NaN bounds and never throws.

Parameters:
- `spec` - Generated track spec.
- `edgeProfile` - The edge profile published by the edge seam for the
ribbon being painted.

Returns: World-space bounds of the sampled rectangle, in meters.

### NEATRACING_ROAD_EDGE_EDGE_COLOR

Bright cyan roadway edge color (0-255 RGB). Its Rec.709 luma exceeds the bloom threshold.

### NEATRACING_ROAD_EDGE_FILL_COLOR

Darker teal roadway fill color (0-255 RGB).

### NEATRACING_ROAD_EDGE_GREEN_EDGE_COLOR

Bright green border-line edge color (0-255 RGB).

### NEATRACING_ROAD_EDGE_GREEN_FILL_COLOR

Darker green border-line fill color (0-255 RGB).

### NEATRACING_ROAD_EDGE_HEX_DIAMETER_M

Corner-to-corner diameter of one flat hex tile, in meters. The roadway is
paved with a honeycomb of these tiles; the same 2 m cell is used for the
single-file red/yellow/green border lines beyond the track edge.

### NEATRACING_ROAD_EDGE_HEX_INNER_RADIUS_M

Inradius that separates the bright edge stroke from the darker fill. The
stroke width is derived from this value so every shared edge is stroked
exactly once. The darker fill ends at this radius and the bright stroke
covers the inner-to-outer radius band on the same paint plane, so the
near-black matte slab below never shows through as a black outline (OBS-23).

### NEATRACING_ROAD_EDGE_HEX_OUTER_RADIUS_M

Circumradius of one flat hex tile: half of the corner-to-corner diameter.

### NEATRACING_ROAD_EDGE_PAINT_Y_OFFSET_M

Small positive world-up offset of the flat paint above the road surface,
in meters. Keeps the paint from z-fighting with the slab top face at y = 0.
Both the darker fill and the bright stroke share this height; there is no
separate stroke y offset (single-plane paint, OBS-23).

### NEATRACING_ROAD_EDGE_RED_EDGE_COLOR

Bright red border-line edge color (0-255 RGB).

### NEATRACING_ROAD_EDGE_RED_FILL_COLOR

Darker red border-line fill color (0-255 RGB).

### NEATRACING_ROAD_EDGE_YELLOW_EDGE_COLOR

Bright yellow border-line edge color (0-255 RGB).

### NEATRACING_ROAD_EDGE_YELLOW_FILL_COLOR

Darker yellow border-line fill color (0-255 RGB).

### resolveEdgeColor

```ts
resolveEdgeColor(
  colorway: TileColorway,
  neighborColorway: TileColorway | null,
): readonly [number, number, number]
```

Chooses the visible color for a shared edge. The colorway with higher
precedence wins, matching the border-line ordering (road → red → yellow →
green). Boundary edges use the tile's own edge color.

### ROAD_EDGE_BORDER_GREEN_M

CUMULATIVE outward distance from the track edge, in meters, at which the
green band ENDS. The green band therefore spans 0-2 m past the edge.

These three limits are CUMULATIVE outer bounds, not per-band widths: each
one is measured from the track edge, so the yellow band is 2 m wide (2-4 m
past the edge) and the red band is 2 m wide (4-6 m past the edge). Do not
read them as the width of their own band.

### ROAD_EDGE_BORDER_RED_M

CUMULATIVE outward distance from the track edge, in meters, at which the
red band ENDS. The red band therefore spans 4-6 m past the edge.

This is a cumulative outer bound measured from the track edge, NOT the
width of the red band itself (the red band is 2 m wide). It is also the
OUTERMOST painted distance: no tile is emitted beyond it.

### ROAD_EDGE_BORDER_YELLOW_M

CUMULATIVE outward distance from the track edge, in meters, at which the
yellow band ENDS. The yellow band therefore spans 2-4 m past the edge.

This is a cumulative outer bound measured from the track edge, NOT the
width of the yellow band itself (the yellow band is 2 m wide).

### RoadEdgeHexBuildInput

Synthetic type alias for the public input tuple.

### RoadEdgeHexLattice

Complete deterministic honeycomb lattice including fill and stroke geometry.

### RoadEdgeHexLatticeEdge

One shared or boundary edge of the honeycomb lattice.

### RoadEdgeHexLatticeTile

One tile of the honeycomb lattice.

### RoadEdgeHexMesh

Output mesh: an ordered list of colored triangles ready for VBO upload.

### RoadEdgeHexTriangle

One triangle of the hex road-edge mesh.

### RoadEdgeHexVertex

One interleaved vertex: world position plus 0-255 RGB color.

### TileColorway

Tile colorway emitted for one honeycomb cell.

`road` covers the drivable ribbon and its curb band; the three border
colorways name the successive rings painted beyond the track edge.

### tileOuterEdgeVertices

```ts
tileOuterEdgeVertices(
  tile: RoadEdgeHexLatticeTile,
  vertexIndex0: number,
  vertexIndex1: number,
): readonly [readonly [number, number, number], readonly [number, number, number]]
```

Computes the two outer vertices that form an edge of a tile.

## track/track.edge.ts

The NAMED track edge seam for the neatRacing example (plan slice
P1-outer-edge-geometry, Step 14U, AC-095).

THE SEAM: the track edge stops being a SYMMETRIC half-width and becomes a
NAMED, CURVATURE-DERIVED PAIR - an INNER half-width and an OUTER
half-width, both measured from the road centreline. The road is WIDER on
the OUTER side of a corner (the drift room the 2026-10-30 user directive
asks for) and UNCHANGED on the INNER side, so the road never narrows and
the road centreline never moves.

WHY IT EXISTS: the edge is read by SEVERAL readers over TWO different
models - the chord-capsule distance (the road rasterizer, the tri-colour
band paint and the containment bound) and the spline-normal distance (the
prop placement and the racing line) - and two readers that derive the
edge themselves can disagree about where the edge IS. This module is the
ONE seam every reader resolves the edge through, so the rasterizer, the
band paint and the containment bound can NEVER disagree about how WIDE the
road is: the WIDTH PROFILE (the inner/outer half-width pair per chord and
per spline sample), the SIDE ASSIGNMENT (loop-global, never flipping at a
straight), the CHORD-DISTANCE MEASURE and the NEAREST-CHORD RULE are all
published here. The readers MEASURE the width through the seam, and each
reader keeps its OWN ratified band policy - the rasterizer and the band
paint keep their min-past-edge UNION over all chords, the containment
bound keeps its arg-min nearest-chord rule - so this seam does NOT change
which chord a reader binds a point to. The corner measure is derived ONCE
here and PUBLISHED - as the ONE corner-measure function of the loop
position, sampled at each publication granularity - so no reader ever
derives its own.

THE TWO FRAMES: the geometry lives in TWO frames and this seam names
BOTH of them. The GENERATOR PLANE is the plane of the segment endpoints
(the generator's own x/y); the WORLD frame MIRRORS that plane as
`worldZ = -generatorY`. ALL THREE READERS MEASURE IN THE GENERATOR PLANE
- the road rasterizer and the tri-colour band paint negate their lattice
z BEFORE they measure, and the containment bound measures the car state
directly - so what differs between them is only the frame their INPUT
arrives in: the rasterizer and the band paint are handed WORLD
coordinates, the containment bound is handed PHYSICS coordinates (which
ARE the generator plane). The side resolution is therefore published in
the GENERATOR PLANE ({@link resolveTrackEdgeSideAt}) and the WORLD-frame
entry point ({@link resolveTrackEdgeSideAtWorldZ}) mirrors the coordinate
for the rasterizer and the band paint, so a world-frame reader can never
invert the side of the road; the chord distance follows the SAME
convention through {@link distanceToTrackChordMAtWorldZ}, so a world-frame
reader can never measure the wrong chord either.

THE ASYMMETRY: the OUTER half-width grows with the local corner measure by
the MEASURED drift space ({@link NEATRACING_TRACK_OUTER_DRIFT_SPACE_M});
the INNER half-width is the EXISTING half-width, UNCHANGED. The widening
is applied on the OUTER side ONLY - the inner side of a corner never
widens and never narrows.

THE CONTINUITY GUARANTEE: the outer/inner side assignment is a property
of the WHOLE loop ({@link TrackEdgeProfile.outerSideSign}), NOT of the
local turn, so it NEVER flips at a straight (where the local turn is zero
and its sign would be meaningless). THAT is the guarantee that holds
EXACTLY, and it is the one the raster, the paint and the bound need:
because the side never flips, the published width profile has NO
discontinuity AT A SIDE FLIP, so no two readers can disagree about which
side of the road a point sits on. The published pair is CONSISTENT at
every publication: the outer half-width is never narrower than the inner
one, the widening is never negative and never exceeds the measured drift
space.

THE ONE CORNER-MEASURE FUNCTION: the corner measure is ONE function of
the loop position - the linear interpolation of the published vertex
measures ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}) along the
chord they join - and the seam publishes that function at TWO
granularities, both of them SAMPLES of it, so the two granularities can
never contradict each other: the per-chord publication is the function's
MIDPOINT sample, HELD over the chord (the chord-capsule model carries ONE
width per chord), and the per-spline-sample publication is the function's
value AT the sample. Where a sample sits at its chord's midpoint, the two
publications carry the SAME corner measure and the SAME drift space, to
the published double precision. The INNER half-width of each publication
is its OWN existing half-width (the chord's own width, the sample's own
interpolated width): the seam never moves an inner edge, it only adds
the OUTER drift room.

THE MEASURED DRIFT SPACE: the numeric value was MEASURED by the RED slice
`P1-outer-edge-red` against the production readers - never decided by an
author - and it is recorded on {@link NEATRACING_TRACK_OUTER_DRIFT_SPACE_M}.

Determinism: every export is a pure function of the track geometry; the
same segments and the same samples always resolve to byte-identical
results on the same runtime (no runtime-random state, no mutation of the
inputs, no side effects).

### buildChordEdge

```ts
buildChordEdge(
  segment: TrackSegment,
  segmentIndex: number,
  turnRadians: readonly number[],
): TrackChordEdge
```

Builds the published edge of one chord: the corner measure (the mean of
the turn magnitudes at the two vertices the chord joins), the drift space
that measure earns, and the named inner/outer half-width pair.

THE PUBLICATION IS A MIDPOINT SAMPLE: the corner measure is the MIDPOINT
sample of the ONE corner-measure function
({@link TrackEdgeProfile.vertexCornerMeasuresDeg}), HELD over the chord
(the chord-capsule model carries ONE width per chord).

THE REFERENCE-CORNER TOLERANCE: the drift factor is EXACTLY 1 at the
90-degree reference corner and at every measure above it, so the published
drift room can never fall below the MEASURED bound - a shortfall AT OR
BELOW the one-nanodegree tolerance
({@link TRACK_EDGE_CORNER_MEASURE_TOLERANCE_DEG}) is absorbed by it
instead of shrinking the drift room. ONLY the saturation is
tolerance-guarded: the INTERIOR ramp is exact only UP TO the double
precision of the published corner measure, so a STRAIGHT (a measure of 0)
earns 0 and a 45-degree corner earns 0.5 to that precision.

Parameters:
- `segment` - The owning raw chord segment.
- `segmentIndex` - Index of the owning chord in the segment list.
- `turnRadians` - The signed turn at every chord's start vertex.

Returns: The published chord edge.

### buildSplineEdge

```ts
buildSplineEdge(
  sample: SplineSample,
  profile: TrackEdgeProfile,
): TrackSplineEdge
```

Builds the published edge of one spline sample from its own interpolated
width and the LOCAL corner measure at the sample.

THE LOCAL MEASURE: the sample's parameter along its owning chord is the
clamped normalised projection of the sample position onto that chord, so
the sample carries the ONE corner-measure function - the linear
interpolation of the profile's published vertex measures along the owning
chord - sampled AT the sample. The owning chord still VALIDATES the
sample: a sample that names a segment the profile does not own throws.

Parameters:
- `sample` - The spline sample (read only).
- `profile` - The edge profile resolved from the same track geometry.

Returns: The published spline edge.

### distanceToTrackChordM

```ts
distanceToTrackChordM(
  segment: Readonly<TrackSegment>,
  pointX: number,
  pointY: number,
): number
```

Measures the distance from a point to a chord in the GENERATOR PLANE: the
Euclidean distance to the CLAMPED nearest point of the chord (the same
clamped-projection measure the production readers use).

A degenerate (zero-length) chord has no projection direction, so its
whole geometry is its start point and the distance is measured to that
point.

THE CANONICAL CHORD DISTANCE: this is the ONE chord-distance measure the
readers should use. The rasterizer, the band paint and the containment
bound keep their own private copies for now; those copies are DELETED in
`P1-outer-edge-paint` when the readers join this seam, so every reader
measures the chord the same way.

Parameters:
- `segment` - The chord segment the distance is measured to (read
only, never mutated; the profile's DEEPLY READONLY segment snapshot is
accepted).
- `pointX` - Generator-plane x of the point, in meters.
- `pointY` - Generator-plane y of the point, in meters.

Returns: The distance from the point to the chord, in meters.

Example:

```ts
distanceToTrackChordM(profile.segments[3], 94, -26); // 4 at the fixture corner
```

### distanceToTrackChordMAtWorldZ

```ts
distanceToTrackChordMAtWorldZ(
  segment: Readonly<TrackSegment>,
  worldX: number,
  worldZ: number,
): number
```

Measures the distance from a WORLD-frame point to a chord, in meters: the
same clamped-projection measure as {@link distanceToTrackChordM}, with the
world z mirrored back into the generator plane.

WHY IT EXISTS: the world frame MIRRORS the generator plane (the plane of
the segment endpoints) as `worldZ = -generatorY`, so a world-frame reader
that measures against the generator-plane chord has to mirror its
coordinate. Mirroring the POINT is enough (the mirror is an isometry, so
the distance is unchanged), but a reader that forgets it would measure the
wrong chord and could read the wrong side of the road. This entry point
performs the mirror for the two world-frame readers (the road rasterizer
and the tri-colour band paint), so the frame convention lives in ONE
place.

Parameters:
- `segment` - The chord segment the distance is measured to (read
only, never mutated; the profile's DEEPLY READONLY segment snapshot is
accepted).
- `worldX` - World-frame x of the point, in meters.
- `worldZ` - World-frame z of the point, in meters (the MIRROR of the
generator-plane y).

Returns: The distance from the point to the chord, in meters.

Example:

```ts
distanceToTrackChordMAtWorldZ(profile.segments[3], 94, 26); // 4 at the fixture corner
```

### measureSignedArea

```ts
measureSignedArea(
  segments: readonly TrackSegment[],
): number
```

Measures the shoelace signed area of the closed loop: a positive area
means the loop winds counter-clockwise.

An EMPTY or ZERO-AREA loop measures exactly `0`, so its winding is
UNDETERMINED; the caller resolves that degenerate case to
`'counter-clockwise'` (see {@link TrackEdgeLoopOrientation}) rather than
throwing.

Parameters:
- `segments` - Ordered closed-loop chord segments (read only).

Returns: The signed area of the loop, in square meters (`0` for an empty
or zero-area loop).

### measureTurnRadians

```ts
measureTurnRadians(
  segments: readonly TrackSegment[],
): readonly number[]
```

Measures the signed turn at the START vertex of every chord: the signed
difference between the incoming chord direction and the outgoing chord
direction, normalised into `(-PI, PI]`. A positive turn is a LEFT turn.

The chord directions are read from the raw chord endpoints (the segment
start and end points), never from a smoothed spline, so the measure is
the geometry the readers actually drive on.

THE INCOMING DIRECTION is the PREVIOUS CHORD'S OWN HEADING (its start
point to its end point), NOT the connector from the previous chord's start
point to this chord's start point. For a closed loop of CONTIGUOUS chords
(the closed-loop invariant `segments.at(-1).end === segments[0].start`,
and every interior chord's end equal to the next chord's start) the two
are identical, and the previous chord's own heading stays well-defined
even when the loop is NOT contiguous.

Parameters:
- `segments` - Ordered closed-loop chord segments (read only).

Returns: One signed turn in radians per chord, in segment order.

### NEATRACING_TRACK_OUTER_DRIFT_SPACE_M

The MEASURED outer drift space of the track edge, in METERS: the extra
half-width the road gains on the OUTER side of a corner. It is exactly the
width of the named curb band beside the road.

MEASURED, NEVER AUTHORED: the RED slice `P1-outer-edge-red` measured the
minimum admissible outer extension against the PRODUCTION readers (the
surface classification, the band paint and the containment bound) and
recorded D >= 1.0 m EXACTLY. The derivation: RED-1's outer probe sits AT
the retired symmetric half-width (EXACTLY 4.0 m from the corner chord)
and MUST classify as ROAD; the production rule is "road iff distance <=
half-width - curb width", so the OUTER half-width at that corner must be
>= 4.0 m + 1.0 m = 5.0 m, hence the outer extension must be >= 1.0 m.
The implemented value is that measured bound.

THE SIDE IT APPLIES TO: the OUTER side of a corner ONLY. The inner side
never widens (and never narrows), and the road centreline never moves.
The value equals the named curb width, so the drift room the outer side
gains is exactly one curb band.

THE CURB-WIDTH COUPLING: the extension is EXACTLY one named curb band -
the drift space EQUALS the named curb width. That curb-width constant is
owned by the READERS (the rasterizer's private curb band and the
physics/containment bound's `NEATRACING_TRACK_CURB_WIDTH_M`) and NOT by
this module: this module owns the WIDENING, the readers own the BAND. If
a reader's curb width ever changes, the drift space must be RE-PINNED in
the SAME slice that changes the curb width, so the two can never drift
apart silently.

Example:

```ts
expect(NEATRACING_TRACK_OUTER_DRIFT_SPACE_M).toBe(1);
```

### normalizeTurnRadians

```ts
normalizeTurnRadians(
  radians: number,
): number
```

Normalises a signed turn into `(-PI, PI]`: the wrap keeps the turn of a
corner whose incoming and outgoing directions straddle the atan2 branch
cut on the correct side.

Parameters:
- `radians` - The raw signed turn difference, in radians.

Returns: The normalised signed turn, in radians.

### resolveChordEdge

```ts
resolveChordEdge(
  profile: TrackEdgeProfile,
  segmentIndex: number,
): TrackChordEdge
```

Reads the published chord edge a sample or a probe names, and fails loudly
when the profile does not own that segment - a silent fallback would let a
reader measure the edge against the WRONG chord.

Parameters:
- `profile` - The edge profile resolved from the track geometry.
- `segmentIndex` - Index of the named chord.

Returns: The published chord edge.

### resolveChordParameter

```ts
resolveChordParameter(
  segment: Readonly<TrackSegment>,
  pointX: number,
  pointY: number,
): number
```

Resolves the NORMALISED parameter of a point along a chord: the clamped
projection of the point onto the chord, in the range `[0, 1]`.

A DEGENERATE (zero-length) chord has no projection direction, so it
resolves to `0` (the chord's own start vertex).

Parameters:
- `segment` - The chord the point is projected onto (read only).
- `pointX` - Generator-plane x of the point, in meters.
- `pointY` - Generator-plane y of the point, in meters.

Returns: The parameter of the point along the chord, in `[0, 1]`.

### resolveCornerMeasureDeg

```ts
resolveCornerMeasureDeg(
  vertexCornerMeasuresDeg: readonly number[],
  segmentIndex: number,
  parameter: number,
): number
```

Resolves the corner measure at a parameter along a chord: the linear
interpolation of the ONE corner-measure function between the two vertex
measures the chord joins.

Parameters:
- `vertexCornerMeasuresDeg` - The published vertex corner measures, in
segment order.
- `segmentIndex` - Index of the owning chord.
- `parameter` - The parameter along the chord, in `[0, 1]`.

Returns: The corner measure at the parameter, in degrees.

### resolveNearestTrackChord

```ts
resolveNearestTrackChord(
  profile: TrackEdgeProfile,
  pointX: number,
  pointY: number,
): TrackNearestChord
```

Resolves the chord a point binds to by the NEAREST-CHORD RULE: the chord
with the smallest {@link distanceToTrackChordM}, measured in the GENERATOR
PLANE.

THE TIE-BREAK: two chords at the SAME distance are a genuine tie, and the
FIRST chord in segment order wins it (the comparison is a strict `<`, so
a later chord never displaces an earlier one). That is the same policy the
containment bound already uses, so the bound and this rule can never
disagree about which chord a point belongs to.

WHAT THIS RULE IS NOT: the rasterizer and the band paint keep their own
ratified min-past-edge UNION band policy (a point is inside a band when
ANY chord's band holds it). That policy is NOT the nearest-chord rule, and
this seam does NOT change it: the two readers may still bind a point to
different chords, because the track widths vary per chord. What the seam
guarantees is that every reader measures the WIDTH through the same
published profile.

Parameters:
- `profile` - The edge profile resolved from the track geometry.
- `pointX` - Generator-plane x of the point, in meters.
- `pointY` - Generator-plane y of the point, in meters.

Returns: The nearest chord: its index in the profile's segment order and
the distance from the point to that chord, in meters.

Example:

```ts
resolveNearestTrackChord(profile, 94, -26); // { segmentIndex: 3, distanceM: 4 }
```

### resolveOuterWideningM

```ts
resolveOuterWideningM(
  cornerMeasureDeg: number,
): number
```

Resolves the OUTER drift space a corner measure earns, in meters: the
MEASURED drift space scaled by the corner measure.

THE SATURATION: the reference corner (90 degrees) and every measure above
it earn the FULL measured drift space, so the published drift room can
never fall below the MEASURED bound - a shortfall AT OR BELOW the
one-nanodegree tolerance
({@link TRACK_EDGE_CORNER_MEASURE_TOLERANCE_DEG}) is absorbed by the
saturation instead of shrinking the drift room. ONLY the saturation is
tolerance-guarded: the INTERIOR ramp is exact only UP TO the double
precision of the published corner measure, which inherits the
double-precision error of the measured turn.

ONE HELPER FOR BOTH PUBLICATIONS: the per-chord publication and the
per-spline-sample publication both derive their drift space from THIS
function, so the corner measure is scaled ONCE and the two granularities
can never disagree about the drift room.

Parameters:
- `cornerMeasureDeg` - The corner measure the drift space is derived
from, in degrees.

Returns: The drift space added on the OUTER side, in meters.

### resolveTrackEdgeHalfWidthM

```ts
resolveTrackEdgeHalfWidthM(
  edge: TrackEdgeHalfWidths,
  side: TrackEdgeSide,
): number
```

Resolves the half-width of one named edge for one named side.

Parameters:
- `edge` - The named edge (the inner/outer half-width pair).
- `side` - The side to resolve: `'inner'` or `'outer'`.

Returns: The half-width of the requested side, in meters.

Example:

```ts
resolveTrackEdgeHalfWidthM(profile.chords[3], 'outer'); // 5 at a 90 degree corner
```

### resolveTrackEdgeProfile

```ts
resolveTrackEdgeProfile(
  segments: readonly TrackSegment[],
): TrackEdgeProfile
```

Resolves the NAMED edge profile of a closed track loop: the published
per-chord edge (the inner/outer half-width pair, the corner measure and
the drift space), the published corner-measure NODES and the loop-global
outer-side assignment.

THE ONE CORNER-MEASURE FUNCTION: the corner measure is ONE function of
the loop position - the linear interpolation of the published vertex
measures ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}) along each
chord they join - and the profile publishes its NODES. The per-chord
publication is that function's MIDPOINT sample, HELD over the chord (the
chord-capsule model carries ONE width per chord), so the widening is
CENTRED on the corner: the STEP in the widening lands SYMMETRICALLY ON
the corner vertex instead of entirely on one side of it. The published
per-chord half-widths are PIECEWISE CONSTANT and STEP between adjacent
chords - the CONTINUITY guarantee that holds EXACTLY is the loop-global
SIDE ASSIGNMENT (it never flips at a straight), together with a
non-negative widening that never exceeds the measured drift space. The
outer half-width grows with the corner measure by the MEASURED drift
space ({@link NEATRACING_TRACK_OUTER_DRIFT_SPACE_M}); the inner
half-width is the existing half-width, UNCHANGED.

THE DEGENERATE TIE-BREAK: an EMPTY or ZERO-AREA loop has no winding, so
its orientation is undetermined and the profile reports it as
`'counter-clockwise'` (the `signedArea >= 0` branch) with the outer-side
sign `-1`; the seam never throws for it.

Parameters:
- `segments` - Ordered closed-loop chord segments (read only, never
 *   mutated; the profile SNAPSHOTS them, so the caller's segment objects
 *   are copied rather than aliased - see  {@link TrackEdgeProfile.segments} ).

Returns: The resolved edge profile: the segments, the per-chord edges, the
vertex corner measures, the loop orientation and the outer-side sign.

Example:

```ts
const profile = resolveTrackEdgeProfile(spec.segments);
profile.outerSideSign; // -1 for a counter-clockwise loop
```

### resolveTrackEdgeSideAt

```ts
resolveTrackEdgeSideAt(
  profile: TrackEdgeProfile,
  segmentIndex: number,
  pointX: number,
  pointY: number,
): TrackEdgeSide
```

Resolves which side of the road centreline a ground-plane point sits on,
relative to the chord it is measured against.

THE COORDINATE FRAME: the point is read in the GENERATOR PLANE - the SAME
plane as the segment endpoints (`startX`/`startY`). The WORLD frame
MIRRORS that plane as `worldZ = -generatorY`, so a world-frame reader must
NOT pass its z straight in here: an un-mirrored z INVERTS the side of the
road on every chord with a non-zero x component. The world-frame entry
point is {@link resolveTrackEdgeSideAtWorldZ}.

The side is resolved from the chord cross product and the PROFILE's
loop-global outer-side sign, so it never flips at a straight and never
depends on the local turn. A point that sits EXACTLY on the chord (a zero
cross product) resolves to `'inner'`: the centreline is road under both
half-widths, and the inner half-width is the conservative one.

Parameters:
- `profile` - The edge profile resolved from the track geometry.
- `segmentIndex` - Index of the chord the point is measured against.
- `pointX` - Generator-plane x of the point, in meters.
- `pointY` - Generator-plane y of the point, in meters (NOT a world z).

Returns: The side of the road centreline the point sits on: `'outer'` or
`'inner'`.

Example:

```ts
resolveTrackEdgeSideAt(profile, 3, 94, -26); // 'outer' on a counter-clockwise loop
```

### resolveTrackEdgeSideAtWorldZ

```ts
resolveTrackEdgeSideAtWorldZ(
  profile: TrackEdgeProfile,
  segmentIndex: number,
  worldX: number,
  worldZ: number,
): TrackEdgeSide
```

Resolves which side of the road centreline a WORLD-frame point sits on,
relative to the chord it is measured against.

THE COORDINATE FRAME: the point is read in the WORLD frame, the frame the
road rasterizer and the tri-colour band paint live in. The world frame
MIRRORS the generator plane (the plane of the segment endpoints) as
`worldZ = -generatorY`, so this entry point mirrors the coordinate back
(`-worldZ`) before it resolves the side.

WHY IT EXISTS: a world-frame reader that passed its z straight into the
generator-plane resolver ({@link resolveTrackEdgeSideAt}) would get an
INVERTED side on every chord with a non-zero x component - the same
physical point would read `'outer'` in one frame and `'inner'` in the
other. Mirroring HERE, at the seam, is what keeps the two frames from
disagreeing.

Parameters:
- `profile` - The edge profile resolved from the track geometry.
- `segmentIndex` - Index of the chord the point is measured against.
- `worldX` - World-frame x of the point, in meters.
- `worldZ` - World-frame z of the point, in meters (the MIRROR of the
generator-plane y).

Returns: The side of the road centreline the point sits on: `'outer'` or
`'inner'`.

Example:

```ts
resolveTrackEdgeSideAtWorldZ(profile, 1, 0, 34); // 'outer' on a counter-clockwise loop
```

### resolveTrackSplineEdges

```ts
resolveTrackSplineEdges(
  samples: readonly SplineSample[],
  profile: TrackEdgeProfile,
): readonly TrackSplineEdge[]
```

Resolves the published edge of every spline sample from the SAME edge
profile the chords use, so the sampled ribbon and the raw chords can
never disagree about the drift room.

The base half-width of a sample is the sample's OWN interpolated width;
the corner measure is the ONE corner-measure function sampled AT the
sample (the linear interpolation of the profile's published vertex
measures along the owning chord), and the drift space is the drift that
measure earns. The owning chord still VALIDATES the sample: a sample
that names a segment the profile does not own throws.

Parameters:
- `samples` - Ordered spline samples of the closed loop (read only,
never mutated).
- `profile` - The edge profile resolved from the same track geometry.

Returns: One published edge per sample, in sample order.

Example:

```ts
const edges = resolveTrackSplineEdges(spec.splineSamples, profile);
edges[0].outerHalfWidthM; // the outer half-width at the first sample
```

### TrackChordEdge

The published edge of ONE raw chord segment: the named half-width pair,
the corner measure the widening was derived from, and the widening itself.

### TrackEdgeHalfWidths

The NAMED edge pair of one place on the track: the INNER half-width (the
existing half-width, UNCHANGED - the road never narrows) and the OUTER
half-width (the inner half-width plus the corner-measured drift space).

Both half-widths are measured from the road CENTRELINE, so widening the
outer side never moves the centreline.

### TrackEdgeLoopOrientation

The winding of the closed loop the edge was resolved from, read from the
sign of the shoelace signed area: `'counter-clockwise'` for a positive
signed area, `'clockwise'` for a negative one.

THE DEGENERATE TIE-BREAK: an EMPTY or ZERO-AREA loop has no winding at
all, so its orientation is undetermined and the seam reports it as
`'counter-clockwise'` (the `signedArea >= 0` branch). The seam never
throws for a degenerate loop.

### TrackEdgeProfile

The resolved edge profile of a whole closed loop: the raw segments, the
published per-chord edges, the published corner-measure NODES, and the
loop-global side assignment.

The side assignment lives on the PROFILE (not on the chord) because the
outer side of a corner is a property of the WHOLE loop: a chord-local
turn sign would flip at a straight, where the local turn is zero and its
sign is meaningless.

### TrackEdgeSide

Which side of the road centreline a point or a half-width belongs to:
`'outer'` is the OUTER side of the loop (the drift side, where the drift
room is added) and `'inner'` is the other side.

THE SIDE IS LOOP-GLOBAL, NEVER PER CORNER: the side is resolved from
{@link TrackEdgeProfile.outerSideSign} - the winding of the WHOLE loop -
and NEVER from the local turn. `'outer'` is the side the loop's winding
turns toward (the drift side) and `'inner'` is the side it turns away
from. A corner whose OWN local turn opposes the loop winding therefore
has NO separately published local outer side: it carries the loop's outer
side, exactly like every other place on the loop. The assignment NEVER
flips at a straight.

### TrackNearestChord

The chord a point binds to, as published by the NEAREST-CHORD RULE: the
index of the owning chord and the distance from the point to that chord.

The rule is measured in the GENERATOR PLANE with the canonical chord
distance ({@link distanceToTrackChordM}).

### TrackSplineEdge

The published edge of ONE spline sample: the named half-width pair at the
sample's own interpolated width, plus the LOCAL corner measure at the
sample and the drift space that measure earns.

THE LOCAL MEASURE: the corner measure is the ONE corner-measure function
of the loop position - the linear interpolation of the published vertex
measures ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}) along the
owning chord - sampled AT this sample, so the sampled ribbon carries the
SAME function the chords do. The owning chord still VALIDATES the sample:
a sample that names a segment the profile does not own throws.

## track/track.slabs.ts

Track slab builder for the neatRacing example (plan slice P1-S7-A,
spec section 6.7 "Track rendering" / decision LD-25).

The track renders as an axis-aligned slab ribbon rasterized onto a
0.5-meter lattice (TP-19) at ground scale, distinct from the voxel-car
world scale (G12: {@link NEATRACING_TRACK_SLAB_PITCH_M} is 0.5 m while
`NEATRACING_VOXEL_WORLD_SIZE` is a much finer per-car lattice size).

Rasterization reuses the deterministic closed-loop generator already
shipped in `examples/racing_curriculum` (spec section 11.1 recycling
inventory), meshes the resulting slab lattice with the shared greedy
mesher (`../voxel/greedy-mesher`), and converts every quad back into
world meters with a 0.5 m multiple minimum-corner position so slab
floors, walls, and the ground plane meet exactly with no z-fighting
seams.

THE EDGE IS NOT THIS MODULE'S TO DECIDE: the road edge is the NAMED
asymmetric pair published by the edge seam (`./track.edge`) - an inner
half-width (the road never narrows) and an outer half-width (the inner
one plus the corner-measured drift room). This module MEASURES the road
through that seam - the published half-width pair, the loop-global side
assignment and the canonical chord distance - and keeps only its OWN
ratified band policy (the min-past-edge UNION over every chord, road
before curb before grass). It derives NO half-width of its own, so the
rasterizer, the tri-colour band paint and the containment bound can never
disagree about how WIDE the road is.

THE RASTERIZED DOMAIN FOLLOWS THE PUBLISHED WIDTHS: the lattice the
ribbon is painted on is the CENTRELINE AABB grown by the WIDEST PUBLISHED
half-width on EVERY side plus the NAMED grass-apron width, so the widened
corner is rasterized instead of being clipped away by a centreline-only
rectangle. The growth is ADDITIVE and DOMAIN-ONLY: it moves no road edge,
no centreline and no band, and it is resolved ONCE per build.

THE BAND GEOMETRY IS PUBLISHED HERE, NOT MIRRORED: this module owns the
two band widths of the tri-colour band - the curb band
({@link NEATRACING_TRACK_CURB_WIDTH_M}) and the grass apron
({@link NEATRACING_TRACK_GRASS_APRON_WIDTH_M}) - and the physics surface
classifier and the ONE containment bound read them from HERE (through the
re-export in `../physics/collision.utils`), so the rasterizer and the
physics model share ONE band boundary (AC-094 (e)) instead of mirroring a
value that could drift apart. The apron width is the width of the
OFF-TRACK BAND: the band stays PAINTED as off-road (the paint ladder is
unchanged - the palette values, their order and their cumulative
semantics are untouched) and only the PHYSICAL meaning of the drivable
edge changes.

Determinism: {@link buildTrackSlabs} is a pure function of
`{ seed, layoutVersion, sizeBucket }` — two calls with the same input
produce byte-identical `JSON.stringify` output in the same runtime
(ordered emission from the shared mesher, no runtime-random state).

Scope note (evidence batch 80 rescoping): the spec defines no bank or
boost-pad track feature, so the slab builder emits road, curb, and
grass surfaces only.

### buildTrackSlabLatticePalette

```ts
buildTrackSlabLatticePalette(): VoxelPaletteEntry[]
```

Builds the slab lattice palette: the RESERVED EMPTY sentinel at index 0
followed by road, curb, then grass. Lattice indices {@link
TRACK_ROAD_SURFACE_INDEX}-{@link TRACK_GRASS_SURFACE_INDEX} must stay
aligned with these positions.

Returns: The four-entry lattice palette handed to the shared mesher.

Example:

```ts
buildTrackSlabLatticePalette(); // [EMPTY, road, curb, grass]
```

### buildTrackSlabs

```ts
buildTrackSlabs(
  input: TrackSlabBuildInput,
): TrackSlabMesh
```

Builds the deterministic track slab mesh for a generateTrack fixture.

Pipeline: recycled closed-loop generator (spec section 11) ->
0.5 m lattice rasterization at ground scale (TP-19) -> shared greedy
mesher (dogfooding LD-28) -> world-meter quad conversion. Two calls
with the same input produce byte-identical meshes (AC-017 Level 2).

Parameters:
- `input` - Deterministic build input
`{ seed, layoutVersion, sizeBucket }` matching the shared generator.

Returns: Track slab mesh with a labeled road/curb/grass palette and
world-meter quads whose positions and extents are 0.5 m multiples.

Example:

```ts
const mesh = buildTrackSlabs({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
mesh.palette.map((entry) => entry.type); // ['road', 'curb', 'grass']
mesh.quads[0]; // { position, normalAxis, width, height, paletteIndex }
```

### buildTrackSlabSurfaceGrid

```ts
buildTrackSlabSurfaceGrid(
  input: TrackSlabBuildInput,
): TrackSlabSurfaceGrid
```

Builds a typed surface grid for the deterministic track tuple, sharing the
same 0.5 m lattice rasterization used by the slab mesh. The returned grid
origin, pitch, and dimensions are exactly those the slab builder uses, so
consumers can place geometry on the same cell centers.

Parameters:
- `input` - Deterministic build input
`{ seed, layoutVersion, sizeBucket }` matching the shared generator.

Returns: Typed surface grid with road, curb, grass, and air cells.

Example:

```ts
const grid = buildTrackSlabSurfaceGrid({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
grid.rows[0][0]; // 'air' | 'road' | 'curb' | 'grass'
```

### classifyTrackSlabSurfaceAt

```ts
classifyTrackSlabSurfaceAt(
  pointX: number,
  pointZ: number,
  segments: readonly TrackSegment[],
  edgeProfile: TrackEdgeProfile,
): TrackSlabSurfaceClassification
```

Classifies an arbitrary ground-plane point using the same band logic that
paints the 0.5 m slab lattice. This lets other builders (for example the
road-edge flat paint) emit geometry whose cells sit exactly on the ribbon's
road/curb/grass boundaries without duplicating classification logic.

THE EDGE IS THE SEAM'S: the half-width of every chord is the PUBLISHED
half-width of the side the point sits on (the inner half-width on the
inner side, the widened outer half-width on the outer side), measured
through the edge seam - this reader derives NO half-width of its own. The
band policy stays THIS reader's: the min-past-edge UNION over every chord
with the road-before-curb-before-grass priority.

Parameters:
- `pointX` - World-frame x of the sample point.
- `pointZ` - World-frame z of the sample point. The edge seam mirrors
the world z back into the generator plane (`worldZ = -y`), the same
handedness the slab rasterizer uses.
- `segments` - Closed-loop segments defining the ribbon.
- `edgeProfile` - The edge profile published by the edge seam for those
segments. It is OPTIONAL: a caller that classifies many points of the
same ribbon passes the profile it resolved ONCE, so the published width
profile is not rebuilt per point. When it is omitted the profile is
resolved from `segments` - a caller that passes a profile MUST pass one
resolved from the SAME segments.

Returns: Surface label plus signed distance past the nearest track edge.
`distancePastEdgeM` is always defined: it is negative inside the roadway,
zero on the curb outer edge, and positive outside.

### measureRasterizedDomainAabb

```ts
measureRasterizedDomainAabb(
  spec: TrackSpec,
  edgeProfile: TrackEdgeProfile,
): TrackDomainAabb
```

Measures the RASTERIZED DOMAIN of the closed loop: the CENTRELINE AABB
grown by the WIDEST PUBLISHED half-width on EVERY side, plus the NAMED
grass-apron width.

THE GROWTH IS ADDITIVE AND DOMAIN-ONLY: the road itself is NOT widened
here - the CENTRELINE, the chord count, the spline samples and the
published half-widths are untouched - only the RECTANGLE the rasterizer
samples is grown, so the widened corner (the published outer half-width)
is rasterized instead of being clipped away.

THE GROWTH IS AXIS-ALIGNED: a lattice rectangle cannot grow on ONE side
of a chord, so the outer/inner distinction is NOT applied to this
rectangle - it is applied to the PROBE and to the WIDENING MEASUREMENT
(the loop-global outer side), never to the lattice bounds.

The apron width is the SAME constant the band policy measures from
({@link NEATRACING_TRACK_GRASS_APRON_WIDTH_M}), so the rasterized domain
contains every cell the band policy can classify: no classified surface
cell is dropped at the border.

Parameters:
- `spec` - Generated track spec.
- `edgeProfile` - The edge profile published by the edge seam for the
ribbon being rasterized.

Returns: World-space bounds of the rasterized domain, in meters (the z
bounds are MIRRORED world z, the scene handedness the rasterizer
paints in).

Example:

```ts
const domain = measureRasterizedDomainAabb(spec, resolveTrackEdgeProfile(spec.segments));
domain.maxX - domain.minX; // the rasterized width, in meters
```

### NEATRACING_TRACK_CURB_WIDTH_M

Width of the curb band on each side of the road core, in meters - the
PUBLISHED curb-band width of the tri-colour band.

THE BAND GEOMETRY IS PUBLISHED HERE, NOT MIRRORED: this module owns the
two band widths of the tri-colour band (the curb band and the grass
apron), and the physics surface classifier and the ONE containment bound
read them from HERE (through the re-export in
`../physics/collision.utils`), so the rasterizer and the physics model
can never disagree about where a band ends.

### NEATRACING_TRACK_GRASS_APRON_WIDTH_M

Width of the grass apron beyond the curb on each side, in meters - the
PUBLISHED width of the OFF-TRACK BAND (AC-094 (e)).

The apron IS the off-track band: it stays PAINTED as off-road (the paint
ladder is unchanged - the palette values, their order and their
cumulative semantics are untouched), and it is also the band the physics
surface classifier reads as the DRIVABLE off-road surface and the ONE
containment bound measures to (the band outer edge). Both read THIS
constant through the re-export in `../physics/collision.utils`, so the
rasterizer and the physics model share ONE band boundary.

### NEATRACING_TRACK_SLAB_PITCH_M

World-space pitch of every track slab, in meters (spec TP-19).

### paintTrackSlabSlices

```ts
paintTrackSlabSlices(
  cellsX: number,
  cellsZ: number,
  originX: number,
  originZ: number,
  edgeProfile: TrackEdgeProfile,
): Uint8Array<ArrayBufferLike>[]
```

Rasterizes the ribbon into one `Uint8Array` slice per lattice z column.

The slab lattice has a single body row (`dims[1] === 1`), so each slice
is a one-row `Uint8Array` indexed by lattice x (`slices[z][x]`), with
cell centers sampled at half-pitch offsets for a symmetric margin.

The lattice z column mirrors the generator y (`worldZ = -y`, the scene
handedness seam shared with the frame composer's car pose and camera):
`originZ` is the mirrored world-meter z of lattice cell z = 0, and each
cell's generator-plane query point negates the world z back into
generator y before the surface classification runs.

Parameters:
- `cellsX` - Lattice width in 0.5 m cells.
- `cellsZ` - Lattice depth in 0.5 m cells.
- `originX` - World-meter x of lattice cell x = 0.
- `originZ` - Mirrored world-meter z of lattice cell z = 0 (the
query point per row is the generator y at -worldZ).
- `edgeProfile` - The edge profile published by the edge seam for the
ribbon being rasterized.

Returns: One painted slice per lattice z, in ascending z order.

Example:

```ts
paintTrackSlabSlices(4, 4, -1, -1, resolveTrackEdgeProfile(spec.segments));
// -> [Uint8Array(4), Uint8Array(4), Uint8Array(4), Uint8Array(4)]
```

### resolveSurfaceIndexAt

```ts
resolveSurfaceIndexAt(
  pointX: number,
  worldZ: number,
  edgeProfile: TrackEdgeProfile,
): number
```

Classifies one ground-plane point into a lattice surface palette index.

Bands are measured from the centerline of every chord, through the edge
seam: the half-width of a chord is the PUBLISHED half-width of the side
the point sits on (the inner half-width on the inner side, the widened
outer half-width on the outer side), so the road is WIDER on the outer
side of a corner and never narrower on the inner side. When a point sits
in the widening of one chord but inside the road core of another, the
highest-priority surface wins (road before curb before grass), keeping
junctions and overlapping corner ribbons drivable.

Parameters:
- `pointX` - World-frame x of the sample point.
- `worldZ` - World-frame z of the sample point.
- `edgeProfile` - The edge profile published by the edge seam for the
ribbon being rasterized.

Returns: Lattice palette index: 1 road, 2 curb, 3 grass, 0 air.

Example:

```ts
resolveSurfaceIndexAt(0, 0, resolveTrackEdgeProfile(spec.segments)); // 1 -> road core
```

### resolveWidestPublishedHalfWidthM

```ts
resolveWidestPublishedHalfWidthM(
  edgeProfile: TrackEdgeProfile,
): number
```

Resolves the WIDEST PUBLISHED half-width of the ribbon: the largest
published OUTER half-width over every chord of the edge profile.

The outer half-width of a corner is the widest one, and the inner
half-width never exceeds it (the road never widens inwards), so this
value bounds the width of EVERY chord band on BOTH sides.

Parameters:
- `edgeProfile` - The edge profile published by the edge seam.

Returns: The widest published half-width, in meters (0 for an empty loop).

Example:

```ts
resolveWidestPublishedHalfWidthM(resolveTrackEdgeProfile(spec.segments));
```

### surfaceIndexToType

```ts
surfaceIndexToType(
  index: number,
): TrackSlabSurfaceCell
```

Maps an internal lattice palette index to the exported surface label.

### synthesizeTrackSlabLattice

```ts
synthesizeTrackSlabLattice(
  spec: TrackSpec,
): TrackSlabLattice
```

Rasterizes the closed-loop ribbon into a `DecodedVoxelModel` lattice on
the world grid, snapping the origin to the 0.5 m grid so every painted
cell and every produced quad maps to exact 0.5 m world coordinates. The
lattice z axis mirrors the generator ground y (`worldZ = -y`), matching
the scene handedness the frame composer applies to the car pose and the
chase camera.

The sampled rectangle is the RASTERIZED DOMAIN (the centreline AABB grown
by the widest published half-width plus the grass apron), so the widened
corner is inside the lattice instead of being clipped away by a
centreline-only rectangle.

Parameters:
- `spec` - Generated track spec (already validated by the generator).

Returns: Lattice model ready for  {@link meshVoxelModel} , plus origin.
 *

Example:

```ts
const lattice = synthesizeTrackSlabLattice(spec);
meshVoxelModel(lattice.model); // greedy-meshed lattice quads
```

### toWorldSlabQuads

```ts
toWorldSlabQuads(
  latticeMesh: VoxelMesh,
  origin: TrackSlabWorldOrigin,
): VoxelQuad[]
```

Converts lattice quads into world-meter track slabs.

Positions become 0.5 m multiple minimum corners
(`origin + lattice * pitch`), extents scale by the pitch, the normal
axis passes through unchanged, and the lattice palette index shifts by
one so index 0 addresses the first exported labeled palette entry
(skipping the internal EMPTY sentinel).

Parameters:
- `latticeMesh` - Merged quads from the shared mesher (cell units).
- `origin` - World-meter origin of lattice cell [0, 0, 0].

Returns: Quads in world meters, preserving deterministic emission order.

Example:

```ts
const quads = toWorldSlabQuads(meshVoxelModel(lattice.model), lattice.origin);
quads[0].position; // world-meter minimum corner at 0.5 m multiples
```

### TrackDomainAabb

Closed interval span of the RASTERIZED track domain on the world ground
plane: the world x axis and the MIRRORED world z axis (the scene
handedness the rasterizer paints in).

### TrackSlabBuildInput

Deterministic track build input mirrored from the recycled generator.

### TrackSlabLattice

Rasterized slab lattice paired with its world-space origin.

### TrackSlabMesh

Mesh of the track: labeled surface palette plus world-meter quads.

### TrackSlabPaletteEntry

One labeled track slab material. Array position is the quad
`paletteIndex` in {@link TrackSlabMesh.quads}; the EMPTY sentinel from
the lattice palette is intentionally not exported.

### TrackSlabSurfaceCell

Surface label for one cell of the track slab lattice, including the
internal air/empty sentinel so callers can reason about boundaries.

### TrackSlabSurfaceClassification

Classification of an arbitrary ground-plane point, including both the
priority-winning surface label and the distance past the nearest track edge
so consumers can place border geometry consistently with the ribbon.

### TrackSlabSurfaceGrid

Typed 2-D grid of track slab surface labels aligned to the same 0.5 m
lattice used by {@link buildTrackSlabs}. This lets other builders emit
geometry whose cells sit exactly on the ribbon's road/curb/grass boundaries
without duplicating classification logic.

### TrackSlabSurfaceType

Surface labels exported with the slab palette (no EMPTY sentinel).

### TrackSlabWorldOrigin

World-meter origin of the slab lattice per axis [x, y, z].

## track/grid.builder.ts

Track grid builder for the neatRacing example (slice P1-grid-footprint,
AC-045).

The Step 14B composition used a 1.0 m engine world-space ground grid as a
baseline (cell footprint area 1 m²). User visual feedback (2026-10-12)
requested larger cells, so the track grid now uses a 25 m² footprint,
giving a linear spacing of 5.0 m. The builder owns the constants and pure
math; the pass in `grid.pass.ts` consumes these values when configuring
the flat-horizon fullscreen draw.

### buildTrackGridSpacing

```ts
buildTrackGridSpacing(): number
```

Returns the track grid line spacing in world meters.

This is a thin helper so tests can assert the same value the pass will
feed to the `uSpacing` uniform, without hard-coding `5.0` in multiple
places.

Returns: The track grid spacing in meters.

Example:

```ts
const spacing = buildTrackGridSpacing(); // 5.0
```

### NEATRACING_TRACK_GRID_BASELINE_CELL_FOOTPRINT_AREA_M2

Baseline cell footprint area from the engine ground grid, in m².

### NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_M2

Target cell footprint area for the track grid, in m².
Equal to the baseline area times the multiplier.

### NEATRACING_TRACK_GRID_CELL_FOOTPRINT_AREA_MULTIPLIER

Multiplier applied to the baseline footprint area for the track grid.

### NEATRACING_TRACK_GRID_SPACING_M

Linear spacing of the track grid lines, derived from the 25 m² cell
footprint area. For a square grid cell, `area = spacing²`, so
`spacing = sqrt(area)`.

### trackGridLineCountForExtent

```ts
trackGridLineCountForExtent(
  extentM: number,
  spacingM: number,
): number
```

Counts how many grid lines fit into a one-axis world extent at the given
spacing. Used by tests to prove that the larger footprint area scales
the line density versus the 1.0 m baseline.

Parameters:
- `extentM` - World extent along one axis, in meters.
- `spacingM` - Distance between adjacent grid lines, in meters.

Returns: The number of whole grid intervals that fit in the extent.

Example:

```ts
const count = trackGridLineCountForExtent(100, buildTrackGridSpacing());
```

## track/public.sprite.ts

Recycled Neatenstein public/agent billboard sprites for the neatRacing
right-world render pass.

Loads a 2D sprite atlas derived from the examples/neatenstein public sprites
and builds screen-facing billboard quads placed on the right-hand side of
the track. Billboards keep a HALF world-space pixel scale, the named
{@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} lever; the right-world
render pass applies the same `focal / distance` perspective projection used
for 3D voxels, so distant crowds shrink while nearby ones stay readable.

THREE IDEAS SHAPE THE MODULE:

- A FOOT ANCHOR. A billboard's `position` is the SURFACE it stands on, not
  its geometric center. The render pass expands the quad upward from that
  point, so a sprite stays glued to the ground or to a prop top no matter
  how far the camera is from it.
- A CAMERA-RELATIVE FACING. Sprites are drawn from one shared atlas of
  eight directions, so the module has to answer "which way is this robot
  looking, relative to the camera?". The direction helpers below turn a
  world-space heading into one of the eight atlas directions by measuring
  the angle between the robot's heading and the camera's forward axis.
- A SHARED SCALE. A billboard is HALF the size of the same pixel count in
  the voxel world - the named {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE}
  lever - so the same perspective anchor shrinks a distant sprite and a
  distant voxel in exactly the same way.

The module exports FOUR GROUPS of public surface:

- THE CROWD ROLE API. {@link PublicRuntimeRole} is the role type, the four
  role constants are {@link PUBLIC_SEATED_RUNTIME_ROLE},
  {@link PUBLIC_WALKER_RUNTIME_ROLE}, {@link PUBLIC_STANDING_RUNTIME_ROLE}
  and {@link PUBLIC_MECHANIC_RUNTIME_ROLE}, {@link PUBLIC_RUNTIME_ROLES}
  lists them in their stable order, and {@link resolvePublicRuntimeRole}
  applies the role-less default.
- THE CROWD SIZE API. {@link NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER}
  is the seated quota of ONE bleacher,
  {@link NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE} is the cap on the
  agents OUTSIDE the bleachers, and
  {@link NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET} is the total the two
  shares add up to.
- THE ATLAS AND FRAME API. {@link PublicSpriteFrame},
  {@link PublicSpriteAtlas}, {@link PublicCamera} and {@link PublicBillboard}
  are the data shapes; {@link loadPublicSpriteAtlas} returns the non-empty
  atlas whose source is 'neatenstein-public';
  {@link selectPublicSpriteFrame} returns a deterministic frame for a given
  (atlas, seed, index) tuple; {@link selectPublicWalkFrame} returns a
  deterministic walk frame; {@link computePublicRuntimeFrameIndex} resolves
  the live frame index of a billboard; {@link resolveDirectionId} maps a
  world yaw onto one of the eight atlas directions of
  {@link DIRECTION_ORDER}; the pose indices
  {@link PUBLIC_POSE_INDEX_STAND}, {@link PUBLIC_POSE_INDEX_WALK1} and
  {@link PUBLIC_POSE_INDEX_WALK2}, together with
  {@link PUBLIC_POSE_COUNT}, address the direction x pose lookup table;
  {@link PUBLIC_TEAM_ACCENT_COLOR_IDS} lists the accents of the team accent
  family; and {@link computeBillboardScale} returns the constant
  {@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} half world scale so
  sprites share the same perspective anchor as voxels.
- THE SCENE BUILDER. {@link buildPublicBillboards} returns the right-side
  track billboards with positive screen-space quads and perspective-correct
  scale.

The billboard representation is 2D; it carries no voxel fallback fields.

THE CROWD CARRIES FOUR ROLES: the SEATED spectators on the bleacher seats,
the STANDING spectators outside the ring, the WALKERS on the trackside and
the MECHANICS of the inner pit band. Every role is built by
{@link buildPublicBillboards}, and every built billboard carries an
explicit role, so the render pass can pick the right pose and accent.

### billboardWorldHalfSize

```ts
billboardWorldHalfSize(
  frame: PublicSpriteFrame,
): readonly [number, number]
```

Computes the world-space half-width and half-height of a billboard.

The footprint reads the SAME named constants as the rendered quad - the
{@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} lever and the
{@link NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M} meter scale - so the
placement footprint and the rendered quad can never disagree.

Parameters:
- `frame` - Sprite frame whose pixel dimensions are known.

Returns: Half-width and half-height in meters.

### buildBillboard

```ts
buildBillboard(
  position: readonly [number, number, number],
  frame: PublicSpriteFrame,
  camera: PublicCamera,
  cameraUp: readonly [number, number, number],
): PublicBillboard
```

Projects one world-space billboard into screen space.

Parameters:
- `position` - World-space position in meters.
- `frame` - Selected 2D sprite frame.
- `camera` - Camera used for projection and distance scaling.
- `cameraUp` - Normalized camera-up vector.

Returns: Billboard with positive screen-space quad and perspective-correct scale.

### buildBleacherSeatPositions

```ts
buildBleacherSeatPositions(
  bleachers: readonly RightWorldProp[],
): readonly [number, number, number][]
```

Builds the deterministic seat positions of the spectator rows.

A bleacher offers exactly ONE sprite row per level, and the rows are read
from the real voxel geometry, so spectator billboards stand on actual seat
surfaces instead of on an arbitrary bounding-box top. The anchors come
back LEVEL-MAJOR - the lowest platform first, then the next platform, and
so on - and every bleacher carries its OWN quota of seated spectators, so
no bleacher of the ring is left empty.

Parameters:
- `bleachers` - Bleacher props emitted by the right-world builder.

Returns: Seat anchors in world-space [x, topY, z] order, at most
 *    {@link NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER} per bleacher.

### buildPublicBillboards

```ts
buildPublicBillboards(
  input: TrackSlabBuildInput,
  camera: PublicCamera,
  atlas: PublicSpriteAtlas,
): readonly PublicBillboard[]
```

Builds public/agent billboards for the right-world pass.

EVERY bleacher of the ring carries its own quota of seated spectators, so
no bleacher of the right world is empty, and the share of the density
budget the seated spectators did not use is spent on the agents OUTSIDE
the bleachers - spread over the WHOLE track instead of over its tail. The
crowd size follows from the world itself: the seated share comes from the
bleacher geometry, and the off-bleacher share is a cap on top of it.

The crowd carries FOUR roles, and they are built in THAT order:

1. The SEATED spectators on the bleacher seats. They come FIRST, because
   the personal-space filter keeps the FIRST billboard of every band, so a
   seated seat must never be starved by an off-bleacher agent.
2. The MECHANICS of the inner pit band. They stand on their own pit
   pedestal and wear a team accent.
3. The STANDING spectators OUTSIDE the ring, beyond the SEAM edge of the
   road.
4. The WALKERS on the trackside, in two bands per people-ring sample.

The STANDING spectators and the WALKERS are placed through a CLEARANCE
search: the agent is pushed further out in steps until its footprint is
clear of every scenery prop AND it keeps the 1 m personal space of every
billboard placed before it. Every candidate clearance is MEASURED beyond
the SEAM edge - the published edge of the track - so the crowd never
relies on the retired spline-normal distance model. An agent with no
clear spot is SKIPPED - the crowd is a cap, never a quota. The MECHANICS
are NOT searched: they are anchored to their own pit pedestal, so they
keep their spot by construction.

The seated spectators, the standing spectators and the mechanics face the
camera with a stand frame so they look at the car, while the walkers use a
slow deterministic walk frame. All billboards keep the foot-anchor
convention and the named
{@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} half world scale.

Parameters:
- `input` - Determinism tuple for the track layout.
- `camera` - Camera used for projection and 8-direction frame selection.
- `atlas` - Public/agent sprite atlas.

Returns: Ordered list of 2D billboards on the right side of the track.

Example:

```ts
const atlas = loadPublicSpriteAtlas();
const camera: PublicCamera = {
  position: [0, 2, 10],
  forward: [0, 0, -1],
  right: [1, 0, 0],
  near: 0.1,
  far: 1000,
  fovDegrees: 60,
};

const billboards = buildPublicBillboards(
  { seed: 42, layoutVersion: 1, sizeBucket: 'large' },
  camera,
  atlas,
);

console.log(billboards.length > 0); // true
console.log(billboards[0]?.runtimeRole); // 'seated' - the first bleacher seat
console.log(billboards[0]?.scale); // 0.5 - sprites share the half voxel world scale
```

### buildPublicSpriteAtlasFrames

```ts
buildPublicSpriteAtlasFrames(): readonly PublicSpriteFrame[]
```

Builds a stable frame list from the Neatenstein robot sprite atlas.

Frames are emitted in COLOR-major order: for each palette color, every
direction×pose frame is recolored and appended. The frame id stays in the
base `direction-pose` form so existing id-split lookup tables keep working.
The last color block is therefore the one a simple direction×pose table maps
to; runtime helpers offset back to each robot's stable color identity.

Returns: Decoded 2D RGBA frames recycled from `ROBOT_SPRITE_FRAMES`.

### computeBillboardScale

```ts
computeBillboardScale(
  distance: number,
): number
```

Returns the billboard world-space scale relative to its raw sprite pixel size.

The returned value is the named
{@link NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE} HALF world scale, so a
crowd billboard is half the size of the same pixel count in the voxel
world. The scale stays DISTANCE-INDEPENDENT: the right-world pass applies
the same `focal / distance` perspective projection used for 3D voxels, so
distant sprites shrink at the same rate as geometry of the same world size.

Parameters:
- `distance` - Distance from the camera in meters (kept for API
stability; the caller is responsible for perspective projection).

Returns: The named half world scale, for every distance.

Example:

```ts
const nearScale = computeBillboardScale(5);
const farScale = computeBillboardScale(50);
console.log(nearScale).toBeCloseTo(0.5, 6); // 1:2 proportion with the voxel world
console.log(farScale).toBeCloseTo(0.5, 6);  // perspective handles shrink
```

### computeCameraUp

```ts
computeCameraUp(
  camera: PublicCamera,
): readonly [number, number, number]
```

Computes the camera-up vector from the forward and right basis vectors.

Parameters:
- `camera` - Camera whose forward/right basis is known.

Returns: Normalized world-up vector perpendicular to forward and right.

### computeHueBucket

```ts
computeHueBucket(
  hueDegrees: number,
): number
```

Rounds a hue angle to its hue bucket.

Parameters:
- `hueDegrees` - Hue angle in degrees.

Returns: The bucket centre in degrees, on the 30 degree grid.

### computeHueBucketDistance

```ts
computeHueBucketDistance(
  left: number,
  right: number,
): number
```

Measures the distance between two hue buckets on the wrapped colour circle.

The distance wraps around 360 degrees, so a bucket at 350 degrees is only
10 degrees away from a bucket at 0 degrees.

Parameters:
- `left` - First hue bucket in degrees.
- `right` - Second hue bucket in degrees.

Returns: The wrapped distance between the two buckets, in degrees.

### computePublicRobotColorId

```ts
computePublicRobotColorId(
  seed: number,
  index: number,
): number
```

Stable per-robot accent color index derived from deterministic identity.

### computePublicRuntimeFrameIndex

```ts
computePublicRuntimeFrameIndex(
  billboard: PublicBillboard,
  cameraWorld: readonly [number, number, number],
  carPosition: readonly [number, number, number],
  timeSeconds: number,
  indexTable: Int16Array<ArrayBufferLike>,
  cameraForward: readonly [number, number, number] | undefined,
): number
```

Computes the runtime atlas frame index for a public/agent billboard.

No strings or objects are allocated: the result is read directly from the
prebuilt direction×pose index table.

The stand frame is CAMERA-RELATIVE: direction index 0 ('front') renders the
sprite facing the camera, which is derived from the camera heading and the
robot-to-car bearing. EVERY role but the walker role renders that frame:
the seated spectators, the standing spectators and the mechanics all show
the camera-relative look-at-car STAND pose, and the walker role is the
ONLY role of the crowd on the walk cycle.

The accent of a billboard follows its role as well: the mechanics wear a
TEAM accent, and every other role keeps the colour-diverse accent of the
public palette.

A billboard WITHOUT a role is read as {@link PUBLIC_WALKER_RUNTIME_ROLE},
which selects the walk poses. Every billboard produced by
{@link buildPublicBillboards} carries an explicit role, so the case does
not arise in a built scene.

Parameters:
- `billboard` - Billboard with runtime role metadata.
- `cameraWorld` - Current camera world position. It is the camera
position of the legacy fallback heading.
- `carPosition` - Live car world position for gaze.
- `timeSeconds` - Monotonic frame time in seconds.
- `indexTable` - Prebuilt direction×pose atlas index table.
- `cameraForward` - Optional camera forward axis. When supplied, the
camera heading comes from that axis; when omitted, the legacy fallback
derives the heading from the camera-to-car bearing (the neatRacing chase
rig always looks at the car).

Returns: Atlas frame index, or -1 if the frame is missing.

Example:

```ts
const atlas = loadPublicSpriteAtlas();

// The lookup table maps a direction and a pose to the frame of the LAST
// colour block, which is how the render pass builds it.
const indexTable = new Int16Array(DIRECTION_ORDER.length * PUBLIC_POSE_COUNT)
  .fill(-1);
indexTable[PUBLIC_POSE_INDEX_STAND] = 120; // 'front-stand' of colour 5

const billboard: PublicBillboard = {
  position: [0, 0, 0],
  frameId: 'front-stand',
  scale: 1,
  distance: 10,
  quad: [0, 0, 0, 0],
  runtimeRole: 'seated',
  seed: 42,
  index: 0,
};

// The robot looks at the car, and the camera looks down -Z from behind it.
const frameIndex = computePublicRuntimeFrameIndex(
  billboard,
  [0, 2, 10], // camera world position
  [0, 0, 0],  // car position
  0,          // frame time in seconds
  indexTable,
  [0, 0, -1], // camera forward axis
);

// The atlas is laid out color-major with 24 frames per color, and the
// accent color of this robot is color 4, so the frame lands in the
// color-4 block that starts at 4 * 24 = 96.
console.log(frameIndex); // 96 - the 'front-stand' frame of this robot's colour

// A table without that cell has no frame to show.
console.log(
  computePublicRuntimeFrameIndex(
    billboard,
    [0, 2, 10],
    [0, 0, 0],
    0,
    new Int16Array(DIRECTION_ORDER.length * PUBLIC_POSE_COUNT).fill(-1),
    [0, 0, -1],
  ),
); // -1
```

### computeRgbHueDegrees

```ts
computeRgbHueDegrees(
  red: number,
  green: number,
  blue: number,
): number
```

Converts an RGB triple into a hue angle in degrees.

The classic max-delta hue formula is used, so a grey pixel - the same max
and min channel - has no hue at all and yields NaN.

Parameters:
- `red` - Red channel in the 0..255 range.
- `green` - Green channel in the 0..255 range.
- `blue` - Blue channel in the 0..255 range.

Returns: The hue angle in the 0..360 degree range, or NaN for a grey pixel.

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

### decodePublicFrame

```ts
decodePublicFrame(
  frame: EncodedFrame,
  id: string,
): PublicSpriteFrame
```

Decodes one encoded Neatenstein robot frame into a 2D RGBA sprite.

Parameters:
- `frame` - Encoded palette-index rows.
- `id` - Stable frame identifier.

Returns: Decoded RGBA sprite frame with positive dimensions.

### DIRECTION_ORDER

Canonical 8-direction order used by the Neatenstein robot sprite data.

The order is the shared index space of the atlas: a direction index is a
position in this list, and the runtime lookup table multiplies it by the
number of poses per direction.

Example:

```ts
console.log(DIRECTION_ORDER.length); // 8
console.log(DIRECTION_ORDER[0]); // 'front'
console.log(DIRECTION_ORDER[4]); // 'back'
```

### EncodedFrame

Encoded robot sprite frame shape reused from Neatenstein.

### isLookAroundPause

```ts
isLookAroundPause(
  seed: number,
  index: number,
  timeSeconds: number,
): boolean
```

Deterministic slow look-around pause for a subset of walkers.

A walker glances at the car only during a short window every 16 seconds,
and only for ~20% of walkers chosen deterministically from seed+index.
The math is branch-free aside from the boolean result and allocates no
objects, keeping the per-frame draw path allocation-free.

### isOffBleacherSpotClear

```ts
isOffBleacherSpotClear(
  x: number,
  z: number,
  halfWidth: number,
  halfDepth: number,
  props: readonly RightWorldProp[],
  placed: readonly PublicBillboard[],
): boolean
```

Reports whether an off-bleacher spot is clear.

The spot is clear when the sprite footprint overlaps NO scenery prop and
when the agent keeps the personal space of every billboard placed before
it.

Parameters:
- `x` - World-space x of the candidate spot.
- `z` - World-space z of the candidate spot.
- `halfWidth` - World-space half-width of the sprite footprint.
- `halfDepth` - World-space half-depth of the sprite footprint.
- `props` - Scenery props of the scene.
- `placed` - Billboards placed before the agent.

Returns: True when the spot is clear of props and of the personal space.

### loadPublicSpriteAtlas

```ts
loadPublicSpriteAtlas(): PublicSpriteAtlas
```

Loads the recycled Neatenstein public/agent sprite atlas.

Returns: A non-empty 2D sprite atlas whose source is 'neatenstein-public'.

Example:

```ts
const atlas = loadPublicSpriteAtlas();
console.log(atlas.source); // 'neatenstein-public'
console.log(atlas.frames.length > 0); // true
```

### NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M

Meters per sprite pixel used to turn frame pixels into world-space size.

The value is the SHARED meter scale of the crowd sprites. It is EXPORTED so
the right-world render pass CAN import it instead of keeping its own copy,
and so the placement footprint of a billboard and the rendered quad can never
disagree about the size of a sprite pixel.

Example:

```ts
console.log(NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M); // 0.05 - meters per sprite pixel
```

### NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET

Density budget of the public/agent billboards per track.

The budget is DERIVED from the world instead of being a magic number: the
seated spectators of every bleacher of the ring (20 bleachers times 12
seats = 240 billboards) PLUS the off-bleacher allowance (34 billboards).
The derivation keeps both shares in step - a world with more bleachers
seats more spectators - so the budget never silently truncates the crowd.

Example:

```ts
console.log(NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET); // 274
```

### NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER

Number of seated spectators every bleacher of the ring carries.

The quota is PER BLEACHER: every bleacher of the right world is populated
with its own share of seated spectators, so no bleacher of the ring is
left empty. The quota is DERIVED from the bleacher geometry - the seats
per level times the number of levels - so the seated crowd covers the FULL
platform of every stand instead of crowding one end of it, and a geometry
change moves the quota with it instead of drifting away from it.

Example:

```ts
console.log(NEATRACING_PUBLIC_BILLBOARD_SEATED_PER_BLEACHER); // 12
```

### NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE

World-space scale of the crowd billboards relative to the voxel world.

The lever is NAMED and EXPORTED so the halving of the crowd sprite world
size has exactly ONE source of truth: the module, the placement footprint
and the right-world render pass all read this constant. With this value the
world width of a billboard is
`frame.width * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M`,
which is HALF the width of the same pixel count in the voxel world. The
result is still perspective-projected by the right-world pass exactly like a
voxel of that world size, so distant sprites shrink only because of the
shared `focal / distance` projection and never because of an extra
per-distance scale factor.

Example:

```ts
console.log(NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE); // 0.5 - half the voxel world scale
```

### NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE

Number of public agents allowed OUTSIDE the bleachers.

The allowance is the share of the density budget that is spent on the
agents that are NOT seated on a stand: the trackside walkers, the standing
spectators outside the ring and the mechanics of the inner pit band. It is
a CAP and not a target: the off-bleacher count never exceeds it, and it
never exceeds the share of the budget the seated spectators did not use.

Example:

```ts
console.log(NEATRACING_PUBLIC_OFF_BLEACHER_AGENT_ALLOWANCE); // 34
```

### OffBleacherAgentPlacement

Placement request of one off-bleacher crowd agent.

### placeBillboardOnSurface

```ts
placeBillboardOnSurface(
  x: number,
  z: number,
  frame: PublicSpriteFrame,
  camera: PublicCamera,
  cameraUp: readonly [number, number, number],
  props: readonly RightWorldProp[],
): PublicBillboard
```

Places a billboard so its base rests on the ground or on top of any scenery
prop its footprint overlaps. The returned position is the foot/base anchor
(y = surface + a small clearance), not the geometric center. The render pass
expands the quad upward from this anchor so the sprite remains surface-aligned
at any runtime camera distance.

Parameters:
- `x` - World-space x center.
- `z` - World-space z center.
- `frame` - Selected sprite frame.
- `camera` - Camera used for projection.
- `cameraUp` - Normalized camera-up vector.
- `props` - Scenery props that can elevate the billboard.

Returns: Billboard with a surface-aligned base anchor and consistent distance/scale.

### placeOffBleacherAgent

```ts
placeOffBleacherAgent(
  request: OffBleacherAgentPlacement,
): PublicBillboard | undefined
```

Places ONE off-bleacher agent through the clearance candidate search.

The candidate offsets are the base clearance and the base clearance plus
the clearance steps, and the FIRST candidate whose footprint is clear of
every scenery prop AND keeps the personal space of every billboard placed
before it wins. An agent with no clear candidate is SKIPPED: the crowd is
a cap, never a quota.

EVERY candidate is MEASURED against the SEAM edge: the candidate offset is
the outward offset whose measured clearance beyond the seam edge reaches
the clearance of that candidate, so the retired spline-normal distance
model never places the crowd again.

Parameters:
- `request` - Placement request of the agent.

Returns: The placed billboard, or undefined when no candidate was clear.

### PUBLIC_MECHANIC_RUNTIME_ROLE

Runtime role of the MECHANICS of the inner pit band.

The mechanic role is the only role that wears a TEAM accent, so the pit
band reads as the crew of the teams instead of the colour-diverse public
crowd.

Example:

```ts
console.log(PUBLIC_MECHANIC_RUNTIME_ROLE); // 'mechanic'
```

### PUBLIC_POSE_COUNT

Number of poses stored per direction in the runtime lookup table.

Example:

```ts
console.log(PUBLIC_POSE_COUNT); // 3
```

### PUBLIC_POSE_INDEX_STAND

Numeric index of the stand pose inside a direction×pose lookup table.

Example:

```ts
console.log(PUBLIC_POSE_INDEX_STAND); // 0
```

### PUBLIC_POSE_INDEX_WALK1

Numeric index of the first walk pose inside a direction×pose lookup table.

Example:

```ts
console.log(PUBLIC_POSE_INDEX_WALK1); // 1
```

### PUBLIC_POSE_INDEX_WALK2

Numeric index of the second walk pose inside a direction×pose lookup table.

Example:

```ts
console.log(PUBLIC_POSE_INDEX_WALK2); // 2
```

### PUBLIC_RUNTIME_ROLES

Every runtime role of the public crowd, in the stable order.

The order is SEATED, WALKER, STANDING, MECHANIC and never changes, so a
consumer can index the list as a stable enum of the crowd.

Example:

```ts
console.log(PUBLIC_RUNTIME_ROLES.length); // 4
console.log(PUBLIC_RUNTIME_ROLES[0]); // 'seated'
```

### PUBLIC_SEATED_RUNTIME_ROLE

Runtime role of the SEATED spectators on the bleacher seats.

The seated role is the ONLY role of a billboard that is anchored to a
bleacher seat: the render pass keeps it on the stand pose and never puts
it on the walk cycle.

Example:

```ts
console.log(PUBLIC_SEATED_RUNTIME_ROLE); // 'seated'
```

### PUBLIC_STANDING_RUNTIME_ROLE

Runtime role of the STANDING spectators outside the ring.

The standing role shares the stand pose with the seated spectators, but it
describes an agent that is NOT anchored to a bleacher: it stands on the
ground outside the ring and looks at the car.

Example:

```ts
console.log(PUBLIC_STANDING_RUNTIME_ROLE); // 'standing'
```

### PUBLIC_TEAM_ACCENT_COLOR_IDS

Accent ids of the public palette that belong to the TEAM accent family.

The family rule is the "same 30 degree hue band": an accent belongs to the
family when its hue bucket is within one bucket of the hue bucket of one
of the team colors. The ids are DERIVED from the accent palette and the
team colors at module load, so a palette or a team-color change moves the
family with it instead of drifting away from it.

Example:

```ts
console.log(PUBLIC_TEAM_ACCENT_COLOR_IDS); // [0, 2, 4]
```

### PUBLIC_WALKER_RUNTIME_ROLE

Runtime role of a billboard that walks instead of sitting.

This is the SINGLE definition of the role-less default: a billboard that
carries no `runtimeRole` is read as this role by
{@link resolvePublicRuntimeRole}. Keeping one definition is what makes the
atlas frame-index helper and the right-world render pass AGREE.

Example:

```ts
console.log(PUBLIC_WALKER_RUNTIME_ROLE); // 'walker'
```

### PublicBillboard

A 2D billboard instance placed in world space and projected to screen space.

A billboard is a camera-facing SPRITE, not a voxel. It carries the world
anchor the sprite stands on, the atlas frame it was built with, a small set
of runtime hints that let the render pass re-derive the live pose of the
agent on every frame, and the projection the building camera produced.

The shape carries FOUR kinds of information:

- THE ANCHOR. {@link PublicBillboard.position} is the SURFACE the sprite
  stands on, and the render pass expands the world-space quad upward from
  it, so the sprite stays glued to the ground or to a prop top.
- THE FRAME. {@link PublicBillboard.frameId} is the atlas frame the builder
  selected; the pass recomputes the live frame index from the role, the
  camera and the clock.
- THE RUNTIME HINTS. The role, the seed, the index and the walk direction
  are the deterministic identity of the agent, and they are what makes the
  live pose reproducible from frame to frame.
- THE PROJECTION. The scale, the distance and the quad describe how the
  billboard looked to the camera that built it.

Example:

```ts
const atlas = loadPublicSpriteAtlas();
const camera: PublicCamera = {
  position: [0, 2, 20],
  forward: [0, -0.1, -1],
  right: [1, 0, 0],
  near: 0.1,
  far: 200,
  fovDegrees: 60,
};

const billboards = buildPublicBillboards(
  { seed: 42, layoutVersion: 1, sizeBucket: 'large' },
  camera,
  atlas,
);

const first = billboards[0]!;

// The first billboard is a seated spectator on a bleacher seat, and its
// anchor is the seat SURFACE, not the geometric center of the sprite.
console.log(first.runtimeRole); // 'seated'
console.log(first.position[1].toFixed(1)); // '0.8' - the seat top of level one

// The quad is a positive-extent [left, top, right, bottom] rectangle.
console.log(first.quad[2] > first.quad[0]); // true
console.log(first.quad[3] > first.quad[1]); // true
```

### PublicCamera

Minimal camera contract used to project world billboards into screen space.

The crowd needs three things from a camera: where it is, which way it looks
and how wide its field of view is. The basis is expected to be orthonormal -
`forward` and `right` are unit vectors and the world-up axis is derived
from their cross product - because the billboard projection measures the
anchor against that basis.

The near and far planes belong to the camera contract but are NOT read by
the billboard projection: the crowd quads are drawn in world space and
depth-tested by the renderer, so the sprite module never has to clip.

Example:

```ts
const camera: PublicCamera = {
  position: [0, 2, 20],
  forward: [0, -0.1, -1],
  right: [1, 0, 0],
  near: 0.1,
  far: 200,
  fovDegrees: 60,
};
```

### PublicRuntimeRole

Runtime role of a billboard of the public crowd.

The re-populated right world carries FOUR crowd roles: the SEATED
spectators on the bleacher seats, the WALKERS on the trackside, the
STANDING spectators outside the ring and the MECHANICS of the inner pit
band. The role drives the runtime pose and the runtime accent of the
billboard.

Example:

```ts
const role: PublicRuntimeRole = 'mechanic';

// The role list is the stable enum of the crowd.
console.log(PUBLIC_RUNTIME_ROLES.includes(role)); // true
console.log(PUBLIC_RUNTIME_ROLES); // ['seated', 'walker', 'standing', 'mechanic']
```

### PublicSpriteAtlas

Sprite atlas loaded from the recycled Neatenstein public sprites.

The atlas is the single texture source of the crowd: one packed image that
holds every accent color, every facing direction and every pose of the
public sprite. The render path therefore uploads ONE texture and samples it
with a per-frame UV rectangle instead of uploading a texture per sprite.

The frame list is COLOR-MAJOR - the frames of one accent color are
contiguous - so a frame index can be shifted by a whole color block to
reach the same direction and pose in a different accent.

Example:

```ts
const atlas = loadPublicSpriteAtlas();

// 6 accent colors x 8 directions x 3 poses = 144 frames.
console.log(atlas.source); // 'neatenstein-public'
console.log(atlas.frames.length); // 144
```

### PublicSpriteFrame

One recycled Neatenstein public/agent sprite frame.

A frame is a decoded, ready-to-upload 2D image: the palette-encoded
Neatenstein robot sprite is expanded to its pixel size and stored as
straight RGBA bytes, with row 0 at the logical sprite TOP, so a frame can be
sampled by a texture upload without any further conversion.

The frame id keeps the plain `direction-pose` form, and the atlas is
COLOR-MAJOR, so the frames of one accent color are contiguous. An index
that addresses one color block therefore addresses the same direction and
the same pose of every other color block once it is shifted by a whole
block.

The pixel dimensions are the contract the world size is derived from: a
frame of `w x h` pixels covers `w * NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE
* NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` meters of world width, and the
same factor of world height.

Example:

```ts
const atlas = loadPublicSpriteAtlas();
const frame = atlas.frames[0]!;

// The id keeps the `direction-pose` form, whatever the accent color is.
console.log(frame.id); // 'front-stand'

// The RGBA buffer is a straight, top-down image of the frame's pixels.
console.log(frame.rgba.length); // 147456 - frame.width * frame.height * 4
```

### recolorFrameRgba

```ts
recolorFrameRgba(
  frame: PublicSpriteFrame,
  accent: readonly [number, number, number, number],
): PublicSpriteFrame
```

Recolors the saturated red/orange accent pixels of a decoded sprite to a neon
accent. This is applied once at atlas build time, so the render path stays
allocation-free.

Parameters:
- `frame` - Decoded sprite frame.
- `accent` - RGBA accent color to apply.

Returns: The same frame reference, with accent pixels recolored.

### resolveBillboardBaseY

```ts
resolveBillboardBaseY(
  x: number,
  z: number,
  halfWidth: number,
  halfDepth: number,
  props: readonly RightWorldProp[],
): number
```

Returns the highest surface a billboard footprint must rest on.

Parameters:
- `x` - World-space x center of the billboard.
- `z` - World-space z center of the billboard.
- `halfWidth` - World-space half-width of the billboard footprint.
- `halfDepth` - World-space half-depth of the billboard footprint.
- `props` - Scenery props to test against.

Returns: Y value of the highest overlapping prop top, or 0 for the ground plane.

### resolveCameraRelativeDirectionIndex

```ts
resolveCameraRelativeDirectionIndex(
  cameraForward: readonly [number, number, number] | undefined,
  cameraWorld: readonly [number, number, number],
  robotPosition: readonly [number, number, number],
  targetPosition: readonly [number, number, number],
): number
```

Resolves the camera-relative 8-direction index of a sprite facing a target.

The conversion is expressed in the neatRacing world compass
(yaw = atan2(dx, dz), 0 rad = +Z = 'front'):

  cameraYaw   = atan2(forwardX, forwardZ)
  facingYaw   = atan2(targetX - robotX, targetZ - robotZ)
  relativeYaw = facingYaw - cameraYaw + PI

The canonical Neatenstein contract
(examples/neatenstein/browser-entry/renderer/sprites.ts:154-156) states the
same contract as `cameraYaw + PI - facingYaw`, but Neatenstein measures yaw
in the OPPOSITE (counter-clockwise, atan2(dy, dx)) compass. Expressed in
the neatRacing compass the same contract is the mirrored form used here,
so the atlas anchor is preserved:

  index 0 ('front') - the sprite FACES the camera
  index 2 ('right') - the sprite nose points to the camera's SCREEN-RIGHT
  index 4 ('back')  - the sprite faces AWAY from the camera
  index 6 ('left')  - the sprite nose points to the camera's SCREEN-LEFT

The neatRacing camera right axis is `(-forwardZ, 0, forwardX)` (the
`lookAtInto` xAxis in examples/neatRacing/browser-entry/browser-entry.ts),
i.e. the forward axis rotated by -90 degrees in the neatRacing compass,
and the public atlas is uploaded without a U flip, so atlas image-right
maps to camera screen-right. A seated spectator therefore has to use the
'right' frame when its nose points to the camera's screen-right.

The helper is allocation-free: it allocates no arrays, objects, or
strings, so it is safe on the per-frame draw path.

Parameters:
- `cameraForward` - Camera forward axis. When omitted, the legacy
five-argument callers fall back to the camera-to-target bearing, which
matches the neatRacing chase rig because it always looks at the car.
- `cameraWorld` - Camera world position, consulted by the legacy
fallback heading only.
- `robotPosition` - Sprite world-space foot position.
- `targetPosition` - World-space position the sprite faces.

Returns: Index in  {@link DIRECTION_ORDER} (0..7).
 *

Example:

```ts
// The camera looks down -Z from [0, 2, 10]. The robot stands at [5, 0, 0]
// and looks at the origin, so its nose points to the camera's screen-left.
const index = resolveCameraRelativeDirectionIndex(
  [0, 0, -1], // camera forward axis
  [0, 2, 10], // camera world position
  [5, 0, 0],  // robot foot position
  [0, 0, 0],  // point the robot faces
);

console.log(index); // 6
console.log(DIRECTION_ORDER[index]); // 'left'
```

### resolveDirectionId

```ts
resolveDirectionId(
  yawDegrees: number,
): string
```

Resolves a world yaw angle in degrees to the nearest 8-direction sprite label.

Parameters:
- `yawDegrees` - World yaw in degrees. 0 faces +Z (front), positive
values rotate toward +X (right).

Returns: One of the eight Neatenstein direction labels.

Example:

```ts
console.log(resolveDirectionId(0)); // 'front'
console.log(resolveDirectionId(90)); // 'right'
console.log(resolveDirectionId(180)); // 'back'
console.log(resolveDirectionId(-45)); // 'frontLeft'
```

### resolvePropHalfExtent

```ts
resolvePropHalfExtent(
  prop: RightWorldProp,
): readonly [number, number]
```

Resolves the world-space half-width and half-depth of a scenery prop.

A bleacher is a stepped voxel structure, so its half size comes from the
real voxel geometry. Every other prop uses the half size of its kind.

Parameters:
- `prop` - Scenery prop of the scene.

Returns: The world-space half-width and half-depth of the prop, in meters.

### resolvePropTopY

```ts
resolvePropTopY(
  prop: RightWorldProp,
  x: number,
  z: number,
): number
```

Resolves the top surface of a scenery prop a billboard can rest on.

Parameters:
- `prop` - Scenery prop of the scene.
- `x` - World-space x of the billboard.
- `z` - World-space z of the billboard.

Returns: The world-space top surface of the prop, in meters.

### resolvePublicRuntimeRole

```ts
resolvePublicRuntimeRole(
  billboard: Pick<PublicBillboard, "runtimeRole">,
): PublicRuntimeRole
```

Resolves the runtime role of a billboard, applying the role-less default.

The atlas frame-index helper and the right-world render pass read the role
through this resolver, so a billboard without an explicit role is never
rendered by one consumer and dropped by the other. A MISSING role is read
as the WALKER role - the only role of the crowd on the walk cycle.

Parameters:
- `billboard` - Billboard (or role carrier) to resolve.

Returns: The explicit role, or  {@link PUBLIC_WALKER_RUNTIME_ROLE} when the
 *   billboard carries none.
 *

Example:

```ts
console.log(resolvePublicRuntimeRole({ runtimeRole: 'seated' })); // 'seated'
console.log(resolvePublicRuntimeRole({})); // 'walker'
```

### resolvePublicSpriteSeed

```ts
resolvePublicSpriteSeed(
  input: TrackSlabBuildInput,
  index: number,
): number
```

Resolves a deterministic unsigned 32-bit seed for the public billboard builder.

Parameters:
- `input` - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
- `index` - Index mixed into the seed, so index streams that start from
the same base seed still draw independent sequences.

Returns: Unsigned 32-bit seed for the local PRNG.

### resolvePublicTeamAccentColorId

```ts
resolvePublicTeamAccentColorId(
  seed: number,
  index: number,
): number
```

Maps the stable robot accent identity into the TEAM accent family.

The mapping is deterministic: the same seed and index always resolve to
the same team-family accent, so the mechanics of a frame keep the same
team-coloured accent across frames. The accent is a member of the TEAM
FAMILY - a hue neighbour of one of the named team colours - and not the
exact colour of a single named team, because the public accent palette
carries no entry for every named team colour.

Parameters:
- `seed` - Deterministic seed of the billboard.
- `index` - Deterministic billboard index.

Returns: The team-family accent id, or the plain robot accent id when the
team family is empty.

### resolvePublicTeamAccentColorIds

```ts
resolvePublicTeamAccentColorIds(): readonly number[]
```

Resolves the palette accent ids that belong to the TEAM accent family.

An accent belongs to the family when its hue bucket is within one bucket
of the hue bucket of one of the team colors.

Returns: The accent ids of the team family, in palette order.

### selectPublicLookAtCarFrameId

```ts
selectPublicLookAtCarFrameId(
  atlas: PublicSpriteAtlas,
  robotPosition: readonly [number, number, number],
  targetPosition: readonly [number, number, number],
  cameraForward: readonly [number, number, number],
): string
```

Returns the 8-direction stand frame id for a robot looking at a target point.

The direction is CAMERA-RELATIVE: the selected frame faces the camera and
matches the frame the render pass recomputes at runtime with
{@link computePublicRuntimeFrameIndex}.

Parameters:
- `atlas` - Public sprite atlas (unused; kept for call-site symmetry).
- `robotPosition` - Robot world-space foot position.
- `targetPosition` - World-space position the robot faces. At build
time this is the camera position, so it doubles as the camera world
position of the legacy fallback heading.
- `cameraForward` - Camera forward axis used as the camera heading.

Returns: Stand frame id, e.g. `'front-stand'`.

Example:

```ts
const atlas = loadPublicSpriteAtlas();

// The camera sits at [0, 2, 10] and looks down -Z. The robot stands at
// [5, 0, 0] and faces the camera, so it is drawn turned towards it.
const frameId = selectPublicLookAtCarFrameId(
  atlas,
  [5, 0, 0],  // robot foot position
  [0, 2, 10], // point the robot faces - the camera position
  [0, 0, -1], // camera forward axis
);

console.log(frameId); // 'frontLeft-stand'
```

### selectPublicSpriteFrame

```ts
selectPublicSpriteFrame(
  atlas: PublicSpriteAtlas,
  seed: number,
  index: number,
): PublicSpriteFrame
```

Selects a deterministic sprite frame from the atlas.

The selection is a seeded draw over the frames of the atlas, so the same
(atlas, seed, index) tuple always picks the same frame. An atlas that
carries no frames has nothing to draw from, and the call returns
`undefined` in that case.

Parameters:
- `atlas` - Public sprite atlas.
- `seed` - Deterministic seed for frame selection.
- `index` - Billboard index used to vary the selected frame.

Returns: The selected frame with positive dimensions, or `undefined` when
the atlas carries no frames.

Example:

```ts
const atlas = loadPublicSpriteAtlas();
const frame = selectPublicSpriteFrame(atlas, 42, 0);
console.log(frame.width > 0); // true
console.log(frame.height > 0); // true
```

### selectPublicWalkFrame

```ts
selectPublicWalkFrame(
  atlas: PublicSpriteAtlas,
  seed: number,
  index: number,
  timeSeconds: number,
): string
```

Selects a deterministic slow two-step walk frame.

The direction comes from a seeded generator keyed by the robot's identity,
so the same robot always walks the same way, and the pose alternates with
the shared walk-cycle period.

Parameters:
- `atlas` - Public sprite atlas (unused; kept for call-site symmetry).
- `seed` - Deterministic seed of the robot.
- `index` - Deterministic index of the robot.
- `timeSeconds` - Monotonic clock time in seconds.

Returns: Walk frame id, e.g. `'front-walk1'`.

Example:

```ts
const atlas = loadPublicSpriteAtlas();

// The first half of the 2.4 s cycle shows walk pose 1, the second half
// walk pose 2.
console.log(selectPublicWalkFrame(atlas, 42, 0, 0)); // 'front-walk1'
console.log(selectPublicWalkFrame(atlas, 42, 0, 3)); // 'front-walk2'
```

### selectPublicWalkPoseIndex

```ts
selectPublicWalkPoseIndex(
  timeSeconds: number,
): number
```

Selects the current walk pose index for a walker at a given clock time.

The cycle is slow (2.4s per pose) and deterministic for a fixed clock.

Parameters:
- `timeSeconds` - Monotonic clock time in seconds.

Returns: 0 for walk1, 1 for walk2.

## track/decoration.lod.ts

Decoration LOD/distance projection for the neatRacing right-world pass
(slice P1-decoration-scale, Step 14D, AC-051).

Decides how many right-world decorations to draw, how they scale with
camera distance, and whether a 2D billboard or voxel fallback should be
used. The P1 decision record keeps 2D sprites as the preferred path and
reserves the voxel branch for future phases.

GREEN TARGET (04-implementing must make exactly these pass):
- Exports {@link NEATRACING_DECORATION_LOD_BUDGET} as the per-frame cap.
- Exports {@link computeDecorationScale} returning a larger positive scale
  for near distances and a smaller positive scale for far distances.
- Exports {@link resolveDecorationRenderMode} returning '2d' for the P1
  decision at all distances, and 'voxel' only when an explicit fallback
  threshold is supplied and the distance exceeds it.
- Exports {@link computeDecorationLod} exposing visibility, scale, and mode.
- Exports {@link selectDecorationsForCamera} culling distant decorations and
  keeping only the nearest ones up to the budget.

### computeDecorationLod

```ts
computeDecorationLod(
  distance: number,
  decision: DecorationRenderDecision,
): DecorationLod
```

Computes the full LOD record for a decoration at a given distance.

Parameters:
- `distance` - Distance from the camera in meters.
- `decision` - Render decision; defaults to the P1 2D-preferred record.

Returns: LOD tier with visibility, scale, and mode.

### computeDecorationScale

```ts
computeDecorationScale(
  distance: number,
): number
```

Computes a distance-aware scale for a decoration.

The anchor is the near reference distance, where the returned scale is 1.0.
Distant decorations shrink inversely with distance and are floored at the
configured minimum so they remain visible rather than disappearing.

Parameters:
- `distance` - Distance from the camera in meters.

Returns: Positive scale that is larger near the camera and smaller far away.

### createDecorationSelectionScratch

```ts
createDecorationSelectionScratch(
  capacity: number,
): DecorationSelectionScratch
```

Creates a preallocated scratch buffer for zero-allocation per-frame
decoration selection.

Parameters:
- `capacity` - Maximum number of decorations that will be selected.

Returns: Scratch buffer ready for  {@link selectVisibleDecorationsForCamera} .

### DecorationLod

LOD tier returned for a decoration at a given distance.

### DecorationRenderDecision

Decision record documenting the 2D-preferred / voxel-fallback choice.

### DecorationSelectionScratch

Preallocated scratch buffer used by
{@link selectVisibleDecorationsForCamera}. Reusing one scratch object per
pass keeps the steady-state path free of per-frame JavaScript allocations.

### NEATRACING_DECORATION_CULL_DISTANCE_M

Distance beyond which decorations are culled as too far to matter.

### NEATRACING_DECORATION_LOD_BUDGET

Maximum number of visible right-world decorations drawn per frame.

### NEATRACING_DECORATION_MIN_SCALE

Floor value so distant decorations remain visible instead of vanishing.

### NEATRACING_DECORATION_NEAR_REFERENCE_M

Reference distance where decoration scale equals 1.0.

### NEATRACING_DECORATION_P1_DECISION

P1 decision record: 2D billboards are preferred for all visible right-world
decorations. The voxel fallback branch is explicit but disabled in P1; it
is reserved for future phases where 2D sprites might be unsuitable.

### PositionedDecoration

Decoration with a world-space position.

### resolveDecorationRenderMode

```ts
resolveDecorationRenderMode(
  distance: number,
  decision: DecorationRenderDecision,
): "2d" | "voxel"
```

Resolves the 2D-vs-voxel render mode for a decoration at the given distance.

P1 always prefers 2D. When an explicit voxel fallback threshold is supplied,
distances strictly greater than the threshold switch to the voxel branch;
all other distances stay in 2D.

Parameters:
- `distance` - Distance from the camera in meters.
- `decision` - Render decision; defaults to the P1 2D-preferred record.

Returns: '2d' or 'voxel' depending on the decision threshold.

### selectDecorationsForCamera

```ts
selectDecorationsForCamera(
  decorations: readonly T[],
  cameraPosition: readonly [number, number, number],
  budget: number,
): readonly T[]
```

Sorts decorations by distance from the camera and culls those beyond the
budget or the far cull distance.

The returned array is front-to-back (nearest first) and capped at the
budget. It is built without mutating the input list.

This helper is convenient for scene-build or test paths; the per-frame
render path uses {@link selectVisibleDecorationsForCamera} with a
preallocated scratch buffer to preserve the zero per-frame allocation
contract (P9).

Parameters:
- `decorations` - Unsorted candidate decorations.
- `cameraPosition` - Camera world position in meters.
- `budget` - Maximum decorations to keep; defaults to the LOD budget.

Returns: Nearest decorations, front-to-back, capped at the budget.

### selectVisibleDecorationsForCamera

```ts
selectVisibleDecorationsForCamera(
  decorations: readonly T[],
  cameraPosition: readonly [number, number, number],
  cullDistanceM: number,
  budget: number,
  scratch: DecorationSelectionScratch,
): number
```

Fills the scratch buffer with decorations visible from the runtime camera,
sorted nearest-first, without allocating any new objects.

The returned count is the number of valid entries in the scratch arrays.
Callers can iterate `scratch.indices[0..count)` for near-to-far drawing, or
iterate in reverse for far-to-near translucent sorting.

Parameters:
- `decorations` - Candidate decorations.
- `cameraPosition` - Runtime camera world position in meters.
- `cullDistanceM` - Distance beyond which decorations are dropped.
- `budget` - Maximum decorations to keep.
- `scratch` - Preallocated scratch buffer.

Returns: Number of visible decorations written to the scratch buffer.

## track/right-world.pass.ts

Right-world scenery render pass for the neatRacing example.

Draws the TRON-style neon scenery props generated by
{@link buildRightWorldProps} after the track slab pass so they sit on top of
the ground plane and behind the car glass pass, then overlays the recycled
Neatenstein public/agent billboard sprites produced by
{@link buildPublicBillboards}.

WHAT THE PASS IS RESPONSIBLE FOR:

- GEOMETRY. Neon props are solid boxes; a bleacher contributes its stepped
  voxel structure instead of a single box, so the crowd really stands in
  the stands.
- TEXTURES. A single packed sprite atlas is uploaded once and shared by
  every billboard, so the upload loop stays short.
- LIVE BILLBOARDS. The billboard `quad` fields carried by the sprite
  helpers are expressed in projection-space units (focal * dot /
  distance), not screen pixels. The pass IGNORES those precomputed quads
  and rebuilds world-space camera-facing billboard geometry each frame from
  the chase camera, so the unit system stays consistent with the WebGL
  vertex path.
- A STEADY-STATE PATH WITHOUT GARBAGE. The pass compiles two WebGL2
  programs (solid props + textured billboards) and draws camera-facing
  billboards each frame without allocating new JavaScript objects. All
  mutable draw state is refreshed in-place from the frame supplied by the
  host.

The sprite atlas is PUBLIC-ONLY (144 frames = 6 accents x 8 directions x 3
poses) and the pass draws the FOUR crowd roles of the re-populated world:
the seated spectators on the bleacher seats, the standing spectators outside
the ring, the walkers on the trackside and the inner-ring mechanics.

The sprite module exports a scale helper that IGNORES the distance: it
returns the NAMED half world scale
`NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE` (0.5) for every billboard,
because the right-world pass applies the shared `focal / distance`
perspective projection itself. The pass uses that public helper
`computeBillboardScale` for all billboards to keep the upload loop single
and avoid duplicating the formula, and multiplies its result with the
SHARED meters-per-pixel export `NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M`
(0.05 m/px), so the placement footprint and the rendered quad can never
disagree about the size of a sprite pixel.

### buildAtlasIndexTables

```ts
buildAtlasIndexTables(
  atlas: RightWorldSpriteAtlas,
): { readonly publicTable: Int16Array<ArrayBufferLike>; }
```

Builds the direction×pose atlas-index lookup table used by the runtime
frame-selection helper. Public table entries are atlas indices.

Parameters:
- `atlas` - Public sprite atlas.

Returns: Public lookup table.

### buildAtlasTextureAndUvs

```ts
buildAtlasTextureAndUvs(
  gl: WebGL2RenderingContext,
  atlas: RightWorldSpriteAtlas,
): { readonly texture: WebGLTexture; readonly uvs: readonly SpriteFrameUv[]; }
```

Packs the public sprite atlas frames into a single RGBA texture and computes
per-frame UV rectangles. The layout is a square-ish 2D grid, and the packer
checks the resulting width and height against the GPU's
`gl.MAX_TEXTURE_SIZE` before uploading anything. The packer throws
BEFORE any `gl.texImage2D` call when the grid would exceed the GPU limit.

Parameters:
- `gl` - WebGL2 context to allocate the texture on.
- `atlas` - Public sprite atlas frames.

Returns: GPU texture handle and per-frame UV table.

### buildBillboardUpload

```ts
buildBillboardUpload(
  target: Float32Array<ArrayBufferLike>,
  billboards: readonly PublicBillboard[],
  atlasFrames: readonly PublicSpriteFrame[],
  uvs: readonly SpriteFrameUv[],
  billboardCount: number,
  billboardScratch: DecorationSelectionScratch,
  billboardFrameScratch: Int16Array<ArrayBufferLike>,
  cameraRight: readonly [number, number, number],
  cameraUp: readonly [number, number, number],
): number
```

Fills the preallocated billboard upload buffer with camera-facing quads and
returns the number of vertices to draw. Missing runtime frame indices are
skipped.

Parameters:
- `target` - Preallocated Float32Array for position+uv vertices.
- `billboards` - Billboard instances.
- `atlasFrames` - Original atlas frames for pixel dimensions.
- `uvs` - Per-frame UV rectangles inside the packed atlas.
- `billboardCount` - Number of visible billboards already selected into `billboardScratch` (nearest-first).
- `billboardScratch` - Preallocated selection scratch with indices/distances sorted nearest-first.
- `billboardFrameScratch` - Preallocated runtime atlas frame index per selected slot.
- `cameraRight` - World-space camera right unit vector.
- `cameraUp` - World-space camera up unit vector.

Returns: Total number of vertices to draw.

### buildRightWorldSceneData

```ts
buildRightWorldSceneData(
  input: TrackSlabBuildInput,
  camera: PublicCamera,
): RightWorldSceneData
```

Builds the deterministic CPU-side scene data once: TRON props plus the
recycled public/agent billboard list. The billboard screen-space quads are
ignored; only world positions and frame ids are retained by the pass. The
atlas is PUBLIC-ONLY: 144 frames (6 accents x 8 directions x 3 poses) are
packed into the single texture the billboard draw path samples.

The assembled crowd keeps ALL FOUR roles of the re-populated world: the
seated spectators on the bleacher seats, the standing spectators outside the
ring, the walkers on the trackside and the inner-ring mechanics. The
billboard pipeline has TWO steps: the anchoring-capacity filter drops the
hosted billboards whose host prop would carry more robots than its top
area affords (the props themselves are never thinned), and the
personal-space filter keeps the 1 m XZ spacing of the crowd. The crowd
arrives ALREADY personal-spaced from `buildPublicBillboards`, so the
personal-space pass here is an idempotent safety net: the anchoring-capacity
filter only DROPS billboards (it never re-hosts or moves them), so dropping
never relaxes a separation and the pass is a no-op on an already
personal-spaced crowd. Both passes keep the FIRST billboard of every band,
so a bleacher seat is never starved.

The default right-world camera position is wired through
`buildPublicBillboards` so the 8-direction sprite frame selection uses the
same heading everywhere.

Parameters:
- `input` - Determinism tuple for the track layout. Defaults to the
shared right-world tuple.
- `camera` - Camera used to place billboards and to pick their
8-direction frame. Defaults to the shared right-world camera.

Returns: Scene data ready to upload to the GPU pass.

Example:

```ts
const scene = buildRightWorldSceneData();

console.log(scene.props.length > 0); // true
console.log(scene.atlas.frames.length); // 144
console.log(scene.billboards.length > 0); // true
```

### compileRightWorldShader

```ts
compileRightWorldShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader
```

Compiles one GLSL ES 3.0 shader and throws a descriptive message on failure.

Parameters:
- `gl` - WebGL2 context.
- `type` - Shader type enum.
- `source` - Shader source.
- `label` - Debug label for error messages.

Returns: Compiled shader handle.

### createRightWorldPass

```ts
createRightWorldPass(
  gl: WebGL2RenderingContext,
): RightWorldPass
```

Creates the right-world scenery draw pass: compiles shaders, builds the
deterministic CPU-side scene data, packs the public sprite atlas, and
preallocates dynamic prop and billboard VBOs. The returned handle selects
visible props each frame against the runtime camera (opaque, depth-write)
and then draws blended billboards far-to-near (depth-read, no depth-write)
without allocating JavaScript objects on the steady-state path.

Parameters:
- `gl` - WebGL2 rendering context.

Returns: The right-world pass handle.

Example:

```ts
// The pass owns its own GPU resources, so it is created once per context
// and disposed when the context goes away.
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2');
if (!gl) {
  throw new Error('WebGL2 is required for the right-world pass');
}

const pass = createRightWorldPass(gl);

pass.draw({
  // Column-major view-projection matrix; the host supplies the real one
  // every frame.
  viewProjection: new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]),
  cameraWorld: [0, 2, 20],
  cameraRight: [1, 0, 0],
  cameraUp: [0, 1, 0],
  timeSeconds: 0,
});

pass.dispose();
```

### deriveCameraForwardFromBasis

```ts
deriveCameraForwardFromBasis(
  cameraUp: readonly [number, number, number],
  cameraRight: readonly [number, number, number],
  out: [number, number, number],
): [number, number, number]
```

Derives the camera forward axis from the draw-state basis vectors.

The browser entry feeds `cameraUp = cameraRight x cameraForward`
(examples/neatRacing/browser-entry/browser-entry.ts), so the exact inverse
is `cameraForward = cameraUp x cameraRight`. Deriving the axis here keeps
{@link RightWorldPassDrawState} free of a new field (no extra construction
site to maintain) and stays allocation-free.

The result is a direction, so its length is irrelevant to the yaw the
sprite conversion reads (`atan2(forwardX, forwardZ)` is scale-invariant).

Parameters:
- `cameraUp` - World-space camera up basis vector.
- `cameraRight` - World-space camera right basis vector.
- `out` - Scratch tuple that receives the derived forward axis.

Returns: The same `out` tuple, so the call site can read it inline.

### emitBleacherVoxelBox

```ts
emitBleacherVoxelBox(
  target: Float32Array<ArrayBufferLike>,
  offset: number,
  voxel: BleacherVoxel,
): number
```

Writes one bleacher voxel as a small role-tagged box. The box top is aligned
to the voxel's reported y so seat rows, walk surface, stairs, and guardrail
render at their real heights.

Parameters:
- `target` - Preallocated Float32Array to write into.
- `offset` - Float offset into `target`.
- `voxel` - Bleacher voxel to emit.

Returns: The number of floats written.

### emitBox

```ts
emitBox(
  target: Float32Array<ArrayBufferLike>,
  offset: number,
  cx: number,
  cy: number,
  cz: number,
  hx: number,
  hy: number,
  hz: number,
  r: number,
  g: number,
  b: number,
): number
```

Writes one axis-aligned box into an interleaved position+color VBO target.
No JavaScript arrays are allocated on the hot path; all vertex coordinates
are written directly into `target`.

Parameters:
- `target` - Preallocated Float32Array to write into.
- `offset` - Float offset into `target`.
- `cx` - Box center x.
- `cy` - Box center y.
- `cz` - Box center z.
- `hx` - Half width.
- `hy` - Half height.
- `hz` - Half depth.
- `r` - Red color component.
- `g` - Green color component.
- `b` - Blue color component.

Returns: The number of floats written.

### emitPropBox

```ts
emitPropBox(
  target: Float32Array<ArrayBufferLike>,
  offset: number,
  prop: RightWorldProp,
): number
```

Writes a single non-bleacher prop as one axis-aligned box.

Parameters:
- `target` - Preallocated Float32Array to write into.
- `offset` - Float offset into `target`.
- `prop` - Scenery prop to emit.

Returns: The number of floats written.

### filterBillboardsByAnchoringCapacity

```ts
filterBillboardsByAnchoringCapacity(
  billboards: readonly PublicBillboard[],
  props: readonly RightWorldProp[],
): readonly PublicBillboard[]
```

Drops any hosted billboard whose host prop would carry more robots than
its top area affords at 1 m spacing. The PROPS themselves are never
thinned: the prop list and the MAX_PROPS LOD are untouched, only the
hosted billboards are dropped. Hosting is restricted to bleacher props.

### findHostProp

```ts
findHostProp(
  billboard: PublicBillboard,
  props: readonly RightWorldProp[],
): RightWorldProp | undefined
```

Finds the prop whose top surface hosts a billboard foot anchor. Only
bleacher props are considered hosts; the bleacher footprint and seat-top
surface come from the real voxel geometry.

### linkRightWorldProgram

```ts
linkRightWorldProgram(
  gl: WebGL2RenderingContext,
  vertexShader: WebGLShader,
  fragmentShader: WebGLShader,
  label: string,
): WebGLProgram
```

Links a program from a vertex/fragment shader pair and validates link status.

Parameters:
- `gl` - WebGL2 context.
- `vertexShader` - Compiled vertex shader.
- `fragmentShader` - Compiled fragment shader.
- `label` - Debug label for error messages.

Returns: Linked program handle.

### NEATRACING_BILLBOARD_FRAME_BUDGET

Per-frame capacity of the billboard draw path.

The budget is the PER-FRAME CAPACITY of the re-populated crowd: it is sized
from the public crowd density budget of the sprite module (240 seated
spectators + 34 off-bleacher agents = 274 quads), so the whole crowd fits
into one upload buffer and one draw call.

The budget is INDEPENDENT of the voxel-prop LOD budget: the prop budget
still governs the opaque scenery through {@link MAX_PROPS}, but the
billboard capacity is derived from the crowd density instead of from the
prop budget. The two budgets therefore never shrink the crowd.

The per-frame selection is NOT reduced by the visible prop count: the
billboard budget is a constant, so a frame that shows many props does not
silently truncate the crowd that is inside the cull distance.

Example:

```ts
// 240 seated billboards (20 bleachers x 12 seats) + 34 off-bleacher agents
console.log(NEATRACING_BILLBOARD_FRAME_BUDGET); // 274
```

### propTopArea

```ts
propTopArea(
  prop: RightWorldProp,
): number
```

Horizontal top surface area of a prop in square meters.

### RightWorldPass

Handle to the compiled right-world scenery draw pass.

The handle owns the GPU resources of the pass: the two compiled programs
and their shaders, the prop and billboard buffers, and the packed sprite
atlas texture. `draw` renders one frame from the camera state the host
supplies, and `dispose` releases every GPU object the pass created.

Example:

```ts
const canvas = document.querySelector('canvas') as HTMLCanvasElement;
const gl = canvas.getContext('webgl2');
if (!gl) {
  throw new Error('WebGL2 is required for the right-world pass');
}

const pass: RightWorldPass = createRightWorldPass(gl);

// The host owns the frame: the pass reads the camera basis and the
// view-projection matrix, and it allocates nothing per frame.
pass.draw({
  viewProjection: new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
  ]),
  cameraWorld: [0, 2, 20],
  cameraRight: [1, 0, 0],
  cameraUp: [0, 1, 0],
  timeSeconds: 0,
});

pass.dispose();
```

### RightWorldPassDrawState

Mutable frame draw state forwarded to {@link RightWorldPass.draw}.

The state carries the camera BASIS, not a camera forward axis: the forward
axis is DERIVED as `cameraUp x cameraRight` by
{@link deriveCameraForwardFromBasis}. That derivation is the exact inverse
of the browser-entry basis (`cameraUp = cameraRight x cameraForward`), so
the pass needs no extra field and no extra construction site.

The host OWNS the frame. The pass only READS the state, and the
steady-state path expects the SAME object to be refreshed in place, so no
per-frame object is allocated.

Example:

```ts
// One draw-state bag per canvas, refreshed in place for every frame.
const drawState: RightWorldPassDrawState = {
  viewProjection: new Float32Array(16),
  cameraWorld: [0, 2, 20],
  cameraRight: [1, 0, 0],
  cameraUp: [0, 1, 0],
  timeSeconds: 0,
};

// Every frame: overwrite the basis and the clock, never replace the bag.
drawState.cameraRight[0] = 0.7071;
drawState.cameraRight[2] = 0.7071;
drawState.timeSeconds += 1 / 60;

// `pass` is the handle returned by createRightWorldPass(gl).
pass.draw(drawState);
```

### RightWorldSceneData

CPU-side scene data consumed by the right-world pass.

This is the CPU half of the scenery boundary: the solid neon props the
opaque program draws, the packed PUBLIC-ONLY sprite atlas the billboard
program samples, and the world-space billboard list that says WHERE the
crowd stands. {@link buildRightWorldSceneData} builds the payload ONCE and
the pass treats it as immutable afterwards, so the per-frame path never
rebuilds the world.

The atlas is PUBLIC-ONLY: 144 frames (6 accents x 8 directions x 3 poses)
are packed into one texture, and every crowd role is drawn from it. The
atlas field type is module-private, so read it through its `source` and
`frames` fields instead of importing the type.

The billboard list carries WORLD positions and frame hints only. The
screen-space quads that the sprite helpers precompute are IGNORED by the
pass, which rebuilds camera-facing world-space quads every frame from the
chase camera.

Example:

```ts
const scene = buildRightWorldSceneData();

// 50 solid neon props, 20 of them spectator bleachers.
console.log(scene.props.length); // 50
console.log(
  scene.props.filter((prop) => prop.kind === 'bleacher').length,
); // 20

// The packed atlas is public-only: 144 frames in one texture.
console.log(scene.atlas.source); // 'neatenstein-public'
console.log(scene.atlas.frames.length); // 144

// The crowd: 240 seated spectators + 34 off-bleacher agents.
console.log(scene.billboards.length); // 274
```

### RightWorldSpriteAtlas

Sprite atlas shape of the public sprite source.

### SpriteFrameUv

UV rectangle for one sprite frame inside the packed atlas.

## track/track.racing-line.ts

Deterministic, curvature-derived racing line for the Neat Racing example
(plan slice `P1-ai-racing-line`; Step 14M; the AC-082 and AC-086
contracts).

THE SURFACE (AC-086): the module publishes ONE point per spline sample -
the signed lateral offset from the road centerline, the signed curvature
profile and the curvature-derived target speed - as a stable
TRACK-RELATIVE table the Phase 2 observation assembler (AC-027) and the
scripted controller (AC-028) can read WITHOUT a second source of truth:
the table is built from the SAME frozen spline samples the autopilot
already reads (`examples/neatRacing/physics/ai.autopilot.ts`), so there
is NO second geometry anywhere in the repo.

THE GEOMETRY: the signed curvature is the Menger curvature of the
three-sample window around each sample of the closed ring - POSITIVE
when the loop turns LEFT, the repo's left-normal convention - and it is
smoothed over a SYMMETRIC circular window, so the published profile is
free of the single-sample jitter the raw three-point estimate carries
while staying exactly invariant under a rigid rotation of the layout
(the rotation negates and swaps the coordinate differences, so the
cross product and the three side lengths are bit-identical).

THE OFFSET: the lateral offset is proportional to the EXCESS curvature
above the straight threshold, so the line is EXACTLY ZERO on a straight
and leans toward the INSIDE of a corner - the inside of a
counter-clockwise corner is to the LEFT, i.e. a POSITIVE offset - with
the peak at the apex, where the curvature peaks.

THE CLAMP (Step 14V, AC-095): the offset is clamped against the SEAM's
ASYMMETRIC half-width PAIR - the OUTER half-width (the inner
half-width plus the corner-measured drift space) on the OUTER side of
the loop and the INNER half-width on the INNER side - so the AI may use
the drift room on the OUTSIDE of a corner and NEVER on the inside. The
RETIRED symmetric `sample.width / 2` clamp is DELETED (no shim, no
deferred cleanup): the road never narrows, so the inner bound is the
half-width the seam publishes and the outer bound is that half-width
plus the drift space the corner measure earns.

THE SEAM (Step 14V, AC-095): the clamp reads the SAME edge the road
reads - the seam published by `examples/neatRacing/track/track.edge.ts`,
the edge the rasterizer, the band paint and the containment bound all
read - so there is NO second edge anywhere in the repo. The racing line
resolves that seam through the closed-loop chord ring the frozen
samples describe: the CONTROL ring when the samples name their control
segments (the production layout - the reconstruction is BIT-IDENTICAL to
the spec's own chord list, because the first sample of a control
segment sits exactly on its control point), and the samples' OWN ring
when they name a single segment (a sample-only fixture, whose own ring
is the only ring it describes).

THE SPEED TARGET: the cornering limit `v = sqrt(a_lat / |k|)` capped at
the physics top speed, so a straight publishes the top speed and a tight
corner publishes a slower one. The lateral limit is derived from the top
speed and the straight threshold, so the two meet exactly: the flattest
corner the line still reacts to is drivable AT the top speed.

THE PROFILE: the offset rises from ZERO at the corner entry to the apex
peak and falls back to ZERO at the corner exit - INSIDE through the
corner and back on the centerline after it. The classic out-in-out line
(a NEGATIVE offset on entry and exit) is deliberately NOT built: the
RED contracts pin only the apex-inside clause, and a negative exit
offset would fight the AC-085 centerline bound and the zero-on-straights
clause.

DETERMINISM: the table is a pure function of the frozen sample list - no
RNG, no clock reads, no shared state - so two builds of the same layout
are identical and the line is SEED-INVARIANT for a fixed layout.

ALLOCATION: the table is built ONCE per track layout and cached by the
caller; the per-frame AI path reads the published table and allocates
nothing.

### buildControlChordRing

```ts
buildControlChordRing(
  samples: readonly SplineSample[],
): RacingLineSeamRing
```

Builds the CONTROL chord ring the samples describe.

THE RECONSTRUCTION IS BIT-IDENTICAL to the spec's own chord list: the
first sample of a control segment carries an interpolation factor of
ZERO, so it sits EXACTLY on that segment's control point, and the
control chord of segment `k` runs from control point `k` to control
point `k + 1` with the width that segment publishes.

Parameters:
- `samples` - The frozen closed-ring spline samples.

Returns: The control chord ring and the samples that name it.

### buildSampleChordRing

```ts
buildSampleChordRing(
  samples: readonly SplineSample[],
): RacingLineSeamRing
```

Builds the SAMPLE chord ring: one chord per consecutive sample pair,
each sample owning the chord that STARTS at it.

A fixture whose samples name a SINGLE control segment describes no
control ring of its own, so the samples' OWN ring is the only ring it
describes and the only ring the seam can be read through.

Parameters:
- `samples` - The frozen closed-ring spline samples.

Returns: The sample chord ring and the samples that name it.

### buildTrackRacingLine

```ts
buildTrackRacingLine(
  samples: readonly SplineSample[],
): readonly TrackRacingLinePoint[]
```

Builds the deterministic racing-line table of a closed spline ring.

THE CONTRACT (AC-082): the table is a pure function of the frozen
spline samples - ONE point per sample, in the sample order - with a
signed lateral offset that is apex-biased toward the inside of a
corner, EXACTLY zero on a straight, clamped to the SEAM's half-width
PAIR (Step 14V, AC-095) and invariant for a fixed track layout.

THE SURFACE (AC-086): the same table publishes the signed curvature
profile and the curvature-derived target speed, so the Phase 2
observation assembler and the scripted controller read ONE track-relative
surface instead of a second source of truth.

Parameters:
- `samples` - The frozen closed-ring spline samples of the track.

Returns: The racing-line points, one per sample, in the sample order.

Example:

const line = buildTrackRacingLine(trackSpec.splineSamples);
// line[0].offsetM            -> the signed lateral offset (m)
// line[0].curvaturePerM      -> the signed curvature (1/m)
// line[0].targetSpeedUnitsPerS -> the curvature target speed (m/s)

### clampOffsetToSeamPair

```ts
clampOffsetToSeamPair(
  offsetM: number,
  edge: TrackSplineEdge,
  outerSideSign: 1 | -1,
): number
```

Clamps a signed lateral offset into the SEAM's half-width PAIR.

THE PAIR, NOT THE SYMMETRIC HALF-WIDTH: the OUTER half-width (the
inner half-width plus the corner-measured drift space) bounds the
offset on the OUTER side of the loop and the INNER half-width bounds
it on the INNER side, so the AI may use the drift room on the outside
of a corner and NEVER on the inside.

THE SIDE IS LOOP-GLOBAL: the side is resolved from the seam's own
loop-global outer-side sign (the winding of the WHOLE loop), exactly
like the seam's own side resolver, so the bound NEVER flips at a
straight.

Parameters:
- `offsetM` - The unclamped signed offset (m).
- `edge` - The published half-width pair of the sample.
- `outerSideSign` - The loop-global sign of the outer side.

Returns: The clamped signed offset (m).

### excessCurvaturePerM

```ts
excessCurvaturePerM(
  curvaturePerM: number,
): number
```

Resolves the curvature a corner actually bites with: the curvature
ABOVE the straight threshold, keeping the sign of the profile.

Parameters:
- `curvaturePerM` - The signed curvature profile value (1/m).

Returns: The excess curvature (1/m); zero on a straight.

### NEATRACING_RACING_LINE_APEX_BIAS_GAIN_M2

The apex-bias gain of the lateral offset (m^2).

The offset is `gain x the excess curvature`, so the gain is the lateral
offset the line takes per unit of curvature: a TIGHTER corner is cut
deeper. The value is calibrated so the tight R = 8 m corner of the
AI-center-keeping fixtures takes a 1.92 m apex offset - inside the 6 m
drivable half-width and inside the 3 m AC-085 centerline bound - while
the R = 60 m lap fixture takes a 0.19 m offset.

### NEATRACING_RACING_LINE_CURVATURE_WINDOW_RADIUS_SAMPLES

The radius of the symmetric curvature-smoothing window (samples).

The window is a circular moving average over the closed ring, so the
published profile is smooth AND rotation- and reversal-equivariant. The
radius is deliberately small: a wide window would bleed the corner
curvature back onto the straight and break the straight-zero clause.

### NEATRACING_RACING_LINE_LATERAL_ACCELERATION_LIMIT_M_PER_S2

The lateral acceleration limit the published speed target respects
(m/s^2).

DERIVED, not tuned: `top speed^2 / the straight corner radius`, so the
flattest corner the line reacts to is exactly drivable AT the top speed
and every tighter corner publishes a proportionally slower target.

### NEATRACING_RACING_LINE_STRAIGHT_CORNER_RADIUS_M

The radius of the FLATTEST corner the racing line still treats as a
corner (m).

A curve flatter than this is a straight for the racing line: the
published offset is exactly zero there and the published speed target is
the top speed. The threshold is the corner radius the top speed can
still hold, so the lateral limit and the top speed meet exactly at the
threshold and the speed profile has no step at the straight/corner
boundary.

### NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M

The curvature at or below which the centerline counts as a STRAIGHT
(1/m) - the reciprocal of
{@link NEATRACING_RACING_LINE_STRAIGHT_CORNER_RADIUS_M}.

The straight zero is a GEOMETRIC property, not a fixture artifact: a
generated track's nominal straight is a sampled curve, so its curvature
is small but not exactly zero. The threshold is what makes the AC-082
"exactly zero on straights" clause hold on a real layout.

### RacingLineSeam

The SEAM the racing line is clamped by: the published half-width pair
of every sample and the loop-global outer-side sign.

### RacingLineSeamRing

The closed-loop chord ring the racing line reads the SEAM through.

THE RING IS RECONSTRUCTED FROM THE SAMPLES: the racing line is a pure
function of the frozen sample list (AC-086), so it resolves the seam
from the chord ring those samples describe - never from a second
geometry.

### resolveRacingLineChordRing

```ts
resolveRacingLineChordRing(
  samples: readonly SplineSample[],
): RacingLineSeamRing
```

Resolves the chord ring the seam is read through.

THE CONTROL RING WHEN THE SAMPLES NAME THEIR CONTROL SEGMENTS: the
production layout, whose reconstruction is bit-identical to the spec's
own chord list. THE SAMPLES' OWN RING OTHERWISE: a fixture whose
samples name a single segment describes no control ring of its own.

Parameters:
- `samples` - The frozen closed-ring spline samples.

Returns: The chord ring the seam profile is resolved from.

### resolveRacingLineSeam

```ts
resolveRacingLineSeam(
  samples: readonly SplineSample[],
): RacingLineSeam
```

Resolves the SEAM the racing line is clamped by.

THE ONE EDGE: the profile is resolved from the chord ring the samples
describe and the per-sample edges from THAT profile, so the racing
line is bounded by the SAME edge the rasterizer, the band paint and
the containment bound read.

Parameters:
- `samples` - The frozen closed-ring spline samples.

Returns: The per-sample half-width pairs and the outer-side sign.

### signedCurvaturePerM

```ts
signedCurvaturePerM(
  samples: readonly SplineSample[],
  index: number,
): number
```

Computes the signed Menger curvature of the closed sample ring at one
sample.

THE SIGN: the Menger curvature of the three-sample triangle is POSITIVE
when the triangle turns counter-clockwise, i.e. when the loop turns
LEFT - the repo's left-normal convention.

THE ROTATION INVARIANCE: the side lengths are computed as
`sqrt(dx * dx + dy * dy)` and NOT with `Math.hypot`, because the
explicit form is bit-identical when a rigid rotation negates and swaps
the coordinate differences (IEEE-754 addition and multiplication are
commutative, so the cross product and the three side lengths are
bit-identical under a rotation).

Parameters:
- `samples` - The frozen closed-ring spline samples.
- `index` - The index of the sample to measure.

Returns: The signed curvature at the sample (1/m); zero when the three
samples are collinear or coincident.

### smoothCurvatureProfile

```ts
smoothCurvatureProfile(
  rawCurvaturePerM: Float64Array<ArrayBufferLike>,
  radius: number,
): Float64Array<ArrayBufferLike>
```

Smooths the raw curvature profile over a symmetric circular window.

THE WINDOW: a moving average over `2 x radius + 1` samples of the
closed ring, so the profile is smooth at the corner entry and exit and
the smoothing stays rotation- and reversal-equivariant (the window is
symmetric and the ring is closed).

Parameters:
- `rawCurvaturePerM` - The raw signed curvature profile (1/m).
- `radius` - The window radius in samples.

Returns: The smoothed signed curvature profile (1/m).

### targetSpeedUnitsPerS

```ts
targetSpeedUnitsPerS(
  curvaturePerM: number,
): number
```

Resolves the curvature-derived target speed at one sample.

Parameters:
- `curvaturePerM` - The signed curvature profile value (1/m).

Returns: The target speed (m/s); the top speed on a straight.

### TrackRacingLinePoint

One published point of the racing-line table.

ONE point per spline sample, in the sample order.

## track/right-world.builder.ts

Deterministic TRON-style neon scenery prop builder for the neatRacing
example.

The builder turns one determinism tuple - `{ seed, layoutVersion,
sizeBucket }` - into an ordered list of neon props that line the
right-hand side of the track: pillars, gantries, signs, and stepped
spectator bleachers. Every prop stands at least
{@link NEATRACING_PROP_CLEARANCE_FLOOR_M} beyond the SEAM EDGE, so the
driving line is never obstructed, and the total count stays inside a
density budget the render pass can afford.

Four properties make the output usable by a renderer and by the crowd
builder that reads it:

- DETERMINISM. The same tuple always produces the same prop list, byte for
  byte, on the same runtime. The builder owns its own seeded PRNG, so it
  never depends on ambient randomness, and every later prop stays exactly
  where it was when an earlier rule changed.
- A GUARANTEED CROWD. The outer ring reserves EXACTLY
  {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT} bleacher slots BY
  CONSTRUCTION at evenly spaced slot indices, so every scene offers the
  same number of spectator stands whatever the seed and whatever the track
  size bucket is. The bleacher slots are reserved BY CONSTRUCTION rather
  than drawn at random, and the reserved slots are spread far enough
  apart that no two bleachers overlap in world space.
- A PIT BAND. The inner ring carries the mechanic, crew and team-banner
  props of the re-populated right world. The pit band sits closer to the
  road than the bleachers and at the midpoints between the bleacher
  blocks, so the two rings never share a track position.
- A SEAM-RELATIVE PLACEMENT. Every clearance is MEASURED against the
  SEAM EDGE - the published edge profile of the track, read through the
  road's own union reader - and NEVER against the retired
  spline-normal distance model, so the props and the road can never
  disagree about where the edge is.

A bleacher is a stepped voxel structure rather than a box: seats, a walking
surface, stairs, and a track-facing guardrail, all expressed in the real
world-space geometry that the crowd builder and the renderer both read.

### BLEACHER_VOXEL_HALF_EXTENT_M

World-space horizontal half-extent used when drawing a bleacher voxel box.

The pair of voxel half-extents - this one and
{@link BLEACHER_VOXEL_HALF_HEIGHT_M} - turns the unit cell of a bleacher
voxel into a world-space box: the horizontal half-extent is the half of the
1 m cell width, and the vertical half-height is the half of the 0.2 m cell
height.

Example:

```ts
console.log(BLEACHER_VOXEL_HALF_EXTENT_M * 2); // 1 - one voxel wide
console.log(BLEACHER_VOXEL_HALF_HEIGHT_M * 2); // 0.2 - one voxel tall
```

### BLEACHER_VOXEL_HALF_HEIGHT_M

World-space vertical half-height used when drawing a bleacher voxel box.

The half-height turns the unit cell of a bleacher voxel into a world-space
box of 0.2 m, so the seat surface of a stand is the voxel height plus this
half-height.

Example:

```ts
console.log(BLEACHER_VOXEL_HALF_HEIGHT_M * 2); // 0.2 - one voxel tall
```

### BleacherGeometry

Geometry carried by a bleacher prop.

The geometry is the STEP of the stand: seats, walking surface, stairs, and
guardrails as individual voxels. A bleacher without geometry is a plain
prop, so the geometry-aware helpers fall back to their documented default
shapes for it.

Example:

```ts
const geometry: BleacherGeometry = {
  voxels: [
    {
      position: [0, 0.8, 0],
      color: [0, 255, 255],
      role: 'seat',
      localU: 3,
      localV: 0,
    },
  ],
};

console.log(geometry.voxels.length); // 1
```

### BleacherVoxel

One voxel emitted by a stepped bleacher prop.

A voxel carries BOTH frames of the stand: the world-space `position` the
renderer draws, and the local `localU` / `localV` coordinates the crowd
builder measures the stand in. The local coordinates are what makes a
rotated bleacher measurable - a stand that is turned relative to the
world axes still reports the same footprint area in its own frame.

Example:

```ts
const voxel: BleacherVoxel = {
  position: [0, 0.8, 0],
  color: [0, 255, 255],
  role: 'seat',
  localU: 3,
  localV: 0,
};

console.log(voxel.role); // 'seat'
```

### BleacherVoxelRole

Role of a single voxel inside an expanded bleacher prop.

The role tells the renderer and the crowd builder what a voxel is FOR:
a `seat` carries a spectator, a `walk` voxel is the standing surface of
the platform, a `stair` voxel climbs between two levels, and a
`guardrail` voxel caps the track-facing edge of a level.

Example:

```ts
const role: BleacherVoxelRole = 'seat';

console.log(role === 'seat'); // true
```

### buildBleacherGeometry

```ts
buildBleacherGeometry(
  position: readonly [number, number, number],
  color: readonly [number, number, number],
  frame: SplineSampleFrame,
): BleacherGeometry
```

Builds the stepped voxel geometry for one spectator bleacher.

The local frame is aligned to the spline tangent so the front of the
bleacher (guardrail side) faces the track and the back side (stairs) faces
away. Seats are arranged in ascending rows toward the back, walking space
covers the full horizontal footprint, and the guardrail band caps the front
edge above the top row.

Parameters:
- `position` - World-space center of the bleacher prop.
- `color` - Neon color applied to every voxel.
- `frame` - Spline sample frame at the nearest track sample.

Returns: Bleacher voxel geometry carrying seats, walk, stairs, guardrail.

### buildRightWorldProps

```ts
buildRightWorldProps(
  input: TrackSlabBuildInput,
): readonly RightWorldProp[]
```

Builds the deterministic right-world neon scenery prop list for the given
track tuple.

The scene carries TWO rings. The OUTER ring holds the neon scenery kinds
and exactly {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT} bleachers
reserved BY CONSTRUCTION at evenly spaced slot indices. The INNER ring
holds the pit props - the mechanic, crew, and team-banner kinds - placed
closer to the road and at the midpoints between the bleacher blocks.

Parameters:
- `input` - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.

Returns: Ordered list of scenery props ready for the right-world render
 *   pass. Props are placed on the right-hand side of the track centerline,
 *   at least  {@link NEATRACING_PROP_CLEARANCE_FLOOR_M} beyond the SEAM edge
 *   of the road, and never exceed the density budget.
 *   The bleachers keep their reserved slot order, because every scene
 *   carries the same number of equally usable bleachers.
 *

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

console.log(props.length > 0); // true - a track always carries props
console.log(props.filter((prop) => prop.kind === 'bleacher').length); // 20
```

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

### emitRightWorldPeopleRing

```ts
emitRightWorldPeopleRing(
  samples: readonly SplineSample[],
  random: () => number,
  plan: RightWorldPeopleRingPlan,
  seam: RightWorldSeamContext,
): readonly RightWorldProp[]
```

Emits the inner ring of pit props: the mechanic, crew, and team-banner
kinds of the re-populated right world.

DRAW CONTRACT: every slot consumes EXACTLY three draws from `random` - the
offset, the archetype, and the colour - in that order, mirroring the
outer ring. The people kind is assigned BY CONSTRUCTION in a fixed round
robin, so the archetype draw is consumed and only the RESOLVED archetype
is overridden afterwards: the draw sequence never shifts, and every kind
is present at every seed.

The three draws of a slot are consumed BEFORE the track sample of the slot
is looked up, so a missing track sample can never shift the draw sequence
of the following slots.

The pit band sits closer to the road EDGE than the bleachers - the gap is
measured beyond the SEAM EDGE, so the two bands are compared on the same
scale - and its slots sit at the midpoints between the reserved bleacher
slots, so the two rings never share a track position. The rings are
separated ALONG the track: the distance of a prop from the centerline
alone does not order the two bands, because the road is wider on one part
of the track than on another.

Parameters:
- `samples` - Spline samples of the track the ring is placed along.
- `random` - Seeded PRNG the ring consumes its draws from.
- `plan` - Clearances and the track sample index of every people slot.
- `seam` - Seam context the clearances are measured against. It is
OPTIONAL: a caller that owns the track segments passes the seam it
resolved for them, and a caller that only owns the spline samples gets
the seam recovered from those samples.

Returns: The props of the inner ring, in emission order.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });

const random = (() => {
  let state = 42;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
})();

const props = emitRightWorldPeopleRing(spec.splineSamples, random, {
  baseOffset: 3,
  offsetRange: 2,
  sampleIndices: [4, 9, 14],
});

console.log(props.map((prop) => prop.kind)); // ['mechanic', 'crew', 'team-banner']
```

### emitRightWorldRing

```ts
emitRightWorldRing(
  samples: readonly SplineSample[],
  random: () => number,
  plan: RightWorldRingPlan,
  seam: RightWorldSeamContext,
): readonly RightWorldProp[]
```

Emits one ring of right-world props.

THE PLACEMENT: every slot is placed at the outward offset whose MEASURED
clearance beyond the SEAM edge reaches the clearance band of the slot, so
the ring never relies on the retired spline-normal distance model. A
bleacher is probed at its front seat row, so the row that carries the
seated crowd clears the floor and the stand's own anchor sits further
out.

DRAW CONTRACT: every slot consumes EXACTLY three draws from `random` - the
clearance band, the archetype, and the colour - in that order. A reserved
bleacher slot consumes its archetype draw like every other slot and only
OVERRIDES the resolved archetype afterwards, so the reservation never
shifts the draw sequence and every later prop stays byte-identical to the
same draw sequence without the reservation.

The function is exported so the draw contract can be exercised with a
reserved slot that is NOT the first outer slot. The reachable track
geometry reserves evenly spaced slots (see
{@link selectBleacherSlotIndices}), and because every reserved slot
consumes its archetype draw before the later slots, a skipped archetype
draw shifts EVERY later prop deterministically - the defect is directly
observable in the reachable space. A reservation further down the ring
exposes the same shift at a known interior position of the ring.

Parameters:
- `samples` - Spline samples of the track the ring is placed along.
- `random` - Seeded PRNG the ring consumes its draws from.
- `plan` - Slot count, archetypes, clearances, and reserved bleacher
slots.
- `seam` - Seam context the clearances are measured against. It is
OPTIONAL: a caller that owns the track segments passes the seam it
resolved for them, and a caller that only owns the spline samples gets
the seam recovered from those samples.

Returns: The props of the ring, in emission order.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });

// A tiny linear congruential generator, so the example needs no extra
// imports: the ring only ever asks for values in [0, 1).
const random = (() => {
  let state = 42;
  return () => {
    state = (state * 1664525 + 1013904223) % 4294967296;
    return state / 4294967296;
  };
})();

const props = emitRightWorldRing(
  spec.splineSamples,
  random,
  {
    slotCount: 4,
    kinds: ['pillar', 'gantry', 'sign'],
    baseOffset: 6,
    offsetRange: 4,
    reservedBleacherSlots: [0, 2],
  },
);

console.log(props[0]?.kind); // 'bleacher' - the reserved slot
```

### filterRobotsByPersonalSpace

```ts
filterRobotsByPersonalSpace(
  robots: readonly T[],
): readonly T[]
```

Deterministically thins a robot billboard list so every retained robot keeps
at least {@link NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M} meters of XZ
clearance from every other retained robot.

Robots are evaluated in input order; the first robot is always kept, and each
subsequent robot is kept only when its XZ distance to all previously kept
robots is >= the personal-space threshold. The threshold is the nominal
{@link NEATRACING_ROBOT_PERSONAL_SPACE_M} minus the named
{@link NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M} margin, so the
effective clearance is still 1 m while the comparison tolerates the
floating-point shortfall of the rotated seat lattice. This reduces density
when the track band cannot host the requested number of robots without
overlap, while staying pure, deterministic, and allocation-free on the
per-frame draw path.

Parameters:
- `robots` - Candidate robot billboards in deterministic order.

Returns: Subset of `robots` whose pairwise XZ separation is at least
 *  {@link NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M} - the nominal 1 m
 * minus the named 1e-12 m margin. A robot closer than that to an already
 * kept robot is dropped.
 *

Example:

```ts
const candidates = [
  { position: [0, 0, 0] },
  { position: [2, 0, 0] },
  { position: [0.5, 0, 0] },
];

const kept = filterRobotsByPersonalSpace(candidates);
console.log(kept.length); // 2 - the robot 0.5 m from the first one is dropped
console.log(kept[0]?.position); // [0, 0, 0]
console.log(kept[1]?.position); // [2, 0, 0]
```

### getBleacherAxisAlignedHalfSize

```ts
getBleacherAxisAlignedHalfSize(
  bleacher: RightWorldProp,
): readonly [number, number]
```

Returns the axis-aligned world-space half-width and half-depth that tightly
enclose all voxels of a bleacher.

The box is the world-space AABB of the stand, so it grows with the
rotation of the bleacher. It is used for the anchoring capacity and the
sprite placement; the crowd-host rule measures the stand in its OWN local
frame instead - see {@link getBleacherHostingFootprintArea}.

Parameters:
- `bleacher` - Bleacher prop whose voxels are measured.

Returns: The half-width and half-depth in meters of the world-space
axis-aligned box that tightly encloses the bleacher voxels, as
`[halfWidth, halfDepth]`. The box follows the rotation of the stand, so
the values differ from one bleacher to the next. A bleacher without voxel
geometry falls back to `[2, 1]`.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const bleacher = props.find((prop) => prop.kind === 'bleacher');
if (bleacher) {
  const [halfWidth, halfDepth] = getBleacherAxisAlignedHalfSize(bleacher);

  // The stand is turned against the world axes, so the box is larger
  // than the local footprint of the stand.
  console.log(halfWidth.toFixed(1)); // '7.6'
  console.log(halfDepth.toFixed(1)); // '6.5'
}
```

### getBleacherHostingFootprintArea

```ts
getBleacherHostingFootprintArea(
  bleacher: RightWorldProp,
): number
```

Returns the horizontal hosting footprint area of a bleacher in square
meters, derived from the real voxel geometry instead of the coarse bounding
box.

The area is measured in the bleacher's OWN local frame - `localU` runs
along the track normal and `localV` along the tangent - so it describes the
STAND and not the axis-aligned bounding box of the stand in world space.
A rotated stand has a bounding box that grows with the rotation, so the
world-space box would rank the stands by how they happen to be turned.

Every bleacher of the right world is the SAME stand, so every bleacher
measures the same hosting area and the named crowd-host rule falls back to
its documented tie-break (the lowest prop index, which is the bleacher of
the FIRST reserved slot).

Parameters:
- `bleacher` - Bleacher prop to measure.

Returns: Footprint area in square meters.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const host = props.find((prop) => prop.kind === 'bleacher');
if (host) {
  console.log(getBleacherHostingFootprintArea(host)); // 126
}
```

### getBleacherSeatAnchors

```ts
getBleacherSeatAnchors(
  bleacher: RightWorldProp,
): readonly [number, number, number][]
```

Returns the world-space anchors of the evenly spread front seat row of
every bleacher level, emitted level-major in ascending platform order.

The candidate set is the front-most stand row of each level of the bleacher
- the row closest to the track - so a bleacher offers ONE standing row per
level rather than every row of every level. Levels and rows are both read
from the bleacher's own voxel geometry, so the helper stays generic: any
number of levels and any number of rows per level is handled, and the level
heights are taken from the geometry instead of being assumed.

The anchors of a level are the EVENLY SPREAD subset of that level's front
row: exactly {@link NEATRACING_BLEACHER_SEATS_PER_LEVEL} anchors per level,
the first one on the first cell of the row and the last one on the last
cell, so the seated crowd covers the FULL platform length instead of
bunching at one end. The anchors are emitted LEVEL-MAJOR in ascending
platform order - the anchors of the lowest platform first, then the ones of
the next platform, and so on - so the stand is walked from the bottom up.
The returned y coordinate is the rendered seat-top surface for that cell;
callers that place billboards should add their own surface clearance
epsilon.

Parameters:
- `bleacher` - Bleacher prop whose seat voxels are inspected.

Returns: The evenly spread front-row seat anchors, level-major in
ascending platform order.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const host = props.find((prop) => prop.kind === 'bleacher');
if (host) {
  const anchors = getBleacherSeatAnchors(host);
  console.log(anchors.length); // 12 - four cells per level, three levels
  // The three levels sit at three distinct heights.
  console.log(new Set(anchors.map((anchor) => anchor[1])).size); // 3
  console.log(anchors[0]?.[1]?.toFixed(1)); // '0.8' - the seat-top of level one
}
```

### getBleacherSeatTopY

```ts
getBleacherSeatTopY(
  bleacher: RightWorldProp,
  x: number | undefined,
  z: number | undefined,
): number
```

Returns the top Y of the seat surface at a given world-space XZ location.

When `x` and `z` are omitted, this returns the highest seat row in the
bleacher (used for rendering bounding-box checks). When provided, it returns
the y of the nearest seat voxel so a billboard anchored to a lower level is
not hoisted to the top row.

Parameters:
- `bleacher` - Bleacher prop whose seat voxels are inspected.
- `x` - Optional world-space X of the query point. It is read only
together with `z`; when both are given, the seat voxel nearest to that
XZ position decides the returned height.
- `z` - Optional world-space Z of the query point, read only together
with `x`.

Returns: The world-space Y of the seat surface: the seat-top of the nearest
seat voxel for a query point, the highest seat-top of the bleacher
without one. A bleacher without voxel geometry falls back to
`bleacher.position[1] + 1.5`, and a bleacher whose geometry carries no
seat voxel falls back to `bleacher.position[1] + 3.0`.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const bleacher = props.find((prop) => prop.kind === 'bleacher');
if (bleacher) {
  console.log(
    getBleacherSeatTopY(bleacher).toFixed(1),
  ); // '3.0' - the top seat row
}
```

### getBleacherWalkSurfaceY

```ts
getBleacherWalkSurfaceY(
  bleacher: RightWorldProp,
): number
```

Returns the top Y of the bleacher walking surface.

Parameters:
- `bleacher` - Bleacher prop whose walk voxels are inspected.

Returns: The world-space Y of the topmost walking surface of the bleacher.
A bleacher without voxel geometry falls back to
`bleacher.position[1] + 0.1`, and a bleacher whose geometry carries no
walk voxel falls back to `bleacher.position[1] + 0.2`.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const bleacher = props.find((prop) => prop.kind === 'bleacher');
if (bleacher) {
  console.log(
    getBleacherWalkSurfaceY(bleacher).toFixed(1),
  ); // '0.2' - the walk surface
}
```

### measureSeamClearanceM

```ts
measureSeamClearanceM(
  seam: RightWorldSeamContext,
  worldX: number,
  worldZ: number,
): number
```

Measures the clearance of a world point beyond the SEAM edge, in meters.

The measurement uses the road's OWN union reader: the minimum past-edge
distance over EVERY published chord of the seam profile. A point that
clears the floor through this reader clears it through the nearest-chord
reader too, because the union is never larger than the nearest chord's
own reading.

Parameters:
- `seam` - The seam context of the track.
- `worldX` - World-frame x of the measured point.
- `worldZ` - World-frame z of the measured point.

Returns: The signed clearance beyond the seam edge, in meters.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
const seam = resolveRightWorldSeam(spec.segments);

// The road centreline sits INSIDE the seam edge.
console.log(measureSeamClearanceM(seam, spec.splineSamples[0]!.x, 0) < 0); // true
```

### NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2

Tolerance (square meters) used when comparing bleacher hosting footprint
areas inside the named crowd-host rule.

The tolerance turns the comparison into a TIE test: two bleachers whose
areas differ by less than the tolerance share the largest area, so a
floating-point wobble in the voxel geometry cannot break the tie.

Example:

```ts
console.log(NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2); // 1e-9
```

### NEATRACING_BLEACHER_LEVEL_COUNT

Number of flat seating levels of a bleacher.

The constant is the level count of the stock bleacher stand: the levels
are ordered by ascending seat height, so level 0 is the lowest platform
(the one nearest the track). The per-bleacher seat quota is DERIVED from
this count and from {@link NEATRACING_BLEACHER_SEATS_PER_LEVEL}, so the
crowd density follows the geometry instead of a hand-written literal.

Example:

```ts
console.log(NEATRACING_BLEACHER_LEVEL_COUNT); // 3
console.log(
  NEATRACING_BLEACHER_SEATS_PER_LEVEL * NEATRACING_BLEACHER_LEVEL_COUNT,
); // 12 - the seat anchors of one bleacher
```

### NEATRACING_BLEACHER_SEATS_PER_LEVEL

Number of seated spectators placed on EVERY bleacher level.

The constant is the seat density of ONE level of a bleacher: the anchors
of a level are the evenly spread subset of that level's front-row cell
lattice, so the seated crowd covers the FULL platform length instead of
bunching at one end of the row. Every bleacher of the right world carries
the same number of levels, so the constant also fixes the per-bleacher
anchor count.

Example:

```ts
console.log(
  NEATRACING_BLEACHER_SEATS_PER_LEVEL * NEATRACING_BLEACHER_LEVEL_COUNT,
); // 12 - the seat anchors of one bleacher

const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const bleacher = props.find((prop) => prop.kind === 'bleacher');
if (bleacher) {
  console.log(getBleacherSeatAnchors(bleacher).length); // 12
}
```

### NEATRACING_PROP_CLEARANCE_FLOOR_M

The USER-NAMED clearance floor of every right-world prop, in meters.

The floor is the distance EVERY prop stands beyond the SEAM EDGE - the
one edge the road's own readers publish - and it is a USER directive, not
an author decision: it is never lowered, never widened by a tolerance and
never satisfied by dropping a prop. The placement clamps EVERY clearance
target to this floor, so the floor holds for every prop kind, at every
seed and for every track size bucket.

Example:

```ts
console.log(NEATRACING_PROP_CLEARANCE_FLOOR_M); // 10
```

### NEATRACING_RIGHT_WORLD_BLEACHER_COUNT

Number of spectator bleachers the right world carries.

The count is a CONSTANT of the world, not a function of the seed or of
the track size bucket: the outer ring reserves this many bleacher slots BY
CONSTRUCTION at evenly spaced slot indices, so every scene offers the same
number of spectator stands. The reserved slots sit in the outer ring, so
they are placed farther from the centerline than the inner-ring pit props.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

console.log(
  props.filter((prop) => prop.kind === 'bleacher').length,
); // 20 - the reserved slots, whatever the seed
```

### NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M

Depth of the bleacher footprint ACROSS the track, in meters.

The walking surface runs from the track-facing guardrail row back to the
stair column, so the structure is fourteen meters deep in the lateral
direction.

Example:

```ts
console.log(NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_DEPTH_M); // 14
```

### NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M

Width of the bleacher footprint ALONG the track, in meters.

The seats of a bleacher span seven cells plus the free outer rows, so the
structure is nine meters wide across the track direction.

Example:

```ts
console.log(NEATRACING_RIGHT_WORLD_BLEACHER_FOOTPRINT_WIDTH_M); // 9
```

### NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M

Minimum world-space center-to-center separation between two bleacher
FOOTPRINT CENTRES.

Two bleachers whose footprint centres are closer than the sum of their
circumradii COULD overlap: the Minkowski difference of two rectangles of
the same shape is contained in the disk of that radius. The sum of the
two circumradii of a bleacher footprint is
`2 * hypot(footprintWidth / 2, footprintDepth / 2)` = 16.64 m, so the
named floor is that threshold rounded UP to the next tenth of a meter -
16.7 m - which is a sufficient - and conservative - no-overlap condition
for the NOMINAL footprint rectangle. The rendered stand is the voxel
boxes, and every box adds a half extent per side, so the RENDERED
footprint is 10 m by 15 m and carries its own sufficient bound of
18.03 m; the realized separation is far beyond both bounds.
The evenly spaced reserved slots realize a separation of the track length
divided by the bleacher count, far beyond this floor.

The footprint centre is NOT the prop anchor: the anchor sits 4 m inboard
of the track-facing guardrail row, so the footprint centre lies 2.5 m
outward of it. The separation contract therefore measures the footprint
centres - the quantity this floor is derived from - and the prop anchors
as well.

Example:

```ts
console.log(NEATRACING_RIGHT_WORLD_BLEACHER_MIN_SEPARATION_M); // 16.7
```

### NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS

Minimum number of bleacher blocks of the outer ring.

The floor keeps the twenty reserved bleachers on every reachable track,
even a short one. The block count is never lowered to fit the track: a
track that is too short for twenty non-overlapping bleachers shows up as a
FAILING separation contract instead of a silently reduced crowd.

Example:

```ts
console.log(
  resolveOuterRingSlotCount(108) / NEATRACING_RIGHT_WORLD_BLEACHER_COUNT,
); // 2 - the floor governs a reachable track
```

### NEATRACING_RIGHT_WORLD_PEOPLE_RING_COUNT

Number of inner-ring pit props the right world carries.

The pit band places one prop at every midpoint between two reserved
bleacher slots, so the band holds one prop per bleacher pair - half the
bleacher count. The constant is part of the density budget arithmetic:
the budget covers the outer ring AND the band, so the outer ring reserves
the band before it counts its own blocks.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

console.log(props.filter((prop) => prop.ringSide === 'inner').length); // 10
```

### NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET

Maximum number of right-world neon scenery props emitted per track.

The budget is the density ceiling of the right world and it covers BOTH
rings: the outer-ring slots AND the inner pit band. The outer ring
therefore reserves the pit band before it counts its own blocks, so the
emitted prop list never exceeds the budget. The budget is large enough for
the twenty reserved bleachers AND the twenty scenery slots beside them to
fit on every reachable track.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

console.log(props.length); // 50 - 20 bleachers, 20 scenery props, 10 pit props
console.log(props.length <= NEATRACING_RIGHT_WORLD_PROP_DENSITY_BUDGET); // true
```

### NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT

Number of track samples one outer-ring slot needs.

The outer ring is spread over the whole track, so a denser track can host
more slots. The constant is the DENSITY INPUT of the block rule in
{@link resolveOuterRingSlotCount}: one more block of twenty slots needs
`SAMPLES_PER_RING_SLOT * BLEACHER_COUNT` more track samples.

On every REACHABLE track the constant does not bind: a reachable track
carries 108 to 144 samples, fewer than the 200 samples an extra block
needs, so the density term of the rule is zero and the floor of
{@link NEATRACING_RIGHT_WORLD_MIN_RING_BLOCKS} blocks governs the slot
count. The constant stays the named input of the rule, so a longer track
can widen the ring without the rule losing its density guard.

Example:

```ts
console.log(NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT); // 10
console.log(
  Math.floor(108 / NEATRACING_RIGHT_WORLD_SAMPLES_PER_RING_SLOT),
); // 10 - fewer samples than one extra block of twenty slots needs
```

### NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M

Effective personal-space clearance: the nominal 1 m minus the named margin.

A robot is dropped when its XZ distance to an already kept robot is BELOW
this value, so every retained pair keeps at least
`NEATRACING_ROBOT_PERSONAL_SPACE_M` minus
`NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M` meters of separation.

Example:

```ts
console.log(NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M); // 0.999999999999
console.log(
  NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M <
    NEATRACING_ROBOT_PERSONAL_SPACE_M,
); // true - the effective clearance carries the named floating-point margin
```

### NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M

Named tolerance applied to the personal-space comparison.

The bleacher stand-cell lattice is mathematically 1 m wide, but the
world-space seat positions are produced through a rotated basis, so the
realized separation of two neighbouring cells measures a few 1e-14 m short
of 1 m. A strict `>= 1 m` comparison therefore rejects every second cell of
the lattice and truncates the crowd pick budget.

The margin is 1e-12 m - roughly 30x the worst measured shortfall of
2.99e-14 m - so the EFFECTIVE clearance stays 1 m and the personal-space
contract (no two robots closer than 1 m) is never relaxed; it is only
expressed against the named margin so the contract and the implementation
agree.

Example:

```ts
console.log(NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M); // 1e-12
console.log(
  NEATRACING_ROBOT_PERSONAL_SPACE_M -
    NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M,
); // 0.999999999999 - the effective clearance
```

### NEATRACING_ROBOT_PERSONAL_SPACE_M

Minimum XZ center-to-center separation enforced between robot billboards
(the public crowd) so two spectators never occupy the same spot.

The personal space is a FIXED world-space constant, and it is deliberately
NOT derived from the size of a crowd sprite. The world size of a crowd
billboard belongs to a different pair of levers - the shared
meters-per-pixel scale `NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M` and the
world-scale lever `NEATRACING_PUBLIC_BILLBOARD_WORLD_SCALE` of
`public.sprite.ts` - and a change to either of them must never move this
threshold. A crowd keeps its one-meter separation whatever the sprite
size is.

The one-meter value is the PITCH of the bleacher stand lattice: a stand
cell is one 1 m voxel cell on the integer local lattice, so neighbouring
cells are one meter apart. A personal space of one meter therefore never
drops a spectator that stands on a cell of its own; it thins away the
billboards that would otherwise land on the same spot and z-fight there.

The comparison runs against
{@link NEATRACING_ROBOT_PERSONAL_SPACE_CLEARANCE_M} - the nominal value
minus the named {@link NEATRACING_ROBOT_PERSONAL_SPACE_EPSILON_M} margin
- so every retained pair still keeps a full meter of separation.

Example:

```ts
console.log(NEATRACING_ROBOT_PERSONAL_SPACE_M); // 1 - one meter of space
```

### PositionedInWorld

Any object that carries a world-space `position` triplet.

The personal-space filter only reads the XZ pair of the triplet, so any
billboard-shaped object with a position satisfies the contract.

Example:

```ts
const billboard: PositionedInWorld = { position: [0, 0, 0] };

console.log(billboard.position[2]); // 0 - the Z the filter reads
```

### recoverRightWorldSeam

```ts
recoverRightWorldSeam(
  samples: readonly SplineSample[],
): RightWorldSeamContext
```

Recovers the SEAM context of a track from its spline samples.

THE RECOVERY IS EXACT: the FIRST sample of every segment strip carries
the segment's own start point and the segment's own width, so the closed
loop of chords can be rebuilt from the samples themselves - the start of
a chord is the start point of its segment strip, and its end is the start
point of the next strip. The recovered context is the DEFAULT seam of the
emitters, so a caller that only owns the spline samples still measures
the clearances against the published edge; a caller that owns the track
segments passes the seam it resolved for them instead.

Parameters:
- `samples` - Spline samples of the track.

Returns: The seam context recovered from the samples.

### resolveOuterRingSlotCount

```ts
resolveOuterRingSlotCount(
  sampleCount: number,
): number
```

Resolves the number of outer-ring slots for a track with the given number
of spline samples.

The slot count is a whole multiple of
{@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT}, so the reserved bleacher
slots can be spread evenly over the ring. The multiple is bounded by the
density budget - the budget covers the outer ring AND the inner pit band,
so the band is reserved first - and by the track density, so a short track
never asks for more slots than it has samples to place them on. The floor
of two blocks keeps the twenty bleachers on every reachable track.

On every REACHABLE track the density term of the rule is ZERO - it sits
BELOW the floor, not beside it - and the budget allows four blocks, so
the floor of two blocks governs: the resolved count is always forty
slots, twenty bleachers and twenty scenery props, plus the ten
inner-ring pit props of the band.

Parameters:
- `sampleCount` - Number of spline samples of the track.

Returns: The number of outer-ring slots.

Example:

```ts
console.log(resolveOuterRingSlotCount(108)); // 40
```

### resolveRightWorldSeam

```ts
resolveRightWorldSeam(
  segments: readonly TrackSegment[],
): RightWorldSeamContext
```

Resolves the SEAM context of a track from its raw chord segments.

The profile is the seam's OWN publication for those segments, so the
context carries the same half-widths, corner measures and loop orientation
the road's own readers use.

Parameters:
- `segments` - Ordered closed-loop chord segments of the track.

Returns: The seam context of the track.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
const seam = resolveRightWorldSeam(spec.segments);

console.log(seam.profile.chords.length); // 6 - one published edge per chord
```

### resolveRightWorldSeed

```ts
resolveRightWorldSeed(
  input: TrackSlabBuildInput,
): number
```

Resolves a deterministic unsigned 32-bit seed for the right-world builder.

Parameters:
- `input` - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.

Returns: Unsigned 32-bit seed for the local PRNG.

### resolveSeamClearanceOffsetM

```ts
resolveSeamClearanceOffsetM(
  seam: RightWorldSeamContext,
  sample: SplineSample,
  frame: SplineSampleFrame,
  targetClearanceM: number,
  probeInsetM: number,
): number
```

Resolves the outward offset that puts a prop's PROBE point at the target
clearance beyond the SEAM edge.

THE SEARCH: the ray starts at the track sample and walks OUTWARD - the
side the right world is built on - in steps of
{@link NEATRACING_SEAM_CLEARANCE_STEP_M}. The FIRST candidate whose
MEASURED clearance reaches the target wins, so the placement is the
closest one that clears the floor. The clearance along the ray is NOT
assumed to be monotone: the winning candidate is always a MEASURED one,
never an interpolated one, and a target the ray never reaches falls back
to the candidate with the best measured clearance.

THE PROBE POINT: the clearance is measured at the probe - the prop's own
anchor pulled `probeInsetM` meters TOWARD the track - so a prop whose
body reaches closer to the road than its anchor (a bleacher with its
front seat row) is measured by the part that comes closest to the road.

Parameters:
- `seam` - The seam context of the track.
- `sample` - Track sample the prop belongs to.
- `frame` - Spline frame of the sample.
- `targetClearanceM` - The clearance the probe has to reach, in meters.
- `probeInsetM` - Distance from the anchor to the probe point, toward
the track, in meters.

Returns: The outward offset of the prop, in meters.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });
const seam = resolveRightWorldSeam(spec.segments);
const sample = spec.splineSamples[0]!;
const frame = resolveSplineSampleFrame(spec.splineSamples, sample.globalIndex);

const offset = resolveSeamClearanceOffsetM(seam, sample, frame, 10, 0);
console.log(offset > 0); // true - the sample itself is inside the road
```

### RightWorldPeopleRingPlan

Placement parameters of the inner-ring people band.

The plan carries the two offsets of the band and the TRACK SAMPLE of every
people slot. The sample indices are the midpoints between the reserved
bleacher slots, so the pit band never shares a track position with a
bleacher.

Example:

```ts
const plan: RightWorldPeopleRingPlan = {
  baseOffset: 3,
  offsetRange: 2,
  sampleIndices: [4, 9, 14],
};

console.log(plan.sampleIndices.length); // 3
```

### RightWorldProp

One TRON-style neon scenery prop.

A prop is a POSITION, a COLOR, an ARCHETYPE, and the RING it belongs to.
The ring side is what separates the two bands of the right world: the
outer ring carries the neon scenery and the spectator bleachers, the
inner ring carries the pit props of the mechanic, crew, and team-banner
kinds.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const bleacher: RightWorldProp | undefined = props.find(
  (prop) => prop.kind === 'bleacher',
);

console.log(bleacher?.ringSide); // 'outer'
```

### RightWorldRingPlan

Placement parameters of one emitted right-world prop ring.

The plan is the GEOMETRY of the ring, not its content: how many slots the
ring is divided into, which archetypes the slots may draw, how far the
ring sits beyond the SEAM edge, and which slot indices are reserved as
bleachers. The builder fills the plan, and the emitters only read it.

Example:

```ts
const plan: RightWorldRingPlan = {
  slotCount: 40,
  kinds: ['pillar', 'gantry', 'sign'],
  baseOffset: 6,
  offsetRange: 4,
  reservedBleacherSlots: selectBleacherSlotIndices(40),
};

console.log(plan.reservedBleacherSlots.length); // 20
```

### RightWorldSeamContext

The SEAM context of one track: the published edge profile and the raw
chord segments it was resolved from.

The context is the ONE place where the right world reads the track edge.
Every clearance the builder measures - and every offset it derives from
that measurement - is read through the context, so the props and the road
can never disagree about where the edge is.

### selectBleacherHostProp

```ts
selectBleacherHostProp(
  props: readonly RightWorldProp[],
): RightWorldProp | undefined
```

Returns the bleacher selected by the named crowd-host rule: the one with
the LARGEST hosting footprint area, with ties broken by the LOWEST prop
index.

The right world builds every bleacher as the SAME stand, so the rule falls
back to the tie-break and selects the bleacher of the FIRST reserved slot.

The rule selects on the hosting footprint area alone, so it does not
single out the bleacher that carries the crowd: the seated spectators are
placed on EVERY bleacher of the ring, and the selection stays a
deterministic function of the prop list.

Parameters:
- `props` - Scene props in their final prop order.

Returns: The rule-selected crowd host, or `undefined` without bleachers.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const host = selectBleacherHostProp(props);
console.log(host?.kind); // 'bleacher'
```

### selectBleacherSlotIndices

```ts
selectBleacherSlotIndices(
  slotCount: number,
): readonly number[]
```

Selects the outer-ring slot indices reserved as bleachers.

The reserved slots are evenly spaced over the ring, so the bleacher count
is exactly {@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT} and the spacing
between two bleachers is the same everywhere along the track.

PRECONDITION: `slotCount` is a whole multiple of
{@link NEATRACING_RIGHT_WORLD_BLEACHER_COUNT}, which
{@link resolveOuterRingSlotCount} guarantees. The stride itself is always
uniform; for a slot count that is NOT a whole multiple the leftover slots
all fall at the end of the ring, so the reserved slots stay inside the
ring but no longer cover it symmetrically.

Parameters:
- `slotCount` - Number of slots of the outer ring.

Returns: The reserved slot indices, in ascending order.

Example:

```ts
console.log(selectBleacherSlotIndices(40)); // [0, 2, 4, ..., 38]
```

### selectEvenlySpacedCellIndices

```ts
selectEvenlySpacedCellIndices(
  cellCount: number,
  seatCount: number,
): number[]
```

Selects the evenly spaced subset of the cells of one row.

The selection spreads the requested seats over the FULL row: the first
selected cell is the FIRST cell of the row and the last selected cell is
the LAST cell of the row, and the cells in between are spread evenly. The
effective seat count never exceeds the number of available cells, so a row
that is shorter than the requested seat count yields every cell of the row
instead of a duplicated cell. The returned indices are ascending and
distinct, so a caller can map them straight onto the cells of the row.

Parameters:
- `cellCount` - Number of cells the row carries.
- `seatCount` - Number of seats the caller wants to place on the row.

Returns: The selected cell indices, ascending and distinct. A
non-positive seat count selects nothing.

Example:

```ts
console.log(selectEvenlySpacedCellIndices(7, 4)); // [0, 2, 4, 6]
console.log(selectEvenlySpacedCellIndices(3, 4)); // [0, 1, 2]
```

### selectMaxAreaBleachers

```ts
selectMaxAreaBleachers(
  props: readonly RightWorldProp[],
): readonly RightWorldProp[]
```

Returns every bleacher that ties for the LARGEST hosting footprint area of
the scene, in prop order.

Only bleachers that carry real voxel geometry take part in the rule; a
bleacher without geometry has no measurable hosting footprint. Bleachers
whose area is within
{@link NEATRACING_BLEACHER_FOOTPRINT_AREA_TOLERANCE_M2} of the largest area
are treated as a tie.

The right world builds every bleacher as the SAME stand, so the whole
bleacher set ties and the returned tie-set is the bleacher set itself, in
the order of the reserved slots.

The rule is a PURE function of the prop list: it selects on the hosting
footprint area alone and does not read where the seated crowd is placed.
The seated crowd covers EVERY bleacher of the ring, so a right-world scene
resolves to the whole bleacher set.

Parameters:
- `props` - Scene props in their final prop order.

Returns: The tie-set of the largest-area bleachers, in prop order. An empty
array when the scene carries no bleacher with voxel geometry.

Example:

```ts
const props = buildRightWorldProps({
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
});

const largest = selectMaxAreaBleachers(props);
console.log(largest.length); // 20 - every bleacher of the ring is the same stand
console.log(largest[0]?.kind); // 'bleacher'
```

### selectPeopleRingSampleIndices

```ts
selectPeopleRingSampleIndices(
  samples: readonly SplineSample[],
  bleacherSlotIndices: readonly number[],
  outerSlotCount: number,
): readonly number[]
```

Selects the track sample indices of the inner-ring people props.

The people slots sit at the midpoints between the reserved bleacher
slots, so the pit band is spread evenly over the track and never shares a
track position with a bleacher.

Parameters:
- `samples` - Spline samples of the track.
- `bleacherSlotIndices` - Reserved bleacher slots of the outer ring.
- `outerSlotCount` - Number of slots of the outer ring.

Returns: The track sample index of every people slot, in slot order.

Example:

```ts
const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'large' });

const indices = selectPeopleRingSampleIndices(
  spec.splineSamples,
  selectBleacherSlotIndices(40),
  40,
);

console.log(indices.length); // 10 - one midpoint per bleacher pair
```

## track/right-world.prop-sizes.ts

Canonical axis-aligned half-sizes for every prop archetype emitted by the
right-world builder.

One table drives three consumers that have to agree exactly:
- solid voxel box emission in the right-world render pass,
- surface anchoring of billboard sprites in `public.sprite.ts`,
- collision and elevation assertions in the right-world test suite.

Sharing the table is what guarantees that a billboard's resolved foot
surface matches the rendered prop top, so a sprite never floats above - or
sinks into - the geometry it is supposed to stand on.

The table covers the outer-ring scenery archetypes (pillar, gantry, sign)
and the spectator bleacher, plus the inner-ring people archetypes
(mechanic, crew, team-banner). Its y component is the vertical half-extent
measured from the base plane, so the prop top sits at
`position.y + 2 * halfHeight`.

The BLEACHER row is the coarse exception: a bleacher is a stepped voxel
structure, so the render pass and the sprite anchoring read its size from
the real voxel geometry and use the row only as a fallback.

### resolvePropHalfSize

```ts
resolvePropHalfSize(
  kind: "mechanic" | "pillar" | "gantry" | "sign" | "bleacher" | "crew" | "team-banner",
): readonly [number, number, number]
```

Resolves the canonical axis-aligned half-size of a scenery prop in meters.

The lookup is a table read, so the answer is stable and cheap enough to
call per prop and per billboard.

Parameters:
- `kind` - Prop archetype emitted by the right-world builder.

Returns: Half-width, half-height, half-depth in meters.

Example:

```ts
const [halfWidth, halfHeight, halfDepth] = resolvePropHalfSize('pillar');
console.log(halfWidth, halfHeight, halfDepth); // 0.4 2 0.4
```

### RIGHT_WORLD_PROP_HALF_SIZES

Half-width, half-height, half-depth in meters for every right-world prop
kind. The y component is the vertical half-extent from the base plane; the
prop top is at `position.y + 2 * halfHeight`.

Example:

```ts
const [halfWidth, halfHeight, halfDepth] = RIGHT_WORLD_PROP_HALF_SIZES.pillar;
console.log(halfWidth, halfHeight, halfDepth); // 0.4 2 0.4
```
