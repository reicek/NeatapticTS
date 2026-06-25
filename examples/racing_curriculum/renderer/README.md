# renderer

Canvas 2D renderer for the Tier 0 racing curriculum demo.

Rendering is intentionally flat and stateless relative to game logic —
the renderer consumes a frozen `TrackSpec`, the current `EnvironmentState`,
and a small mutable `RacingRenderState` (tire marks) and produces one frame.

Visual style: neon-retro-arcade — dark background, cyan/blue structure,
square-outline car, fading tire marks, neon-white bumper lighting.

The world coordinate system is math-convention (Y increases upward).
Canvas pixels use Y-down convention. The affine `WorldTransform` absorbs
both the scale and the canvas centre offset; no explicit Y-flip is needed
because positive world-Y maps to positive canvas-Y (lower on screen) and
both the car heading render and the physics movement are consistent in the
same coordinate frame.

## renderer/racing.renderer.ts

### advanceTireMarks

```ts
advanceTireMarks(
  renderState: RacingRenderState,
  envState: EnvironmentState,
): void
```

Appends a new tire mark at the car's current position and ages all marks.
Marks older than `TIRE_MARK_MAX_AGE_TICKS` are evicted from the front.

Parameters:
- `renderState` - Mutable render state (mutated in-place).
- `envState` - Current car position source.

### buildGuidingLineForTeam

```ts
buildGuidingLineForTeam(
  trackSpec: TrackSpec,
  teamIndex: number,
): { readonly x: number; readonly y: number; }[]
```

Builds a fresh per-team guiding line parallel to the inner-lane centerline.

Team A (`teamIndex = 0`) starts exactly on the inner-lane centerline so the
first point matches the car start position. Team B (`teamIndex = 1`) uses a
small constant inward offset that keeps the line inside the inner lane. Each
call returns a distinct array, so callers may mutate or cache freely.

Parameters:
- `trackSpec` - Frozen track geometry.
- `teamIndex` - Team index: 0 = Team A, 1 = Team B.

Returns: Fresh ordered list of world-space `{x, y}` guiding points.

Example:

```ts
const teamAGuidingLine = buildGuidingLineForTeam(trackSpec, 0);
const teamBGuidingLine = buildGuidingLineForTeam(trackSpec, 1);
```

### CanvasPoint

Canvas-space coordinate pair produced by the world transform.

### computeWorldTransform

```ts
computeWorldTransform(
  canvas: HTMLCanvasElement,
  spec: TrackSpec,
  options: WorldTransformOptions,
): WorldTransform
```

Computes a world-to-canvas affine transform that fits the entire track
inside the canvas with uniform padding on all sides.

Parameters:
- `canvas` - Target canvas (uses `width` and `height` attributes).
- `spec` - Frozen track spec whose bounding box determines the scale.

Returns: Affine transform for use in `renderRacingFrame`.

Example:

```ts
const transform = computeWorldTransform(canvasElement, trackSpec);
```

### createRacingRenderState

```ts
createRacingRenderState(): RacingRenderState
```

Creates a zeroed `RacingRenderState` ready for first use.

Returns: Fresh render state with an empty tire-mark list.

Example:

```ts
const renderState = createRacingRenderState();
renderRacingFrame(canvas, spec, envState, renderState, transform);
```

### drawCar

```ts
drawCar(
  ctx: CanvasRenderingContext2D,
  state: RenderCarState,
  transform: WorldTransform,
  tireState: TireStateTuple,
  carOutlineColor: string,
): void
```

Draws the car as a square outline with a neon-white front bumper and
forward headlight projection.

The car rectangle is drawn in local space (car centre at origin, facing
positive local X), then rotated and translated to world position via the
canvas transform stack.

Parameters:
- `ctx` - 2D rendering context.
- `state` - Current physics state (position + heading).
- `transform` - World-to-canvas affine transform.

### drawCarTireCorners

```ts
drawCarTireCorners(
  ctx: CanvasRenderingContext2D,
  halfLengthCanvas: number,
  halfWidthCanvas: number,
  tireState: TireStateTuple,
): void
```

Draws live tire-health corner markers on the car body.

Parameters:
- `ctx` - 2D rendering context.
- `halfLengthCanvas` - Half car length in canvas pixels.
- `halfWidthCanvas` - Half car width in canvas pixels.
- `tireState` - Ordered tire-health tuple `[FL, FR, RL, RR]`.

### drawGuidingLinePath

```ts
drawGuidingLinePath(
  ctx: CanvasRenderingContext2D,
  worldPoints: readonly { readonly x: number; readonly y: number; }[],
  transform: WorldTransform,
  rgbColor: string,
  alpha: number,
): void
```

Traces and strokes one guiding line path with the requested team color.

Parameters:
- `ctx` - 2D rendering context.
- `worldPoints` - Ordered world-space guiding points.
- `transform` - World-to-canvas affine transform.
- `rgbColor` - RGB color string without alpha wrapper.
- `alpha` - Stroke alpha in [0, 1].

### drawOptimalLineGuidance

```ts
drawOptimalLineGuidance(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
  guidanceAlpha: number,
): void
```

Draws the faded optimal-line overlay used by the Tier 1 browser harness.

Parameters:
- `ctx` - 2D rendering context.
- `trackRenderGeometry` - Cached spline-derived track geometry.
- `transform` - World-to-canvas transform.
- `guidanceAlpha` - Overlay alpha in [0, 1].

### drawPitOverlayCenterDetails

```ts
drawPitOverlayCenterDetails(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  strokeColor: string,
  occupied: boolean,
): void
```

Draws centered pit scanline and side beacons for extra Tron surface detail.

Parameters:
- `ctx` - 2D rendering context.
- `canvasWidth` - Overlay width in canvas pixels.
- `canvasHeight` - Overlay height in canvas pixels.
- `strokeColor` - Team stroke color.
- `occupied` - Whether the parent pit overlay is occupied.

### drawPitOverlayCornerAccents

```ts
drawPitOverlayCornerAccents(
  ctx: CanvasRenderingContext2D,
  canvasWidth: number,
  canvasHeight: number,
  strokeColor: string,
  dashedOverlay: boolean,
): void
```

Draws compact corner accent marks for pit overlays in local overlay space.

Parameters:
- `ctx` - 2D rendering context.
- `canvasWidth` - Overlay width in canvas pixels.
- `canvasHeight` - Overlay height in canvas pixels.
- `strokeColor` - Team stroke color.
- `dashedOverlay` - Whether the parent overlay uses dashed lines.

### drawPitOverlays

```ts
drawPitOverlays(
  ctx: CanvasRenderingContext2D,
  spec: TrackSpec,
  transform: WorldTransform,
  pitStatus: Uint8Array<ArrayBufferLike> | Uint16Array<ArrayBufferLike> | Int16Array<ArrayBufferLike> | undefined,
  visiblePitTeamIndex: 0 | 1 | undefined,
): void
```

Draws the Tier 4 pit entrance and stall overlays.

`pitStatus` uses the packed tuple `[teamA_car, teamA_ticks, teamB_car,
teamB_ticks]`. A positive tick count marks that team's pit as occupied and
causes both the stall and entrance corridor AABB to render with the occupied
fill overlay.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Frozen track geometry.
- `transform` - World-to-canvas affine transform.
- `pitStatus` - Optional packed pit-status tuple.

### drawRotatedOverlay

```ts
drawRotatedOverlay(
  ctx: CanvasRenderingContext2D,
  worldBox: { readonly x: number; readonly y: number; readonly width: number; readonly height: number; },
  transform: WorldTransform,
  strokeColor: string,
  occupied: boolean,
  dashPattern: readonly number[],
  rotationRadians: number,
): void
```

Draws one overlay rectangle in world space with optional rotation.

Parameters:
- `ctx` - 2D rendering context.
- `worldBox` - World-space rectangle.
- `transform` - World-to-canvas affine transform.
- `strokeColor` - Outline/fill color.
- `occupied` - Whether the box is currently occupied.
- `dashPattern` - Optional dashed outline pattern.
- `rotationRadians` - World-space rotation in radians.

### drawStartLineCrosswalk

```ts
drawStartLineCrosswalk(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void
```

Draws a neon-white outlined square crosswalk at the lane start sample.

Squares are aligned with the local tangent direction and distributed across
the lane width using the local normal direction.

Parameters:
- `ctx` - 2D rendering context.
- `trackRenderGeometry` - Cached spline-derived track geometry.
- `transform` - World-to-canvas affine transform.

### drawTeamGuidingLines

```ts
drawTeamGuidingLines(
  ctx: CanvasRenderingContext2D,
  spec: TrackSpec,
  transform: WorldTransform,
  guidanceAlpha: number,
): void
```

Draws a per-team dashed guiding line over the track when the guidance overlay
is enabled.

Team A uses cyan and Team B uses magenta so each agent has a visually distinct
lane marker. The lines are drawn before car bodies because this helper runs
inside the track-drawing pass.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Frozen track geometry.
- `transform` - World-to-canvas affine transform.
- `guidanceAlpha` - Overlay alpha in [0, 1].

### drawTireMarks

```ts
drawTireMarks(
  ctx: CanvasRenderingContext2D,
  marks: readonly TireMark[],
  transform: WorldTransform,
): void
```

Draws the fading tire-mark trail behind the car.

Each mark fades from `COLOR_TIRE_MARK_MAX_ALPHA` to fully transparent as
its age increases toward `TIRE_MARK_MAX_AGE_TICKS`.

Parameters:
- `ctx` - 2D rendering context.
- `marks` - Tire mark list from the render state.
- `transform` - World-to-canvas transform.

### drawTrack

```ts
drawTrack(
  ctx: CanvasRenderingContext2D,
  spec: TrackSpec,
  transform: WorldTransform,
  guidanceAlpha: number,
  pitStatus: Uint8Array<ArrayBufferLike> | Uint16Array<ArrayBufferLike> | Int16Array<ArrayBufferLike> | undefined,
  visiblePitTeamIndex: 0 | 1 | undefined,
): void
```

Draws all track layers onto the canvas context.

Layers (back to front): glow halo, asphalt surface, left/right edge lines,
dashed centerline, start-line neon square crosswalk.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Frozen track geometry.
- `transform` - World-to-canvas affine transform.

### drawTrackCenterline

```ts
drawTrackCenterline(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void
```

Draws a dashed centerline along each segment.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Track geometry.
- `transform` - World-to-canvas transform.

### drawTrackEdgeLines

```ts
drawTrackEdgeLines(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void
```

Draws the bright cyan edge lines on both sides of each segment using
perpendicular offset geometry.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Track geometry.
- `transform` - World-to-canvas transform.

### drawTrackGlowLayer

```ts
drawTrackGlowLayer(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void
```

Draws a wide semi-transparent glow behind each track segment.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Track geometry.
- `transform` - World-to-canvas transform.

### drawTrackSurfaceLayer

```ts
drawTrackSurfaceLayer(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void
```

Fills the track surface with the dark asphalt colour.

Parameters:
- `ctx` - 2D rendering context.
- `spec` - Track geometry.
- `transform` - World-to-canvas transform.

### getTrackRenderGeometry

```ts
getTrackRenderGeometry(
  spec: TrackSpec,
): TrackRenderGeometry
```

Resolves cached spline-derived geometry for the given `TrackSpec`.

Parameters:
- `spec` - Frozen track geometry.

Returns: Cached centerline/boundary geometry.

### isPitsEnabledForCurrentTier

```ts
isPitsEnabledForCurrentTier(
  overlayFrame: RacingRenderOverlayFrame | undefined,
  carCount: number,
  pitBoxCount: number,
): boolean
```

Resolves whether pit-gated visuals are active for the current render tier.

Worker-backed paths expose explicit feature flags; browser-local fallback
treats multi-car tiers with authored pit geometry as pit-enabled.

Parameters:
- `overlayFrame` - Optional packed worker frame.
- `carCount` - Number of cars in the current render roster.
- `pitBoxCount` - Number of generated pit boxes on the current track.

Returns: True when pit visuals should be enabled for this frame.

### normalizeHeadingRadiansCandidate

```ts
normalizeHeadingRadiansCandidate(
  headingCandidate: number,
): number
```

Normalizes a heading candidate to radians in [-pi, pi], tolerating degree
inputs from future pit metadata producers.

Parameters:
- `headingCandidate` - Metadata heading candidate.

Returns: Normalized radian heading.

### normalizeRadians

```ts
normalizeRadians(
  valueRadians: number,
): number
```

Wraps an angle in radians to the interval [-pi, pi].

Parameters:
- `valueRadians` - Input angle in radians.

Returns: Wrapped angle in radians.

### PitOrientationMetadata

Optional pit-orientation fields that may be attached by track generators.

### RacingRenderOptions

Optional overlays layered on top of the base track render.

Tier 4 uses `frame.tireState` for live corner colors and `frame.pitStatus`
for pit occupancy overlays. `pitStatus` follows the packed tuple
`[teamA_car, teamA_ticks, teamB_car, teamB_ticks]`.

### RacingRenderOverlayFrame

Narrow worker-frame fields consumed by the Tier 4 renderer overlays.

### RacingRenderState

Mutable render state owned by the animation loop.

Isolated from the physics `EnvironmentState` so that rendering artefacts
(trail length, mark density) can be tuned without touching the simulation.

### RenderCarState

Pose and optional tire tuple for one renderable car slot.

### renderRacingFrame

```ts
renderRacingFrame(
  canvas: HTMLCanvasElement,
  spec: TrackSpec,
  envState: EnvironmentState,
  renderState: RacingRenderState,
  transform: WorldTransform,
  renderOptions: RacingRenderOptions,
): void
```

Renders one animation frame onto the canvas.

Rendering order: background → track glow → track surface → track edges →
centerline dashes → tire marks → car body + front lighting accents.

When `renderOptions.frame` is present, the renderer also colors the four tire
corners from the packed Tier 4 tire tuple and draws pit entrance/stall
overlays from the packed pit-status tuple.

Mutates `renderState.tireMarks` and `renderState.ticksSinceLastMark`.

Parameters:
- `canvas` - Target canvas element.
- `spec` - Frozen track spec (geometry only).
- `envState` - Current physics state from the simulation.
- `renderState` - Mutable tire-mark accumulator.
- `transform` - World-to-canvas affine transform.
- `renderOptions` - Optional overlay configuration; defaults to no guidance overlay.

### resolvePitMetadataHeadingRadians

```ts
resolvePitMetadataHeadingRadians(
  pitBox: TrackPitBox,
): number | undefined
```

Resolves a pit heading when the pit metadata already carries orientation.

Parameters:
- `pitBox` - Team pit metadata descriptor.

Returns: Optional heading value in radians.

### resolvePitOverlayHeadingRadians

```ts
resolvePitOverlayHeadingRadians(
  spec: TrackSpec,
  pitBox: TrackPitBox,
  pitCenter: WorldPoint,
): number
```

Resolves pit-overlay rotation from pit metadata or nearby spline tangent.

Parameters:
- `spec` - Frozen track geometry used for nearest-sample lookup.
- `pitBox` - Team pit metadata descriptor.
- `pitCenter` - World-space center used to locate the nearest spline sample.

Returns: Overlay heading in world radians.

### resolveRenderCars

```ts
resolveRenderCars(
  envState: EnvironmentState,
): readonly RenderCarState[]
```

Resolves the render roster from the authoritative environment state.

Parameters:
- `envState` - Current environment snapshot.

Returns: Ordered list of car states to draw for this frame.

### resolveRenderCarTeamIndex

```ts
resolveRenderCarTeamIndex(
  overlayFrame: RacingRenderOverlayFrame | undefined,
  envState: EnvironmentState,
  renderCar: RenderCarState,
  carIndex: number,
  focusCarIndex: number,
): 0 | 1
```

Resolves the team index used for per-car pit-colored outlines.

Parameters:
- `overlayFrame` - Optional packed worker frame.
- `envState` - Current simulation state.
- `renderCar` - Car being rendered.
- `carIndex` - Render roster index.
- `focusCarIndex` - Focused car index for packed fallbacks.

Returns: Normalized team index (`0` or `1`).

### resolveRenderTireState

```ts
resolveRenderTireState(
  envState: EnvironmentState,
  renderCar: RenderCarState,
  packedTireState: Float32Array<ArrayBufferLike> | undefined,
  carIndex: number,
  focusCarIndex: number,
): TireStateTuple
```

Resolves the rendered tire-health tuple for the focused car.

Parameters:
- `envState` - Current environment snapshot.
- `packedTireState` - Optional packed worker tire array.
- `focusCarIndex` - Focused car row inside the packed worker frame.

Returns: Ordered tire-health tuple for renderer use.

### resolveSmoothedPitHeadingRadians

```ts
resolveSmoothedPitHeadingRadians(
  spec: TrackSpec,
  pitCenter: WorldPoint,
  nearestSampleIndex: number,
): number
```

Resolves a stable pit heading using a weighted circular mean of nearby
spline tangents around the nearest lane-center sample.

Parameters:
- `spec` - Frozen track geometry containing spline samples.
- `pitCenter` - World-space pit center used for proximity weighting.
- `nearestSampleIndex` - Index of the nearest spline sample.

Returns: Smoothed tangent heading in radians.

### resolveStartLineSquareCount

```ts
resolveStartLineSquareCount(
  laneWidthWorldUnits: number,
  squarePitchWorldUnits: number,
): number
```

Resolves how many start-line squares can fit across the current lane width.

Parameters:
- `laneWidthWorldUnits` - Current lane width in world units.
- `squarePitchWorldUnits` - Square side plus inter-square gap.

Returns: Clamped square count for crosswalk readability.

### resolveStartLineSquareSide

```ts
resolveStartLineSquareSide(
  laneWidthWorldUnits: number,
): number
```

Resolves one start-line square side length from the current lane width.

Parameters:
- `laneWidthWorldUnits` - Current lane width in world units.

Returns: Clamped square side length in world units.

### resolveTeamPitColor

```ts
resolveTeamPitColor(
  teamIndex: 0 | 1,
): string
```

Resolves the pit palette color for the supplied team index.

Parameters:
- `teamIndex` - Team index (`0 = Team A`, `1 = Team B`).

Returns: Team pit color used by both pit overlays and car outlines.

### resolveTireHealthColor

```ts
resolveTireHealthColor(
  tireHealth: number,
): string
```

Resolves the live tire marker color from tire health.

Parameters:
- `tireHealth` - Normalized tire-health value.

Returns: CSS color string for the tire marker.

### resolveVisiblePitTeamIndex

```ts
resolveVisiblePitTeamIndex(
  overlayFrame: RacingRenderOverlayFrame | undefined,
  envState: EnvironmentState,
  focusCarIndex: number,
): 0 | 1 | undefined
```

Resolves which team's pit overlays should be visible in the current frame.

Parameters:
- `overlayFrame` - Optional packed worker frame.
- `envState` - Current simulation state.
- `focusCarIndex` - Focused car index for packed frames.

Returns: Team index for pit visibility, or `undefined` when unavailable.

### resolveWorldEdgePaddingPx

```ts
resolveWorldEdgePaddingPx(
  canvas: HTMLCanvasElement,
  requestedEdgePaddingPx: number | undefined,
): number
```

Resolves the edge padding used by world-to-canvas fitting.

Parameters:
- `canvas` - Target canvas.
- `requestedEdgePaddingPx` - Optional explicit edge padding.

Returns: Clamped edge padding in canvas pixels.

### TireMark

One sampled point in the fading tire-mark trail.

### traceClosedSamplePath

```ts
traceClosedSamplePath(
  ctx: CanvasRenderingContext2D,
  centerlinePoints: readonly TrackSamplePoint[],
  transform: WorldTransform,
): void
```

Traces the sampled centerline path.

Parameters:
- `ctx` - 2D rendering context.
- `centerlinePoints` - Ordered sampled centerline points.
- `transform` - World-to-canvas affine transform.

### traceClosedWorldPath

```ts
traceClosedWorldPath(
  ctx: CanvasRenderingContext2D,
  worldPoints: readonly WorldPoint[],
  transform: WorldTransform,
): void
```

Traces one closed polyline derived from world-space points.

Parameters:
- `ctx` - 2D rendering context.
- `worldPoints` - Ordered world-space points.
- `transform` - World-to-canvas affine transform.

### traceTrackRibbonPath

```ts
traceTrackRibbonPath(
  ctx: CanvasRenderingContext2D,
  trackRenderGeometry: TrackRenderGeometry,
  transform: WorldTransform,
): void
```

Traces the filled ribbon path bounded by the left and right track edges.

Parameters:
- `ctx` - 2D rendering context.
- `trackRenderGeometry` - Cached world-space track geometry.
- `transform` - World-to-canvas affine transform.

### TrackRenderGeometry

Cached spline-derived geometry used by transform and draw helpers.

### TrackSamplePoint

One sampled centerline point with its interpolated track width.

### WorldPoint

World-space coordinate pair used while building cached track geometry.

### WorldTransform

Affine transform mapping world units to canvas pixels.

Computed once per episode from the track bounding box and canvas size.
Recompute if the canvas is resized.

### WorldTransformOptions

Optional tuning for world-to-canvas fit behavior.
