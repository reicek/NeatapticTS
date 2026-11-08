# engine/passes

NeatRacing car render pass (slice P1-S4-D; spec sections 6.5, 6.6, 7.3.1,
7.4 in the NeatRacing spec document).

Owns the on-track car draw: body and wheel voxel-bin meshes under the
shared lighting model, plus the per-car blob shadow. Structural sibling of
`grid.pass.ts` (numeric GL enums, exported GLSL ES 3.00 sources, a compile
helper that throws with the pass label, throw-on-missing uniform
resolution, a lazy warmup seam) with two programs under one warmup key:

1. The lit voxel program draws bins-derived meshes two-sided (spec 7.3.1:
   CULL_FACE disabled, per-fragment view-flip when the world normal points
   away from the viewer), reconstructs the axis normal in world space
   through the model matrix, applies the draw-time team remap (AC-014:
   uTeamColor swaps albedo wherever paletteIndex equals
   NEATRACING_TEAM_REMAP_SLOT — never baked into vertex data), and embeds
   NEATRACING_LIGHTING_GLSL_CHUNK so the shader cannot drift from the
   lighting module's TS mirror. exp2 fog mixes to uFogColor over each
   fragment's world distance to the camera.
2. The blob shadow program draws one ground quad per car at world
   y + NEATRACING_BLOB_SHADOW_Y_OFFSET_M (spec-pinned 0.01 m anti
   z-fight) with depth-write off, depth-test on, and SRC_ALPHA blending;
   a smoothstep soft edge falls off from named constants and the XZ scale
   covers the body's full XZ footprint with a margin.

Draw order per frame: cars sort far -> near by squared XZ camera distance
inside a reused scratch array (strict-compare insertion sort keeps equal
distances stable), then all blob shadows draw (loop 1), then each car's
body plus its four wheel-mounted wheels draw (loop 2). The arcade drift
yaw (AC-047C) rotates the body into the slide, and because the tires are
ATTACHED to the body, the wheel chain and the car shadow inherit that same
drift yaw (AC-078) instead of staying on the base yaw. No per-frame
allocation: every matrix and sort scratch is module-scope and reused (P9).

## engine/passes/car.pass.ts

### CAR_LIT_FRAGMENT_SHADER_SOURCE

Lit fragment shader: two-sided axis normal reconstructed through the model
matrix, per-fragment view flip (spec 7.3.1), draw-time team remap
(AC-014), and the shared lighting chunk over exp2 fog.

### CAR_LIT_UNIFORM_NAMES

Lit-program uniform names: shared lighting uniforms + car-specific feeds.

### CAR_LIT_VERTEX_SHADER_SOURCE

Lit vertex shader: transforms a car voxel vertex into world space and
passes color, axis, emissive, palette slot, and camera distance through.

### CAR_MESH_FLOATS_PER_VERTEX

Floats per expanded car vertex: position, color, axis, emissive, slot.

### CAR_MESH_VERTEX_STRIDE_BYTES

Bytes per expanded car vertex (9 floats x 4 bytes).

### CAR_SHADOW_FRAGMENT_SHADER_SOURCE

Blob shadow fragment shader: smoothstep soft edge over the dark tint,
peaking at NEATRACING_BLOB_SHADOW_MAX_ALPHA and shaped by
NEATRACING_BLOB_SHADOW_FALLOFF_POWER (both interpolated from named
constants so the GLSL cannot drift from the module contract).

### CAR_SHADOW_VERTEX_SHADER_SOURCE

Blob shadow vertex shader: unit quad centered on the car's XZ footprint.

### CarMeshYAnchor

Expands a voxel bin into the car pass vertex format (pure, GL-free).

Each quad becomes six non-indexed vertices (triangles c0-c1-c2 and
c0-c2-c3 over corners (0,0), (w,0), (w,h), (0,h)) on a 9-float stride
[x, y, z, r, g, b, normalAxis, emissive, paletteIndex]. X/Z grid positions
map to model-space meters centered on the model origin (lattice-centered);
the Y mapping is pinned by the `yAnchor` argument so the two authored
model kinds land where the G9 mount contract expects them:
- `'ground'` (car bodies, authored with row 0 down on the floor):
  `modelY = row * voxelSize`, so the body's floor row sits exactly on the
  car-frame ground plane `y = 0` and the body rises above the road.
  Centering the lattice instead would sink a bottom-authored body meters
  underground where every fragment fails the depth test against the road
  (invisible car bug).
- `'center'` (wheels, authored as a symmetric 32³ disc whose surface
  center is the lattice midpoint `dimY * 0.5` = 16 cells = the hub
  height): `modelY = (row - dimY * 0.5) * voxelSize`, so the disc's
  geometric center is the rotation pivot AND the tire's bottom face sits
  at exactly `-NEATRACING_WHEEL_HUB_HEIGHT_M`; the mount then raises the
  disc by the hub height, landing the contact patch at exactly `y = 0`
  (AC-015). A `(dimY - 1) * 0.5` offset would float the tire half a cell
  (17.6 mm) above the road.

Colors resolve from `buildVoxelMaterialTable` (bytes / 255) and emissive is
glow / 255 with the LD-23 black exception already folded in (black palette
entries carry emissive 0). The NEATRACING_TEAM_REMAP_SLOT color stays
UNBAKED because the team remap happens at draw time (AC-014).

Tangent rule (matching the shared mesher `FACE_TANGENT_AXES` exactly —
the mesher records `position[widthAxis] = u` and `position[heightAxis] =
v`, so a mismatch here transposes every non-square face and stretches it
to the swapped extents): width runs along the first tangent axis, height
along the second: axis 0 -> (y, z), axis 1 -> (x, z), axis 2 -> (x, y);
the fixed axis is `quad.position[axis]`.

Parameters:
- `model` - Decoded voxel bin model (dims, palette, quads).
- `yAnchor` - Vertical anchoring contract: `'ground'` for bodies
authored with row 0 on the floor, `'center'` for lattice-centered wheel
discs. Defaults to `'ground'`.

Returns: The expanded mesh plus its XZ half-extent in meters.

Example:

const mesh = expandVoxelBinModelToCarMesh(bodyBin);
// mesh.vertexCount === 6 * bodyBin.quads.length; 9 floats per vertex
const wheelMesh = expandVoxelBinModelToCarMesh(wheelBin, 'center');

### CarPass

Car pass handle returned by createCarPass.

### CarPassCarState

Car pose inside one draw frame.

### CarPassDrawState

Per-frame draw state accepted by CarPass.draw.

### CarPassModels

Models the pass renders: the car body bin plus one wheel bin.

### CarVoxelMesh

Mesh produced by {@link expandVoxelBinModelToCarMesh}: interleaved
non-indexed triangles ready for one FLOAT-attributed VBO per model.

### compileAndLinkCarProgram

```ts
compileAndLinkCarProgram(
  gl: WebGL2RenderingContext,
  vertexShaderSource: string,
  fragmentShaderSource: string,
  programLabel: string,
): CompiledCarProgram
```

Compiles and links one of the car pass programs from its two stages.

Parameters:
- `gl` - WebGL2 context.
- `vertexShaderSource` - Source for the vertex stage.
- `fragmentShaderSource` - Source for the fragment stage.
- `programLabel` - Label used in every rejection message.

Returns: The linked program plus its two shader handles.

### compileCarShader

```ts
compileCarShader(
  gl: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
  shaderLabel: string,
): WebGLShader
```

Compiles one car pass shader stage, rejecting failures with the pass label
(grid.pass.ts parity).

Parameters:
- `gl` - WebGL2 context.
- `shaderType` - gl.VERTEX_SHADER (0x8B31) or gl.FRAGMENT_SHADER (0x8B30).
- `shaderSource` - GLSL ES 3.00 source.
- `shaderLabel` - Stage label used in the rejection message.

Returns: The compiled shader handle.

### CompiledCarProgram

Compiled program bundle: the linked program plus its two live shaders.

### composeSortedCarOrder

```ts
composeSortedCarOrder(
  state: CarPassDrawState,
): void
```

Fills the sort scratch with the far -> near draw order. Stable insertion
sort over squared XZ camera distances: equal distances keep array order,
so the ordering is deterministic frame to frame.

Parameters:
- `state` - Frame draw state (cars + camera world position).

### computeFrontWheelSteerYawRad

```ts
computeFrontWheelSteerYawRad(
  car: CarPassCarState,
): number
```

Computes the EFFECTIVE front-wheel steer yaw (AC-088, Step 14R).

G-V5: the tire yaw-follow used to be the RAW steer echo, so a dramatic
turn was rendered even when the physics state never latched a slide. The
rendered yaw is now the angle the front axle can actually HOLD: the raw
request scaled by the front grip share the published severity leaves
behind. On a straight line (severity 0) the front axle still holds the
whole request, so the legacy driving feel is untouched; while the tires
are sliding the authority shrinks and the rendered yaw is no longer the
echo.

The pass does NOT invent a countersteer angle: the physics state
publishes no wheel angle, so the pass presents the grip-limited request
instead of fabricating a steering angle the simulation never computed.

Parameters:
- `car` - The car being drawn.

Returns: The front-wheel steer yaw in radians.

### computeTireGrip

```ts
computeTireGrip(
  tractionSeverity: number,
  isFrontWheel: boolean,
): number
```

Computes the GRIP share of one tire from the PUBLISHED traction severity
(AC-088, Step 14R).

The severity arrives already published by the physics state; this helper
only turns it into the per-axle grip shares the fragment stage consumes.
The severity itself is NEVER re-derived here - not from the fixed sun,
not from the wheel spin, not from the steer input.

The two axles never share one grip value: the REAR axle breaks away
first (the drive tires do the sliding) and the FRONT axle keeps most of
its grip, so the four tires are shaded differently while the car slides
(G-V10).

Parameters:
- `tractionSeverity` - The published severity in [0, 1].
- `isFrontWheel` - True for the steered front axle, false for the rear.

Returns: The grip share in [0, 1] (1 = fully hooked, 0 = fully sliding).

### createCarPass

```ts
createCarPass(
  gl: WebGL2RenderingContext,
  models: CarPassModels,
): CarPass
```

Creates the car render pass: compiles the lit + blob shadow programs
eagerly, resolves uniforms throw-on-missing, expands and uploads both
voxel bins plus the shadow quad, and resolves the four wheel mount
transforms once at creation (they allocate per call; keeping them
pre-resolved keeps the draw path allocation-free, P9).

Parameters:
- `gl` - WebGL2 context.
- `models` - Body and wheel voxel bins to render ( {@link CarPassModels} ).

Returns: The car pass handle ( {@link CarPass} ).

Example:

const carPass = createCarPass(gl, { body: bodyBin, wheel: wheelBin });
carPass.draw({ viewProjection, cameraWorld, cars });
carPass.dispose();

### ensureCarSortScratchCapacity

```ts
ensureCarSortScratchCapacity(
  carCount: number,
): void
```

Grows the sort scratches to the fleet size once; steady-state frames never
reallocate because scratch capacity only ever grows (P9).

Parameters:
- `carCount` - Cars in the current frame.

### ensureWarmupCarPass

```ts
ensureWarmupCarPass(
  context: WebGL2RenderingContext,
): CarPass
```

Hydrates the singleton car pass for the warmup seam, rebuilding only when
the context changes. Compilation stays lazy until compileAndLink runs
(grid.pass.ts parity).

Parameters:
- `context` - WebGL2 context the warmup shader will run against.

Returns: The hydrated singleton pass.

### expandVoxelBinModelToCarMesh

```ts
expandVoxelBinModelToCarMesh(
  model: VoxelBinModel,
  yAnchor: CarMeshYAnchor,
): CarVoxelMesh
```

### identityMat4

```ts
identityMat4(
  out: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBufferLike>
```

Writes the identity matrix into a scratch buffer.

Parameters:
- `out` - Column-major 16-element destination.

Returns: The same buffer for call-site chaining.

### multiplyMat4

```ts
multiplyMat4(
  out: Float32Array<ArrayBufferLike>,
  a: Float32Array<ArrayBufferLike>,
  b: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBufferLike>
```

Multiplies two column-major matrices into an output scratch buffer:
`out = a * b` (a point is transformed by b first, then a). Call sites must
pass a dedicated scratch that aliases neither input; every scratch here is
preallocated at module scope to keep the draw path allocation-free (P9).

Parameters:
- `out` - 16-element output scratch (must not alias a or b).
- `a` - Left operand, applied second in the transform chain.
- `b` - Right operand, applied first in the transform chain.

Returns: The same output buffer for call-site chaining.

### NEATRACING_BLOB_SHADOW_BOUNDS_SCALE

Blob shadow XZ scale margin over the car's full footprint including the
tire contact patches. The shadow quad world width equals 2 x the combined
body/wheel half extent times this scale, so 1.35 covers the full car
footprint plus tires with a generous safety margin (OBS-21).

### NEATRACING_BLOB_SHADOW_WHEEL_PATCH_SCALE_M

World-space side length of each per-wheel contact-patch shadow quad.
Slightly larger than the tire radius so the darkened footprint covers the
ground directly under every tire (OBS-24).

### NEATRACING_BLOB_SHADOW_Y_OFFSET_M

World y offset for blob shadows (spec section 6.6). The shadow is drawn
just above the opaque road-edge paint layer so it is not depth-tested
away by the honeycomb floor paint; depth-write stays off so it reads as
a ground decal.

### NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR

Bright end of the fixed-space tire gradient (toward the top-front light).

### NEATRACING_TIRE_GRADIENT_DARK_FACTOR

Dark end of the fixed-space tire gradient (away from the top-front light).

### NEATRACING_TIRE_GRADIENT_MIN_SPREAD

Minimum non-degenerate spread between the two ends of the fixed-space tire
gradient (AC-088): the bright factor minus the dark factor, in the
dimensionless multiplier units the fragment stage mixes between.

This is a FLOOR, not a live value: the production spread is
`NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR - NEATRACING_TIRE_GRADIENT_DARK_FACTOR`
= 1.35 (1.7 - 0.35), which already clears the floor with margin. The named
export exists so the contract is a checkable constant instead of a bare
literal in an assertion: the fragment stage can only differentiate the tire
when the two ends of the gradient differ, and this floor is the minimum
difference that keeps the effect visible.

The rendered gradient itself is a fragment-stage effect, so it is NOT
measurable from JavaScript; the visible outcome is carried by the AC-090
user visual gate, and this constant pins the seam INPUTS.

Example:

0.5

### NEATRACING_TIRE_GRADIENT_RUBBER_MAX_CHANNEL

Palette colors at or below this max channel are treated as tire rubber and
receive the fixed-space vertical gradient; brighter rim/hub colors are
skipped.

### quadCornerGridX

```ts
quadCornerGridX(
  quad: VoxelBinQuad,
  u: number,
): number
```

Resolves a corner's grid x coordinate per the shared mesher tangent rule.

Parameters:
- `quad` - Source quad (minimum lattice corner + tangent extents).
- `u` - Width extent along the quad's first tangent axis.

Returns: Grid x of the expanded corner.

### quadCornerGridY

```ts
quadCornerGridY(
  quad: VoxelBinQuad,
  u: number,
  v: number,
): number
```

Resolves a corner's grid y coordinate per the ascending tangent rule.

Parameters:
- `quad` - Source quad (minimum lattice corner + tangent extents).
- `u` - Width extent along the quad's first tangent axis.
- `v` - Height extent along the quad's second tangent axis.

Returns: Grid y of the expanded corner.

### quadCornerGridZ

```ts
quadCornerGridZ(
  quad: VoxelBinQuad,
  v: number,
): number
```

Resolves a corner's grid z coordinate per the shared mesher tangent rule.

Parameters:
- `quad` - Source quad (minimum lattice corner + tangent extents).
- `v` - Height extent along the quad's second tangent axis.

Returns: Grid z of the expanded corner.

### resolveCarLitAttributeLocations

```ts
resolveCarLitAttributeLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
): Record<"aPosition" | "aColor" | "aNormalAxis" | "aEmissive" | "aPaletteIndex", number>
```

Resolves the five lit-program attribute locations in expansion float order
(position, color, normal axis, emissive, palette index).

Parameters:
- `gl` - WebGL2 context.
- `program` - Linked lit program handle.

Returns: Attribute location table used by vertexAttribPointer feeds.

### resolveCarUniformLocations

```ts
resolveCarUniformLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  names: readonly string[],
): Record<string, WebGLUniformLocation>
```

Resolves one program's uniform locations with throw-on-missing semantics
so a shader/name drift fails loudly at pass creation (grid.pass.ts parity).

Parameters:
- `gl` - WebGL2 context.
- `program` - Linked program handle.
- `names` - Uniform names to resolve, in feed order.

Returns: Location table keyed by uniform name.

### rotationYMat4

```ts
rotationYMat4(
  out: Float32Array<ArrayBufferLike>,
  angleRad: number,
): Float32Array<ArrayBufferLike>
```

Writes a rotation-around-world-up matrix into a scratch buffer
(column-major: col0 = (cos, 0, -sin), col2 = (sin, 0, cos)).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `angleRad` - Yaw angle in radians.

Returns: The same buffer for call-site chaining.

### rotationZMat4

```ts
rotationZMat4(
  out: Float32Array<ArrayBufferLike>,
  angleRad: number,
): Float32Array<ArrayBufferLike>
```

Writes a rotation-around-the-model-depth-axis matrix into a scratch buffer
(the pre-mount wheel model's spin axis; column-major).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `angleRad` - Spin angle in radians.

Returns: The same buffer for call-site chaining.

### scaleXYZMat4

```ts
scaleXYZMat4(
  out: Float32Array<ArrayBufferLike>,
  sx: number,
  sy: number,
  sz: number,
): Float32Array<ArrayBufferLike>
```

Writes an axis scale matrix into a scratch buffer (column-major).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `sx` - X scale factor.
- `sy` - Y scale factor.
- `sz` - Z scale factor.

Returns: The same buffer for call-site chaining.

### toCarWarmupProgram

```ts
toCarWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Builds the car warmup program descriptor for the shared warmup registry
(key 'car', NEATRACING_WARMED_PROGRAM_KEYS[4]). Compilation is lazy: no GL
program exists until compileAndLink runs, and drawOffscreen pushes the
zero-pose frame through the same draw path production uses.

Parameters:
- `context` - WebGL2 context the warmup shader will run against.

Returns: The warmup program descriptor keyed 'car'.

Example:

programRegistry.register(toCarWarmupProgram(gl));

### translationMat4

```ts
translationMat4(
  out: Float32Array<ArrayBufferLike>,
  x: number,
  y: number,
  z: number,
): Float32Array<ArrayBufferLike>
```

Writes a translation matrix into a scratch buffer (column-major).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `x` - Translation x in meters.
- `y` - Translation y in meters.
- `z` - Translation z in meters.

Returns: The same buffer for call-site chaining.

### uniform3fColor

```ts
uniform3fColor(
  gl: WebGL2RenderingContext,
  location: WebGLUniformLocation,
  color: LightingVec3,
): void
```

Feeds a normalized [0, 1] color triple into a vec3 uniform; no raw 0-255
byte ever reaches a uniform through this helper.

Parameters:
- `gl` - WebGL2 context.
- `location` - Resolved uniform location.
- `color` - Color already normalized to [0, 1].

### uploadCarFloat32Buffer

```ts
uploadCarFloat32Buffer(
  gl: WebGL2RenderingContext,
  data: Float32Array<ArrayBufferLike>,
): WebGLBuffer
```

Uploads one Float32 array into a STATIC_DRAW buffer.

Parameters:
- `gl` - WebGL2 context.
- `data` - Vertex data to upload.

Returns: The uploaded buffer handle.

## engine/passes/grid.pass.ts

NeatRacing world-space ground grid pass (spec P1-S2, slice P1-S2-B,
AC-005): a full-screen raycast ground plane whose fragment shader draws
the world-coherent 1.0-unit grid with constant *screen-space* line width.

Contract highlights (spec section 6.4.2 + AC-005):
- Line half-widths are derived per fragment from `fwidth(vWorldXZ)` so a
  2 px core + 4 px halo stay constant in pixels at any zoom (G7: no
  `gl.lineWidth`, which WebGL2/ANGLE does not honor above 1).
- Line alpha fades with camera distance from 0.58 (near) to a 0.05 far
  floor across `NEATRACING_DRAW_DISTANCE_M` (320 m, TP-16).
- The grid is world-anchored: line phase comes from world-space XZ only,
  so it stays stable as the player moves.
- The plane sits at `NEATRACING_GRID_PLANE_Y_OFFSET_M` (-0.01 m) below the
  ground plane and the pass renders with depth test/write disabled, so it
  can never z-fight with track geometry drawn above it.

The pure TypeScript functions below are the Jest-testable mirrors of the
fragment-shader math; the GLSL sources encode the identical formulas.

### compileGridShader

```ts
compileGridShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader
```

Compiles one shader stage or throws a descriptive error carrying the stage
label and the driver info log.

### createGridPass

```ts
createGridPass(
  gl: WebGL2RenderingContext,
): GridPass
```

Creates the grid pass: compiles and links the program exactly once and
resolves all uniform locations. The returned handle never recompiles or
re-uploads; `draw` only sets uniforms and issues a full-screen triangle
draw with blending enabled and depth test/write disabled (background-layer
composition, anti-z-fighting vs the track).

Parameters:
- `gl` - WebGL2 rendering context to bind the pass to.

Returns: The grid pass handle.

### gridAxisLineDistance

```ts
gridAxisLineDistance(
  worldCoord: number,
  spacing: number,
): number
```

Distance from a world coordinate to the nearest grid axis line for a given
spacing: the TS mirror of `abs(fract(coord / spacing + 0.5) - 0.5) *
spacing`. Periodic per spacing, symmetric around each line, and purely
world-anchored (camera-independent).

Parameters:
- `worldCoord` - World-space coordinate along one axis.
- `spacing` - Grid spacing in world units (must be positive).

Returns: Distance in world units to the nearest line, in [0, spacing / 2].

### gridBandCoverage01

```ts
gridBandCoverage01(
  lineDistance: number,
  threshold: number,
): number
```

Smoothstep band coverage for a fragment at `lineDistance` from a line with
half-width `threshold`: 1 at the line center, 0.5 at half the threshold,
0 at and beyond the threshold. Degenerate (non-positive) thresholds map to
1 only exactly on the line. TS mirror of the shader band term.

Parameters:
- `lineDistance` - Distance from the line center in world units.
- `threshold` - World-space band half-width (from the fwidth mirror).

Returns: Coverage in [0, 1].

### gridFogFactor01

```ts
gridFogFactor01(
  distanceM: number,
  drawDistanceM: number,
): number
```

Normalized camera-distance fog factor clamped to [0, 1]: 0 at the camera,
1 at (and beyond) the draw distance. TS mirror of the shader fog term.

Parameters:
- `distanceM` - Camera distance in meters.
- `drawDistanceM` - Draw distance in meters (must be positive).

Returns: Fog factor in [0, 1].

### gridFragmentAlpha

```ts
gridFragmentAlpha(
  worldX: number,
  worldZ: number,
  fwidthWorld: number,
  distanceM: number,
  drawDistanceM: number,
): number
```

Full fragment alpha for the grid at a world position: the maximum of the
core contribution (`baseAlpha * coreCoverage`) and the halo contribution
(`baseAlpha * GLOW_ALPHA_MULTIPLIER * glowCoverage`), where coverages are
the per-axis maxima over X and Z. TS mirror of the shader `main` body.

Parameters:
- `worldX` - World-space X coordinate of the fragment.
- `worldZ` - World-space Z coordinate of the fragment.
- `fwidthWorld` - `fwidth` derivative double of the world coordinate.
- `distanceM` - Camera distance in meters.
- `drawDistanceM` - Draw distance in meters (must be positive).

Returns: Fragment alpha in [0, MAX_ALPHA].

### gridLineAlpha

```ts
gridLineAlpha(
  distanceM: number,
  drawDistanceM: number,
): number
```

Depth-fade line-center alpha: `max(MIN_ALPHA, MAX_ALPHA * (1 - fogFactor))`,
i.e. 0.58 at the camera fading linearly to the 0.05 far floor across the
draw distance. TS mirror of the shader alpha envelope (G7 MIN/MAX form).

Parameters:
- `distanceM` - Camera distance in meters.
- `drawDistanceM` - Draw distance in meters (must be positive).

Returns: Line-center alpha in [MIN_ALPHA, MAX_ALPHA].

### gridLineThresholdWorld

```ts
gridLineThresholdWorld(
  widthPx: number,
  fwidthWorld: number,
): number
```

Computes the world-space line half-width for a pixel width at the current
fragment footprint: the TS mirror of `threshold = widthPx * fwidth(coord)`.
Because `fwidth` is the per-pixel world-space derivative of the coordinate,
the resulting band is constant in screen pixels at any distance.

Parameters:
- `widthPx` - Desired band width in screen pixels.
- `fwidthWorld` - `fwidth` derivative double of the world coordinate.

Returns: World-space half-width threshold to feed the band coverage.

### GridPass

The grid pass handle: create once, draw every frame.

### GridPassDrawState

Per-draw inputs for the grid pass.

### GridUniformLocations

Resolved uniform locations keyed by uniform name.

### invertMatrix44

```ts
invertMatrix44(
  out: Float32Array<ArrayBufferLike>,
  m: Float32Array<ArrayBufferLike>,
): boolean
```

Inverts a 4x4 column-major matrix into `out` using the adjugate/determinant
method. The source and destination must not alias.

Parameters:
- `out` - 16-float destination (column-major).
- `m` - 16-float source matrix (column-major).

Returns: `true` when the matrix was invertible, `false` if singular.

### NEATRACING_DRAW_DISTANCE_M

Draw distance in meters spanning the depth fade (TP-16).

### NEATRACING_GRID_FRAGMENT_SHADER

Fragment shader (GLSL ES 3.00) for the grid pass: unprojects each fragment
through `uInvViewProjection` from its exact interpolated clip position,
casts the camera ray, and intersects the horizontal ground plane at
`uPlaneY`. Upward rays (sky) discard, so the grid stays flat on the ground
and the horizon is always horizontal. Line half-widths are derived per
fragment from `fwidth(worldXZ)` so a 2 px core + 4 px halo stay constant in
pixels at any zoom (G7: no `gl.lineWidth`, which WebGL2/ANGLE does not
honor above 1).

### NEATRACING_GRID_GLOW_ALPHA_MULTIPLIER

Alpha multiplier applied to the halo band relative to the line-center alpha.

### NEATRACING_GRID_GLOW_WIDTH_PX

Halo (glow) grid line width in screen pixels around the core line.

### NEATRACING_GRID_LINE_COLOR_RGB

Grid line color as normalized RGB. The spec pins no grid color (AC-005
references only the LD-24 neon family), so this mirrors the AC-004
starfield cyan rgba(95,255,255,1) to keep the palette small and coherent.

### NEATRACING_GRID_LINE_WIDTH_PX

Core grid line width in screen pixels (constant via fwidth, not gl.lineWidth).

### NEATRACING_GRID_MAX_ALPHA

Near-camera alpha for grid line centers.

### NEATRACING_GRID_MIN_ALPHA

Far-floor alpha for grid lines (the visible-at-distance floor).

### NEATRACING_GRID_PLANE_Y_OFFSET_M

Grid plane height offset below the ground plane (anti-z-fighting, G9).

### NEATRACING_GRID_PROGRAM_KEY

Warmup/draw registry key; must stay a member of the program registry.

### NEATRACING_GRID_SPACING_WORLD

World spacing between grid lines, mirroring the 1.0 world-unit floor grid.

### NEATRACING_GRID_VERTEX_SHADER

Vertex shader (GLSL ES 3.00) for the grid pass: full-screen triangle that
covers normalized device coordinates. The vertex stage only forwards the
clip-space x/y as the `vClip` varying — clip coordinates are an affine
function of the interpolated barycentrics, so `vClip` interpolates exactly
across the giant triangle. Every projective quantity (unprojection, ray
direction, ground-plane hit, world position) is computed per fragment in
the fragment shader, because linearly interpolating per-vertex unprojections
across a fullscreen triangle is only exact at the 3 vertices and produces
inclined garbage lines everywhere else (AC-005: the grid must be flat,
world-anchored ground, never camera-tilted lines in the sky).

### toGridWarmupProgram

```ts
toGridWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Exposes the grid pass as a shader warmup program (AC-040 seam).

The returned entry keeps compilation lazy: `compileAndLink` builds the
pass (compiling and linking the program), and `drawOffscreen` issues the
zero-pose ground-grid draw through the shared pass path so the driver
warms the exact render pipeline used on screen. The context passed to
`drawOffscreen` is accepted for interface parity; the pass is bound to
the context supplied at construction.

Parameters:
- `context` - WebGL2 context the pass resources will be built against.

Returns: Warmup entry keyed to the grid registry slot (second warmed key).

Example:

const warmup = toGridWarmupProgram(gl);
warmup.compileAndLink();
warmup.drawOffscreen(gl);

## engine/passes/bloom.pass.ts

NeatRacing cheap bloom post-processing pass (slice P1-S4-E; spec section
6.6 G8, AC-016, AC-035).

G8 concretizes the glow-byte bloom pipeline as THREE fullscreen programs
behind one pass and three warmup registry keys:

1. `bloom-threshold` (stage 1): the bright pass samples the scene texture
   and keeps only pixels above NEATRACING_BLOOM_BRIGHT_THRESHOLD, scaled
   by a Rec709 luminance dot (named weights 0.2126 / 0.7152 / 0.0722).
   Bright-pass input selection (glow voxels minus starfield/pulses per
   section 6.6) is the host's composition contract: this pass consumes
   exactly ONE scene texture.
2. `bloom-blur` (stage 2): ONE separable blur, run twice — horizontally
   into ping-pong target A, then vertically into ping-pong target B — on
   the half-res 640x360 target with NEATRACING_BLOOM_MAX_TAPS = 13
   (the spec's named tap cap; the used NEATRACING_BLOOM_TAP_COUNT equals
   the cap exactly). Tap offsets/weights are gaussian with
   NEATRACING_BLOOM_BLUR_SIGMA_TEXELS, normalized to sum to one, built in
   TypeScript by {@link computeBloomBlurTaps} and interpolated as GLSL
   const arrays so the shader cannot drift from the TS mirror.
3. `bloom-additive` (stage 3): the composite samples the scene texture
   and the blurred bloom and emits
   `scene + bloom * NEATRACING_BLOOM_COMPOSITE_INTENSITY` through
   `blendFunc(GL_ONE, GL_ONE)` additive blending onto the DEFAULT
   framebuffer — which the host has cleared and pre-filled with its
   additive background (starfield/pulses drawn direct, pulse-pass
   pattern). The world exists per-frame ONLY as the host-owned scene
   FBO texture, so this composite draw is the ONLY scene transfer
   to the display, and the world is rendered exactly once there.
   Never emit bloom-only output (the world would vanish from the
   display) and never pre-draw the world to the default framebuffer
   (the scene would double).

Internal render targets are RGBA8 half-res 640x360 FBOs (bright, blur A,
blur B) validated with `checkFramebufferStatus` against
`GL_FRAMEBUFFER_COMPLETE` at creation. The AC-035 off-switch is the
`enabled === false` frame state: zero GL operations, no viewport change,
no state churn. Bloom sits FIRST on the drop list before any LOD swap.

Structural sibling of `grid.pass.ts` / `car.pass.ts` (numeric GL enums,
exported GLSL ES 3.00 sources, compile helpers that throw with the pass
label, throw-on-missing uniform resolution, warmup seam) — with three
programs and no vertex-mesh meshing: every stage draws the same shared
fullscreen NDC TRIANGLE_STRIP quad.

### BloomBlurTap

One blur tap: separable offset in texels plus its normalized weight.

### BloomPass

Bloom pass handle returned by createBloomPass.

### BloomPassDrawFrame

Per-frame draw state accepted by BloomPass.draw.

### BloomRenderTarget

One internal half-res render target: an RGBA texture plus complete FBO.

### compileAndLinkBloomProgram

```ts
compileAndLinkBloomProgram(
  gl: WebGL2RenderingContext,
  fragmentShaderSource: string,
  programLabel: string,
): CompiledBloomProgram
```

Compiles and links one bloom stage program from its two stages.

Parameters:
- `gl` - WebGL2 context.
- `fragmentShaderSource` - GLSL ES 3.00 fragment source.
- `programLabel` - Label used in every rejection message.

Returns: The linked program plus its two shader handles.

### compileBloomShader

```ts
compileBloomShader(
  gl: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
  shaderLabel: string,
): WebGLShader
```

Compiles one bloom pass shader stage, rejecting failures with the pass
label (grid.pass.ts parity).

Parameters:
- `gl` - WebGL2 context.
- `shaderType` - gl.VERTEX_SHADER (0x8B31) or gl.FRAGMENT_SHADER (0x8B30).
- `shaderSource` - GLSL ES 3.00 source.
- `shaderLabel` - Stage label used in the rejection message.

Returns: The compiled shader handle.

### CompiledBloomProgram

Compiled program bundle: the linked program plus its two live shaders.

### computeBloomBlurTaps

```ts
computeBloomBlurTaps(): readonly BloomBlurTap[]
```

Builds the separable blur tap table (pure, GL-free): a symmetric gaussian
kernel of {@link NEATRACING_BLOOM_TAP_COUNT} taps centered on zero with
weights normalized to sum to one, capped by
{@link NEATRACING_BLOOM_MAX_TAPS}. GLSL const arrays interpolated from
this table guarantee the TS mirror and the shader cannot drift.

Returns: One tap per lattice offset from -6 to +6 texels (inclusive).

Example:

const taps = computeBloomBlurTaps();
// taps.length === 13; taps[0].offsetTexels === -6; center weight is the
// maximum; Math.hypot(taps.reduce(...) ... sums to one within 1e-12.

### computeBloomBrightFactor

```ts
computeBloomBrightFactor(
  red: number,
  green: number,
  blue: number,
): number
```

Computes the bright-pass factor of one scene pixel (pure, GL-free mirror
of the threshold shader). Pixels below the threshold are dropped to zero;
above it, the factor ramps linearly to one at full white.

Parameters:
- `red` - Linear channel in [0, 1].
- `green` - Linear channel in [0, 1].
- `blue` - Linear channel in [0, 1].

Returns: The bright-pass multiplier clamped to [0, 1].

Example:

computeBloomBrightFactor(0, 0, 0); // 0 — black never blooms (LD-23)
computeBloomBrightFactor(1, 1, 1); // 1

### computeBloomLuminance

```ts
computeBloomLuminance(
  red: number,
  green: number,
  blue: number,
): number
```

Computes the Rec709 luminance of one color triple (pure, GL-free).

Parameters:
- `red` - Linear channel in [0, 1].
- `green` - Linear channel in [0, 1].
- `blue` - Linear channel in [0, 1].

Returns: The Rec709 luminance dot with NEATRACING_BLOOM_LUMA_WEIGHTS.

Example:

computeBloomLuminance(1, 0, 0); // 0.2126

### createBloomPass

```ts
createBloomPass(
  gl: WebGL2RenderingContext,
): BloomPass
```

Creates the bloom post-processing pass: compiles the threshold, separable
blur, and additive composite programs eagerly, resolves uniforms and the
`aCorner` attribute throw-on-missing, uploads the shared fullscreen quad,
and creates the three half-res 640x360 internal targets with
completeness checks (bright, blur ping-pong A, blur ping-pong B).

Parameters:
- `gl` - WebGL2 context.

Returns: The bloom pass handle ( {@link BloomPass} ).

Example:

const bloomPass = createBloomPass(gl);
bloomPass.draw({ sceneTexture, targetWidth: 1280, targetHeight: 720, enabled: true });
bloomPass.dispose();

### createBloomRenderTarget

```ts
createBloomRenderTarget(
  gl: WebGL2RenderingContext,
  label: string,
): BloomRenderTarget
```

Creates one half-res RGBA8 render target validated with
`checkFramebufferStatus`.

Parameters:
- `gl` - WebGL2 context.
- `label` - Label used in every rejection message.

Returns: The framebuffer + texture bundle.

### createWarmupZeroPoseFrame

```ts
createWarmupZeroPoseFrame(): BloomPassDrawFrame
```

Builds the enabled zero-pose warmup frame (1x1 white placeholder source,
1x1 composite viewport) against the hydrated singleton's placeholder
texture so the warmup drawOffscreen exercises the full 4-draw pipeline
(AC-040).

Returns: The `enabled` warmup draw frame.

### disposeWarmupBloomPass

```ts
disposeWarmupBloomPass(): void
```

Releases the warmup singleton (pass + placeholder texture) — used when
the context changes so stale GL handles are never reused.

### ensureWarmupBloomPass

```ts
ensureWarmupBloomPass(
  context: WebGL2RenderingContext,
): BloomPass
```

Hydrates the singleton bloom pass for the warmup seam, rebuilding only
when the context changes. Compilation stays lazy until compileAndLink
runs (car.pass.ts parity); the scene placeholder is created on the passed
context.

Parameters:
- `context` - WebGL2 context the warmup shaders will run against.

Returns: The hydrated singleton pass.

### formatBloomTapFloat

```ts
formatBloomTapFloat(
  value: number,
): string
```

Formats one tap number as a GLSL float literal (ES ES requires a decimal
point or exponent; integer offsets gain an explicit `.0` suffix).

Parameters:
- `value` - Tap value to format.

Returns: A GLSL-compatible float literal.

### NEATRACING_BLOOM_ADDITIVE_FRAGMENT_SHADER_SOURCE

Additive composite fragment shader (G8 stage 3): emits the full
`scene + bloom * kCompositeIntensity` color, which the pass draws
additively onto the default framebuffer that already holds the host's
additive background (starfield/pulses drawn direct). This composite is
the ONLY scene transfer to the display — the world is rendered exactly
once into the host-owned scene FBO texture this shader samples.

### NEATRACING_BLOOM_ADDITIVE_PROGRAM_KEY

Warmup program key (AC-040): additive composite stage, registry index 8.

### NEATRACING_BLOOM_BLUR_FRAGMENT_SHADER_SOURCE

Separable blur fragment shader (G8 stage 2): sums
{@link NEATRACING_BLOOM_TAP_COUNT} gaussian taps mirrored from
{@link computeBloomBlurTaps} as GLSL const arrays, offset along
`uBlurDirection` (set to (1/W, 0) or (0, 1/H) per draw).

### NEATRACING_BLOOM_BLUR_PROGRAM_KEY

Warmup program key (AC-040): separable blur stage, registry index 7.

### NEATRACING_BLOOM_BLUR_SIGMA_TEXELS

Gaussian sigma of the separable blur kernel, in half-res texels.

### NEATRACING_BLOOM_BRIGHT_THRESHOLD

Bright-pass luminance threshold in linear [0, 1] units (G8 stage 1):
pixels luminance-below this value contribute zero bloom.

### NEATRACING_BLOOM_COMPOSITE_INTENSITY

Composite intensity multiplier for the additive bloom contribution
(G8 stage 3: additive composite onto the scene, TP-10 tuning value).

### NEATRACING_BLOOM_LUMA_WEIGHTS

Rec709 luminance weights used by both the TS threshold mirror and the
interpolated GLSL const (shader cannot drift from the TS math).

### NEATRACING_BLOOM_MAX_TAPS

Named tap cap (G8: "named tap cap <= 13 taps"); per-direction tap counts
must never exceed this value.

### NEATRACING_BLOOM_TAP_COUNT

Tap count the separable blur actually uses per direction; pinned at the
cap (13, odd so the gaussian keeps a center tap).

### NEATRACING_BLOOM_TARGET_HEIGHT

Half-res bloom target height (G8: derived from the fixed 720p backing
store height; NEATRACING_FIXED_RENDER_HEIGHT / 2).

### NEATRACING_BLOOM_TARGET_WIDTH

Half-res bloom target width (G8: "half-res 640x360 target" derived from
the fixed 720p backing store's 1280 width; NEATRACING_FALLBACK_CANVAS_
WIDTH / 2).

### NEATRACING_BLOOM_THRESHOLD_FRAGMENT_SHADER_SOURCE

Threshold fragment shader (G8 stage 1): Rec709 luminance dot interpolated
from the TS mirror constants so the shader cannot drift.

### NEATRACING_BLOOM_THRESHOLD_PROGRAM_KEY

Warmup program key (AC-040): threshold stage, registry index 6.

### NEATRACING_BLOOM_VERTEX_SHADER_SOURCE

Fullscreen vertex shader shared by all three bloom stages: maps the unit
NDC strip corner to UV space.

### resolveBloomCornerAttributeId

```ts
resolveBloomCornerAttributeId(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  programLabel: string,
): number
```

Resolves the `aCorner` fullscreen attribute for one program.

Parameters:
- `gl` - WebGL2 context.
- `program` - Linked program handle.
- `programLabel` - Stage label used in the rejection message.

Returns: The attribute location for `aCorner`.

### resolveBloomUniformLocations

```ts
resolveBloomUniformLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  names: readonly string[],
): Record<string, WebGLUniformLocation>
```

Resolves one program's uniform locations with throw-on-missing semantics
so a shader/name drift fails loudly at pass creation.

Parameters:
- `gl` - WebGL2 context.
- `program` - Linked program handle.
- `names` - Uniform names to resolve, in feed order.

Returns: Location table keyed by uniform name.

### toBloomWarmupPrograms

```ts
toBloomWarmupPrograms(
  context: WebGL2RenderingContext,
): readonly [threshold: ShaderWarmupProgram, blur: ShaderWarmupProgram, additive: ShaderWarmupProgram]
```

Builds the three bloom warmup program descriptors for the shared warmup
registry (keys 'bloom-threshold', 'bloom-blur', 'bloom-additive' —
NEATRACING_WARMED_PROGRAM_KEYS[6..8]). Compilation is lazy: no GL program
exists until compileAndLink runs, and each drawOffscreen pushes the
enabled zero-pose frame through the shared four-draw pipeline so all
stages warm in the sweep.

Parameters:
- `context` - WebGL2 context the warmup shaders will run against.

Returns: The three warmup program descriptors in registry-key order.

Example:

for (const program of toBloomWarmupPrograms(gl)) {
  programRegistry.register(program);
}

### uploadBloomFullscreenQuad

```ts
uploadBloomFullscreenQuad(
  gl: WebGL2RenderingContext,
): WebGLBuffer
```

Uploads the shared fullscreen NDC TRIANGLE_STRIP quad corner data into a
STATIC_DRAW buffer used by all three stages.

Parameters:
- `gl` - WebGL2 context.

Returns: The uploaded buffer handle.

## engine/passes/glass.pass.ts

NeatRacing glass render pass (slice P1-S4-E; spec section 6.5 G4, AC-013
part 3, and the section 7.3.2 winding/glass exemption note).

G4 concretizes the LD-23 glass voxels: a dedicated glass pass runs after
ALL opaque geometry with **depth-test ON, depth-write OFF, back-face
culling ON**, and every glass quad is tinted with **premultiplied-alpha
color** plus an **additive glow contribution** sampled from the palette's
glow byte. Cars draw far -> near as cheap insurance; stacked glass inside
one car double-tints, which the spec explicitly locks as accepted.

Structural sibling of `car.pass.ts` (numeric GL enums, exported GLSL ES
3.00 sources, compile helpers that throw with the pass label,
throw-on-missing uniform resolution, a lazy warmup seam) with ONE program:

1. The glass tint program draws bins-INDEPENDENT geometry. Glass cells are
   alpha-0 air for the mesher (section 7.3.2), so the greedy mesher emits
   no glass quads and this pass derives its quads straight from
   `DecodedVoxelModel` slices: six full-faces per glass cell, emitted in a
   deterministic z-asc -> y-asc -> x-asc lattice scan with a fixed face
   order (-x, +x, -y, +y, -z, +z) and TRUE winding (front faces point
   outward; verified by cross-product tests below).

Fragment math is premultiplied alpha: the shader emits
`fragColor = vec4(tint.rgb * (kTintAlpha + vEmissive), kTintAlpha)` and the
pass pins `gl.blendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`, so a glass layer
contributes its tint plus glow additively while softly attenuating only
what is behind it. Black palette entries classify as `black` BEFORE glass
in the shared material table (LD-23), so black can never fabricate glass
geometry: black never shines in the glass pass either (AC-013).

No per-frame allocation: every matrix, sort scratch, and mesh is
preallocated at module or creation scope (P9).

### compileAndLinkGlassProgram

```ts
compileAndLinkGlassProgram(
  gl: WebGL2RenderingContext,
  vertexShaderSource: string,
  fragmentShaderSource: string,
  programLabel: string,
): CompiledGlassProgram
```

Compiles and links the glass pass program from its two stages.

Parameters:
- `gl` - WebGL2 context.
- `vertexShaderSource` - Source for the vertex stage.
- `fragmentShaderSource` - Source for the fragment stage.
- `programLabel` - Label used in every rejection message.

Returns: The linked program plus its two shader handles.

### CompiledGlassProgram

Compiled program bundle: the linked program plus its two live shaders.

### compileGlassShader

```ts
compileGlassShader(
  gl: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
  shaderLabel: string,
): WebGLShader
```

Compiles one glass pass shader stage, rejecting failures with the pass
label (grid.pass.ts parity).

Parameters:
- `gl` - WebGL2 context.
- `shaderType` - gl.VERTEX_SHADER (0x8B31) or gl.FRAGMENT_SHADER (0x8B30).
- `shaderSource` - GLSL ES 3.00 source.
- `shaderLabel` - Stage label used in the rejection message.

Returns: The compiled shader handle.

### composeSortedGlassOrder

```ts
composeSortedGlassOrder(
  state: GlassPassDrawState,
): void
```

Fills the sort scratch with the far -> near glass draw order, anchored at
the camera's XZ position. Stable insertion sort over squared camera
distances: equal distances keep array order, so the ordering is
deterministic frame to frame (spec section 6.5 "cheap insurance").

Parameters:
- `state` - Frame draw state (cars + camera world position).

### createGlassPass

```ts
createGlassPass(
  gl: WebGL2RenderingContext,
  models: GlassPassModels,
): GlassPass
```

Creates the glass render pass: compiles the tint program eagerly, resolves
uniforms and attributes throw-on-missing, expands both decoded models into
glass meshes (AC-013 material-table filter; black contributes nothing),
and resolves the wheel mount transforms once at creation (they allocate
per call; keeping them pre-resolved keeps the draw path allocation-free,
P9).

Parameters:
- `gl` - WebGL2 context.
- `models` - Body and wheel decoded models to render
 *   ( {@link GlassPassModels} ).

Returns: The glass pass handle ( {@link GlassPass} ).

Example:

const glassPass = createGlassPass(gl, { body: decodedBody, wheel: decodedWheel });
glassPass.draw({ viewProjection, cameraWorld, cars });
glassPass.dispose();

### ensureGlassSortScratchCapacity

```ts
ensureGlassSortScratchCapacity(
  carCount: number,
): void
```

Grows the sort scratches to the fleet size once; steady-state frames
never reallocate because scratch capacity only ever grows (P9).

Parameters:
- `carCount` - Cars in the current frame.

### ensureWarmupGlassPass

```ts
ensureWarmupGlassPass(
  context: WebGL2RenderingContext,
): GlassPass
```

Hydrates the singleton glass pass for the warmup seam, rebuilding only
when the context changes. Compilation stays lazy until compileAndLink
runs (car.pass.ts parity).

Parameters:
- `context` - WebGL2 context the warmup shader will run against.

Returns: The hydrated singleton pass.

### expandGlassModelToGlassMesh

```ts
expandGlassModelToGlassMesh(
  model: DecodedVoxelModel,
  yAnchor: CarMeshYAnchor,
): GlassVoxelMesh
```

Expands a decoded voxel model into the glass pass vertex format (pure,
GL-free).

Every palette entry classified `glass` by the shared material table
(fully transparent, colored — spec section 7.2) contributes six full
faces with TRUE winding; every other kind — including `black`, which the
shared table resolves BEFORE glass with zero emissive — contributes
nothing, so black can never shine in the glass pass (AC-013).

Grid positions map to model-space meters exactly like the car pass
(`expandVoxelBinModelToCarMesh`): `modelX = (gx - dimX * 0.5) *
voxelSize` and `modelZ = (gz - dimZ * 0.5) * voxelSize` are
lattice-centered, while `modelY` follows the `yAnchor` argument —
`'ground'` (`modelY = gridY * voxelSize`, bodies authored with row 0 on
the floor) or `'center'` (`modelY = (gridY - dimY * 0.5) * voxelSize`,
lattice-centered wheel discs whose tire bottom must land at the negative
hub height) — so glass aligns one-to-one with the bins-derived body
instead of floating a Y-flipped copy above the car. Colors resolve from
`buildVoxelMaterialTable` (bytes / 255) and emissive is glow / 255. The
lattice scan runs z-asc -> y-asc -> x-asc with the fixed face order
(-x, +x, -y, +y, -z, +z), so the same model always expands to the same
byte-stable vertex data.

Parameters:
- `model` - Decoded voxel model to filter.
- `yAnchor` - Vertical anchoring contract: `'ground'` for bodies,
`'center'` for wheel discs. Defaults to `'ground'`.

Returns: The expanded glass mesh plus its glass cell count.

Example:

const mesh = expandGlassModelToGlassMesh(decodedBody);
// mesh.vertexCount === mesh.glassCellCount * 36; 7 floats per vertex

### GLASS_MESH_FLOATS_PER_VERTEX

Floats per expanded glass vertex: position, color, emissive.

### GLASS_MESH_VERTEX_STRIDE_BYTES

Bytes per expanded glass vertex (7 floats x 4 bytes).

### GlassPass

Glass pass handle returned by createGlassPass.

### GlassPassCarState

Glass pose inside one draw frame.

### GlassPassDrawState

Per-frame draw state accepted by GlassPass.draw.

### GlassPassModels

Models the pass renders: the car body and wheel DECODED voxel models.

### GlassVoxelMesh

Mesh produced by {@link expandGlassModelToGlassMesh}: interleaved
non-indexed triangles ready for one FLOAT-attributed VBO per model.

### identityMat4

```ts
identityMat4(
  out: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBufferLike>
```

Writes the identity matrix into a scratch buffer.

Parameters:
- `out` - Column-major 16-element destination.

Returns: The same buffer for call-site chaining.

### multiplyMat4

```ts
multiplyMat4(
  out: Float32Array<ArrayBufferLike>,
  a: Float32Array<ArrayBufferLike>,
  b: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBufferLike>
```

Multiplies two column-major matrices into an output scratch buffer:
`out = a * b` (a point is transformed by b first, then a). Call sites
must pass a dedicated scratch that aliases neither input; every scratch
here is preallocated at module scope to keep the draw path
allocation-free (P9).

Parameters:
- `out` - 16-element output scratch (must not alias a or b).
- `a` - Left operand, applied second in the transform chain.
- `b` - Right operand, applied first in the transform chain.

Returns: The same output buffer for call-site chaining.

### NEATRACING_GLASS_FACES_PER_CELL

Glass faces emitted per glass cell, in fixed face order
(-x, +x, -y, +y, -z, +z); derived from the face table length so the
count and the winding table can never drift.

### NEATRACING_GLASS_FRAGMENT_SHADER_SOURCE

Glass fragment shader (G4): premultiplied-alpha tint plus additive glow.
`fragColor.rgb = tint.rgb * (kTintAlpha + vEmissive)` with
`fragColor.a = kTintAlpha` and the pass blending
`blendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)`, so the tint softens what is
behind the pane while the palette glow contribution adds on top.

### NEATRACING_GLASS_PROGRAM_KEY

Warmup program key: registry index 5 of NEATRACING_WARMED_PROGRAM_KEYS
('glass') is the contract pinned by the warmup seam test (AC-040).

### NEATRACING_GLASS_TINT_ALPHA

Mild premultiplied glass tint alpha (G4: "arcade glass is thin and tint
is mild"; double-tinting stacked panes is explicitly accepted, so the
per-layer alpha stays low). Pinned so GLSL interpolation cannot drift.

### NEATRACING_GLASS_VERTEX_SHADER_SOURCE

Glass vertex shader: transforms one glass face corner into clip space and
passes the palette tint plus glow contribution through.

### NEATRACING_GLASS_VERTICES_PER_FACE

Expanded vertices per glass face: two non-indexed triangles.

### resolveGlassAttributeLocations

```ts
resolveGlassAttributeLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
): Record<"aPosition" | "aColor" | "aEmissive", number>
```

Resolves the glass program's attribute locations in expansion float order
(position, color, emissive).

Parameters:
- `gl` - WebGL2 context.
- `program` - Linked glass program handle.

Returns: Attribute location table used by vertexAttribPointer feeds.

### resolveGlassUniformLocations

```ts
resolveGlassUniformLocations(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
  names: readonly string[],
): Record<string, WebGLUniformLocation>
```

Resolves the glass program's uniform locations with throw-on-missing
semantics so a shader/name drift fails loudly at pass creation (grid and
car pass parity).

Parameters:
- `gl` - WebGL2 context.
- `program` - Linked program handle.
- `names` - Uniform names to resolve, in feed order.

Returns: Location table keyed by uniform name.

### rotationYMat4

```ts
rotationYMat4(
  out: Float32Array<ArrayBufferLike>,
  angleRad: number,
): Float32Array<ArrayBufferLike>
```

Writes a rotation-around-world-up matrix into a scratch buffer
(column-major: col0 = (cos, 0, -sin), col2 = (sin, 0, cos)).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `angleRad` - Yaw angle in radians.

Returns: The same buffer for call-site chaining.

### rotationZMat4

```ts
rotationZMat4(
  out: Float32Array<ArrayBufferLike>,
  angleRad: number,
): Float32Array<ArrayBufferLike>
```

Writes a rotation-around-the-model-depth-axis matrix into a scratch
buffer (the pre-mount wheel model's spin axis; column-major).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `angleRad` - Spin angle in radians.

Returns: The same buffer for call-site chaining.

### toGlassWarmupProgram

```ts
toGlassWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Builds the glass warmup program descriptor for the shared warmup registry
(key 'glass', NEATRACING_WARMED_PROGRAM_KEYS[5]). Compilation is lazy: no
GL program exists until compileAndLink runs, and drawOffscreen pushes the
zero-pose frame through the same draw path production uses (G4 state
machine included).

Parameters:
- `context` - WebGL2 context the warmup shader will run against.

Returns: The warmup program descriptor keyed 'glass'.

Example:

programRegistry.register(toGlassWarmupProgram(gl));

### translationMat4

```ts
translationMat4(
  out: Float32Array<ArrayBufferLike>,
  x: number,
  y: number,
  z: number,
): Float32Array<ArrayBufferLike>
```

Writes a translation matrix into a scratch buffer (column-major).

Parameters:
- `out` - Column-major 16-element scratch destination.
- `x` - Translation x in meters.
- `y` - Translation y in meters.
- `z` - Translation z in meters.

Returns: The same buffer for call-site chaining.

### uploadGlassFloat32Buffer

```ts
uploadGlassFloat32Buffer(
  gl: WebGL2RenderingContext,
  data: Float32Array<ArrayBufferLike>,
): WebGLBuffer
```

Uploads one Float32 array into a STATIC_DRAW buffer.

Parameters:
- `gl` - WebGL2 context.
- `data` - Vertex data to upload.

Returns: The uploaded buffer handle.

## engine/passes/pulse.pass.ts

NeatRacing ambient pulse pass (AC-006 / spec §6.4.3, slice P1-S2-C).

Renders non-interactive, world-space pulse rings on the track ground
plane: each pulse is a `#B7FF00` (185, 255, 0) glow ring at alpha 0.42
(mirroring the neatenstein `PULSE_GLOW_ALPHA` constant) that expands from
the ring start radius to the ring end radius over a fixed lifetime while
its alpha follows a deterministic attack-then-decay envelope. Placement
is derived deterministically from the TP-20 visual seed, emission follows
a fixed cadence capped at the concurrency limit, and every pulse is
depth-tested (GL_LESS semantics) so world geometry occludes it.

Structure follows the orchestration-first layout: the public pure-math
contract (constants, types, envelope, radius, placement, cadence,
visuals, depth predicate) is testable in a plain Node Jest environment,
while the WebGL2 orchestration (`createPulsePass`, `toPulseWarmupProgram`,
draw loop, GLSL sources, GL helpers) sits below the fold and is exercised
through the recording mock-context harness in `pulse.pass.test.ts`.

The pass is deliberately stateless between frames: active pulses are
re-collected from the simulation timestamp every draw (pure time-window
model), which keeps the visual output deterministic and replayable and
avoids incremental spawn state on the render side.

### AmbientPulse

A single active ambient pulse collected for a frame.

### clamp01

```ts
clamp01(
  value: number,
): number
```

Clamps a scalar into the unit interval.

Parameters:
- `value` - Value to clamp.

Returns: Value constrained to `[0, 1]`.

### collectActivePulses

```ts
collectActivePulses(
  simTimeMs: number,
  seed: number,
): AmbientPulse[]
```

Collects the ambient pulses alive at a simulation timestamp.

The model is a pure time window rather than incremental state: spawn `k`
exists for `[k * interval, k * interval + lifetime)`. Iterating newest
first and capping at the concurrency limit yields the deterministic,
replayable alive-set the plan requires. Pulses whose age reached the
lifetime expire, and negative timestamps yield an empty set. Each call
returns a fresh array; inputs are never mutated.

Parameters:
- `simTimeMs` - Simulation time in milliseconds.
- `seed` - Visual seed driving deterministic placement.

Returns: Fresh array of active pulses ordered newest first.

Example:

collectActivePulses(0, 1337); // exactly the first pulse, age 0
collectActivePulses(1_999, 1337); // 4 pulses at the concurrency cap

### compileShaderOrThrow

```ts
compileShaderOrThrow(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
): WebGLShader
```

Compiles one shader stage, throwing with pass context on failure.

Parameters:
- `gl` - WebGL2 context to compile against.
- `type` - GL shader stage constant.
- `source` - GLSL ES 3.00 source for the stage.

Returns: Compiled shader handle.

### computePulseEnvelope

```ts
computePulseEnvelope(
  progress: number,
): number
```

Computes the lifetime alpha envelope for a pulse.

The envelope is zero at progress `0`, ramps linearly from `0` to the glow
alpha across the attack window, then decays linearly back to zero across
the remainder of the lifetime. It is continuous, peaks exactly at the
glow alpha at the attack boundary, and is fully deterministic, which lets
the plan's red test assert the expansion/decay curve.

Parameters:
- `progress` - Pulse lifetime progress; clamped into `[0, 1]`.

Returns: Alpha in `[0, NEATRACING_PULSE_GLOW_ALPHA]`.

Example:

computePulseEnvelope(0); // 0
computePulseEnvelope(NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION); // 0.42
computePulseEnvelope(1); // 0

### computePulseRingRadius

```ts
computePulseRingRadius(
  progress: number,
): number
```

Computes the expanding ring radius for a pulse.

The radius grows linearly from the start radius at birth to the end
radius at end of lifetime, giving the deterministic expansion curve the
plan binds to AC-006. Progress outside `[0, 1]` clamps onto the
endpoints so out-of-window inputs cannot produce out-of-range geometry.

Parameters:
- `progress` - Pulse lifetime progress; clamped into `[0, 1]`.

Returns: Ring radius in meters between start and end radius.

Example:

computePulseRingRadius(0); // 0.5
computePulseRingRadius(1); // 6

### createPulsePass

```ts
createPulsePass(
  gl: WebGL2RenderingContext,
): NeatRacingPulsePass
```

Creates the WebGL2 pulse pass handle for AC-006 rendering.

Compiles and links the ground-plane ring program once, uploads the unit
quad, and returns a stateless draw handle. The draw call owns the glow
state machine per frame: depth test on (world occlusion), depth writes
off, face culling off (ground-plane winding is view-dependent), and
premultiplied additive blending `blendFunc(ONE, ONE)` so overlapping
rings accumulate light without sorting. One `drawArrays` call is issued
per pulse in the frame; zero-alpha pulses are issued anyway and the
fragment shader discards them, keeping the call count deterministic.

Parameters:
- `gl` - WebGL2 context owned by the rendering context service.

Returns: Pulse pass handle keyed to the `'pulse'` registry entry.

Example:

const pass = createPulsePass(gl);
pass.draw({ pulses: collectActivePulses(simTimeMs, seed), viewProjection });
// later: pass.dispose();

### derivePulseOrigin

```ts
derivePulseOrigin(
  seed: number,
  spawnIndex: number,
): PulseOrigin
```

Derives the deterministic world-space origin for a pulse emission.

Two independent hash streams (X and Z) are seeded by the visual seed and
the spawn index, so the same seed always reproduces the same placement
while different seeds produce different layouts. Values map the `[0, 1)`
hash range onto the square spawn bounds centered on the world origin.

Parameters:
- `seed` - Visual seed (TP-20 base 1337 by default).
- `spawnIndex` - Monotonic emission index to derive placement for.

Returns: Origin in meters, bounded by the spawn half extent.

Example:

derivePulseOrigin(1337, 0); // e.g. { x: -23.41, z: 12.07 }

### hashPulseSeed

```ts
hashPulseSeed(
  seed: number,
  stream: number,
): number
```

Hashes a seed and stream index into a deterministic `[0, 1)` fraction.

Uses integer-only mixing (`Math.imul` with xor-shift finalization) so the
result is stable across platforms and Jest runs, giving placement its
replayability guarantee. Streams are independent integers derived from
the spawn index, so X and Z never collapse onto one stream.

Parameters:
- `seed` - Visual seed to salt the hash with.
- `stream` - Independent stream index (X uses even, Z uses odd).

Returns: Deterministic fraction in `[0, 1)`.

### isPulseDepthVisible

```ts
isPulseDepthVisible(
  pulseDepth: number,
  bufferDepth: number,
): boolean
```

Depth-visibility predicate for a pulse against the depth buffer.

Uses strict GL_LESS semantics matching the default WebGL2 depth test: a
pulse is visible only when its depth value is strictly nearer (smaller)
than the buffer value, so the sky clear depth of 1.0 never occludes a
pulse while equal or nearer geometry always does.

Parameters:
- `pulseDepth` - Pulse depth value in `[0, 1]` window space.
- `bufferDepth` - Depth buffer value in `[0, 1]` window space.

Returns: `true` when the pulse passes the depth test.

Example:

isPulseDepthVisible(0.5, 1); // true (sky never occludes)
isPulseDepthVisible(0.5, 0.5); // false (equal depth occludes)

### NEATRACING_PULSE_AMBIENT_INTERVAL_MS

Fixed cadence between pulse emissions in milliseconds.

### NEATRACING_PULSE_COLOR_HEX

Hex token of the pulse glow color, as written in spec §6.4.3.

### NEATRACING_PULSE_COLOR_RGB

RGB decomposition of {@link NEATRACING_PULSE_COLOR_HEX} (185, 255, 0),
normalized to the `[0, 1]` range the fragment shader consumes: the shader
multiplies the raw channels, so 0-255 values would clamp the glow to
yellow-white (the defect the starfield/grid passes avoid by feeding
normalized channels).

Spec defect (flagged to the spec owner, intentionally not "fixed" here):
the spec pins the hex token `#B7FF00` (hex 0xB7 decodes to 183) and the
decimal triplet (185, 255, 0) in the same section. The decimal pin stays
authoritative for this pass, so 185 is kept.

### NEATRACING_PULSE_ENVELOPE_ATTACK_FRACTION

Fraction of the lifetime spent ramping up to the glow alpha peak.

### NEATRACING_PULSE_GLOW_ALPHA

Glow alpha at the envelope peak; mirrors the neatenstein 0.42 constant.

### NEATRACING_PULSE_GROUND_LIFT_M

Ground lift in meters keeping the ring coplanar-ish above the ground.

### NEATRACING_PULSE_LIFETIME_MS

Pulse lifetime in milliseconds; drives both envelope and cadence windows.

### NEATRACING_PULSE_MAX_CONCURRENT

Maximum number of pulses kept alive at once; defensive cap.

### NEATRACING_PULSE_PROGRAM_KEY

Warmup/draw registry key; must stay a member of the program registry.

### NEATRACING_PULSE_RING_END_RADIUS_M

Ring radius in meters at pulse end of lifetime.

### NEATRACING_PULSE_RING_START_RADIUS_M

Ring radius in meters at pulse birth.

### NEATRACING_PULSE_SPAWN_HALF_EXTENT_M

Half extent in meters of the square spawn bounds around the origin.

### NEATRACING_PULSE_VISUAL_SEED_DEFAULT

Default visual seed (TP-20) used to derive deterministic placement.

### NeatRacingPulsePass

Draw-pass handle returned by {@link createPulsePass}.

### PulseOrigin

World-space origin of a pulse, derived from the visual seed.

### PulsePassDrawFrame

Frame payload handed to {@link NeatRacingPulsePass.draw}.

### PulseVisuals

Renderable visuals resolved for one pulse.

### resolvePulseVisuals

```ts
resolvePulseVisuals(
  pulse: AmbientPulse,
): PulseVisuals
```

Resolves the renderable visuals for one pulse from its age.

Composition point binding the envelope and radius curves to a concrete
pulse: alpha comes from the lifetime envelope, radius from the expansion
curve, both driven by the same normalized progress.

Parameters:
- `pulse` - Pulse to resolve visuals for.

Returns: Alpha and radius in meters for the pulse's current age.

Example:

const pulse = collectActivePulses(1_000, 1337)[0];
resolvePulseVisuals(pulse); // { alpha: <decay tail>, radiusM: 3.25 }

### toPulseWarmupProgram

```ts
toPulseWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Exposes the pulse pass as a shader warmup program (AC-040 seam).

The returned entry keeps compilation lazy: `compileAndLink` builds the
pass (compiling and linking the program), and `drawOffscreen` issues the
zero-pose frame draw through the shared pass path so the driver warms the
exact render pipeline used on screen. The context passed to
`drawOffscreen` is accepted for interface parity; the pass is bound to
the context supplied at construction.

Parameters:
- `context` - WebGL2 context the pass resources will be built against.

Returns: Warmup entry keyed to the `'pulse'` registry entry.

Example:

const warmup = toPulseWarmupProgram(gl);
warmup.compileAndLink();
warmup.drawOffscreen(gl);

## engine/passes/track.pass.ts

NeatRacing track slab draw pass (spec section 6.7, slice P1-ground-composition,
AC-017). Uploads the deterministic {@link buildTrackSlabs} ribbon as an
interleaved position+color VBO and draws it with depth testing enabled so
the track sits on the ground behind/on top of the grid plane.

### buildRoadEdgeVboData

```ts
buildRoadEdgeVboData(
  mesh: RoadEdgeHexMesh,
): Float32Array<ArrayBufferLike>
```

Builds an interleaved lit Float32Array of position+color+normalAxis
vertices from the deterministic road-edge hex mesh. Each triangle is already
triangulated, so we emit three vertices per triangle with consistent winding.
The road-edge paint is a flat horizontal plane, so every vertex uses the
world-up normal axis (1).

### buildTrackLitVboData

```ts
buildTrackLitVboData(
  mesh: TrackSlabMesh,
): Float32Array<ArrayBufferLike>
```

Builds an interleaved lit Float32Array of position+color+normalAxis
vertices from the deterministic track slab mesh. Same layout as
{@link buildTrackVboData} plus an encoded normal axis per vertex so the
track shares the same sun/hemisphere/fog lighting as the car.

### buildTrackVboData

```ts
buildTrackVboData(
  mesh: TrackSlabMesh,
): Float32Array<ArrayBufferLike>
```

Builds an interleaved Float32Array of position+color vertices from the
deterministic track slab mesh. Each greedy-meshed quad is triangulated as
two triangles (six vertices) with a consistent counter-clockwise winding.

Top faces (normalAxis === 1, world y = 0) are painted with a flat
near-black matte so the old road/curb/grass slab colors do not show through
the hex floor paint (OBS-9). Side and bottom faces keep their palette
colors for readable ribbon edges.

Exported as a test seam so the VBO color contract can be asserted
directly without a full WebGL mock.

### compileTrackShader

```ts
compileTrackShader(
  gl: WebGL2RenderingContext,
  type: number,
  source: string,
  label: string,
): WebGLShader
```

Compiles one shader stage or throws a descriptive error carrying the stage
label and the driver info log.

### createTrackPass

```ts
createTrackPass(
  gl: WebGL2RenderingContext,
): TrackPass
```

Creates the track slab draw pass: builds the default deterministic ribbon,
compiles and links the position+color program, uploads the interleaved VBO,
and returns a handle whose `draw(state)` sets the view-projection uniform,
enables depth testing, and issues one `gl.drawArrays(GL_TRIANGLES, 0, count)`.

Parameters:
- `gl` - WebGL2 rendering context to bind the pass to.

Returns: The track pass handle.

### expandSlabQuadCorners

```ts
expandSlabQuadCorners(
  quad: VoxelQuad,
): readonly [readonly [number, number, number], readonly [number, number, number], readonly [number, number, number], readonly [number, number, number]]
```

Expands a slab quad into its four corner positions in consistent winding
order. The quad lies on the plane perpendicular to its normal axis, with
width along the first tangent axis and height along the second.

### NEATRACING_TRACK_PROGRAM_KEY

Warmup/draw registry key for the track pass (AC-040).

### normalizeColorBytes

```ts
normalizeColorBytes(
  color: readonly [number, number, number],
): readonly [number, number, number]
```

Converts 0-255 RGB bytes to 0..1 normalized floats for the vertex color
attribute.

### resolveTangentAxes

```ts
resolveTangentAxes(
  normalAxis: number,
): readonly [number, number]
```

Resolves the width and height tangent axes for a given face normal axis,
mirroring the {@link FACE_TANGENT_AXES} mapping in the shared greedy mesher.

### toTrackWarmupProgram

```ts
toTrackWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Exposes the track pass as a shader warmup program (AC-040 seam).

The returned entry keeps compilation lazy: `compileAndLink` builds the
pass (compiling and linking the program), and `drawOffscreen` issues the
zero-pose track draw through the shared pass path so the driver warms the
exact render pipeline used on screen. The context argument is accepted for
interface parity; the pass is bound to the context supplied at construction.

Parameters:
- `context` - WebGL2 context the pass resources will be built against.

Returns: Warmup entry keyed to the track registry slot (fourth warmed key).

### TrackPass

The track pass handle: create once, draw every frame.

### TrackPassDrawState

Per-draw inputs for the track pass.

## engine/passes/starfield.pass.ts

### buildStarLayerLayout

```ts
buildStarLayerLayout(
  spec: StarfieldLayerSpec,
  tileWidthPx: number,
  tileHeightPx: number,
  layerIndex: number,
  maxYPx: number,
): StarfieldLayerLayout
```

Builds the deterministic point layout for one starfield layer.

Called once per layer at pass construction; repeated calls with the same
arguments return deep-equal layouts because the xorshift32 stream is fully
seeded by the layer spec.

Parameters:
- `spec` - Layer recipe (seed, counts, size/alpha ranges).
- `tileWidthPx` - Tile width the x positions are generated against.
- `tileHeightPx` - Tile height the y positions are generated against.

Returns: Immutable layout holding exactly `spec.starCount` placements.

Example:

const layout = buildStarLayerLayout(
  NEATRACING_STARFIELD_LAYER_SPECS[0],
  NEATRACING_STARFIELD_TILE_WIDTH_PX,
  NEATRACING_STARFIELD_TILE_HEIGHT_PX,
  0,
);
// → { spec, placements: StarPlacement[spec.starCount] }

### compileStarfieldProgram

```ts
compileStarfieldProgram(
  context: WebGL2RenderingContext,
): WebGLProgram
```

Compiles and links the starfield shader program.

Parameters:
- `context` - WebGL2 context owning the program.

Returns: The linked program handle.

### compileStarfieldShader

```ts
compileStarfieldShader(
  context: WebGL2RenderingContext,
  shaderType: number,
  shaderSource: string,
): WebGLShader
```

Compiles one shader stage and fails loudly with the driver info log.

Parameters:
- `context` - WebGL2 context owning the shader.
- `shaderType` - `VERTEX_SHADER` or `FRAGMENT_SHADER`.
- `shaderSource` - GLSL ES 3.00 source text.

Returns: The compiled shader handle.

### computeStarfieldBlink

```ts
computeStarfieldBlink(
  timeSeconds: number,
  layerIndex: number,
  starIndex: number,
): number
```

Compute the raw normalized blink value for a given star at a point in time.

The blink value is in `[-1, 1]` and uses a layer-specific beat so consecutive
layers do not pulse in lockstep. It is deterministic from `(timeSeconds,
layerIndex, starIndex)`.

### computeStarfieldBlinkPhaseOffset

```ts
computeStarfieldBlinkPhaseOffset(
  layerIndex: number,
  starIndex: number,
): number
```

Deterministic phase offset (radians) for a single star's blink oscillator.

Depends on the layer id and star index only, so the same star always
twinkles at the same phase regardless of current time. The phase is
distributed across `[0, 2π)` using a seeded Xorshift step.

### computeStarfieldGloomAlphaScale

```ts
computeStarfieldGloomAlphaScale(
  timeSeconds: number,
): number
```

Compute the short gloom alpha scale for a star layer at a point in time.

The gloom cycle dims the whole layer slightly over time without ever dropping
to zero. It is layer-agnostic but deterministic for a given time.

### computeStarfieldNdcFromWrappedTilePosition

```ts
computeStarfieldNdcFromWrappedTilePosition(
  wrappedXPx: number,
  wrappedYPx: number,
  tileWidthPx: number,
  tileHeightPx: number,
  horizonNdcY: number,
): { ndcX: number; ndcY: number; }
```

Maps a wrapped tile-space pixel position to NDC using the generation tile
size, mirroring the vertex shader formula (OBS-G coverage seam).

Parameters:
- `wrappedXPx` - Wrapped x position in `[0, tileWidthPx)`.
- `wrappedYPx` - Wrapped y position in `[0, tileHeightPx)`.
- `tileWidthPx` - Generation tile width in pixels.
- `tileHeightPx` - Generation tile height in pixels.
- `horizonNdcY` - NDC y coordinate of the visible horizon.

Returns: The NDC `(x, y)` coordinate produced by the shader.

### computeStarLayerOffsets

```ts
computeStarLayerOffsets(
  cameraPose: StarfieldCameraPose,
  layerIndex: number,
): { offsetX: number; offsetY: number; }
```

Resolves both scroll offsets for one layer from the camera pose.

Pure convenience wrapper over {@link computeStarLayerOffsetX} and
{@link computeStarLayerOffsetY}; the draw path calls the two scalar
functions directly so the per-frame path never allocates.

Parameters:
- `cameraPose` - Current camera pose in radians.
- `layerIndex` - Layer index into the pinned spec order.

Returns: Both pixel scroll offsets for the layer.

Example:

computeStarLayerOffsets({ yawRadians: 0, pitchRadians: 0 }, 0);
// → { offsetX: -0, offsetY: -0 }

### computeStarLayerOffsetX

```ts
computeStarLayerOffsetX(
  yawRadians: number,
  layerIndex: number,
): number
```

Resolves the horizontal scroll offset for one layer from the camera yaw.

G6 formula: `offsetX = −(yaw / 2π) × tileWidthPx × scrollRatio`. Positive
yaw (camera rotating right) shifts stars left, and each layer scrolls at
its pinned ratio (far 3%, mid 4%, near 5%) for the parallax shear.

Parameters:
- `yawRadians` - Camera yaw in radians.
- `layerIndex` - Layer index into the pinned spec order.

Returns: Horizontal scroll offset in pixels.

Example:

computeStarLayerOffsetX(Math.PI / 2, 0); // → −3.84 (quarter turn, far layer)

### computeStarLayerOffsetY

```ts
computeStarLayerOffsetY(
  pitchRadians: number,
  layerIndex: number,
): number
```

Resolves the vertical scroll offset for one layer from the camera pitch.

G6 formula: `offsetY = −(pitch / 2π) × tileHeightPx × scrollRatio`. A
camera pitching down (positive pitch) scrolls stars up the full backdrop
so the parallax shear reaches the horizon.

Parameters:
- `pitchRadians` - Camera pitch in radians.
- `layerIndex` - Layer index into the pinned spec order.

Returns: Vertical scroll offset in pixels.

Example:

computeStarLayerOffsetY(Math.PI / 2, 0); // → −5.4 (quarter pitch, far layer)

### createLayerVertexArray

```ts
createLayerVertexArray(
  context: WebGL2RenderingContext,
  layout: StarfieldLayerLayout,
): { vertexArrayObject: WebGLVertexArrayObject; buffer: WebGLBuffer; vertices: Float32Array<ArrayBufferLike>; }
```

Uploads one layer's stars into a vertex array object and reusable buffer.

Runs once per layer at construction; the returned VAO owns the interleaved
buffer and all attribute pointers so the per-frame draw path only needs
`bindVertexArray` + `drawArrays` (P9 zero-allocation contract). The buffer
and vertex array are returned separately so `setBackdropSize` can rewrite
the same buffer on resize without allocating a new GPU object.

Parameters:
- `context` - WebGL2 context owning the GPU resources.
- `layout` - Precomputed layer layout to upload.

Returns: VAO, GPU buffer, and the reusable CPU-side vertex array.

### createStarfieldPass

```ts
createStarfieldPass(
  context: WebGL2RenderingContext,
  options: CreateStarfieldPassOptions,
): StarfieldPass
```

Creates the starfield background pass (AC-004).

Compiles the point-sprite program once, precomputes every layer layout and
vertex buffer once (P9), configures the static backdrop state (cyan tint,
alpha blending, no depth writes), and returns the draw seam. Every later
`draw` call only rebinds precomputed state and feeds the pure G6 scroll
offsets, so the per-frame path allocates nothing.

Parameters:
- `context` - WebGL2 context acquired by the rendering context service.
- `options` - Optional backing-store width override for NDC mapping.

Returns: The starfield pass seam.

Example:

const starfieldPass = createStarfieldPass(gl);
starfieldPass.draw({ yawRadians: camera.yaw, pitchRadians: camera.pitch });

### CreateStarfieldPassOptions

Optional construction knobs for the starfield pass.

### createStarfieldRng

```ts
createStarfieldRng(
  seed: number,
): () => number
```

Creates a deterministic pseudo-random generator for starfield layouts.

Direct port of the flappy `createSeededRandom` xorshift32 generator so the
voxel racer reproduces the exact same value stream per seed. The seed is
coerced to unsigned 32-bit state; every call advances the state through
the pinned 13/17/5 shift triple and normalizes into `[0, 1)`.

Parameters:
- `seed` - Unsigned integer seed (layer spec seed or visual seed base).

Returns: Function yielding deterministic values in `[0, 1)`.

Example:

const random = createStarfieldRng(NEATRACING_VISUAL_SEED);
random(); // → deterministic value in [0, 1)

### packLayerVertices

```ts
packLayerVertices(
  layout: StarfieldLayerLayout,
  out: Float32Array<ArrayBufferLike> | undefined,
): Float32Array<ArrayBufferLike>
```

Packs one layer's placements into the interleaved star vertex buffer.

When an existing `out` array is supplied it is filled in place so resize
re-uploads avoid allocating a fresh Float32Array (P9 contract).

Parameters:
- `layout` - Precomputed layer layout.
- `out` - Optional existing buffer to overwrite; must be large enough.

Returns: Float buffer laid out as `[xPx, yPx, sizePx, alpha, phaseRadians]` per star.

### requireLayerSpec

```ts
requireLayerSpec(
  layerIndex: number,
): StarfieldLayerSpec
```

Resolves the layer spec for an index, failing loudly on out-of-range use.

Parameters:
- `layerIndex` - Index into the pinned spec order (0 = far, 2 = near).

Returns: The pinned layer spec.

### requireUniformLocation

```ts
requireUniformLocation(
  context: WebGL2RenderingContext,
  program: WebGLProgram,
  name: string,
): WebGLUniformLocation
```

Resolves a uniform location, failing loudly when the driver drops it.

Parameters:
- `context` - WebGL2 context owning the program.
- `program` - Linked starfield program.
- `name` - Uniform name declared in the shader source.

Returns: The live uniform location.

### resolveStarPlacement

```ts
resolveStarPlacement(
  options: { random: () => number; spec: StarfieldLayerSpec; layerIndex: number; starIndex: number; tileWidthPx: number; tileHeightPx: number; maxYPx: number; },
): StarPlacement
```

Resolves one deterministic star placement and appearance from the RNG.

Port of the flappy `resolveStarPlacement`: exactly four draws per star in
the pinned order (x, y, size, alpha) so any change to the sequence would
shift every later star. The inclusive size offset makes the upper size
bound reachable, matching the flappy tile builder.

Parameters:
- `options` - Random source, layer bounds, and tile extent.

Returns: Pixel location, square size, and alpha for one rendered star.

### StarfieldCameraPose

Camera pose subset the starfield reacts to (G6).

### StarfieldLayerLayout

Precomputed point layout for one parallax starfield layer.

### StarfieldLayerRenderTarget

Renderable handle for one starfield layer: layout plus its GPU buffers.

### StarfieldPass

The starfield background pass seam consumed by the render loop.

### StarPlacement

One deterministic star placement inside a starfield tile (flappy parity).

### toStarfieldWarmupProgram

```ts
toStarfieldWarmupProgram(
  context: WebGL2RenderingContext,
): ShaderWarmupProgram
```

Exposes the starfield pass as a shader warmup program (AC-040 seam).

The returned entry keeps compilation lazy: `compileAndLink` builds the
pass (compiling and linking the program), and `drawOffscreen` issues the
three-layer draw through the shared pass path so the driver warms the
exact render pipeline used on screen. The context passed to
`drawOffscreen` is accepted for interface parity; the pass is bound to the
context supplied at construction.

Parameters:
- `context` - WebGL2 context the pass resources will be built against.

Returns: Warmup entry keyed as the first pinned registry key.

Example:

const warmup = toStarfieldWarmupProgram(gl);
warmup.compileAndLink();
warmup.drawOffscreen(gl);
