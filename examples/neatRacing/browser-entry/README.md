# browser-entry

The Neat Racing chase camera (Step 14P, slice
`P1-drift-rewire-travel-camera`; the DRIFT-CAMERA law of Step 14X, slice
`P1-driftcam-law`).

THE DEFECT THIS MODULE CLOSES: the chase camera was a pure function of the
car NOSE heading. The camera eye, the look target and the starfield yaw
were all re-derived from `car.heading` every frame, and because the eye
was re-placed along the nose FROM the car's own position, the frame
cancelled the off-axis travel BY CONSTRUCTION: a 25.8 degree slide
produced EXACTLY ZERO screen-space translation (the frozen-camera trap,
RED-4). The 3D view could not move, so the drift could not be seen.

THE LAW: the camera aims along the TRAVEL direction - the travel heading
the physics publishes on the car state - and the follow that gets it
there is PROPORTIONAL to the speed times the shared drift amount (the
user law S-1/S-4), never a fixed time constant (a fixed tau is the S-1
violation the retired `NEATRACING_CHASE_CAMERA_FOLLOW_TAU_S` and
`NEATRACING_CHASE_CAMERA_TRAVEL_AIM_GAIN` names stood for; those names
are FORBIDDEN and do not exist).

THE DRIFT-CAMERA LAW (Step 14X, AC-097): the yaw aim offset is CAPPED
at the NAMED `NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD` - EXACTLY HALF
of the MEASURED pre-wave peak offset - and the cap is enforced AT THE
SOURCE (inside this law), so EVERY reader can only ever see the CAPPED
offset. The follow is DESENSITISED by the NAMED
`NEATRACING_CHASE_CAMERA_FOLLOW_GAIN` (HALF of the MEASURED pre-wave
gain), so the camera reacts LESS to the same published slide and drifts
into it instead of snapping onto it. The RELEASE is a CONSTANT fraction
of the aim (`NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION`), so the
RELEASE term never closes more than that fraction of the aim's value in
ONE tick: the MULTIPLICATIVE snap-back enabler (the pre-wave release
rate was MULTIPLIED by the camera's OWN offset, so the camera snapped
back harder the further it already sat from the nose) is DELETED at its
source.

THE OWNERSHIP: this module owns the chase GEOMETRY (the eye, the look
target, the forward vector) and the follow LAW. The browser entry owns
the renderer and reads BOTH through the ONE aim law, so the camera
placement, the look target and the starfield yaw can never disagree about
which way the car is travelling.

## browser-entry/constants.ts

### NEATRACING_FALLBACK_CANVAS_WIDTH

Fallback backing-store width in pixels used when no client or viewport
dimensions are measurable (the final step of the resolution ladder).

1280x720 keeps the fallback at a 16:9 aspect ratio matching the fixed
render height, mirroring the Neatenstein fallback ladder shape.

### NEATRACING_FIXED_RENDER_HEIGHT

Fixed backing-store render height in pixels for the neatRacing canvas.

The strategy (spec §6.2) always renders into a 720-pixel-tall backing
store and derives the backing width from the client aspect ratio, so the
render loop and the letterbox layout can be computed against a known
vertical resolution regardless of the window size.

Example:

resolveCanvasBackingDimensions({ clientWidth: 1920, clientHeight: 1080, ... });
// → { width: 1280, height: NEATRACING_FIXED_RENDER_HEIGHT }

### NEATRACING_MAX_BACKING_WIDTH

Maximum backing-store width in pixels for the neatRacing canvas.

Ultrawide displays (for example 32:9) would otherwise double the fill
rate compared with a 16:9 backing store; the backing width is therefore
clamped at this constant and the CSS letterbox absorbs the difference.

## browser-entry/chase-camera.ts

### ChaseCameraBasis

The chase-camera basis in scene space: the camera eye, the look target
and the normalized camera forward.

### ChaseCameraFollowState

The chase-camera follow state: the camera's slide offset from the nose
heading and the deterministic tick the follow was last advanced to.

EXACTLY TWO fields, by contract: the offset is the ONLY carried state of
the follow, and the tick is the frame-rate-independent clock that
advances it.

### ChaseCameraFollowStep

The step descriptor the follow consumes: the simulation ticks that
elapsed, the nose heading, the PUBLISHED travel heading, the car speed and
the tick index the step belongs to.

### clampChaseCameraFollowOffsetRad

```ts
clampChaseCameraFollowOffsetRad(
  offsetRad: number,
): number
```

Clamps a chase-camera aim offset to the NAMED symmetric cap.

WHY: the cap is a PUBLISHED invariant (the cap is enforced AT THE
SOURCE, so EVERY reader can only ever see the CAPPED offset), so every
path that publishes an offset - the supported aim, the next aim and the
HOLD path - clamps through this ONE helper and no reader can ever see an
over-cap offset.

Parameters:
- `offsetRad` - The aim offset to clamp (rad).

Returns: The offset clamped to the NAMED cap (rad).

### createChaseCameraFollowState

```ts
createChaseCameraFollowState(): ChaseCameraFollowState
```

Creates the neutral chase-camera follow state: the camera aims along the
nose heading and the follow has not advanced yet.

Returns: A fresh follow state.

### NEATRACING_CHASE_CAMERA_BACK

Chase-camera distance behind the car (world units).

### NEATRACING_CHASE_CAMERA_FOLLOW_DT_S

The chase-camera follow timestep (s): the NAMED 60 Hz physics tick dt.

The follow is stepped in TICKS, never in frames, so the aim is frame-rate
independent: sixty ticks delivered as one frame or as sixty frames
produce the IDENTICAL aim (RED-6). The dt is IMPORTED from the physics
service instead of being inlined, so the camera can never tick at a
different rate than the simulation it follows.

### NEATRACING_CHASE_CAMERA_FOLLOW_GAIN

The chase-camera follow gain: how fast the camera aim closes on the
PUBLISHED travel direction, in rad/s per (m/s of car speed) per rad of
shared drift amount.

UNIT: rad/s per (m/s * rad). SIDE: SIGNLESS - the gain scales the
magnitude of the published drift amount and never flips the side of the
slide.

MEASURED: the RED slice (P1-driftcam-red) MEASURED the pre-wave gain
as 1 and RECORDED the pre-wave peak aim offset of 0.34897858999999976
rad; this gain is HALF of that MEASURED pre-wave gain, so the camera
responds DESENSITISED (a weaker pull on the published slide). The
follow stays PROPORTIONAL to the speed times the shared drift amount
(the user law S-1/S-4), never a fixed time constant.

### NEATRACING_CHASE_CAMERA_HEIGHT

Chase-camera height above the ground plane (world units).

### NEATRACING_CHASE_CAMERA_LOOKAHEAD

Chase-camera look-ahead distance along the camera aim (world units).

### NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD

The chase-camera MAXIMUM aim offset (rad): the NAMED CAP on the yaw aim
offset the camera may carry away from the nose heading.

UNIT: rad (an angle). SIDE: the cap is SYMMETRIC - it bounds the
magnitude of the signed offset, so a left-hand slide (negative offset,
the camera aiming below the nose) and a right-hand slide (positive
offset) are bounded by the SAME value.

MEASURED: the RED slice (P1-driftcam-red) MEASURED the pre-wave peak
aim offset of 0.34897858999999976 rad over a 1200-tick hold and
RECORDED this cap as EXACTLY HALF of that measured peak (the half is
the only USER-NAMED number of the wave). The cap is enforced AT THE
SOURCE (inside this law), so every reader - the HUD reader and the
browser reader - can only ever see the CAPPED offset.

### NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION

The chase-camera release smooth fraction: the maximum fraction of its
own value the aim may close in ONE release tick.

UNIT: a dimensionless fraction (0..1) of the aim offset. SIDE:
SIGNLESS - it bounds the MAGNITUDE of the release step, and the release
never flips the side of the aim.

MEASURED: the RED slice (P1-driftcam-red) MEASURED the pre-wave
one-tick release close fraction as 0.10498288025082125 at the peak
offset (the pre-wave rate was MULTIPLIED by the camera's own offset -
the multiplicative snap-back enabler this law deletes) and DECIDED
this RECORDED smooth fraction below it, so the one-tick snap-back is
impossible.

### resolveChaseCameraAimRad

```ts
resolveChaseCameraAimRad(
  headingRad: number,
  followOffsetRad: number,
): number
```

The ONE aim law: the camera aim is the nose heading plus the follow
offset.

Every live consumer (the camera placement, the look target and the
starfield yaw) reads THIS function, so the frame can never show a camera
that looks one way and a sky that slides another (RED-3).

Parameters:
- `headingRad` - The car nose heading (rad).
- `followOffsetRad` - The chase-camera follow offset (rad).

Returns: The camera aim (rad).

### resolveChaseCameraBasis

```ts
resolveChaseCameraBasis(
  carX: number,
  carY: number,
  aimRad: number,
  out: ChaseCameraBasis | undefined,
): ChaseCameraBasis
```

Resolves the chase-camera basis from the camera aim ALONE: the camera
eye, the look target and the normalized camera forward.

The basis is a pure function of the aim, so the three live call sites
(the camera placement, the look target and the starfield yaw) can only
agree if they all read the ONE aim law (RED-3), and a straight lane (a
published slip of EXACTLY 0) reproduces the pre-wave basis BIT-FOR-BIT
(RED-4).

The scene mirrors the top-down y into z (`worldZ = -y`) - the SAME
handedness seam as `carSceneWorldZ` in the browser entry. The mirror is
kept LOCAL here because the browser entry imports this module: importing
the helper back would close an import cycle.

Parameters:
- `carX` - The car x coordinate (m).
- `carY` - The car top-down y coordinate (m).
- `aimRad` - The camera aim (rad).
- `out` - Optional destination bag. When supplied, the basis is
written INTO it and returned, so a hot render path can refresh a
STABLE bag with no per-frame allocation. When omitted, a fresh basis
is returned.

Returns: The chase-camera basis.

### resolveChaseCameraFollowStepRad

```ts
resolveChaseCameraFollowStepRad(
  offsetRad: number,
  step: ChaseCameraFollowStep,
): number
```

Resolves the ONE per-tick follow step of the camera aim.

THE LAW (Step 14X, AC-097): the aim is pulled toward the SUPPORTED aim
- the PUBLISHED travel divergence CAPPED at the NAMED
`NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD` - by a PROPORTIONAL rate
(the car SPEED times the shared drift amount, scaled by the DESENSITISED
named gain, never a fixed time constant), and the part of the aim the
published slide no longer supports is RELEASED at a CONSTANT fraction of
the aim.

THE MULTIPLICATIVE SNAP-BACK ENABLER IS DELETED AT ITS SOURCE: the
pre-wave law fed the camera's OWN remaining divergence into the follow
rate (`max(|published slip|, |aim error|)`), so the release closed
FASTER the further the camera already sat from the nose. The rate here
reads the PUBLISHED slide ONLY, so the release never accelerates and the
one-tick snap-back is impossible.

The per-tick change is bounded (no snap) and the aim is approached along
the shortest path (no overshoot, no discontinuity).

Parameters:
- `offsetRad` - The current camera aim offset from the nose (rad).
- `step` - The step descriptor.

Returns: The signed offset change for this tick (rad).

### signedShortestDeltaRad

```ts
signedShortestDeltaRad(
  fromRad: number,
  toRad: number,
): number
```

The SIGNED SHORTEST-PATH delta from one angle to another, so a heading
crossing plus or minus pi never turns the camera by 2 pi (RED-7).

Parameters:
- `fromRad` - The angle travelled from (rad).
- `toRad` - The angle travelled to (rad).

Returns: The shortest signed delta (rad).

### stepChaseCameraFollow

```ts
stepChaseCameraFollow(
  follow: ChaseCameraFollowState,
  step: ChaseCameraFollowStep,
  out: ChaseCameraFollowState | undefined,
): ChaseCameraFollowState
```

Advances the chase-camera follow over the elapsed simulation ticks.

THE LAW (RED-5, the user law S-1/S-4): the per-tick aim change is
PROPORTIONAL to the car speed times the shared drift amount, so the
follow is a slight ANIMATION with no snap: it eases onto the published
slide at low speed and locks onto the CAPPED slide at speed. The
steady-state error against the travel divergence is ZERO only while the
published divergence stays INSIDE the cap; when the divergence is WIDER
the aim HOLDS AT THE CAP and the residual steady-state error is the
divergence MAGNITUDE minus the cap. A fixed time constant (the retired,
FORBIDDEN `NEATRACING_CHASE_CAMERA_FOLLOW_TAU_S`) would leave the rate
unchanged when the speed doubles - the S-1 violation.

THE MECHANICS:

- `ticks: 0` HOLDS the aim (a frame that released no simulation tick
  must not move the camera);
- a tick REGRESSION (`tick < follow.lastTick` - a new car or a mode
  switch rewinds the deterministic tick counter) resets the offset to
  EXACTLY 0, so no stale slide survives a respawn (RED-7). This is the
  ONE reset owner;
- the aim is always approached along the SHORTEST PATH, so a heading
  crossing plus or minus pi never turns the camera by 2 pi (RED-7).

Parameters:
- `follow` - The current follow state (read-only; never mutated).
- `step` - The step descriptor (the elapsed ticks, the nose heading,
the published travel heading, the speed and the tick index).
- `out` - Optional destination bag. When supplied, the next state is
written INTO it and returned, so a hot render path can refresh a
STABLE bag with no per-frame allocation. When omitted, a fresh state
is returned.

Returns: The NEXT follow state.

### wrapRad

```ts
wrapRad(
  angleRad: number,
): number
```

Wraps an angle into the `[-pi, pi)` band.

Parameters:
- `angleRad` - The angle to wrap (rad).

Returns: The wrapped angle (rad).

### writeChaseCameraFollowState

```ts
writeChaseCameraFollowState(
  out: ChaseCameraFollowState | undefined,
  offsetRad: number,
  lastTick: number,
): ChaseCameraFollowState
```

Writes a chase-camera follow state into the destination bag.

Parameters:
- `out` - Optional destination bag; a fresh state is created when it
is omitted.
- `offsetRad` - The next camera aim offset (rad).
- `lastTick` - The tick the follow is now anchored on.

Returns: The written follow state.

## browser-entry/browser-entry.ts

### applyRendererSize

```ts
applyRendererSize(
  canvas: HTMLCanvasElement,
): void
```

Applies the fixed-canvas letterbox strategy from the live viewport once.

Resizes the canvas backing store, positions the CSS display box with the
matching letterbox layout, and records a window-size annotation so the
next browser smoke test can cross-check the full-page render box.

Parameters:
- `canvas` - Canvas whose backing store and CSS box are resized.

### attachResizeListener

```ts
attachResizeListener(
  canvas: HTMLCanvasElement,
  session: BootSession,
  starfieldPass: StarfieldPass | null,
): void
```

Installs the non-debounced viewport-resize listener that re-applies the
letterbox strategy from the live viewport (spec §6.2).

Also re-uploads the starfield pass backdrop size so the star map stays
aligned with the computed backing width after a resize (OBS-F).

Parameters:
- `canvas` - Canvas whose backing store tracks the viewport.
- `session` - Boot session storing the listener for teardown.
- `starfieldPass` - Starfield pass whose NDC mapping is updated.

### BootSession

Boot-session teardown bookkeeping (frame loop, resize listener, fallback
notice, stopped flag) shared by the wired boot surfaces.

### BootShellOverlays

Boot-mounted overlay controllers handed to the live drive loop.

### buildNeatRacingTrackChordIndex

```ts
buildNeatRacingTrackChordIndex(
  segments: readonly TrackSegment[],
): NeatRacingTrackChordIndex
```

Builds the chord index the host-owned progress meter rides.

The chords are the generated track's segments in loop order; the arc-length
offsets are ONE prefix-sum over the chord lengths, so the index is built
once per boot and never rebuilt per frame.

Parameters:
- `segments` - The generated track's segments in loop order.

Returns: The chord index and the total loop length.

Example:

```ts
const index = buildNeatRacingTrackChordIndex(trackSpec.segments);
```

### carSceneWorldZ

```ts
carSceneWorldZ(
  topDownY: number,
): number
```

Maps a top-down math-plane y coordinate into the mirrored scene z axis
(`worldZ = -y`), the single ground-plane handedness seam for the frame
composer's car pose, camera eye/target, and baked track slabs.

Parameters:
- `topDownY` - Physics-plane y coordinate in meters.

Returns: Scene z coordinate in meters.

### carSceneYawRad

```ts
carSceneYawRad(
  topDownHeadingRadians: number,
): number
```

Maps a top-down math heading into the mirrored scene's car yaw so the
model +z nose points along the mirrored scene forward.

Parameters:
- `topDownHeadingRadians` - Physics-plane heading (0 = +x, CCW positive).

Returns: Scene yaw fed to the car body matrix and camera pose.

### createBootProgramRegistry

```ts
createBootProgramRegistry(
  gl: WebGL2RenderingContext,
): readonly ShaderWarmupProgram[]
```

Builds the per-boot warmup registry in the pinned AC-040 key order:
starfield, grid, pulse, track, car, glass, bloom-threshold, bloom-blur,
bloom-additive (`NEATRACING_WARMED_PROGRAM_KEYS`).

The legacy 1 m² engine grid pass has been replaced by the 2 m² track-grid
pass (AC-045). The `grid` registry slot now warms the track-grid shader so
the canonical key count stays unchanged and no dual-path draw code remains.

Parameters:
- `gl` - Live WebGL2 context the warmup programs compile against.

Returns: The warmup programs in the pinned registry-key order, keyed so
the `grid` slot compiles the track-grid program.

### createBootSession

```ts
createBootSession(): BootSession
```

Creates the boot-session teardown bookkeeping with every surface unset.

Returns: A fresh boot session for the current boot call.

### createBootShellOverlays

```ts
createBootShellOverlays(): BootShellOverlays
```

Mounts the HUD overlay, the race mode selector, and the track selector
into the reserved host container at boot (AC-024 partial; batch 94
handoff; AC-101 track selection) and returns the controllers.

The HUD mounts first, then the race mode selector, then the track
selector (the read model before the control surfaces), mirroring the
overlay ordering. The factories resolve the reserved
{@link NEATRACING_HUD_OUTPUT_ID} container themselves and throw when the
shell omits it (index.html owns the reservation), so the boot path owns
only the ordering. Stubbed service doubles may mount nothing; the drive
loop stays wired on inert fallback controllers in that case
(P1-drive-loop-live: AC-024 contract requires the wiring chain to survive
undefined controllers).

Returns: The mounted HUD, race-mode-selector, and track-selector
controllers (inert fallback controllers when a mount factory yields
nothing or cannot resolve the shell container).

### createHudFallback

```ts
createHudFallback(): { update(state: NeatRacingHudState): void; }
```

Inert HUD fallback for stubbed mount surfaces (no-op updates).

Returns: An HUD-compatible controller whose updates do nothing.

### createSelectorFallback

```ts
createSelectorFallback(): { readonly mode: NeatRacingRaceMode; onSwitch(listener: (mode: NeatRacingRaceMode) => void): void; }
```

Inert selector fallback for stubbed mount surfaces (manual mode).

Returns: A selector-compatible controller with a no-op switch carrier.

### createStopHandle

```ts
createStopHandle(
  canvas: HTMLCanvasElement,
  session: BootSession,
): NeatRacingStop
```

Builds the boot stop handle that tears down every wired surface
symmetrically: the RAF loop, the keyboard listeners, the resize listener,
the composer + passes, any appended fallback notice, and the mounted game
canvas (AC-001).

Parameters:
- `canvas` - Mounted game canvas removed by the handle.
- `session` - Boot session holding the surfaces to tear down.

Returns: The stop handle.

### createTrackSelectorFallback

```ts
createTrackSelectorFallback(): { readonly trackId: string; setSelectedTrack(trackId: string): void; onSwitch(listener: (trackId: string) => void): void; }
```

Inert track-selector fallback for stubbed mount surfaces (the catalog
default — the retired `NEATRACING_DEMO_TRACK_ID` boot literal's catalog
home; no dual path).

Returns: A track-selector-compatible controller with a no-op switch
carrier selected to the catalog default.

### finishFallbackBoot

```ts
finishFallbackBoot(
  output: HTMLElement,
  canvas: HTMLCanvasElement,
  session: BootSession,
  fallbackNotice: HTMLElement | null,
): NeatRacingStop
```

Completes the degraded boot when WebGL2 is unavailable.

Appends the service-built fallback notice into the boot container (the
editor-entry precedent for a reduced view) and returns the same stop-handle
contract so an embedding host can always unmount the boot cleanly. Rendering
is never started on this path: there is no live context to draw into.

Parameters:
- `output` - Host container owning the boot content.
- `canvas` - Mounted game canvas removed by the stop handle.
- `session` - Boot session tracking the notice for symmetric teardown.
- `fallbackNotice` - Service-built notice element, or null when the
init result carried no DOM notice (headless hosts).

Returns: A stop handle that removes the mounted boot content.

### getOutputContainer

```ts
getOutputContainer(
  outputId: string,
): HTMLElement
```

Resolves the host output container, throwing a descriptive error when the
shell does not provide it (mounted-canvas contract, AC-001).

Parameters:
- `outputId` - Host container element id to resolve.

Returns: The resolved host container.

### lookAtInto

```ts
lookAtInto(
  out: Float32Array<ArrayBufferLike>,
  eyeX: number,
  eyeY: number,
  eyeZ: number,
  targetX: number,
  targetY: number,
  targetZ: number,
): void
```

Writes a right-handed look-at view matrix (column-major) into `out` with a
world-up vector of `(0, 1, 0)`.

Parameters:
- `out` - 16-float matrix destination (overwritten).
- `eyeX` - Camera eye X.
- `eyeY` - Camera eye Y.
- `eyeZ` - Camera eye Z.
- `targetX` - Look target X.
- `targetY` - Look target Y.
- `targetZ` - Look target Z.

### mountGameCanvas

```ts
mountGameCanvas(
  output: HTMLElement,
): HTMLCanvasElement
```

Creates the game canvas and mounts it into the host container.

The canvas is mounted before context acquisition so the service memoizes
the WebGL2 context against the actual mounted element.

Parameters:
- `output` - Host container that will own the canvas.

Returns: The freshly mounted game canvas.

### mountNeatRacingTrackSelector

```ts
mountNeatRacingTrackSelector(): { readonly trackId: string; setSelectedTrack(trackId: string): void; onSwitch(listener: (trackId: string) => void): void; }
```

Mounts the track-selector overlay into the reserved HUD container,
degrading to the inert fallback when the resolver-or-throw factory cannot
mount (AC-101).

WHY the real factory is caught here (the sibling overlays use the plain
`??` tolerance instead): the track selector is a SECONDARY control — the
race remains fully usable without it (the selection then simply never
leaves the CATALOG DEFAULT track), and the sibling mount doubles are
stubbed in the boot-composition suites while this factory runs REAL, so
a stubbed shell (no reserved HUD container) must keep the boot alive on
the inert fallback instead of failing it. The degradation is silent on
purpose (the `??` fallback paths never warn either), and the live shell
always reserves the container (index.html owns the reservation), so no
real boot-order bug is hidden by this tolerance.

Returns: The mounted track selector, or the inert fallback controller.

### multiplyViewProjectionInto

```ts
multiplyViewProjectionInto(
  out: Float32Array<ArrayBufferLike>,
  view: Float32Array<ArrayBufferLike>,
  projection: Float32Array<ArrayBufferLike>,
): void
```

Writes `projection * view` (column-major) into `out`. All three matrices
are 16-float column-major; `out` must be distinct from the inputs.

The GLSL world passes consume `uViewProjection * worldPosition`, so the
combined matrix must apply the projection after the view transform. This
helper is intentionally named with the operands `(out, view, projection)`
to match the call-site wording while computing `projection * view`.

Parameters:
- `out` - 16-float matrix destination.
- `view` - View matrix.
- `projection` - Projection matrix.

### NEATRACING_CAR_SCENE_YAW_OFFSET_RAD

Scene handedness (Step 14B round-3): the physics integrator works in the
top-down math plane where positive heading rotates counter-clockwise, and
the WebGL2 camera reads left-handed screen axes from `lookAtInto` (with the
camera looking along scene +x at yaw 0, screen right is scene +z). Mirroring
the top-down y into scene z (`worldZ = -y`) aligns the two: positive heading
tips the car nose toward scene -z, which the chase camera reads as screen
LEFT, matching the A key. The yaw offset then rotates the car model's +z
nose (per the wheel mount car-frame contract) so the nose points along the
mirrored scene forward `(cos heading, 0, -sin heading)` instead of the raw
mirrored top-down forward.

### NeatRacingAiModeGoalState

The host-owned AI-mode goal lifecycle state.

### NeatRacingAiModeRebaseDecision

The host decision surface for one AI-mode goal crossing.

### NeatRacingCarProgressState

The host-owned along-track progress state of ONE car.

UNIT: the progress meter measures METERS along the track's centerline
(the sum of the generated segments' chord lengths).

### NeatRacingDriftState

The 14N drift debug seam: the read-only, EXACTLY eight-field view of the
live drift channel that the browser half measures. It is a debug
surface, not a gameplay API.

### NeatRacingDriftStateBag

The mutable drift-seam BAG: the SAME field set as the published view
with the `readonly` modifiers stripped, so the frame builder can
refresh the seam IN PLACE.

Step 14Q2 (AC-088): the zero per-frame allocation rule holds - the seam
object is a STABLE BAG, not a fresh object per frame. ONE bag is born
per boot and the window handle keeps pointing at the SAME object, so
this seam growth (and the next one) adds a field WITHOUT adding a
per-frame allocation.

### NeatRacingHostRaceState

The host-owned race state: what the drive loop reads from the worker
protocol on every frame.

### NeatRacingLapFactAdvance

The host-owned lap-fact bookkeeping for one rendered frame.

### NeatRacingNetworkPanelState

The host-owned most-forward network panel state.

### neatRacingStart

```ts
neatRacingStart(
  outputId: string,
): NeatRacingStop
```

Mount the game canvas into the host shell, wire the live render path, and
return a stop handle.

The mount → overlays → init → warmup → letterbox → RAF loop order is the
AC-002 boot contract: the canvas is mounted first, the HUD overlay and the
race mode selector are mounted into the reserved shell container (AC-024
partial), the context is acquired exactly once for it, the shader warmup
runs against the per-boot program registry before the first frame, the
letterbox strategy is applied from the live viewport, and the
fixed-timestep accumulator is advanced from `requestAnimationFrame` (one
request at boot, one re-request per tick). Without WebGL2 the boot degrades
gracefully by appending the service-built fallback notice (editor-entry
precedent) while returning a functional stop handle.

Parameters:
- `outputId` - Host container element id that will own the game canvas.

Returns: A stop handle that removes the mounted canvas.

Example:

```ts
const stop = window.neatRacingStart('neatracing-output');
stop(); // unmounts the demo canvas
```

### NeatRacingStart

```ts
NeatRacingStart(
  outputId: string,
): NeatRacingStop
```

Boot function installed on `window` by the host bundle.

Parameters:
- `outputId` - Host container element id that will own the game canvas.

Returns: A stop handle that removes the mounted canvas.

### NeatRacingStop

```ts
NeatRacingStop(): void
```

Teardown handle returned by the boot stub.

Calling it stops the render loop, removes the resize listener and any
appended fallback notice, and removes the mounted game canvas so an
embedding host can unmount the demo cleanly (used by tests and future
hot-reload paths).

### NeatRacingTrackChord

One track chord: a straight piece of the generated loop plus its
along-track arc-length offset.

WHY: the host-owned along-track progress meter measures a car's position
by the NEAREST-CHORD rule over the generated segments, so the meter needs
the chord geometry (start, end, length) and the arc-length offset of each
chord in loop order.

### NeatRacingTrackChordIndex

The boot-time chord index the host-owned progress meter rides.

### NeatRacingWheelSpinState

The host-owned display wheel-spin state of ONE car.

### perspectiveInto

```ts
perspectiveInto(
  out: Float32Array<ArrayBufferLike>,
  fovYRadians: number,
  aspect: number,
  near: number,
  far: number,
): void
```

Writes a right-handed perspective projection (column-major, WebGL clip
conventions) into `out` in place.

Parameters:
- `out` - 16-float matrix destination (overwritten).
- `fovYRadians` - Vertical field of view in radians.
- `aspect` - Width/height aspect ratio.
- `near` - Near clip distance.
- `far` - Far clip distance.

### resolveAdaptiveQualityStageClamped

```ts
resolveAdaptiveQualityStageClamped(
  stageIndex: number,
): AdaptiveQualityStage
```

Resolves the defensively clamped adaptive-quality stage record for a
controller stage index (AC-035): negative, out-of-range or non-finite
controller outputs clamp into the stage ladder so the off-switch knob is
ALWAYS a defined stage record (never `undefined`).

Parameters:
- `stageIndex` - The raw stage index handed back by the ticked
controller.

Returns: The resolved (defensively clamped) stage record.

### resolveChordAlongM

```ts
resolveChordAlongM(
  chord: NeatRacingTrackChord,
  pointX: number,
  pointY: number,
): number
```

Resolves the along-track offset of a point on a chord (meters).

Parameters:
- `chord` - The chord to measure against.
- `pointX` - The point's generator x coordinate.
- `pointY` - The point's generator y coordinate.

Returns: The along-track offset from the loop start (meters).

### resolveChordDistanceSq

```ts
resolveChordDistanceSq(
  chord: NeatRacingTrackChord,
  pointX: number,
  pointY: number,
): number
```

Resolves the squared distance from a point to the nearest point of a
chord (the clamped projection).

Parameters:
- `chord` - The chord to measure against.
- `pointX` - The point's generator x coordinate.
- `pointY` - The point's generator y coordinate.

Returns: The squared distance.

### resolveNearestChordIndex

```ts
resolveNearestChordIndex(
  index: NeatRacingTrackChordIndex,
  pointX: number,
  pointY: number,
  incumbentIndex: number,
): number
```

Resolves the nearest chord of a point (the nearest-chord rule).

The scan is WINDOWED around the incumbent chord index (a car moves a few
chords per frame) and falls back to a FULL scan whenever the windowed
winner sits on the window boundary, so the nearest chord is always the
true nearest one.

Parameters:
- `index` - The boot-time chord index.
- `pointX` - The point's generator x coordinate.
- `pointY` - The point's generator y coordinate.
- `incumbentIndex` - The incumbent chord index of the previous frame.

Returns: The index of the nearest chord.

### resolveNeatRacingAiModeGoalSessionForMode

```ts
resolveNeatRacingAiModeGoalSessionForMode(
  mode: NeatRacingRaceMode,
  bootLevel: NeatRacingObservationLevel | undefined,
): NeatRacingAiModeGoalSession | null
```

Resolves the AI-mode goal session a mode switch must boot.

A switch INTO `ai` boots a FRESH goal session; a switch to `manual`
parks the AI-mode lifecycle (the adopted session stays adopted and NO
further goal facts are fed), so the resolver returns null.

Educational note — the carried boot level: the goal session is built at
the level the ROUND boots (or switches) at, not the mode's default
start level. The lifecycle's goals measure the observation roster the
goal state materializes, so the lifecycle must describe the SAME level
the player is driving (a switch through the debug deck's rung pins the
carried level on the lifecycle too). The legacy two-arg-free call keeps
the mode's default start level.

Parameters:
- `mode` - The UI race mode the host switched to.
- `bootLevel` - The observation level the round boots (or switches)
at; the mode's default start level when omitted.

Returns: The fresh goal session, or null when the lifecycle parks.

Example:

```ts
const booted = resolveNeatRacingAiModeGoalSessionForMode('ai');
const carried = resolveNeatRacingAiModeGoalSessionForMode('ai', 2);
```

### resolveNeatRacingAiModeRebaseRung

```ts
resolveNeatRacingAiModeRebaseRung(
  outcome: NeatRacingAiModeGoalCrossingOutcome,
): NeatRacingAiModeRebaseDecision
```

Resolves the host's rebase decision for one AI-mode goal crossing.

The decision surface is PURE: the host applies the rebase (re-boot the
worker round on the promoted rung) and the suite can exercise the same
decision without a live promoted round.

Parameters:
- `outcome` - The goal machine's outcome for the crossing.

Returns: The rebase decision (null when the round does not rebase).

Example:

```ts
const decision = resolveNeatRacingAiModeRebaseRung(outcome);
if (decision.rebaseRung !== null) {
  rebaseRoundAtPromotion(decision.rebaseRung);
}
```

### resolveNeatRacingCarAlongProgressM

```ts
resolveNeatRacingCarAlongProgressM(
  index: NeatRacingTrackChordIndex,
  carX: number,
  carY: number,
  previous: NeatRacingCarProgressState | null,
): NeatRacingCarProgressState
```

Resolves the host-owned along-track progress of one car.

UNIT: METERS along the track's centerline (the sum of the generated
segments' chord lengths). The position inside the loop comes from the
NEAREST-CHORD rule (the clamped projection onto the nearest chord); the
total accumulates the forward distance and detects a half-loop wrap.

WRAP LAW: the wrap direction is read from the SIGN of `deltaM`; the
loop-length correction applies OPPOSITE that sign. A FORWARD crossing
of the loop start DROPS the global along-track position (`deltaM` very
negative) and grows the total by the loop tail + head
(`deltaM + loopLengthM`, the +L correction opposite the negative
sign). A BACKWARD crossing RAISES the position (`deltaM` very
positive) and DECREMENTS the total by the wrapped shorter displacement
(`deltaM - loopLengthM`): the rivals never own the rules' laps (the
worker's lap detector is car-0-only), so the host accepts the
decrement instead of inventing a lap counter of its own.

Parameters:
- `index` - The boot-time chord index.
- `carX` - The car's generator x coordinate.
- `carY` - The car's generator y coordinate.
- `previous` - The previous progress state, or null on the first frame.

Returns: The car's progress state.

Example:

```ts
const progress = resolveNeatRacingCarAlongProgressM(
  index,
  car.x,
  car.y,
  previous,
);
```

### resolveNeatRacingLapFactAdvance

```ts
resolveNeatRacingLapFactAdvance(
  lapBaseline: number,
  lapsCompleted: number,
): NeatRacingLapFactAdvance
```

Resolves the host-owned lap-fact bookkeeping for one rendered frame.

The worker owns the lap detector (car 0 only); the host keeps the
baseline it adopted at the round boot and derives the laps completed since
the last crossing from the protocol's forward-lap counter.

Parameters:
- `lapBaseline` - The laps completed the host adopted as its baseline.
- `lapsCompleted` - The laps completed the rules clone now reports.

Returns: The lap-fact advance (no crossing when nothing changed).

Example:

```ts
const advance = resolveNeatRacingLapFactAdvance(0, 1);
// { lapBaseline: 1, lapsSinceLastCrossing: 1, crossed: true }
```

### resolveNeatRacingObservationRung

```ts
resolveNeatRacingObservationRung(
  level: number,
): NeatRacingObservationLevel
```

Narrows a round's boot level onto the observation ladder's rung.

Educational note — why the narrowing is needed: the race rules carry the
round's level as a plain NUMBER (the ladder's rung is a number there),
while the adaptation session's observation level is the ladder's TYPED
rung. Both ladders carry the same rungs (1, 2 and 3), so the round's own
boot level narrows onto the observation rung unchanged; a level outside
the ladder is CLAMPED onto it (a corrupted rules state can never seed the
session with an unknown observation tier).

Parameters:
- `level` - The round's carried boot level.

Returns: The observation rung the level names.

### resolveStarfieldBackingDimensions

```ts
resolveStarfieldBackingDimensions(
  canvas: HTMLCanvasElement,
): { width: number; height: number; }
```

Resolves the computed backing dimensions for the starfield pass.

Uses the letterbox-strategy resolver from canvas-dimensions.utils so the
pass maps stars across the intended fixed-render-width backing store rather
than whatever the raw canvas.width happens to be at construction time
(OBS-F). This is also the source of truth for the resize listener.

Parameters:
- `canvas` - Mounted game canvas (or structural stub).

Returns: Computed backing-store dimensions in pixels.

### startLiveDriveLoop

```ts
startLiveDriveLoop(
  gl: WebGL2RenderingContext,
  session: BootSession,
  overlays: BootShellOverlays,
  canvas: HTMLCanvasElement,
): { handleFrame: (timestampMs: number) => void; }
```

Builds the live drive session and starts the RAF-driven drive: keyboard
presses are tracked on the window, the composer scene is assembled once
(decode + mesh the static voxel assets, per-pass program construction), and
every frame runs through the simulation-worker protocol
(`sample → worker frame → rebind → lap facts → HUD → panel → composer`).

Composition is best-effort within a live render path: when the context
cannot compile programs (fake/headless GL surfaces) or the scene assembly
throws, the session keeps running against an inert composer that swallows
draw frames, so the drive loop (input, the worker protocol, the HUD)
stays live
without render-path noise. The composer and its passes are disposed by the
boot stop handle through the session teardown hook.

Step 05 (P2-S4-C) moved the SIMULATION out of the host: the worker owns
the spawning, the physics, the race rules and the lap-line detector, and
the host owns the input sampling, the composer bindings, the AI-mode goal
lifecycle, the most-forward network panel and the teardown. Car 0 is the
AUTHORITY for the chase camera, the HUD, the drift seam and the
right-world pass; the rivals (cars 1..n) are the worker's fleet.

Parameters:
- `gl` - Live WebGL2 context for the mounted canvas.
- `session` - Boot session receiving teardown bookkeeping.
- `overlays` - Boot-mounted HUD + race mode + track selector
controllers.
- `canvas` - Mounted game canvas (aspect source for the camera).

Returns: The drive session handed to the RAF loop.

### startRenderLoop

```ts
startRenderLoop(
  session: BootSession,
  driveSession: { handleFrame(timestampMs: number): void; },
): void
```

Runs the session-driven render loop.

Exactly one `requestAnimationFrame` is requested at boot; each frame hands
the timestamp to the drive session, which forwards it through the
simulation-worker protocol (sample → worker frame → rebind → HUD → panel →
composer) and re-requests the next frame so the loop stays alive. The boot
itself carries no accumulator: the worker owns the fixed-timestep
accumulator (P2-S4-C), so the host's per-frame entry point is a pure
protocol hop. A stopped session never schedules again, so the stop handle
terminates the loop deterministically.

Parameters:
- `session` - Boot session storing the current frame handle.
- `driveSession` - Session whose `handleFrame` drives one frame.

## browser-entry/constants.starfield.ts

### NEATRACING_SKY_BAND_HEIGHT_PX

Old above-horizon sky-band height in pixels, retained as the horizon
reference for the starfield reach test (AC-059).

The starfield originally stopped at this 240-pixel line; the new layout
spans the full {@link NEATRACING_FIXED_RENDER_HEIGHT} backdrop so stars
are generated all the way down to the horizon.

### NEATRACING_STARFIELD_BLINK_AMPLITUDE

Subtle per-star brightness oscillation amplitude (AC-060).

Alpha is multiplied by `1 + blinkAmplitude * sin(...)`, so a value of 0.08
produces an ±8% swing around the base alpha. Kept small to avoid strobing.

### NEATRACING_STARFIELD_BLINK_BASE_HZ

Base blink beat frequency in hertz for the far layer (AC-060).

Each layer adds {@link NEATRACING_STARFIELD_BLINK_BEAT_DELTA_HZ} so the
three layers have slightly different beats and the field stays uneven.

### NEATRACING_STARFIELD_BLINK_BEAT_DELTA_HZ

Per-layer blink beat frequency delta in hertz (AC-060).

Layer 0 = base, layer 1 = base + delta, layer 2 = base + 2*delta.

### NEATRACING_STARFIELD_BLINK_PHASE_SEED_STEP

Seed step used to spread per-star blink phase offsets deterministically.

The phase offset for a star is derived from an xorshift32 stream seeded
with `layerSeed + starIndex * step`. The value is odd so the low 32 bits
walk through distinct offsets for the small star counts used here.

### NEATRACING_STARFIELD_CYAN_FILL_STYLE

Fill color used for every star in the flappy-parity cyan tint.

The string keeps source-text parity with the flappy
`FLAPPY_STARFIELD_CYAN_FILL_STYLE` constant; the WebGL pass converts the
same RGB triple into a shader uniform.

### NEATRACING_STARFIELD_FAR_SCROLL_RATIO

Horizontal scroll ratio for the farthest parallax layer (flappy parity).

One full camera revolution scrolls the far layer by 3% of a tile width.

### NEATRACING_STARFIELD_FAR_STAR_COUNT

Restored per-layer star counts after the AC-059 horizon stretch.

The original sky-band counts (far 35, mid 28, near 23) covered only the
240 px top band. Stretching the same count over the full 720 px backdrop
cut visible sky-band density to roughly one third, so counts are scaled by
the ratio 720/240 = 3 to keep the same visual density in the sky band.

### NEATRACING_STARFIELD_GLOOM_AMPLITUDE

Subtle short gloom-cycle dim amplitude (AC-060).

Alpha is multiplied by `1 - gloomAmplitude * (0.5 + 0.5 * sin(...))`,
so a value of 0.05 dims stars by at most 5%. Always kept below the blink
amplitude so the gloom is the quieter of the two effects.

### NEATRACING_STARFIELD_GLOOM_PERIOD_SECONDS

Short gloom cycle period in seconds (AC-060).

A 4.2 s period keeps the dimming slow enough to read as atmosphere but
short enough to be visible.

### NEATRACING_STARFIELD_INCLUSIVE_RANGE_OFFSET

Inclusive range offset used when converting size bounds into random spans.

Adding 1 before `Math.floor` makes the upper size bound reachable, matching
the flappy placement algorithm exactly.

### NEATRACING_STARFIELD_LAYER_SPECS

Declarative layer specs for the three flappy-parity parallax bands.

Ordered far → mid → near; the pass draws them back-to-front. The far layer
seed is the visual seed base itself (TP-20), and no cloud layers exist in
the voxel racer sky (AC-004 pins exactly these three star layers).

### NEATRACING_STARFIELD_MID_SCROLL_RATIO

Horizontal scroll ratio for the middle parallax layer (flappy parity).

### NEATRACING_STARFIELD_MID_STAR_COUNT

### NEATRACING_STARFIELD_NEAR_SCROLL_RATIO

Horizontal scroll ratio for the nearest parallax layer (flappy parity).

### NEATRACING_STARFIELD_NEAR_STAR_COUNT

### NEATRACING_STARFIELD_TILE_HEIGHT_PX

Height of one repeated starfield tile: the full backdrop height so stars
cover the whole sky region down to the horizon (AC-059).

### NEATRACING_STARFIELD_TILE_WIDTH_PX

Width of one repeated starfield tile in pixels (flappy parity).

The tile wraps horizontally across the backing store, so this width fixes
both the generated point layout and the horizontal wrap period.

### NEATRACING_STARFIELD_UNSIGNED_NORMALIZATION_DIVISOR

Divisor that normalizes unsigned 32-bit RNG state into the unit interval.

### NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_FINAL

Final left-shift used by the xorshift32 random transition.

### NEATRACING_STARFIELD_XORSHIFT_LEFT_SHIFT_PRIMARY

Primary left-shift used by the xorshift32 random transition.

### NEATRACING_STARFIELD_XORSHIFT_RIGHT_SHIFT

Middle right-shift used by the xorshift32 random transition.

### NEATRACING_VISUAL_SEED

Visual-only seed base for every deterministic visual effect (TP-20).

Starfield layers derive their xorshift32 seeds from this base (the far
layer uses it directly). Because the value only shifts which pseudo-random
stars appear, it is exempt from replay byte parity per spec §9.7.2.

### StarfieldLayerSpec

Layer recipe describing one deterministic parallax starfield band.

The fields mirror the flappy layer spec so the placement algorithm can be
ported one-to-one; `id` names the layer for diagnostics and draw ordering.

## browser-entry/debug-rung.utils.ts

### NeatRacingDebugRungBootLevelSources

The boot-level sources the precedence resolver reads, in the exact shape
the host passes at the round boot.

### resolveNeatRacingBootLevel

```ts
resolveNeatRacingBootLevel(
  sources: NeatRacingDebugRungBootLevelSources,
): number
```

Resolves the level the host boots a round at, from the sources in
precedence order: the promotion ladder's pending level first, the debug
rung second, the session level last.

Educational note — THE LADDER KEEPS PRECEDENCE. A debug rung only seeds
the boot when the promotion ladder is silent: a pending `nextRaceLevel`
(a race-completion boundary set it) always boots first, so `?level` can
never fight a freshly promoted round.
Educational note — THE RUNG SEEDS A BOOT, IT GRANTS NO RUNG. The seeded
level feeds the SAME boot path every round takes, so the observation
tier and the fleet size still resolve through the ladder's own
resolvers — a debug boot races the identical rules, it only opens the
door to a rung.
Educational note — THE DEFAULT IS BYTE-IDENTICAL. With a null debug rung
this is exactly the host's former boot expression
(`nextRaceLevel ?? sessionLevel`).

Parameters:
- `sources` - The boot-level sources: `nextRaceLevel` (the ladder's
pending level, null until a race-completion boundary sets it),
`debugRungLevel` (the rung resolved from the optional `?level`
parameter, null when absent) and `sessionLevel` (the session level the
boot falls back to).

Returns: The boot level.

Example:

// The default boot (no ?level parameter) is byte-identical
expect(resolveNeatRacingBootLevel({
  nextRaceLevel: null,
  debugRungLevel: null,
  sessionLevel: 1,
})).toBe(1);
// A debug rung seeds only a silent ladder's boot
expect(resolveNeatRacingBootLevel({
  nextRaceLevel: null,
  debugRungLevel: 3,
  sessionLevel: 1,
})).toBe(3);

### resolveNeatRacingDebugRungLevel

```ts
resolveNeatRacingDebugRungLevel(
  queryString: string,
): number | null
```

Resolves the debug rung the OPTIONAL `level` query parameter names, or
null when the parameter is absent, malformed or outside the ladder.

Educational note — WHAT and WHY. The W2 debug rung lets a developer boot
the racer straight at a rung (`?level=2`) so the rival-bearing manual
rung is reachable in ONE boot — no promotion driving needed. The rung is
a debug/harness affordance: it is read ONCE at the opening boot, it
seeds a boot level, and it never writes the promotion ladder's state.
THE DEFAULT: with no `level` parameter this resolver answers null and
the host boots the byte-identical default (`nextRaceLevel` else the
session level), so the shipped demo is unaffected. THE ACCEPTED RUNGS:
exactly the ladder band, level 1 to `NEATRACING_PROMOTION_MAX_LEVEL`
(1, 2, 3) — the ceiling is IMPORTED from the promotion ladder so a
ladder change moves the seam. NOTHING IS COERCED: a fraction (`2.5`),
a signed name (`-1`), a non-numeric name (`abc`) and an out-of-band name
(`0` and above the ceiling) all answer null even though the LADDER'S own
clamp would have coerced some of them — the debug rung is a key to a
rung, not a re-derivation of it.
WHEN: the host calls this exactly once per page boot, inside the live
drive loop's round setup; the resolver may also be called from tests,
scripts and embeddings because it reads nothing global.

Parameters:
- `queryString` - The boot query string, sampled from
`window.location.search` by the host (a leading `?` is fine).

Returns: The ladder rung (1..3) the parameter names, or null when the
parameter is absent, malformed or out of the ladder band.

Example:

// The seeded rival-bearing rung
expect(resolveNeatRacingDebugRungLevel('?level=2')).toBe(2);
// The absent-parameter default stays byte-identical (null)
expect(resolveNeatRacingDebugRungLevel('?track=demo')).toBeNull();
// Malformed and out-of-ladder names are ignored, never coerced
expect(resolveNeatRacingDebugRungLevel('?level=2.5')).toBeNull();
expect(resolveNeatRacingDebugRungLevel('?level=9')).toBeNull();

## browser-entry/canvas-dimensions.utils.ts

### applyCanvasBackingStore

```ts
applyCanvasBackingStore(
  surface: ResizableCanvasSurface,
  dimensions: CanvasDimensions,
): void
```

Writes resolved dimensions onto the canvas backing store.

Thin DOM adapter in the Neatenstein parity shape: the host assigns the
backing-store width and height so the render loop can draw at the fixed
strategy resolution while CSS stretches the element to the layout box.

Parameters:
- `surface` - Canvas element (or structural stub) to resize.
- `dimensions` - Backing-store dimensions to apply.

### applyFullWindowStretchStyle

```ts
applyFullWindowStretchStyle(
  canvas: HTMLCanvasElement,
): void
```

Applies a full-window stretch style to the canvas.

The canvas fills the entire output container (which is itself sized to the
viewport), so the rendered image stretches edge-to-edge like Neateinstein
rather than being letterboxed. Backing-store dimensions are matched to the
CSS client size so the projection aspect ratio stays correct.

Parameters:
- `canvas` - Canvas element whose style box is stretched to the container.

### applyLetterboxLayoutStyle

```ts
applyLetterboxLayoutStyle(
  canvas: HTMLCanvasElement,
  letterbox: LetterboxLayout,
): void
```

Applies the letterbox layout to the canvas CSS box.

The canvas backing store is fixed at 720 px height, so CSS owns the
on-screen display box. The layout box is centered with absolute positioning
inside the full-viewport output container; the computed offsets keep the
rendered box inside the viewport bounds without page scroll.

Parameters:
- `canvas` - Canvas element whose style box is positioned.
- `letterbox` - Layout box produced by  {@link resolveLetterboxLayout} .

### CanvasDimensions

Resolved backing-store dimensions in pixels.

### CanvasDimensionSources

Reference dimensions available to the backing-store resolution ladder.

### FullWindowRendererSizeUpdateSources

Inputs for the {@link updateFullWindowRendererSize} orchestration.

### LetterboxLayout

Letterboxed layout box in pixels, sized to fit inside the viewport.

### LetterboxLayoutSources

Reference dimensions available to the letterbox layout resolution.

### logRendererSizeAnnotation

```ts
logRendererSizeAnnotation(
  canvas: HTMLCanvasElement,
  dimensions: CanvasDimensions,
  letterbox: LetterboxLayout,
  viewportWidth: number,
  viewportHeight: number,
): void
```

Records a deterministic, human-readable size annotation for evidence
capture. The message includes the backing-store resolution, the CSS
letterbox box, and the host viewport dimensions so full-page screenshots
can be cross-checked against the source of truth.

Parameters:
- `canvas` - Resized canvas.
- `dimensions` - Backing-store dimensions.
- `letterbox` - CSS layout box.
- `viewportWidth` - Host viewport width in pixels.
- `viewportHeight` - Host viewport height in pixels.

### proportionalBackingDimensions

```ts
proportionalBackingDimensions(
  referenceWidth: number,
  referenceHeight: number,
): CanvasDimensions
```

Derives the proportional backing dimensions for a measurable reference size.

The backing width is the reference aspect ratio applied to the fixed render
height, rounded to the nearest pixel and clamped between one pixel and the
{@link NEATRACING_MAX_BACKING_WIDTH} ultrawide guard.

Parameters:
- `referenceWidth` - Measurable reference width in pixels.
- `referenceHeight` - Measurable reference height in pixels.

Returns: Backing-store dimensions at the fixed render height.

### RendererSizeUpdate

Result of a renderer resize: the applied backing dimensions and the letterbox layout.

### RendererSizeUpdateSources

Inputs for the {@link updateRendererSize} orchestration.

### ResizableCanvasSurface

Minimal structural surface the strategy reads and writes (a canvas or a stub).

### resolveCanvasBackingDimensions

```ts
resolveCanvasBackingDimensions(
  sources: CanvasDimensionSources,
): CanvasDimensions
```

Resolves the backing-store dimensions from the available reference sizes.

The fallback ladder mirrors the Neatenstein canvas strategy: proportional
scaling from the live client dimensions when measurable, otherwise
proportional scaling from the viewport dimensions, and finally the
{@link NEATRACING_FALLBACK_CANVAS_WIDTH} constant at the fixed render
height when nothing is measurable.

Parameters:
- `sources` - Client and viewport reference dimensions; zeros mark
unmeasurable sources and advance the ladder.

Returns: Backing-store dimensions at  {@link NEATRACING_FIXED_RENDER_HEIGHT} *   height with the width clamped to  {@link NEATRACING_MAX_BACKING_WIDTH} .
 *

Example:

resolveCanvasBackingDimensions({ clientWidth: 3840, clientHeight: 1080, viewportWidth: 0, viewportHeight: 0 });
// → { width: 1920, height: 720 } (ultrawide 32:9 clamped)

### resolveFullWindowBackingDimensions

```ts
resolveFullWindowBackingDimensions(
  sources: CanvasDimensionSources,
): CanvasDimensions
```

Resolves backing-store dimensions that match the CSS client size.

Falls back to the viewport dimensions when the canvas has not been laid out
yet, then to the fixed fallback constants when both sources are missing.

Parameters:
- `sources` - Client and viewport reference dimensions.

Returns: Backing-store dimensions equal to the client size or fallback.

### resolveLetterboxLayout

```ts
resolveLetterboxLayout(
  sources: LetterboxLayoutSources,
): LetterboxLayout
```

Resolves the letterboxed layout box for the given viewport and content.

The box scales the content uniformly to fit inside the viewport and is
centered with floor arithmetic so offsets stay non-negative and the box
never exceeds the viewport bounds (no page scroll). Degenerate inputs
(non-positive dimensions) collapse to a zero-size box at the origin.

Parameters:
- `sources` - Viewport bounds and content dimensions in pixels.

Returns: Letterbox layout box with centered, non-negative offsets.

Example:

resolveLetterboxLayout({ viewportWidth: 2560, viewportHeight: 720, contentWidth: 1920, contentHeight: 720 });
// → { width: 1920, height: 720, offsetX: 320, offsetY: 0 } (horizontal letterbox)

### updateFullWindowRendererSize

```ts
updateFullWindowRendererSize(
  sources: FullWindowRendererSizeUpdateSources,
): CanvasDimensions
```

Applies a full-window stretch resize to the renderer.

Sets the canvas CSS box to fill its container, then resolves the backing
store from the live client dimensions (with viewport fallback) and applies
it. The returned dimensions match the pixel size the GPU will render into.

Parameters:
- `sources` - Canvas surface plus viewport fallback dimensions.

Returns: The applied backing-store dimensions.

### updateRendererSize

```ts
updateRendererSize(
  sources: RendererSizeUpdateSources,
): RendererSizeUpdate
```

Applies a resize to the renderer: resolves, applies, and reports the layout.

Orchestration entry in the Neatenstein parity shape. The backing-store
dimensions are resolved from the canvas client dimensions with a viewport
fallback, applied to the canvas surface, and the matching letterbox layout
is resolved so the host can position the canvas CSS box without scrolling.

Parameters:
- `sources` - Canvas surface plus viewport fallback dimensions.

Returns: The applied backing-store dimensions and the letterbox layout.

Example:

updateRendererSize({ canvas, viewportWidth: window.innerWidth, viewportHeight: window.innerHeight });
// → { dimensions: { width: 1280, height: 720 }, letterbox: { ... } }
