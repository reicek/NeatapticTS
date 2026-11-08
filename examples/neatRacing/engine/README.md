# engine

NeatRacing lighting model (slice P1-S4-D; spec sections 6.5-6.6).

Pure TypeScript owner of the car-surface lighting model shared by every lit
pass: a single directional sun, hemisphere ambient that mixes ground into
sky over the surface normal's up component, and exp2 distance fog matching
the grid pass draw distance. Every function here is a numeric mirror of the
GLSL chunk, and `NEATRACING_LIGHTING_GLSL_CHUNK` ships the uniform names
plus the equivalent GLSL so `car.pass.ts` can embed the chunk itself and
the shader cannot drift from this mirror.

Color discipline: all colors are authored as 0-255 bytes and normalized to
[0, 1] here once (byte / 255); no raw byte ever reaches a uniform feed.
Distances are meters; the fog density carries the 1/m unit in its name.

## engine/lighting.ts

### computeFogFactor

```ts
computeFogFactor(
  distanceMeters: number,
  density: number,
): number
```

Computes the exp2 distance-fog transmittance: 1 at the camera, exactly 0.5
at 1/density, and ~0.53 at half the 320 m grid-pass draw distance.

This mirrors the shader's `fogFactor(distanceMeters)` from
{@link NEATRACING_LIGHTING_GLSL_CHUNK}; keep both in lockstep.

Parameters:
- `distanceMeters` - Eye-space distance in meters (>= 0 expected).
- `density` - Extinction coefficient in 1/m (defaults to the tuned
 *    {@link NEATRACING_FOG_DENSITY_1_PER_M} ).

Returns: Transmittance in [0, 1]: 1 = clear, 0 = fully fogged.

Example:

computeFogFactor(0);                       // 1
computeFogFactor(1 / 0.006);               // ~0.5
computeFogFactor(160);                     // ~0.53 at half draw distance

### computeHemisphereAmbient

```ts
computeHemisphereAmbient(
  normalY: number,
): LightingVec3
```

Computes hemisphere ambient: ground/sky mixed over the normal's up
component with the mirrored `clamp(normal.y * 0.5 + 0.5)` weight.

Parameters:
- `normalY` - World-space normal y component (unclamped range).

Returns: Mixed ambient color, per channel, normalized [0, 1].

Example:

computeHemisphereAmbient(1);   // pure sky
computeHemisphereAmbient(-1);  // pure ground
computeHemisphereAmbient(0);   // ground/sky midpoint

### computeSunDiffuse

```ts
computeSunDiffuse(
  normal: LightingVec3,
): number
```

Computes the directional sun term: `max(0, dot(normal, sunDirection))`.

Parameters:
- `normal` - Unit surface normal in the same space as the sun direction.

Returns: Clamped diffuse factor in [0, 1].

Example:

computeSunDiffuse(NEATRACING_SUN_DIRECTION); // 1
computeSunDiffuse([-1, 0, 0]);               // 0

### LightingVec3

RGB triple normalized to [0, 1]; also used for directions and normals.

### NEATRACING_FOG_COLOR

Frozen normalized exp2 fog tint.

### NEATRACING_FOG_DENSITY_1_PER_M

Fog extinction coefficient in 1/m (fogFactor = exp2(-(density * d)^2)).

### NEATRACING_FOG_HALF_TRANSMITTANCE_DISTANCE_M

Distance in meters at which the exp2 fog transmittance is exactly 0.5.

### NEATRACING_HEMISPHERE_GROUND_COLOR

Frozen normalized hemisphere ground tint (down-facing ambient).

### NEATRACING_HEMISPHERE_SKY_COLOR

Frozen normalized hemisphere sky tint (up-facing ambient).

### NEATRACING_LIGHTING_GLSL_CHUNK

GLSL chunk for lit passes: the lighting uniforms from
{@link NEATRACING_LIGHTING_UNIFORM_NAMES} plus the mirrored shade
functions. Embedding this chunk in a fragment shader is the seam that
guarantees the GL and TypeScript lighting models cannot drift.

### NEATRACING_LIGHTING_UNIFORM_NAMES

Shared lighting uniform names resolved by every lit pass.

`uCameraWorld` enables view-dependent shading inputs (spec 7.3.1 view-flip
uses the reconstructable view direction; the fog distance term uses the
world-fragment distance to the camera).

### NEATRACING_SUN_COLOR

Frozen normalized sun tint.

### NEATRACING_SUN_DIRECTION

Frozen, normalized sun direction (unit length, y-dominant).

### shadeLitSurface

```ts
shadeLitSurface(
  albedo: LightingVec3,
  normal: LightingVec3,
  emissive: number,
): LightingVec3
```

Full fragment shade mirror: `albedo * (hemisphere + sunColor * sunDiffuse)
+ albedo * emissive`. The emissive term is glow/255 resolved at vertex
build time, with LD-23's black-entry exception already folded in by
`buildVoxelMaterialTable` (black palette entries always emit 0).

Parameters:
- `albedo` - Vertex color normalized to [0, 1].
- `normal` - Unit surface normal.
- `emissive` - Additive emissive scalar in [0, 1] (glow byte / 255).

Returns: Final lit color normalized [0, 1] (may exceed 1 before the
shader's output clamp, mirroring GL behavior).

Example:

shadeLitSurface([0.5, 0.5, 0.5], [0, 1, 0], 0); // plain shaded white plate

## engine/context.service.ts

WebGL2 context service for the Neat Racing voxel racer engine (slice
P1-S1-C).

The module owns three responsibilities behind one small seam so later
engine slices can rely on a stable contract:

1. Context acquisition (AC-002): acquire the `webgl2` context exactly once
   per canvas — WeakMap memoization keyed by the canvas element — and
   degrade gracefully, with a human-readable fallback notice element,
   instead of throwing when WebGL2 is unavailable.
2. Context-loss handling (G5): `preventDefault()` is called on
   `webglcontextlost` so the browser keeps the context restorable, a lost
   flag lets the render loop suppress rendering while the context is gone,
   and an `onContextRestored` hook is where callers rebuild GPU resources.
3. Shader warmup sweep (AC-040, P8): every registered program is compiled,
   linked, and drawn exactly once against a 1×1 offscreen target before the
   first visible frame, eliminating first-draw shader-compilation hitches.
   The registry is pinned by {@link NEATRACING_WARMED_PROGRAM_COUNT}.

### ContextLostState

Mutable lost-flag shared between the loss listeners and the predicate.

### createFallbackResult

```ts
createFallbackResult(
  documentRef: Pick<Document, "createElement"> | undefined,
  isContextLost: () => boolean,
): RenderingContextInitResult
```

Build the graceful-fallback result used when WebGL2 is unavailable (AC-002).

Parameters:
- `documentRef` - Document factory for the notice element; when omitted
(headless environments) the result carries `null` for `fallbackElement`.
- `isContextLost` - Predicate bound to the shared lost-flag; always
reports false because no context exists to lose.

Returns: Result with `ok: false`, the fallback message, and the notice.

### createLiveResult

```ts
createLiveResult(
  canvas: HTMLCanvasElement,
  gl: WebGL2RenderingContext,
  onContextRestored: (() => void) | undefined,
  lostState: ContextLostState,
  isContextLost: () => boolean,
): RenderingContextInitResult
```

Register the G5 context-loss listeners and build the live-context result.

Parameters:
- `canvas` - Canvas the loss/restore events fire on.
- `gl` - Acquired WebGL2 context carried on the result.
- `onContextRestored` - Caller hook that reinitializes GPU resources.
- `lostState` - Shared mutable lost-flag toggled by the listeners.
- `isContextLost` - Predicate bound to the shared lost-flag.

Returns: Result with `ok: true` and the acquired context.

### initRenderingContext

```ts
initRenderingContext(
  options: InitRenderingContextOptions,
): RenderingContextInitResult
```

Acquire the WebGL2 rendering context for a canvas, exactly once per canvas.

The result is memoized in a WeakMap keyed by the canvas element, so repeated
boot calls never re-request the context. When WebGL2 is unavailable the
service degrades gracefully: the result carries the fallback message plus a
DOM notice element (when a `documentRef` is supplied) instead of throwing.

On success the service registers the G5 context-loss listeners:
`webglcontextlost` calls `preventDefault()` and sets the lost flag (which
{@link RenderingContextInitResult.isContextLost} reports), and
`webglcontextrestored` clears the flag and fires the `onContextRestored`
hook where callers rebuild GPU resources.

Parameters:
- `options` - Canvas plus optional document factory and restore hook.

Returns: The memoized acquisition result; inspect `ok` before using `gl`.

Example:

```ts
const init = initRenderingContext({ canvas, documentRef: document });
if (init.ok && init.gl) {
  const warmup = runShaderWarmup(init.gl, buildWarmupPrograms());
  console.log(`warmed ${warmup.warmedProgramCount} programs`);
}
```

### InitRenderingContextOptions

Options for {@link initRenderingContext}.

### NEATRACING_FALLBACK_MESSAGE

Message surfaced to players when WebGL2 is unavailable (AC-002).

Used both as the init result's `message` and as the text content of the
fallback notice element, so the DOM notice and the programmatic result can
never drift apart.

### NEATRACING_WARMED_PROGRAM_COUNT

Number of shader programs warmed by the pre-first-frame sweep (AC-040).

Derived from {@link NEATRACING_WARMED_PROGRAM_KEYS} so the count and the
registry can never disagree; the spec pins the value to the 7–9 range.

### NEATRACING_WARMED_PROGRAM_KEYS

Registry keys for every shader program warmed before the first frame.

Six world passes (starfield, grid, pulse, track, car, glass) plus the three
bloom stages from G8 (threshold, separable blur, additive composite) give
the pinned nine-key sweep. Later engine slices register their programs under
exactly these keys, so the warmup sweep and the pass registry cannot drift
apart (AC-040).

### prepareWarmupRenderTarget

```ts
prepareWarmupRenderTarget(
  gl: WebGL2RenderingContext,
): void
```

Configure the invisible offscreen render target used by the warmup sweep.

Parameters:
- `gl` - Live WebGL2 context to configure before the sweep draws.

### RenderingContextInitResult

Outcome of the WebGL2 context acquisition attempt (AC-002).

### runShaderWarmup

```ts
runShaderWarmup(
  gl: WebGL2RenderingContext,
  programs: readonly ShaderWarmupProgram[],
): ShaderWarmupResult
```

Run the shader warmup sweep against a 1×1 offscreen target (AC-040, P8).

Every registered program is compiled, linked, and drawn exactly once so the
driver finishes pipeline compilation before the first visible frame; the
offscreen target keeps warmup pixels invisible to players. Call this right
after {@link initRenderingContext} succeeds and again from the
`onContextRestored` hook so restored contexts re-warm their pipelines.

Parameters:
- `gl` - Live WebGL2 context to warm.
- `programs` - Programs registered for the sweep, keyed per
 *    {@link NEATRACING_WARMED_PROGRAM_KEYS} .

Returns: Warmup counters: programs warmed and offscreen draws issued.

Example:

```ts
const warmup = runShaderWarmup(gl, registeredPrograms);
console.log(warmup.warmedProgramCount === NEATRACING_WARMED_PROGRAM_COUNT);
```

### ShaderWarmupProgram

A single GPU program registered for the warmup sweep (AC-040).

### ShaderWarmupResult

Counters returned by the shader warmup sweep.

## engine/adaptive-quality.ts

NeatRacing adaptive quality controller (slice P3-S1-A; spec sections 6.3,
13.3 AC-035, G8, TP-15 in `plans/neatracing-voxel-racer.spec.md`).

Quality-first degradation: when the frame budget is exceeded the
controller drops QUALITY in a fixed ladder — the G8 cheap-bloom
pipeline's explicit off-switch first (the `enabled === false` frame state
already consumed by `passes/bloom.pass.ts`), then the cheap-tier LOD
bias, then full-vs-sparse prop density — BEFORE any frame is dropped.
Dropping frames stays the host's last resort; this module never drops
frames.

The decision function is:

- DETERMINISTIC (Level 2 ordered deterministic, reproducibility-contracts
  language): the same frameMs sequence drives the same state trajectory
  on any runtime — EMA blending plus threshold comparisons only; no
  randomness, no wall-clock reads, no cross-tick float reordering.
- HYSTERESIS-BOUNDED: drops fire immediately below
  {@link NEATRACING_ADAPTIVE_FPS_DROP_FLOOR}; recovery requires smoothed
  fps at or above {@link NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING} for a
  named {@link NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES} consecutive
  frame run, and the mid-band between floor and ceiling holds quality
  while burning recovery credit, so boundary noise cannot thrash the
  ladder (R-03's boundary-hysteresis philosophy applied to quality).
- ALLOCATION-FREE per frame (P9): the state object is created once and
  mutated in place — the same in-place executor pattern as
  `host/render-loop.utils.ts` (`advanceFrameSteps`).

This module owns ORDER and hysteresis only; it renders nothing. The host
reads the resolved stage record from
{@link NEATRACING_ADAPTIVE_QUALITY_STAGES}`[state.stage]` and applies each
knob at composition time (bloom pass `frame.enabled`, LOD tier selection,
sparse prop subset). Threshold values are Phase-3 tunables per spec
TP-15: retuning is a named-constant change only.

### AdaptiveQualityStage

One quality stage in the fixed degradation ladder (AC-035).

Stage 0 is full quality; every later entry is the previous entry minus
exactly one knob, so the ladder encodes the drop ORDER and never skips a
knob or re-enables a dropped one while quality is still dropping.

### AdaptiveQualityState

Mutable per-frame controller state (render-loop executor pattern: created
once, mutated in place, zero per-frame allocation — P9).

### createInitialAdaptiveQualityState

```ts
createInitialAdaptiveQualityState(): AdaptiveQualityState
```

Creates the initial controller state: full quality (stage 0), empty
recovery streak, and the assume-target smoothing seed so startup frames
never drop quality before real telemetry exists.

Call once at render-loop boot; every later frame reuses this state object
in place.

Returns: The initial full-quality controller state.

Example:

const state = createInitialAdaptiveQualityState();
console.log(state.stage); // 0 (full quality)

### decideAdaptiveQualityTransition

```ts
decideAdaptiveQualityTransition(
  state: AdaptiveQualityState,
): number
```

Applies the hysteresis-bounded ladder decision to the state (helper below
the fold) and returns the resolved stage index.

- smoothed fps < drop floor: drop exactly ONE stage immediately (never a
  multi-stage jump, even under a severe breach) and reset recovery credit.
- smoothed fps >= recover ceiling: accumulate one good-frame credit; at
  the named streak, recover exactly ONE stage and restart the streak.
- mid-band: hold quality and burn credit (hysteresis gap).
- non-finite estimate (defensive): hold quality and burn credit.

Parameters:
- `state` - Controller state mutated in place.

Returns: The resolved stage index after the decision.

### NEATRACING_ADAPTIVE_FPS_DROP_FLOOR

Drop floor in smoothed fps (spec TP-15): the first quality knob drops as
soon as smoothed fps falls below this floor. Chosen ~9% below the 60 fps
target so quality drops happen before visible stutter becomes the norm.

### NEATRACING_ADAPTIVE_FPS_EMA_ALPHA

Exponential-moving-average weight given to each new fps sample
(`smoothed += alpha * (sample - smoothed)`). With 0.1 a sustained breach
converges past the drop floor within a handful of frames (~60 ms) while
isolated hitches are absorbed, keeping decisions stable.

### NEATRACING_ADAPTIVE_FPS_RECOVER_CEILING

Recovery ceiling in smoothed fps (spec TP-15): at or above this value a
frame counts toward the sustained recovery streak. Kept strictly below
the drop floor's distance from the target so the hold band
`[DROP_FLOOR, RECOVER_CEILING)` is non-empty — that band IS the
hysteresis gap (no drop, no recover, no boundary thrash).

### NEATRACING_ADAPTIVE_FPS_TARGET

Smoothed-fps estimate seed and tuning reference (spec 6.3, TP-15).

The controller assumes the 60 fps target until smoothed telemetry says
otherwise, so a healthy startup never drops quality before real evidence.

### NEATRACING_ADAPTIVE_QUALITY_MAX_STAGE_INDEX

Highest valid stage index (`stage ladder length - 1`), derived from the
ladder so the clamp target and the table can never disagree.

### NEATRACING_ADAPTIVE_QUALITY_STAGES

The fixed quality-first degradation ladder in drop order
(AC-035): stage 0 full quality -> 1 bloom off (G8 explicit off-switch)
-> 2 cheap LOD tier -> 3 sparse props. Frozen in place, read-only at
runtime.

### NEATRACING_ADAPTIVE_RECOVER_STREAK_FRAMES

Consecutive at-or-above-ceiling frames required to recover exactly ONE
quality stage (spec TP-15). 120 frames is 2 sustained seconds at 60 Hz:
recovery is always deliberately slower than a drop, and each recovered
stage restarts the streak, bounding how fast quality unwinds.

### NEATRACING_ADAPTIVE_SMOOTHED_FPS_CEILING

Upper clamp on the smoothed-fps estimate (finite-boundary guard, spec 6.3
AC-003-adjacent). An unclamped geometric EMA could asymptote toward a
non-finite value if a corrupted sub-millisecond delta ever reached the
fold; clamping keeps every stored estimate finite and the decision
comparisons bounded without affecting normal 0-1000 fps telemetry.

### tickAdaptiveQuality

```ts
tickAdaptiveQuality(
  state: AdaptiveQualityState,
  frameMs: number,
): number
```

Advances the controller for one frame and returns the resolved stage
index after the decision.

The orchestrator runs exactly two steps: the smoothing fold of
`frameMs` into the state's EMA estimate, then the hysteresis-bounded
stage decision over that estimate. The state is mutated in place and
the per-frame path performs no heap allocations (P9, executor pattern
of `host/render-loop.utils.ts`).

Determinism (Level 2, ordered deterministic): identical frameMs
sequences produce identical stage/streak/smoothed trajectories on any
runtime; the fold ignores non-finite and non-positive deltas instead of
emitting machine-dependent values.

Parameters:
- `state` - Controller state mutated in place.
- `frameMs` - Latest frame delta in milliseconds (clamped upstream by
the render loop's `NEATRACING_MAX_FRAME_DELTA_MS`).

Returns: The resolved stage index after this frame's decision (`0..3`).

Example:

const state = createInitialAdaptiveQualityState();
const stage = tickAdaptiveQuality(state, 20); // 50 fps sample
console.log(NEATRACING_ADAPTIVE_QUALITY_STAGES[stage].label);

### updateSmoothedFps

```ts
updateSmoothedFps(
  state: AdaptiveQualityState,
  frameMs: number,
): void
```

Folds one frame delta into the state's EMA fps estimate (helper below the
fold). Usable deltas produce a bounded update; unusable deltas (non-finite
or non-positive — first-frame zeros, corrupted timers) are ignored so the
estimate can never become NaN/Infinity and the decision input stays
finite and deterministic.

Parameters:
- `state` - Controller state mutated in place.
- `frameMs` - Latest frame delta in milliseconds.
