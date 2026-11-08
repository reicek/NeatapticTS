# host

NeatRacing host HUD overlay — plan slice P1-S7-E (plan
`neatracing-voxel-racer.plans.md`, phase 1, step 12), implementing spec
AC-024: the speed/lap/time DOM overlay into the reserved host output
container plus the mode-switch control surface (spec sections 4.2,
10.2, 11.2, 13.1). The cyan/magenta palette is recycled verbatim from
the Neatenstein host HUD (spec section 11.2 recycling inventory: "same
CSS tokens, cyan/magenta"), pinned here as `NEATRACING_HUD_COLOR_CYAN`
and `NEATRACING_HUD_COLOR_MAGENTA` (rgb() literal strings, matching the
Neatenstein `hud.constants.ts` precedent).

RED -> GREEN contract: `examples/neatRacing/host/mode-selector.test.ts`
(the header contract of this slice) pins the display string grammar —
speed = whole km/h (TP-01 physical anchor: 1 world unit = 1 meter, so
~89 units/s ≈ ~320 km/h arcade feel; never a decimal part), lap =
`Lap c/20` clamped at `NEATRACING_HUMAN_RACE_LAPS` (a completed race
never displays 21/20; the selector restart contract AC-033 is
pre-verified by `mode-selector.ts`), and time = `M:SS.d` (unpadded
minutes, zero-padded seconds, one tenths digit).

The TRACTION READOUT (AC-088; plan slice `P1-drift-presentation-hud`)
is a DISPLAY of the ONE published traction severity
(`PhysicsCarState.tractionSeverity`, published by the physics step):
the HUD SHOWS the number it is handed and NEVER re-derives it, so the
overlay, the car pass tires and the browser debug seam can never
disagree about how much grip is left. The severity reaches the read
model as an OPTIONAL field (a read model without it paints the idle
zero), so the sibling suites that build the classic speed/lap/time
model keep compiling and keep their legacy look.

The arcade reconciliation is deliberate and documented rather than
incidental: spec section 11.1 wants an arcade-speed feel while the
physics layer (spec section 5) is intentionally NOT scale-accurate
(TP-01). The HUD is a display-only read model — it never mutates race
state — so the km/h conversion constant lives here at the seam where
the decision is made, mirroring how `race-rules.ts` keeps its own
budget constants instead of importing physics timing.

Allocation policy: `createHud` is the only DOM allocator (three labels
created once inside one wrapper), and `update()` performs no per-frame
DOM/geometry allocation beyond the unavoidable display strings passed
to `textContent` (zero allocations is not technically achievable for
a text repaint; this is the minimal allocation form — one string per
label, no arrays/closures/new objects per call). Formatting never
touches the DOM: it is pure so the same numbers produce the same
strings on any tick, enabling deterministic replay in tests.

Time formatting note: tenths are computed via
`Math.round(seconds * TENTHS_PER_SECOND)` and then integer modulo
arithmetic (tenths per minute = 600). Floating-point seconds such as
3.04, 75.3, or 9.9999 do not accumulate a double drift error larger
than one rounding of the tenths (the double is 3 units of ulp at
~1e14 tenths, a range far beyond any race duration); negative seconds
are non-domain — race-rules guarantees `elapsedS >= 0` (zero starts,
60s checkpoint budget) and the formatters document that domain instead
of carrying an unreachable guard branch.

Determinism section: every executor here is pure — identical inputs
produce identical strings with no Math.random, no clock reads, no
timers. Not claimed (same-runtime rung only): cross-environment byte
equality (DOM layout differences across browsers do not affect text
content or the pinned style token strings).

## host/hud.ts

### createHud

```ts
createHud(
  outputId: string,
): NeatRacingHud
```

What: binds the HUD labels to a live `RaceRulesState` slice and
mounts them into the reserved host output container (spec section
10.2 folder layout; AC-024). This is the only place that reads
`document` for the HUD display (the selector module resolves its own
container the same way, Neatenstein precedent).

Why: the host shell owns the container id; the HUD never assumes a
particular host DOM beyond that one hook, keeping the module portable
between the dev-server shell and the jsdom test fixture.

When: called once during host boot (slice P1-S8 wires the render
loop's `update` call; AC-033's restart hook pre-verified by
`createRaceModeSelector`). Throws instead of returning a null HUD so
boot order bugs surface immediately instead of as silent blank HUDs.

Allocation contract: this function performs all HUD DOM allocation
for the page lifetime (one wrapper + four labels — speed, lap, time
and traction — each once); `update()` then only writes textContent
strings per frame.

Parameters:
- `outputId` - Reserved host DOM id; `neatracing-hud-output` in
the host shell and the slice fixture suite.

Returns: The HUD controller (labels + stateless update binding).

Example:

```ts
const hud = createHud('neatracing-hud-output');
hud.update({
    speedUnitsPerS: 20,
    raceState: createRaceRulesState(),
});
// speedLabel.textContent === '72 km/h', lapLabel === 'Lap 1/20',
// timeLabel === '0:00.0', traction readout === 'Traction 0%'
```

### createHudLabel

```ts
createHudLabel(
  wrapper: HTMLElement,
  color: string,
): HTMLElement
```

What: builds one HUD readout label inside the wrapper with a pinned
accent color (speed = cyan, lap = magenta, time = cyan — the labels as
primary/progress/primary readouts of the recycled palette). Labels
share the wrapper's typography (monospace 14px) so each label only
needs its own color.

Parameters:
- `wrapper` - The HUD wrapper the label is appended to.
- `color` - The accent token from the Neatenstein palette.

Returns: The created label element.

### createHudWrapper

```ts
createHudWrapper(
  container: HTMLElement,
): HTMLElement
```

What: builds the HUD wrapper element styled like the Neatenstein
death-feedback overlay (absolute top-left, z-index 10, translucent
dark bg, 1px cyan border, monospace) so both examples share one
readable arcade palette (spec section 11.2).

Parameters:
- `container` - The host container to mount the wrapper into.

Returns: The styled, appended wrapper element.

### formatHudLap

```ts
formatHudLap(
  lapsCompleted: number,
): string
```

What: formats the completed-laps counter `lapsCompleted` for the lap
label as the race-day ordinal `Lap c/20` (the lap currently being
raced), clamped so a finished race still reads `Lap 20/20`.

Why: `race-rules` counts completed crossings, so lapsCompleted === 0
during lap 1; the human-facing label is the current lap number. The
clamp uses the shared `NEATRACING_HUMAN_RACE_LAPS` constant (single
source of truth with the rules module, spec section 5.4); without it a
terminal out-of-budget tick could paint `21/20` for one frame.

When: called by `NeatRacingHud.update` per frame and by tests.

Parameters:
- `lapsCompleted` - Number of lap lines fully crossed (race-rules
domain: integer >= 0, clamped at NEATRACING_HUMAN_RACE_LAPS).

Returns: The lap label, e.g. `'Lap 1/20'` for 0 completed laps.

Example:

```ts
formatHudLap(0); // 'Lap 1/20'
formatHudLap(1); // 'Lap 2/20'
formatHudLap(20); // 'Lap 20/20' (clamped, never 21/20)
```

### formatHudSpeed

```ts
formatHudSpeed(
  speedUnitsPerS: number,
): string
```

What: formats forward speed for the speed label, converting world
units-per-second into whole km/h (TP-01: 1 unit = 1 meter, so the
conversion is a fixed 3.6 km/h per unit/s).

Why: the arcade physics layer is deliberately not scale-accurate
(spec section 11.1) but the HUD readout is pinned to a physical anchor
in the slice contract so the displayed number is stable across
refactors of the physics constants; keeping the conversion here means
callers pass the raw sim speed without pre-scaling.

When: called by `NeatRacingHud.update` per frame; also used directly
by tests and debug overlays.

Format contract (slice header): whole number, no decimal part — a
half-unit bump is an imperceptible 1.8 km/h jump against a 320 km/h
readout, so rounding loses no arcade readability.

Parameters:
- `speedUnitsPerS` - Forward speed in world-units-per-second
(TP-01 meters/s); expected in [0, inf).

Returns: The speed formatted as whole `km/h`, e.g. `'320'` for 89.

Example:

```ts
formatHudSpeed(89); // '320 km/h'
formatHudSpeed(0); // '0 km/h'
formatHudSpeed(20); // '72 km/h'
```

### formatHudTime

```ts
formatHudTime(
  seconds: number,
): string
```

What: formats race elapsed seconds as `M:SS.d` — unpadded minutes,
zero-padded 2-digit seconds, one tenths digit — for the time label.

Why: tenths are resolved first via `Math.round(seconds *
TENTHS_PER_SECOND)` and then by integer arithmetic (minutes =
floor of tenths / TENTHS_PER_MINUTE; remainder split into seconds and
tenths), which is immune to the classic double pitfall of decomposing
fractional seconds directly — e.g. 75.3s stored as
75.299999999999997... would floor to the wrong `:15.2` but rounds to
the contract-pinned `1:15.3`, and 3.04s pins to `0:03.0` not `0:03.1`
even though 3.04 * TENTHS_PER_SECOND lands on 30.4.

When: called by `NeatRacingHud.update` per frame and by tests.

Parameters:
- `seconds` - Elapsed race time in seconds (race-rules domain:
non-negative; zero starts, checkpoint budget keeps it at or above 0).

Returns: The race clock, e.g. `'1:15.3'` for 75.3.

Example:

```ts
formatHudTime(0); // '0:00.0'
formatHudTime(75.3); // '1:15.3'
formatHudTime(3.04); // '0:03.0'
formatHudTime(60); // '1:00.0'
```

### formatHudTraction

```ts
formatHudTraction(
  tractionSeverity: number | undefined,
): string
```

What: formats the PUBLISHED traction severity for the traction
readout as a whole percent (e.g. `'Traction 42%'`).

Why: the severity is the ONE shared scalar the physics step publishes
(AC-088), so the HUD only CONVERTS it for display — it never
re-derives it from the speed, the slip or any other HUD input, and
the overlay therefore cannot disagree with the car pass tires. A
missing severity (the OPTIONAL read-model field) is the IDLE zero:
the readout is always on screen, so the player never has to guess
whether the overlay lost the value.

When: called by `NeatRacingHud.update` per frame and by the boot
paint.

Parameters:
- `tractionSeverity` - The published severity in [0, 1];
`undefined` when the read model carries no severity (the idle zero).

Returns: The traction readout, e.g. `'Traction 42%'` for 0.42.

Example:

```ts
formatHudTraction(0.42); // 'Traction 42%'
formatHudTraction(0); // 'Traction 0%'
formatHudTraction(undefined); // 'Traction 0%' (the idle zero)
```

### NEATRACING_HUD_COLOR_CYAN

Cyan token used for the speed label, time label, and selector border.

### NEATRACING_HUD_COLOR_MAGENTA

Magenta token used for the lap label (the second accent of the
Neatenstein palette kept as a distinct token so future HUD elements
can share one import source).

### NeatRacingHud

The constructed HUD controller handed to the host shell.

### NeatRacingHudState

The race state slice the HUD reads from. Mirroring `input.ts`, this is
a named input contract instead of the whole `RaceRulesState` so the
render loop can pass exactly what the HUD needs.

### resolveOutputContainer

```ts
resolveOutputContainer(
  outputId: string,
  hudName: string,
): HTMLElement
```

What: resolves the reserved host output container by id, matching the
Neatenstein `resolveHudContainer` precedent (`hud.dom.utils.ts`)
flattened into this slice per the P1-S7-E file contract (no extra
shared files inside the slice boundary).

Parameters:
- `outputId` - Reserved host DOM id (spec section 10.2 folder
layout; `neatracing-hud-output` in the host shell and fixtures).
- `hudName` - Human-readable surface name used in the banner text.

Returns: The DOM container to append the HUD overlay into.

Example:

```ts
resolveOutputContainer('neatracing-hud-output', HUD_NAME);
```

## host/input.ts

Keyboard input map for the neatRacing example (plan slice P1-S7-C;
AC-024 partial, spec section 5.1 decision D5).

D5 sampling rule: keyboard state is sampled ONCE per rAF frame by the
host into a transport-neutral pressed-key map
({@link NeatRacingKeyboardPresses}); this module maps that sample onto
the analog axes the physics step consumes, and the host replays the
SAME sample object unchanged into EVERY catch-up tick of that frame.
Key-to-press binding is deliberately the host's DOM concern (key
codes, focus, preventDefault), so this mapper holds no browser state
and reads no DOM: it is a pure boolean-to-axis map.

The output type is the `PhysicsStepInput` axes shape itself, so a
sampled keyboard sample replays into `stepPhysics` verbatim with no
adapter step that could re-order or re-quantize the axes.

Determinism: `sampleKeyboardInput` is pure — identical pressed-key
maps always produce byte-identical samples (`JSON.stringify`-equal)
with no runtime-random state.

### NeatRacingKeyboardPresses

The once-per-rAF-frame pressed-key map the host feeds to
{@link sampleKeyboardInput} (D5). Transport neutral: the map carries
plain boolean pressed state; how `up`/`down`/`left`/`right` bind to
physical key codes is the host's DOM concern, not the sim's.

### NeatRacingKeyboardSample

One-keyboard-frame analog axes sample (D5): exactly the
`PhysicsStepInput` shape, so the host can apply the same sample
object unchanged to every catch-up tick of the frame it was sampled
in, and fresh identical samples per tick always land byte-identical.

### sampleKeyboardInput

```ts
sampleKeyboardInput(
  presses: NeatRacingKeyboardPresses,
): PhysicsStepInput
```

Maps one once-per-rAF-frame pressed-key map onto the analog axes
sample (`spec section 5.1` D5).

The up key maps to full positive throttle, the down key to full
positive brake, the left key to steer +1 (counterclockwise per the
steer axis contract), and the right key to steer -1. A neutral map
yields three zero axes. Pressing left and right together resolves to
a zero steer axis (the symmetric analog difference of the two
directions); pressing up and down together keeps both axes at full
ratio — brake precedence is enforced by the physics tick itself (the
`PhysicsStepInput` contract: "a positive brake wins over throttle"),
not by remapping keys here.

Parameters:
- `presses` - Host-collected pressed-key map sampled once for this
rAF frame.

Returns: The axes sample applied, unchanged, to every catch-up tick
of that frame before the next per-frame sample is taken (D5).

Example:

```ts
const sample = sampleKeyboardInput({
  up: true, down: false, left: true, right: false,
});
console.log(sample.throttle, sample.steer); // 1, 1 — forward and left
```

## host/mode-selector.ts

NeatRacing host race-mode selector — plan slice P1-S7-E (plan
`neatracing-voxel-racer.plans.md`, phase 1, step 12), implementing the
AC-024 selector overlay ("top-right; ai, manual options; initial
manual mode"; spec sections 4.2, 10.2, 11.2, 13.1) and serving the
AC-033 restart-on-switch surface: `onSwitch` is the single hook the
host shell connects to a race restart, so the "toggle the race mode"
step is delivered as a notify contract here rather than as an owned
restart call (spec section 4.2 keeps mode authority in the shell, not
the selector).

Phase-2 update (P2-S4-A, AC-033): the AC-033 mode half is now served
with the Phase-2 AI goal lifecycle (`../ai/ai-mode.goals.ts`): AI mode
runs a minimum 20-lap session and then continues until the current
level's promotion-threshold-driven growth goal is met, and level 3
continues until the user stops the session.

Structural precedent (cited in the slice contract header):
`examples/neatenstein/browser-entry/host/hud.human-mode.utils.ts` —
resolve the reserved container or throw; build the `<select>` once and
style it from named constants; render both options with ai FIRST;
keep the selected mode in a closure guarded by an early same-value
exit; notify subscribers exactly once per actual switch (both for a
user `change` event and a programmatic `setMode`); never rebuild DOM
on mode change (allocation policy: only allocation is mount-time).

Boot mode (Step 14P / AC-087, the L8 mask, re-pointing the batch-108
user-gap patch): the overlay still exposes both 'ai' and 'manual'
options, but the INITIAL mode is 'manual'. WHY: the served page IS
the acceptance surface (the user judges the drift on the page that
is actually served) and the ai-mode page is never acceptable
evidence — the AI autopilot crosses the 0.5 engage cliff on only 176
of 7200 ticks (2.44 percent), so an AI boot masks the very slide
the user has to judge. The 'ai' option stays SELECTABLE through the
same switch, which still notifies its listeners and fires the
AC-033 restart hook. Same-value switches are documented no-ops (and
covered
by the slice contract's jest non-call assertion) rather than silent
restarts: the host shell must not reset a race because a user
re-picked the already-active mode. A mode switch into AI mode makes
the HOST shell (the P2-S4-C wiring) boot a fresh AI goal session at
the first ladder rung (`NEATRACING_AI_MODE_START_LEVEL`,
`../ai/ai-mode.goals.ts`); the selector itself keeps notifying exactly
once per actual switch and owns no race state.

Cyan/magenta discipline (spec section 11.2 "same CSS tokens,
cyan/magenta"): the border recycles `NEATRACING_HUD_COLOR_CYAN`
imported from './hud' (single source of truth for the tokens; the
magenta token stays owned by hud.ts since the selector only uses the
cyan border in this slice).

Determinism section: every executor here is pure or mount-only —
identical inputs produce identical DOM/notify behavior with no
Math.random, no clock reads, no timers. Not claimed (same-runtime
rung only): cross-environment byte equality (jsdom vs browsers agree
on the style strings this module assigns, not on layout metrics).

### createModeSelect

```ts
createModeSelect(): HTMLSelectElement
```

What: builds the `<select>` with both race-mode options in the pinned
order (`ai` first) and the Neatenstein top-right overlay styling from
named constants (spec section 11.2 recycling inventory), without
attaching it to the document yet.

Why: option order and initial value are part of the AC-024 acceptance
contract and the DOM is built exactly once at mount (allocation
policy: no per-switch DOM construction).

Returns: The styled, unattached `<select>` element.

### createRaceModeSelector

```ts
createRaceModeSelector(
  outputId: string,
): NeatRacingRaceModeSelector
```

What: creates the race-mode selector overlay inside the reserved host
output container with the Neatenstein `hud.human-mode.utils.ts`
structure: a single `<select>` (ai, manual — initial 'manual'), a
closure-held current mode, an `onSwitch` subscription list, and one
`change` listener plus one `setMode` entry point that both notify
subscribers exactly once per actual switch and never notify on
same-value switches.

Why (AC-033, Phase 2): the host connects `onSwitch` to a race
restart — the restart runs in BOTH directions (LD-12) and the host
shell (the P2-S4-C wiring) handles it, including the boot of the
fresh AI goal session at the first ladder rung
(`../ai/ai-mode.goals.ts`); the hook is delivered and verified here so
the restart wiring cannot be blocked by a missing selector surface.
Step 14P opens the race in the 'manual' mode (the served
page is the acceptance surface and the ai-mode page is never
acceptable evidence — see `INITIAL_RACE_MODE`); the boot glue never
calls `setMode`; the 'ai' option remains
rendered
and selectable to keep the toggle contract fully exercisable (same
CSS tokens, cyan/magenta; option order `ai, manual` from AC-024).

When: called once during host boot after the HUD mounts into the same
reserved container id (`neatracing-hud-output`). Throws instead of
returning null so boot-order bugs surface immediately.

Parameters:
- `outputId` - Reserved host DOM id; `neatracing-hud-output` in
the host shell and the slice fixture suite.

Returns: The selector controller (select element + mode API).

Example:

```ts
const selector = createRaceModeSelector('neatracing-hud-output');
selector.mode; // 'manual' (the boot default)
selector.setMode('ai'); // notifies onSwitch listeners once with 'ai'
selector.setMode('ai'); // same-value: documented no-op, not notified
selector.onSwitch((mode) => restartRace(mode));
```

### INITIAL_RACE_MODE

Initial race mode: the game page boots in MANUAL mode.

WHY (Step 14P / AC-087, the L8 mask): the served page is the
acceptance surface — the user judges the drift on the page that is
actually served — and the ai-mode page is never acceptable evidence.
The AI autopilot only crosses the 0.5 engage cliff on 176 of 7200
ticks (2.44 percent), so an AI boot masks the very slide the user
has to judge. This is a BOOT DEFAULT and nothing else: the 'ai'
option stays selectable through the mode selector and a switch still
notifies its listeners (AC-033).

### NeatRacingRaceMode

The two race modes supported by the selector overlay.

### NeatRacingRaceModeListener

```ts
NeatRacingRaceModeListener(
  mode: NeatRacingRaceMode,
): void
```

Race-mode change subscriber shape.

### NeatRacingRaceModeSelector

The constructed race-mode selector handed to the host shell.

### resolveSelectorContainer

```ts
resolveSelectorContainer(
  outputId: string,
  surfaceName: string,
): HTMLElement
```

What: resolves the reserved host output container by id, matching the
Neatenstein `resolveHudContainer` precedent (`hud.dom.utils.ts`)
flattened into this slice per the P1-S7-E file contract (no extra
shared files inside the slice boundary; hud.ts keeps its own copy of
this eight-line resolver).

Parameters:
- `outputId` - Reserved host DOM id (spec section 10.2 folder
layout; `neatracing-hud-output` in the host shell and fixtures).
- `surfaceName` - Human-readable surface name for the error body.

Returns: The DOM container to append the selector into.

Example:

```ts
resolveSelectorContainer('neatracing-hud-output', SELECTOR_NAME);
```

## host/frame-composer.ts

### bindSceneFrameBufferAndViewport

```ts
bindSceneFrameBufferAndViewport(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
): void
```

Ensures the scene FBO and its full-size viewport are currently bound.
Always rebinds both: framebuffer binds are cheap, and this guarantees a
previous pass that changed either target or viewport cannot silently blank
the next pass.

Parameters:
- `gl` - WebGL2 context.
- `scene` - The owned scene render target.

### composeBloomComposite

```ts
composeBloomComposite(
  gl: WebGL2RenderingContext,
  bloom: BloomPass,
  bloomFrame: DrawStateBag<BloomPassDrawFrame>,
): void
```

Composites the bloom stage onto the default framebuffer using the owned
scene texture as the transfer carrier (single scene transfer, batch 54).
The default framebuffer is cleared to transparent black before the
additive composite so the `blendFunc(GL_ONE, GL_ONE)` destination is
deterministic per frame — without it the presented result depends on
whatever the browser drawing buffer holds after the previous present
instead of being a pure `scene + bloom * intensity` transfer.

Parameters:
- `gl` - WebGL2 context.
- `bloom` - The injected bloom pass handle.
- `bloomFrame` - The refreshed bloom frame bag.

### countNonTransparentBlackPixels

```ts
countNonTransparentBlackPixels(
  pixels: Uint8Array<ArrayBufferLike>,
): number
```

Reads back a small deterministic sample from the scene FBO and throws
a labeled composer error if every sampled pixel is transparent black.

This is the dev-only tripwire that turns a silent blank-composite
failure into a loud, searchable exception on the first composed frame.
It intentionally samples the center of the scene target because the
surrounding letterbox area may legitimately stay black while the game
content is centered.

Parameters:
- `gl` - WebGL2 context.
- `scene` - The owned scene render target.
- `label` - Error label prefix.

### createDrawStateBags

```ts
createDrawStateBags(
  gl: WebGL2RenderingContext,
  sceneTexture: WebGLTexture,
): NeatRacingFrameComposerDrawStateBags
```

Creates the identity-stable draw-state bags once at construction. Every
bag view persists for the composer lifetime; the bloom bag starts with
the owned scene texture and neutral refresh targets.

Parameters:
- `gl` - WebGL2 context (backing-store dims seed the bloom bag).
- `sceneTexture` - The owned scene texture of the initial bloom bag.

Returns: The stable bag bundle.

### createFrameComposer

```ts
createFrameComposer(
  deps: NeatRacingFrameComposerDeps,
): NeatRacingFrameComposer
```

Creates the frame composer: the host-side carrier that owns one scene
FBO + texture, draws the world passes into it once per tick in the
pinned warmup-key order, then composites bloom onto the default
framebuffer with the owned scene texture.

Parameters:
- `deps` - One WebGL2 context plus the seven required pass handles; the
composer never owns (and never disposes) the injected passes.

Returns: The composer handle with `drawFrame` and `dispose`.

Example:

```ts
const composer = createFrameComposer({ gl, passes });
composer.drawFrame({ viewProjection, cameraWorld, cameraPose, pulses, cars, enabled: true });
composer.dispose();
```

### createSceneRenderTarget

```ts
createSceneRenderTarget(
  gl: WebGL2RenderingContext,
): NeatRacingFrameComposerSceneTarget
```

Creates the owned scene render target at construction: one RGBA8 texture
sized to the canvas backing store, attached to the FBO which is left
bound (the carrier state; no construction-time default-FB bind). The FBO
completeness is validated with `checkFramebufferStatus` (bloom-parity
loudness: a broken attachment can never degrade silently into a blank
composite) and the canvas-backed viewport dims are captured for the
per-frame scene viewport (the warmup sweep leaves the global 1x1
viewport behind, and no world pass sets a viewport itself).

Parameters:
- `gl` - WebGL2 context.

Returns: The texture + framebuffer + viewport-dims bundle.

### DrawStateBag

Converts one pass-facing draw state into its identity-stable local bag
mirror: identical member types with mutable members, so a per-tick
refresh writes values in place while every `draw()` call still receives
a structurally exact draw state (AC-042).

### logPerPassDiagnosticsIfEnabled

```ts
logPerPassDiagnosticsIfEnabled(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passName: string,
  readbackBuffer: Uint8Array<ArrayBufferLike> | null,
): void
```

Temporary deep diagnostic: after each world pass, logs the bound
framebuffer, viewport, GL error, and the non-transparent pixel count across
a 16x16 sample grid covering the whole scene target.

### logPerPassPixelCount

```ts
logPerPassPixelCount(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passName: string,
  readbackBuffer: Uint8Array<ArrayBufferLike>,
): void
```

Reads back a small deterministic sample from the scene FBO and logs the
number of non-transparent-black pixels for one world pass. This is the
per-pass diagnostic used to locate which pass fails to contribute pixels
after a camera-matrix or pass-state fix.

Parameters:
- `gl` - WebGL2 context.
- `scene` - The owned scene render target.
- `passName` - World pass name printed in the console log.
- `readbackBuffer` - Reused `Uint8Array` buffer allocated at construction.

### logPerPassPixelCountIfEnabled

```ts
logPerPassPixelCountIfEnabled(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passName: string,
  readbackBuffer: Uint8Array<ArrayBufferLike> | null,
): void
```

Helper that conditionally logs the per-pass pixel count when the composer
was constructed with `assertPerPassPixelCounts` enabled. The buffer is
reused, so this adds no per-frame allocation on the steady-state hot path.

Parameters:
- `gl` - WebGL2 context.
- `scene` - The owned scene render target.
- `passName` - World pass name printed in the console log.
- `readbackBuffer` - Reused `Uint8Array` buffer, or null when disabled.

### NeatRacingFrameComposer

Handle created by {@link createFrameComposer}.

### NeatRacingFrameComposerDeps

Dependencies of the frame composer: one GL context plus the pass handles.

### NeatRacingFrameComposerDevFlags

Dev-only assertion surface for the frame composer (AC-035 off-switch parity).

### NeatRacingFrameComposerDrawStateBags

Identity-stable draw-state bags created once at construction (AC-042).
Bag identity persists across ticks; only member values are refreshed.

### NeatRacingFrameComposerFrame

One composed frame handed to the composer's draw seam.

### NeatRacingFrameComposerPasses

The pass handles the composer draws once per tick, in pinned order.

### NeatRacingFrameComposerSceneTarget

Composer-owned scene render target: the single-scene-transfer carrier
resources (one RGBA8 texture, one depth renderbuffer, plus their attach
FBO). The captured width/height own the per-frame viewport (the warmup
sweep leaves the global 1x1 viewport behind, and no world pass sets a
viewport itself).

### NeatRacingTrackPassDrawState

Draw state forwarded to the not-yet-extracted track pass seam.

### NeatRacingTrackPassSeam

Minimal draw-only structural seam for the not-yet-extracted track pass
(track/track.slabs.ts exposes only `buildTrackSlabs`, so importing a
track module here would break the module graph). P1-drive-loop-live
replaces it with the real pass module under this same shape.

### PulseDrawStateBag

Pulse draw-state bag: the pulse program samples the matrix from plain
number storage (`readonly number[]` seam), so its matrix buffer is a
plain 16-float array written by a per-tick loop copy.

### refreshDrawStateBags

```ts
refreshDrawStateBags(
  bags: NeatRacingFrameComposerDrawStateBags,
  gl: WebGL2RenderingContext,
  frame: NeatRacingFrameComposerFrame,
): void
```

Refreshes the identity-stable bags in place from one frame (pure member
writes; no object allocation, AC-042).

Parameters:
- `bags` - The stable bag bundle to refresh.
- `gl` - WebGL2 context (backing-store dims for the bloom bag).
- `frame` - The composed frame payload for this tick.

### renderWorldIntoSceneTarget

```ts
renderWorldIntoSceneTarget(
  gl: WebGL2RenderingContext,
  scene: NeatRacingFrameComposerSceneTarget,
  passes: NeatRacingFrameComposerPasses,
  bags: NeatRacingFrameComposerDrawStateBags,
  frame: NeatRacingFrameComposerFrame,
  perPassReadbackBuffer: Uint8Array<ArrayBufferLike> | null,
  perPassDiagnosticBuffer: Uint8Array<ArrayBufferLike> | null,
): void
```

Renders the world passes into the owned scene target: binds the scene
FBO, resets the per-frame viewport to the full scene extent (the
`runShaderWarmup` sweep leaves the global viewport at the shared 1x1
warmup target and `updateRendererSize` only resizes the canvas backing
store — no world pass sets a viewport, so the composer owns it per the
batch 54 host policy), establishes a transparent-black color baseline
(the post-present drawing-buffer clearance the un-composed path
depended on, now explicit), resets the depth mask so the depth clear is
never silently masked off by the previous frame's `depthMask(false)`
pass state, and draws every world pass exactly once in the pinned
warmup-key order (starfield → grid → pulse → track → car → glass).

Defensively re-binds the scene FBO and viewport immediately before each
world pass draw call so a pass that internally changes framebuffer state
cannot silently blank subsequent passes.

Parameters:
- `gl` - WebGL2 context.
- `scene` - The owned scene render target.
- `passes` - The injected pass handles.
- `bags` - The refreshed draw-state bags.
- `frame` - The composed frame payload (starfield camera pose seam).
- `perPassReadbackBuffer` - Reused `Uint8Array` buffer for the dev-only
per-pass pixel-count diagnostic, or null when disabled.

### restoreFrameFramebufferBaseline

```ts
restoreFrameFramebufferBaseline(
  gl: WebGL2RenderingContext,
): void
```

Restores the host-owned frame-level framebuffer baseline (default FB)
after every tick; the bind is idempotent by the batch 54 host policy.

Parameters:
- `gl` - WebGL2 context.

## host/render-loop.utils.ts

Host render-loop utilities for NeatRacing (slice P1-S1-D, plan
plans/neatracing-voxel-racer.plans.md, step 02, spec sections 5.1/6.3).

This module owns the deterministic part of the NeatRacing render loop:
frame-delta clamping, the fixed-timestep accumulator that releases
simulation steps at a strict 60 Hz cadence, the spiral-of-death guard, and
the preallocated matrix/vector buffers that later render passes fill in.
The browser-entry glue (rAF scheduling, WebGL2 calls, input sampling)
lives in sibling slices and consumes these executors.

Allocation policy (AC-042, render-loop share): every Float64Array is
allocated exactly once at state creation through the injectable allocator
seam. The per-frame executors only read/write plain numbers and the
preallocated buffers, so a steady-state render loop performs zero host
allocations per frame.

Floating-point note (AC-003): `Math.floor(accumulatedMs / timestep)`
loses exactly one tick per simulated second when frame timestamps carry
ulp-scale rounding drift (for example `i * (1000 / 144)` timestamps), so
the step boundary is relaxed by an ulp-scale epsilon. The epsilon only
flips boundaries that differ by ~1e-9 ms (a nanosecond), which preserves
the spec cadence exactly while keeping the accumulator ulp-robust.

### advanceFrameSteps

```ts
advanceFrameSteps(
  state: NeatRacingRenderLoopState,
  timestampMs: number,
): number
```

Advances the accumulator for one rAF frame and returns how many simulation
steps the loop must run this frame. Steps are released from the fixed
60 Hz cadence accumulator, capped at `NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME`
(spiral-of-death guard), and excess backlog above the catch-up ceiling is
dropped so the loop cannot fall behind permanently. Sub-timestep frames
release 0 steps (never a forced minimum of 1) and carry their remainder to
the next frame. The call mutates `state` in place and performs no heap
allocations (AC-042).

Parameters:
- `state` - Render-loop state mutated in place.
- `timestampMs` - Current rAF timestamp.

Returns: The number of simulation steps to run this frame (0 to 4).

Example:

const state = createNeatRacingRenderLoopState();
const stepCount = advanceFrameSteps(state, performance.now());
for (let stepIndex = 0; stepIndex < stepCount; stepIndex++) {
  // run one 1/60 s simulation tick
}

### computeDeltaMs

```ts
computeDeltaMs(
  lastTimestampMs: number | null,
  timestampMs: number,
  maxDeltaMs: number,
): number
```

Computes the clamped frame delta between the previous and current rAF
timestamps (recycled verbatim from the Neatenstein render-loop precedent).
The first frame (null previous timestamp) contributes a 0 ms delta so the
accumulator cannot pre-fill before real frames arrive.

Parameters:
- `lastTimestampMs` - Previous rAF timestamp, or null on the first frame.
- `timestampMs` - Current rAF timestamp.
- `maxDeltaMs` - Clamp bound for oversized deltas (tab switches, pauses).

Returns: The clamped non-negative delta in milliseconds.

Example:

computeDeltaMs(null, 16.6, 100); // 0   (first frame)
computeDeltaMs(0, 5000, 100);    // 100 (clamped tab switch)
computeDeltaMs(0, 16.6, 100);    // 16.6 (normal frame)

### createDefaultNeatRacingRenderLoopAllocator

```ts
createDefaultNeatRacingRenderLoopAllocator(): NeatRacingRenderLoopAllocator
```

Creates the default allocator that produces standard Float64Array buffers.

Returns: An allocator backed by plain Float64Array construction.

Example:

const allocator = createDefaultNeatRacingRenderLoopAllocator();
const state = createNeatRacingRenderLoopState(allocator);

### createNeatRacingRenderLoopState

```ts
createNeatRacingRenderLoopState(
  allocator: NeatRacingRenderLoopAllocator,
): NeatRacingRenderLoopState
```

Creates the render-loop state, allocating every matrix and vector exactly
once through the given allocator seam.

Parameters:
- `allocator` - Allocation seam; defaults to the standard allocator.

Returns: The initial accumulator state with preallocated buffers.

Example:

const state = createNeatRacingRenderLoopState();
const steps = advanceFrameSteps(state, timestampMs);

### NEATRACING_CATCHUP_CEILING_MS

Accumulator ceiling in milliseconds: the maximum catch-up backlog the
accumulator ever holds (`max catchup steps * fixed timestep`). Excess
simulated time is dropped instead of accumulating into a spiral.

### NEATRACING_FIXED_TIMESTEP_MS

Fixed simulation timestep in milliseconds: 60 simulation ticks per second (spec section 5.1).

### NEATRACING_FIXED_TIMESTEP_SECONDS

Fixed simulation timestep in seconds, the unit physics and simulation
code consume (spec section 5.1).

### NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME

Maximum number of simulation steps released in a single rAF frame. This is
the spiral-of-death guard: after a stall (tab switch, GC pause) the loop
plays at most 4 ticks per frame instead of freezing at 100% catch-up load
(spec section 6.3).

### NEATRACING_MAX_FRAME_DELTA_MS

Upper bound for a single clamped frame delta in milliseconds. Deltas above
this bound (tab switches, debugger pauses) are clamped so a stall cannot
pump a large backlog into the accumulator at once.

### NEATRACING_RENDER_LOOP_MATRIX_FLOATS

Float length of a column-major 4x4 matrix.

### NEATRACING_RENDER_LOOP_VECTOR_FLOATS

Float length of a 3-component vector.

### NEATRACING_STEP_BOUNDARY_EPSILON

Ulp-scale boundary epsilon in step units. It widens the floor boundary of
the step-count division so ulp-level floating-point drift in accumulated
deltas cannot swallow a step release that ideal frame streams are owed
(AC-003 exactness); it is far below any physical frame delta.

### NeatRacingRenderLoopAllocator

Allocation seam for the render-loop's preallocated buffers. Browser-entry
glue injects the default allocator; tests inject a counting double to
assert the zero-per-frame-allocation contract (AC-042).

### NeatRacingRenderLoopState

Mutable accumulator state shared by the render-loop executors. The matrix
and vector buffers are preallocated once and filled in-place by the later
projection/view/camera passes each frame.
