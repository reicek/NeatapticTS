# AUDIT-VISUALS - read-only code audit of the tire/traction PRESENTATION layer

**Artifact id**: `neatracing-voxel-racer.audit-visuals`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 5` record, `plans/neatracing-voxel-racer.plans.md` L10284-10286, `audit_waves[3]` = `14N-V5-audit-visuals`
**Subject**: `examples/neatRacing/engine/passes/car.pass.ts` (the car/tire draw pass) + `examples/neatRacing/host/hud.ts` (the HUD) + `examples/neatRacing/car/wheel.steer.ts` + `examples/neatRacing/car/generate-car.ts` + `examples/neatRacing/voxel/wheel.mount.utils.ts` + the presentation-facing half of `examples/neatRacing/browser-entry/browser-entry.ts` and `examples/neatRacing/physics/physics.step.service.ts`
**Constraints honoured**: READ-ONLY code audit · NO child agents (per the dispatch's `O_1` "NO child agents inside each") · NO browser runs · NO Chrome MCP · no production byte touched.

---

## 0. Provenance, method and the naming deviation

**Method.** Every claim below is a *static-code* read plus repo-wide readership greps. Line numbers are cited against the files as they stand on disk (2026-10-27). No runtime measurement was performed (the dispatch forbids browser runs; the `O_2` "Chrome MCP BANNED" order is non-negotiable). Numbers quoted as "derived" are arithmetic from source constants and are marked **DERIVED**.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. There is no runtime evidence in this audit, so every finding is `static-code` ranked. Where a source JSDoc disagrees with the code, the **code wins** and the doc drift is reported as a literal gap.

**Naming deviation (disclosed).** The contract names `plans/<PlanName>.research.md`. That exact path is **already occupied** by a different, closed 14N artifact (`plans/neatracing-voxel-racer.research.md`, dated 2026-10-24 - the Step 14N drift-feel *integration coupling* design), and `plans/neatracing-voxel-racer.drift-feel.research.md` is likewise occupied. Clobbering a closed, dated artifact would fuse two different questions under one date. This artifact therefore follows the repo's own topic-suffix family convention (`.14N-red-contracts.md`, `.14N-fixture-gate.md`, `.14N-rebaseline-risk.md`, `.drift-feel.research.md`, `.audit-physics.research.md`) as a **new sibling**: `plans/neatracing-voxel-racer.audit-visuals.research.md`. It remains a `plans/*.research.md` sibling of the tracker and is **not** under `docs/research/`. The orchestrator may rename or merge it without loss.

**Scope boundary.** This is the **PRESENTATION** half. The camera aim path (`browser-entry.ts` L935-937 / L1000-1012), the `cameraPose` trap, the wheel-spin *reset idiom*, the `stepPhysics` wrapper and the tau = 0.08 follow law are `AUDIT-CAMERA`'s scope. The traction law itself (`drift.model.ts` + the scrub in `physics.step.service.ts`) is `AUDIT-PHYSICS`' scope, and the *live input chain* is `14N-V5-audit-input`'s scope. This artifact adjudicates exactly one question: **what VISUALLY expresses losing traction, what feeds it, and which published drift outputs are unwired.** Where a claim crosses a boundary it is flagged as such.

**Non-duplication.** The settled `AUDIT-PHYSICS` facts (F1-F8, B1-B4, D1-D6, R1-R8) are **not** re-proved. The one physics-side fact this audit *must* re-state (the scrub is computed and discarded) is cited as a **cross-reference**, not as a new finding, and is marked as such in G-V2.

---

## 1. Question

The `USER VERDICT 5` dispatch asks:

> INVENTORY the tire/traction PRESENTATION layer: what VISUALLY expresses losing traction (tire yaw-follow, skid/gradient, smoke, speed loss HUD) and what feeds it; enumerate EVERY published-but-unwired drift output (scrub, `car.yawRate`, `pose.poseRate`, seam fields) so the implementation wave knows exactly which wires exist and which must be CREATED.

The user's own verdict supplies the reason the question matters:

> "there are no drifting physics in place, I can take turns at full speed and the car never drifts, it's only confusing due to the odd camera angle. The current implementation is either not wired or a complete failure!"

So the audit must answer three things:

1. **What does the player actually SEE** when the tires lose (or fail to lose) traction?
2. **What is the upstream source of each of those visuals** - is it the traction law, the committed slip, the raw input, or the speed?
3. **Which published drift outputs have no production reader at all**, and therefore which wires the implementation wave must **CREATE** rather than **CONNECT**.

---

## 2. Evidence

### 2.A - The presentation layer, in full  *(confidence 0.96, source: static-code)*

A repo-wide grep over `examples/neatRacing/**/*.ts` for `smoke|skid|trail|tireMark|particle` returns **ZERO product hits** (only unrelated prose such as "smoke test" and "trailing"). A second grep for `audio|squeal|AudioContext` also returns **ZERO hits**. The presentation surface of the whole build is therefore:

| Presentation element | Exists? | Source of its value | Reads the traction law? |
|---|---|---|---|
| Car body yaw (the drift crab) | YES | `car.drift.pose.pose` | **NO** - reads the committed slip |
| Wheel chain yaw (tire follow) | YES | the body's drift-yawed matrix | **NO** - inherits the crab |
| Front-wheel steer yaw | YES | `clamp(steerAxis) * 35 deg` | **NO** - an INPUT ECHO |
| Wheel spin (rotation) | YES | post-traction `car.speed` / radius | indirectly (a SPEED effect) |
| Tire "gradient" | YES | a fixed sun direction + a boolean | **NO** - pure lighting |
| HUD speed | YES | post-traction `car.speed` | indirectly (a SPEED effect) |
| HUD lap / time / controls hint | YES | race rules | n/a |
| Camera aim | YES | `car.heading` (+ the contested offset) | **NO** - `AUDIT-CAMERA` |
| Smoke / skid marks / dust / trails | **NO** | - | - |
| Tire squeal / audio | **NO** | - | - |
| Grip bar / slip indicator / traction lamp | **NO** | - | - |

**The headline: there is exactly ONE drift-driven visual in the entire build (the pose crab), and it is not downstream of the traction law.** Every other traction-adjacent visual is driven by SPEED or by the RAW INPUT. Nothing in the presentation layer can distinguish "the tires lost grip" from "the player is holding the stick" or "the car is braking".

### 2.B - The tire yaw-follow chain, traced end to end  *(confidence 0.95, source: static-code)*

There are **two distinct** "tire yaw" effects, and they have **different and disconnected** sources.

**(1) The body/wheel DRIFT yaw (AC-047C + AC-078).** The chain is:

```
physics.step.service.ts:454-459   stepDriftPoseSpring(state.drift.pose, { slipAngle: nextCommitted.slip, ... })
        -> drift.model.ts:756-758    poseTarget = slipAngle * 0.5 + 0.08 * articulation   (slipAngle = the COMMITTED slip)
        -> drift.model.ts:764        pose = prev.pose + poseRate * dt
browser-entry.ts:972                carPoseBag.driftYawOffsetRad = car.drift.pose.pose
car.pass.ts:1617-1626               NEATRACING_CAR_BODY_MODEL_MATRIX = MODEL * rotY(driftYawOffsetRad ?? 0)
car.pass.ts:1636-1670              the wheel chain is parented onto the SAME drift-yawed matrix (AC-078)
```

Consequences, all load-bearing:
- The **body** and the **wheels** rotate by the SAME offset, so the tires follow the body crab (this is the AC-078 "tire follow" the dispatch names). Good - the *plumbing* of the tire follow is complete and correct.
- But the **source** is the pose spring, whose target is the **COMMITTED slip** (`nextCommitted.slip`), a *latched command* - not the traction loss, and not even the **published** slip (the actual lateral-velocity angle that the scrub law reads).
- The pose is a **second-order spring** (`omega = 18`, `zeta = 0.35`, `drift.model.ts:759-764`) that **overshoots** its target. The `car.pass.ts` JSDoc (L460-495) states the declared target envelope is `0.45 * 0.5 + 0.08 = 0.3050 rad` and that the *rendered* offset is bounded only by measurement.
- The `car.pass.ts` JSDoc also states that the offset is applied **verbatim** and is "never re-derives or rescales it" - i.e. the pass is a pure sink, and all the semantics live in the physics.

**(2) The front-wheel STEER yaw (AC-044).** This is a completely separate path:

```
browser-entry.ts:964               fillCarRenderPose(..., boundSession.state.input?.steer)
generate-car.ts:70                 pose.steerAngleRad = computeWheelSteerAngleRad(steerInput ?? 0)
wheel.steer.ts:55-57               computeWheelSteerAngleRad(steerAxis) = clamp(steerAxis, -1, 1) * NEATRACING_MAX_WHEEL_STEER_RAD (35 deg)
car.pass.ts:1661-1667              rotationY(NEATRACING_WHEEL_STEER_YAW_MATRIX, isFrontWheel ? (car.steerAngleRad ?? 0) : 0)
```

`computeWheelSteerAngleRad` is a **pure linear function of the raw steering axis**, with **no gate of any kind**. The rear wheels are excluded by `isFrontWheel` and receive **no** steer yaw at all.

**The consequence is the sharpest presentation-layer finding in this audit.** The visible front-wheel turn is an **unconditioned echo of the player's input**. At `steer = 0.10` the front wheels are already at 3.5 deg; at `steer = 0.49` they are at 17.2 deg - a dramatic-looking turn - while the drift model has latched **nothing** (the engage threshold is `0.5`) and the traction loss is **exactly zero**. The tires therefore *look* like they are working at any input above roughly `0.03`, and the player is given **no other feedback whatsoever**. This is a direct, code-level explanation of "the tires still never lose traction" from the player's point of view: the one thing the player can read on the tires is the input, echoed back.

The sampled axis is the **same** object the physics replays (`race-session.ts:239-252` samples once per frame and replays the same `PhysicsStepInput` into every catch-up tick), and the source comment at `race-session.ts:237-238` says so explicitly: *"keeps `state.input` current for the composer on zero-step frames, so the front-wheel steer pose reflects the held key immediately."* So the visual and the physics read the same number - the divergence is not the number, it is the **gating**: the visual has none of the drift model's gates (engage threshold `0.5`, MIN_SPEED `15 m/s`, the re-arm latch).

### 2.C - The tire gradient: a LIGHTING effect wearing a traction costume  *(confidence 0.96, source: static-code)*

The dispatch names the "skid/gradient" as one of the traction visuals. It is not one.

**The seam** (`car.pass.ts:1516-1536`) is:

```ts
function feedTireGradientUniforms(enabled: boolean): void {
  gl.uniform1f(litUniforms.uApplyTireGradient, enabled ? 1.0 : 0.0);
  if (enabled) {
    gl.uniform1f(litUniforms.uTireGradientRadius, NEATRACING_WHEEL_HUB_HEIGHT_M);
    gl.uniform1f(litUniforms.uTireGradientBrightFactor, NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR);   // 1.7
    gl.uniform1f(litUniforms.uTireGradientDarkFactor,   NEATRACING_TIRE_GRADIENT_DARK_FACTOR);     // 0.35
    gl.uniform1f(litUniforms.uTireGradientThreshold, NEATRACING_TIRE_GRADIENT_RUBBER_MAX_CHANNEL / NEATRACING_CAR_COLOR_BYTE_MAX); // 80/255
  }
}
```

It takes **a boolean and four constants**. There is **no** slip, no traction, no grip, no speed, no per-wheel identity, no per-frame value of any kind. It is called twice per car: `false` for the body draw (L1633) and `true` for every wheel draw (L1635) - so **all four wheels receive the identical global uniform values**.

**The fragment shader** (`car.pass.ts:325-333`) is where the truth is:

```glsl
// Fixed-space tire lighting (OBS-20): the highlight follows the shared
// scene sun direction (top-front of the car) and stays fixed while the wheel
// voxels spin through it. ...
if (uApplyTireGradient > 0.5 && max(max(albedo.r, albedo.g), albedo.b) <= uTireGradientThreshold) {
  vec3 wheelCenter = vec3(uModel[3][0], uModel[3][1], uModel[3][2]);
  float t = clamp(dot(vWorld - wheelCenter, uSunDirection) / (2.0 * uTireGradientRadius) + 0.5, 0.0, 1.0);
  float factor = mix(uTireGradientDarkFactor, uTireGradientBrightFactor, t);
  albedo *= factor;
}
```

Three independent facts follow, and they are the whole story of this element:

1. **It is a function of the world-space sun direction only.** `t` depends on `vWorld`, `wheelCenter` and `uSunDirection`. None of those is a function of the car's slip, traction, grip, speed or time.
2. **The shader comment states the design intent in plain words**: the highlight *"stays fixed while the wheel voxels spin through it"*. The **uniform feed** is spin-independent (true - it is a constant), and the **rendered result** animates with the spin (also true - the tread voxels move through a fixed band). Both statements are in the source; the visual outcome is the *product* of the two.
3. **Therefore the perceived "tire activity" is proportional to SPEED, not to GRIP.** The tread slides through the light band faster the faster the wheel spins. A wheel that is *slipping* (spinning faster than the road) and a wheel that is *rolling freely* (spinning at the road speed) are visually **identical**; a car under full traction loss and a car coasting at the same speed look the same on the tire.

Two further properties of the element matter for the implementation wave:

- **The AC-088 residual is a static floor, not a traction quantity.** `NEATRACING_TIRE_GRADIENT_MIN_SPREAD = 0.5` (`car.pass.ts:411`) is the *minimum non-degenerate spread* between the bright and dark ends. Its own JSDoc says: *"This is a FLOOR, not a live value: the production spread is 1.35, which already clears the floor with margin ... The rendered gradient itself is a fragment-stage effect, so it is NOT measurable from JavaScript; the visible outcome is carried by the AC-090 user visual gate, and this constant pins the seam INPUTS."* So the Step 14G residual "tire-gradient MAGNITUDE measures 0" is a **static spread** - it is **not** a traction magnitude and cannot become one without a new input.
- **The wheel chain's world orientation does move the band across the tread.** The chain is `BODY_MODEL (incl. the drift yaw) * translate(socket) * rotY(mount yaw) * rotY(steer yaw) * rotZ(spin)`. The drift yaw (AC-078) and the steer yaw therefore rotate the tire in world space, which shifts *where* on the tread the bright band sits. A player can therefore read "the tires are turned" from the gradient - but that is the **steer echo** again, not a grip cue.

### 2.D - The HUD: three readouts, no traction  *(confidence 0.96, source: static-code)*

`host/hud.ts` is the entire HUD. Its state contract (`hud.ts:142-147`) is:

```ts
export interface NeatRacingHudState {
  readonly speedUnitsPerS: number;   // the current forward speed
  readonly raceState: RaceRulesState; // lap + time
}
```

Its DOM surface is **exactly three readouts plus one static hint** (`hud.ts:154-158`, `hud.ts:393`, `hud.ts:401-403`): `speedLabel` (cyan), `lapLabel` (magenta), `timeLabel` (cyan), and a fixed `HUD_CONTROLS_HINT_TEXT` written once at construction.

The one and only update path is `race-session.ts:261-264`:

```ts
deps.hud.update({
  speedUnitsPerS: state.car.speed,   // POST-traction speed
  raceState: state.rules,
});
```

`state.car.speed` is the **post-scrub** speed (`physics.step.service.ts:471`), so the speed number **does** fall when the scrub bites - the loss is *implicitly* visible. But:

- there is **no** grip/slip/traction readout, bar, lamp or gauge;
- there is **no** comparison against a target speed, so the player cannot see "I am losing 12 km/h to the slide";
- there is **no** colour or flash change on the speed label when grip is lost;
- the HUD contract has **no field** that could carry a traction value - adding one is a **contract change**, not a wiring change.

**There is no speed-loss HUD.** The speed number is the only traction-caused pixel in the HUD, and it is indistinguishable from braking.

### 2.E - The wheel spin: the only traction-caused pixel, and it is a speed effect  *(confidence 0.92, source: static-code)*

`browser-entry.ts:948-955` accumulates the visual spin from `computeVisualWheelSpinDeltaRad` (`wheel.mount.utils.ts:419-437`):

```
omega = clampVisualWheelSpinOmega(trueWheelSpinOmega(groundSpeedMPerS, wheelRadiusM))
      = clamp(v / r, +-NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S)   // 2 rev/s = 12.566 rad/s
```

`groundSpeedMPerS` is `car.speed`, the **post-traction** speed. So the wheels **do** visibly slow down when the scrub bites. But:

- it is a **speed** effect - braking produces the identical visual;
- it is **clamped** at 2 rev/s (the AC-015 strobing cap), so the visible delta is bounded;
- a **slipping** wheel spins *faster* than the road speed, not slower - the model has no slip-ratio concept on the visual side, so a sliding tire renders as a *slower* wheel. The visual is backwards for a true slide.

### 2.F - EVERY published-but-unwired drift output  *(confidence 0.96, source: static-code)*

`PhysicsCarState` publishes (`physics.step.service.ts:78-130`):
`x, y, heading, speed, yawRate, slipAngle, lateralVelocity, drift{committed{slip,turnSign,reArmed}, pose{pose,poseRate,articulation}}, tick`.

A repo-wide grep over `examples/neatRacing/**/*.ts` for `\.committed\.(slip|turnSign|reArmed)|pose\.(poseRate|articulation)|\.yawRate` returns hits in **exactly three files, all of them `.test.ts`**: `physics/collision.utils.test.ts`, `physics/drift.model.test.ts`, `physics/physics.step.service.test.ts`. **Zero production readers.** (This independently re-confirms `AUDIT-PHYSICS` F4; it is not a new finding, and it is listed here because the dispatch explicitly asks for the complete enumeration.)

| # | Published output | Where written | Production reader? | What it is |
|---|---|---|---|---|
| 1 | `car.yawRate` | declared `physics.step.service.ts:95`, init `:308`, written `:479` (`yawRatePerTick`, rad/tick) | **NONE** - test only (`physics.step.service.test.ts:481-483`) | the per-tick turning rate - the natural "speed x turning" and proportional-camera input |
| 2 | `car.drift.pose.poseRate` | `drift.model.ts:759-763`, returned `:766` | **NONE** - test only | the visible yaw **rate** (rad/s) of the crab |
| 3 | `car.drift.pose.articulation` | `drift.model.ts:735-753`, returned `:766` | **NONE** - test only (consumed only by the model's own pose target) | the steer-yaw articulation, armed by the committed slip |
| 4 | `car.drift.committed.slip` | `physics.step.service.ts:396-403` | **NONE** - test only | the **latched command** (the commanded slide) |
| 5 | `car.drift.committed.turnSign` | `drift.model.ts` latch | **NONE** - test only | the latched turn direction |
| 6 | `car.drift.committed.reArmed` | `drift.model.ts` latch | **NONE** - test only | the re-arm token |
| 7 | the **scrub** (traction loss) | computed `physics.step.service.ts:419-427` | **NOT PUBLISHED AT ALL** | the literal traction loss; only its *effect* (a reduced speed) survives |
| 8 | `car.drift.pose.pose` | `physics.step.service.ts:454-459` | **YES** - `browser-entry.ts:972` (the crab) and `:981` (the seam) | the **only** production consumer of the drift channel that reaches a visual |
| 9 | `car.slipAngle` | `physics.step.service.ts:474` | **YES** - `collision.utils.ts:386-389` (the bounce recompose) + `browser-entry.ts:980` (the seam) | the published slip; **not read by any visual** |
| 10 | `car.lateralVelocity` | `physics.step.service.ts:476` | **YES** - `collision.utils.ts:386-389` only | the lateral velocity; **not read by any visual** |

**A second, sharper framing of item 8.** The pose and the traction law read **two different slip variables**:

- the **pose** target is `0.5 * COMMITTED slip + 0.08 * articulation` (`drift.model.ts:756-758`, fed `nextCommitted.slip` at `physics.step.service.ts:457`);
- the **scrub** reads the **PUBLISHED** slip (`state.slipAngle`, the tick-start `atan2` of the lateral velocity) at `physics.step.service.ts:422`.

The committed slip is a *latched command* that saturates at the `0.45 rad` clamp; the published slip is the *measured* lateral-velocity angle that the chase has not yet converged to. They are equal only in steady state. Concretely: on the first tick after the commit latches, the committed slip is already at its target while the published slip is still near zero, so **the car crabs while the traction loss is still zero** - and the pose spring (`omega = 18`) reaches its full target in well under a second. This is the mechanism by which the *one* drift visual can be at its full envelope with **no** grip lost, and it is the mechanical form of the user's complaint that the drift is "a poorly implemented camera trick, that has no connection with the drifting". **There is no shared scalar (S-4) between the visual and the tire.**

### 2.G - The debug seam  *(confidence 0.95, source: static-code)*

`window.neatRacingDriftState` (`browser-entry.ts:978-984`) is an **exactly five-field** debug surface, written once per frame:

```ts
window.neatRacingDriftState = {
  poseRad: car.drift.pose.pose,
  slipAngleRad: car.slipAngle,
  speedMps: car.speed,
  tickHz: 1 / NEATRACING_FIXED_TIMESTEP_SECONDS,
  tick: car.tick,
};
```

The interface (`browser-entry.ts:112-152`) is documented as a *"read-only, EXACTLY five-field view of the live drift channel"*. **There is no field for** traction loss / scrub, yaw rate, pose rate, committed slip, turn sign, articulation, or a grip factor.

The W4b packet plans a **6th field**, and `AUDIT-CAMERA` F8 accepts it **only if the follow state is ONE SHARED OBJECT**; the field is written at **render** rate but its source updates at **tick** rate, so the new 44th check must tolerate **60 Hz quantization** above 60 Hz displays.

**This audit's constraint on that decision:** the 6th field must be the **shared traction scalar** (the merged brief's D_B / D_J), **not** merely the camera follow offset. If the 6th field carries only the follow state, the fixture would keep measuring the *camera* and never the *tires*, and the S-4 "one shared scalar drives both" requirement would be re-broken **at the seam** - the exact defect this wave exists to close.

### 2.H - The wire ledger: what EXISTS vs what must be CREATED  *(confidence 0.94, source: static-code)*

This is the table the implementation wave asked for.

| Wire | Status | Where it lives | Reads the traction law? |
|---|---|---|---|
| committed slip -> pose spring -> body crab | **EXISTS** | `physics.step.service.ts:454` -> `browser-entry.ts:972` -> `car.pass.ts:1617` | NO (reads the committed slip) |
| body crab -> wheel chain (tire follow, AC-078) | **EXISTS** | `car.pass.ts:1636-1670` | NO (inherits the crab) |
| raw steer axis -> front-wheel steer yaw (AC-044) | **EXISTS** | `browser-entry.ts:964` -> `wheel.steer.ts:55` -> `car.pass.ts:1661` | NO (an input echo) |
| post-traction speed -> wheel spin | **EXISTS** | `browser-entry.ts:948-955` -> `wheel.mount.utils.ts:419` | indirectly (a speed effect) |
| post-traction speed -> HUD speed | **EXISTS** | `race-session.ts:261-264` -> `hud.ts:401` | indirectly (a speed effect) |
| fixed sun -> tire gradient | **EXISTS but VACUOUS** | `car.pass.ts:1516` + the fragment shader L325-333 | NO (lighting only) |
| pose + slip + speed + tick -> 5-field seam | **EXISTS but INCOMPLETE** | `browser-entry.ts:978-984` | NO (no traction field) |
| **traction-loss scalar published on the physics state** | **MUST BE CREATED** | `physics.step.service.ts` (the scrub is computed at `:419-427` and discarded) | - |
| **car-pass traction channel (per car, and optionally per tire)** | **MUST BE CREATED** | `CarPassCarState` (`car.pass.ts:435-497`) has **no** such field | - |
| **HUD traction/grip readout + contract field** | **MUST BE CREATED** | `hud.ts:142` (`NeatRacingHudState`) | - |
| **smoke / skid marks / dust / trails subsystem** | **MUST BE CREATED FROM ZERO** | no source file exists | - |
| **tire-squeal / audio channel** | **MUST BE CREATED FROM ZERO** | no `AudioContext` anywhere | - |
| **per-tire grip differentiation (the rears breaking away)** | **MUST BE CREATED** | the gradient seam carries a boolean and 4 globals; the shader has no per-wheel input | - |
| **6th seam field carrying the shared traction scalar** | **MUST BE CREATED** | `browser-entry.ts:978-984` | - |

### 2.I - Literal GAPS

**G-V1 - The car-pass contract has NO traction channel.** *(confidence 0.97, gap type: missing)*
`CarPassCarState` (`car.pass.ts:435-497`) has **exactly seven** fields: `worldX`, `worldZ`, `yawRad`, `teamColorId`, `wheelSpinRad`, `steerAngleRad?`, `driftYawOffsetRad?`. **Not one** of them is slip, traction, grip, slip-ratio, scrub or a per-tire state. The presentation pass is **structurally incapable** of expressing traction loss, because it is never told about it. Any traction visual the implementation wave wants must first **extend this contract**.

**G-V2 - No traction-loss scalar is published; the scrub is computed and DISCARDED.** *(confidence 0.95, gap type: missing; CROSS-REFERENCE to `AUDIT-PHYSICS` F5 - not a new finding)*
`physics.step.service.ts:419-427` computes `computeLongitudinalSpeedScrub(state.slipAngle, nextSpeed, speedPercent) * dt` **inline** into `scrubbedSpeed`; **no scalar survives on the state**. Therefore the presentation layer has **nothing to read**, and S-4 ("one shared scalar drives both") is structurally impossible today. The implementation wave must **publish** the scalar (the merged brief's D_J already says so) - the presentation half of this audit confirms that **no presentation consumer exists to receive it**, and that the *only* candidate readers today (the pose spring and the camera) read the *committed* slip instead.

**G-V3 - The ONE drift visual is driven by the COMMITTED slip, not by the traction loss - so the car crabs with zero grip lost.** *(confidence 0.93, gap type: contradicts)*
The pose target is `0.5 * committedSlip + 0.08 * articulation` (`drift.model.ts:756-758`), fed the **committed** slip (`physics.step.service.ts:457`); the traction law reads the **published** slip (`physics.step.service.ts:422`). The two are different quantities. On the tick the commit latches, the committed slip is at its target while the published slip (and therefore the whole traction loss) is still ~zero, and the second-order pose spring (`omega = 18`, `zeta = 0.35`) reaches its full target in well under a second. The car is therefore **visibly at its full drift envelope while the tires have lost nothing** - a direct, code-level instance of the user's "poorly implemented camera trick, that has no connection with the drifting". The reverse also holds: the committed slip is a latched command that can saturate at the `0.45 rad` clamp regardless of the measured slip, so the visual can sit at a large offset with a small actual slide.

**G-V4 - The tire gradient is a fixed-space LIGHTING effect masquerading as a traction surface.** *(confidence 0.96, gap type: contradicts)*
`feedTireGradientUniforms(enabled: boolean)` (`car.pass.ts:1516-1536`) takes a boolean and four constants - **no drift, slip, speed or traction input** - and the fragment shader (`car.pass.ts:325-333`) computes a band from `dot(vWorld - wheelCenter, uSunDirection)`. The shader's own comment says the highlight *"stays fixed while the wheel voxels spin through it"*, so the visible tire activity is the tread sliding through a fixed light band: **proportional to SPEED, blind to GRIP**. A sliding tire and a free-rolling tire render identically. The AC-088 `MIN_SPREAD` residual is a **static spread floor**, not a traction magnitude.

**G-V5 - The tire yaw-follow is an INPUT ECHO, not a physics echo.** *(confidence 0.95, gap type: contradicts)*
`computeWheelSteerAngleRad(steerAxis) = clamp(steerAxis, -1, 1) * 35 deg` (`wheel.steer.ts:55-57`) is a **pure linear function of the raw steering axis**, applied to the front wheels only (`car.pass.ts:1661-1667`); the rear wheels get **no** steer yaw. It bypasses **every** drift gate: the `0.5` engage threshold, the `15 m/s` MIN_SPEED gate and the re-arm latch. At `steer = 0.49` the front wheels are at `17.2 deg` - a dramatic-looking turn - while the drift model has latched nothing and the traction loss is **exactly zero**. This is the strongest single explanation the code offers for "the tires still never lose traction": the *only* thing the player can read on the tires is their own input, echoed back.

**G-V6 - There is no speed-loss HUD and no traction/grip readout.** *(confidence 0.96, gap type: partial)*
`host/hud.ts` has exactly three readouts (speed / lap / time) plus a static controls hint; `NeatRacingHudState` (`hud.ts:142-147`) carries only `{ speedUnitsPerS, raceState }`. The speed number **is** post-traction (`race-session.ts:261-264` -> `state.car.speed` = `physics.step.service.ts:471`), so a speed loss is *implicitly* visible in the number - but there is **no** grip bar, slip indicator, traction lamp, delta-vs-target or colour change, and the HUD contract has **no field** that could carry one. Adding a traction readout is a **contract change**, not a wiring change.

**G-V7 - Every published-but-unwired drift output (the complete enumeration).** *(confidence 0.96, gap type: missing)*
`car.yawRate`, `car.drift.pose.poseRate`, `car.drift.pose.articulation`, `car.drift.committed.slip`, `car.drift.committed.turnSign`, `car.drift.committed.reArmed` have **ZERO production readers** (all readers are `.test.ts`); the **scrub is not published at all**. The only production consumer of the whole drift channel that reaches a visual is `car.drift.pose.pose` -> the crab. `car.slipAngle` and `car.lateralVelocity` are read in production only by the bounce recompose (`collision.utils.ts:386-389`) and the seam. `car.yawRate` and `car.drift.pose.poseRate` are the two channels that a *proportional* camera (S-1) and a *traction-proportional* visual would both want, and both are free to wire.

**G-V8 - The debug seam carries NO traction channel, and the planned 6th field must be the shared scalar.** *(confidence 0.94, gap type: partial)*
`window.neatRacingDriftState` is exactly five fields (`browser-entry.ts:978-984`): `poseRad`, `slipAngleRad`, `speedMps`, `tickHz`, `tick`. There is **no** field for traction loss, yaw rate, pose rate, committed slip or turn sign. Per `AUDIT-CAMERA` F8 the 6th field is accepted only if the follow state is ONE SHARED OBJECT, is written at render rate, updated at tick rate, and the new 44th check tolerates 60 Hz quantization above 60 Hz displays. **This audit's constraint:** the 6th field must be the **shared traction scalar** (D_B / D_J), not only the follow offset - otherwise the fixture keeps measuring the camera and never the tires, and S-4 is re-broken at the seam.

**G-V9 - No smoke / skid-mark / dust / trail subsystem exists, and no audio channel exists either.** *(confidence 0.95, gap type: missing)*
A repo-wide grep over `examples/neatRacing/**/*.ts` for `smoke|skid|trail|tireMark|particle` returns **zero product hits**; a grep for `audio|squeal|AudioContext` returns **zero hits**. The dispatch's "smoke" and the user's "skid" expectation have **no substrate at all**. This is the largest new-build item in the whole wave: a new render pass (or a new draw phase inside the car pass), its own uniform seam, its own draw ordering relative to the blob shadow / body / wheels, a life/spawn policy, and - critically for this repo - a **determinism** story (the physics is a pure 60 Hz tick; particles would have to be tick-stepped, not wall-clock stepped, or they will desync the deterministic replay contract).

**G-V10 - Per-tire grip differentiation is impossible today.** *(confidence 0.92, gap type: missing)*
The tire gradient is fed **per draw** with the **same four global uniform values** for all four wheels, and the seam carries **only a boolean** - no mount identity (`front-left` / `front-right` / rear), no per-tire grip. The shader's `wheelCenter` is derived from `uModel[3]`, so the per-wheel *geometry* differs, but the per-wheel *state* does not exist. The single most legible drift cue in an arcade racer - **the rear tires breaking away while the fronts stay hooked** - is therefore inexpressible. The hook point exists (`feedTireGradientUniforms` is called per wheel), so the work is to widen the seam and add a per-tire scalar, not to find a new location.

**G-V11 - The only traction-caused pixel is the wheel spin, and it is a speed effect that is backwards for a real slide.** *(confidence 0.90, gap type: partial)*
`computeVisualWheelSpinDeltaRad` derives `omega = v/r` from the **post-traction** speed and clamps it to 2 rev/s (`wheel.mount.utils.ts:419-437`, `NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S` at `:121`). The wheels do visibly slow when the scrub bites - but the effect is **indistinguishable from braking**, and a *slipping* wheel physically spins **faster** than the road speed, so the model has no slip-ratio on the visual side and the rendered slide reads backwards.

---

## 3. Decision

**D1 - Gap classification of the dispatch's clauses.**

| Clause | Gap type | Confidence | One-line verdict |
|---|---|---|---|
| "tire yaw-follow" | **contradicts** | 0.95 | The tire follow **plumbing** is complete and correct (AC-078 parents the wheels onto the drift-yawed matrix), but the front-wheel steer yaw is an **unconditioned input echo** and the body crab is driven by the **committed slip** - neither reads the traction loss. |
| "skid / gradient" | **contradicts** | 0.96 | There is **no** skid. The "gradient" is a fixed-sun **lighting** band with a boolean-only seam; its visible motion is the tread sliding through the light, i.e. speed, not grip. |
| "smoke" | **missing** | 0.95 | No smoke, skid mark, dust, trail or particle subsystem exists anywhere in the codebase. |
| "speed-loss HUD" | **partial** | 0.96 | The speed **number** is post-traction, so a loss is implicitly visible; there is **no** traction/grip readout and the HUD contract has no field for one. |
| "every published-but-unwired drift output" | **missing** | 0.96 | Six published outputs have **zero** production readers; the scrub is **not published at all**; the only production consumer of the drift channel is the crab. |
| "which wires exist and which must be CREATED" | **answered** | 0.94 | See the ledger in 2.H: **7 wires exist**, **7 must be created**, and 3 of the 7 existing ones are *structurally* incapable of expressing traction. |

**D2 - The single sentence the implementation wave needs.** *The build has exactly one drift-driven visual, it is fed by the committed slip rather than by the traction loss, the two most tire-like effects (the front-wheel steer and the tire gradient) are echoes of the input and of the speed respectively, and there is no traction-loss scalar, no traction HUD, no smoke, and no per-tire state anywhere in the presentation layer.*

**D3 - The presentation inventory, restated as a contract table (this is the deliverable).**

| Layer | What exists | What it is fed by | Traction-driven? |
|---|---|---|---|
| Body crab | `driftYawOffsetRad` | the pose spring over the **committed** slip | **NO** |
| Wheel chain | the body's drift-yawed matrix (AC-078) | inherits the crab | **NO** |
| Front-wheel steer yaw | `steerAngleRad` | the **raw steering axis** (35 deg lock) | **NO** |
| Rear-wheel steer yaw | nothing (rears are excluded) | - | - |
| Wheel spin | `wheelSpinRad` | **post-traction** speed / radius, clamped at 2 rev/s | indirectly, as a **speed** effect |
| Tire gradient | `uApplyTireGradient` + 4 constants | the **fixed sun direction** | **NO** |
| HUD speed | `speedUnitsPerS` | **post-traction** speed | indirectly, as a **speed** effect |
| HUD lap / time / hint | race rules / a literal | - | - |
| Seam | 5 fields | pose, published slip, speed, tick rate, tick | **NO** (no traction field) |
| Smoke / skid / dust / audio | **nothing** | - | - |

**D4 - The order of creation is not free.** Three things must be created, and the order is forced by the dependency chain:

1. **Publish the traction-loss scalar on the physics state** (the physics half of G-V2; the merged brief's D_J). Until this exists, **nothing** else in the presentation layer can be wired to the traction law - every visual would have to re-derive the scrub in the view layer, which is the duplicated-physics-law hazard `AUDIT-PHYSICS` F5 identified.
2. **Widen the car-pass contract** (`CarPassCarState`) with a traction field, and (if the rears-breaking-away cue is wanted) with a **per-tire** field. Until the contract carries the value, the pass cannot consume it (G-V1, G-V10).
3. **Only then** wire the consumers: the HUD readout (G-V6), the seam's 6th field (G-V8), and any new effects pass (G-V9).

**D5 - The seam's 6th field must be the shared traction scalar.** Per `AUDIT-CAMERA` F8 the 6th field is accepted only if the follow state is ONE SHARED OBJECT, written at render rate and updated at tick rate, with the 44th check tolerating 60 Hz quantization above 60 Hz displays. This audit adds one constraint: **the shared object must contain the traction scalar, not only the follow offset.** A 6th field carrying only the follow offset would let the fixture report GREEN while the tires still show nothing - the `AUDIT-PHYSICS` R-4 failure mode, repeated at the seam. The traction scalar is already speed-proportional (`0.55 * |slip| * speed^2 / 89`), so the 44th check must also be a **ratio**-shaped assertion, not a floor.

**D6 - The visual effect of the two slip variables must be re-thought, not just re-wired.** G-V3 is not fixed by "wiring the scrub to the crab". The crab is a **second-order spring over a latched command**; the traction law reads the **measured** slip. If the implementation wave makes the crab read the published slip, the crab stops overshooting (the spring's far-side pop is the feature, per the `car.pass.ts` JSDoc), and if it keeps the committed slip, the visual still leads the physics. The author must **decide which of the two is the visual's source** and record it - this is a design decision, not an implementation detail.

**D7 - The tire gradient must either be honestly re-labelled or genuinely extended.** Two coherent options, and the author must pick one:
- **(a) HONEST** - rename/re-document the element as a **lighting** effect (OBS-20 fixed-space tire lighting), and build the traction surface elsewhere (a new per-tire gradient input, a new smoke/skid pass, a HUD element). This is the cheaper, more honest option, and it removes the risk that a future author keeps treating the gradient as the traction surface.
- **(b) EXTENDED** - widen `feedTireGradientUniforms` from `enabled: boolean` to a per-tire `(traction, mount)` feed and add a per-tire grip term to the shader. This makes the gradient a real traction surface, but it touches the fragment shader and the AC-088 MIN_SPREAD contract, whose JSDoc explicitly says the rendered gradient is *"NOT measurable from JavaScript"*.
Option (b) re-opens AC-088; option (a) does not. The author must state which.

**D8 - The three un-wired channels are the cheapest traction visuals available, and they should be used first.** `car.yawRate` (the per-tick turning rate) and `car.drift.pose.poseRate` (the crab's rate) are already published, already per-tick, and already unwired. `car.drift.committed.turnSign` is the latched turn direction. Together they are enough to express "the car is turning" and "the car is sideways, and in which direction" **without touching the physics contract at all** - a strictly cheaper first tranche than the full traction-scalar wave, and a good RED-lane target.

**D9 - The numbered gaps in implementation priority order.**

| # | Gap | Type | Confidence | Priority |
|---|---|---|---|---|
| **G-V2** | no published traction-loss scalar (the scrub is computed and discarded) | missing | 0.95 | **P0** - blocks everything (D4) |
| **G-V1** | the car-pass contract has no traction channel | missing | 0.97 | **P0** - blocks everything (D4) |
| **G-V3** | the one drift visual reads the committed slip, not the traction loss | contradicts | 0.93 | **P0** - this is the user's complaint |
| **G-V5** | the tire yaw-follow is an input echo with no gate | contradicts | 0.95 | **P1** - cheap to fix, high visual impact |
| **G-V4** | the tire gradient is lighting, not traction | contradicts | 0.96 | **P1** - decide (a) or (b) (D7) |
| **G-V7** | six published outputs with zero production readers (+ the scrub) | missing | 0.96 | **P1** - D8 is the cheapest tranche |
| **G-V6** | no speed-loss / traction HUD | partial | 0.96 | **P2** - a contract change to `NeatRacingHudState` |
| **G-V8** | the 5-field seam carries no traction channel; the 6th field must be the shared scalar | partial | 0.94 | **P2** - constrained by `AUDIT-CAMERA` F8 |
| **G-V10** | per-tire grip differentiation is impossible | missing | 0.92 | **P2** - depends on G-V4 (b) |
| **G-V11** | the only traction pixel is a clamped, backwards speed effect | partial | 0.90 | **P3** |
| **G-V9** | no smoke / skid / dust / trail / audio subsystem exists | missing | 0.95 | **P3** - the largest new-build, and it needs a determinism decision |

**D10 - What this audit does NOT adjudicate.** The camera aim path, the `cameraPose` trap (`frame-composer.ts:822` feeds `cameraPose` to the **starfield** only - so it is NOT a general camera channel), the wheel-spin reset idiom, the `stepPhysics` wrapper and the `tau = 0.08` follow law remain `AUDIT-CAMERA`'s. The traction law and the `0.45 / 0.5 / 3.5 / 0.55` constants remain `AUDIT-PHYSICS`'s settled facts and were **not** re-proved. The live keyboard chain remains `14N-V5-audit-input`'s.

---

## 4. Risks

- **R1 (highest) - the "presentation" surface may look healthy because the input echo looks like traction.** The front-wheel steer yaw is a *linear* function of the steering axis with no gate, so on screen the tires *always* respond to the player. Any visual review that asks "do the tires move when I steer?" will answer YES and conclude the chain is wired, while the traction law is doing nothing. **Mitigation: every new traction-visual contract must assert the response to a quantity the player does NOT control** (the published slip or the traction scalar) - e.g. a *same-input, different-traction* comparison, never a *different-input* comparison.
- **R2 - the drift envelope is pinned by a pose contract, so a traction visual may be built against the wrong variable.** The `car.pass.ts` JSDoc (L460-495) pins the *target* envelope at `0.3050 rad` and the *fallback* re-record target at `0.2250 rad`, and both are derived from the **committed** slip. If the wave makes the crab read the **published** slip instead, the envelope is derived from a different variable and **every one of those pinned numbers must be re-derived** - including the 44th fixture check. This is a coupled blast radius of the same kind as `AUDIT-PHYSICS` R-2.
- **R3 - the `0.45 rad` commit clamp and the `0.5` engage threshold mean the traction visual can be *saturated* in ordinary driving.** Once the commit clamps, a further steering increase changes nothing in either the pose or the scrub, so a traction visual driven by the committed slip will **plateau** in normal cornering. A visual that saturates in ordinary driving reads as "broken" to the player even when the law is correct.
- **R4 - the smoke / skid work has a determinism trap.** The physics is a pure 60 Hz tick with a documented replay contract (`physics.step.service.ts:27-28`, "the service is PURE"). A wall-clock-stepped particle system would break the deterministic replay and would need its own tick-stepped state. **Mitigation: decide the particle state model (tick-stepped, seeded from the deterministic tick counter) BEFORE the first particle is drawn** - the same discipline the plan already applies to the `pulse` system (`collectActivePulses(car.tick * dt, ...)`).
- **R5 - a 6th seam field that carries only the follow offset would make the fixture report GREEN while the tires show nothing.** This is `AUDIT-PHYSICS` R-4 (the vacuous 340 km/h clause) repeated at the seam. **Mitigation: D5 - the shared object must contain the traction scalar, and the 44th check must be ratio-shaped.**
- **R6 - the 60 Hz quantization constraint on the 6th field cuts against a fast-changing traction scalar.** The seam is written once per frame but the scalar updates once per tick, so at display rates above 60 Hz the field is stale between ticks, and the traction scalar is **speed-proportional** (and therefore moves fast at speed). A 44th check that asserts a *rate* on the seam must tolerate the quantization, or it will fail intermittently.
- **R7 - no runtime measurement was performed.** Every claim here is a static-code read. In particular, the claim that "the car crabs with zero grip lost" is an **argument from the code** (two different slip variables + a spring time constant), not a measurement. Before the contract is re-authored, a code-level lane should confirm: (i) the pose magnitude at the first tick after the commit, versus the published slip and the scrub at the same tick; (ii) the front-wheel steer angle at `steer = 0.49` versus the committed slip and the scrub; (iii) the wheel-spin delta caused by the scrub, isolated from braking.
- **R8 - the "no child agents" constraint.** The dispatch explicitly forbade child agents inside each wave (`O_1`), overriding the generic Tier-1 delegation mandate. This audit is therefore **single-context** and its findings are **not** cross-checked by an independent scout. Every headline claim is a direct code read with a cited line, and the three highest-stakes claims - (a) the zero production readers of the six unwired outputs, (b) the absence of any smoke/skid/particle/audio subsystem, (c) the boolean-only tire-gradient seam - were each re-verified by an **independent repo-wide grep** in this session. Residual risk: a *missed* gap that a second pair of eyes would have found. **Mitigation: the 01-planning author re-runs the three greps before the contract is frozen.**
- **R9 - the audit boundary.** This artifact adjudicates the **presentation** surface only. The physics-side facts it relies on are `AUDIT-PHYSICS`' settled rows, not this audit's findings; if that audit's conclusions are revised by the orchestrator's merge, G-V2's wording (and only its wording) changes. The camera surface remains `AUDIT-CAMERA`'s.

---

## 5. Evidence index (files read, read-only)

| File | Read | Purpose |
|---|---|---|
| `plans/neatracing-voxel-racer.plans.md` | L10000-10291 (the AUDIT-CAMERA findings, the MERGED AUTHOR BRIEF D_A..D_N, USER VERDICT 4, USER VERDICT 5), L10134-10221 (the AUDIT-PHYSICS PlanUpdate, the format template) | the dispatch, the settled facts, the PlanUpdate shape |
| `plans/neatracing-voxel-racer.audit-physics.research.md` | L1-40, L243-400 | the precedent artifact; the settled physics facts (F1-F8, D1-D6, R1-R8) |
| `examples/neatRacing/engine/passes/car.pass.ts` | L225-340 (the lit shader + the tire-gradient block), L370-425 (the gradient constants + the MIN_SPREAD JSDoc), L428-497 (`CarPassCarState` + the drift-yaw contract), L1500-1560 (`feedTireGradientUniforms`), L1600-1680 (the body/wheel draw order + the AC-078 wheel parenting) | the tire + body presentation pass |
| `examples/neatRacing/host/hud.ts` | L136-160 (`NeatRacingHudState` + the three labels), L300-410 (the DOM construction + the update path) | the whole HUD |
| `examples/neatRacing/host/race-session.ts` | L200-270 (the input sampling, the tick replay, the HUD update) | the HUD's input, and the input-object sharing |
| `examples/neatRacing/host/frame-composer.ts` | L140-190, L780-830 | the `cameraPose` starfield-only channel (the trap) |
| `examples/neatRacing/browser-entry/browser-entry.ts` | L112-152 (the 5-field seam interface), L880-1010 (`carPoseBag`, the pose feed, the seam write, the camera forward, the wheel-spin accumulation) | the presentation seam + the only production drift consumer |
| `examples/neatRacing/car/wheel.steer.ts` | **complete, 58 lines** | the front-wheel steer echo |
| `examples/neatRacing/car/generate-car.ts` | L20-75 | the render-pose bag and the steer feed |
| `examples/neatRacing/voxel/wheel.mount.utils.ts` | L80-130 (the hub height, the 2 rev/s cap), L370-440 (the spin clamp + `computeVisualWheelSpinDeltaRad`) | the wheel-spin visual |
| `examples/neatRacing/physics/physics.step.service.ts` | L78-130 (`PhysicsCarState`), L370-482 (the tick pipeline: the yaw, the commit, the chase, the scrub, the pose spring, the published state) | the published contract + the discarded scrub |
| `examples/neatRacing/physics/drift.model.ts` | L720-770 (`stepDriftPoseSpring`) | the pose target and the two slip variables |
| `examples/neatRacing/physics/collision.utils.ts` | L30-40, L220-235, L300-320, L380-500 | the only production readers of `slipAngle` / `lateralVelocity` + the zeroed drift channel on bounce |
| `examples/neatRacing/physics/ai.autopilot.ts` | the input sampler (grep) | the autopilot reads only `x, y, heading, speed` - it never reads the drift channel |
| `examples/neatRacing/engine/lighting.test.ts` | grep (`uTireGradient*`) | the uniform-name contract + the test-only coverage of the gradient seam |
| repo-wide greps | `examples/neatRacing/**/*.ts` for `smoke\|skid\|trail\|tireMark\|particle`; `audio\|squeal\|AudioContext`; `\.committed\.(slip\|turnSign\|reArmed)\|pose\.(poseRate\|articulation)\|\.yawRate`; `uTireGradient\|TireGradient`; `computeWheelSteerAngleRad`; `fillCarRenderPose` | the zero-reader, zero-subsystem and zero-hits findings |

**No file was modified by this audit.** The only writes are this artifact and the `AUDIT-VISUALS` PlanUpdate appended to `plans/neatracing-voxel-racer.plans.md`.
