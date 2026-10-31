# AUDIT-MOTION - read-only code audit of where the committed slip / speed scrub IS (and is NOT) applied to the car's ACTUAL MOTION

**Artifact id**: `neatracing-voxel-racer.audit-motion`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 5` record, `plans/neatracing-voxel-racer.plans.md` (the `audit_waves` block), `audit_waves[0]` = `14N-V5-audit-motion`
**Subject**: `examples/neatRacing/physics/physics.step.service.ts` (the live 60 Hz tick pipeline) + `examples/neatRacing/physics/drift.model.ts` (the drift law) + `examples/neatRacing/browser-entry/browser-entry.ts` (the render bridge) + `examples/neatRacing/engine/passes/car.pass.ts` (the body draw) + `examples/neatRacing/host/race-session.ts` (the live loop) + `examples/neatRacing/physics/collision.utils.ts` (the world-velocity algebra)
**Constraints honoured**: READ-ONLY code audit · NO child agents (the dispatch's `O_1`) · NO browser runs · NO Chrome MCP · no production byte touched.

---

## 0. Provenance, method and the naming deviation

**Method.** Every claim below is a *static-code* read, plus a **read-only in-memory execution of the exact audited sources** (no browser, no Chrome MCP, no repo mutation). The numeric claims marked **MEASURED** were produced by importing the repository's own `physics.step.service.ts` through `esbuild` into a temp file **outside** the repository (`$TEMP`), driving the real `stepPhysics` at the real 60 Hz fixed timestep, and reading the real returned state. The repository working tree was not modified and no build artifact was written inside it.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. This audit is the only wave of the four that produced **runtime** evidence (headless, in-memory, no browser), so its numeric findings outrank the static-code rows of the sibling waves and are ranked `runtime` above.

**Naming deviation (disclosed).** The contract names `plans/<PlanName>.research.md`. That exact path is **already occupied** by a different, closed 14N artifact (the 2026-10-24 drift-feel *integration coupling* design), and `plans/neatracing-voxel-racer.drift-feel.research.md` and `plans/neatracing-voxel-racer.audit-physics.research.md` are likewise occupied, as is the sibling `plans/neatracing-voxel-racer.audit-visuals.research.md` written by wave [3] on the same day. Clobbering or appending to a closed, dated artifact would fuse two different questions under one date. This artifact therefore follows the repo's own topic-suffix family convention as a **new sibling**: `plans/neatracing-voxel-racer.audit-motion.research.md`. It remains a `plans/*.research.md` sibling of the tracker and is **not** under `docs/research/`. The orchestrator may rename or merge it without loss.

**Scope boundary.** This is the **MOTION** half: the physics tick -> the car entity's velocity / heading / travel direction, and the exact line where the drift stops changing how the car moves. The *presentation inventory* is `14N-V5-audit-visuals`' scope; the *keyboard chain* is `14N-V5-audit-input`'s scope; the *numeric engage envelope* is `14N-V5-audit-envelope`'s scope. Cross-references are flagged inline and are not re-proved.

**Non-duplication.** The settled `AUDIT-PHYSICS` facts (the committed-slip clamp 0.45, the 0.5 engage threshold, the 3.5 rad/s ramp, the `0.55*|slip|*speed^2/89` scrub formula, the camera rigid-mirror F1, the `0.5*slip + 0.08*articulation` pose consumption) are **not** re-proved. This audit answers the question the four waves were spawned for: **where does the committed slip and the speed scrub actually reach the car's motion, and where does it stop.**

---

## 1. Question

The `USER VERDICT 5` dispatch (`audit_waves[0]`) asks:

> TRACE where committed slip / speed scrub is (or is NOT) applied to the car's ACTUAL MOTION in the live runtime path: `physics.step.service.ts` integration -> the car entity's velocity/heading/travel direction. MARK THE EXACT BREAK LINE where drift stops changing how the car moves, and name the missing wiring (lateral velocity divergence, travel-vs-heading separation, scrub consumption).

The user's own verdict supplies the reason the question matters:

> "even worse! there is NO drifting! the tires still never loose traction! which is the main fucking point. It's just a poorly implemented camera trick, that has no connection with the drifting, there are no drifiting physics in place, I can take turns at full speed and the car never drifts, it's only confusing due to the odd camera angle. The current implementation is either not wired or a complete failure!"

So the audit must answer three things:

1. **Does the committed slip change where the car goes?** (the travel-vs-heading separation)
2. **Does the speed scrub change how fast the car goes?** (the scrub consumption)
3. **If both are real, why does the player see no drift?** - i.e. where is the exact line where the drift stops changing how the car moves *as seen*.

---

## 2. Evidence

### 2.A - The live tick pipeline, end to end  *(confidence 0.97, source: static-code + runtime)*

`examples/neatRacing/physics/physics.step.service.ts:366` `stepPhysics` is the single motion producer of the whole runtime. Its five stages, with the drift channel inserted at 4a-4e:

| Stage | Lines | What it does | Does the slip change the motion here? |
|---|---|---|---|
| 1 surface resolve | `:373-375` | reads the surface under the car at tick start | no |
| 2 yaw / heading | `:377-381` | `nextHeading = wrapHeading(state.heading + yawRatePerTick)` | no (the nose only) |
| 3 longitudinal integrate | `:384-390` | `nextSpeed` (accel / brake / drag / off-road) | no |
| **4a commitment** | `:396-405` | `stepDriftCommitment(state.drift.committed, ...)` -> `nextCommitted.slip` (the latched command) | **sets up** the slide |
| **4b committed chase** | `:407-416` | `computeNextLateralVelocity(nextCommitted, ...)` -> `nextLateralVelocity` | **yes** (the divergence) |
| **4c speed scrub** | `:419-427` | `computeLongitudinalSpeedScrub(state.slipAngle, nextSpeed, speedPercent)` -> `scrubbedSpeed` | **yes** (the traction loss) |
| **4c-bis off-road floor** | `:436-444` | lifts the post-scrub speed to the off-road floor | indirectly |
| **4d published slip** | `:447-450` | `nextSlipAngle = computeSlipAngle(nextLateralVelocity, nextSpeedAfterTraction)` | - |
| **4e pose spring** | `:454-459` | `stepDriftPoseSpring(...)` -> the cosmetic yaw accumulator | **NO - this is the visual, not the motion** |
| **5 translate** | `:462-475` | `velocityHeading = nextHeading + nextSlipAngle`; `x/y += (cos, sin)(velocityHeading) * nextSpeedAfterTraction * dt` | **yes - THIS is where the drift moves the car** |
| publish | `:477-484` | `heading`, `speed`, `slipAngle`, `lateralVelocity`, `drift` | - |

**MEASURED (runtime, the real `stepPhysics`, 60 Hz, road surface, `throttle=1, steer=1` from a 89 m/s straight line):**

```
tick | speed | committed.slip | published.slip | lateralVel | pose | heading
  1  | 89.000 |    -0.0583    |    -0.0181     |   -1.609   | -0.0002 | 0.0333
  4  | 88.928 |    -0.2333    |    -0.1315     |  -11.761   | -0.0059 | 0.1333
 30  | 87.679 |    -0.4500    |    -0.4507     |  -42.424   | -0.1418 | 0.9944
120 | 83.722 |    -0.4500    |    -0.4507     |  -40.511   | -0.1497 | -2.4073
300 | 80.830 |    -0.4500    |    -0.4502     |  -39.062   | -0.1523 | 3.1179

MAX |published slip|   = 0.4512 rad = 25.85 deg
MAX |committed slip|  = 0.4500 rad
MAX |pose|            = 0.1523 rad =  8.73 deg
MAX |lateralVelocity| = 42.442 m/s
MAX realized travel-vs-heading separation (measured from the x/y integration) = 0.4199 rad = 24.06 deg
```

**MEASURED scrub:** straight-line top speed `89.000 m/s`; after 300 ticks of a full-steer drift `80.830 m/s` - a loss of **8.170 m/s = 9.2 %**. The steady-state scrub term is `0.55 * 0.4502 * 80.83 * (80.83/89) = 18.18 m/s^2` against an engine accel of `89/5 = 17.8 m/s^2` (`physics.step.service.ts:149`), so the car settles on a drift terminal speed of ~80.8 m/s.

**Conclusion of 2.A**: **the drift law is NOT absent and the motion is NOT unwired inside the physics.** The slip produces a 24-26 degree travel-vs-heading separation and a 42 m/s lateral divergence; the scrub removes 9.2 % of the speed. Both are real, and both land in the car's `x`/`y`.

**So the "no drifting" symptom is NOT in the physics.** It is downstream. That is the finding.

### 2.B - THE EXACT BREAK LINE: the render bridge is built from the NOSE, and the travel direction is a discarded local  *(confidence 0.97, source: static-code + runtime)*

The travel direction is computed at **`physics.step.service.ts:462`**:

```ts
const velocityHeading = nextHeading + nextSlipAngle;
```

and is used at `:469` and `:474` to advance `x` and `y`. **It is then discarded.** It is a local `const`; it is never written onto the state (`:477-484`), never published, never rendered.

`PhysicsCarState` (`physics.step.service.ts:78-130`, initialized at `:296-314`) carries `heading` (**the nose**), `slipAngle` (a scalar), `lateralVelocity` (a scalar), `yawRate`, `speed`, `x`, `y`, `drift` - and **no velocity vector and no travel heading**. So the direction the car is actually going cannot be read from the state; it can only be re-derived as `heading + slipAngle`, and **nothing in the live runtime does that.**

Repo-wide grep over `examples/neatRacing/**/*.ts` (excluding `*.test.ts`) for `slipAngle|lateralVelocity` returns production hits in exactly three files:

- `physics/physics.step.service.ts` - the pipeline itself (the next tick's own reads at `:402`, `:412`, `:413`, `:423`, and the publishes at `:480`, `:481`).
- `physics/collision.utils.ts:386-389` - `resolveWorldVelocity`, the **only** place a world velocity VECTOR is assembled from `car.heading + car.slipAngle` and `car.lateralVelocity`.
- `browser-entry/browser-entry.ts:982` - `slipAngleRad: car.slipAngle` inside `window.neatRacingDriftState`, the **debug seam**.

**The render bridge reads the NOSE and the COSMETIC POSE, never the slip:**

| Line | Expression | What it binds the frame to |
|---|---|---|
| `browser-entry.ts:935-936` | `forwardX = Math.cos(car.heading)`, `sceneForwardZ = -Math.sin(car.heading)` | the **nose** |
| `browser-entry.ts:938-940` | `cameraWorld = car.x/y - 5.5 m * noseDir` | the **nose** (re-derived from the car's position every frame) |
| `browser-entry.ts:963` | `fillCarRenderPose(..., carSceneYawRad(car.heading), ...)` | the **nose** |
| `browser-entry.ts:972` | `carPoseBag.driftYawOffsetRad = car.drift.pose.pose` | the **cosmetic pose spring** |
| `browser-entry.ts:1000` | `cameraPose.yawRadians = carPoseBag.yawRad` | the **nose** |
| `browser-entry.ts:1010-1012` | `target = car.x/y + 8 m * noseDir` | the **nose** |
| `car.pass.ts:1608` | `rotationYMat4(NEATRACING_CAR_YAW_MATRIX, car.yawRad)` | the **nose** |
| `car.pass.ts:1614-1620` | body yaw = base yaw + `driftYawOffsetRad` | the **cosmetic pose** |
| `car.pass.ts:1415-1425` | shadow yaw = `car.yawRad + driftYawOffsetRad` | the **cosmetic pose** |

**Why this erases the drift (the mechanism, MEASURED):**

1. The camera is re-placed **every frame from the car's current position** (`browser-entry.ts:938-940` uses `car.x` / `carSceneZ`), along the nose. A rigid nose-frame therefore **follows the car's off-axis travel perfectly**: the 25.8-degree off-axis motion produces **no screen-space translation at all**. The one cue a viewer would normally read as "sliding" is cancelled by construction.
2. The camera's **aim** is the nose (`browser-entry.ts:1000`, `:1010-1012`).
3. The **body's** rendered yaw is the nose **plus the cosmetic pose** (`browser-entry.ts:972` -> `car.pass.ts:1614-1620`), and the pose is a second-order spring on `0.5 * COMMITTED slip + 0.08 * articulation` (`drift.model.ts:730-770`). MEASURED settled value: **8.73 deg**, i.e. **33.8 %** of the physical 25.79 deg separation. The articulation term (`+0.08 * 0.9 = +0.072 rad`) partially **cancels** the slip term (`-0.5 * 0.45 = -0.225 rad`).
4. The published `car.slipAngle` - the number that actually describes the slide - is carried across the bridge and **dropped into a debug object** at `browser-entry.ts:982`.

**MEASURED net:** the angle between the car's **rendered heading** and its **actual travel direction** is `25.79 - 8.73 = 17.06 deg`, and **not one degree of it is expressed by the camera or by the body yaw**. The entire physical drift collapses into an 8.73-degree cosmetic crab in a frame that is rigidly attached to the nose.

**This is the user's sentence, mechanically:** *"It's just a poorly implemented camera trick, that has no connection with the drifting."* The camera trick is real, and the connection to the drifting is severed at exactly two lines: **`browser-entry.ts:963`** (the body yaw is bound to `car.heading`) and **`browser-entry.ts:972`** (the only body-yaw offset is the pose spring, not the slip). The break line proper - where the physical quantity stops existing - is **`physics.step.service.ts:462`**, where `velocityHeading` is computed and dropped.

**A non-finding worth recording so the implementation wave does not "fix" a non-bug:** the pose **sign** is coherent with the travel direction (a left turn gives a negative committed slip, a negative pose, and a negative travel separation - `drift.model.ts:735-737` explicitly documents the `+committedSlip` sign repair). There is **no sign inversion** in the render path. The defect is **magnitude and provenance** (a 0.5-gain spring on the committed command instead of the published travel direction), not sign.

### 2.C - The lateral velocity divergence is real, but its only world-velocity consumer is DEAD CODE in the live runtime  *(confidence 0.95, source: static-code)*

`physics.step.service.ts:407-416` produces a genuine divergence: MEASURED peak `42.442 m/s`, steady `-39.062 m/s` at 80.8 m/s forward.

The **only** production consumer of that divergence as a world velocity is `collision.utils.ts:384-390`:

```ts
vx = Math.cos(car.heading + car.slipAngle) * car.speed - Math.sin(car.heading) * car.lateralVelocity,
vy = Math.sin(car.heading + car.slipAngle) * car.speed + Math.cos(car.heading) * car.lateralVelocity,
```

This is the correct decomposition of a sliding car into a world velocity - and **it is never executed in the live runtime.** A repo-wide search (all `*.ts`/`*.tsx`/`*.js`/`*.mjs` outside `node_modules`, `dist`, `docs/assets`, the bundles and `coverage`) for `resolveCarCollisions|reboundPastApronEdge` returns hits in exactly **two** files:

- `physics/collision.utils.ts` - the definition (`:197`), the internal `reboundPastApronEdge` call (`:250`), the JSDoc example (`:191`), and the private helper (`:465`).
- `physics/collision.utils.test.ts` - the test call sites (`:245`, `:270`, `:284`, `:292-293`, `:502`, `:517`, `:531`).

`browser-entry.ts:83` imports **only** `createTrackSurfaceResolver` (used at `:733`). **`resolveCarCollisions` is never imported by any production module.**

Consequences: (a) the live car never bounces and never has its drift channel zeroed by a bounce (`collision.utils.ts:406-414`, `:428-458` are dead in the live path); (b) the world-velocity algebra that would make a bounce respect the slide is never applied; (c) the car drives through walls with only the off-road surface decel to slow it.

### 2.D - The speed scrub IS consumed (this is NOT a discard) - but it is consumed as a bare scalar with no traction event  *(confidence 0.96, source: static-code + runtime)*

The scrub is computed at `physics.step.service.ts:419-427`, is floored at `:436-444`, **is published as the tick-end speed** at `:478` (`speed: nextSpeedAfterTraction`), **and is used to translate the car** at `:469` / `:474`. MEASURED: the car really does lose 9.2 % of its top speed in a sustained drift, and the HUD shows the derated number (`host/race-session.ts:261-264` -> `state.car.speed`).

**Correction to the sibling wave's wording** (`14N-V5-audit-visuals` G-V2, which calls the scrub "computed and discarded"): the scrub is **not** discarded - it is applied to the speed and to the translation. What does not survive is the **scalar**: no traction-loss value is placed on the state, so no consumer can read *how much* traction was lost, only the resulting slower speed. This audit records the correction because the motion question ("does the scrub change how the car moves?") turns on it: **yes, it does - by 9.2 % - and that is the ONLY place the traction law is observable in the live runtime.**

### 2.E - THREE different slip values coexist in one tick  *(confidence 0.95, source: static-code)*

| Value | Where it is produced | Where it is consumed |
|---|---|---|
| the **committed** slip (a latched COMMAND) | `drift.model.ts:484-560` -> `physics.step.service.ts:396` | the chase target `tan(slip) * speed` (`drift.model.ts:690`) and the pose target (`physics.step.service.ts:457`) |
| the **published** slip (a MEASUREMENT) | `physics.step.service.ts:447-450` (`atan2(nextLateralVelocity, nextSpeedAfterTraction)`) | the scrub (`:423`), the next tick (`:402`, `:413`), the debug seam (`browser-entry.ts:982`) |
| the **scrub input** | the tick-start published slip, `state.slipAngle` (`:423`) | the traction law |

Consequences: (a) the traction loss is **one tick (16.7 ms) behind** the slide that causes it; (b) the published slip is computed from the **post-scrub** speed, so the scrub lowers the denominator, which inflates `|atan2|`, which increases the scrub - a **positive feedback loop** (MEASURED: published 0.4502 vs committed 0.4500 at the steady state, a small but real self-amplification); (c) the pose reads the committed command while the traction law reads the published measurement, so the **visual and the traction law can disagree** (this is the sibling wave's G-V3, re-confirmed here from the motion side, and it is why the body can crab at full envelope with the tires losing little).

### 2.F - The slide magnitude is set by the STEER AXIS, not by the speed  *(confidence 0.94, source: static-code + runtime)*

- `drift.model.ts:567-580` `computeGripBudgetFactor(slipAngle, speedPercent)` **ignores** `speedPercent` - the parameter is explicitly eslint-disabled-unused at `:572-573` ("the law is speed-INDEPENDENT (G-2)"). The grip budget is `1 / (1 + 2 * max(0, |slip| - 0.2))`.
- `drift.model.ts:201` `NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S = 3.5` is a constant rate in rad/s - speed-independent.
- `drift.model.ts:66` `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45` clamps the commit.
- The **only** speed-coupled term in the entire drift channel is the scrub (`drift.model.ts:618-646`).

**MEASURED (a 300-tick sweep from 89 m/s, same surface):**

```
steer=0.30 -> max |published slip| = 0.0000 rad   (NO DRIFT)
steer=0.45 -> max |published slip| = 0.0000 rad   (NO DRIFT)
steer=0.49 -> max |published slip| = 0.0000 rad   (NO DRIFT)
steer=0.50 -> max |published slip| = 0.4512 rad   (25.8 deg)
steer=0.55 -> max |published slip| = 0.4512 rad   (25.8 deg)
steer=0.70 -> max |published slip| = 0.4512 rad   (25.8 deg)
steer=1.00 -> max |published slip| = 0.4512 rad   (25.8 deg)
```

Two things fall out: (a) the engage gate at `drift.model.ts:189` (`NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD = 0.5`, enforced at `:515-517` together with `reArmed` and `slip === 0`) is a **binary cliff at exactly 0.5** - there is no partial slide; (b) once past the gate the slide magnitude is **completely independent of the steer magnitude and of the speed** - 0.50 and 1.00 give the identical 0.4512 rad. The "loss of traction" is therefore a **steer-axis step function plus a speed-squared drag term**, not a tire model.

### 2.G - The live default mode is `ai`, and the autopilot's proportional steer barely reaches the engage gate  *(confidence 0.90, source: static-code + runtime; CROSS-WAVE to `14N-V5-audit-input` / `14N-V5-audit-envelope`)*

- `host/mode-selector.ts:23` - the served page must open as a running race, i.e. in the **AI autopilot** mode.
- `host/race-session.ts:235-240` - in `ai` mode the input comes from `deps.sampleAutoInput(state.car)`, not from the keyboard.
- `physics/ai.autopilot.ts:153-181` - `steer = clampSteer(headingError * 2.2)` (`NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD = 2.2`, `:32`). The engage gate needs `|steer| >= 0.5`, i.e. **|headingError| >= 0.227 rad (13 degrees)**.

Steady-state analysis: to hold a radius `R` the autopilot needs a yaw rate of `v/R`, and the yaw rate law (`physics.step.service.ts:192` `STEERING_RATE = 2`, scaled by `speedPercent = v/89`) gives `yawRate = steer * 2 * v/89`, so `steer = 44.5 / R` **independent of speed**. The drift therefore engages only in corners **tighter than R = 89 m** (plus whatever the off-line error contributes), and otherwise the slip is **exactly 0.0000 rad**.

The **motion** consequence (this audit's lane; the chain itself is the input wave's): on the live page the car is in one of exactly two states - a 25.8-degree slide, or **literally zero** traction loss. A player who "takes a turn at full speed" and sees nothing is in the second state, and no intermediate behaviour exists to hint that the feature is present.

### 2.H - The MIN_SPEED gate is a hard cliff, and the off-road floor gate is one tick stale  *(confidence 0.93, source: static-code)*

- `drift.model.ts:184` `NEATRACING_DRIFT_MIN_SPEED_MPS = 15`; applied at `:495` (the commitment - below it the commit is released and `reArmed` is cleared) and at `:679` (the chase - below it the lateral velocity is **exactly 0**). There is no fade-in: the slide jumps from nothing to the full 3.5 rad/s entry ramp at 15 m/s.
- `physics.step.service.ts:524-546` `isOffRoadSpeedFloorEngaged(surface, input, state.speed)` reads the **tick-start** speed; the same gate then lifts the **post-scrub** speed at `:436-444`. A car that scrubs below the 22.25 m/s floor on the road and then rolls onto dirt gets the floor decided by the previous tick's speed - a one-tick lag in the traction regime.

### 2.I - NON-FINDING: the served bundle is current - the break is NOT a staleness break  *(confidence 0.99, source: static-code + runtime)*

`examples/neatRacing/index.html:97-100` selects the bundle by path:

```js
var bundlePath = repoPath
  ? '../../docs/assets/neatracing.bundle.js?v=20260928-1'
  : '../../assets/neatracing.bundle.js?v=20260928-1';
```

Verified: **both** files exist, are **byte-identical** (SHA-256 `3C255A7F91A99AA8B3C7172B2BADCD23426DC63746A286E81778A16E33FFD9D8`, 398,699 bytes, 10/27/2026 09:56:21), and the bundle's `.map` `sourcesContent` for `drift.model.ts` (30,224 B), `physics.step.service.ts` (25,851 B) and `browser-entry/browser-entry.ts` (49,377 B) are **byte-identical to the on-disk sources**. No source file is newer than the bundle (the only newer artifact is `editor/editor.bundle.js`, same timestamp).

**Conclusion: the live page executes the audited code. The break is a design break, not a build break.**

### 2.J - NON-FINDING: the physics fixture reproduces the live behaviour, so no existing test can see the break  *(confidence 0.96, source: static-code + runtime)*

`physics/physics.step.service.test.ts:626-631` asserts `Math.abs(drifted.slipAngle) > 0.4` from `runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road')`. The live-equivalent run reproduces `0.4512 rad`. The fixture and the live path agree on the physics, so **every drift test in the repo passes while the feature is invisible to the player** - the break lives entirely downstream of the physics state, where there is no assertion at all.

---

## 3. Decision

**The drift is NOT unwired in the physics. It is severed in the render bridge, and the severed quantity is never carried by the state.**

The chain, end to end:

```
steer axis (|steer| >= 0.5, cliff)
   -> committed slip            drift.model.ts:484-560     (a COMMAND, clamped at 0.45)
   -> lateral velocity          drift.model.ts:676-712     (MEASURED 42.4 m/s)
   -> published slip            physics.step.service.ts:447
   -> speed scrub               physics.step.service.ts:419-427 (MEASURED 9.2 % of the top speed)
   -> travel heading            physics.step.service.ts:462      <-- COMPUTED, USED AT :469/:474, THEN DISCARDED
   -> car x/y                   physics.step.service.ts:467-475   <-- the drift DOES move the car
   ------------------------------------------------------------------------------------
   x  published slip has NO state field and NO render consumer
   x  the state carries the NOSE heading, not the travel direction
   x  the camera is bound to the nose (position AND aim)         browser-entry.ts:935-940, :1000, :1010-1012
   x  the body yaw is bound to the nose + a 0.5-gain pose spring  browser-entry.ts:963, :972 -> car.pass.ts:1614-1620
   x  the only live read of car.slipAngle is a debug object       browser-entry.ts:982
```

**The exact break line is `examples/neatRacing/physics/physics.step.service.ts:462`.** That is where the physical travel direction exists and dies: `velocityHeading` is the last expression in the codebase that knows the direction the car is actually going, and it is a local `const` that is never stored, published, or rendered.

**The three named wirings, answered:**

| Named wiring | Verdict | Evidence |
|---|---|---|
| **lateral velocity divergence** | **PRESENT in the physics, ABSENT from the runtime** | produced at `physics.step.service.ts:407-416` (MEASURED 42.4 m/s); its only world-velocity consumer `collision.utils.ts:384-390` is **dead code** (`resolveCarCollisions` is imported by no production module) |
| **travel-vs-heading separation** | **PRESENT in the physics, ABSENT from the state and from the render** | `physics.step.service.ts:462` (MEASURED 24.06 deg realized); the state has no travel heading (`:78-130`, `:477-484`); the render reads the nose (`browser-entry.ts:935-940`, `:963`, `:1000`, `:1010-1012`) |
| **scrub consumption** | **PRESENT and REAL** | `physics.step.service.ts:419-427` -> `:436-444` -> `:478` -> `:469`/`:474` (MEASURED 89.000 -> 80.830 m/s = 9.2 %); but no traction scalar survives, and the HUD is its only observer (`race-session.ts:261-264`) |

**What the implementation wave must CREATE (not merely connect):**

1. **A travel direction on the state.** The state must carry the direction the car is going (or the velocity vector), because no consumer can re-derive it correctly and cheaply, and because the camera and the body both need it. This is a **contract change** to `PhysicsCarState`.
2. **A camera that can express the divergence.** The camera is currently a rigid nose-frame re-derived from the car's position every frame - it cancels the off-axis motion by construction. A camera that follows the **travel** direction (or blends nose and travel) is required before any drift is visible.
3. **A body yaw that expresses the published slip**, not the committed command through a 0.5-gain spring with a partially cancelling articulation term. The `0.5` gain and the `+0.08 * articulation` term are the *measured* reason the body shows 8.73 deg where the slide is 25.79 deg.
4. **One shared slip/traction scalar** for the seam and for any traction visual (cross-referenced from the visuals wave's G-V8 and this audit's 2.D) - the scrub's *magnitude* is currently unobservable.

**What must NOT be "fixed":** the sign of the pose (2.B - coherent), the scrub's application (2.D - real and applied), the committed-slip clamp, the 0.5 threshold, the 3.5 rad/s ramp, and the `0.55*|slip|*speed^2/89` scrub formula - all settled.

---

### 3.A - The twelve numbered gaps, with the exact break line marked

Each gap is classified against the Spec-Kit gap types (missing / partial / contradicts / non-finding) and carries its own confidence and provenance. `**BREAK LINE**` marks the exact seam where the committed slip stops changing how the car moves.

| # | Gap | Type | Conf. | Break site / file:line |
|---|---|---|---|---|
| **G-M1** | **THE BREAK LINE.** `velocityHeading` - the last expression in the codebase that knows the direction the car is actually going - is a local `const` that is used to advance x/y and then **DISCARDED**: never stored on the state, never published, never rendered. | missing | 0.97 | **`examples/neatRacing/physics/physics.step.service.ts:462`** (used at `:469`, `:474`; discarded at `:475`) |
| **G-M2** | The **state cannot express the travel direction**. `PhysicsCarState` carries the NOSE heading, `slipAngle` and `lateralVelocity` as scalars, and **no velocity vector and no travel heading**; no production code re-derives them. Fixing the camera therefore requires a **contract change**, not a wiring change. | missing | 0.97 | `physics.step.service.ts:78-130` (contract), `:296-314` (init), `:477-484` (publish) |
| **G-M3** | The **render bridge is bound to the nose and the camera cancels the drift by construction**: the camera is re-placed every frame from the car's position along the nose, its aim is the nose, and the body yaw is the nose + a cosmetic pose spring. The measured 25.85 deg off-axis travel produces **no screen-space translation** - the user's "poorly implemented camera trick" in mechanical form. | missing | 0.96 | `browser-entry/browser-entry.ts:935-936`, `:938-940`, `:963`, `:972`, `:1000`, `:1010-1012` -> `engine/passes/car.pass.ts:1608`, `:1614-1620`, `:1415-1425` |
| **G-M4** | The **published slip has exactly ONE production reader and it is a debug object** (`window.neatRacingDriftState`). The number that actually describes the slide is carried across the render bridge and dropped into a debug field. | missing | 0.96 | `browser-entry/browser-entry.ts:982`; producers `physics.step.service.ts:480-481`; the only other non-test readers are the next tick's own `:402`, `:412-413`, `:423` and the dead `collision.utils.ts:386-389` |
| **G-M5** | The lateral divergence's **only world-velocity consumer is DEAD CODE** in the live runtime: `resolveCarCollisions` is imported by **no production module** (only `createTrackSurfaceResolver` is wired), so the live car never bounces, never has its drift channel zeroed, and drives through walls. The divergence is measured (42.44 m/s peak, 39.06 m/s steady) but reaches no world velocity. | missing | 0.95 | `physics/collision.utils.ts:197` (`resolveCarCollisions`), `:384-390` (`resolveWorldVelocity`), `:406-414`, `:428-458`; wired at `browser-entry/browser-entry.ts:83`, `:733` |
| **G-M6** | **CORRECTION to the sibling AUDIT-VISUALS G-V2.** The **speed scrub IS consumed, not discarded**: computed, floored, published as `state.speed`, and translated with. Measured 89.000 -> 80.830 m/s = **9.2 %** of the top speed. What does not survive is the **scalar** - no traction value reaches the state - so a consumer can read only the resulting slower speed. The HUD is its only live observer. | partial | 0.96 | `physics.step.service.ts:419-427` -> `:436-444` -> `:478` -> `:469`/`:474`; observer `host/race-session.ts:261-264` |
| **G-M7** | **Three slip values coexist in one tick**: the committed COMMAND, the published MEASUREMENT, and the scrub INPUT (the tick-start published slip). The traction law is one tick (16.7 ms) behind the slide that causes it, and the published slip is computed from the **post-scrub** speed, so the scrub lowers the denominator, inflates `\|atan2\|` and increases the scrub (measured published 0.4502 vs committed 0.4500 - a small positive feedback). | contradicts | 0.95 | `physics/drift.model.ts:484-560` (command) -> `physics.step.service.ts:396`; `physics.step.service.ts:447` (published); `physics.step.service.ts:423` (scrub input); chase target `drift.model.ts:690`; pose `drift.model.ts:457` |
| **G-M8** | The **slide magnitude is set by the steer axis, not by the speed**: `computeGripBudgetFactor` ignores `speedPercent` (the arg is eslint-disabled-unused) and the entry rate 3.5 rad/s is speed-independent. Measured sweep from 89 m/s: steer 0.30/0.45/0.49 -> **0.0000 rad**; steer 0.50/0.55/0.70/1.00 -> **0.4512 rad, identical**. The tires never degrade with speed. | contradicts | 0.95 | `physics/drift.model.ts:567-580` (budget), `:572-573` (unused arg), `:201` (entry rate), `:66` (clamp 0.45); the only speed-coupled term is the scrub `:618-646` |
| **G-M9** | The **engage gate is a binary cliff at exactly \|steer\| = 0.5** with no partial slide and no fade. Measured: 0.49 -> 0.0000 rad, 0.50 -> 0.4512 rad. A player who steers at 0.49 sees literally nothing. | missing | 0.94 | `physics/drift.model.ts:189` (threshold), `:515-517` (enforcement, `reArmed`) |
| **G-M10** | **The served page opens in `ai` mode** and the autopilot steers proportionally, so it rarely reaches the gate (needs \|headingError\| >= 0.227 rad; steady-state steer = 44.5/R, i.e. the drift engages only in corners tighter than R = 89 m). Motion consequence: on the live page the car is either in a 25.8 deg slide or in **literally zero** traction loss - there is no in-between behaviour. | missing | 0.90 | `host/mode-selector.ts:23`; `host/race-session.ts:235-240`; `physics/ai.autopilot.ts:153-181` (gain at `:32`); yaw law `physics.step.service.ts:192` |
| **G-M11** | The **MIN_SPEED gate is a hard cliff** (no fade-in) and the **off-road floor gate is one tick stale**: `isOffRoadSpeedFloorEngaged` reads the TICK-START speed and then guards the POST-SCRUB speed. | partial | 0.93 | `physics/drift.model.ts:184`, `:495`, `:679`; `physics.step.service.ts:524-546` -> `:436-444` |
| **G-M12** | **NON-FINDING - the served bundle is current, so the break is a DESIGN break, not a build break.** Both served copies are byte-identical (SHA-256 `3C255A7F...FFD9D8`, 398,699 bytes) and the bundle map's `sourcesContent` for `drift.model.ts`, `physics.step.service.ts` and `browser-entry.ts` matches the on-disk sources byte for byte. **Corollary:** the fixture asserts `\|slip\| > 0.4` and the live-equivalent run measures 0.4512, so every drift test **passes** while the feature is invisible - the break lives entirely downstream of the physics state, where there is no assertion at all. | non-finding | 0.99 | `examples/neatRacing/index.html:97-100`; `physics/physics.step.service.test.ts:626-631` |

---

## 4. Risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| R-M1 | Implementing the fix by re-pointing the **camera** at the travel direction without adding a state field leaves the body yaw reading the pose (33.8 %) and the frame internally inconsistent (a travel-locked camera with a crabbing body). | HIGH | Land the state field and the camera change in the same slice; the body yaw must read the published slip, not the pose. |
| R-M2 | The **0.5 engage cliff** (2.F) means the feature is binary. Once the camera is fixed, a user who steers at 0.49 still sees literally nothing, and will report "still no drift". | HIGH | Decide deliberately: keep the cliff (and say so) or add a proportional entry ramp. The `14N-V5-audit-envelope` wave owns the numeric envelope. |
| R-M3 | The **ai-mode default** (2.G) means the *served* page may still show no drift after the fix, because the autopilot rarely reaches `|steer| >= 0.5`. | MEDIUM | Verify with a manual-mode full-steer run first; do not accept the ai-mode page as the acceptance evidence. |
| R-M4 | The live path has **no collision resolution at all** (2.C). A drift fix will make the car slide - and slide straight through walls, because `resolveCarCollisions` is never called. | MEDIUM | Wire `resolveCarCollisions` (or state explicitly that walls are out of scope for this slice). |
| R-M5 | The scrub's **one-tick staleness and self-amplifying feedback** (2.E) will become visible once the drift is visible; a reviewer may misread the rising scrub as a runaway. | LOW | Read the published slip for the scrub, or document the lag explicitly. |
| R-M6 | The **MIN_SPEED cliff at 15 m/s** (2.H) means the slide appears abruptly; on the ai-mode track (which slows for corners) the car may cross 15 m/s mid-corner and pop into a slide. | LOW | Observe in a manual run; the envelope wave owns the fade decision. |
| R-M7 | This audit is the only wave with **runtime** numbers; the sibling waves' static-code rows were written without them and one of them (visuals G-V2) mis-describes the scrub as discarded. | MEDIUM | The merge must apply the 2.D correction: the scrub IS applied to the speed and to the translation. |

---

## 5. Evidence index

**Files read (production, read-only):** `examples/neatRacing/physics/physics.step.service.ts` (L78-130 the state contract, L142-195 the constants, L290-320 the initial state, L355-500 the tick pipeline, L505-580 the helpers) · `examples/neatRacing/physics/drift.model.ts` (L1-140 the sign convention + the constants, L480-600 the commitment + the grip budget, L600-800 the scrub + the chase + the pose spring + `computeSlipAngle`) · `examples/neatRacing/browser-entry/browser-entry.ts` (L83 the collision import, L112-152 the seam interface, L519-527 the camera constants, L557-585 the scene yaw/position helpers, L733 the surface resolver, L920-1070 the render bridge, L1149 the `stepPhysics` seam) · `examples/neatRacing/engine/passes/car.pass.ts` (L435-497 the pass state, L1405-1430 the shadow yaw, L1595-1630 the body yaw) · `examples/neatRacing/host/race-session.ts` (L210-275 the live loop) · `examples/neatRacing/host/input.ts` (the whole file - the keyboard axis map) · `examples/neatRacing/host/mode-selector.ts` (L23 the default mode) · `examples/neatRacing/physics/ai.autopilot.ts` (L20-60 the constants, L140-185 the steer law) · `examples/neatRacing/physics/collision.utils.ts` (L155-260 the resolver + `resolveCarCollisions`, L378-460 the world velocity + the drift zeroing) · `examples/neatRacing/index.html` (L80-140 the bundle path) · `examples/neatRacing/physics/physics.step.service.test.ts` (L575-660 the drift fixtures).

**Repo-wide greps:** `slipAngle|lateralVelocity` (non-test) · `drift\.pose|driftYawOffsetRad|car\.heading|velocityHeading|car\.yawRad` (non-test) · `resolveCarCollisions|reboundPastApronEdge` (all `*.ts`/`*.tsx`/`*.js`/`*.mjs`, excluding `node_modules`, `dist`, `docs/assets`, the bundles and `coverage`).

**Runtime method (read-only, no browser):** the repository's own `physics.step.service.ts` was bundled with the repo's `esbuild` into `$TEMP` (outside the repository) and driven headlessly at the real 60 Hz fixed timestep with the real `stepPhysics`; the repository working tree was not modified. Two runs: (1) a 900-tick straight-line spin-up to the 89 m/s top speed followed by 300 ticks of `throttle=1, steer=1`; (2) a 7-point steer sweep (0.30 / 0.45 / 0.49 / 0.50 / 0.55 / 0.70 / 1.00), 300 ticks each from the same 89 m/s start.

**Git state (read-only inspection only; no mutating command was run):** the 14N drift work is **uncommitted** - `git diff --stat` shows `drift.model.ts` (+917/-), `physics.step.service.ts` (+179), `car.pass.ts` (+52), `collision.utils.ts` (+36) plus test churn, on top of HEAD `023a45b6 14g`. The audit's line numbers are cited against the **working tree**, which is what the served bundle was built from (2.I).

**No production byte was touched.** The only writes are this artifact and the `PlanUpdate` block in `plans/neatracing-voxel-racer.plans.md`.
