# Research artifacts — the neatRacing voxel-racer plan family

**Plan**: `plans/neatracing-voxel-racer.plans.md` (Phase 1 — Graphics engine + manual keyboard driving, `[WIP]`)
**Canonical home**: this file.
**Folded**: 2026-10-31, under the USER consolidation order. Seven read-only research artifacts that had
been living as sibling side files — the former canonical `plans/neatracing-voxel-racer.research.md` plus
six `plans/neatracing-voxel-racer.<topic>.research.md` siblings — are merged HERE and the side files are
DELETED. The finding IDs, the section numbers, the measured numbers, the confidences and the provenance
of every artifact are PRESERVED: nothing was re-derived, nothing was re-scoped, nothing was summarised
away. The fold changes only WHERE an artifact lives and HOW it is addressed.

**Address form.** A folded artifact keeps its original identifier and is addressed as
`plans/neatracing-voxel-racer.research.md (Part A)` … `(Part B-6)`. Every pointer that named a side
file was RE-POINTED to its new home; the pointed-to CONTENT was NOT rewritten in substance. The one
exception is a **historical naming-deviation paragraph** (each audit artifact explains, in its `0.`
provenance section, that on 2026-10-27 it was written as a sibling because the canonical path was then
occupied): that paragraph is preserved verbatim as history and is superseded by the fold note that
precedes it — read it as "this is why the file was a sibling", not as a current path claim.

**Source-of-truth order applied throughout**: `runtime/validation > static code > comments/docs > external`.
Each artifact states its own evidence class; where two artifacts touch the same constant the disagreement
is recorded inside the artifact that owns it (AUDIT-ENVELOPE `R6`, AUDIT-PHYSICS `R2`/`R-6`), never merged
away here.

---

## Merged index — the seven folded artifacts

| Part    | Artifact id                                        | Title                                                                                                    | Question it answers                                                                                                                                                                               | Status                                | Evidence class                                                                             |
| ------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------ |
| **A**   | `neatracing-voxel-racer` (the 14N coupling design) | Step 14N drift-feel integration coupling design                                                          | where the transient drift-yaw accumulator must live and how it must feed the car body matrix without breaking AC-078 / AC-079 / the DRIFT_PROXY fixture gates / zero straight-line yaw            | RESEARCH COMPLETE (closed 2026-10-24) | static code + source reads                                                                 |
| **B-1** | `neatracing-voxel-racer.audit-physics`             | AUDIT-PHYSICS — read-only code audit of the drift/traction model                                         | what traction/scrub loss ACTUALLY scales with; where speed × turning-angle coupling exists or is missing; the MIN_SPEED gate, the clamps, the latch, the D1.3 deficit form; the literal gaps      | COMPLETE (2026-10-27)                 | static code (numbers marked DERIVED are arithmetic from the source constants)              |
| **B-2** | `neatracing-voxel-racer.audit-envelope`            | AUDIT-ENVELOPE — numeric re-derivation of the drift engage envelope                                      | the NUMERIC engage/hold envelope; why the user sees zero drift; the S-1/S-2/S-3 impossibility proofs under the frozen constants                                                                   | COMPLETE (2026-10-27)                 | **RUNTIME** — the live modules bundled headlessly and executed in Node (not a browser run) |
| **B-3** | `neatracing-voxel-racer.audit-input`               | AUDIT-INPUT — live keyboard steering chain vs the fixture's scripted steer injection                     | what an interactive, full-speed, HELD turn actually feeds into the drift model on the live path; whether the engage gates ever open; how the fixture's scripted injection differs                 | COMPLETE (2026-10-27)                 | static code + served-bundle check                                                          |
| **B-4** | `neatracing-voxel-racer.audit-visuals`             | AUDIT-VISUALS — read-only code audit of the tire/traction PRESENTATION layer                             | what VISUALLY expresses losing traction, what feeds it, and which published drift outputs are unwired                                                                                             | COMPLETE (2026-10-27)                 | static code                                                                                |
| **B-5** | `neatracing-voxel-racer.audit-motion`              | AUDIT-MOTION — where the committed slip / speed scrub IS (and is NOT) applied to the car's ACTUAL MOTION | where the committed slip and the speed scrub actually reach the car's motion, and the exact line where they stop                                                                                  | COMPLETE (2026-10-27)                 | **RUNTIME** — headless, in-memory, the real 60 Hz `stepPhysics`                            |
| **B-6** | `neatracing-voxel-racer.drift-feel`                | Step 14N arcade drift references (drift transient tuning) + the 2026-10-25 spring-damper addendum        | how the classic arcade racers tune the drift transient (entry attack → hold → countersteer snap-back → overshoot → numeric rates) and the concrete constant pattern `drift.model.ts` should adopt | RESEARCH COMPLETE (closed 2026-10-25) | web / ROM-decomp source sweep + repo source reads                                          |

### Finding-id index (a number stays findable after the fold)

| Part    | Finding ids                                                                                                                                                     |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A**   | E1–E7, D1–D7                                                                                                                                                    |
| **B-1** | F-series, L-series, B1–B5, D1–D6, R1–R8, plus the `G-` literal-gap rows                                                                                         |
| **B-2** | D1–D5 (the numeric envelope, the ranked contribution table, the S-1/S-2/S-3 adjudication, the numbered gaps, the confidence summary), R1–R8, the evidence index |
| **B-3** | G-1 … G-14 (severity-ranked literal gaps, CRITICAL → LOW)                                                                                                       |
| **B-4** | G-V1 … G-V11, D1–D10, R1–R9, plus the 7-exist / 7-create wire ledger                                                                                            |
| **B-5** | G-M1 … G-M12, plus the 2.A–2.J evidence rows and the break line                                                                                                 |
| **B-6** | E1–E5, D1–D5, R1–R10 (the arcade-refs first pass) and E6.1–E6.10, D6.1–D6.10, R11–R15 (the 2026-10-25 spring-damper addendum)                                   |

> The `GF-1 … GF-16` decision rows are **not** in Part B-6 — they are the game-feel scout's DECISIVE FINDINGS
> and they live in the tracker's plan tail under `## PlanUpdate - Step 14N SCOUT game-feel`
> (`plans/neatracing-voxel-racer.plans.md`, the GF-1 row onward). They are cited, not duplicated, here.

---

## PART A — Step 14N drift-feel integration coupling design

> **Provenance.** Folded 2026-10-31 from the former canonical `plans/neatracing-voxel-racer.research.md`.
> Author role `02-researching` (Tier-1 drift-feel scout), read-only research, 2026-10-24 (Step 14N planning
> round). Sub-orchestrators used: `boundary-mapper`, `implementation-pattern-scout` (both read-only, both
> re-verified by the orchestrator against source). Status at fold: RESEARCH COMPLETE — the feel direction, the
> floor value and the AC-088 lever remained RED-slice decisions and were deliberately NOT invented here.
> Content is UNCHANGED apart from the heading demotion and the path re-pointing.

- **Plan**: `plans/neatracing-voxel-racer.plans.md` (Step 14N, `[WIP]`)
- **Step**: 14N — Arcade drift FEEL sharpening
- **Author role**: 02-researching (Tier-1 drift-feel scout), read-only research
- **Date**: 2026-10-24 (Step 14N planning round)
- **Sub-orchestrators used**: `boundary-mapper`, `implementation-pattern-scout` (both read-only, both re-verified by the orchestrator against source)
- **Status**: RESEARCH COMPLETE — the feel direction, the floor value, and the AC-088 lever remain RED-slice decisions (deliberately NOT invented here)

---

#### Question

Where should a **transient drift-yaw accumulator** live, and how must it feed the car body
matrix so that the Step 14N work does not break:

1. **AC-078** — the tire-follow pin (`TIRE_YAW_FOLLOW_TOLERANCE_RAD = 0.002` rad),
2. **AC-079** — `proportionalTractionLoss` (the 290 km/h ceiling),
3. the **DRIFT_PROXY** fixture gates (`DRIFT_PROXY_MIN_RAD = 0.01`, 3 consecutive frames, 6-frame window),
4. **zero straight-line yaw**.

Plus: state ownership, tick order, and which constants move.

---

#### Evidence

##### E1 — The drift-yaw production chain (verified in source)

| Stage     | Location                                                      | Behaviour                                                                                       |
| --------- | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Constant  | `examples/neatRacing/physics/drift.model.ts:97`               | `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD = 0.5`                                                    |
| Producer  | `examples/neatRacing/physics/physics.step.service.ts:416-417` | `nextDriftYawOffset = nextSlipAngle * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`                     |
| Return    | `physics.step.service.ts:440`                                 | returned from `stepPhysics` (pipeline at `:365-450`)                                            |
| Field     | `physics.step.service.ts:108`                                 | `PhysicsCarState.driftYawOffset` (spawn init `0` at `:312`)                                     |
| Transport | `examples/neatRacing/browser-entry/browser-entry.ts:941`      | `carPoseBag.driftYawOffsetRad = car.driftYawOffset` — the **only** production read of the field |
| Shadow    | `examples/neatRacing/engine/passes/car.pass.ts:1365`          | Euler sum `car.yawRad + (car.driftYawOffsetRad ?? 0)`                                           |
| Body      | `car.pass.ts:1565-1573`                                       | matrix product `T · Ry(yaw) · Ry(drift)`                                                        |
| Wheels    | `car.pass.ts:1598`                                            | wheel chain parented onto `NEATRACING_CAR_BODY_MODEL_MATRIX`                                    |

`drift.model.ts:363` `computeDriftYawOffset()` is a **dead export** — production inlines the
multiplication, so the "shape" of the body animation is literally `slipAngle × gain`.

Because `Ry(yaw)` and `Ry(drift)` share the same world-up axis, `Ry(a)·Ry(b) ≡ Ry(a+b)`; the
shadow's Euler sum and the body's matrix product agree **exactly** today. Any change that keeps
a single scalar offset preserves that identity.

`examples/neatRacing/engine/passes/glass.pass.ts:1048,1086` reads `car.yawRad` /
`steerAngleRad` but **not** `driftYawOffsetRad` — the glass canopy stays on base yaw. This is a
pre-existing inconsistency, but feel work amplifies it (the body will visibly yaw away from the
canopy). Flagged, not owned by Step 14N.

##### E2 — The DRIFT_PROXY is measured from the **rendered body matrix** vs the travel chord

`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:789-822`:

- `bodyYawRad = yawFromMatrixColumn3(m)` — the yaw column of the **rendered car body model matrix**;
- the window's oldest/newest positions give a world-XZ **chord** direction;
- `proxyRad = |wrapAngle(endpointMidYawRad - offsetAngle(chordDx, chordDz))|`.

The endpoint-anchored window mean cancels the mid-window lag exactly; the residual is bounded by
`alpha·T²/8` plus a speed-variation term ≤ 0.002 rad. `maxDriftProxyRad` is the **max** over the
run; `driftProxyRunFrames` counts **consecutive** frames ≥ `DRIFT_PROXY_MIN_RAD = 0.01` and
resets to 0 on any frame below it.

##### E3 — DECISIVE: the drift proxy is a **V-shaped** function of the visual gain

The physics translates the car along `heading + slipAngle` (`physics.step.service.ts`, step 4c);
the rendered body yaw is `heading + driftYawOffset`. Therefore

```
proxy  = |bodyYaw - travelDirection|
       = |(heading + slip*GAIN_eff) - (heading + slip)|
       = |slipAngle| * |GAIN_eff - 1|
```

The recorded Step 14G baseline is `0.049388550574608114 rad` at `GAIN = 0.5`, which implies
`|slipAngle| = 0.049388550574608114 / 0.5 = 0.098777101149216228 rad` at that operating point.
The baseline is _exactly_ `0.5 × |slip|` — the identity above is confirmed against measured data.

| `GAIN_eff`       | proxy (rad, at `\|slip\| = 0.09878`) | meaning                                   |
| ---------------- | ------------------------------------ | ----------------------------------------- |
| 0.00             | 0.0988                               | body on the heading — **no slide at all** |
| **0.50 (today)** | **0.0494**                           | body halfway to the travel direction      |
| 0.75             | 0.0247                               | slide _shrinks_                           |
| **1.00**         | **0.0000**                           | **NULL — the slide becomes invisible**    |
| 1.25             | 0.0247                               | slide grows again, on the other side      |
| 1.50             | 0.0494                               | equals today's baseline                   |
| 2.00             | 0.0988                               | 2× baseline                               |

**Consequences the RED slice must internalise before pinning anything:**

- The proxy is **non-monotonic** in the gain with a **null at `GAIN_eff = +1.0`**. A naive
  "raise the gain to feel more drift" move to 1.0–1.25 makes the slide _weaker_, and at exactly
  1.0 the slide vanishes. Any floor pinned against the gain must record the forbidden null.
- Raising the gain **above** 1.0 is a _different visual branch_ from the current one: the body
  rotates past the travel direction instead of sitting short of it.
- The current `GAIN = 0.5` means the body yaw sits at only **half** the physical slip. The crisp
  arcade-drift look (nose further into the turn than the path) is the branch where the body is
  _further_ from the travel direction than the heading is — i.e. an **effective gain below the
  current 0.5** (or negative), which is the direction that **increases** the proxy monotonically
  and never crosses the `+1.0` null.

##### E4 — State ownership and tick order

**Ownership.** `PhysicsCarState` (`physics.step.service.ts:73-112`) owns every drift field.
`NeatRacingRaceSessionState.car` (`examples/neatRacing/host/race-session.ts:81-89`) holds the
instance; each tick **replaces the record wholesale** (immutable record replacement), so any new
field must be carried into the new record. There is **no worker / `postMessage` boundary today** —
everything runs on the browser main thread (the spec's "simulation worker" is a Phase-2
aspiration), so no structured-clone or transfer-list concerns apply to a new numeric field.

**Tick order** (`race-session.ts:231-269`, `host/render-loop.utils.ts:195-218`):

```
rAF
 └─ handleFrame
     ├─ advanceFrameSteps        -> 0..4 fixed ticks of 1/60 s
     ├─ input sampled once per frame
     ├─ per tick: stepPhysics  (produces driftYawOffset)
     │            stepRaceRules
     ├─ hud.update
     └─ composer.drawFrame(buildComposerFrame())
          └─ browser-entry.ts:941  -> carPoseBag.driftYawOffsetRad
               └─ carPass.draw  -> body matrix built ONCE per rendered frame
```

Physics therefore runs **before** the transform build, and the body matrix is built once per
**rendered** frame from the last tick's state. The drift offset is sampled at render rate but
**produced** at tick rate. Precedent for a per-tick accumulated visual term exists: the wheel-spin
accumulator at `browser-entry.ts:906-925`.

##### E5 — The constraints that pin the design (all verified)

| #   | Constraint                                                                                                                                                                           | Location                                | Effect on the design                                                                                                                                       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1  | `spawn.driftYawOffset === 0`                                                                                                                                                         | `physics.step.service.test.ts:156-163`  | the offset must start at 0                                                                                                                                 |
| C2  | `\|driftYawOffset\| < Math.PI / 2` (and finite, and `> 0` after 60 ticks)                                                                                                            | `physics.step.service.test.ts:162`      | the **total** offset must stay under π/2 — this is the hard bound on any gain/accumulator magnitude                                                        |
| C3  | determinism = run A vs run B of the **same** code (never a golden literal)                                                                                                           | `physics.step.service.test.ts:130-137`  | adding a field to `PhysicsCarState` is safe                                                                                                                |
| C4  | `serializeDriftState(state) === serializeDriftState(createZeroDriftState())` after 120 straight-line steps                                                                           | `drift.model.test.ts:309-327` (AC-047D) | `DriftModelState` has exactly **two** fields (`lateralVelocity`, `slipAngle`); **adding a field to it touches this byte-identity pin**                     |
| C5  | `computeLongitudinalTractionLoss(0, 60, 0.67) === 0` exactly                                                                                                                         | `drift.model.test.ts:176-183`           | straight-line traction must stay exactly 0                                                                                                                 |
| C6  | 600 straight-line full-throttle steps → `speed === MAX` and `slipAngle === 0`                                                                                                        | `physics.step.service.test.ts:247-252`  | the trajectory must not change                                                                                                                             |
| C7  | `DRIFT_VISUAL_YAW_LIMIT_RAD = NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` (= π/2 × 0.5 ≈ 0.785) and the monotonic sweep uses `[-L, -L/2, 0, L/2, L]` | `car.pass.test.ts:371-373, 479-516`     | the limit is **derived from the gain** and auto-scales; the sweep asserts the _pass_ is a faithful `Ry(drift)`, so it constrains the pass, not the physics |
| C8  | `tireYawDriftEngaged` needs ≥ 3 **consecutive** frames with proxy ≥ 0.01                                                                                                             | fixture `:876-881`                      | a transient that drives the proxy below 0.01 for even one frame **resets** the run counter and can break **both** AC-078 and AC-079                        |
| C9  | `tireYawFollowsBodyDuringDrift` = `maxRelativeYawError ≤ 0.002 && maxRelativeOffsetError ≤ 0.002`                                                                                    | fixture `:926-928`                      | the 0.002 rad pin is a **relative** measure (`wheelYaw − bodyYaw == mount.rotationYawRad`) — **invariant to the offset's magnitude**                       |
| C10 | `proportionalTractionLoss` = `tireYawDriftEngaged && 0 < maxSpeedKmhDrift < 290`                                                                                                     | fixture `:935-937`                      | a **ceiling only**, no ratio; satisfied iff the drift stays engaged and the speed stays under 290                                                          |
| C11 | `NEATRACING_WHEEL_HUB_HEIGHT_M = NEATRACING_WHEEL_WORLD_RADIUS_M = 0.5625 m`                                                                                                         | `voxel/wheel.mount.utils.ts:86`         | the tire-gradient radius is currently the **hub height**, not a dedicated constant                                                                         |

**There is NO assertion anywhere that the body yaw/heading is zero on a straight line.** The
"zero straight-line yaw" invariant is currently unenforced and must be closed by a new RED
contract.

##### E6 — DECISIVE: the AC-088 tire-gradient measurement is **blind to the shader**

`car.pass.ts:320-329` applies the tire gradient in the **fragment** shader:

```glsl
if (uApplyTireGradient > 0.5 && max(max(albedo.r, albedo.g), albedo.b) <= uTireGradientThreshold) {
  vec3 wheelCenter = vec3(uModel[3][0], uModel[3][1], uModel[3][2]);
  float t = clamp(dot(vWorld - wheelCenter, uSunDirection) / (2.0 * uTireGradientRadius) + 0.5, 0.0, 1.0);
  float factor = mix(uTireGradientDarkFactor, uTireGradientBrightFactor, t);
  albedo *= factor;
}
```

But the fixture's `analyzeWheelShading()` (fixture `:605-624`) reads the **wheel VBO**:

```js
var y = data[i * 9 + 1];                       // local mesh Y
var brightness = Math.max(r, g, b);            // VERTEX ALBEDO, not the shaded result
if (y > 0) { upperSum += brightness; ... }
else if (y < 0) { lowerSum += brightness; ... }
```

And the mesh builder says so explicitly (`car.pass.ts:555-556`):

> `// Fixed-space tire lighting is applied in the fragment shader so the`
> `// highlight follows the shared sun direction and does not rotate with the`
> `// spinning voxels (OBS-20). Vertex colors stay uniform rubber.`

Tire vertices carry the uniform base colour `NEATRACING_BAKED_TIRE_BASE_COLOR = [28, 28, 30]`
(`car.pass.ts:383-385, 542-544`). The mesh is top/bottom symmetric, so
`upperAvg === lowerAvg === 0.28565` and the measured gradient is **identically zero** — and it is
**invariant to every named AC-088 lever**, because `uTireGradientRadius`,
`uTireGradientBrightFactor` and `uTireGradientDarkFactor` are fragment-stage uniforms that this
measurement path never observes.

Secondary observation: the sun is `[0, 2, 0.5]` normalised → **97 % vertical**
(`engine/lighting.ts:62, 75-81`). The shader's gradient axis is therefore _almost_ the local Y
axis that the fixture splits on, so the measurement **axis** is roughly right; it is the
**sampling** (albedo instead of shaded output) that is wrong.

**AC-088 as currently worded is therefore not closable by tuning the three named constants.** The
RED contract must either (a) sample the shaded framebuffer instead of the VBO, or (b) be
re-scoped to the jest-level contract that already exists in
`examples/neatRacing/engine/lighting.test.ts` (`describe('fixed-space tire lighting (OBS-20)')`),
which owns the shader-source / uniform-feed contract. The jest owner of the tire gradient is
`engine/lighting.test.ts`, **not** `car.pass.test.ts` — the Step 14N RED packet names the latter.

##### E7 — Secondary risks (verified, dormant)

- `examples/neatRacing/physics/collision.utils.ts:410,417-418` zeroes `speed`, `slipAngle` and
  `lateralVelocity` on a bounce but **not** `driftYawOffset` — dormant today, latent the moment a
  transient is wired.
- Spec drift is already live: the spec's illustrative formula is
  `clamp(lateralVelocity/speed · GAIN, ±max)` while the implementation is `atan2(...) · GAIN`
  with **no clamp**.

---

#### Decision

##### D1 — The model: a **transient gain**, not an additive yaw offset

> **The accumulator is a transient, bounded, dimensionless _gain modulation_ on the drift yaw,
> and the drift-yaw offset stays exactly proportional to the slip angle.**

```
offset  = slipAngle * (NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD + transientGain)
proxy   = |slipAngle| * |GAIN_eff - 1|,   GAIN_eff = GAIN + transientGain
```

with `transientGain` integrated per tick by a **pure** step function and carried as its own
small state.

**Why this shape and not `offset = slip*GAIN + accumulatedYaw`:**

1. **Zero straight-line yaw is exact by construction.** With slip ≡ 0 the offset ≡ 0 _whatever_
   the accumulator holds. An additive accumulator would carry a **stale yaw** onto the next
   straight line and would need an explicit reset contract (and a hard-reset rule) to satisfy
   the user's zero-straight-line-yaw requirement.
2. **The "body-animation shape" is retained**, as AC-087 demands: the offset is still
   `slipAngle × gain`; only the _gain_ becomes transient. `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`
   keeps owning the body yaw (it becomes the _base_ gain of a transient gain).
3. **The proxy stays monotone in the chosen direction.** Driving `GAIN_eff` **down** (0.5 → 0.25
   → −0.5) increases `|GAIN_eff − 1|` monotonically and **never crosses the `+1.0` null**, so
   the 3-consecutive-frame drift gate (C8) and the `tireYawDriftEngaged` precondition of AC-078
   and AC-079 are never endangered. Driving the gain _up_ toward 1.0 would approach the null and
   would break C8 at the worst possible moment.
4. **The accumulator is visual-only**: it never touches `x`, `y`, `heading`, `speed`,
   `slipAngle` or `lateralVelocity`. The physics trajectory is therefore byte-identical
   (C5, C6, AC-089) and the `proportionalTractionLoss` 290 km/h ceiling (C10) is untouched.
5. **The pass is not re-authored** (AC-087), so the body matrix composition, the shadow Euler sum
   and the wheel parenting are structurally unchanged.

The **alternative** (additive yaw accumulator) is documented here for completeness but is
**not recommended**: it (a) breaks the zero-straight-line-yaw guarantee unless hard-reset,
(b) has a _null_ in the middle of its range — an additive offset equal to `slip·(1 − GAIN)`
puts the body exactly on the travel direction and the proxy collapses to 0 for a frame or two,
resetting the 3-consecutive counter, and (c) decouples the offset from the slip, so the
`car.pass.test.ts` monotonicity contract (C7) would have to be re-authored against a quantity
that is no longer slip-derived.

##### D2 — Where the accumulator lives

| Element                                | Home                                                                                                                                                 | Rationale                                                                                                                                                                                 |
| -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| The pure step function + its constants | `examples/neatRacing/physics/drift.model.ts` (new export)                                                                                            | pure, side-effect free, no `Math.random` (AC-089); unit-testable in isolation; the drift model is already the owner of the drift _shape_                                                  |
| The accumulator **state**              | a **new, separate** state type in `drift.model.ts` (e.g. `DriftTransientGainState { transientGain: number }`) — **NOT** a field of `DriftModelState` | C4: `DriftModelState` is byte-identity pinned by AC-047D after 120 straight-line steps; a separate type leaves that pin **untouched**                                                     |
| The accumulator's carrier on the car   | a new numeric field on `PhysicsCarState` (`physics.step.service.ts:73-112`), init `0` in `createInitialPhysicsCarState`                              | the physics state is the only per-car, per-tick, JSON-serializable, deterministic carrier (C3); the session replaces the record wholesale each tick, so the field must be carried forward |
| The step call site                     | `physics.step.service.ts` inside the drift step (next to `:416-417`)                                                                                 | the accumulator must advance in **tick** order, with the same `dt = 1/60` as the rest of the pipeline                                                                                     |
| The render side                        | **nothing**                                                                                                                                          | `car.pass.ts` and `browser-entry.ts` are untouched: the accumulator's effect is already inside the single `driftYawOffset` scalar they consume                                            |

##### D3 — How it feeds the body matrix

The accumulator feeds **one** scalar — `PhysicsCarState.driftYawOffset` — and nothing else.
Everything downstream is therefore structurally identical to today:

```
physics.step.service.ts :416-417
  driftYawOffset = slipAngle * (GAIN + transientGain)     <-- the ONLY changed line
        |
        v  (browser-entry.ts:941)
  carPoseBag.driftYawOffsetRad
        |
        +--> car.pass.ts:1365  shadow  : yaw + driftYawOffset          (Euler sum)
        +--> car.pass.ts:1565-1573 body : T * Ry(yaw) * Ry(drift)      (matrix product)
                 |
                 +--> car.pass.ts:1598  wheels parented onto NEATRACING_CAR_BODY_MODEL_MATRIX
```

**AC-078 (the 0.002 rad pin) is provably invariant.** The fixture's check
(`tireYawFollowsBodyDuringDrift`, `:926-928`) measures `wheelYaw − bodyYaw` and
`wheelOffset − bodyOffset` against `mount.rotationYawRad`. Because the wheels are _parented onto_
the body matrix, the wheel yaw is `bodyYaw + mount.rotationYawRad` for **any** body yaw, and the
wheel offset is likewise identical. The residual is a floating-point difference, independent of
the offset's magnitude. **Enlarging the drift cannot move the 0.002 rad residual** — the only
way it could move is if the offset grew so large that the `Ry` composition lost float precision
(irrelevant at ≤ 0.2 rad), or if the _pass_ stopped composing `Ry(drift)` as a matrix product.

The `car.pass.test.ts` monotonicity contract (C7) also survives: the limit is **derived** as
`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`, so it auto-scales
with the base gain, and the sweep passes explicit offsets to the pass — it constrains the _pass_,
not the physics.

##### D4 — State ownership

| State                          | Owner                                | Lifetime           | Notes                                                                               |
| ------------------------------ | ------------------------------------ | ------------------ | ----------------------------------------------------------------------------------- |
| `transientGain`                | `PhysicsCarState` (new field)        | per car, per tick  | replaced wholesale each tick; JSON-serializable; carried by `race-session.ts:81-89` |
| `DriftTransientGainState`      | `drift.model.ts` (new type)          | per car            | **separate** from `DriftModelState` (C4)                                            |
| `driftYawOffset`               | `PhysicsCarState` (existing, `:108`) | per car, per tick  | unchanged field; the accumulator is folded into its value                           |
| `carPoseBag.driftYawOffsetRad` | `browser-entry.ts:941`               | per rendered frame | unchanged; the render never sees the accumulator                                    |

##### D5 — Tick order

```
per fixed tick (dt = 1/60 s):
  1. surface
  2. yaw / heading
  3. longitudinal speed
  4. drift model      -> slipAngle, lateralVelocity
  4b. longitudinal traction loss
  4c. TRANSIENT ACCUMULATOR STEP  <-- NEW, inserted here
  4d. driftYawOffset = slipAngle * (GAIN + transientGain)   <-- the ONE changed line
  5. translate along (heading + slipAngle)
  6. tick count

per rendered frame:
  buildComposerFrame() -> carPoseBag.driftYawOffsetRad -> carPass.draw -> body matrix
```

The accumulator advances at **tick** rate (deterministic, 1/60 s) and is sampled at **frame**
rate, exactly like the existing offset. The accumulator must therefore be stepped **after** the
drift model has produced `slipAngle` (step 4) and **before** the offset is computed (step 4d) —
otherwise it would use the previous tick's slip.

##### D6 — Constants that move

**Move (the levers):**

| Constant                                      | Current            | Home                        | Role                                                                                                                                                                                                                                                                                                                         |
| --------------------------------------------- | ------------------ | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`        | `0.5`              | `drift.model.ts:97`         | the **base gain**; the magnitude lever. **Its relation to the AC-087 proxy is V-shaped with a null at `1.0`** (E3) — the RED slice must record the null as a forbidden region                                                                                                                                                |
| `NEATRACING_DRIFT_TRANSIENT_GAIN_MAX`         | new                | `drift.model.ts`            | magnitude bound of the transient. **Must be chosen so that `\|GAIN_eff\| · \|slipAngle\| < π/2`** (C2). With `\|slip\| ≤ π/2` in the worst case this argues for `\|GAIN_eff\| ≤ 1`; at the measured operating point (`\|slip\| ≈ 0.0988`) the headroom is far larger                                                         |
| `NEATRACING_DRIFT_TRANSIENT_GAIN_RATE_PER_S`  | new                | `drift.model.ts`            | how fast the slide **builds** (the "sustain" half of AC-087)                                                                                                                                                                                                                                                                 |
| `NEATRACING_DRIFT_TRANSIENT_GAIN_DECAY_PER_S` | new                | `drift.model.ts`            | how fast the slide **unwinds** on drift exit (the "crisper recovery" half). **These two trade off** — a slow decay sustains the slide but softens the recovery; this is the concrete form of the feel-direction decision                                                                                                     |
| `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD`        | new (AC-087 floor) | `drift.model.ts` (exported) | the AC-087 floor. **Coupling note:** the browser fixture hardcodes its own constants (`DRIFT_PROXY_MIN_RAD = 0.01` at fixture `:128`), so the floor needs a single source of truth _and_ a mirrored literal in the fixture; the plan should name both                                                                        |
| `NEATRACING_TIRE_GRADIENT_RADIUS_M`           | new (candidate)    | `car.pass.ts`               | AC-088 names the gradient radius, but today `uTireGradientRadius` is fed `NEATRACING_WHEEL_HUB_HEIGHT_M` (`car.pass.ts:1469`) — a **wheel-mount** constant, not a lighting constant. Decoupling it into a lighting-owned named constant is correct hygiene — **but see E6: it cannot by itself close the measured residual** |

**Must NOT move (physics constants — moving them changes the trajectory and breaks AC-079/AC-089):**

`NEATRACING_DRIFT_MIN_SPEED_MPS = 15`, `NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD = 0.25`,
`NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD = 2`,
`NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S = 0.6`,
`NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S = 12`,
`NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S = 4`, `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = π/2`.

**Moves as a side effect (derived, auto-consistent):**
`car.pass.test.ts`'s `DRIFT_VISUAL_YAW_LIMIT_RAD = NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`
— it re-derives itself from the base gain, so the monotonicity sweep stays green.

##### D7 — What the RED slice `P1-drift-feel-red` must author (recommended contracts)

1. **AC-087 floor.** Pin `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD` **against the measured baseline
   `0.049388550574608114 rad`** and record the _direction_ (bigger slide vs. crisper recovery) plus
   the chosen named lever, per the plan. Record the **`GAIN_eff = +1.0` null** as a forbidden
   region, and state whether the lever drives `GAIN_eff` down (recommended: monotone, null-free)
   or up (crosses the null).
2. **Zero straight-line yaw — a NEW contract.** No assertion exists today (E5). It must assert
   that the **body yaw** (not just the slip) is zero/unchanged on a straight line, and — because
   the recommended model is slip-proportional — that the offset is _exactly_ 0 whenever
   `slipAngle === 0`. Under the additive alternative this contract would need a hard-reset rule.
3. **Drift-exit reset / recovery.** Assert the transient returns to its rest value when the drift
   ends, and that the recovery time matches the recorded decay constant.
4. **The proxy never dips below the gate mid-drift.** A regression guard for C8: assert that the
   3-consecutive-frame run is not reset by the transient (i.e. the effective gain never approaches
   `+1.0` during an established drift).
5. **AC-088 — the measurement, not the constant.** Per E6, the RED contract must be authored
   against a measurement that can observe the fragment-shader gradient (shaded-framebuffer
   sampling), or the AC-088 lever must be re-scoped to the existing jest owner
   (`engine/lighting.test.ts`, OBS-20). **A RED contract of the form "the _current_ fixture's
   measured gradient is strictly non-zero" is unsatisfiable by the named constants** and would
   strand the tune slice. This is a **plan-wording correction**, not a tuning task.
6. **Owner correction.** The Step 14N RED packet names `examples/neatRacing/engine/passes/car.pass.test.ts`
   as the file for the tire-gradient magnitude assertion, but the tire-gradient contract is owned
   by `examples/neatRacing/engine/lighting.test.ts` (`describe('fixed-space tire lighting (OBS-20)')`).
   The RED slice should correct the file list.

---

#### Risks

| #   | Risk                                                                                                                                                                                                                                        | Severity                     | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                   | Mitigation                                                                                                                                                                                       |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| R1  | **The gain is a V-shaped proxy lever with a null at 1.0.** A tune that raises the gain toward 1.0 makes the slide _weaker_, and at 1.0 the slide is invisible — while the offset magnitude looks _bigger_.                                  | **HIGH**                     | E3                                                                                                                                                                                                                                                                                                                                                                                                                         | Record the null as forbidden; prefer driving `GAIN_eff` down; add a RED regression guard on the proxy floor that would catch a null crossing                                                     |
| R2  | **AC-088's named lever set cannot close the measured residual.** The fixture measures vertex albedo, which is uniform by construction; the gradient is a fragment-stage effect.                                                             | **HIGH**                     | E6                                                                                                                                                                                                                                                                                                                                                                                                                         | Re-scope AC-088's lever to the _measurement_; correct the file owner to `engine/lighting.test.ts`; treat this as a plan amendment in the RED slice, not a tuning task                            |
| R3  | **The 3-consecutive-frame gate (C8) is fragile.** Any transient that dips the proxy below `0.01` for one frame resets `driftProxyRunFrames`, which can break `tireYawDriftEngaged` and therefore **both** AC-078 and AC-079 simultaneously. | **HIGH**                     | E5 (C8), fixture `:876-881`                                                                                                                                                                                                                                                                                                                                                                                                | Visual-only design; drive the gain away from the null; add the RED guard in D7.4                                                                                                                 |
| R4  | **The `\|driftYawOffset\| < π/2` pin (C2) bounds the total gain.** A large `GAIN_eff` combined with a large slip could breach it at the `physics.step.service.test.ts` fixture.                                                             | MEDIUM                       | E5 (C2)                                                                                                                                                                                                                                                                                                                                                                                                                    | Bound `NEATRACING_DRIFT_TRANSIENT_GAIN_MAX` so `\|GAIN_eff\| · \|slip\| < π/2`; the RED slice should measure the slip envelope at the pinned fixture before choosing the bound                   |
| R5  | **A transient that ramps slowly may be measured before it reaches steady state**, so the RED floor could be red for the wrong reason (the transient had not engaged yet).                                                                   | MEDIUM                       | E2 (window mean), E3                                                                                                                                                                                                                                                                                                                                                                                                       | The 6-frame window mean is endpoint-anchored and cancels mid-window lag (residual ≤ 0.002 rad), but the RED contract must sample **after** the ramp completes, and must name that sampling point |
| R6  | **The build-rate / decay-rate trade-off is the feel decision itself** and cannot be tuned independently.                                                                                                                                    | MEDIUM                       | D6                                                                                                                                                                                                                                                                                                                                                                                                                         | The RED slice must record the direction; the tune slice gets one named pair of constants, not four                                                                                               |
| R7  | **The glass canopy does not follow the drift yaw** (`glass.pass.ts:1048,1086` has no `driftYawOffsetRad`). Feel work amplifies the visible mismatch between the body and the canopy.                                                        | MEDIUM                       | E1                                                                                                                                                                                                                                                                                                                                                                                                                         | Out of Step 14N's scope (the pass is not re-authored); record as a carried-forward residual for a later step                                                                                     |
| R8  | **The bounce path zeroes `speed`/`slipAngle`/`lateralVelocity` but not the offset** (`collision.utils.ts:410,417-418`) — dormant today, latent once the accumulator is wired.                                                               | LOW                          | E7                                                                                                                                                                                                                                                                                                                                                                                                                         | With the slip-proportional model the offset is zeroed automatically by `slipAngle = 0`; still worth a residual note                                                                              |
| R9  | **Spec drift is already live** — the spec's illustrative formula (`clamp(lateralVelocity/speed · GAIN, ±max)`) does not match the implementation (`atan2(...) · GAIN`, no clamp).                                                           | LOW                          | E7                                                                                                                                                                                                                                                                                                                                                                                                                         | Record as a residual; the spec amendment belongs to a documentation slice, not this one                                                                                                          |
| R10 | **The car pass is not re-authored (AC-087)**, so a transient larger than the pass's current assumptions would need a pass change — out of scope.                                                                                            | LOW                          | D3                                                                                                                                                                                                                                                                                                                                                                                                                         | Keep the total offset well inside `DRIFT_VISUAL_YAW_LIMIT_RAD`'s order of magnitude; the pass's monotonicity contract is unaffected either way                                                   |
| R11 | **The workflow-update-sync hook cannot see letter-SUFFIXED step ids.** `.github/hooks/workflow-update-sync.mjs` parses step headings with `/^#{4,5}\s+(?:Step                                                                               | Packet)\s+([A-Z]?\d+)\s+(?:— | -)\s+(.+)\s\[([A-Z]+)\]/`— an id pattern that accepts`14`and`E1`but not`14N`. The heading `#### Step 14N — … [WIP]`is therefore never parsed, so the hook reports`currentWipStep: null`/`actionTaken: 'between-steps'`even though Step 14N **is**`[WIP]`, and it will not auto-advance Step 14N → Step 15. The hook still returns `ok: true`/`pass: true`and`plan-sync` still passes, so this is a **silent** degradation. | MEDIUM (workflow)                                                                                                                                                                                | `.github/hooks/workflow-update-sync.mjs:196-198` | Pre-existing and **not** introduced by this research pass. Outside this slice's edit scope; the step advancement must be driven manually by the orchestrator. Owner: the workflow-hook maintainer |

---

#### Certainty statement

- **E1, E2, E4, E5, E7** — confidence **0.95** (direct source reads, re-verified after the scout
  reports).
- **E3 (the V-shaped proxy identity)** — confidence **0.93**. The identity is derived from the
  source and **confirmed against the recorded measured baseline** to the last digit
  (`0.049388550574608114 = 0.5 × 0.098777101149216228`). The residual uncertainty is the
  operating-point slip magnitude, which the RED slice should re-measure.
- **E6 (the AC-088 measurement blindness)** — confidence **0.90**. The shader path, the mesh
  builder's own "Vertex colors stay uniform rubber" comment, and the fixture's VBO read are all
  directly verified. The residual uncertainty is only whether a shaded-framebuffer sampling path
  already exists elsewhere in the fixture that could be reused.
- **D1 (the transient-gain model)** — confidence **0.85** as a _design recommendation_; it is
  **not** a decision. The feel direction and the floor value are explicitly owned by the RED
  slice `P1-drift-feel-red` and are **not** invented here.
- **R1, R2** — confidence **0.85**. Both are plan-level corrections that the RED slice must
  record before any production edit.

## PART B — the six folded audit / reference artifacts

> **Provenance (collective).** All six were authored by `02-researching` (Tier 1) on 2026-10-27 or 2026-10-24
> under the `USER VERDICT 3` / `USER VERDICT 5` dispatch records in the tracker, READ-ONLY, with NO browser
> runs, NO Chrome DevTools MCP, NO child agents inside the audits and no production byte touched. Each
> artifact's own `0. Provenance` section is preserved below verbatim — it carries the method, the evidence
> ranking, the scope boundary and the non-duplication contract, and it is the authority for that artifact's
> findings. Where an artifact cites a sibling, the sibling is now a Part of THIS file.

### B-1 — AUDIT-PHYSICS — read-only code audit of the drift/traction model

> **Provenance.** Artifact id `neatracing-voxel-racer.audit-physics`; author `02-researching` (Tier 1),
> 2026-10-27; dispatch origin the `USER VERDICT 3` record in the tracker (`dispatch_plan[0]` =
> `AUDIT-PHYSICS`); subject `examples/neatRacing/physics/drift.model.ts` (787 lines) +
> `examples/neatRacing/physics/physics.step.service.ts` (647 lines); constraints honoured — READ-ONLY code
> audit, NO child agents, NO browser runs, NO Chrome MCP, no production byte touched.
> **Fold note (2026-10-31).** The naming-deviation paragraph in section `0.` below is HISTORICAL: it records
> why the artifact was written as a sibling on 2026-10-27. It is folded here as Part B-1 and the sibling
> file is DELETED.

**Artifact id**: `neatracing-voxel-racer.audit-physics`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 3` record, `plans/neatracing-voxel-racer.plans.md` L9988+, `dispatch_plan[0]` = `AUDIT-PHYSICS`
**Subject**: `examples/neatRacing/physics/drift.model.ts` (787 lines) + `examples/neatRacing/physics/physics.step.service.ts` (647 lines)
**Constraints honoured**: READ-ONLY code audit · NO child agents · NO browser runs · NO Chrome MCP · no production byte touched.

---

#### 0. Provenance, method and the naming deviation

**Method.** Every claim below is a _static-code_ read plus a repo-wide readership grep. Line numbers are cited against the two subject files as they stand on disk (2026-10-27). No runtime measurement was performed (the directive forbids browser runs; a headless physics harness was also not in scope). The numeric sensitivities that are quoted as "derived" are arithmetic from the source constants (0.55 / 0.45 / 89 / 15 / 3.5 / 2.5 / 0.045 / 70) and are marked **DERIVED** where they are not literally printed in the source.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. There is no runtime evidence in this audit, so every finding is `static-code` ranked. Where a source JSDoc disagrees with the code, the **code wins** and the JSDoc drift is reported as a literal gap (G5).

**Naming deviation (disclosed).** The contract names `plans/<PlanName>.research.md`. That exact path was **already occupied** by a different, closed 14N artifact (the Step 14N drift-feel _integration coupling_ design, dated 2026-10-24 — now **Part A** of this merged file), and the drift-feel brief was likewise occupied (now **Part B-6**). Clobbering or appending to a closed, dated artifact would fuse two different questions under one date. This artifact therefore followed the repo's own topic-suffix family convention as a **new sibling** — `plans/neatracing-voxel-racer.audit-physics.research.md` (the file created on 2026-10-24, DELETED in the 2026-10-31 fold) — and is now folded as **Part B-1** of this merged file. It remains a `plans/*.research.md` sibling of the tracker and is **not** under `docs/research/`. The orchestrator may rename or merge it without loss.

**Scope boundary.** This is the PHYSICS half only. The camera aim path, the wheel-spin idiom seam, the `stepPhysics` wrapper in `browser-entry.ts`, and the τ=0.08 follow law are `AUDIT-CAMERA`'s scope and are **not** adjudicated here. The single cross-boundary fact established by this audit is recorded in **F4**.

---

#### 1. Question

The `USER VERDICT 3` dispatch asks, of the live physics model:

> what does traction/scrub loss **ACTUALLY** scale with; where speed × turning-angle coupling exists or is missing; the MIN_SPEED gate, the clamps, the latch, the D1.3 deficit form; literal GAPS (TODOs, dead branches, constants defined but unread, channels never wired).

And the verdict opens a NEW SPEC surface (S-2/S-3) asking whether the live model implements speed × steer proportionality, and **where fixed constants defeat the proportionality at game speeds**.

The audit must therefore answer two things: **what the law is**, and **what the contract can and cannot prove about it**.

---

#### 2. Evidence

##### 2.A — What the traction/scrub loss ACTUALLY scales with _(confidence 0.95, source: static-code)_

The traction loss in the live model is a single term: the **longitudinal speed scrub**.

`drift.model.ts:632-646`

```ts
export function computeLongitudinalSpeedScrub(
  slipAngle: number,
  speed: number,
  speedPercent: number,
): number {
  return (
    NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S * // 0.55   (drift.model.ts:282)
    Math.abs(slipAngle) *
    speed *
    speedPercent
  );
}
```

`physics.step.service.ts:419-427` (the only call site)

```ts
const scrubbedSpeed = Math.max(
  0,
  nextSpeed -
    computeLongitudinalSpeedScrub(
      state.slipAngle, // tick-START published slip
      nextSpeed, // tick-END speed
      speedPercent, // tick-START speed percent
    ) *
      NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
);
```

With `speedPercent = speed / 89` (`physics.step.service.ts:506-513`, `resolveSpeedPercent`, against `NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S = 89` at `:142`):

> **scrub [m/s²] = 0.55 · |published slip| · speed² / 89**

Four properties follow, and they are the whole answer to "what does it scale with":

| #   | Property                                                                                                                                                                                                                                                           | Evidence                                                                                                                      |
| --- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| A1  | The **speed** dependence is **quadratic**, not proportional. The `speed` and `speedPercent` factors are _the same quantity twice_ — `speedPercent` is a normalized copy of `speed`. The `speed²` term is a duplication artifact, not a designed second-order term. | `drift.model.ts:632-646` + `physics.step.service.ts:506-513`                                                                  |
| A2  | The **slip** factor is the **PUBLISHED** slip (`state.slipAngle`, the tick-start `atan2` of the lateral velocity), **not** the committed slip, **not** the steering angle, **not** the yaw rate.                                                                   | `physics.step.service.ts:422` (`state.slipAngle`) vs `:396` (`stepDriftCommitment` → `nextCommitted.slip`, a different value) |
| A3  | The scrub is **exactly zero on a straight line** (published slip = 0 ⇒ term = 0). Pinned by `[RED-G5-SPEED-SCRUB]`.                                                                                                                                                | `drift.model.test.ts:724-736`                                                                                                 |
| A4  | The scrub is applied as a **reduction of the tick-end speed**, floored at 0, and is _not_ a separate "traction" channel — the value is computed inside `stepPhysics` and **discarded**.                                                                            | `physics.step.service.ts:419-430`                                                                                             |

**DERIVED sensitivity table** (saturated committed slip 0.45 rad, so the published slip saturates there too):

| speed               | speedPercent | scrub accel [m/s²]                      |
| ------------------- | ------------ | --------------------------------------- |
| 89 m/s (320.4 km/h) | 1.0          | 22.03                                   |
| 80 m/s              | 0.899        | 17.80                                   |
| 30 m/s              | 0.337        | 2.50                                    |
| 20 m/s              | 0.225        | 1.11                                    |
| ≤ 15 m/s            | ≤ 0.169      | **0.000 — the gate is closed** (see B1) |

**DERIVED equilibrium**: the engine accel is `89/5 = 17.8` m/s² (`NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2`, `physics.step.service.ts:148`). The scrub balances it at `0.55 · 0.45 · v²/89 = 17.8` ⇒ **v ≈ 80.0 m/s (287.5 km/h)**, matching the documented measured 79.858 m/s / 287.49 km/h deficit of 32.55 km/h at 600 ticks. The source's own JSDoc records the **4/9 crossover** (`0.4444`): the engine wins while the coefficient `c ≤ 4/9`; the landed `c = 0.55` is above the crossover, so the **scrub binds**. The scrub's maximum (22.03 m/s²) exceeds the engine's 17.8 m/s², so the scrub is the limiting term at top speed.

##### 2.B — Where speed × turning-angle coupling exists, and where it is missing _(confidence 0.93)_

**The coupling is MISSING.** There is **no steering-angle term and no yaw-rate term anywhere in the traction law.** The only "turning" quantity that reaches the speed path is the _committed slide_, and that slide's magnitude is destroyed by a clamp.

The full chain, as implemented:

```
steer ──▶ [0.5 threshold] ──▶ latch (turnSign) ──▶ [3.5 rad/s · |steer| · dt]  (SPEED-INDEPENDENT)
      ──▶ committed slip ──▶ [clamp ±0.45 rad]  ──▶ chase ──▶ published slip ──▶ scrub
```

Four independent blocks stand between "turning" and "traction loss". Each must be re-decided for S-3.

**B1 — The MIN_SPEED gate is a hard cliff, not a proportional ramp.** _(confidence 0.95)_
`drift.model.ts:184` — `NEATRACING_DRIFT_MIN_SPEED_MPS = 15` (54 km/h).

- `drift.model.ts:494-501` (stage 0 of `stepDriftCommitment`): below the gate the commitment is **released** (`slip = 0`, `turnSign = 0`) and the **re-arm token is cleared** (`reArmed = false`).
- `drift.model.ts:680-683` (stage 0 of `computeNextLateralVelocity`): the lateral velocity is returned as **exactly 0**.

Consequence: at ≤ 54 km/h the traction loss is **exactly 0.000**, with no fade-in. A car at 55 km/h and a car at 320 km/h differ by a _step_, not by a curve. The verdict's named constant ("the MIN_SPEED gate") is real and is a genuine discontinuity in the speed→traction map.

**B2 — The engage threshold is a fixed steer magnitude, and the engage ramp is speed-independent.** _(confidence 0.93)_
`drift.model.ts:189` — `NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD = 0.5`.
`drift.model.ts:201` — `NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S = 3.5`.

```ts
slip -= turnSign * NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S * Math.abs(steer) * dt;
```

(`drift.model.ts:520-524` on engage, `:534-536` while drifting.)

The ramp rate scales with `|steer|` but **not** with speed. The slide therefore builds at the same rad/s at 20 m/s as at 89 m/s.

**Provable consequence (the sharpest finding in this audit).** A **full-speed turn with `|steer| < 0.5`** produces:

- `reArmed && |steer| >= 0.5` fails ⇒ no latch ⇒ `turnSign = 0`;
- ⇒ `computeNextLateralVelocity` stage 5 forces `nextLateralVelocity = 0`;
- ⇒ `computeSlipAngle(0, speed) = atan2(0, speed) = 0` ⇒ **published slip = 0**;
- ⇒ **scrub = 0.55 · 0 · … = exactly 0**.

The car **rotates** at `yawRatePerTick = steer · 2 · speedPercent · dt` (`physics.step.service.ts:515-522`, `NEATRACING_PHYSICS_STEERING_RATE = 2`) and **loses no speed whatsoever**. At full speed, a half-lock turn is a free turn. This is the sharpest possible demonstration that **S-3 (speed × turning angle → traction loss) is not implemented**: the coupling term does not merely under-scale, it is _absent_.

**B3 — The commit clamp erases the turning angle.** _(confidence 0.92)_
`drift.model.ts:66` — `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45`; applied at `drift.model.ts:551-555`:

```ts
slip = clamp(
  slip,
  -NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
);
```

Once `|committed slip| ≥ 0.45 rad`, **further steering magnitude has zero effect on the traction law** — the clamp is a hard saturation on the only turning-shaped input to the scrub. Above the clamp the scrub reduces to `0.55 · 0.45 · speed²/89`: a function of **speed alone**. **This is the mechanism by which a fixed constant defeats the proportionality: the 0.45 rad commit clamp converts a "speed × turning" law into a "speed-only" law in the entire operating envelope the model actually reaches at speed.**

**B4 — The tire's grip budget is contractually SPEED-INDEPENDENT (the G-2 pin).** _(confidence 0.95)_
`drift.model.ts:567-579`:

```ts
export function computeGripBudgetFactor(
  slipAngle: number,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- the law is speed-INDEPENDENT (G-2)
  speedPercent: number,
): number {
  const demand = Math.max(
    0,
    Math.abs(slipAngle) - NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD,
  );
  return 1 / (1 + NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD * demand);
}
```

The `speedPercent` parameter is accepted and **ignored** — with an explicit eslint-disable naming the reason. The factor is `1/(1 + 2·max(0, |slip| − 0.2))`: at the 0.45 rad saturation it is **0.667**, a 33 % falloff, and it is **speed-independent by design**.

**This is a RATIFIED RED pin, not an oversight.** `drift.model.test.ts` `[RED-G2]` asserts `computeGripBudgetFactor(0.9, 1) === computeGripBudgetFactor(0.9, 0.2)` — the tire grip budget is contractually a function of the slip only. The model's own JSDoc (`drift.model.ts:558-562`) records the decision: _"The budget is a function of the SLIP only — the speed demand is charged by the SPEED SCRUB (stage 6), not by the tire (G-2)."_

**Consequence for the verdict:** the `USER VERDICT 3` contested set names "the 20 km/h floor, MAX_SLIP_ANGLE_RAD 0.45, the commit clamp, the MIN_SPEED gate" as the fixed constants that defeat the proportionality. **It does not name G-2** — and G-2 is the pin that most directly forbids a speed term in the tire. Any S-2/S-3 implementation that adds speed to the grip law **must re-open the ratified `[RED-G2]` pin**, which the RED lane will otherwise re-pin. G-2 must be added to the contested set.

**B5 — The chase deliberately ignores the yaw rate.** _(confidence 0.90)_
`drift.model.test.ts` `[RED-G6-COMMITTED-CHASE]` pins that _"the 14N model has NO outward push — stage 3 is the committed chase, and the chase tracks the COMMITTED slip, **not the yaw rate**."_ The chase target is `Math.tan(slip) * speed` (`drift.model.ts:694`). So the yaw rate — the one quantity that is literally "how much the car is turning" — is deliberately excluded from the lateral dynamics, and (per B4) it is also excluded from the tire.

##### 2.C — The clamps, complete _(confidence 0.94)_

| Clamp / bound                       | Where                                                                                  | Value                 | Runtime?                | Effect on S-2/S-3                                                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------- | --------------------- | ----------------------- | ----------------------------------------------------------------------------------------- |
| Commit clamp                        | `drift.model.ts:551-555` (const `:66`)                                                 | ± 0.45 rad            | **YES**                 | **Kills the turning term above 0.45 rad** (B3)                                            |
| MIN_SPEED gate                      | `drift.model.ts:494-501`, `:680-683` (const `:184`)                                    | > 15 m/s              | **YES**                 | **Traction loss is exactly 0 below 54 km/h** (B1)                                         |
| Speed floor                         | `physics.step.service.ts:420`                                                          | 0 m/s                 | YES                     | The scrub can only bleed speed to 0                                                       |
| Speed ceiling                       | `physics.step.service.ts:567-602` (const `:142`)                                       | 89 m/s                | YES                     | The engine limit; the scrub's top-speed bound                                             |
| Off-road floor                      | `physics.step.service.ts:436-445` (const `:182`)                                       | 22.25 m/s (80.1 km/h) | YES                     | **Lifts a post-scrub speed back up — masks the scrub off-road**                           |
| Published-slip bound                | `drift.model.ts:779-787`                                                               | (−π/2, π/2)           | YES                     | Never reached (the published slip saturates ≈ 0.45)                                       |
| Published-slip **assertion** bounds | `drift.model.ts:94` (0.002 overshoot), `:128` (0.452 max), `:163` (0.493 off-road max) | —                     | **NO — assertion only** | They never clip; they are read by tests (and mirrored as literals in the browser fixture) |

**The off-road floor is a real measurement hazard.** `isOffRoadSpeedFloorEngaged` (`physics.step.service.ts:543-565`) re-lifts the **post-scrub** speed to 22.25 m/s. On grass, the drift scrub's speed deficit is therefore **partially cancelled by the floor**. Any new proportional-traction contract must be measured and asserted **on road** — the browser fixture already enforces `straightCaptureOnRoad && driftCaptureOnRoad`, so the existing lane is on-road-safe.

##### 2.D — The latch (three-state machine) _(confidence 0.90 for L1, 0.88 for L2, 0.85 for L3)_

State: `DriftCommittedState { slip, turnSign, reArmed }` (`drift.model.ts:385-396`).

- **ENGAGE** (`drift.model.ts:514-525`): requires `reArmed && |steer| >= 0.5 && slip === 0`; sets `turnSign = sign(steer)`.
- **RE-ARM** (`drift.model.ts:526-530`): `|steer| < 0.15` ⇒ `reArmed = true`. **Lives only inside the `turnSign === 0` branch.**
- **RELEASE** (`drift.model.ts:537-548`): at `|slip| <= 0.02` ⇒ `slip = 0, turnSign = 0, reArmed = false`; and at the MIN_SPEED gate (`:494-501`).
- **CONTRACTION** (`drift.model.ts:537-549` + `computeCountersteerAuthority` `:591-618`): an opposite-sign steer unwinds the slide at a flat `EXIT_RATE_RAD_PER_S = 2.5` rad/s and **never flips `turnSign`**.

**L1 — a NEUTRAL wheel does NOT unwind the slide; it FREEZES it.** _(confidence 0.90)_
`drift.model.ts:506-508`:

```ts
const countersteering = turnSign !== 0 && Math.sign(steer) === -turnSign;
const drifting = turnSign !== 0 && !countersteering;
```

With `steer = 0`, `Math.sign(0) = 0 ≠ -turnSign` ⇒ `countersteering = false` ⇒ `drifting = true` ⇒ the **entry** ramp runs with `|steer| = 0`:

```ts
slip -= turnSign * 3.5 * Math.abs(steer) * dt; // -= turnSign * 3.5 * 0 * dt  →  NO CHANGE
```

The slide is **frozen at its current value indefinitely**. The chase's hard-zero (stage 5, `drift.model.ts:709-714`) is gated on `!drifting`, so it does **not** fire either — the lateral velocity keeps chasing the frozen target and the slide persists. The slide is unwound **only** by an opposite-sign steer (`|steer| ≥ 0.9`, `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER`), or by the speed gate closing. A player who lifts the wheel mid-drift sees the car keep sliding, forever.

**L2 — the neutral re-arm branch is unreachable while committed.** _(confidence 0.88)_
The `|steer| < 0.15` re-arm sits inside `if (turnSign === 0)`. A committed car that returns the wheel to neutral therefore does **not** re-arm (it cannot — it is committed) and does **not** unwind (per L1). The latch is self-consistent, but the "neutral = let go" behaviour a player expects is absent, and the two branches of the latch have disjoint reachable conditions.

**L3 — the ENGAGE gate uses an exact float comparison.** _(confidence 0.85)_
`drift.model.ts:517` — `slip === 0`. It is reachable today (both release paths assign the literal `0`), so it is safe as written, but it is a fragile equality on a float that a future non-zeroing release path would break silently.

**L4 — the countersteer authority is NOT slide-magnitude-proportional.** _(confidence 0.90)_
`drift.model.ts:612-618` returns `turnSign · 70 · |steer| · speedPercent` m/s², gated only by `|slip| > 0.02` (`:231`). Catching a 0.45 rad slide and catching a 0.03 rad slide take the **same** authority. This is the one place in the model where a `speedPercent` factor genuinely enters an acceleration — it is the closest existing precedent for a "speed-proportional" term, and the 01-planning author should note it as such.

**L5 — the chase response rate is speed-independent.** _(confidence 0.85)_
`drift.model.ts:219` — `NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S = 0.045`, a **fixed** time constant, and `gripFactor` is speed-independent by G-2 (B4). So the chase's _response rate_ does not scale with speed; only the chase _target_ (`tan(slip)·speed`) does. The physics path therefore carries its own fixed τ, parallel to the camera's contested τ = 0.08. The verdict's contested set names only the camera's τ; this one is not in it.

##### 2.E — The D1.3 deficit form, and why it cannot prove S-2/S-3 _(confidence 0.94)_

The D1.3 contract is a **single-point, absolute, floor-only** assertion.

**Jest lane** — `physics.step.service.test.ts:697-715`:

```ts
const straight = runSteps(600, 42, STRAIGHT_INPUT, 'road');
const drift = runStepsFrom(atTopSpeed(), 600, FIXTURE_INPUT, 'road');
const deficitKmh = (straight.speed - drift.speed) * 3.6;
expect(deficitKmh).toBeGreaterThanOrEqual(
  NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH,
);
```

**Browser lane** — `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:2502-2506`:

```js
proportionalTractionLoss: tireYawDriftEngaged &&
  driftSpeedDeficitKmh !== null &&
  driftSpeedDeficitKmh >= DRIFT_MIN_SPEED_DEFICIT_KMH &&
  maxSpeedKmhDrift <= TRACTION_LOSS_MAX_SPEED_KMH &&
  straightCaptureOnRoad && driftCaptureOnRoad && !captureRegimeSplit,
```

**Why the existing contract cannot prove proportionality — five independent reasons:**

1. **It is a floor.** A MINIMUM. Over-scrubbing can never fail it; only under-scrubbing can. It constrains one direction only.
2. **It is a single capture point.** One speed (top speed / the 90 km/h accel target) × one steer input. A speed-proportional law and a speed-constant law are **indistinguishable at one point**. No ratio, no second speed, no second steer.
3. **It is absolute km/h, not a ratio.** The plan's own wording calls the D1.3 form _"the scale-free scrub-rate / speed-deficit assertion"_ (`plans/neatracing-voxel-racer.plans.md` L4024) — but the assertion is **literally an absolute km/h delta**. It is not scale-free, and the plan's description overstates it.
4. **The upper-bound clause is PROVABLY VACUOUS.** `TRACTION_LOSS_MAX_SPEED_KMH = 340` (fixture L201). The service clamps the speed to 89 m/s = **320.4 km/h** (`physics.step.service.ts:142`, `:567-602`), and the scrub only ever **reduces** the speed. Therefore `maxSpeedKmhDrift ≤ 320.4 < 340` **always** — the clause can never fail. It is a dead term inside a check **named** `proportionalTractionLoss`: the name promises proportionality, the form asserts a floor.
5. **The unit-lane speed pin is a MONOTONICITY, not a proportionality.** `drift.model.test.ts` `[RED-G5-SPEED-SCRUB-2]` asserts only `expect(fast).toBeGreaterThan(slow)` (fast = the scrub at 80 m/s, slow = at 20 m/s). **Any strictly increasing function of speed passes this** — a logarithm, a `√v`, a `v^0.1`. The current contract set **cannot distinguish a proportional scrub from a merely monotone one**, and therefore cannot establish S-2.

**Correction to the verdict's premise (load-bearing).** `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH = 20` (`drift.model.ts:341`) is **never read by production code**. A repo-wide grep over `examples/`, `docs/` and `plans/` returns it in exactly two executable places: `physics.step.service.test.ts:85,704,1094` (the jest lane) and the **mirrored literal** `var DRIFT_MIN_SPEED_DEFICIT_KMH = 20;` (`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:227`, used at `:2504` and reported at `:2736`). The browser fixture **mirrors** the constant rather than importing it (the debug seam is fixed at five fields), so the two can drift apart silently.

**Therefore the 20 km/h floor does not "defeat the proportionality at game speeds" — it has no runtime effect at all.** What it defeats is the ability to **prove** proportionality: a floor at a single point constrains only the under-scrub direction and is blind to the shape of the speed→traction curve. The verdict's characterisation of this constant should be corrected in the amended plan.

##### 2.F — Literal GAPS

**F1 — Fully dead exports (defined, exported, read by NOBODY).** _(confidence 0.96)_
A repo-wide grep over `examples/neatRacing/**/*.ts` for `COUNTERSTEER_HOLD_TICKS` returns **exactly one match — the declaration itself**:
`drift.model.ts:251` — `export const NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS = 8;`
It is read by **no production code and not even by a test**. Its own JSDoc records it as an open GAP (_"EXPORTED but NOT ENFORCED"_). It is a **hard** gap: a named, documented, contract-surface constant with **no reader at all**.

**F2 — Exports read only by tests (never by production).** _(confidence 0.95)_

| Symbol                                            | Where                    | Value | Production reader?                                  |
| ------------------------------------------------- | ------------------------ | ----- | --------------------------------------------------- |
| `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD`            | `drift.model.ts:288`     | 0.2   | **NO** — jest only + a mirrored fixture literal     |
| `NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD`       | `drift.model.ts:294`     | 0.01  | **NO** — jest only + a mirrored fixture literal     |
| `NEATRACING_DRIFT_PROXY_MIN_FRAMES`               | `drift.model.ts:299`     | 3     | **NO** — jest only + a mirrored fixture literal     |
| `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH`          | `drift.model.ts:341`     | 20    | **NO** — jest only + a mirrored fixture literal     |
| `computeDriftSlideProxyRad()`                     | `drift.model.ts:655-662` | fn    | **NO** — jest only; the fixture mirrors the formula |
| `NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD`         | `drift.model.ts:128`     | 0.452 | **NO** — jest only + a mirrored fixture literal     |
| `NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD` | `drift.model.ts:163`     | 0.493 | **NO** — jest only + a mirrored fixture literal     |

The last four (the AC-087 prominence-floor family) are the notable ones: **the deterministic unit proxy never reaches production.** The proxy exists in the test lane and as a **hand-copied literal** in the browser fixture — the "one source of truth" is a **duplication** that can drift. The published-slip ceilings are legitimate as _assertion_ bounds, but they are asserted in a lane that cannot import them.

**F3 — Dead fields and dead parameters.** _(confidence 0.95)_

- `DriftStepInput.slipAngle` (`drift.model.ts:421-437`) is **passed but never read**: `stepDriftCommitment` destructures `{ steer, speed, dt }` (`:489`); `computeNextLateralVelocity` destructures `{ speed, speedPercent, dt, lateralVelocity }` (`:674`); `computeCountersteerAuthority` destructures `{ steer, speedPercent }` (`:595`). Yet `physics.step.service.ts:402` and `:416` still pass it. **A dead field on the model's own input contract.**
- `computeGripBudgetFactor`'s `speedPercent` parameter — accepted and ignored, eslint-disabled at `drift.model.ts:569`. It encodes the G-2 decision _in the signature_, which is why the pin is easy to miss on a read.

**F4 — Channels published but never wired (the S-4 surface).** _(confidence 0.96)_
`PhysicsCarState` (`physics.step.service.ts:80-124`) publishes `x, y, heading, speed, yawRate, slipAngle, lateralVelocity, drift{committed{slip,turnSign,reArmed}, pose{pose,poseRate,articulation}}, tick`.

**Production consumers (verified):**

- `browser-entry.ts:972` — `car.drift.pose.pose` → `carPoseBag.driftYawOffsetRad`
- `browser-entry.ts:981-982` — `car.drift.pose.pose` → `poseRad`; `car.slipAngle` → `slipAngleRad`
- `collision.utils.ts:386-389` — `car.slipAngle`, `car.lateralVelocity` (the bounce recompose)

**Published but read by NO production consumer:**

- **`car.yawRate`** — declared `physics.step.service.ts:95`, initialised `:308`, written `:479`, and read **only** by `physics.step.service.test.ts:481-483`. **The turning rate is already on the channel and entirely unwired** — and it is exactly the signal a "speed × turning angle" coupling (S-3) and a proportional camera follow (S-1) would consume.
- `car.drift.committed.slip` — the commanded slide.
- `car.drift.committed.turnSign` — the latched turn direction.
- `car.drift.committed.reArmed` — the re-arm token.
- `car.drift.pose.poseRate` — the visible yaw **rate** (the natural proportional camera input).
- `car.drift.pose.articulation` — the steer-yaw articulation.

**F5 — No traction-loss scalar exists on the channel.** _(confidence 0.95)_
There is **no** traction-loss / scrub / grip field on `PhysicsCarState`. The scrub is computed at `physics.step.service.ts:419-430` and **discarded**; only its _effect_ (a reduced speed) survives. Therefore:

- The camera **cannot** read "the same drift/traction-loss amount the tires experience" (S-4) from the physics today.
- To satisfy S-4 the view layer would have to **re-derive** the scrub from `car.slipAngle` and `car.speed` by re-implementing `computeLongitudinalSpeedScrub` outside the model — a **duplicated physics law** in the presentation layer, which is the classic drift hazard, and which would silently diverge from the model the moment a constant moves.
- **The physics must publish the shared scalar** for S-4 to be implementable without duplication. Candidate scalars, in order of directness: **(a)** the scrub acceleration `0.55·|slip|·speed·speedPercent` — the literal "traction loss", already speed-proportional, already the quantity the tire actually loses; **(b)** the grip factor `1/(1+2·max(0,|slip|−0.2))`; **(c)** the committed slip.

**F6 — Stale JSDoc describing the retired TP-02 model.** _(confidence 0.90)_
`physics.step.service.ts:97-108`:

- `slipAngle`: _"Zero while the car is not moving; **grows only under lateral push, decays via grip**"_ — this describes the **retired TP-02 outward-push** model. The 14N model has **no outward push** (pinned by `[RED-G6-COMMITTED-CHASE]`, B5); the slip is produced by the committed chase.
- `lateralVelocity`: _"**Driven by the TP-02 outward push**, drained by the grip rate"_ — the same staleness; the lateral velocity is now driven by the committed chase toward `tan(slip)·speed`.

Both are on the **public state contract** that the camera and the collision code read, so a reader of the interface is actively misinformed about where the lateral motion comes from.

**F7 — A mixed tick read in the scrub.** _(confidence 0.88)_
`physics.step.service.ts:419-427` feeds the scrub the **tick-start** published slip (`state.slipAngle`) and the **tick-start** `speedPercent` (`:374`) but the **tick-end** speed (`nextSpeed`). The inline comment claims _"the scrub reads the tick-start slip, like every other tick-start read in this integrator"_ — true for the slip, but the **speed** in the same expression is a tick-end read. The choice is self-consistent (the scrub bleeds the tick-end speed), but the comment overstates the uniformity, and the mixed read makes the feedback loop (scrub → speed ↓ → chase target ↓ → published slip ↓ → scrub ↓) order-dependent in a way that is hard to reason about from the comment alone.

**F8 — The "single source of truth" is a duplication in the fixture.** _(confidence 0.90)_
The browser fixture mirrors `DRIFT_MIN_SPEED_DEFICIT_KMH` (L227), `DRIFT_MIN_SPEED_KMH` (L233), `ACCEL_TARGET_SPEED_KMH` (L236), `DRIFT_PROXY_MIN_RAD` (L256) and the published-slip ceilings as **hand-written literals**, because the debug seam is fixed at five fields and cannot import. Every one of these is a drift hazard on the next constant change, and the mirrored `proportionalTractionLoss` check's dead 340 km/h clause will keep passing regardless of what the physics does.

---

#### 3. Decision

**D1 — Gap classification of the four verdict clauses.**

| Clause                                                   | Gap type         | Confidence | One-line verdict                                                                                                                                                                                                                                                              |
| -------------------------------------------------------- | ---------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **S-2** (traction loss scales proportionally with speed) | **partial**      | 0.92       | The scrub is **quadratic** in speed (via the duplicated `speedPercent`), but only through a redundancy; the slip factor is speed-independent, the tire's grip budget is **contractually** speed-independent (G-2), and below the MIN_SPEED gate the loss is **exactly zero**. |
| **S-3** (traction loss = speed × turning angle)          | **missing**      | 0.93       | There is **no steering-angle or yaw-rate term anywhere in the traction law**. The only turning-shaped input is the committed slide, whose magnitude is erased by the 0.45 rad clamp, and a full-speed turn below the 0.5 engage threshold loses **exactly zero** speed.       |
| **S-4** (one shared scalar drives both)                  | **missing**      | 0.95       | **No traction-loss scalar is published on the channel.** The camera cannot read the tires' loss amount; the only way to satisfy S-4 today is to duplicate the physics law in the view layer.                                                                                  |
| **S-1** (camera animates proportionally)                 | **out of scope** | —          | `AUDIT-CAMERA`'s surface. The physics-side fact that bears on it: `car.yawRate` and `car.drift.pose.poseRate` are already published and **entirely unwired** (F4).                                                                                                            |

**D2 — The four fixed constants that defeat the proportionality, in dependency order.** All four must be re-decided for S-3; the order matters because each one gates the next.

| #      | Constant                                                                                   | Where                                 | What it does to the proportionality                                                            | Contested by the verdict?                               |
| ------ | ------------------------------------------------------------------------------------------ | ------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| **B4** | **G-2** (the speed-independent tire grip budget)                                           | `drift.model.ts:567-579` + `[RED-G2]` | **Forbids a speed term in the tire outright**                                                  | **NO — must be added**                                  |
| B1     | `NEATRACING_DRIFT_MIN_SPEED_MPS = 15`                                                      | `drift.model.ts:184`                  | A hard cliff: **zero** traction loss below 54 km/h                                             | YES                                                     |
| B2     | `NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD = 0.5` + the speed-independent `ENTRY_RATE = 3.5` | `drift.model.ts:189`, `:201`          | A full-speed turn under half-lock is **free**; the slide builds at the same rad/s at any speed | Partially (the ramp is not named)                       |
| B3     | `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45`                                               | `drift.model.ts:66`                   | **Erases the turning term** above 0.45 rad ⇒ the law becomes speed-only                        | YES (as "the commit clamp" / "MAX_SLIP_ANGLE_RAD 0.45") |

**D3 — The contract must be RE-AUTHORED, not merely re-pinned.** The current AC-079 set is floor-only, single-point, and (in one term) provably vacuous. To _establish_ S-2/S-3 the contract needs at minimum:

1. A **multi-speed comparison** — the same steer input at ≥ 2 speeds, asserted as a **ratio** or a **normalized deficit**, not a monotonicity. `[RED-G5-SPEED-SCRUB-2]`'s `fast > slow` cannot distinguish proportional from monotone and must be **strengthened**, not merely re-run.
2. A **turn-angle comparison** — the same speed at ≥ 2 steer magnitudes (one above, one below the 0.5 engage threshold), asserting that the loss **increases** with turning. This is the assertion that would have caught the missing coupling.
3. A **clamp-region assertion** — that the loss continues to rise with turning **past** the 0.45 rad commit clamp, or a documented decision that the clamp **is** the saturation and the proportionality is defined only below it.
4. The **dead 340 km/h clause must be removed or made reachable** (it can never fail; see 2.E reason 4). The plan's own B-9 disposition replaced the retained 290 km/h ceiling; a second provably-vacuous ceiling has replaced it and carries the same defect.
5. The mirrored fixture literals (F8) must be re-derived from the model, or the seam must be widened to carry the values.

**D4 — S-4 requires a physics-side decision BEFORE the camera contract can be written.** The shared scalar must be **published**, not re-derived. Recommended: publish the **scrub acceleration** (F5-a) as the canonical "traction-loss amount" and let the camera consume it. The two channels already published and unwired — `car.yawRate` and `car.drift.pose.poseRate` — are the natural _proportional_ camera inputs and should be wired in the same wave, so S-1 and S-4 are satisfied by one shared read rather than two parallel re-derivations.

**D5 — The corrections the amended plan must carry forward.**

1. **G-2 joins the contested set** (D2, B4) — it is the pin that forbids a speed term in the tire, and it is not currently listed.
2. **The 20 km/h floor is a test-only constant** (2.E) — it has **no runtime effect**; it defeats the _proof_, not the _law_. The verdict's premise needs correcting.
3. **The D1.3 form is not "scale-free"** (2.E reason 3) — it is an absolute km/h floor at one point.
4. **The camera's τ = 0.08 is not the only fixed τ** — the tire chase has its own fixed `SLIP_RESPONSE_TAU_S = 0.045` (L5) and its response rate is speed-independent by G-2.

**D6 — The literal gaps are, in priority order:** **F1** (a named contract constant with **no reader at all** — `COUNTERSTEER_HOLD_TICKS`), **F5** (no traction-loss scalar — blocks S-4), **F4** (`yawRate`/`poseRate`/`committed.*` unwired), **F2+F8** (the duplicated mirror set), **F3** (the dead `slipAngle` input field), **F6** (the stale TP-02 JSDoc on the public contract), **F7** (the mixed tick read), and **L1** (the neutral-wheel freeze — a _behavioural_ gap that plausibly contributes to the user's "the implementation we have now is terrible").

---

#### 4. Risks

- **R1 (highest) — the G-2 pin.** Implementing speed in the tire without re-opening `[RED-G2]` breaks a **ratified RED contract**; the RED lane will re-pin it and the GREEN lane will fail. G-2 is currently absent from the verdict's contested set, so the amended plan would be authored against a pin nobody has been told is contested. **Mitigation: add G-2 to the contested set in the 01-planning amendment, explicitly.**
- **R2 — the overshoot tolerance is coupled to the scrub coefficient.** `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD = 0.002` (`drift.model.ts:94`) is **calibrated** to the measured chase overshoot at `c = 0.55`, and the source JSDoc records the calibration envelope as **exhausted** at `c ≈ 0.61` (only 0.00018 rad of headroom). **Any change to the scrub coefficient, the entry rate, or the chase τ forces a re-derivation of the overshoot tolerance and of the published-slip ceilings** (`0.452` / `0.493`), and by extension of the browser fixture's mirrored literals (F8). This is a **coupled blast radius**, not an isolated constant — the 01-planning author must schedule the re-derivation with the change, not after it.
- **R3 — the off-road floor masks the scrub.** On grass, `isOffRoadSpeedFloorEngaged` re-lifts the post-scrub speed to 22.25 m/s, so the speed deficit is partially cancelled. **Any new proportional-traction measurement must be taken on road** (the existing fixture lane already enforces `straightCaptureOnRoad && driftCaptureOnRoad`, so it is on-road-safe — the risk is for _new_ lanes, not the retained one).
- **R4 — the mirrors can drift silently.** The proxy floor, the deficit floor and the published-slip ceilings are hand-copied into the fixture. The `proportionalTractionLoss` check's dead 340 km/h clause will keep passing no matter what the physics does, so the fixture will report GREEN while the model is wrong. **Mitigation: remove or make reachable the vacuous clause before the amended contract is authored, and widen the seam or re-derive the mirrors with every constant change.**
- **R5 — the neutral-wheel freeze (L1) is a live behaviour, not a theoretical one.** If the amended contract keeps the latch mechanics (the verdict retains the seam mechanics but re-decides the follow/aim law), a player who releases the wheel mid-drift sees a **frozen slide and a frozen camera**. This is a plausible contributor to the user's "terrible" report and should be an explicit decision, not an accident.
- **R6 — no runtime measurement was performed.** Every claim here is a static-code read; the sensitivities in 2.A are **DERIVED** arithmetically from the source constants. Before the contract is re-authored, a measurement lane should confirm: (i) the scrub at 25 / 50 / 89 m/s at a fixed committed slip; (ii) the **zero-traction** case for `|steer| < 0.5` at top speed; (iii) the scrub at two steer magnitudes at a fixed speed; and (iv) the neutral-wheel freeze (L1). Without (ii) and (iii) the S-3 gap is argued, not measured — and the S-3 RED contract should be _born_ from the measurement.
- **R7 — the audit boundary.** This artifact adjudicates the **physics** surface only. The camera aim path, the wheel-spin idiom seam, the `browser-entry.ts` `stepPhysics` wrapper and the τ = 0.08 follow law belong to `AUDIT-CAMERA`; **S-1 and S-4 remain open until that audit returns.** The one cross-boundary fact established here is F4/F5: the physics publishes **no** traction-loss scalar, and the camera's only drift inputs today are `car.drift.pose.pose` and `car.slipAngle`. If `AUDIT-CAMERA` reaches a different conclusion about the available inputs, the merge point is the orchestrator's (per the verdict's `merge_point`), and this artifact's D4 stands only as a _recommendation_.
- **R8 — the "no child agents" constraint.** The dispatch explicitly forbade child agents, overriding the generic Tier-1 delegation mandate. This audit is therefore **single-context** and its findings are **not cross-checked by an independent scout**. Every headline claim is a direct code read with a cited line, and the three highest-stakes claims (the dead constant, the unwired `yawRate`, the test-only deficit floor) were re-verified by independent repo-wide greps in this session (F1, F4, 2.E). Residual risk: a _missed_ gap that a second pair of eyes would have found. Mitigation: the 01-planning author re-runs the F-series greps before the contract is frozen.

---

#### 5. Evidence index (files read, read-only)

| File                                                          | Read                                                                      | Purpose                                                                                  |
| ------------------------------------------------------------- | ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `plans/neatracing-voxel-racer.plans.md`                       | L9988–10120 (the `USER VERDICT 3` record), L9740–10093 (the tail packets) | the dispatch, the contested set, the plan-family conventions                             |
| `examples/neatRacing/physics/drift.model.ts`                  | **complete, 787 lines**                                                   | the model under audit                                                                    |
| `examples/neatRacing/physics/physics.step.service.ts`         | **complete, 647 lines**                                                   | the tick pipeline                                                                        |
| `examples/neatRacing/physics/drift.model.test.ts`             | L695–760                                                                  | `[RED-G2]`, `[RED-G5-SPEED-SCRUB]`, `[RED-G5-SPEED-SCRUB-2]`, `[RED-G6-COMMITTED-CHASE]` |
| `examples/neatRacing/physics/physics.step.service.test.ts`    | L690–800, L481–483                                                        | the D1.3 deficit pin; the `yawRate` test-only reads                                      |
| `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html` | L195–275, L2488–2516, L2736                                               | the mirrored literals and the `proportionalTractionLoss` check                           |
| `examples/neatRacing/browser-entry/browser-entry.ts`          | L972, L981–982                                                            | the only production reader of the drift channel                                          |
| `examples/neatRacing/collision/collision.utils.ts`            | L386–389                                                                  | the only production reader of `slipAngle` / `lateralVelocity`                            |
| `plans/neatracing-voxel-racer.research.md`                    | L1–60                                                                     | the pre-existing 14N artifact (to avoid clobbering it)                                   |

**No file was modified by this audit.** The only writes are this artifact and the `AUDIT-PHYSICS` PlanUpdate appended to `plans/neatracing-voxel-racer.plans.md`.

### B-2 — AUDIT-ENVELOPE — numeric re-derivation of the drift engage envelope

> **Provenance.** Artifact id `neatracing-voxel-racer.audit-envelope`; author `02-researching` (Tier 1),
> 2026-10-27; dispatch origin the `USER VERDICT 5` record in the tracker (wave id `14N-V5-audit-envelope`);
> subject `drift.model.ts`, `physics.step.service.ts`, `host/input.ts`, `host/mode-selector.ts`,
> `host/race-session.ts`, `physics/ai.autopilot.ts`, `browser-entry/browser-entry.ts`; constraints honoured —
> READ-ONLY, NO child agents, NO browser runs, NO Chrome MCP, no production byte touched, no git mutation.
> This is the only one of the four audit waves with **runtime** evidence (the live modules bundled headlessly
> and executed in Node), so its numbers outrank the static-code rows of its siblings.
> **Fold note (2026-10-31).** Folded here as Part B-2; the sibling file is DELETED.

**Artifact id**: `neatracing-voxel-racer.audit-envelope`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 5` record, `plans/neatracing-voxel-racer.plans.md` L10267–10291, wave id `14N-V5-audit-envelope`
**Subject**: `examples/neatRacing/physics/drift.model.ts` (787 lines), `examples/neatRacing/physics/physics.step.service.ts` (647 lines), `examples/neatRacing/host/input.ts`, `examples/neatRacing/host/mode-selector.ts`, `examples/neatRacing/host/race-session.ts`, `examples/neatRacing/physics/ai.autopilot.ts`, `examples/neatRacing/browser-entry/browser-entry.ts`
**Constraints honoured**: READ-ONLY code audit · NO child agents (caller order overrides the generic Tier-1 delegation mandate) · NO browser runs · NO Chrome MCP · no production byte touched · no git mutation.
**Sibling artifacts**: `neatracing-voxel-racer.research.md (Part B-1)` (constant inventory F-series, L-series, R6), `neatracing-voxel-racer.research.md (Part B-4)` (presentation surface). This artifact is the ENVELOPE half and does not duplicate either.

---

#### 0. Provenance, method and the evidence class used

**Method — MEASURED, not derived.** The caller's objective was a _numeric_ re-derivation, so arithmetic-by-hand was not sufficient. The real production modules were bundled headlessly and executed in Node:

```
$probe | node_modules/.bin/esbuild.cmd --bundle --platform=node --format=cjs --loader=ts \
        --outfile=$env:TEMP/drift-probe.cjs
node $env:TEMP/drift-probe.cjs
```

The probe imports the **live** `physics.step.service.ts`, `drift.model.ts`, `ai.autopilot.ts`, `collision.utils.ts` and `racing_curriculum/track/track.generator.ts`. It is a _headless Node_ run — **not** a browser run, so the NO-browser / NO-Chrome-MCP order is honoured. Bundle output and probe sources were written **outside the repository** (`%TEMP%`); no repository file other than this artifact was created or modified, and no git command was run.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. This wave is the only one of the four with **runtime** evidence, so its findings outrank the static-code findings of `AUDIT-PHYSICS` wherever they touch the same constant. Where the two disagree, the disagreement is recorded explicitly in **R6**.

**Reconciliation (why the numbers may be trusted).** The measured per-tick scrub was reconstructed by inverting the _real_ integrator (`physics.step.service.ts:567-601` → `min(89, v + 17.8·dt)`) and compared against the model's own law `computeLongitudinalSpeedScrub` (`drift.model.ts:632-646`):

```
scrub_measured − scrub_model = 0.000000000   at every sampled tick (t = 1 … 1200)
```

The reconstruction is therefore **exact**, not a fit. All figures below are measured, not derived.

**Scope boundary.** This artifact adjudicates the _envelope_ (what magnitudes of slip and traction loss are reachable, and at what speed/steer). It does **not** adjudicate the camera aim path (`AUDIT-VISUALS`), the wheel-spin idiom seam, or whether the live rAF loop really steps the drift channel (`AUDIT-MOTION`). It does adjudicate the **input** surface, because the engage threshold is only meaningful against the axis the host actually supplies.

---

#### 1. Question

> Re-derive **numerically** the drift engage envelope (the `drift.model.ts` constants) for a **full-speed held steer turn**; show **why the user sees ZERO drift**; and determine whether **proportional traction loss (S-1 / S-2 / S-3) is possible under the B1–B4 fixed constants** — or prove the model cannot lose traction at full speed.

Three sub-questions, three answers, in **D1** (envelope), **D2** (zero drift) and **D3** (S-1/S-2/S-3).

---

#### 2. Evidence

##### 2.A The committed ramp reaches its clamp in 8 ticks, at any speed

`stepDriftCommitment` (`drift.model.ts:489-560`) ramps the committed slip by
`slip −= turnSign · 3.5 · |steer| · dt` (`:520-524` on engage, `:534-536` while drifting)
and clamps it to ±0.45 (`:66`, `:551-555`).

Measured at |steer| = 1 (dt = 1/60, ramp = 0.0583333 rad/tick):

| tick  | committed slip      | published slip | note                                                |
| ----- | ------------------- | -------------- | --------------------------------------------------- |
| 1     | −0.058333           | −0.018074      | first ramp step                                     |
| 5     | −0.291667           | −0.175635      |                                                     |
| **8** | **−0.45 (clamped)** | −0.308278      | clamp hit at tick 8 (0.133 s)                       |
| 20    | −0.45               | −0.444018      |                                                     |
| 45    | −0.45               | **−0.451153**  | chase overshoot (published max 0.452, `:94`/`:128`) |

The ramp is **speed-independent** (B2). Measured at every entry speed 5 … 89 m/s, the committed slip reaches exactly −0.45 (S3 sweep). ⇒ **The slide depth is a constant 0.45 rad (25.8°) at 5 m/s and at 89 m/s alike.**

##### 2.B The traction loss IS real at full speed (the "no traction" premise is FALSE)

Road, throttle = 1, |steer| = 1, entry 89 m/s, paired against a steer-0 baseline of the same throttle:

| tick     | v (m/s)     | v (km/h)   | published slip | scrub (m/s²) | deficit (km/h) |
| -------- | ----------- | ---------- | -------------- | ------------ | -------------- |
| 1        | 89.0000     | 320.40     | −0.018074      | 0.00000      | 0.000          |
| 8        | 88.7837     | 319.62     | −0.308278      | 12.98067     | 0.779          |
| 12       | 88.6866     | 319.27     | −0.399347      | 18.76633     | 1.128          |
| 20       | 88.3010     | 317.88     | −0.444018      | 21.39787     | 2.516          |
| 30       | 87.6788     | 315.64     | −0.450654      | **21.50395** | 4.756          |
| 45       | 86.8045     | 312.50     | −0.451153      | 21.10655     | 7.904          |
| 60       | 86.0298     | 309.71     | −0.451044      | 20.72444     | 10.693         |
| 120      | 83.7221     | 301.40     | −0.450662      | 19.60453     | 19.000         |
| 300      | 80.8299     | 290.99     | −0.450169      | 18.24589     | 29.412         |
| 600      | 79.9570     | 287.85     | −0.450018      | 17.84564     | 32.555         |
| 900      | 79.8672     | 287.52     | −0.450000      | 17.80472     | 32.878         |
| **1200** | **79.8579** | **287.49** | −0.450000      | 17.80049     | **32.911**     |

The scrub peaks at **21.50 m/s²** and relaxes to **17.80 m/s²**, which is exactly the engine accel (`NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2 = 89/5 = 17.8`, `physics.step.service.ts:148`). The terminal state is an **equilibrium at 79.858 m/s = 287.49 km/h**, i.e. a **32.91 km/h** deficit against the 320.40 km/h top speed.

Closed-form check: `0.55·0.45·(v + 17.8/60)·(v/89) = 17.8` ⇒ v = 79.858 m/s. **Measured = 79.8579.** The law and the measurement agree to 4 significant figures.

⇒ **The model DOES lose traction at full speed. "The model cannot lose traction at full speed" is FALSE.** The magnitude is ~10 % of top speed, and it is a _terminal_ shift reached over 15–20 s, not a rate the player feels in a 1–3 s corner (at t = 30 the deficit is 4.76 km/h; at t = 60, 10.69 km/h).

##### 2.C The traction loss is a STEP FUNCTION of |steer| — not a proportion

Road, throttle = 1, entry 89 m/s, 600 ticks, measured deficit against the paired steer-0 baseline:

| \|steer\| | published slip | committed slip | turnSign | deficit (km/h) |
| --------- | -------------- | -------------- | -------- | -------------- |
| 0.00      | 0              | 0              | 0        | **0.000000**   |
| 0.10      | 0              | 0              | 0        | **0.000000**   |
| 0.15      | 0              | 0              | 0        | **0.000000**   |
| 0.20      | 0              | 0              | 0        | **0.000000**   |
| 0.25      | 0              | 0              | 0        | **0.000000**   |
| 0.40      | 0              | 0              | 0        | **0.000000**   |
| **0.49**  | 0              | 0              | 0        | **0.000000**   |
| **0.50**  | −0.450018      | −0.45          | 1        | **32.5392**    |
| 0.51      | −0.450018      | −0.45          | 1        | 32.5399        |
| 0.60      | −0.450018      | −0.45          | 1        | 32.5448        |
| 0.75      | −0.450018      | −0.45          | 1        | 32.5500        |
| 0.90      | −0.450018      | −0.45          | 1        | 32.5532        |
| 0.95      | −0.450018      | −0.45          | 1        | 32.5541        |
| **1.00**  | −0.450018      | −0.45          | 1        | **32.5547**    |

Two facts, both load-bearing:

1. **Below 0.5 the loss is EXACTLY ZERO** — not small, _bit-exact_ zero (the latch never opens, `drift.model.ts:514-525`). B2 is a **hard cliff**, not a fade.
2. **Above 0.5 the loss is essentially CONSTANT.** The whole 0.50 → 1.00 span is 32.5392 → 32.5547 km/h, a spread of **0.0155 km/h = 0.048 %**.

⇒ **There is no severity gradient in the traction model.** A gentle turn and a violent turn at the same speed produce the _same_ 0.45 rad slide and the _same_ terminal speed. This is the physical meaning of the user's "the tires still never lose traction".

##### 2.D The traction loss is also entry-speed-INDEPENDENT

Road, throttle = 1, |steer| = 1, 600 ticks. The committed slip is −0.45 at **every** entry speed:

| entry (m/s) | terminal v (m/s) | published slip | deficit (km/h) |
| ----------- | ---------------- | -------------- | -------------- |
| 89          | 79.9570          | −0.450018      | 32.555         |
| 80          | 79.8831          | −0.450005      | 32.821         |
| 70          | 79.7656          | −0.449984      | 33.244         |
| 60          | 79.6312          | −0.449960      | 33.728         |
| 50          | 79.4759          | −0.449933      | 34.287         |
| 40          | 79.2948          | −0.449901      | 34.939         |
| 30          | 79.0807          | −0.449864      | 35.709         |
| 20          | 78.8241          | −0.449818      | 36.633         |
| 16          | 78.7066          | −0.449798      | 37.056         |
| **15.1**    | 78.6788          | −0.449793      | 37.156         |
| **15.0**    | 78.6761          | −0.449792      | 37.166         |
| **14.0**    | 78.6457          | −0.449787      | 37.275         |
| 10          | 78.5167          | −0.449764      | 37.740         |
| 5           | 78.3364          | −0.449732      | 38.389         |

The slide depth is _identical_ at 5 m/s and 89 m/s (B2 × B3), and the deficit is a **near-constant 32.6 – 38.4 km/h** that _grows_ as the entry speed falls — i.e. it tracks the **equilibrium shift**, not the speed at which the turn began. **A real friction circle would show the opposite: the faster the entry, the harder the bite.** The tire's grip budget is speed-blind by the G-2 pin (`drift.model.ts:567-579`), so nothing in the model can express that.

##### 2.E The host supplies a TWO-STATE steer axis — the continuous envelope is unreachable

`host/input.ts:51` `PRESSED_AXIS_RATIO = 1`, `:54` `NEUTRAL_AXIS_RATIO = 0`, `:87-91`:

```
throttle: presses.up    ? 1 : 0
brake:    presses.down  ? 1 : 0
steer:    (presses.left ? 1 : 0) − (presses.right ? 1 : 0)
```

⇒ the live steer axis takes **only the values {−1, 0, +1}**. The engage threshold 0.5 (`drift.model.ts:189`) is therefore **always met** whenever a steer key is held, and the sub-threshold regime of 2.C is **unreachable on the keyboard path**.

This **falsifies the standing hypothesis** carried over from `AUDIT-PHYSICS` F-B2 _as applied to the live path_ ("a full-speed turn with |steer| < 0.5 loses exactly zero speed"). That statement remains true **as a model property** but is **not a candidate explanation for the user's report**, because the host never produces |steer| < 0.5 except 0.

Measured, with a hypothetical analog ramp (steer 0 → 1 over 60 ticks): the latch opens at the first tick where |steer| ≥ 0.5 (t = 30, committed slip −0.02917 — the ramp is seeded on the engage tick), and by t = 60 the slip is at the 0.45 clamp. **The model is written for a continuous axis that the host does not provide.**

##### 2.F The page BOOTS in AI mode, and the AI almost never crosses the threshold

- `host/mode-selector.ts:62-63` — `INITIAL_RACE_MODE = MODE_OPTION_AI` ("the page opens as a running AI-driven race").
- `host/race-session.ts:239-242` — in `ai` mode the session samples `sampleAutoInput(state.car)` **instead of** the keyboard sample.
- `physics/ai.autopilot.ts:30, :174` — `steer = clampSteer(headingError · 2.2)`.

The AI must therefore have a heading error of **0.5 / 2.2 = 0.2273 rad = 13.02°** before it engages a drift at all. Measured over a **120 s AI lap on the real demo track** (real `generateTrack`, real `createTrackSurfaceResolver`, real `stepPhysics`):

```
maxAbsSteer   = 0.9046
ticks |steer| ≥ 0.5   = 176 / 7200  =  2.444 %
ticks |steer| <  0.5  = 7024 / 7200 = 97.556 %
max published slip     = 0.466686
surface histogram      = { road 3444, curb 484, grass 3272 }   (45 % of the lap is GRASS)
|steer| histogram     = { <0.1: 1680, 0.1: 3251, 0.2: 1256, 0.3: 526, 0.4: 311,
                           0.5: 71, 0.6: 69, 0.7: 30, 0.8: 5, 0.9: 1 }
```

⇒ **What the user sees on load is an AI-driven race in which the tires are engaged 2.4 % of the time.** That is the single most likely proximate cause of "there is NO drifting".

##### 2.G On the real demo track the OFF-ROAD model dominates the drift model

Measured with the real `generateTrack({seed: 42, layoutVersion: 1, sizeBucket: 'large'})` (`browser-entry.ts:512-516`), the real `createTrackSurfaceResolver` (`browser-entry.ts:733`), the real spawn pose and full throttle from the real spawn:

```
segment widths      = [31.103, 27.808, 30.483, 31.882, 28.147, 27.449, 31.876, 32.169] m  (8 segments)
first GRASS tick    = 280   (4.67 s)  at 82.325 m/s
max speed reached   = 82.77 m/s  (297.97 km/h)   — the car never even reaches the 89 m/s cap
surface histogram   = { road 278, curb 1, grass 921 }  over 1200 ticks
```

The car leaves the ribbon **while driving dead straight**, because the demo track is a loop. Once on grass:

```
t = 420   surf grass   v = 22.25 m/s (80.10 km/h)   slip 0        com 0
t = 430   surf grass   v = 22.25                  slip −0.36523  com −0.45   pose −0.19759
t = 450   surf grass   v = 22.25                  slip −0.44941  com −0.45   lat −10.73
t = 480+  surf grass   v = 22.25                  slip −0.45     com −0.45   lat −10.75
```

Two constants do this: `NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2 = −89/2 = −44.5` (`physics.step.service.ts:173`) and `NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S = 89/4 = 22.25` (`:182`).

⇒ **The "full-speed held steer turn" the user performs in the live game is physically an 80.1 km/h turn on grass.** The deceleration the player attributes to "the tires" is, at the floor speed, `44.5 m/s²` of **off-road drag** against a drift scrub of `0.55·0.45·22.25·(22.25/89) = 1.39 m/s²` — a ratio of **32 : 1**. **The traction loss the player feels is 97 % off-road drag and 3 % tire slide.**

⇒ The claim "the tires never lose traction" is, on the live track, _almost_ literally true: the surface, not the tire, is the whole story.

##### 2.H The neutral wheel FREEZES the slide; only a countersteer unwinds it

`stepDriftCommitment` computes `countersteering = turnSign !== 0 && Math.sign(steer) === −turnSign` (`drift.model.ts:511-512`). `Math.sign(0) = 0 ≠ −turnSign`, so **releasing the key is not countersteering** — the "drifting" branch runs the entry ramp with `|steer| = 0` (`drift.model.ts:534-536`), i.e. `slip −= turnSign·3.5·0·dt`, i.e. **no change**. Measured (steer = 1 for 60 ticks, then steer = 0):

```
t = 60   com −0.45   slip −0.451044   scrub 2.92   deficit 10.693 km/h
t = 120  com −0.45   slip −0.450662   scrub 1.80   deficit 19.000 km/h
t = 300  com −0.45   slip −0.450169   scrub 0.45   deficit 29.412 km/h
t = 600  com −0.45   slip −0.450018   scrub 0.05   deficit 32.555 km/h
```

⇒ **Releasing the steering key leaves a permanent 0.45 rad slide that keeps scrubbing the car down to the 287.5 km/h terminal speed, forever.** The player cannot straighten out of a slide by lifting the wheel.

The **countersteer** does unwind it — and, measured, it unwinds it _identically_ for every magnitude tested:

| countersteer | com @63 | com @66 | com @69 | com @72      | lat @63 | lat @66 | lat @69 |
| ------------ | ------- | ------- | ------- | ------------ | ------- | ------- | ------- |
| −0.40        | −0.325  | −0.200  | −0.075  | **0**        | −36.273 | −25.916 | −14.715 |
| −0.50        | —       | —       | —       | **0** (t=90) |         |         |         |
| −0.89        | —       | —       | —       | **0** (t=90) |         |         |         |
| −0.90        | −0.325  | −0.200  | −0.075  | **0**        | −33.933 | −22.875 | −11.518 |
| −1.00        | —       | —       | —       | **0** (t=90) |         |         |         |

The committed-slip unwind is **entirely** the `EXIT_RATE = 2.5 rad/s` ramp (`:537-549`): 0.45 / 2.5 = 0.18 s = 10.8 ticks, identical for all magnitudes. The 70 m/s² `COUNTERSTEER_AUTHORITY` (`:225`) only changes the _lateral-velocity_ decay, by 2–3 m/s out of 15–40. **The authority is decorative in the recovery path.** And `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS = 8` (`:251`) has **exactly one occurrence in the whole repository — its own declaration** (repo-wide grep over `*.ts`): it is dead.

After the unwind the latch is **released with `reArmed = false`** (`:541-544`), so the car **cannot re-engage a drift until the wheel is returned to neutral** (|steer| < 0.15, `:526-530`).

##### 2.I The reachable slip set is EIGHT discrete values, and the only control is tap length

Because the entry ramp is 0.0583333 rad/tick and the clamp is 0.45, and because the neutral wheel freezes the slide (2.H), the _reachable_ committed slips are exactly:

| key held (ticks) | committed slip | terminal v | note          |
| ---------------- | -------------- | ---------- | ------------- |
| 1                | −0.058333      | 88.952     |               |
| 2                | −0.116667      | 88.905     |               |
| 3                | −0.175000      | 88.857     |               |
| 5                | −0.291667      | 88.763     |               |
| 7                | −0.408333      | 83.910     |               |
| **8 and up**     | **−0.45**      | **79.957** | clamp         |
| released / 0     | 0              | 89.000     | never engaged |

⇒ the slip domain is **{0} ∪ {0.0583333·k : k = 1…7} ∪ {0.45}** — eight values, quantised to 1/60 s of key-hold time. This is an **accidental** control channel (a by-product of the L1 freeze), not a designed envelope, and it is not humanly modulable.

##### 2.J The visual surface under-reports the physical slide by 2×

- `computeDriftSlideProxyRad` (`:655-659`) = `|slip| · |GAIN − 1|` = `0.45 · 0.5` = **0.225 rad (12.9°)** against a physical slide of **0.45 rad (25.8°)**.
- The body yaw the renderer draws is the pose spring `slip·0.5 + 0.08·articulation` (`:727-777`, GAIN = 0.5 at `:332`). Measured: **−0.153 rad (8.8°)** on road at full lock; **−0.205 rad (11.7°)** on grass at the floor.
- The camera rigid-mirrors the body (`browser-entry.ts:972` → `carPoseBag.driftYawOffsetRad = car.drift.pose.pose`).

⇒ the "odd camera angle" the user names is the **8.8° pose spring**, and the body-vs-travel angle is 25.8° − 8.8° ≈ 17° on road / ≈ 14° on grass. **The physical slide IS wired into the travel direction** (`physics.step.service.ts:470-486` integrates x, y along `velocityHeading = nextHeading + nextSlipAngle`) — so it is _not_ a camera trick in the sense of "the physics is absent"; it is a camera trick in the sense that **the rendered body yaw under-reports the physical slide by 2×**. Adjudicated in `AUDIT-VISUALS`; recorded here only as the envelope-side cause of the perception.

---

#### 3. Decision

##### D1 — The envelope, re-derived numerically

| quantity                                         | value                                                 | source                       |
| ------------------------------------------------ | ----------------------------------------------------- | ---------------------------- |
| engage threshold                                 | 0.5                                                   | `drift.model.ts:189`         |
| entry ramp                                       | 3.5 rad/s × \|steer\|, **speed-independent**          | `:201`, `:520-536`           |
| ticks to the 0.45 clamp at \|steer\| = 1         | **8** (0.133 s)                                       | measured                     |
| reachable committed slips                        | **{0} ∪ {0.0583333·k, k=1…7} ∪ {0.45}**               | measured (2.I)               |
| chase time constant                              | 0.045 s (published slip reaches −0.4511 in ~45 ticks) | `:219`, measured             |
| scrub law                                        | `0.55·\|published slip\|·v·(v/89)` m/s²               | `:632-646`                   |
| peak scrub at full lock                          | **21.50 m/s²** (at t = 30)                            | measured                     |
| terminal drift speed (engine-balanced)           | **79.858 m/s = 287.49 km/h**                          | measured, analytic-identical |
| terminal speed deficit                           | **32.91 km/h** (10.3 % of the 320.40 km/h cap)        | measured                     |
| traction loss for \|steer\| < 0.5                | **exactly 0.000000**                                  | measured (2.C)               |
| traction loss spread over \|steer\| 0.5 → 1.0    | **0.0155 km/h (0.048 %)**                             | measured (2.C)               |
| traction loss spread over entry speed 5 → 89 m/s | 32.56 → 38.39 km/h (equilibrium shift, not a bite)    | measured (2.D)               |
| slide depth at 5 m/s vs 89 m/s                   | **identical (−0.45)**                                 | measured (2.D)               |

##### D2 — Why the user sees ZERO drift (ranked by measured contribution)

1. **The page boots in AI mode** (`mode-selector.ts:63`, `race-session.ts:239-242`) and **the AI engages a drift on 2.444 % of ticks** (2.F). For 97.6 % of the run the tires are _bit-exactly_ not slipping. **This is the strongest and most likely proximate cause.**
2. **On the real demo track the car is on grass 921/1200 ticks and pinned at the 80.1 km/h off-road floor** (2.G). There, the off-road decel (−44.5 m/s²) is **32 : 1** larger than the drift scrub (1.39 m/s²). Whatever the player feels is surface drag, not tire slip.
3. **The model has no severity gradient** (2.C): every turn above the threshold yields the same 0.45 rad slide and the same terminal speed. There is nothing to "feel" as a deeper or a harder drift.
4. **The traction loss is a terminal shift, not a rate** (2.B): a 1–3 s corner yields 4.8 – 19.0 km/h, which reads as "the car slowed a bit", not as "the tires let go".
5. **The rendered body yaw under-reports the physical slide by 2×** (2.J) — the perception of "just a camera angle" (cross-ref `AUDIT-VISUALS`).
6. **Residual, not adjudicated here**: whether the live rAF loop actually steps the drift channel (`AUDIT-MOTION`).

**Correction to the dispatch premise.** "The model cannot lose traction at full speed" is **FALSE**: the model loses 32.91 km/h at full lock on road, and the scrub reaches 21.50 m/s² — larger than the engine's 17.8 m/s². The premise survives only in the _live-track_ sense (2.G), where the surface dominates, and in the _AI-mode_ sense (2.F), where the threshold is never crossed.

##### D3 — S-1 / S-2 / S-3 under the frozen B1–B4 constants

| proposal                                                                      | verdict                                           | proof                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **S-1** traction loss proportional to **speed** (a real friction circle)      | **IMPOSSIBLE without re-opening G-2 (B4) and B2** | the grip budget is speed-blind by contract (`:567-579`); the slide depth is 0.45 rad at every speed (2.D); the observable deficit is a constant equilibrium shift, not a bite (2.D)                                                                                      |
| **S-2** traction loss proportional to the **slide magnitude**                 | **FORMALLY PRESENT, DEGENERATE IN PRACTICE**      | the law is already `∝ \|published slip\|` (`:632-646`) — but the reachable slip set is **8 discrete values** quantised to 1/60 s of tap length (2.I), because B2 gates it, B3 clamps it, and the L1 freeze is the only (accidental) way to land on an intermediate value |
| **S-3** traction loss proportional to the **steering effort / turn severity** | **IMPOSSIBLE**                                    | measured: 0.50 → 32.5392 km/h, 1.00 → 32.5547 km/h — a **0.048 %** spread (2.C). The scrub reads the _clamped published_ slip (B3), not the steer and not the yaw rate (`:632-646`)                                                                                      |

⇒ **Under B1–B4 the drift model can emit exactly ONE traction-loss magnitude: a step function at |steer| ≥ 0.5.** There is no continuous, steer-proportional, speed-proportional traction loss available. Delivering S-1/S-2/S-3 requires re-opening **B2** (the speed-independent entry rate), **B3** (the 0.45 clamp) **and G-2 (B4)** — and **G-2 is NOT in the verdict's contested set and must be added to it.** G-2 is the binding constraint: without it no friction circle can exist at any speed.

##### D4 — Numbered gaps (file:line)

| #      | gap                                                                                                                                                                                                                                                                                                   | file:line                                                                                                                 | confidence | evidence                                  |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------- |
| **1**  | The traction loss is a **step function** of \|steer\|, not a proportion: 0.49 → exactly 0 km/h, 0.50 → 32.5392 km/h, 1.00 → 32.5547 km/h (0.048 % spread). No severity gradient exists for the player to feel.                                                                                        | `drift.model.ts:189`, `:632-646`, `:66`                                                                                   | 0.95       | measured (2.C)                            |
| **2**  | **S-3 is impossible** under the frozen constants. The scrub reads the _clamped published_ slip, so above the 0.45 clamp the law is speed-only.                                                                                                                                                        | `drift.model.ts:66`, `:632-646`, `:551-555`                                                                               | 0.93       | measured (2.C) + source                   |
| **3**  | **G-2 (the speed-independent grip budget) is the binding constraint on S-1 and is NOT in the verdict's contested set** — it must be added, or S-1 can never be delivered.                                                                                                                             | `drift.model.ts:567-579`                                                                                                  | 0.90       | static + measured (2.D)                   |
| **4**  | The **engage threshold is a bit-exact cliff**: below 0.5 the traction loss is not "small", it is **0.000000** (the latch never opens). No fade-in exists.                                                                                                                                             | `drift.model.ts:189`, `:514-525`                                                                                          | 0.95       | measured (2.C)                            |
| **5**  | The **host supplies a two-state steer axis** {−1, 0, +1}; the model's continuous-control envelope is unreachable from the keyboard, so the sub-threshold regime is dead code in practice and every turn is a full-lock 0.45 rad slide.                                                                | `host/input.ts:51`, `:54`, `:87-91`                                                                                       | 0.95       | static (2.E)                              |
| **6**  | The **page boots in AI mode** and the **AI engages a drift on 2.444 % of ticks** (heading error must exceed 13.02°). The zero-drift report is fully explained for the default boot.                                                                                                                   | `host/mode-selector.ts:62-63`, `host/race-session.ts:239-242`, `physics/ai.autopilot.ts:30`, `:174`, `drift.model.ts:189` | 0.90       | static + measured (2.F)                   |
| **7**  | The **off-road model dominates the drift model on the demo track**: grass 921/1200 ticks, the 22.25 m/s floor pins the car at 80.1 km/h, and the off-road decel is **32 : 1** larger than the drift scrub at that speed. The "full-speed turn" is an 80 km/h turn on grass.                           | `physics.step.service.ts:173`, `:182`, `browser-entry.ts:512-516`, `:733`                                                 | 0.92       | measured (2.G)                            |
| **8**  | The **neutral wheel FREEZES the slide** (L1 confirmed): releasing the key leaves a permanent 0.45 rad slide that keeps scrubbing to the terminal speed. The player cannot straighten out by lifting the wheel.                                                                                        | `drift.model.ts:511-512`, `:534-536`                                                                                      | 0.93       | measured (2.H)                            |
| **9**  | The **countersteer authority is decorative** in the recovery path: the 2.5 rad/s exit ramp unwinds the slide identically at \|steer\| = 0.4 and 1.0; the 70 m/s² authority moves the lateral velocity by only 2–3 m/s of 15–40. And `COUNTERSTEER_HOLD_TICKS` has **no reader anywhere in the repo**. | `drift.model.ts:214`, `:225`, `:237`, `:251`, `:537-549`                                                                  | 0.88       | measured (2.H) + repo-wide grep           |
| **10** | The **reachable slip domain is 8 discrete values** quantised to 1/60 s of key-hold time — an accidental by-product of the L1 freeze, not a designed control surface. S-2 is therefore formally present but unusable.                                                                                  | `drift.model.ts:66`, `:201`, `:520-536`, `:551-555`                                                                       | 0.92       | measured (2.I)                            |
| **11** | The **rendered body yaw under-reports the physical slide by 2×** (proxy 0.225 rad vs physical 0.45 rad; pose 0.153 rad). The physical slide _is_ wired into the travel direction, so this is a _presentation_ under-report, not an absent physics.                                                    | `drift.model.ts:332`, `:655-659`, `:727-777`, `browser-entry.ts:972`                                                      | 0.90       | static + measured (2.J) → `AUDIT-VISUALS` |
| **12** | The **traction loss has no published channel**: the scrub is the only tire-load term and is never surfaced on the drift channel, so neither the HUD nor the composer can express "the tires let go". (Inherited F5, now with a measured magnitude.)                                                   | `physics.step.service.ts:419-427`, `drift.model.ts:282`                                                                   | 0.80       | static                                    |

##### D5 — Confidence summary

| finding                                                               | confidence | provenance                             | act                                                                                                                                 |
| --------------------------------------------------------------------- | ---------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| The model loses 32.91 km/h at full lock on road                       | **0.95**   | runtime (measured, reconciled to 1e-9) | ACT                                                                                                                                 |
| The loss is steer-independent above 0.5                               | **0.95**   | runtime                                | ACT                                                                                                                                 |
| The loss is exactly 0 below 0.5                                       | **0.95**   | runtime                                | ACT                                                                                                                                 |
| The slide depth is speed-independent (0.45 at 5 and 89 m/s)           | **0.95**   | runtime                                | ACT                                                                                                                                 |
| The AI engages a drift 2.444 % of the time; the page boots in AI mode | **0.90**   | runtime + static                       | ACT (see D2.6 for the residual)                                                                                                     |
| The off-road model dominates on the demo track                        | **0.92**   | runtime                                | ACT (cross-ref `AUDIT-MOTION`)                                                                                                      |
| The neutral-wheel freeze (L1)                                         | **0.93**   | runtime                                | ACT                                                                                                                                 |
| `COUNTERSTEER_HOLD_TICKS` is dead                                     | **0.90**   | static (repo-wide grep, 1 hit)         | ACT                                                                                                                                 |
| The countersteer authority is decorative                              | **0.88**   | runtime                                | DELEGATE — a reviewer should confirm the intended feel contract before the constant is deleted                                      |
| The 2× visual under-report                                            | **0.90**   | static + runtime                       | DELEGATE — `AUDIT-VISUALS` owns the decision                                                                                        |
| "The user sees zero drift" — full causal attribution                  | **0.78**   | runtime (parts 1–5) + residual         | DELEGATE — `AUDIT-MOTION` (does the live loop step the drift?) and `AUDIT-VISUALS` (is the slide visible?) must close the last 22 % |

---

#### 4. Risks

- **R1 — The zero-drift attribution is 78 % confident, not higher.** D2.1 (AI mode) and D2.2 (off-road dominance) are each sufficient to produce the report; D2.3–D2.5 are contributory. The residual 22 % sits in the motion loop (`AUDIT-MOTION`) and the presentation (`AUDIT-VISUALS`). **Do not close VERDICT 5's O_3 on this artifact alone.**
- **R2 — G-2 is a ratified contract, not a free constant.** `[RED-G2]` asserts `computeGripBudgetFactor(0.9, 1) === computeGripBudgetFactor(0.9, 0.2)`. Delivering S-1 requires **re-opening a ratified pin** and re-writing that test. The verdict's contested set (B1–B4) does not include it; **the orchestrator must add it**, or S-1 is undeliverable by construction.
- **R3 — The measured terminal speed (79.8579 m/s) is the equilibrium, not a transient.** A fixture that asserts the deficit at a fixed 600 ticks (as the D1.3 pin does) will read 32.555 km/h, not 32.911 km/h. Any new RED contract must pin the _equilibrium_, or it will be a vacuous floor assertion (the inherited F-E criticism).
- **R4 — The off-road floor can MASK a drift defect.** A car pinned at 22.25 m/s shows a scrub of 1.39 m/s²; a traction-loss regression would be invisible on the demo track. New traction contracts must run on a **road-only** surface resolver, or they will be vacuous.
- **R5 — The neutral-wheel freeze makes the drift STICKY, which is the opposite failure mode from the user's report and could mask a fix.** A fix that only touches the ramp will not change the reported symptom, because the symptom is dominated by D2.1/D2.2. **Order the fixes: mode/AI → off-road → envelope.**
- **R6 — `AUDIT-PHYSICS` F-B2 must be re-read as a _model_ property, not a _live-path_ explanation.** This wave's 2.E falsifies the live-path reading of it (the host never produces |steer| < 0.5). Rank: runtime (this wave) > static (`AUDIT-PHYSICS`).
- **R7 — The off-road published-slip ceiling (0.493, `:163`) is reachable on the demo track** (the AI measured max 0.466686) and the grass run reached −0.467015, so the grass path has its own envelope, not a rescaled road envelope. Any traction contract must state the surface.
- **R8 — Measurement hygiene.** The probe wrote only to `%TEMP%`; no repository byte and no git ref was touched. Reproduce with: `esbuild --bundle --platform=node --format=cjs --loader=ts` over a stdin probe importing `examples/neatRacing/physics/physics.step.service.ts` + `examples/racing_curriculum/track/track.generator.ts`, then `node` the bundle. **The reconciliation invariant to re-check is `scrub_measured − scrub_model == 0` at every tick** — if it is non-zero, the integrator assumption (`min(89, v + 17.8·dt)`, `physics.step.service.ts:567-601`) has drifted and every deficit figure here is void.

---

#### 5. Evidence index (file:line)

| file:line                                                         | fact used                                                                                                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/neatRacing/physics/drift.model.ts:66`                   | `MAX_SLIP_ANGLE_RAD = 0.45` (B3)                                                                                                             |
| `examples/neatRacing/physics/drift.model.ts:94`, `:128`, `:163`   | published-slip overshoot 0.002 / max 0.452 / off-road max 0.493                                                                              |
| `examples/neatRacing/physics/drift.model.ts:169`, `:177`          | grip falloff threshold 0.2, rate 2                                                                                                           |
| `examples/neatRacing/physics/drift.model.ts:184`                  | `MIN_SPEED_MPS = 15` (B1)                                                                                                                    |
| `examples/neatRacing/physics/drift.model.ts:189`                  | `STEER_ENGAGE_THRESHOLD = 0.5` (B2)                                                                                                          |
| `examples/neatRacing/physics/drift.model.ts:195`                  | `STEER_NEUTRAL_THRESHOLD = 0.15`                                                                                                             |
| `examples/neatRacing/physics/drift.model.ts:201`, `:214`          | `ENTRY_RATE 3.5`, `EXIT_RATE 2.5` rad/s                                                                                                      |
| `examples/neatRacing/physics/drift.model.ts:219`                  | `SLIP_RESPONSE_TAU_S = 0.045`                                                                                                                |
| `examples/neatRacing/physics/drift.model.ts:225`, `:237`, `:251`  | countersteer 70 m/s² / min steer 0.9 / **HOLD_TICKS 8 (dead)**                                                                               |
| `examples/neatRacing/physics/drift.model.ts:282`                  | `SPEED_SCRUB_COEFF_PER_S = 0.55`                                                                                                             |
| `examples/neatRacing/physics/drift.model.ts:314`, `:332`          | `POSE_STEER_YAW_RAD 0.08`, `VISUAL_YAW_GAIN_RAD 0.5`                                                                                         |
| `examples/neatRacing/physics/drift.model.ts:464-472`              | `createZeroDriftChannelState` — `reArmed: true` at spawn                                                                                     |
| `examples/neatRacing/physics/drift.model.ts:489-560`              | `stepDriftCommitment`: MIN_SPEED gate `:494-501`; ENGAGE `:514-525`; RE-ARM `:526-530`; ramp `:531-536`; unwind `:537-549`; clamp `:551-555` |
| `examples/neatRacing/physics/drift.model.ts:567-579`              | `computeGripBudgetFactor` — **`speedPercent` accepted and ignored (G-2)**                                                                    |
| `examples/neatRacing/physics/drift.model.ts:591-618`              | `computeCountersteerAuthority`                                                                                                               |
| `examples/neatRacing/physics/drift.model.ts:632-646`              | `computeLongitudinalSpeedScrub` — `0.55·\|slip\|·v·speedPercent`                                                                             |
| `examples/neatRacing/physics/drift.model.ts:655-659`              | `computeDriftSlideProxyRad` = `\|slip\|·\|GAIN−1\|`                                                                                          |
| `examples/neatRacing/physics/drift.model.ts:670-725`              | `computeNextLateralVelocity` — MIN_SPEED hard zero, chase, countersteer, hard zero                                                           |
| `examples/neatRacing/physics/drift.model.ts:727-777`              | `stepDriftPoseSpring` — pose target `slip·0.5 + 0.08·articulation`, **unclamped**                                                            |
| `examples/neatRacing/physics/physics.step.service.ts:133`         | fixed timestep 1/60                                                                                                                          |
| `examples/neatRacing/physics/physics.step.service.ts:142`, `:148` | max speed 89; accel 89/5 = 17.8                                                                                                              |
| `examples/neatRacing/physics/physics.step.service.ts:173`, `:182` | off-road decel −89/2 = −44.5; off-road floor 89/4 = 22.25                                                                                    |
| `examples/neatRacing/physics/physics.step.service.ts:192`         | `STEERING_RATE = 2` rad/s                                                                                                                    |
| `examples/neatRacing/physics/physics.step.service.ts:419-427`     | the scrub call (tick-start slip, tick-end speed, tick-start speedPercent)                                                                    |
| `examples/neatRacing/physics/physics.step.service.ts:455-461`     | the pose spring step                                                                                                                         |
| `examples/neatRacing/physics/physics.step.service.ts:470-486`     | motion integrated along `velocityHeading = nextHeading + nextSlipAngle`                                                                      |
| `examples/neatRacing/physics/physics.step.service.ts:506-513`     | `resolveSpeedPercent = clamp(speed/89)`                                                                                                      |
| `examples/neatRacing/physics/physics.step.service.ts:567-601`     | `integrateLongitudinalSpeed` — `min(89, v + 17.8·dt)` on road                                                                                |
| `examples/neatRacing/host/input.ts:51`, `:54`, `:87-91`           | `PRESSED_AXIS_RATIO = 1`; the two-state steer axis                                                                                           |
| `examples/neatRacing/host/mode-selector.ts:57-63`                 | `INITIAL_RACE_MODE = 'ai'`                                                                                                                   |
| `examples/neatRacing/host/race-session.ts:239-242`                | ai mode samples the autopilot instead of the keyboard                                                                                        |
| `examples/neatRacing/physics/ai.autopilot.ts:30`, `:174`          | `STEER_GAIN_PER_RAD = 2.2`; `steer = clampSteer(headingError · 2.2)`                                                                         |
| `examples/neatRacing/browser-entry/browser-entry.ts:512-516`      | the demo track generation spec                                                                                                               |
| `examples/neatRacing/browser-entry/browser-entry.ts:733`          | `createTrackSurfaceResolver(trackSpec.segments)`                                                                                             |
| `examples/neatRacing/browser-entry/browser-entry.ts:972`          | `driftYawOffsetRad = car.drift.pose.pose`                                                                                                    |
| `examples/neatRacing/browser-entry/browser-entry.ts:1147-1149`    | the keyboard / autopilot / `stepPhysics` seams                                                                                               |

---

#### 6. Merge notes for Agent Zero

1. **The dispatch premise is corrected**: the model _does_ lose traction at full speed (32.91 km/h, 21.50 m/s² peak scrub). The "no traction" experience is explained by **AI-mode boot (2.444 % engagement)**, **off-road dominance (32 : 1)**, **the absent severity gradient**, and the **2× visual under-report** — not by an empty envelope.
2. **G-2 must be added to the contested set.** Without re-opening the speed-independent grip budget, S-1 is undeliverable. S-3 is undeliverable under B1–B4 **at all**, and S-2 exists only as an 8-value accidental domain.
3. **Fix order**: mode/AI → off-road → envelope. A pure envelope fix will not move the reported symptom (R5).
4. **The shared plan tracker was deliberately NOT edited** by this wave — four audit waves are running in parallel against the same plan tail and a concurrent append risks a merge conflict. This sibling artifact is the durable record; the orchestrator owns the `PlanUpdate` row.
5. **No child agents were dispatched** — the caller's "NO child agents" order overrides the generic Tier-1 delegation mandate, consistent with `AUDIT-PHYSICS` R8.
6. **No browser and no Chrome MCP were used.** The measurement was a headless Node run of the production modules.

### B-3 — AUDIT-INPUT — live keyboard steering chain vs the fixture's scripted steer injection

> **Provenance.** Artifact id `audit-input` (VERDICT-5 wave, plan record `14N-V5-audit-input`); author
> `02-researching` (Tier 1); dispatch origin Agent Zero (USER VERDICT 5, 2026-10-27) — audit wave
> `14N-V5-audit-input`; subject the `examples/neatRacing` live input path (browser key handlers → steer value
> → the drift engage/hold/commit gates) compared against
> `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html`; constraints honoured — READ-ONLY code only, NO
> browser, NO Chrome MCP, NO child agents, no implementation.
> **Fold note (2026-10-31).** Folded here as Part B-3; the sibling file is DELETED.

**Artifact id:** `audit-input` (VERDICT-5 wave, plan record `14N-V5-audit-input`)
**Author:** `02-researching` (Tier 1)
**Dispatch origin:** Agent Zero (USER VERDICT 5, 2026-10-27) — audit wave `14N-V5-audit-input`
**Subject:** `examples/neatRacing` live input path — browser key handlers → steer value → drift engage/hold/commit gates — compared against `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html`
**Constraints honoured:** READ-ONLY code only (no source, fixture, plan or bundle was modified); NO browser; NO Chrome MCP; NO child agents; no implementation.

---

#### 0. Provenance & method

| Item            | Value                                                                                                                                                                                                                                                                                                                    |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Plan            | `plans/neatracing-voxel-racer.plans.md` (active phase 1 "Graphics engine + manual keyboard driving" [WIP], active step **14N** [WIP], active slice `P1-drift-feel-green`)                                                                                                                                                |
| Context load    | Cortex MCP `freshness_check` → `{"fresh":true,"stale":[]}`; `search_corpus` (RRF + DiskANN, `dense_state: warm`); `load_chunk` on chunk_index 604/605 = the USER VERDICT 5 record and the `14N-V5-audit-input` row                                                                                                       |
| Plan tail       | 120-line tail of `plans/neatracing-voxel-racer.plans.md` (sibling AUDIT-PHYSICS rows F-F0..F-F8, L1, R-2, R-3, R-8, MERGED AUTHOR BRIEF D_A..D_N, USER VERDICT 4, USER VERDICT 5) — read to avoid duplicating sibling rows                                                                                               |
| Native fallback | `view` / `grep` on known file paths only (documented degradation: Cortex `load_chunk` returns plan prose, not whole source files). All findings below were cross-checked against a second read of the same region.                                                                                                       |
| Bundle check    | `examples/neatRacing/assets/neatracing.bundle.js` and `docs/assets/neatracing.bundle.js` — 398 699 B each, written 2026-10-27 09:56, i.e. **after** the drift sources (2026-10-26 23:42 / 23:58) → the served bundle on disk IS current. The bundle source map contains the RE-ARM token and the engage-threshold JSDoc. |

**Source-of-truth order applied:** runtime/validation > static code > comments/docs > external. Sibling findings already in the plan are **cited, not re-proved** (F-F0..F-F8, L1, R-2, R-3, R-8).

---

#### 1. Question

1. What does an **interactive, full-speed, held turn** actually feed into the drift model on the live path — threshold value, ramp ticks, MIN_SPEED gate direction, RE-ARM / hold semantics?
2. Do the engage gates **ever** open on the live path?
3. How does the fixture's "scripted steer injection" differ from the live path — is it a separate injection point, or the same chain under different preconditions?

---

#### 2. Evidence

##### 2.1 The live chain (read end to end)

```
window 'keydown'/'keyup'  (browser-entry.ts:775-798 / :801-825)
  → presses.left / presses.right                      (browser-entry.ts:775-798)
    → sampleKeyboardInput(deps.presses)              (race-session.ts:242)
      → steer = left?1:0 − right?1:0                 (input.ts:89-91, PRESSED_AXIS_RATIO=1 at :51)
        → stepPhysics(steer, throttle, brake, …)      (race-session.ts:252)
          → updateDriftChannel(...)                   (physics.step.service.ts:396-460)
            → committed.slip / committed.reArmed        (drift.model.ts:495-545)
```

##### 2.2 The fixture's injection point

`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:818-822` — `dispatchKey()` builds a **real** `KeyboardEvent{code}` and dispatches it on `window`. There is no private steer setter, no direct model poke, no mocked input module. The fixture therefore drives the **identical** live chain; only its _preconditions and phase sequence_ differ (see G-7).

##### 2.3 The engage gate, evaluated for a held key

| Gate                                 | Source                                                                        | Value            | Held-key result                        |
| ------------------------------------ | ----------------------------------------------------------------------------- | ---------------- | -------------------------------------- |
| `\|steer\| ≥ STEER_ENGAGE_THRESHOLD` | `drift.model.ts:189`, tested at `:513`                                        | 0.5              | **PASS** (live \|steer\| = 1)          |
| `speed > MIN_SPEED_MPS`              | `drift.model.ts:184`, tested at `:495`                                        | 15 m/s (54 km/h) | **PASS** above 54 km/h; **FAIL** below |
| RE-ARM token `reArmed`               | `drift.model.ts:464-470` (spawn), cleared at `:499`, re-latched at `:522-524` | TRUE at spawn    | **PASS** on the first tick             |
| `committed.slip === 0`               | `drift.model.ts:509-512`                                                      | 0 at spawn       | **PASS** on the first tick             |

**Conclusion (2.3):** the four gates are all satisfied **on the first tick** of a held key above 54 km/h. The gates are _not_ the break.

---

#### 3. Numbered gaps

> Severity is my own ranking (CRITICAL = explains the user's symptom outright; HIGH = materially changes the drive feel or hides evidence; MEDIUM/LOW = latent or hygiene).
> Every row is `file:line`. Rows marked _(cited)_ are already in the plan via a sibling audit and are **not** re-proved.

##### G-1 — CRITICAL — the live path never feeds a non-zero steer: the boot mode is `ai`, so the keyboard is dead

- `examples/neatRacing/host/mode-selector.ts:70` — `INITIAL_RACE_MODE = MODE_OPTION_AI` → the page boots in **`ai`** mode.
- `examples/neatRacing/host/mode-selector.ts:263-267` — `select.value = mode` mirrors `ai` into the DOM select; the user _sees_ a mode dropdown set to AI.
- `examples/neatRacing/host/race-session.ts:239-242` — the frame branch prefers the autopilot sample whenever `mode === 'ai'`:
  `deps.modeSelector.mode === 'ai' && deps.sampleAutoInput !== undefined ? deps.sampleAutoInput(state.car) : deps.sampleKeyboardInput(deps.presses)`
  → `sampleKeyboardInput` is **never called** in the boot mode.
- `examples/neatRacing/browser-entry/browser-entry.ts:1148` — `sampleAutoInput` is always supplied, so the AI branch is always the one taken in `ai`.
- `examples/neatRacing/browser-entry/browser-entry.ts:775-798` — the keydown handler still latches `presses.left/right`; those flags are **dead** in the boot mode.
- `examples/neatRacing/browser-entry/browser-entry.ts:323-330` — `createBootShellOverlays` never calls `setMode`, so nothing flips the mode at boot.

**Proof that the engage gates never open on the default live path:** the live steer is _exactly_ `0` for every tick (G-2), so the `|steer| ≥ 0.5` gate at `drift.model.ts:513` can never be satisfied. The gates are not closed — **they are never fed**.

##### G-2 — CRITICAL — the live keyboard axis is binary ±1; there is no analog ramp, so the 0.5 threshold cannot discriminate

- `examples/neatRacing/host/input.ts:51` — `PRESSED_AXIS_RATIO = 1`.
- `examples/neatRacing/host/input.ts:89-91` — `steer: (presses.left ? 1 : 0) - (presses.right ? 1 : 0)`.
- `examples/neatRacing/physics/drift.model.ts:189` — `STEER_ENGAGE_THRESHOLD = 0.5`.

The live steer can only be `0`, `+1` or `−1`. The nominal "engage band 0.4–0.6" is unreachable from a keyboard, so the threshold is a constant, not a control: a key press either fully engages or does nothing.

##### G-3 — HIGH — the commit ramp is **speed-independent** and saturates in 8 ticks at any speed above 15 m/s

- `examples/neatRacing/physics/drift.model.ts:201` — `ENTRY_RATE_RAD_PER_S = 3.5`.
- `examples/neatRacing/physics/drift.model.ts:66` — `MAX_SLIP_ANGLE_RAD = 0.45` (the clamp).
- `examples/neatRacing/physics/drift.model.ts:169` — `GRIP_FALLOFF_THRESHOLD_RAD = 0.2`.
- `examples/neatRacing/physics/drift.model.ts:517-521` (engage ramp) and `:528-530` (drift ramp) — `slip -= turnSign · 3.5 · |steer| · dt`, `dt = 1/60`.

**Derived:** per tick `|Δslip| = 3.5/60 = 0.0583333 rad`. Grip falloff begins at **tick 4** (0.2 rad); the 0.45 rad clamp is reached at **tick 8** (0.133 s) — identically at 16 m/s and at 89 m/s. The live driver has no speed-dependent build-up: the slide is a step response to the key, not a curve.

##### G-4 — HIGH — the MIN_SPEED gate is **not** inverted, but it is a cliff, not a taper

- `examples/neatRacing/physics/drift.model.ts:184` — `MIN_SPEED_MPS = 15` (54 km/h).
- `examples/neatRacing/physics/drift.model.ts:495-501` — `if (!(speed > 15)) { … slip = 0; turnSign = 0; reArmed = false; return …; }`.

**Derived:** the gate **opens** as the speed rises (correct direction — a slow car cannot drift) and **closes** below 15 m/s, where it also _clears the RE-ARM token_ (`:499`). There is no taper: at 15.0 m/s the lateral velocity is exactly `0` (`drift.model.ts:679-681`) and one tick later it is full. The live driver therefore experiences a binary "drift on/off at 54 km/h" boundary.

##### G-5 — HIGH — the RE-ARM token can only be re-latched while **no slide is latched**, and only by a fully released key

- `examples/neatRacing/physics/drift.model.ts:464-470` — the spawn channel: `reArmed: true` (wired at `physics.step.service.ts:311`).
- `examples/neatRacing/physics/drift.model.ts:509-524` — the re-latch sits **inside** `if (turnSign === 0)` and requires `|steer| < 0.15` (`:195`).
- `examples/neatRacing/physics/drift.model.ts:522-524` — the re-latch itself.

**Consequence:** a live keyboard can re-arm (steer `0 < 0.15`) but only in the window where no slide is latched. Once a slide is latched, releasing the key neither re-arms nor unwinds (G-6). A second drift therefore requires: countersteer to unwind past `SNAP_BACK_ZERO_SLIP_RAD = 0.02` (`drift.model.ts:257`) **and then** a ≥1-tick neutral-steer window.

##### G-6 — HIGH — a held _or released_ key never releases the slide: the only live unwind is an exact opposite-key countersteer _(cited: plan row L1 — from the input side)_

- `examples/neatRacing/physics/drift.model.ts:506` — `countersteering = turnSign !== 0 && Math.sign(steer) === -turnSign`. With `steer = 0`, `Math.sign(0) = 0 ≠ -turnSign` → `countersteering = false` → `drifting = true` (`:507`) → the ramp **keeps growing**.
- `examples/neatRacing/physics/drift.model.ts:526-530` (ramp branch) vs `:531-545` (unwind branch).
- `examples/neatRacing/physics/drift.model.ts:237` — `COUNTERSTEER_MIN_STEER = 0.9`; the keyboard's ±1 clears the magnitude bar, but the **direction** must be exactly opposite.

**Live-specific consequence:** the interactive driver's natural "lift the wheel to straighten up" gesture **keeps the slide at 0.45 rad** with its latched turn direction, and the ramp continues to push it. The only unwind is pressing the opposite key (the fixture's P2-RELEASE idiom, `p1-turn-drift-green-smoke.html:2187`).

##### G-7 — CRITICAL — the fixture's "scripted steer injection" is **not a separate path**; the divergence is preconditions and phase sequence

- `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:818-822` — `dispatchKey` → `window.dispatchEvent(new KeyboardEvent('keydown', { code, bubbles: true, cancelable: true, composed: true }))` → the **live** `keydownListener` (`browser-entry.ts:775-798`) → `presses` → `sampleKeyboardInput` (`input.ts:83-92`) → steer ±1.

⇒ **The fixture exercises the identical chain.** Its green drift capture (committed slip ≈ 0.452 rad, a real speed deficit) therefore _proves the model and the chain work_. The live failure is a **precondition** failure, not a wiring failure.

The live path does **not** reproduce the fixture's preconditions:

| #   | Fixture precondition                                                                                             | Fixture site                                                                         | Live path                                                                                                           |
| --- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| a   | `setManualMode()` — bypasses the mode gate                                                                       | `:1895`, `:2106`                                                                     | user must find the dropdown (`mode-selector.ts:246-262`); boot is `ai` (G-1)                                        |
| b   | `reentryCarOnRibbon()` — two-step `ai`→`manual` toggle so the restart carrier re-seeds the car **on the ribbon** | `:2148`, `:2167`, `:2226`; impl `:1240-1250`                                         | no equivalent: the user commits the steer **wherever the car happens to be** (typically mid-corner, often off-road) |
| c   | `setFeelThrottle(true)` — throttle explicitly held                                                               | `:2149`; impl `:1212-1216`                                                           | throttle depends on the user holding **W** _at the same time_ as the steer                                          |
| d   | **Uncapped** 210-tick straight accel before the commit                                                           | `:2154` (P1-STRAIGHT), `:2168-2177` (P2-ACCEL 140 + P2-COMMIT 30 + P2-HOLD 40 = 210) | no warm-up; the user's key press lands at an arbitrary speed                                                        |
| e   | Capture pair taken at the **end** of a 40-tick **neutral-steer** hold (the key is released at `:2175-2177`)      | `:2175-2177`                                                                         | the user typically **holds** the steer into the capture                                                             |

**Derived for (d):** 210 ticks ÷ 60 × `ACCEL = 17.8` m/s² (`physics.step.service.ts:148`) → **62.3 m/s (224 km/h) on-ribbon**; the capture is therefore taken **on-road at speed**, on the axis. The live turn is taken at whatever speed the user was already doing, usually mid-corner.

**Derived for (e):** the fixture **never holds the steer** through the capture — the committed window is 30 ticks (0.5 s) and the drift capture follows 40 ticks of _released_ steer. There is no fixture evidence for "steer held through the corner", which is precisely the live case.

##### G-8 — HIGH — no road containment on the live path: a held full-lock turn leaves the 30 m ribbon in ≈ 0.46 s, and the off-road floor then masks the only traction-loss evidence

- `examples/neatRacing/physics/collision.utils.ts:234-256` — `resolveTrackBoundaryCar` (the boundary bounce) is referenced **only** by `collision.utils.ts` and `collision.utils.test.ts`; it is **never called** by `browser-entry.ts` or `race-session.ts`.
- `examples/neatRacing/physics/collision.utils.ts:197` — `resolveCarCollisions` (car-car) is likewise unwired.
- `examples/neatRacing/browser-entry/browser-entry.ts:733`, `:1149` — only `createTrackSurfaceResolver` (`collision.utils.ts:165-169`) is wired, i.e. the surface _type_ is known but no containment is applied.
- `examples/neatRacing/physics/physics.step.service.ts:173` — `OFF_ROAD_DECEL_UNITS_PER_S = -44.5`; `:182` — `OFF_ROAD_SPEED_FLOOR_UNITS_PER_S = 22.25`; `:436-445` — the floor re-lifts the **post-scrub** speed.
- `examples/racing_curriculum/track/track.generator.ts:14-18`, `:30` — the large bucket is radius 240 m (±18 %) with a 30 m road (±10 %).

**Derived:** a full-lock turn has a **speed-independent radius of 44.5 m** (yaw ∝ `speedPercent`, speed ∝ `speedPercent`, `physics.step.service.ts:192` `STEERING_RATE = 2`) against a ~240 m corner radius and a 15 m half-width — so a held full lock at top speed leaves the ribbon in ≈ **28 ticks (0.46 s)**. Off-road the speed settles at the 22.25 m/s floor (80.1 km/h), where the scrub `0.55·|slip|·v·(v/89)` (`drift.model.ts:264`, applied at `physics.step.service.ts:419-427`) collapses to ≈ **1.4 m/s²** instead of ≈ 20 m/s² on-road, and is then re-lifted by the floor. _(Cross-ref sibling R-3.)_

The fixture documents this itself as the reason it needed the re-entry: `p1-turn-drift-green-smoke.html:1218-1239` — "the recorded browser lane enters the 14N block **OFF-ROAD** … an off-road car can never produce the ON-ROAD straight capture the D1.3 capture pair needs — off-road the speed settles at the 22.25 m/s floor (80.1 km/h)".

##### G-9 — MEDIUM — the bounce reset of the drift channel is never wired, so the "a bounce releases the slide" rule does not exist at runtime

- `examples/neatRacing/physics/collision.utils.ts:406-450` — `zeroBouncedDriftChannel()` clears the channel and the RE-ARM token; referenced **only** inside `collision.utils.ts`.
- `examples/neatRacing/physics/drift.model.ts:455-462` — the model documents the bounce release that the runtime never performs.

Not the live break (the live path never bounces, because the boundary bounce is unwired — G-8), but it is an unwired published contract, and it means the live RE-ARM token can only ever be cleared by the MIN_SPEED cliff (G-4).

##### G-10 — MEDIUM — input latching hazards that the fixture structurally cannot catch

- `examples/neatRacing/host/race-session.ts:236-243` — the input is sampled **once per rAF frame** and replayed to every catch-up tick of that frame; `examples/neatRacing/host/input.ts:83-92` reads the latched booleans.
- `examples/neatRacing/browser-entry/browser-entry.ts:775-798` / `:801-825` — the booleans are latched on keydown/keyup only: **no `blur` handler, no `visibilitychange` handler**, no `event.repeat` handling.

**Consequences:** (a) a press-and-release inside one frame (≤ ~16 ms, or up to 100 ms of catch-up per `host/render-loop.utils.ts:48`) is **invisible** to the physics; (b) if the window loses focus while a key is held, the keyup is never delivered and `presses.left` stays TRUE → a **permanent full-lock steer** with no way to recover except clicking back into the page. The fixture always dispatches and releases keys in order, so neither hazard is exercised.

##### G-11 — MEDIUM — the HUD advertises a control the boot mode ignores, and there is no traction readout at all

- `examples/neatRacing/host/hud.ts:132` — the controls hint is `'WASD / ARROWS to drive'`, mounted as a static label (`hud.ts:392-393`).
- `examples/neatRacing/host/mode-selector.ts:70` — the boot mode is `ai`, so the hint is **false on load** (G-1).
- `examples/neatRacing/host/hud.ts:383-393` — the HUD mounts exactly four labels (speed, lap, time, hint). There is **no slip / traction / grip indicator**, so the only user-visible traction evidence is the raw speed number, and the user cannot correlate it with the slide.

##### G-12 — MEDIUM — in the default `ai` mode the autopilot's own steer barely clears the engage threshold, and its throttle/brake schedule fights the slide

- `examples/neatRacing/physics/ai.autopilot.ts:167` — `steer: clampSteer(headingError * NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD)`; gain `2.2` at `:31`.
- `examples/neatRacing/physics/ai.autopilot.ts:160-162` (throttle cut above 0.35 rad), `:163-165` (brake above 0.7 rad).

**Derived:** `|steer| ≥ 0.5` requires `|headingError| ≥ 0.2273 rad`; the AI cuts the throttle above 0.35 rad and brakes above 0.7 rad. In the boot mode the drift gate therefore opens only in a narrow error band in which the car is simultaneously coasting or braking — any slide seen in `ai` is a transient artifact of the AI's own cornering error, not a driver-visible drift.

##### G-13 — LOW — the served bundle's cache-buster predates the entire drift wave

- `examples/neatRacing/index.html:95-99` — `bundlePath` is built with the frozen query `?v=20260928-1`; `:119` — `script.src = bundlePath`.

The file on disk is current (398 699 B, 2026-10-27 09:56, after the 2026-10-26 sources; the source map contains the RE-ARM token and the engage-threshold JSDoc), so this is **not** the break — but any browser holding the pre-drift response for the same URL keeps serving the old bundle. Confidence that this _contributed_ to the user's observation: **0.40** (no browser was run; cache state is unverifiable by read-only code).

##### G-14 — LOW — the fixture could not read the HUD speed and fell back to the debug seam

- `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:1465-1467` — "the HUD text read returned 0 at the capture points in the recorded runs" → the fixture fell back to the `window.neatRacingDriftState` seam.
- `examples/neatRacing/browser-entry/browser-entry.ts:980-986` — the 5-field debug seam (`slip`, `speed`, `reArmed`, `phase`, …).
- `examples/neatRacing/host/hud.ts:335-342` — the HUD labels carry **no DOM id**, so the fixture had to reach them by structural query.

This corroborates G-11: the live HUD is not a reliable traction observation surface, so "the tires never lose traction" is unobservable in the live build without the debug seam.

---

#### 4. Decision

**What an interactive full-speed held turn feeds:** `steer = ±1` (binary, `input.ts:51`, `:89-91`), sampled once per frame (`race-session.ts:236-243`) and replayed to each catch-up tick. Above 15 m/s the engage gates are all satisfied on the **first tick**; the slip then ramps at **0.0583333 rad/tick** (3.5 rad/s × 1/60, `drift.model.ts:201`, `:517-521`, `:528-530`) to the 0.45 rad clamp in **8 ticks / 0.133 s**, independent of speed; the RE-ARM token is **not** re-latchable while the slide is latched (`:509-524`); and the slide is unwound **only** by an exact opposite-key press (`:506`, `:237`).

**Do the engage gates ever open on the live path?** **No — not on the default path, and not because a gate is closed.** The page boots in `ai` (`mode-selector.ts:70`) and the frame branch prefers the autopilot sample (`race-session.ts:239-242`), so `sampleKeyboardInput` is never called and the live steer is exactly `0` for every tick — the `|steer| ≥ 0.5` gate at `drift.model.ts:513` is never fed. In `manual` mode the gates **do** open on the first tick above 54 km/h; the live failure then becomes a _feel + evidence_ failure (speed-independent 8-tick snap to 0.45 rad, no re-arm while latched, release only by opposite key, and an off-road floor that masks the speed scrub).

**Live vs fixture:** the fixture's steer injection is **not** a separate path — `dispatchKey` (`p1-turn-drift-green-smoke.html:818-822`) dispatches a real `KeyboardEvent` into the live listener. The divergence is entirely in **preconditions and phase sequence** (G-7 a–e): manual mode, an on-ribbon re-seed, an explicit throttle, a 210-tick uncapped straight to 62.3 m/s, and a **neutral-steer** capture window. The live user has none of these, and the live user **holds** the steer through the corner — a case the fixture never covers.

**Merge routing:** the plan file is **not** edited by this audit (Agent Zero owns the merge). This artifact is the deliverable.

---

#### 5. Risks / gaps in this audit

- **R-A (single-context, no cross-check).** The user's "NO child agents" order overrides the generic Tier-1 delegation mandate, so this audit was not independently re-checked by a second scout. A missed gap in the live input path is possible. _(Same disclosure shape as the sibling's R-8.)_
- **R-B (no runtime evidence).** No browser was run, so every claim here is static. The "gates never open" conclusion is a control-flow proof (mode → branch → steer value), not a captured frame.
- **R-C (cache state unverifiable).** G-13's contribution to the user's observation is confidence 0.40; the on-disk bundle is current, so a stale cached bundle is _possible_ but unproven.
- **R-D (boundary respected).** The camera aim law, the wheel-spin idiom, the `browser-entry` `stepPhysics` wrapper, the numeric envelope and the presentation inventory are owned by AUDIT-CAMERA / AUDIT-MOTION / AUDIT-ENVELOPE / AUDIT-VISUALS and are **not** re-litigated here.
- **R-E (no duplication).** F-F0..F-F8, L1, R-2, R-3 and the MERGED AUTHOR BRIEF D_A..D_N were read from the plan tail and are cited rather than re-proved; G-6 and G-8 are the input-side views of L1 and R-3 respectively.

### B-4 — AUDIT-VISUALS — read-only code audit of the tire/traction PRESENTATION layer

> **Provenance.** Artifact id `neatracing-voxel-racer.audit-visuals`; author `02-researching` (Tier 1),
> 2026-10-27; dispatch origin the `USER VERDICT 5` record in the tracker (`audit_waves[3]` =
> `14N-V5-audit-visuals`); subject `examples/neatRacing/engine/passes/car.pass.ts` +
> `examples/neatRacing/host/hud.ts` + `examples/neatRacing/car/wheel.steer.ts` +
> `examples/neatRacing/car/generate-car.ts` + `examples/neatRacing/voxel/wheel.mount.utils.ts` + the
> presentation-facing half of `browser-entry.ts` and `physics.step.service.ts`; constraints honoured —
> READ-ONLY, NO child agents, NO browser runs, NO Chrome MCP, no production byte touched.
> **Fold note (2026-10-31).** Folded here as Part B-4; the sibling file is DELETED.

**Artifact id**: `neatracing-voxel-racer.audit-visuals`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 5` record, `plans/neatracing-voxel-racer.plans.md` L10284-10286, `audit_waves[3]` = `14N-V5-audit-visuals`
**Subject**: `examples/neatRacing/engine/passes/car.pass.ts` (the car/tire draw pass) + `examples/neatRacing/host/hud.ts` (the HUD) + `examples/neatRacing/car/wheel.steer.ts` + `examples/neatRacing/car/generate-car.ts` + `examples/neatRacing/voxel/wheel.mount.utils.ts` + the presentation-facing half of `examples/neatRacing/browser-entry/browser-entry.ts` and `examples/neatRacing/physics/physics.step.service.ts`
**Constraints honoured**: READ-ONLY code audit · NO child agents (per the dispatch's `O_1` "NO child agents inside each") · NO browser runs · NO Chrome MCP · no production byte touched.

---

#### 0. Provenance, method and the naming deviation

**Method.** Every claim below is a _static-code_ read plus repo-wide readership greps. Line numbers are cited against the files as they stand on disk (2026-10-27). No runtime measurement was performed (the dispatch forbids browser runs; the `O_2` "Chrome MCP BANNED" order is non-negotiable). Numbers quoted as "derived" are arithmetic from source constants and are marked **DERIVED**.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. There is no runtime evidence in this audit, so every finding is `static-code` ranked. Where a source JSDoc disagrees with the code, the **code wins** and the doc drift is reported as a literal gap.

**Naming deviation (disclosed).** The contract names `plans/<PlanName>.research.md`. That exact path was **already occupied** by a different, closed 14N artifact (the Step 14N drift-feel _integration coupling_ design, dated 2026-10-24 — now **Part A** of this merged file), and the drift-feel brief was likewise occupied (now **Part B-6**). Clobbering a closed, dated artifact would fuse two different questions under one date. This artifact therefore followed the repo's own topic-suffix family convention as a **new sibling** — `plans/neatracing-voxel-racer.audit-visuals.research.md` (the file created on 2026-10-24, DELETED in the 2026-10-31 fold) — and is now folded as **Part B-4** of this merged file. It remains a `plans/*.research.md` sibling of the tracker and is **not** under `docs/research/`. The orchestrator may rename or merge it without loss.

**Scope boundary.** This is the **PRESENTATION** half. The camera aim path (`browser-entry.ts` L935-937 / L1000-1012), the `cameraPose` trap, the wheel-spin _reset idiom_, the `stepPhysics` wrapper and the tau = 0.08 follow law are `AUDIT-CAMERA`'s scope. The traction law itself (`drift.model.ts` + the scrub in `physics.step.service.ts`) is `AUDIT-PHYSICS`' scope, and the _live input chain_ is `14N-V5-audit-input`'s scope. This artifact adjudicates exactly one question: **what VISUALLY expresses losing traction, what feeds it, and which published drift outputs are unwired.** Where a claim crosses a boundary it is flagged as such.

**Non-duplication.** The settled `AUDIT-PHYSICS` facts (F1-F8, B1-B4, D1-D6, R1-R8) are **not** re-proved. The one physics-side fact this audit _must_ re-state (the scrub is computed and discarded) is cited as a **cross-reference**, not as a new finding, and is marked as such in G-V2.

---

#### 1. Question

The `USER VERDICT 5` dispatch asks:

> INVENTORY the tire/traction PRESENTATION layer: what VISUALLY expresses losing traction (tire yaw-follow, skid/gradient, smoke, speed loss HUD) and what feeds it; enumerate EVERY published-but-unwired drift output (scrub, `car.yawRate`, `pose.poseRate`, seam fields) so the implementation wave knows exactly which wires exist and which must be CREATED.

The user's own verdict supplies the reason the question matters:

> "there are no drifting physics in place, I can take turns at full speed and the car never drifts, it's only confusing due to the odd camera angle. The current implementation is either not wired or a complete failure!"

So the audit must answer three things:

1. **What does the player actually SEE** when the tires lose (or fail to lose) traction?
2. **What is the upstream source of each of those visuals** - is it the traction law, the committed slip, the raw input, or the speed?
3. **Which published drift outputs have no production reader at all**, and therefore which wires the implementation wave must **CREATE** rather than **CONNECT**.

---

#### 2. Evidence

##### 2.A - The presentation layer, in full _(confidence 0.96, source: static-code)_

A repo-wide grep over `examples/neatRacing/**/*.ts` for `smoke|skid|trail|tireMark|particle` returns **ZERO product hits** (only unrelated prose such as "smoke test" and "trailing"). A second grep for `audio|squeal|AudioContext` also returns **ZERO hits**. The presentation surface of the whole build is therefore:

| Presentation element                      | Exists? | Source of its value                    | Reads the traction law?           |
| ----------------------------------------- | ------- | -------------------------------------- | --------------------------------- |
| Car body yaw (the drift crab)             | YES     | `car.drift.pose.pose`                  | **NO** - reads the committed slip |
| Wheel chain yaw (tire follow)             | YES     | the body's drift-yawed matrix          | **NO** - inherits the crab        |
| Front-wheel steer yaw                     | YES     | `clamp(steerAxis) * 35 deg`            | **NO** - an INPUT ECHO            |
| Wheel spin (rotation)                     | YES     | post-traction `car.speed` / radius     | indirectly (a SPEED effect)       |
| Tire "gradient"                           | YES     | a fixed sun direction + a boolean      | **NO** - pure lighting            |
| HUD speed                                 | YES     | post-traction `car.speed`              | indirectly (a SPEED effect)       |
| HUD lap / time / controls hint            | YES     | race rules                             | n/a                               |
| Camera aim                                | YES     | `car.heading` (+ the contested offset) | **NO** - `AUDIT-CAMERA`           |
| Smoke / skid marks / dust / trails        | **NO**  | -                                      | -                                 |
| Tire squeal / audio                       | **NO**  | -                                      | -                                 |
| Grip bar / slip indicator / traction lamp | **NO**  | -                                      | -                                 |

**The headline: there is exactly ONE drift-driven visual in the entire build (the pose crab), and it is not downstream of the traction law.** Every other traction-adjacent visual is driven by SPEED or by the RAW INPUT. Nothing in the presentation layer can distinguish "the tires lost grip" from "the player is holding the stick" or "the car is braking".

##### 2.B - The tire yaw-follow chain, traced end to end _(confidence 0.95, source: static-code)_

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

- The **body** and the **wheels** rotate by the SAME offset, so the tires follow the body crab (this is the AC-078 "tire follow" the dispatch names). Good - the _plumbing_ of the tire follow is complete and correct.
- But the **source** is the pose spring, whose target is the **COMMITTED slip** (`nextCommitted.slip`), a _latched command_ - not the traction loss, and not even the **published** slip (the actual lateral-velocity angle that the scrub law reads).
- The pose is a **second-order spring** (`omega = 18`, `zeta = 0.35`, `drift.model.ts:759-764`) that **overshoots** its target. The `car.pass.ts` JSDoc (L460-495) states the declared target envelope is `0.45 * 0.5 + 0.08 = 0.3050 rad` and that the _rendered_ offset is bounded only by measurement.
- The `car.pass.ts` JSDoc also states that the offset is applied **verbatim** and is "never re-derives or rescales it" - i.e. the pass is a pure sink, and all the semantics live in the physics.

**(2) The front-wheel STEER yaw (AC-044).** This is a completely separate path:

```
browser-entry.ts:964               fillCarRenderPose(..., boundSession.state.input?.steer)
generate-car.ts:70                 pose.steerAngleRad = computeWheelSteerAngleRad(steerInput ?? 0)
wheel.steer.ts:55-57               computeWheelSteerAngleRad(steerAxis) = clamp(steerAxis, -1, 1) * NEATRACING_MAX_WHEEL_STEER_RAD (35 deg)
car.pass.ts:1661-1667              rotationY(NEATRACING_WHEEL_STEER_YAW_MATRIX, isFrontWheel ? (car.steerAngleRad ?? 0) : 0)
```

`computeWheelSteerAngleRad` is a **pure linear function of the raw steering axis**, with **no gate of any kind**. The rear wheels are excluded by `isFrontWheel` and receive **no** steer yaw at all.

**The consequence is the sharpest presentation-layer finding in this audit.** The visible front-wheel turn is an **unconditioned echo of the player's input**. At `steer = 0.10` the front wheels are already at 3.5 deg; at `steer = 0.49` they are at 17.2 deg - a dramatic-looking turn - while the drift model has latched **nothing** (the engage threshold is `0.5`) and the traction loss is **exactly zero**. The tires therefore _look_ like they are working at any input above roughly `0.03`, and the player is given **no other feedback whatsoever**. This is a direct, code-level explanation of "the tires still never lose traction" from the player's point of view: the one thing the player can read on the tires is the input, echoed back.

The sampled axis is the **same** object the physics replays (`race-session.ts:239-252` samples once per frame and replays the same `PhysicsStepInput` into every catch-up tick), and the source comment at `race-session.ts:237-238` says so explicitly: _"keeps `state.input` current for the composer on zero-step frames, so the front-wheel steer pose reflects the held key immediately."_ So the visual and the physics read the same number - the divergence is not the number, it is the **gating**: the visual has none of the drift model's gates (engage threshold `0.5`, MIN_SPEED `15 m/s`, the re-arm latch).

##### 2.C - The tire gradient: a LIGHTING effect wearing a traction costume _(confidence 0.96, source: static-code)_

The dispatch names the "skid/gradient" as one of the traction visuals. It is not one.

**The seam** (`car.pass.ts:1516-1536`) is:

```ts
function feedTireGradientUniforms(enabled: boolean): void {
  gl.uniform1f(litUniforms.uApplyTireGradient, enabled ? 1.0 : 0.0);
  if (enabled) {
    gl.uniform1f(
      litUniforms.uTireGradientRadius,
      NEATRACING_WHEEL_HUB_HEIGHT_M,
    );
    gl.uniform1f(
      litUniforms.uTireGradientBrightFactor,
      NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR,
    ); // 1.7
    gl.uniform1f(
      litUniforms.uTireGradientDarkFactor,
      NEATRACING_TIRE_GRADIENT_DARK_FACTOR,
    ); // 0.35
    gl.uniform1f(
      litUniforms.uTireGradientThreshold,
      NEATRACING_TIRE_GRADIENT_RUBBER_MAX_CHANNEL /
        NEATRACING_CAR_COLOR_BYTE_MAX,
    ); // 80/255
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
2. **The shader comment states the design intent in plain words**: the highlight _"stays fixed while the wheel voxels spin through it"_. The **uniform feed** is spin-independent (true - it is a constant), and the **rendered result** animates with the spin (also true - the tread voxels move through a fixed band). Both statements are in the source; the visual outcome is the _product_ of the two.
3. **Therefore the perceived "tire activity" is proportional to SPEED, not to GRIP.** The tread slides through the light band faster the faster the wheel spins. A wheel that is _slipping_ (spinning faster than the road) and a wheel that is _rolling freely_ (spinning at the road speed) are visually **identical**; a car under full traction loss and a car coasting at the same speed look the same on the tire.

Two further properties of the element matter for the implementation wave:

- **The AC-088 residual is a static floor, not a traction quantity.** `NEATRACING_TIRE_GRADIENT_MIN_SPREAD = 0.5` (`car.pass.ts:411`) is the _minimum non-degenerate spread_ between the bright and dark ends. Its own JSDoc says: _"This is a FLOOR, not a live value: the production spread is 1.35, which already clears the floor with margin ... The rendered gradient itself is a fragment-stage effect, so it is NOT measurable from JavaScript; the visible outcome is carried by the AC-090 user visual gate, and this constant pins the seam INPUTS."_ So the Step 14G residual "tire-gradient MAGNITUDE measures 0" is a **static spread** - it is **not** a traction magnitude and cannot become one without a new input.
- **The wheel chain's world orientation does move the band across the tread.** The chain is `BODY_MODEL (incl. the drift yaw) * translate(socket) * rotY(mount yaw) * rotY(steer yaw) * rotZ(spin)`. The drift yaw (AC-078) and the steer yaw therefore rotate the tire in world space, which shifts _where_ on the tread the bright band sits. A player can therefore read "the tires are turned" from the gradient - but that is the **steer echo** again, not a grip cue.

##### 2.D - The HUD: three readouts, no traction _(confidence 0.96, source: static-code)_

`host/hud.ts` is the entire HUD. Its state contract (`hud.ts:142-147`) is:

```ts
export interface NeatRacingHudState {
  readonly speedUnitsPerS: number; // the current forward speed
  readonly raceState: RaceRulesState; // lap + time
}
```

Its DOM surface is **exactly three readouts plus one static hint** (`hud.ts:154-158`, `hud.ts:393`, `hud.ts:401-403`): `speedLabel` (cyan), `lapLabel` (magenta), `timeLabel` (cyan), and a fixed `HUD_CONTROLS_HINT_TEXT` written once at construction.

The one and only update path is `race-session.ts:261-264`:

```ts
deps.hud.update({
  speedUnitsPerS: state.car.speed, // POST-traction speed
  raceState: state.rules,
});
```

`state.car.speed` is the **post-scrub** speed (`physics.step.service.ts:471`), so the speed number **does** fall when the scrub bites - the loss is _implicitly_ visible. But:

- there is **no** grip/slip/traction readout, bar, lamp or gauge;
- there is **no** comparison against a target speed, so the player cannot see "I am losing 12 km/h to the slide";
- there is **no** colour or flash change on the speed label when grip is lost;
- the HUD contract has **no field** that could carry a traction value - adding one is a **contract change**, not a wiring change.

**There is no speed-loss HUD.** The speed number is the only traction-caused pixel in the HUD, and it is indistinguishable from braking.

##### 2.E - The wheel spin: the only traction-caused pixel, and it is a speed effect _(confidence 0.92, source: static-code)_

`browser-entry.ts:948-955` accumulates the visual spin from `computeVisualWheelSpinDeltaRad` (`wheel.mount.utils.ts:419-437`):

```
omega = clampVisualWheelSpinOmega(trueWheelSpinOmega(groundSpeedMPerS, wheelRadiusM))
      = clamp(v / r, +-NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S)   // 2 rev/s = 12.566 rad/s
```

`groundSpeedMPerS` is `car.speed`, the **post-traction** speed. So the wheels **do** visibly slow down when the scrub bites. But:

- it is a **speed** effect - braking produces the identical visual;
- it is **clamped** at 2 rev/s (the AC-015 strobing cap), so the visible delta is bounded;
- a **slipping** wheel spins _faster_ than the road speed, not slower - the model has no slip-ratio concept on the visual side, so a sliding tire renders as a _slower_ wheel. The visual is backwards for a true slide.

##### 2.F - EVERY published-but-unwired drift output _(confidence 0.96, source: static-code)_

`PhysicsCarState` publishes (`physics.step.service.ts:78-130`):
`x, y, heading, speed, yawRate, slipAngle, lateralVelocity, drift{committed{slip,turnSign,reArmed}, pose{pose,poseRate,articulation}}, tick`.

A repo-wide grep over `examples/neatRacing/**/*.ts` for `\.committed\.(slip|turnSign|reArmed)|pose\.(poseRate|articulation)|\.yawRate` returns hits in **exactly three files, all of them `.test.ts`**: `physics/collision.utils.test.ts`, `physics/drift.model.test.ts`, `physics/physics.step.service.test.ts`. **Zero production readers.** (This independently re-confirms `AUDIT-PHYSICS` F4; it is not a new finding, and it is listed here because the dispatch explicitly asks for the complete enumeration.)

| #   | Published output               | Where written                                                                                   | Production reader?                                                                                | What it is                                                                              |
| --- | ------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| 1   | `car.yawRate`                  | declared `physics.step.service.ts:95`, init `:308`, written `:479` (`yawRatePerTick`, rad/tick) | **NONE** - test only (`physics.step.service.test.ts:481-483`)                                     | the per-tick turning rate - the natural "speed x turning" and proportional-camera input |
| 2   | `car.drift.pose.poseRate`      | `drift.model.ts:759-763`, returned `:766`                                                       | **NONE** - test only                                                                              | the visible yaw **rate** (rad/s) of the crab                                            |
| 3   | `car.drift.pose.articulation`  | `drift.model.ts:735-753`, returned `:766`                                                       | **NONE** - test only (consumed only by the model's own pose target)                               | the steer-yaw articulation, armed by the committed slip                                 |
| 4   | `car.drift.committed.slip`     | `physics.step.service.ts:396-403`                                                               | **NONE** - test only                                                                              | the **latched command** (the commanded slide)                                           |
| 5   | `car.drift.committed.turnSign` | `drift.model.ts` latch                                                                          | **NONE** - test only                                                                              | the latched turn direction                                                              |
| 6   | `car.drift.committed.reArmed`  | `drift.model.ts` latch                                                                          | **NONE** - test only                                                                              | the re-arm token                                                                        |
| 7   | the **scrub** (traction loss)  | computed `physics.step.service.ts:419-427`                                                      | **NOT PUBLISHED AT ALL**                                                                          | the literal traction loss; only its _effect_ (a reduced speed) survives                 |
| 8   | `car.drift.pose.pose`          | `physics.step.service.ts:454-459`                                                               | **YES** - `browser-entry.ts:972` (the crab) and `:981` (the seam)                                 | the **only** production consumer of the drift channel that reaches a visual             |
| 9   | `car.slipAngle`                | `physics.step.service.ts:474`                                                                   | **YES** - `collision.utils.ts:386-389` (the bounce recompose) + `browser-entry.ts:980` (the seam) | the published slip; **not read by any visual**                                          |
| 10  | `car.lateralVelocity`          | `physics.step.service.ts:476`                                                                   | **YES** - `collision.utils.ts:386-389` only                                                       | the lateral velocity; **not read by any visual**                                        |

**A second, sharper framing of item 8.** The pose and the traction law read **two different slip variables**:

- the **pose** target is `0.5 * COMMITTED slip + 0.08 * articulation` (`drift.model.ts:756-758`, fed `nextCommitted.slip` at `physics.step.service.ts:457`);
- the **scrub** reads the **PUBLISHED** slip (`state.slipAngle`, the tick-start `atan2` of the lateral velocity) at `physics.step.service.ts:422`.

The committed slip is a _latched command_ that saturates at the `0.45 rad` clamp; the published slip is the _measured_ lateral-velocity angle that the chase has not yet converged to. They are equal only in steady state. Concretely: on the first tick after the commit latches, the committed slip is already at its target while the published slip is still near zero, so **the car crabs while the traction loss is still zero** - and the pose spring (`omega = 18`) reaches its full target in well under a second. This is the mechanism by which the _one_ drift visual can be at its full envelope with **no** grip lost, and it is the mechanical form of the user's complaint that the drift is "a poorly implemented camera trick, that has no connection with the drifting". **There is no shared scalar (S-4) between the visual and the tire.**

##### 2.G - The debug seam _(confidence 0.95, source: static-code)_

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

The interface (`browser-entry.ts:112-152`) is documented as a _"read-only, EXACTLY five-field view of the live drift channel"_. **There is no field for** traction loss / scrub, yaw rate, pose rate, committed slip, turn sign, articulation, or a grip factor.

The W4b packet plans a **6th field**, and `AUDIT-CAMERA` F8 accepts it **only if the follow state is ONE SHARED OBJECT**; the field is written at **render** rate but its source updates at **tick** rate, so the new 44th check must tolerate **60 Hz quantization** above 60 Hz displays.

**This audit's constraint on that decision:** the 6th field must be the **shared traction scalar** (the merged brief's D_B / D_J), **not** merely the camera follow offset. If the 6th field carries only the follow state, the fixture would keep measuring the _camera_ and never the _tires_, and the S-4 "one shared scalar drives both" requirement would be re-broken **at the seam** - the exact defect this wave exists to close.

##### 2.H - The wire ledger: what EXISTS vs what must be CREATED _(confidence 0.94, source: static-code)_

This is the table the implementation wave asked for.

| Wire                                                             | Status                        | Where it lives                                                                       | Reads the traction law?       |
| ---------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------------------------------------ | ----------------------------- |
| committed slip -> pose spring -> body crab                       | **EXISTS**                    | `physics.step.service.ts:454` -> `browser-entry.ts:972` -> `car.pass.ts:1617`        | NO (reads the committed slip) |
| body crab -> wheel chain (tire follow, AC-078)                   | **EXISTS**                    | `car.pass.ts:1636-1670`                                                              | NO (inherits the crab)        |
| raw steer axis -> front-wheel steer yaw (AC-044)                 | **EXISTS**                    | `browser-entry.ts:964` -> `wheel.steer.ts:55` -> `car.pass.ts:1661`                  | NO (an input echo)            |
| post-traction speed -> wheel spin                                | **EXISTS**                    | `browser-entry.ts:948-955` -> `wheel.mount.utils.ts:419`                             | indirectly (a speed effect)   |
| post-traction speed -> HUD speed                                 | **EXISTS**                    | `race-session.ts:261-264` -> `hud.ts:401`                                            | indirectly (a speed effect)   |
| fixed sun -> tire gradient                                       | **EXISTS but VACUOUS**        | `car.pass.ts:1516` + the fragment shader L325-333                                    | NO (lighting only)            |
| pose + slip + speed + tick -> 5-field seam                       | **EXISTS but INCOMPLETE**     | `browser-entry.ts:978-984`                                                           | NO (no traction field)        |
| **traction-loss scalar published on the physics state**          | **MUST BE CREATED**           | `physics.step.service.ts` (the scrub is computed at `:419-427` and discarded)        | -                             |
| **car-pass traction channel (per car, and optionally per tire)** | **MUST BE CREATED**           | `CarPassCarState` (`car.pass.ts:435-497`) has **no** such field                      | -                             |
| **HUD traction/grip readout + contract field**                   | **MUST BE CREATED**           | `hud.ts:142` (`NeatRacingHudState`)                                                  | -                             |
| **smoke / skid marks / dust / trails subsystem**                 | **MUST BE CREATED FROM ZERO** | no source file exists                                                                | -                             |
| **tire-squeal / audio channel**                                  | **MUST BE CREATED FROM ZERO** | no `AudioContext` anywhere                                                           | -                             |
| **per-tire grip differentiation (the rears breaking away)**      | **MUST BE CREATED**           | the gradient seam carries a boolean and 4 globals; the shader has no per-wheel input | -                             |
| **6th seam field carrying the shared traction scalar**           | **MUST BE CREATED**           | `browser-entry.ts:978-984`                                                           | -                             |

##### 2.I - Literal GAPS

**G-V1 - The car-pass contract has NO traction channel.** _(confidence 0.97, gap type: missing)_
`CarPassCarState` (`car.pass.ts:435-497`) has **exactly seven** fields: `worldX`, `worldZ`, `yawRad`, `teamColorId`, `wheelSpinRad`, `steerAngleRad?`, `driftYawOffsetRad?`. **Not one** of them is slip, traction, grip, slip-ratio, scrub or a per-tire state. The presentation pass is **structurally incapable** of expressing traction loss, because it is never told about it. Any traction visual the implementation wave wants must first **extend this contract**.

**G-V2 - No traction-loss scalar is published; the scrub is computed and DISCARDED.** _(confidence 0.95, gap type: missing; CROSS-REFERENCE to `AUDIT-PHYSICS` F5 - not a new finding)_
`physics.step.service.ts:419-427` computes `computeLongitudinalSpeedScrub(state.slipAngle, nextSpeed, speedPercent) * dt` **inline** into `scrubbedSpeed`; **no scalar survives on the state**. Therefore the presentation layer has **nothing to read**, and S-4 ("one shared scalar drives both") is structurally impossible today. The implementation wave must **publish** the scalar (the merged brief's D_J already says so) - the presentation half of this audit confirms that **no presentation consumer exists to receive it**, and that the _only_ candidate readers today (the pose spring and the camera) read the _committed_ slip instead.

**G-V3 - The ONE drift visual is driven by the COMMITTED slip, not by the traction loss - so the car crabs with zero grip lost.** _(confidence 0.93, gap type: contradicts)_
The pose target is `0.5 * committedSlip + 0.08 * articulation` (`drift.model.ts:756-758`), fed the **committed** slip (`physics.step.service.ts:457`); the traction law reads the **published** slip (`physics.step.service.ts:422`). The two are different quantities. On the tick the commit latches, the committed slip is at its target while the published slip (and therefore the whole traction loss) is still ~zero, and the second-order pose spring (`omega = 18`, `zeta = 0.35`) reaches its full target in well under a second. The car is therefore **visibly at its full drift envelope while the tires have lost nothing** - a direct, code-level instance of the user's "poorly implemented camera trick, that has no connection with the drifting". The reverse also holds: the committed slip is a latched command that can saturate at the `0.45 rad` clamp regardless of the measured slip, so the visual can sit at a large offset with a small actual slide.

**G-V4 - The tire gradient is a fixed-space LIGHTING effect masquerading as a traction surface.** _(confidence 0.96, gap type: contradicts)_
`feedTireGradientUniforms(enabled: boolean)` (`car.pass.ts:1516-1536`) takes a boolean and four constants - **no drift, slip, speed or traction input** - and the fragment shader (`car.pass.ts:325-333`) computes a band from `dot(vWorld - wheelCenter, uSunDirection)`. The shader's own comment says the highlight _"stays fixed while the wheel voxels spin through it"_, so the visible tire activity is the tread sliding through a fixed light band: **proportional to SPEED, blind to GRIP**. A sliding tire and a free-rolling tire render identically. The AC-088 `MIN_SPREAD` residual is a **static spread floor**, not a traction magnitude.

**G-V5 - The tire yaw-follow is an INPUT ECHO, not a physics echo.** _(confidence 0.95, gap type: contradicts)_
`computeWheelSteerAngleRad(steerAxis) = clamp(steerAxis, -1, 1) * 35 deg` (`wheel.steer.ts:55-57`) is a **pure linear function of the raw steering axis**, applied to the front wheels only (`car.pass.ts:1661-1667`); the rear wheels get **no** steer yaw. It bypasses **every** drift gate: the `0.5` engage threshold, the `15 m/s` MIN_SPEED gate and the re-arm latch. At `steer = 0.49` the front wheels are at `17.2 deg` - a dramatic-looking turn - while the drift model has latched nothing and the traction loss is **exactly zero**. This is the strongest single explanation the code offers for "the tires still never lose traction": the _only_ thing the player can read on the tires is their own input, echoed back.

**G-V6 - There is no speed-loss HUD and no traction/grip readout.** _(confidence 0.96, gap type: partial)_
`host/hud.ts` has exactly three readouts (speed / lap / time) plus a static controls hint; `NeatRacingHudState` (`hud.ts:142-147`) carries only `{ speedUnitsPerS, raceState }`. The speed number **is** post-traction (`race-session.ts:261-264` -> `state.car.speed` = `physics.step.service.ts:471`), so a speed loss is _implicitly_ visible in the number - but there is **no** grip bar, slip indicator, traction lamp, delta-vs-target or colour change, and the HUD contract has **no field** that could carry one. Adding a traction readout is a **contract change**, not a wiring change.

**G-V7 - Every published-but-unwired drift output (the complete enumeration).** _(confidence 0.96, gap type: missing)_
`car.yawRate`, `car.drift.pose.poseRate`, `car.drift.pose.articulation`, `car.drift.committed.slip`, `car.drift.committed.turnSign`, `car.drift.committed.reArmed` have **ZERO production readers** (all readers are `.test.ts`); the **scrub is not published at all**. The only production consumer of the whole drift channel that reaches a visual is `car.drift.pose.pose` -> the crab. `car.slipAngle` and `car.lateralVelocity` are read in production only by the bounce recompose (`collision.utils.ts:386-389`) and the seam. `car.yawRate` and `car.drift.pose.poseRate` are the two channels that a _proportional_ camera (S-1) and a _traction-proportional_ visual would both want, and both are free to wire.

**G-V8 - The debug seam carries NO traction channel, and the planned 6th field must be the shared scalar.** _(confidence 0.94, gap type: partial)_
`window.neatRacingDriftState` is exactly five fields (`browser-entry.ts:978-984`): `poseRad`, `slipAngleRad`, `speedMps`, `tickHz`, `tick`. There is **no** field for traction loss, yaw rate, pose rate, committed slip or turn sign. Per `AUDIT-CAMERA` F8 the 6th field is accepted only if the follow state is ONE SHARED OBJECT, is written at render rate, updated at tick rate, and the new 44th check tolerates 60 Hz quantization above 60 Hz displays. **This audit's constraint:** the 6th field must be the **shared traction scalar** (D_B / D_J), not only the follow offset - otherwise the fixture keeps measuring the camera and never the tires, and S-4 is re-broken at the seam.

**G-V9 - No smoke / skid-mark / dust / trail subsystem exists, and no audio channel exists either.** _(confidence 0.95, gap type: missing)_
A repo-wide grep over `examples/neatRacing/**/*.ts` for `smoke|skid|trail|tireMark|particle` returns **zero product hits**; a grep for `audio|squeal|AudioContext` returns **zero hits**. The dispatch's "smoke" and the user's "skid" expectation have **no substrate at all**. This is the largest new-build item in the whole wave: a new render pass (or a new draw phase inside the car pass), its own uniform seam, its own draw ordering relative to the blob shadow / body / wheels, a life/spawn policy, and - critically for this repo - a **determinism** story (the physics is a pure 60 Hz tick; particles would have to be tick-stepped, not wall-clock stepped, or they will desync the deterministic replay contract).

**G-V10 - Per-tire grip differentiation is impossible today.** _(confidence 0.92, gap type: missing)_
The tire gradient is fed **per draw** with the **same four global uniform values** for all four wheels, and the seam carries **only a boolean** - no mount identity (`front-left` / `front-right` / rear), no per-tire grip. The shader's `wheelCenter` is derived from `uModel[3]`, so the per-wheel _geometry_ differs, but the per-wheel _state_ does not exist. The single most legible drift cue in an arcade racer - **the rear tires breaking away while the fronts stay hooked** - is therefore inexpressible. The hook point exists (`feedTireGradientUniforms` is called per wheel), so the work is to widen the seam and add a per-tire scalar, not to find a new location.

**G-V11 - The only traction-caused pixel is the wheel spin, and it is a speed effect that is backwards for a real slide.** _(confidence 0.90, gap type: partial)_
`computeVisualWheelSpinDeltaRad` derives `omega = v/r` from the **post-traction** speed and clamps it to 2 rev/s (`wheel.mount.utils.ts:419-437`, `NEATRACING_MAX_VISUAL_WHEEL_OMEGA_RAD_PER_S` at `:121`). The wheels do visibly slow when the scrub bites - but the effect is **indistinguishable from braking**, and a _slipping_ wheel physically spins **faster** than the road speed, so the model has no slip-ratio on the visual side and the rendered slide reads backwards.

---

#### 3. Decision

**D1 - Gap classification of the dispatch's clauses.**

| Clause                                        | Gap type        | Confidence | One-line verdict                                                                                                                                                                                                                                                        |
| --------------------------------------------- | --------------- | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "tire yaw-follow"                             | **contradicts** | 0.95       | The tire follow **plumbing** is complete and correct (AC-078 parents the wheels onto the drift-yawed matrix), but the front-wheel steer yaw is an **unconditioned input echo** and the body crab is driven by the **committed slip** - neither reads the traction loss. |
| "skid / gradient"                             | **contradicts** | 0.96       | There is **no** skid. The "gradient" is a fixed-sun **lighting** band with a boolean-only seam; its visible motion is the tread sliding through the light, i.e. speed, not grip.                                                                                        |
| "smoke"                                       | **missing**     | 0.95       | No smoke, skid mark, dust, trail or particle subsystem exists anywhere in the codebase.                                                                                                                                                                                 |
| "speed-loss HUD"                              | **partial**     | 0.96       | The speed **number** is post-traction, so a loss is implicitly visible; there is **no** traction/grip readout and the HUD contract has no field for one.                                                                                                                |
| "every published-but-unwired drift output"    | **missing**     | 0.96       | Six published outputs have **zero** production readers; the scrub is **not published at all**; the only production consumer of the drift channel is the crab.                                                                                                           |
| "which wires exist and which must be CREATED" | **answered**    | 0.94       | See the ledger in 2.H: **7 wires exist**, **7 must be created**, and 3 of the 7 existing ones are _structurally_ incapable of expressing traction.                                                                                                                      |

**D2 - The single sentence the implementation wave needs.** _The build has exactly one drift-driven visual, it is fed by the committed slip rather than by the traction loss, the two most tire-like effects (the front-wheel steer and the tire gradient) are echoes of the input and of the speed respectively, and there is no traction-loss scalar, no traction HUD, no smoke, and no per-tire state anywhere in the presentation layer._

**D3 - The presentation inventory, restated as a contract table (this is the deliverable).**

| Layer                       | What exists                            | What it is fed by                                    | Traction-driven?                  |
| --------------------------- | -------------------------------------- | ---------------------------------------------------- | --------------------------------- |
| Body crab                   | `driftYawOffsetRad`                    | the pose spring over the **committed** slip          | **NO**                            |
| Wheel chain                 | the body's drift-yawed matrix (AC-078) | inherits the crab                                    | **NO**                            |
| Front-wheel steer yaw       | `steerAngleRad`                        | the **raw steering axis** (35 deg lock)              | **NO**                            |
| Rear-wheel steer yaw        | nothing (rears are excluded)           | -                                                    | -                                 |
| Wheel spin                  | `wheelSpinRad`                         | **post-traction** speed / radius, clamped at 2 rev/s | indirectly, as a **speed** effect |
| Tire gradient               | `uApplyTireGradient` + 4 constants     | the **fixed sun direction**                          | **NO**                            |
| HUD speed                   | `speedUnitsPerS`                       | **post-traction** speed                              | indirectly, as a **speed** effect |
| HUD lap / time / hint       | race rules / a literal                 | -                                                    | -                                 |
| Seam                        | 5 fields                               | pose, published slip, speed, tick rate, tick         | **NO** (no traction field)        |
| Smoke / skid / dust / audio | **nothing**                            | -                                                    | -                                 |

**D4 - The order of creation is not free.** Three things must be created, and the order is forced by the dependency chain:

1. **Publish the traction-loss scalar on the physics state** (the physics half of G-V2; the merged brief's D_J). Until this exists, **nothing** else in the presentation layer can be wired to the traction law - every visual would have to re-derive the scrub in the view layer, which is the duplicated-physics-law hazard `AUDIT-PHYSICS` F5 identified.
2. **Widen the car-pass contract** (`CarPassCarState`) with a traction field, and (if the rears-breaking-away cue is wanted) with a **per-tire** field. Until the contract carries the value, the pass cannot consume it (G-V1, G-V10).
3. **Only then** wire the consumers: the HUD readout (G-V6), the seam's 6th field (G-V8), and any new effects pass (G-V9).

**D5 - The seam's 6th field must be the shared traction scalar.** Per `AUDIT-CAMERA` F8 the 6th field is accepted only if the follow state is ONE SHARED OBJECT, written at render rate and updated at tick rate, with the 44th check tolerating 60 Hz quantization above 60 Hz displays. This audit adds one constraint: **the shared object must contain the traction scalar, not only the follow offset.** A 6th field carrying only the follow offset would let the fixture report GREEN while the tires still show nothing - the `AUDIT-PHYSICS` R-4 failure mode, repeated at the seam. The traction scalar is already speed-proportional (`0.55 * |slip| * speed^2 / 89`), so the 44th check must also be a **ratio**-shaped assertion, not a floor.

**D6 - The visual effect of the two slip variables must be re-thought, not just re-wired.** G-V3 is not fixed by "wiring the scrub to the crab". The crab is a **second-order spring over a latched command**; the traction law reads the **measured** slip. If the implementation wave makes the crab read the published slip, the crab stops overshooting (the spring's far-side pop is the feature, per the `car.pass.ts` JSDoc), and if it keeps the committed slip, the visual still leads the physics. The author must **decide which of the two is the visual's source** and record it - this is a design decision, not an implementation detail.

**D7 - The tire gradient must either be honestly re-labelled or genuinely extended.** Two coherent options, and the author must pick one:

- **(a) HONEST** - rename/re-document the element as a **lighting** effect (OBS-20 fixed-space tire lighting), and build the traction surface elsewhere (a new per-tire gradient input, a new smoke/skid pass, a HUD element). This is the cheaper, more honest option, and it removes the risk that a future author keeps treating the gradient as the traction surface.
- **(b) EXTENDED** - widen `feedTireGradientUniforms` from `enabled: boolean` to a per-tire `(traction, mount)` feed and add a per-tire grip term to the shader. This makes the gradient a real traction surface, but it touches the fragment shader and the AC-088 MIN_SPREAD contract, whose JSDoc explicitly says the rendered gradient is _"NOT measurable from JavaScript"_.
  Option (b) re-opens AC-088; option (a) does not. The author must state which.

**D8 - The three un-wired channels are the cheapest traction visuals available, and they should be used first.** `car.yawRate` (the per-tick turning rate) and `car.drift.pose.poseRate` (the crab's rate) are already published, already per-tick, and already unwired. `car.drift.committed.turnSign` is the latched turn direction. Together they are enough to express "the car is turning" and "the car is sideways, and in which direction" **without touching the physics contract at all** - a strictly cheaper first tranche than the full traction-scalar wave, and a good RED-lane target.

**D9 - The numbered gaps in implementation priority order.**

| #         | Gap                                                                                   | Type        | Confidence | Priority                                                            |
| --------- | ------------------------------------------------------------------------------------- | ----------- | ---------- | ------------------------------------------------------------------- |
| **G-V2**  | no published traction-loss scalar (the scrub is computed and discarded)               | missing     | 0.95       | **P0** - blocks everything (D4)                                     |
| **G-V1**  | the car-pass contract has no traction channel                                         | missing     | 0.97       | **P0** - blocks everything (D4)                                     |
| **G-V3**  | the one drift visual reads the committed slip, not the traction loss                  | contradicts | 0.93       | **P0** - this is the user's complaint                               |
| **G-V5**  | the tire yaw-follow is an input echo with no gate                                     | contradicts | 0.95       | **P1** - cheap to fix, high visual impact                           |
| **G-V4**  | the tire gradient is lighting, not traction                                           | contradicts | 0.96       | **P1** - decide (a) or (b) (D7)                                     |
| **G-V7**  | six published outputs with zero production readers (+ the scrub)                      | missing     | 0.96       | **P1** - D8 is the cheapest tranche                                 |
| **G-V6**  | no speed-loss / traction HUD                                                          | partial     | 0.96       | **P2** - a contract change to `NeatRacingHudState`                  |
| **G-V8**  | the 5-field seam carries no traction channel; the 6th field must be the shared scalar | partial     | 0.94       | **P2** - constrained by `AUDIT-CAMERA` F8                           |
| **G-V10** | per-tire grip differentiation is impossible                                           | missing     | 0.92       | **P2** - depends on G-V4 (b)                                        |
| **G-V11** | the only traction pixel is a clamped, backwards speed effect                          | partial     | 0.90       | **P3**                                                              |
| **G-V9**  | no smoke / skid / dust / trail / audio subsystem exists                               | missing     | 0.95       | **P3** - the largest new-build, and it needs a determinism decision |

**D10 - What this audit does NOT adjudicate.** The camera aim path, the `cameraPose` trap (`frame-composer.ts:822` feeds `cameraPose` to the **starfield** only - so it is NOT a general camera channel), the wheel-spin reset idiom, the `stepPhysics` wrapper and the `tau = 0.08` follow law remain `AUDIT-CAMERA`'s. The traction law and the `0.45 / 0.5 / 3.5 / 0.55` constants remain `AUDIT-PHYSICS`'s settled facts and were **not** re-proved. The live keyboard chain remains `14N-V5-audit-input`'s.

---

#### 4. Risks

- **R1 (highest) - the "presentation" surface may look healthy because the input echo looks like traction.** The front-wheel steer yaw is a _linear_ function of the steering axis with no gate, so on screen the tires _always_ respond to the player. Any visual review that asks "do the tires move when I steer?" will answer YES and conclude the chain is wired, while the traction law is doing nothing. **Mitigation: every new traction-visual contract must assert the response to a quantity the player does NOT control** (the published slip or the traction scalar) - e.g. a _same-input, different-traction_ comparison, never a _different-input_ comparison.
- **R2 - the drift envelope is pinned by a pose contract, so a traction visual may be built against the wrong variable.** The `car.pass.ts` JSDoc (L460-495) pins the _target_ envelope at `0.3050 rad` and the _fallback_ re-record target at `0.2250 rad`, and both are derived from the **committed** slip. If the wave makes the crab read the **published** slip instead, the envelope is derived from a different variable and **every one of those pinned numbers must be re-derived** - including the 44th fixture check. This is a coupled blast radius of the same kind as `AUDIT-PHYSICS` R-2.
- **R3 - the `0.45 rad` commit clamp and the `0.5` engage threshold mean the traction visual can be _saturated_ in ordinary driving.** Once the commit clamps, a further steering increase changes nothing in either the pose or the scrub, so a traction visual driven by the committed slip will **plateau** in normal cornering. A visual that saturates in ordinary driving reads as "broken" to the player even when the law is correct.
- **R4 - the smoke / skid work has a determinism trap.** The physics is a pure 60 Hz tick with a documented replay contract (`physics.step.service.ts:27-28`, "the service is PURE"). A wall-clock-stepped particle system would break the deterministic replay and would need its own tick-stepped state. **Mitigation: decide the particle state model (tick-stepped, seeded from the deterministic tick counter) BEFORE the first particle is drawn** - the same discipline the plan already applies to the `pulse` system (`collectActivePulses(car.tick * dt, ...)`).
- **R5 - a 6th seam field that carries only the follow offset would make the fixture report GREEN while the tires show nothing.** This is `AUDIT-PHYSICS` R-4 (the vacuous 340 km/h clause) repeated at the seam. **Mitigation: D5 - the shared object must contain the traction scalar, and the 44th check must be ratio-shaped.**
- **R6 - the 60 Hz quantization constraint on the 6th field cuts against a fast-changing traction scalar.** The seam is written once per frame but the scalar updates once per tick, so at display rates above 60 Hz the field is stale between ticks, and the traction scalar is **speed-proportional** (and therefore moves fast at speed). A 44th check that asserts a _rate_ on the seam must tolerate the quantization, or it will fail intermittently.
- **R7 - no runtime measurement was performed.** Every claim here is a static-code read. In particular, the claim that "the car crabs with zero grip lost" is an **argument from the code** (two different slip variables + a spring time constant), not a measurement. Before the contract is re-authored, a code-level lane should confirm: (i) the pose magnitude at the first tick after the commit, versus the published slip and the scrub at the same tick; (ii) the front-wheel steer angle at `steer = 0.49` versus the committed slip and the scrub; (iii) the wheel-spin delta caused by the scrub, isolated from braking.
- **R8 - the "no child agents" constraint.** The dispatch explicitly forbade child agents inside each wave (`O_1`), overriding the generic Tier-1 delegation mandate. This audit is therefore **single-context** and its findings are **not** cross-checked by an independent scout. Every headline claim is a direct code read with a cited line, and the three highest-stakes claims - (a) the zero production readers of the six unwired outputs, (b) the absence of any smoke/skid/particle/audio subsystem, (c) the boolean-only tire-gradient seam - were each re-verified by an **independent repo-wide grep** in this session. Residual risk: a _missed_ gap that a second pair of eyes would have found. **Mitigation: the 01-planning author re-runs the three greps before the contract is frozen.**
- **R9 - the audit boundary.** This artifact adjudicates the **presentation** surface only. The physics-side facts it relies on are `AUDIT-PHYSICS`' settled rows, not this audit's findings; if that audit's conclusions are revised by the orchestrator's merge, G-V2's wording (and only its wording) changes. The camera surface remains `AUDIT-CAMERA`'s.

---

#### 5. Evidence index (files read, read-only)

| File                                                  | Read                                                                                                                                                                                                                                                                               | Purpose                                                                                             |
| ----------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `plans/neatracing-voxel-racer.plans.md`               | L10000-10291 (the AUDIT-CAMERA findings, the MERGED AUTHOR BRIEF D_A..D_N, USER VERDICT 4, USER VERDICT 5), L10134-10221 (the AUDIT-PHYSICS PlanUpdate, the format template)                                                                                                       | the dispatch, the settled facts, the PlanUpdate shape                                               |
| `plans/neatracing-voxel-racer.research.md (Part B-1)` | L1-40, L243-400                                                                                                                                                                                                                                                                    | the precedent artifact; the settled physics facts (F1-F8, D1-D6, R1-R8)                             |
| `examples/neatRacing/engine/passes/car.pass.ts`       | L225-340 (the lit shader + the tire-gradient block), L370-425 (the gradient constants + the MIN_SPREAD JSDoc), L428-497 (`CarPassCarState` + the drift-yaw contract), L1500-1560 (`feedTireGradientUniforms`), L1600-1680 (the body/wheel draw order + the AC-078 wheel parenting) | the tire + body presentation pass                                                                   |
| `examples/neatRacing/host/hud.ts`                     | L136-160 (`NeatRacingHudState` + the three labels), L300-410 (the DOM construction + the update path)                                                                                                                                                                              | the whole HUD                                                                                       |
| `examples/neatRacing/host/race-session.ts`            | L200-270 (the input sampling, the tick replay, the HUD update)                                                                                                                                                                                                                     | the HUD's input, and the input-object sharing                                                       |
| `examples/neatRacing/host/frame-composer.ts`          | L140-190, L780-830                                                                                                                                                                                                                                                                 | the `cameraPose` starfield-only channel (the trap)                                                  |
| `examples/neatRacing/browser-entry/browser-entry.ts`  | L112-152 (the 5-field seam interface), L880-1010 (`carPoseBag`, the pose feed, the seam write, the camera forward, the wheel-spin accumulation)                                                                                                                                    | the presentation seam + the only production drift consumer                                          |
| `examples/neatRacing/car/wheel.steer.ts`              | **complete, 58 lines**                                                                                                                                                                                                                                                             | the front-wheel steer echo                                                                          |
| `examples/neatRacing/car/generate-car.ts`             | L20-75                                                                                                                                                                                                                                                                             | the render-pose bag and the steer feed                                                              |
| `examples/neatRacing/voxel/wheel.mount.utils.ts`      | L80-130 (the hub height, the 2 rev/s cap), L370-440 (the spin clamp + `computeVisualWheelSpinDeltaRad`)                                                                                                                                                                            | the wheel-spin visual                                                                               |
| `examples/neatRacing/physics/physics.step.service.ts` | L78-130 (`PhysicsCarState`), L370-482 (the tick pipeline: the yaw, the commit, the chase, the scrub, the pose spring, the published state)                                                                                                                                         | the published contract + the discarded scrub                                                        |
| `examples/neatRacing/physics/drift.model.ts`          | L720-770 (`stepDriftPoseSpring`)                                                                                                                                                                                                                                                   | the pose target and the two slip variables                                                          |
| `examples/neatRacing/physics/collision.utils.ts`      | L30-40, L220-235, L300-320, L380-500                                                                                                                                                                                                                                               | the only production readers of `slipAngle` / `lateralVelocity` + the zeroed drift channel on bounce |
| `examples/neatRacing/physics/ai.autopilot.ts`         | the input sampler (grep)                                                                                                                                                                                                                                                           | the autopilot reads only `x, y, heading, speed` - it never reads the drift channel                  |
| `examples/neatRacing/engine/lighting.test.ts`         | grep (`uTireGradient*`)                                                                                                                                                                                                                                                            | the uniform-name contract + the test-only coverage of the gradient seam                             |
| repo-wide greps                                       | `examples/neatRacing/**/*.ts` for `smoke\|skid\|trail\|tireMark\|particle`; `audio\|squeal\|AudioContext`; `\.committed\.(slip\|turnSign\|reArmed)\|pose\.(poseRate\|articulation)\|\.yawRate`; `uTireGradient\|TireGradient`; `computeWheelSteerAngleRad`; `fillCarRenderPose`    | the zero-reader, zero-subsystem and zero-hits findings                                              |

**No file was modified by this audit.** The only writes are this artifact and the `AUDIT-VISUALS` PlanUpdate appended to `plans/neatracing-voxel-racer.plans.md`.

### B-5 — AUDIT-MOTION — where the committed slip / speed scrub IS (and is NOT) applied to the car's ACTUAL MOTION

> **Provenance.** Artifact id `neatracing-voxel-racer.audit-motion`; author `02-researching` (Tier 1),
> 2026-10-27; dispatch origin the `USER VERDICT 5` record in the tracker (`audit_waves[0]` =
> `14N-V5-audit-motion`); subject `examples/neatRacing/physics/physics.step.service.ts` (the live 60 Hz tick
> pipeline) + `drift.model.ts` + `browser-entry/browser-entry.ts` + `engine/passes/car.pass.ts` +
> `host/race-session.ts` + `physics/collision.utils.ts`; constraints honoured — READ-ONLY, NO child agents,
> NO browser runs, NO Chrome MCP, no production byte touched.
> **Fold note (2026-10-31).** Folded here as Part B-5; the sibling file is DELETED.

**Artifact id**: `neatracing-voxel-racer.audit-motion`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 5` record, `plans/neatracing-voxel-racer.plans.md` (the `audit_waves` block), `audit_waves[0]` = `14N-V5-audit-motion`
**Subject**: `examples/neatRacing/physics/physics.step.service.ts` (the live 60 Hz tick pipeline) + `examples/neatRacing/physics/drift.model.ts` (the drift law) + `examples/neatRacing/browser-entry/browser-entry.ts` (the render bridge) + `examples/neatRacing/engine/passes/car.pass.ts` (the body draw) + `examples/neatRacing/host/race-session.ts` (the live loop) + `examples/neatRacing/physics/collision.utils.ts` (the world-velocity algebra)
**Constraints honoured**: READ-ONLY code audit · NO child agents (the dispatch's `O_1`) · NO browser runs · NO Chrome MCP · no production byte touched.

---

#### 0. Provenance, method and the naming deviation

**Method.** Every claim below is a _static-code_ read, plus a **read-only in-memory execution of the exact audited sources** (no browser, no Chrome MCP, no repo mutation). The numeric claims marked **MEASURED** were produced by importing the repository's own `physics.step.service.ts` through `esbuild` into a temp file **outside** the repository (`$TEMP`), driving the real `stepPhysics` at the real 60 Hz fixed timestep, and reading the real returned state. The repository working tree was not modified and no build artifact was written inside it.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. This audit is the only wave of the four that produced **runtime** evidence (headless, in-memory, no browser), so its numeric findings outrank the static-code rows of the sibling waves and are ranked `runtime` above.

**Naming deviation (disclosed).** The contract names `plans/<PlanName>.research.md`. That exact path is **already occupied** by a different, closed 14N artifact (the 2026-10-24 drift-feel _integration coupling_ design), and `plans/neatracing-voxel-racer.research.md (Part B-6)` and `plans/neatracing-voxel-racer.research.md (Part B-1)` are likewise occupied, as is the sibling `plans/neatracing-voxel-racer.research.md (Part B-4)` written by wave [3] on the same day. Clobbering or appending to a closed, dated artifact would fuse two different questions under one date. This artifact therefore follows the repo's own topic-suffix family convention as a **new sibling**: `plans/neatracing-voxel-racer.research.md (Part B-5)`. It remains a `plans/*.research.md` sibling of the tracker and is **not** under `docs/research/`. The orchestrator may rename or merge it without loss.

**Scope boundary.** This is the **MOTION** half: the physics tick -> the car entity's velocity / heading / travel direction, and the exact line where the drift stops changing how the car moves. The _presentation inventory_ is `14N-V5-audit-visuals`' scope; the _keyboard chain_ is `14N-V5-audit-input`'s scope; the _numeric engage envelope_ is `14N-V5-audit-envelope`'s scope. Cross-references are flagged inline and are not re-proved.

**Non-duplication.** The settled `AUDIT-PHYSICS` facts (the committed-slip clamp 0.45, the 0.5 engage threshold, the 3.5 rad/s ramp, the `0.55*|slip|*speed^2/89` scrub formula, the camera rigid-mirror F1, the `0.5*slip + 0.08*articulation` pose consumption) are **not** re-proved. This audit answers the question the four waves were spawned for: **where does the committed slip and the speed scrub actually reach the car's motion, and where does it stop.**

---

#### 1. Question

The `USER VERDICT 5` dispatch (`audit_waves[0]`) asks:

> TRACE where committed slip / speed scrub is (or is NOT) applied to the car's ACTUAL MOTION in the live runtime path: `physics.step.service.ts` integration -> the car entity's velocity/heading/travel direction. MARK THE EXACT BREAK LINE where drift stops changing how the car moves, and name the missing wiring (lateral velocity divergence, travel-vs-heading separation, scrub consumption).

The user's own verdict supplies the reason the question matters:

> "even worse! there is NO drifting! the tires still never loose traction! which is the main fucking point. It's just a poorly implemented camera trick, that has no connection with the drifting, there are no drifiting physics in place, I can take turns at full speed and the car never drifts, it's only confusing due to the odd camera angle. The current implementation is either not wired or a complete failure!"

So the audit must answer three things:

1. **Does the committed slip change where the car goes?** (the travel-vs-heading separation)
2. **Does the speed scrub change how fast the car goes?** (the scrub consumption)
3. **If both are real, why does the player see no drift?** - i.e. where is the exact line where the drift stops changing how the car moves _as seen_.

---

#### 2. Evidence

##### 2.A - The live tick pipeline, end to end _(confidence 0.97, source: static-code + runtime)_

`examples/neatRacing/physics/physics.step.service.ts:366` `stepPhysics` is the single motion producer of the whole runtime. Its five stages, with the drift channel inserted at 4a-4e:

| Stage                     | Lines      | What it does                                                                                                        | Does the slip change the motion here?           |
| ------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| 1 surface resolve         | `:373-375` | reads the surface under the car at tick start                                                                       | no                                              |
| 2 yaw / heading           | `:377-381` | `nextHeading = wrapHeading(state.heading + yawRatePerTick)`                                                         | no (the nose only)                              |
| 3 longitudinal integrate  | `:384-390` | `nextSpeed` (accel / brake / drag / off-road)                                                                       | no                                              |
| **4a commitment**         | `:396-405` | `stepDriftCommitment(state.drift.committed, ...)` -> `nextCommitted.slip` (the latched command)                     | **sets up** the slide                           |
| **4b committed chase**    | `:407-416` | `computeNextLateralVelocity(nextCommitted, ...)` -> `nextLateralVelocity`                                           | **yes** (the divergence)                        |
| **4c speed scrub**        | `:419-427` | `computeLongitudinalSpeedScrub(state.slipAngle, nextSpeed, speedPercent)` -> `scrubbedSpeed`                        | **yes** (the traction loss)                     |
| **4c-bis off-road floor** | `:436-444` | lifts the post-scrub speed to the off-road floor                                                                    | indirectly                                      |
| **4d published slip**     | `:447-450` | `nextSlipAngle = computeSlipAngle(nextLateralVelocity, nextSpeedAfterTraction)`                                     | -                                               |
| **4e pose spring**        | `:454-459` | `stepDriftPoseSpring(...)` -> the cosmetic yaw accumulator                                                          | **NO - this is the visual, not the motion**     |
| **5 translate**           | `:462-475` | `velocityHeading = nextHeading + nextSlipAngle`; `x/y += (cos, sin)(velocityHeading) * nextSpeedAfterTraction * dt` | **yes - THIS is where the drift moves the car** |
| publish                   | `:477-484` | `heading`, `speed`, `slipAngle`, `lateralVelocity`, `drift`                                                         | -                                               |

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

##### 2.B - THE EXACT BREAK LINE: the render bridge is built from the NOSE, and the travel direction is a discarded local _(confidence 0.97, source: static-code + runtime)_

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

| Line                         | Expression                                                                   | What it binds the frame to                                    |
| ---------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `browser-entry.ts:935-936`   | `forwardX = Math.cos(car.heading)`, `sceneForwardZ = -Math.sin(car.heading)` | the **nose**                                                  |
| `browser-entry.ts:938-940`   | `cameraWorld = car.x/y - 5.5 m * noseDir`                                    | the **nose** (re-derived from the car's position every frame) |
| `browser-entry.ts:963`       | `fillCarRenderPose(..., carSceneYawRad(car.heading), ...)`                   | the **nose**                                                  |
| `browser-entry.ts:972`       | `carPoseBag.driftYawOffsetRad = car.drift.pose.pose`                         | the **cosmetic pose spring**                                  |
| `browser-entry.ts:1000`      | `cameraPose.yawRadians = carPoseBag.yawRad`                                  | the **nose**                                                  |
| `browser-entry.ts:1010-1012` | `target = car.x/y + 8 m * noseDir`                                           | the **nose**                                                  |
| `car.pass.ts:1608`           | `rotationYMat4(NEATRACING_CAR_YAW_MATRIX, car.yawRad)`                       | the **nose**                                                  |
| `car.pass.ts:1614-1620`      | body yaw = base yaw + `driftYawOffsetRad`                                    | the **cosmetic pose**                                         |
| `car.pass.ts:1415-1425`      | shadow yaw = `car.yawRad + driftYawOffsetRad`                                | the **cosmetic pose**                                         |

**Why this erases the drift (the mechanism, MEASURED):**

1. The camera is re-placed **every frame from the car's current position** (`browser-entry.ts:938-940` uses `car.x` / `carSceneZ`), along the nose. A rigid nose-frame therefore **follows the car's off-axis travel perfectly**: the 25.8-degree off-axis motion produces **no screen-space translation at all**. The one cue a viewer would normally read as "sliding" is cancelled by construction.
2. The camera's **aim** is the nose (`browser-entry.ts:1000`, `:1010-1012`).
3. The **body's** rendered yaw is the nose **plus the cosmetic pose** (`browser-entry.ts:972` -> `car.pass.ts:1614-1620`), and the pose is a second-order spring on `0.5 * COMMITTED slip + 0.08 * articulation` (`drift.model.ts:730-770`). MEASURED settled value: **8.73 deg**, i.e. **33.8 %** of the physical 25.79 deg separation. The articulation term (`+0.08 * 0.9 = +0.072 rad`) partially **cancels** the slip term (`-0.5 * 0.45 = -0.225 rad`).
4. The published `car.slipAngle` - the number that actually describes the slide - is carried across the bridge and **dropped into a debug object** at `browser-entry.ts:982`.

**MEASURED net:** the angle between the car's **rendered heading** and its **actual travel direction** is `25.79 - 8.73 = 17.06 deg`, and **not one degree of it is expressed by the camera or by the body yaw**. The entire physical drift collapses into an 8.73-degree cosmetic crab in a frame that is rigidly attached to the nose.

**This is the user's sentence, mechanically:** _"It's just a poorly implemented camera trick, that has no connection with the drifting."_ The camera trick is real, and the connection to the drifting is severed at exactly two lines: **`browser-entry.ts:963`** (the body yaw is bound to `car.heading`) and **`browser-entry.ts:972`** (the only body-yaw offset is the pose spring, not the slip). The break line proper - where the physical quantity stops existing - is **`physics.step.service.ts:462`**, where `velocityHeading` is computed and dropped.

**A non-finding worth recording so the implementation wave does not "fix" a non-bug:** the pose **sign** is coherent with the travel direction (a left turn gives a negative committed slip, a negative pose, and a negative travel separation - `drift.model.ts:735-737` explicitly documents the `+committedSlip` sign repair). There is **no sign inversion** in the render path. The defect is **magnitude and provenance** (a 0.5-gain spring on the committed command instead of the published travel direction), not sign.

##### 2.C - The lateral velocity divergence is real, but its only world-velocity consumer is DEAD CODE in the live runtime _(confidence 0.95, source: static-code)_

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

##### 2.D - The speed scrub IS consumed (this is NOT a discard) - but it is consumed as a bare scalar with no traction event _(confidence 0.96, source: static-code + runtime)_

The scrub is computed at `physics.step.service.ts:419-427`, is floored at `:436-444`, **is published as the tick-end speed** at `:478` (`speed: nextSpeedAfterTraction`), **and is used to translate the car** at `:469` / `:474`. MEASURED: the car really does lose 9.2 % of its top speed in a sustained drift, and the HUD shows the derated number (`host/race-session.ts:261-264` -> `state.car.speed`).

**Correction to the sibling wave's wording** (`14N-V5-audit-visuals` G-V2, which calls the scrub "computed and discarded"): the scrub is **not** discarded - it is applied to the speed and to the translation. What does not survive is the **scalar**: no traction-loss value is placed on the state, so no consumer can read _how much_ traction was lost, only the resulting slower speed. This audit records the correction because the motion question ("does the scrub change how the car moves?") turns on it: **yes, it does - by 9.2 % - and that is the ONLY place the traction law is observable in the live runtime.**

##### 2.E - THREE different slip values coexist in one tick _(confidence 0.95, source: static-code)_

| Value                                      | Where it is produced                                                                     | Where it is consumed                                                                                            |
| ------------------------------------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| the **committed** slip (a latched COMMAND) | `drift.model.ts:484-560` -> `physics.step.service.ts:396`                                | the chase target `tan(slip) * speed` (`drift.model.ts:690`) and the pose target (`physics.step.service.ts:457`) |
| the **published** slip (a MEASUREMENT)     | `physics.step.service.ts:447-450` (`atan2(nextLateralVelocity, nextSpeedAfterTraction)`) | the scrub (`:423`), the next tick (`:402`, `:413`), the debug seam (`browser-entry.ts:982`)                     |
| the **scrub input**                        | the tick-start published slip, `state.slipAngle` (`:423`)                                | the traction law                                                                                                |

Consequences: (a) the traction loss is **one tick (16.7 ms) behind** the slide that causes it; (b) the published slip is computed from the **post-scrub** speed, so the scrub lowers the denominator, which inflates `|atan2|`, which increases the scrub - a **positive feedback loop** (MEASURED: published 0.4502 vs committed 0.4500 at the steady state, a small but real self-amplification); (c) the pose reads the committed command while the traction law reads the published measurement, so the **visual and the traction law can disagree** (this is the sibling wave's G-V3, re-confirmed here from the motion side, and it is why the body can crab at full envelope with the tires losing little).

##### 2.F - The slide magnitude is set by the STEER AXIS, not by the speed _(confidence 0.94, source: static-code + runtime)_

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

##### 2.G - The live default mode is `ai`, and the autopilot's proportional steer barely reaches the engage gate _(confidence 0.90, source: static-code + runtime; CROSS-WAVE to `14N-V5-audit-input` / `14N-V5-audit-envelope`)_

- `host/mode-selector.ts:23` - the served page must open as a running race, i.e. in the **AI autopilot** mode.
- `host/race-session.ts:235-240` - in `ai` mode the input comes from `deps.sampleAutoInput(state.car)`, not from the keyboard.
- `physics/ai.autopilot.ts:153-181` - `steer = clampSteer(headingError * 2.2)` (`NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD = 2.2`, `:32`). The engage gate needs `|steer| >= 0.5`, i.e. **|headingError| >= 0.227 rad (13 degrees)**.

Steady-state analysis: to hold a radius `R` the autopilot needs a yaw rate of `v/R`, and the yaw rate law (`physics.step.service.ts:192` `STEERING_RATE = 2`, scaled by `speedPercent = v/89`) gives `yawRate = steer * 2 * v/89`, so `steer = 44.5 / R` **independent of speed**. The drift therefore engages only in corners **tighter than R = 89 m** (plus whatever the off-line error contributes), and otherwise the slip is **exactly 0.0000 rad**.

The **motion** consequence (this audit's lane; the chain itself is the input wave's): on the live page the car is in one of exactly two states - a 25.8-degree slide, or **literally zero** traction loss. A player who "takes a turn at full speed" and sees nothing is in the second state, and no intermediate behaviour exists to hint that the feature is present.

##### 2.H - The MIN_SPEED gate is a hard cliff, and the off-road floor gate is one tick stale _(confidence 0.93, source: static-code)_

- `drift.model.ts:184` `NEATRACING_DRIFT_MIN_SPEED_MPS = 15`; applied at `:495` (the commitment - below it the commit is released and `reArmed` is cleared) and at `:679` (the chase - below it the lateral velocity is **exactly 0**). There is no fade-in: the slide jumps from nothing to the full 3.5 rad/s entry ramp at 15 m/s.
- `physics.step.service.ts:524-546` `isOffRoadSpeedFloorEngaged(surface, input, state.speed)` reads the **tick-start** speed; the same gate then lifts the **post-scrub** speed at `:436-444`. A car that scrubs below the 22.25 m/s floor on the road and then rolls onto dirt gets the floor decided by the previous tick's speed - a one-tick lag in the traction regime.

##### 2.I - NON-FINDING: the served bundle is current - the break is NOT a staleness break _(confidence 0.99, source: static-code + runtime)_

`examples/neatRacing/index.html:97-100` selects the bundle by path:

```js
var bundlePath = repoPath
  ? '../../docs/assets/neatracing.bundle.js?v=20260928-1'
  : '../../assets/neatracing.bundle.js?v=20260928-1';
```

Verified: **both** files exist, are **byte-identical** (SHA-256 `3C255A7F91A99AA8B3C7172B2BADCD23426DC63746A286E81778A16E33FFD9D8`, 398,699 bytes, 10/27/2026 09:56:21), and the bundle's `.map` `sourcesContent` for `drift.model.ts` (30,224 B), `physics.step.service.ts` (25,851 B) and `browser-entry/browser-entry.ts` (49,377 B) are **byte-identical to the on-disk sources**. No source file is newer than the bundle (the only newer artifact is `editor/editor.bundle.js`, same timestamp).

**Conclusion: the live page executes the audited code. The break is a design break, not a build break.**

##### 2.J - NON-FINDING: the physics fixture reproduces the live behaviour, so no existing test can see the break _(confidence 0.96, source: static-code + runtime)_

`physics/physics.step.service.test.ts:626-631` asserts `Math.abs(drifted.slipAngle) > 0.4` from `runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road')`. The live-equivalent run reproduces `0.4512 rad`. The fixture and the live path agree on the physics, so **every drift test in the repo passes while the feature is invisible to the player** - the break lives entirely downstream of the physics state, where there is no assertion at all.

---

#### 3. Decision

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

| Named wiring                     | Verdict                                                               | Evidence                                                                                                                                                                                                          |
| -------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **lateral velocity divergence**  | **PRESENT in the physics, ABSENT from the runtime**                   | produced at `physics.step.service.ts:407-416` (MEASURED 42.4 m/s); its only world-velocity consumer `collision.utils.ts:384-390` is **dead code** (`resolveCarCollisions` is imported by no production module)    |
| **travel-vs-heading separation** | **PRESENT in the physics, ABSENT from the state and from the render** | `physics.step.service.ts:462` (MEASURED 24.06 deg realized); the state has no travel heading (`:78-130`, `:477-484`); the render reads the nose (`browser-entry.ts:935-940`, `:963`, `:1000`, `:1010-1012`)       |
| **scrub consumption**            | **PRESENT and REAL**                                                  | `physics.step.service.ts:419-427` -> `:436-444` -> `:478` -> `:469`/`:474` (MEASURED 89.000 -> 80.830 m/s = 9.2 %); but no traction scalar survives, and the HUD is its only observer (`race-session.ts:261-264`) |

**What the implementation wave must CREATE (not merely connect):**

1. **A travel direction on the state.** The state must carry the direction the car is going (or the velocity vector), because no consumer can re-derive it correctly and cheaply, and because the camera and the body both need it. This is a **contract change** to `PhysicsCarState`.
2. **A camera that can express the divergence.** The camera is currently a rigid nose-frame re-derived from the car's position every frame - it cancels the off-axis motion by construction. A camera that follows the **travel** direction (or blends nose and travel) is required before any drift is visible.
3. **A body yaw that expresses the published slip**, not the committed command through a 0.5-gain spring with a partially cancelling articulation term. The `0.5` gain and the `+0.08 * articulation` term are the _measured_ reason the body shows 8.73 deg where the slide is 25.79 deg.
4. **One shared slip/traction scalar** for the seam and for any traction visual (cross-referenced from the visuals wave's G-V8 and this audit's 2.D) - the scrub's _magnitude_ is currently unobservable.

**What must NOT be "fixed":** the sign of the pose (2.B - coherent), the scrub's application (2.D - real and applied), the committed-slip clamp, the 0.5 threshold, the 3.5 rad/s ramp, and the `0.55*|slip|*speed^2/89` scrub formula - all settled.

---

##### 3.A - The twelve numbered gaps, with the exact break line marked

Each gap is classified against the Spec-Kit gap types (missing / partial / contradicts / non-finding) and carries its own confidence and provenance. `**BREAK LINE**` marks the exact seam where the committed slip stops changing how the car moves.

| #         | Gap                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Type        | Conf. | Break site / file:line                                                                                                                                                                                                            |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **G-M1**  | **THE BREAK LINE.** `velocityHeading` - the last expression in the codebase that knows the direction the car is actually going - is a local `const` that is used to advance x/y and then **DISCARDED**: never stored on the state, never published, never rendered.                                                                                                                                                                                                                                                                                                                                          | missing     | 0.97  | **`examples/neatRacing/physics/physics.step.service.ts:462`** (used at `:469`, `:474`; discarded at `:475`)                                                                                                                       |
| **G-M2**  | The **state cannot express the travel direction**. `PhysicsCarState` carries the NOSE heading, `slipAngle` and `lateralVelocity` as scalars, and **no velocity vector and no travel heading**; no production code re-derives them. Fixing the camera therefore requires a **contract change**, not a wiring change.                                                                                                                                                                                                                                                                                          | missing     | 0.97  | `physics.step.service.ts:78-130` (contract), `:296-314` (init), `:477-484` (publish)                                                                                                                                              |
| **G-M3**  | The **render bridge is bound to the nose and the camera cancels the drift by construction**: the camera is re-placed every frame from the car's position along the nose, its aim is the nose, and the body yaw is the nose + a cosmetic pose spring. The measured 25.85 deg off-axis travel produces **no screen-space translation** - the user's "poorly implemented camera trick" in mechanical form.                                                                                                                                                                                                      | missing     | 0.96  | `browser-entry/browser-entry.ts:935-936`, `:938-940`, `:963`, `:972`, `:1000`, `:1010-1012` -> `engine/passes/car.pass.ts:1608`, `:1614-1620`, `:1415-1425`                                                                       |
| **G-M4**  | The **published slip has exactly ONE production reader and it is a debug object** (`window.neatRacingDriftState`). The number that actually describes the slide is carried across the render bridge and dropped into a debug field.                                                                                                                                                                                                                                                                                                                                                                          | missing     | 0.96  | `browser-entry/browser-entry.ts:982`; producers `physics.step.service.ts:480-481`; the only other non-test readers are the next tick's own `:402`, `:412-413`, `:423` and the dead `collision.utils.ts:386-389`                   |
| **G-M5**  | The lateral divergence's **only world-velocity consumer is DEAD CODE** in the live runtime: `resolveCarCollisions` is imported by **no production module** (only `createTrackSurfaceResolver` is wired), so the live car never bounces, never has its drift channel zeroed, and drives through walls. The divergence is measured (42.44 m/s peak, 39.06 m/s steady) but reaches no world velocity.                                                                                                                                                                                                           | missing     | 0.95  | `physics/collision.utils.ts:197` (`resolveCarCollisions`), `:384-390` (`resolveWorldVelocity`), `:406-414`, `:428-458`; wired at `browser-entry/browser-entry.ts:83`, `:733`                                                      |
| **G-M6**  | **CORRECTION to the sibling AUDIT-VISUALS G-V2.** The **speed scrub IS consumed, not discarded**: computed, floored, published as `state.speed`, and translated with. Measured 89.000 -> 80.830 m/s = **9.2 %** of the top speed. What does not survive is the **scalar** - no traction value reaches the state - so a consumer can read only the resulting slower speed. The HUD is its only live observer.                                                                                                                                                                                                 | partial     | 0.96  | `physics.step.service.ts:419-427` -> `:436-444` -> `:478` -> `:469`/`:474`; observer `host/race-session.ts:261-264`                                                                                                               |
| **G-M7**  | **Three slip values coexist in one tick**: the committed COMMAND, the published MEASUREMENT, and the scrub INPUT (the tick-start published slip). The traction law is one tick (16.7 ms) behind the slide that causes it, and the published slip is computed from the **post-scrub** speed, so the scrub lowers the denominator, inflates `\|atan2\|` and increases the scrub (measured published 0.4502 vs committed 0.4500 - a small positive feedback).                                                                                                                                                   | contradicts | 0.95  | `physics/drift.model.ts:484-560` (command) -> `physics.step.service.ts:396`; `physics.step.service.ts:447` (published); `physics.step.service.ts:423` (scrub input); chase target `drift.model.ts:690`; pose `drift.model.ts:457` |
| **G-M8**  | The **slide magnitude is set by the steer axis, not by the speed**: `computeGripBudgetFactor` ignores `speedPercent` (the arg is eslint-disabled-unused) and the entry rate 3.5 rad/s is speed-independent. Measured sweep from 89 m/s: steer 0.30/0.45/0.49 -> **0.0000 rad**; steer 0.50/0.55/0.70/1.00 -> **0.4512 rad, identical**. The tires never degrade with speed.                                                                                                                                                                                                                                  | contradicts | 0.95  | `physics/drift.model.ts:567-580` (budget), `:572-573` (unused arg), `:201` (entry rate), `:66` (clamp 0.45); the only speed-coupled term is the scrub `:618-646`                                                                  |
| **G-M9**  | The **engage gate is a binary cliff at exactly \|steer\| = 0.5** with no partial slide and no fade. Measured: 0.49 -> 0.0000 rad, 0.50 -> 0.4512 rad. A player who steers at 0.49 sees literally nothing.                                                                                                                                                                                                                                                                                                                                                                                                    | missing     | 0.94  | `physics/drift.model.ts:189` (threshold), `:515-517` (enforcement, `reArmed`)                                                                                                                                                     |
| **G-M10** | **The served page opens in `ai` mode** and the autopilot steers proportionally, so it rarely reaches the gate (needs \|headingError\| >= 0.227 rad; steady-state steer = 44.5/R, i.e. the drift engages only in corners tighter than R = 89 m). Motion consequence: on the live page the car is either in a 25.8 deg slide or in **literally zero** traction loss - there is no in-between behaviour.                                                                                                                                                                                                        | missing     | 0.90  | `host/mode-selector.ts:23`; `host/race-session.ts:235-240`; `physics/ai.autopilot.ts:153-181` (gain at `:32`); yaw law `physics.step.service.ts:192`                                                                              |
| **G-M11** | The **MIN_SPEED gate is a hard cliff** (no fade-in) and the **off-road floor gate is one tick stale**: `isOffRoadSpeedFloorEngaged` reads the TICK-START speed and then guards the POST-SCRUB speed.                                                                                                                                                                                                                                                                                                                                                                                                         | partial     | 0.93  | `physics/drift.model.ts:184`, `:495`, `:679`; `physics.step.service.ts:524-546` -> `:436-444`                                                                                                                                     |
| **G-M12** | **NON-FINDING - the served bundle is current, so the break is a DESIGN break, not a build break.** Both served copies are byte-identical (SHA-256 `3C255A7F...FFD9D8`, 398,699 bytes) and the bundle map's `sourcesContent` for `drift.model.ts`, `physics.step.service.ts` and `browser-entry.ts` matches the on-disk sources byte for byte. **Corollary:** the fixture asserts `\|slip\| > 0.4` and the live-equivalent run measures 0.4512, so every drift test **passes** while the feature is invisible - the break lives entirely downstream of the physics state, where there is no assertion at all. | non-finding | 0.99  | `examples/neatRacing/index.html:97-100`; `physics/physics.step.service.test.ts:626-631`                                                                                                                                           |

---

#### 4. Risks

| #    | Risk                                                                                                                                                                                                                                       | Severity | Mitigation                                                                                                                                     |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| R-M1 | Implementing the fix by re-pointing the **camera** at the travel direction without adding a state field leaves the body yaw reading the pose (33.8 %) and the frame internally inconsistent (a travel-locked camera with a crabbing body). | HIGH     | Land the state field and the camera change in the same slice; the body yaw must read the published slip, not the pose.                         |
| R-M2 | The **0.5 engage cliff** (2.F) means the feature is binary. Once the camera is fixed, a user who steers at 0.49 still sees literally nothing, and will report "still no drift".                                                            | HIGH     | Decide deliberately: keep the cliff (and say so) or add a proportional entry ramp. The `14N-V5-audit-envelope` wave owns the numeric envelope. |
| R-M3 | The **ai-mode default** (2.G) means the _served_ page may still show no drift after the fix, because the autopilot rarely reaches `                                                                                                        | steer    | >= 0.5`.                                                                                                                                       | MEDIUM | Verify with a manual-mode full-steer run first; do not accept the ai-mode page as the acceptance evidence. |
| R-M4 | The live path has **no collision resolution at all** (2.C). A drift fix will make the car slide - and slide straight through walls, because `resolveCarCollisions` is never called.                                                        | MEDIUM   | Wire `resolveCarCollisions` (or state explicitly that walls are out of scope for this slice).                                                  |
| R-M5 | The scrub's **one-tick staleness and self-amplifying feedback** (2.E) will become visible once the drift is visible; a reviewer may misread the rising scrub as a runaway.                                                                 | LOW      | Read the published slip for the scrub, or document the lag explicitly.                                                                         |
| R-M6 | The **MIN_SPEED cliff at 15 m/s** (2.H) means the slide appears abruptly; on the ai-mode track (which slows for corners) the car may cross 15 m/s mid-corner and pop into a slide.                                                         | LOW      | Observe in a manual run; the envelope wave owns the fade decision.                                                                             |
| R-M7 | This audit is the only wave with **runtime** numbers; the sibling waves' static-code rows were written without them and one of them (visuals G-V2) mis-describes the scrub as discarded.                                                   | MEDIUM   | The merge must apply the 2.D correction: the scrub IS applied to the speed and to the translation.                                             |

---

#### 5. Evidence index

**Files read (production, read-only):** `examples/neatRacing/physics/physics.step.service.ts` (L78-130 the state contract, L142-195 the constants, L290-320 the initial state, L355-500 the tick pipeline, L505-580 the helpers) · `examples/neatRacing/physics/drift.model.ts` (L1-140 the sign convention + the constants, L480-600 the commitment + the grip budget, L600-800 the scrub + the chase + the pose spring + `computeSlipAngle`) · `examples/neatRacing/browser-entry/browser-entry.ts` (L83 the collision import, L112-152 the seam interface, L519-527 the camera constants, L557-585 the scene yaw/position helpers, L733 the surface resolver, L920-1070 the render bridge, L1149 the `stepPhysics` seam) · `examples/neatRacing/engine/passes/car.pass.ts` (L435-497 the pass state, L1405-1430 the shadow yaw, L1595-1630 the body yaw) · `examples/neatRacing/host/race-session.ts` (L210-275 the live loop) · `examples/neatRacing/host/input.ts` (the whole file - the keyboard axis map) · `examples/neatRacing/host/mode-selector.ts` (L23 the default mode) · `examples/neatRacing/physics/ai.autopilot.ts` (L20-60 the constants, L140-185 the steer law) · `examples/neatRacing/physics/collision.utils.ts` (L155-260 the resolver + `resolveCarCollisions`, L378-460 the world velocity + the drift zeroing) · `examples/neatRacing/index.html` (L80-140 the bundle path) · `examples/neatRacing/physics/physics.step.service.test.ts` (L575-660 the drift fixtures).

**Repo-wide greps:** `slipAngle|lateralVelocity` (non-test) · `drift\.pose|driftYawOffsetRad|car\.heading|velocityHeading|car\.yawRad` (non-test) · `resolveCarCollisions|reboundPastApronEdge` (all `*.ts`/`*.tsx`/`*.js`/`*.mjs`, excluding `node_modules`, `dist`, `docs/assets`, the bundles and `coverage`).

**Runtime method (read-only, no browser):** the repository's own `physics.step.service.ts` was bundled with the repo's `esbuild` into `$TEMP` (outside the repository) and driven headlessly at the real 60 Hz fixed timestep with the real `stepPhysics`; the repository working tree was not modified. Two runs: (1) a 900-tick straight-line spin-up to the 89 m/s top speed followed by 300 ticks of `throttle=1, steer=1`; (2) a 7-point steer sweep (0.30 / 0.45 / 0.49 / 0.50 / 0.55 / 0.70 / 1.00), 300 ticks each from the same 89 m/s start.

**Git state (read-only inspection only; no mutating command was run):** the 14N drift work is **uncommitted** - `git diff --stat` shows `drift.model.ts` (+917/-), `physics.step.service.ts` (+179), `car.pass.ts` (+52), `collision.utils.ts` (+36) plus test churn, on top of HEAD `023a45b6 14g`. The audit's line numbers are cited against the **working tree**, which is what the served bundle was built from (2.I).

**No production byte was touched.** The only writes are this artifact and the `PlanUpdate` block in `plans/neatracing-voxel-racer.plans.md`.

### B-6 — Step 14N arcade drift references (drift transient tuning) + the spring-damper addendum

> **Provenance.** The 2026-10-24 drift-feel scout artifact (author role `02-researching`, Tier-1 drift-feel
> scout; charter: "Mario Kart / OutRun 2 / Ridge Racer / Daytona / Virtua Racing / TrackMania drift
> transients — entry attack, hold, countersteer snap-back, overshoot, numeric rates") plus the 2026-10-25
> game-feel scout ADDENDUM (section E6 / decision D6 / risks R11–R15). Sub-orchestrators used: `research`
> (harness built-in web/ROM-decomp source sweep), `implementation-pattern-scout`, `boundary-mapper`.
> Status at fold: RESEARCH COMPLETE — the feel direction and the floor value remained RED-slice decisions; the
> model and the constant set were settled and numerically verified.
> **Fold note (2026-10-31).** Folded here as Part B-6; the sibling file is DELETED. The decisive FINDINGS
> (`GF-1 … GF-16`) live in the tracker plan tail (`## PlanUpdate - Step 14N SCOUT game-feel`) and are cited,
> not duplicated, here.

---

- **Plan**: `plans/neatracing-voxel-racer.plans.md` (Step 14N, `[WIP]`)
- **Step**: 14N — Arcade drift FEEL sharpening
- **Scout charter**: _"Mario Kart / OutRun 2 / Ridge Racer / Daytona / Virtua Racing / TrackMania drift transients — entry attack, hold, countersteer snap-back, overshoot, numeric rates."_
- **Author role**: 02-researching (Tier-1 drift-feel scout), read-only research
- **Date**: 2026-10-24
- **Primary sibling artifact**: `plans/neatracing-voxel-racer.research.md` (integration-coupling design — unchanged by this file)
- **Sub-orchestrators used**: `research` (harness built-in, web/ROM-decomp source sweep), `implementation-pattern-scout` (repo Tier-3), `boundary-mapper` (repo Tier-3)
- **Status**: RESEARCH COMPLETE — the _feel direction_ and the _floor value_ remain RED-slice decisions; the **model** and the **constant set** below are settled and numerically verified.
- **Addendum (2026-10-25, game-feel scout)**: a **second pass** on this artifact adds the
  **spring-damper response layer** — _attack rates, damping ratio, overshoot, anticipation_ —
  as **section E6** (evidence) and **D6** (decision) below, plus risks **R11–R15** and the
  matching certainty rows. **E1–E5, D1–D5 and R1–R10 above are UNCHANGED** (the arcade-refs
  brief stands); D6 **amends** D1's stage 7 and D2's constants only. The decisive findings
  are appended to `plans/neatracing-voxel-racer.plans.md` as
  `## PlanUpdate - Step 14N SCOUT game-feel`.

---

#### Question

How do the classic arcade racers tune the **drift transient**, decomposed into
**entry attack → hold → countersteer snap-back → overshoot → numeric rates**, and what
**concrete constant pattern** should `examples/neatRacing/physics/drift.model.ts` adopt to
answer the user verdict _"tilt too slow/icy; bounce-back must be sharper; want exciting
arcade drift"_?

**Addendum question (game-feel, 2026-10-25).** Given the committed-drift model and its
second-order **pose spring**, what **spring-damper response** delivers the arcade feel —
specifically (a) the **attack rate**, (b) the **damping ratio**, (c) the **overshoot**
behaviour, and (d) the **anticipation** (the body leading the slide) — and what are the
**concrete accumulator parameters**? The sub-questions that decide it: _is the entry
overshoot a defect to be damped, or the energy reservoir that produces the far-side pop?_
_is the pop caused by the pose target's discontinuity, or by the spring?_ _does the spring
set the perceived attack at all?_ — answered in **E6** and **D6**.

---

#### Evidence

##### E1 — The current model, reproduced from source (verified)

`examples/neatRacing/physics/drift.model.ts` (373 lines) owns eight constants and
`computeNextLateralVelocity` (`:179-201`). Per 60 Hz tick, with
`yawRatePerTick = steer · STEERING_RATE · speedPercent · dt`
(`physics.step.service.ts:473-479`, `STEERING_RATE = 2` at `:193`):

```
outwardPush  = -centrifugalMultiplier · yawRatePerTick · (speed - 15)      // zero below 15 m/s
slipAngle    = atan2(lateralVelocity, speed)
traction     = 1 / (1 + max(0, |slip| - 0.25) · 2 · speedPercent)²
counterDelta = -sign(slip) · |steer| · speedPercent · 12 · dt               // only when steer·sign(slip) > 0
nextLateralVelocity = (lat + outwardPush + counterDelta) · (1 - 4 · traction · dt)
```

Production constants: `CENTRIFUGAL_MULTIPLIER = 1.5` (`physics.step.service.ts:184`),
`MAX_SPEED_UNITS_PER_S = 89` (`:134`), `ACCEL = 89/5 = 17.8 m/s²` (`:140`).

> **C1 — the JSDoc lies.** `drift.model.ts:119` states the centrifugal multiplier "defaults
> to 0.3"; the production value is **1.5**. Every number below uses **1.5**.

##### E2 — FIRST-HAND SIMULATION of the current model (reproduced equations, 60 Hz)

| Scenario                                                               | Result                                                                                                                                                                        |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C, 25 m/s (the fixture's 90 km/h operating point), full left steer** | slip plateaus at **-4.50°** (0.0785 rad); visual proxy plateaus at **0.0392 rad**; takes **~0.25 s** to reach 63 % and **~0.75 s** to reach 95 %                              |
| **A, accelerating from rest, full left steer at t=1 s**                | slip grows monotonically with speed: -1.51° (1.25 s) → -6.69° (2 s) → -10.2° (2.5 s) → -17.2° (3.5 s) → -23.4° (4 s) — **never plateaus** while the car is still accelerating |
| **E, 89 m/s (top speed), full left steer**                             | -21.97° (0.25 s) → -37.47° (0.5 s) → -56.69° (1 s) → **-65.82° (1.5 s) and still growing**; effective damping collapses **4.0/s → 0.51/s** as the slip grows                  |
| **D, 25 m/s, full COUNTERSTEER at t=2.5 s**                            | slip goes **-4.49° → +1.58° (+0.25 s) → +4.48° (settled)** — it crosses zero and **parks in a mirror-image equilibrium of the same size**                                     |
| **E, 89 m/s, full COUNTERSTEER at t=2.5 s**                            | **-65.82° → -27.16° (+0.5 s) → +10.55° (+0.75 s) → +52.15° (+1.5 s)** — a full inversion, never recovered                                                                     |

> **C4 — the slide is not "too small", it is UNBOUNDED.** The damping term is
> `4 · traction`, and `traction` _falls_ as the slip grows ⇒ more slip ⇒ less damping ⇒
> more slip. That is **positive feedback, not a grip budget**. The "icy" read is a
> **first-0.5-seconds** problem; after that the slide runs away.

> **C5 — THE COUNTERSTEER CANNOT CATCH THE SLIDE. IT MIRRORS IT.** Two independent
> causes, both verified in source:
>
> 1. **The outward push is a function of the yaw rate, and the yaw rate is a function of
>    the steer.** Countersteering therefore _reverses_ the "centrifugal" push, and the two
>    terms **add** instead of opposing. (At 25 m/s: push +8.4 m/s², counter +3.4 m/s², both
>    positive, against a -4.5° slide.)
> 2. **The countersteer gate closes the instant the slip crosses zero**
>    (`steer · sign(slip) > 0`). The authority that could catch the slide is switched off
>    exactly when it is needed, leaving the reversed push to drive the car to the
>    opposite lock.
>
> ⇒ **"Bounce-back must be sharper" is a MISSING-AUTHORITY / WRONG-GATE problem, not a
> rate problem.** Raising `NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S` (the
> code-map scout's lever #5) would make the inversion **faster and harder**, not better.

> **C6 — the countersteer authority is also dimensionally tiny.** `12 · speedPercent` =
> 3.4 m/s² at 25 m/s and 12 m/s² at 89 m/s, versus the push's 8.4 / 222 m/s².

> **C7 — `outwardPush` is dimensionally inconsistent.** `yawRatePerTick` is _radians per
> tick_; `rad/tick · m/s` is not m/s. The correct centripetal term is
> `yawRate[rad/s] · speed[m/s] · dt`. The current form is a per-tick velocity delta
> (`-222 m/s` per tick at 89 m/s), which is why the top-speed slide explodes.

> **C8 — the longitudinal scrub can out-accelerate the engine.**
> `0.6 · |slip| · speed · speedPercent` = **24.2 m/s²** at 26°/89 m/s, versus the
> engine's **17.8 m/s²**. The car cannot hold speed in a deep slide.

##### E3 — The proxy identity (independently confirmed by three agents)

`plans/neatracing-voxel-racer.research.md:66-89` (E3) and, separately, the repo
`boundary-mapper` (RISK-1) and the repo `implementation-pattern-scout` (§3) all agree:

```
rendered body yaw = heading + slipAngle · GAIN
travel direction  = heading + slipAngle
proxy = |bodyYaw - travelDir| = |slipAngle| · |GAIN - 1|
```

- The recorded browser baseline **0.049388550574608114 rad** ⇒ `|slip| ≈ 0.0988 rad = 5.66°`
  (confirmed to the last digit: `0.049388550574608114 = 0.5 × 0.098777101149216228`).
- **The proxy is V-shaped in the gain with a NULL at `GAIN = 1.0`.** Raising the gain
  toward 1.0 makes the slide _weaker_. The only monotone direction is **down**.
- The browser proxy is a **render-level, render-rate-dependent** quantity (recorded spread
  0.0494 / 0.0603 / 0.3599 rad). The deterministic twin is the model's own `slipAngle`.

##### E4 — Arcade references: the five phases, per title

Labels: `[SRC]` = value recovered from source/decompilation · `[CONSENSUS]` = community
measurement, not from source · `[SIM]` = derived by simulation · `[NOT FOUND]` = no public
data; **nothing invented**.

| Game                                                                                           | Entry attack                                                                                                                                                      | Hold                                                                                                                                 | Countersteer snap-back                                                                                                                     | Overshoot                                                 | Speed scrub                                                                                                                                              |
| ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mario Kart Wii**                                                                             | commanded drift angle ramps **+150 °/frame** toward target; auto ±30 °/frame; steering lag `reactivity` `[SRC]`                                                   | clamp **±60°**; charge tiers 270/300/75; `stabilizationFactor` 0.5/0.22 `[SRC]`                                                      | **linear −2.0 °/frame = −120 °/s**; velocity filter `dirDiff *= 0.1` ⇒ **τ ≈ 167 ms** `[SRC]`                                              | prevented by sign-flip guard + turn fade                  | accel swaps to the drift curve `[SRC]`                                                                                                                   |
| **Mario Kart DS**                                                                              | **rate-limited ramp ±200 units/tick** to a hard clamp `[SRC]`                                                                                                     | 2 tiers, 10-frame alternation lockout `[SRC]`                                                                                        | **linear ±200 units/tick**, hard clamp at 0 `[SRC]`                                                                                        | impossible (clamped)                                      | `speed *= 0.95/frame` above cap `[SRC]`                                                                                                                  |
| **Super Mario Kart**                                                                           | two independent ramps: velocity lag **13.5°/18°**, pose offset **46°/57°**; hop 17 frames `[SRC]`                                                                 | no mini-turbo (wiki "3 s" claim conflicts with the ROM)                                                                              | linear per-frame decay **with a residual** (−64/−96) `[SRC]`                                                                               | spin-out at ±$480/frame `[SRC]`                           | ~30 % of pace lost `[SRC]`                                                                                                                               |
| **Mario Kart 64**                                                                              | first-order lag **τ = 12 frames**, seeded `unk_078 · 0.14` `[SRC]`                                                                                                | clamp **±40.08°** (0x1C70); mini-turbo tiers 10/18 frames `[SRC]`                                                                    | **exponential τ = 12 frames ≈ 0.4 s @30 Hz**; stick remap eats 13/53 countersteer units `[SRC]`                                            | impossible (+0.33° deadband)                              | `decel +=                                                                                                                                                | angle_deg           | · 0.004/frame`— **2.5× cheaper than a hard turn**`[SRC]`                                                                                                                                                                   |
| **OutRun 2**                                                                                   | **NOT FOUND** — no assist; 4-step manual technique (lift→brake→steer→throttle)                                                                                    | NOT FOUND; steering-motor power 100/90/80/60 %                                                                                       | NOT FOUND                                                                                                                                  | catch-the-slide is designed in, unquantified              | NOT FOUND                                                                                                                                                |
| **Ridge Racer (PS1)**                                                                          | wheel ±0x500 of ±0x1000 per frame (**0→full lock 3.2 frames ≈ 107 ms**); **velocity direction chases heading at 1/5 per frame ⇒ τ = 5 frames = 166.7 ms** `[SRC]` | grip = **hard projection** (lateral velocity discarded each frame); drift gate `speed > 320` raw; per-car `lim` 9 100–21 000 `[SRC]` | **exponential, τ = 5 frames = 166.7 ms**; wheel to neutral in 2 frames (τ ≈ 48 ms) `[SRC]`                                                 | **no anti-overshoot clamp**; spin cap 0x249 BAM12 ≈ 51.4° | `speed · cos ψ` per frame; coast drag ×996/1000 `[SRC]`                                                                                                  |
| **Ridge Racer Type 4**                                                                         | ~0.5 s accelerator release `[CONSENSUS]`                                                                                                                          | 3 drift stages, no seconds                                                                                                           | NOT FOUND                                                                                                                                  | rotation clamp provably exists, angle NOT FOUND           | NOT FOUND                                                                                                                                                |
| **Ridge Racer 7**                                                                              | accel-drift / brake-drift; minimum brake-pulse length exists, value NOT FOUND                                                                                     | scrub saturates **≈30°** in the driver's view                                                                                        | assist-mediated, setup variable; **fails open** (mis-countersteer ⇒ the car drifts the _other_ way) — a **relay / re-latch** `[CONSENSUS]` | 360° drifts possible ⇒ no hard rotation clamp             | per-car tiers only                                                                                                                                       |
| **Daytona USA** _(Sega Rally '95 sibling proxy — Daytona USA's own ROM physics is not public)_ | yaw gated on `                                                                                                                                                    | steer                                                                                                                                | > 0.0349`and`speed > 0.3704`; slide blends toward the road tangent `[SRC]`                                                                 | road-relative; self-limiting                              | **`error *= cos Δθ` per frame** ⇒ quadratic convergence: 30° → ×0.866 (τ 125 ms), 60° → ×0.500 (τ 33 ms), 90° → instant, 15° → ×0.966 (τ 489 ms) `[SRC]` | **none** (monotone) | speed converges to the corner target (τ 2.2–3.6 s) `[SRC]`                                                                                                                                                                 |
| **Virtua Racing (32X)** — _the grip-based CONTRAST_                                            | steering accel 24 units/frame, EMA α=0.5 (τ = 24 ms); **drift threshold `                                                                                         | slip                                                                                                                                 | > 0.215 rad = 12.3°`** `[SRC]`                                                                                                             | saturated grip budget (max 1.504× slip); spin-out at `    | lateral_vel                                                                                                                                              | ≥ $C0F2`            | **the reference arcade countersteer: steering accel HALVED and `drift -= (drift >> 3)` ⇒ ×7/8 per frame ⇒ τ = 7.49 frames = 124.8 ms @60 Hz**; 90 % gone in 284 ms; **hard snap to zero on sign reversal; hard zero when ` | v   | ≤ 15`** `[SRC]` | structurally excluded | `final_force = (net_force >> 1) · grip >> 7` `[SRC]` |
| **TrackMania**                                                                                 | tire-saturation event (DIRECT→CAPTURED on `slip_seen`); steering slew 20.0 ⇒ 0→full lock in **50 ms @100 Hz** `[SRC]`                                             | **captured-angle rail**, clamp **π/4 = 45°**; grip clamped to `min(requested, μN)` `[SRC]`                                           | **hard state reset to zero** on captured-angle sign cross (`capture_exit_side_speed_max = 1.0`) `[SRC]`                                    | none                                                      | proportional to over-limit demand, saturating `[SRC]`                                                                                                    |

**Cross-game synthesis.** All eleven titles use one of three architectures:
(a) a **velocity-direction lag filter**, (b) a **saturated grip budget**, or
(c) a **clamped design angle + a separate pose channel**. In _every_ case the slide is a
**commanded, clamped quantity that the body chases** — never an emergent force-balance
equilibrium. The shared grammar is:

> **entry = a rate-limited ramp to a hard clamp; exit = a linear unwind; the countersteer
> has its own authority applied _against_ the slide; the sign is latched at entry.**

Only two real overshoot sources exist in the whole corpus: RR7's **failing-open countersteer
relay**, and the SMK/MKWii **pose-vs-velocity split** (a dedicated second channel that the
body swings on).

**Anchor band** (for picking a clamp): Ridge Racer PS1 steady slip **14.1°**, Virtua Racing
drift threshold **12.3°**, Mario Kart 64 clamp **40.1°**, Mario Kart Wii clamp **60°**,
TrackMania clamp **45°**, Ridge Racer 7 saturation **≈30°**.

##### E5 — Local repo constraints (repo Tier-3 scouts)

- **Naming grammar** (`implementation-pattern-scout`): one `export const` block at the top
  of the file; `NEATRACING_DRIFT_<SUBJECT>_<UNIT>`; unit suffixes `_RAD`, `_MPS`, `_PER_S`,
  `_PER_RAD`; floor/ceiling infixes `MIN_` / `MAX_`; JSDoc = one-line summary, blank line,
  then a 2–6-line "why" naming the AC id and the measured rationale; **no** `@param`/`@returns`
  on constants.
- **Only one production importer**: `physics.step.service.ts` (calls `computeNextLateralVelocity`
  at `:388`, `computeLongitudinalTractionLoss` at `:406`, `computeSlipAngle` at `:412`, and
  inlines `nextSlipAngle · GAIN` at `:417`). **The car pass (`engine/passes/car.pass.ts`)
  imports nothing from the drift model** — it consumes one opaque radian scalar
  (`car.driftYawOffsetRad` at `:438`, `:1365`, `:1567`). ⇒ **no drift lever requires
  re-authoring the car pass.**
- **Drift levers that live _outside_ the declared AC-087 file set**:
  `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER` and `NEATRACING_PHYSICS_STEERING_RATE` are in
  `physics.step.service.ts` — a slice file-set amendment is required to touch them.
- **The visual-yaw formula exists twice**: `drift.model.ts:363 computeDriftYawOffset()` is
  **mathematically identical** to `physics.step.service.ts:417` but has **zero production
  consumers** (dead export / divergent twin).
- **AC-088 is unmeasurable through the named levers** (both repo scouts agree): the tire
  gradient is a **fragment-stage** effect, while the fixture's `analyzeWheelShading()` reads
  the **wheel VBO vertex colours**, which the mesh builder writes _uniform_ ("Vertex colors
  stay uniform rubber", `car.pass.ts:559-561`) ⇒ `upperAvg == lowerAvg == 0.28565` **by
  construction**. No value of `NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR`/`_DARK_FACTOR`/
  `NEATRACING_WHEEL_HUB_HEIGHT_M` can make the measured gradient non-zero. This is a
  **plan-wording correction**, not a tuning task.
- **Spec §5.2 is stale** against the code (`MIN_GRIP_FACTOR = 0.25` deleted, countersteer 4.0
  vs 12, lateral grip 6.0 vs 4, and the **countersteer sign rule is stated backwards**).

##### E6 — Game-feel response design (the spring-damper layer) — SECOND-HAND SIMULATION

_Added by the game-feel scout (2026-10-25). E1–E5 and D1–D5 above are UNCHANGED._

**Method.** I reproduced the D1 seven-stage arcade model at 60 Hz in a scratch harness
(`tmp/driftfeel-final.mjs`, gitignored) and **validated the harness against this artifact's
own published D3 numbers before trusting any new figure**: entry overshoot 17.6 %, catch dive
−3.67°, far-side pop **+2.61°**, settle 550 ms, max pose-target step **−4.65°**, peak pose
rate 108 °/s — all matched to the digit. Every number in E6/D6 is machine-produced by that
harness. (Scratch harnesses are in `tmp/`, which is gitignored; nothing under `src/` or
`examples/` was touched.)

**E6.1 — The arcade-refs' stage-7 pose target is DISCONTINUOUS in the steering input.**
`poseTarget = slip · GAIN + 0.08 · steer · speedPercent · min(1, |slip|/0.10)`. The steer term
flips sign on every countersteer, so the target **steps** by `2 · 0.08 · speedPercent`:
**1.83° at 20 % speed · 5.13° at 56 % (the measured hold point) · 9.17° at 89 m/s.** Measured
in-sim: **−4.65° in a single 1/60 s tick at 49.8 m/s.** A second-order system turns a target
step into a velocity impulse, so the kick's size is a function of the **step size, not of ζ**.

**E6.2 — The entry overshoot is the pop's ENERGY RESERVOIR, not a defect.** Replacing the
arcade-refs' symmetric `ζ = 0.35` with an asymmetric rule (`ζ_attack = 0.9` / `ζ_release = 0.35`)
collapses the far-side pop from **2.61° to 0.00°** (measured at ω = 18 and ω = 24). The entry
overshoot is **1.79° absolute** (17.6 % of the 10.16° hold target) and is the "settle into the
slide" the arcade feel needs. ⇒ **The symmetric ζ = 0.35 spring IS the pop's engine.**

**E6.3 — The pop is NOT the step.** Replacing the raw articulation with a slew-limited one
keeps the pop at **2.22–2.30°** (vs the arcade's 2.61°) ⇒ the discontinuity contributes only
~0.3–0.4° of the pop and ~0.7° of the outward dive. ⇒ **the step is a TWITCH defect; the pop
must be KEPT and the step must GO.**

**E6.4 — The attack rate belongs to the physical model, not the accumulator.** The onset is
**identical across every variant tested** (raw / τ-filtered / slewed / asymmetric, ω = 12…32):
**25 % of the hold δ at 33.3 ms, 50 % at 66.7 ms** after the steer input. The spring adds ≤ 1–2
ticks of lag; the committed-angle ramp + the slip chase set the attack. ⇒ **ω is free to be
raised for a crisper pop without touching the perceived attack.**

**E6.5 — ω and ζ are the SAME trade-off dial (pop vs ring-out).** Measured (slew 20 /s):
ω = 15 → pop 2.62° / settle 650 ms · **ω = 18 → 2.22° / 550 ms** · ω = 21 → 1.76° / 383 ms ·
ω = 24 → 1.37° / 367 ms; ζ = 0.25 → 3.40° / 717 ms · ζ = 0.35 → 2.22° / 550 ms · ζ = 0.45 →
1.35° / 433 ms. The pop and the settle are strictly monotone in both, in opposite directions.

**E6.6 — NEW: the arcade-refs' pose spring OVERSHOOTS the declared envelope by 2.00 %.** The
declared `DRIFT_VISUAL_YAW_LIMIT_RAD` = `MAX_SLIP_ANGLE_RAD · VISUAL_YAW_GAIN_RAD` =
0.45 · 0.5 = **0.2250 rad (12.89°)**. The measured **max |pose| = 13.15° = 0.2295 rad** during
the entry overshoot ⇒ **2.00 % OVER the limit.** The slew-limited variant lands at 12.87° =
0.2246 rad (0.19 % inside). ⇒ **the limit must be re-derived from the measured peak, not from
`MAX_SLIP · GAIN`** — and ω = 18 is **the largest ω that keeps the pose inside the declared
envelope at ζ = 0.35** (ω = 15 → 13.29° = 0.232 rad, 3 % over).

**E6.7 — Tick placement and the visual channel's reach (repo Tier-3 `boundary-mapper`).** The
drift yaw is written once per 60 Hz tick (`physics.step.service.ts:416-417` → `:440`), bridged
once per render frame (`browser-entry.ts:941`), and read in exactly **TWO** places
(`car.pass.ts:1365` shadow, `:1567` body matrix); the tires inherit it via body-matrix
parenting (`:1598`). **There is NO interpolation anywhere**, and the tick:frame ratio is
`displayHz/60` (1 @ 60 Hz, 2 @ 144 Hz, 2.75 @ 165 Hz; max 4 catch-up steps/frame). ⇒ **a
per-frame spring would run 2.4× faster at 144 Hz ⇒ the accumulator MUST be stepped inside the
60 Hz `stepPhysics` tick.** And **the chase camera does NOT consume `driftYawOffsetRad`**
(`browser-entry.ts:955` — the camera yaw is trajectory-only) ⇒ **the pose channel is the ONLY
visual read of the slide**, and the anticipation term is the first thing the player sees.

**E6.8 — There is NO second-order precedent in the repo (repo Tier-3
`implementation-pattern-scout`).** No spring/damper exists in `src/` or `examples/neatRacing`;
the house idiom is FIRST-ORDER TAU smoothing (`editor.camera.ts:48-56,430-447`;
`NEATRACING_EDITOR_CAMERA_SMOOTHING_TAU_S = 0.08`). A first-order filter cannot overshoot ⇒
the spring is a corpus question, not an idiom question. The state must be a NEW SEPARATE type
on `PhysicsCarState` (safe to extend), **not** on `DriftModelState` (byte-identity pinned,
AC-047D) — and with the slew it is **three** scalars, not two.

**E6.9 — Guarantees and stability (measured).** Straight line (no steering input, 5 s):
**max |pose| = 0.000000000 rad, max |poseTarget| = 0.000000000 rad, max per-tick step =
0.000000°, final pose = 0.000000000 rad** ⇒ the zero-straight-line-yaw guarantee is EXACT.
Residual pose after a settled drift: **−0.000°**. Partial countersteer (−0.4 steer): pop
**1.92°**, dive −1.88°, max step 1.89°, settle 533 ms ⇒ the pop is not a full-input artefact.
Stability at `dt = 1/60` (semi-implicit Euler): ω = 18 → ω·dt 0.300, C·dt 0.210, K·dt² 0.090;
ω = 24 → 0.400 / 0.280 / 0.160; ω = 32 → 0.533 / 0.373 / 0.284 ⇒ **unconditionally stable, no
sub-stepping required.** Unit step at the arcade-refs' point: **t63 = 50.0 ms, t95 = 83.3 ms,
first overshoot 29.06 %, peak at 150 ms, settle-after-t95 450 ms.**

**E6.10 — STRESS (flagged, not fixed): steer chatter.** A 4 Hz square-wave steer on the
committed-drift model drives the far-side body swing to **22.6°** (556 % of the hold δ) and the
settle to ~2 s. The slew bounds the per-tick target step but does NOT suppress the committed
model's chatter response ⇒ **this is a property of the D1 LATCH design, not of the pose
channel** — out of the game-feel charter's scope, raised as a risk on the latch.

---

#### Decision

##### D1 — THE MODEL (stated explicitly)

**Committed-drift-angle arcade model.** The drift becomes a _commanded, clamped state_; the
body chases it; the countersteer gets its **own** authority applied **against** the slide.

**Sign convention (pinned to production, verified):**
`lateralVelocity > 0` = the velocity vector lies to the **LEFT** of the heading;
`slipAngle = atan2(lateralVelocity, speed)`, so a **LEFT** turn (steer > 0) produces a
**NEGATIVE** slip (the nose leads, the velocity lags right). The visible yaw offset is
`slipAngle · GAIN`, hence **negative** in a left turn. _(The upstream draft of this model
wrote the visible-pose target as `-committed`; with this convention that is a sign error and
would mirror the whole drift. Corrected here to `+committedSlip`.)_

```
(0) DRIFT LATCH
    turnSign  = latched at entry = sign(steer that started the drift)
    countersteering = (turnSign != 0) && sign(steer) == -turnSign
    drifting        = (turnSign != 0) && !countersteering

(1) COMMITTED DRIFT ANGLE                       [production slip sign: negative = left turn]
    if (drifting)  committedSlip -= turnSign * ENTRY_RATE_RAD_PER_S * |steer| * dt
    else            committedSlip  = sign(committedSlip)
                                     * max(0, |committedSlip| - EXIT_RATE_RAD_PER_S * dt)
    committedSlip  = clamp(committedSlip, -MAX_SLIP_ANGLE_RAD, +MAX_SLIP_ANGLE_RAD)

(2) TIRE SATURATION                             [a BUDGET, not a damping multiplier]
    gripFactor = 1 / (1 + GRIP_FALLOFF_RATE_PER_RAD
                        * max(0, |slip| - GRIP_FALLOFF_THRESHOLD_RAD))

(3) PHYSICAL CHASE                              [the body chases the commanded angle]
    vTarget = tan(committedSlip) * speed
    vLat   += (vTarget - vLat) * (1 - exp(-dt * gripFactor / SLIP_RESPONSE_TAU_S))

(4) COUNTERSTEER AUTHORITY                      [applied AGAINST the slide]
    if (countersteering && |slip| > COUNTERSTEER_MIN_SLIP_RAD)
        vLat += turnSign * COUNTERSTEER_AUTHORITY_MPS * |steer| * speedPercent * dt

(5) HARD ZERO                                   [VR's "|v| <= 15 => 0"]
    if (!drifting && |slip| < SNAP_BACK_ZERO_SLIP_RAD) { vLat = 0; committedSlip = 0; }

(6) SPEED SCRUB                                 [saturating, cheaper than a hard turn]
    speed -= SPEED_SCRUB_COEFF_PER_S * |slip| * speed * speedPercent * dt

(7) VISIBLE POSE (second order — where the overshoot lives)
    poseTarget = slip * VISUAL_YAW_GAIN_RAD
               + POSE_STEER_YAW_RAD * steer * speedPercent
                 * min(1, |slip| / POSE_STEER_YAW_FULL_SLIP_RAD)     <-- slip-gated
    poseAcc  = POSE_STIFFNESS_RAD_PER_S2 * (poseTarget - pose)
             - POSE_DAMPING_RAD_PER_S * poseRate
    poseRate += poseAcc * dt;  pose += poseRate * dt
```

**Why this shape and not the alternatives:**

1. _It is the only shape that can produce the requested behaviour._ A first-order
   lateral-velocity model is **monotone** — it cannot overshoot (verified: variant (b) below
   produced **0.00°** of swing-past). The overshoot must live in a **second-order** channel.
2. _It fixes C5 at the root._ The countersteer no longer reaches the slide through
   `sign(slip)`; it has its own authority, and the drift sign is **latched** so the
   "reversal" of C5 becomes a controlled **unwind**, not a mirror.
3. _It fixes C4 at the root._ The slide is bounded by `MAX_SLIP_ANGLE_RAD`, not by a force
   balance.
4. _The slip-gated steer-yaw term in (7) is what makes the swing visible_ **and** keeps the
   **zero-straight-line-yaw** guarantee (the term vanishes when `slip = 0`).

##### D2 — THE CONSTANT SET (for `examples/neatRacing/physics/drift.model.ts`)

Repo grammar `NEATRACING_DRIFT_<SUBJECT>_<UNIT>`, one `export const` block at the top of
the file. Values marked **[SIM]** are **verified by my own 60 Hz simulation** of the
equations above; **[REF]** values are the arcade anchor the number was chosen against.

```ts
// ---- drift envelope ----------------------------------------------------------
export const NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45; // rad (25.8°)  [SIM][REF: TrackMania 45°, MK64 40.1°]
export const NEATRACING_DRIFT_MIN_SPEED_MPS = 15; // m/s          (UNCHANGED)

// ---- entry attack / hold -----------------------------------------------------
export const NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S = 3.5; // rad/s (201°/s) [SIM][REF: MKWii 150°/frame]
export const NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S = 2.0; // rad/s (115°/s) [SIM][REF: MKWii -2.0°/frame]
export const NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S = 0.045; // s  (τ of the chase) [SIM][REF: VR 24 ms, RR 167 ms]

// ---- tire saturation (a budget, not a damping multiplier) ---------------------
export const NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD = 0.2; // rad (11.5°) [REF: VR threshold 12.3°]
export const NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD = 2; // 1/rad (UNCHANGED)

// ---- countersteer: its own authority, applied against the slide ---------------
export const NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS = 70; // m/s²  [SIM][REF: VR τ=125 ms]
export const NEATRACING_DRIFT_COUNTERSTEER_MIN_SLIP_RAD = 0.02; // rad (1.15°) [REF: MK64 0.33° deadband]
export const NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD = 0.02; // rad (1.15°) [REF: VR "|v|<=15 => 0"]

// ---- visible pose (second order: the spring-damper) --------------------------
export const NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2 = 324; // rad/s² (ω = 18 rad/s, f = 2.86 Hz)
export const NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S = 12.6; // rad/s  (ζ = 0.35 ⇒ 31 % first overshoot)
export const NEATRACING_DRIFT_POSE_STEER_YAW_RAD = 0.08; // rad (4.6°)
export const NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD = 0.1; // rad (5.7°) — slip at which the term is full

// ---- speed scrub --------------------------------------------------------------
export const NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S = 0.25; // 1/s (down from 0.6) [REF: MK64 2.5x cheaper]

// ---- retained -----------------------------------------------------------------
export const NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD = 0.5; // rad (UNCHANGED — the base gain)
```

**Constants that this model RETIRES** (their jobs are taken over):
`NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S` (4) → becomes `SLIP_RESPONSE_TAU_S` + the
saturation budget; `NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S` (12) → becomes
`COUNTERSTEER_AUTHORITY_MPS`; `NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S`
(0.6) → becomes `SPEED_SCRUB_COEFF_PER_S` (0.25, and now _saturating_ rather than
proportional-to-speed).
`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` changes from `π/2` to `0.45` (it is currently
**unused inside its own module** — `atan2` bounds it by construction — so it is an
envelope pin, not a lever).

**Optional refinement (NOT verified — decide in the RED slice):** a steering-authority fade
so a big slide stays catchable — `NEATRACING_DRIFT_STEER_AUTHORITY_FADE_START_RAD = 0.52`
(30°) and `..._FADE_SPAN_RAD = 0.52` (30°), i.e. zero steering authority at 60° of slip.
Modelled on MKWii's `1 - (angle - 60)/40`.

##### D3 — MEASURED PREDICTIONS of the new model [SIM, my own run, 60 Hz]

Full left steer at t = 1.0 s, full countersteer at t = 3.0 s:

| Metric                           | Current model                    | **New model**                                    |
| -------------------------------- | -------------------------------- | ------------------------------------------------ |
| Entry: 63 % of the hold slide    | ~0.25 s @25 m/s, ~0.5 s @89 m/s  | **0.10 s** @both                                 |
| Entry: 95 % of the hold slide    | ~0.75 s @25 m/s                  | **0.20 s**                                       |
| Hold slide (89 m/s)              | -65.8° and **growing**           | **-25.94° and BOUNDED**                          |
| Hold slide (25 m/s)              | -4.50° (too small to see)        | **-25.83°**                                      |
| Visual proxy at hold             | 0.0392 rad @25 m/s               | **0.2266 rad = 4.6× the 0.0494 baseline**        |
| Countersteer: slip after +0.1 s  | -65.8° → -54.4° (still negative) | **-25.94° → -18.96°**                            |
| Countersteer: slip after +0.2 s  | +10.6° (**mirrored**)            | **-7.01°**                                       |
| Countersteer: slip after +0.3 s  | +28.1° (mirrored)                | **0.00° (hard zero)**                            |
| Visible swing-past               | **none** (0.00°)                 | **-14.02° at +0.1 s = 4.83° past the hold pose** |
| Visible ring after the snap      | none                             | +2.61° → -0.51° → -0.53° → +0.23° → 0 (ζ = 0.35) |
| Residual pose on a straight line | 0 (exact)                        | **-0.07° ≈ 0** (slip-gated steer-yaw)            |
| Speed scrub @89 m/s              | 24.2 m/s² (**> engine 17.8**)    | **10.0 m/s²** (engine still wins)                |

**Variant check — where the swing-past comes from.** With the steer-yaw term removed from
(7) (`poseTarget = slip · GAIN` only) the model produces **0.00°** of swing-past: the
physical recovery is monotone and the visible channel simply follows it. The steer-yaw term
is the swing source; the **slip-gate is what keeps the zero-straight-line guarantee**
(ungated it leaves a -4.58° residual body yaw while driving straight).

##### D4 — Sequencing recommendation for the orchestrator (criterion: fun / exciting)

**Adopt the physical model — but recognize that the already-recorded coupling design's
"Track A" is a strict subset of it.**

- The coupling design's **transient gain** (`offset = slip · (GAIN + transientGain)`) and this
  model's **pose spring-damper** are _the same mechanism_ (a second-order visual channel).
  This model simply **sources it from the physical slip** instead of a synthetic gain, so
  the visual swing and the physical catch become one coherent thing.
- **The visual-only route (Track A) cannot satisfy the user's verdict.** The verdict's
  operative clause is _"bounce-back must be sharper"_, and the bounce-back is **physical**
  (C5). A cosmetic swing on a car that is still sliding the wrong way is not a fix.
- **Track A's gain-down move is still the cheapest first step** (proxy
  `0.0988 × 1.5 = 0.148 rad` = 3.0× baseline, byte-identical trajectory) and is a
  reasonable _interim_ if the budget cannot absorb the re-baselining. But it must be
  recorded as **cosmetic-only**, and it must be driven **down** (0.5 → 0.25 → -0.5) — never
  up, because the proxy is V-shaped with a null at `GAIN = 1.0` (E3).

**⇒ Recommended mix: (1) the physical model as a NEW slice with re-baselining; (2) the pose
spring-damper as the visual half of it (this _is_ the transient the plan already names);
(3) the AC-087 floor re-pinned on the _deterministic_ unit proxy, not the noisy browser
proxy.**

##### D5 — Plan items this research requires to be corrected

1. **The MUST-NOT-MOVE physics pin must be lifted** for the new slice
   (`plans/neatracing-voxel-racer.research.md:341-352`, `plans.md:8233`). A 5× larger slide
   and a latched countersteer are **trajectory** changes ⇒ AC-079 (290 km/h ceiling) and
   AC-089 (no regression) must be **re-baselined**, not preserved.
2. **The AC-087 floor must be pinned in ONE unit system.** The browser proxy (0.0494 rad,
   render-level, render-rate dependent, recorded spread 0.0494–0.3599) and the model
   `slipAngle` (>0.2 rad in the existing unit test) are **different quantities**. Pin the
   floor in the deterministic jest lane; let the fixture carry a looser observable.
3. **AC-088 must be re-scoped to a measurable seam** — the named constants cannot move the
   measured gradient (E5). Narrowest fix: capture the `uTireGradient*` uniforms through the
   fixture's existing `uniform1f` interception hook and assert the factor spread
   `[dark, bright]` is non-degenerate.
4. **The Step 14N packet's file set must widen** to `physics.step.service.ts` (the visual
   yaw is inlined at `:417`; the `CENTRIFUGAL_MULTIPLIER`/`STEERING_RATE` levers live there).
5. **The `computeDriftYawOffset` dead twin** (E5) should be either wired up or deleted in
   the same step, so the tune slice cannot leave two divergent visual-yaw formulas.

##### D6 — THE GAME-FEEL ACCUMULATOR (the spring-damper layer)

_Added by the game-feel scout (2026-10-25). D1–D5 above are UNCHANGED. D6 **AMENDS D1's
stage 7 and D2's constants**; the committed-drift model, the grip budget, the countersteer
authority and the hard zero all stand exactly as written._

**D6.1 — The arcade-refs' spring constants STAY. Do not re-tune them.**
`NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2 = 324` (ω = 18 rad/s = 2.86 Hz) and
`NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S = 12.6` (ζ = 0.35) are **correct as published**.
The asymmetric-damping "fix" for the entry overshoot is **REFUTED** (E6.2): it collapses the
far-side pop 2.61° → 0.00°. **The entry overshoot is the pop's energy reservoir, not a defect.**

**D6.2 — THE CHANGE: replace the RAW articulation term with a SLEW-LIMITED articulation.**

```
artTarget  = steer * speedPercent * min(1, |slip| / ARM_SLIP_RAD)     // ARM = 0.01 rad
art       += clamp(artTarget - art, -ARTICULATION_SLEW_PER_S*dt, +ARTICULATION_SLEW_PER_S*dt)
poseTarget  = slip * VISUAL_YAW_GAIN_RAD + ARTICULATION_RAD * art      // ART = 0.08 (unchanged)
poseRate   += (STIFFNESS * (poseTarget - pose) - DAMPING * poseRate) * dt
pose       += poseRate * dt
```

**Provable per-tick bound:**
`max |ΔposeTarget| ≤ ARTICULATION_RAD · ARTICULATION_SLEW_PER_S · dt + 0.5 · |slip_at_hard_zero|`
— **speed-independent** (the raw term's bound is ∝ `speedPercent`). Measured: **+2.13°/tick
at slew = 20 /s vs the arcade's −4.65°/tick at 49.8 m/s and 9.17°/tick at 89 m/s.**

**D6.3 — THE CONSTANT SET (the concrete accumulator params).**

| Constant                                                        | Value               | Status                      |
| --------------------------------------------------------------- | ------------------- | --------------------------- |
| `NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2`                    | **324** (ω = 18)    | **UNCHANGED** (arcade-refs) |
| `NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S`                       | **12.6** (ζ = 0.35) | **UNCHANGED** (arcade-refs) |
| `NEATRACING_DRIFT_POSE_STEER_YAW_RAD` (articulation amplitude)  | **0.08**            | **UNCHANGED** (arcade-refs) |
| `NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD` (arm threshold) | **0.10 → 0.01**     | **CHANGED**                 |
| `NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S`                 | **20** (band 15–30) | **NEW**                     |

**The slew band** (ω = 18, ζ = 0.35, art 0.08, arm 0.01): slew 10 → pop 1.82° / step 1.47° ·
15 → 2.10 / 1.75 · **20 → 2.22 / 2.13** · 30 → 2.30 / 2.89 · 45 → 2.34 / 3.37 · 60 → 2.33 /
4.06. **The settle is invariant at 550 ms across the whole band** (the ring-out is the
spring's, not the articulation's) and the pop saturates at slew ≥ 30. **The in-band choice is
the RED slice's DECISION, not this scout's.**

**The arm band** (slew 20): arm 0.005 / 0.01 / 0.02 → **identical** (pop 2.22°, step 2.13°);
arm 0.05 → pop 2.43°; arm 0.1 → pop 2.57°. ⇒ **arming earlier is free; the arcade-refs' 0.10
gate defeats the anticipation entirely** (with `ENTRY_RATE 3.5 rad/s` the gate is saturated for
the whole drift, so the arcade-refs' term is a raw-steer OFFSET, not a lead).

**D6.4 — THE ANTICIPATION: use a SLEW, not a τ lag.** A τ lag's per-tick step is
`2 · ART · speedPercent · dt/τ` — **speed- and amplitude-dependent**; the slew's is
`ART · SLEW · dt` — **speed- and amplitude-independent**. The slew therefore keeps the step
bound tight while the articulation amplitude (and hence the pop) stays free. A first-order lag
remains a valid fallback at the cost of a speed-dependent step bound (an open choice, not a
recommendation).

**D6.5 — THE STATE: ONE accumulator, three scalars, on `PhysicsCarState`.** The coupling
design's `DriftTransientGainState` (2 scalars) and the arcade-refs' pose spring (2 scalars) are
**the same mechanism** — with the slew the pose state is **THREE** scalars.
**Ship ONE `DriftPoseSpringState { pose; poseRate; articulation }`** with
`createInitialDriftPoseSpringState()` and `stepDriftPoseSpring(prev, input)`, carried on
`PhysicsCarState` (**NOT** on `DriftModelState` — AC-047D byte-identity) and initialised to
zeros in `createInitialPhysicsCarState`. **Phase B must not ship two competing visual
accumulators.**

**D6.6 — THE TICK: step the spring INSIDE the 60 Hz `stepPhysics` tick.** Never in the render
path (E6.7): a per-frame spring is display-rate dependent (2.4× faster at 144 Hz) and would
break jest/browser parity. The result must be written into the existing single bridge write
(`browser-entry.ts:941`) — **one** scalar, unchanged contract.

**D6.7 — THE RED CONTRACTS (three numbers, not one).** Pin at one
`(ω, ζ, art, arm, slew)` point: **(a) the far-side pop** (≈2.2° ≈ 15 % of the hold δ — a
**named corpus contract with a stated minimum**, so a future "tidy-up" that raises ζ to 0.45 or
ω to 24 cannot silently halve it), **(b) the settle time** (≈550 ms), **(c) the max per-tick
pose-TARGET step** (≤ 2.13°). Plus **the zero-straight-line-yaw guarantee as an EXACT
identity (to 1e-9)**, not a tolerance (E6.9). Pinning only the hold pose would let a future
edit destroy the pop silently.

**D6.8 — THE ENVELOPE MUST BE RE-DERIVED FROM THE MEASURED PEAK.** `DRIFT_VISUAL_YAW_LIMIT_RAD`
= `MAX_SLIP · GAIN` = 0.2250 rad is exceeded by **2.00 %** by the arcade-refs' own entry
overshoot (E6.6). Either the limit moves to the measured peak (0.2295 rad) or the pose stays
inside it (the recommended variant: 0.2246 rad). **The existing pin at
`physics.step.service.test.ts:156-163` (`|driftYawOffset| < π/2`, spawn offset `=== 0`) must
be re-verified against the overshoot.** **ω = 18 is the largest ω that keeps the pose inside
the declared envelope at ζ = 0.35** — a second, independent reason to keep it.

**D6.9 — THE MEASURED PREDICTIONS** (for comparison with D3's arcade-refs column):

| Metric                                  | Arcade-refs (D3)                      | Recommended (D6)                        |
| --------------------------------------- | ------------------------------------- | --------------------------------------- |
| Hold slip / pose / δ                    | −25.39° / −10.16° / 15.23°            | **identical** (the spring is unchanged) |
| Entry overshoot                         | 17.63 % (1.79°)                       | 18.35 % (1.86°) — **kept deliberately** |
| Catch dive                              | −3.67°                                | −4.40°                                  |
| **Far-side pop**                        | **2.61° (17 %)**                      | **2.22° (15 %) — the pop is PRESERVED** |
| Settle (\|δ\| < 0.5°)                   | 550 ms                                | 550 ms                                  |
| Peak pose rate                          | 108 °/s                               | 92 °/s                                  |
| Onset to 25 % / 50 % of δ               | 33.3 / 66.7 ms                        | **33.3 / 66.7 ms (unchanged)**          |
| **Max pose-target step per tick**       | **−4.65° (9.17° @ 89 m/s)**           | **+2.13°, speed-independent**           |
| Max \|pose\| vs the 0.2250 rad envelope | **13.15° = 0.2295 rad — 2.00 % OVER** | **12.87° = 0.2246 rad — 0.19 % inside** |
| Residual pose at rest                   | −0.000°                               | −0.000°                                 |
| Zero-straight-line max \|pose\|         | 0 (exact)                             | **0.000000000 rad (exact)**             |

**D6.10 — WHAT THIS RESEARCH REQUIRES OF PHASE B.** (1) Record the asymmetric-damping idea as
**REFUTED** so it is not re-derived. (2) **AMEND** the arcade-refs D1 stage 7 + D2 constants;
the rest of the arcade-refs model stands. (3) **MERGE** the two visual channels into ONE
accumulator (D6.5). (4) The RED slice pins **three** numbers + the exact identity (D6.7).
(5) Re-derive the yaw limit (D6.8). (6) The pop is a **corpus contract** (D6.7a).

---

#### Risks

| #       | Risk                                                                                                                                                                                                                                                                                                                     | Severity | Evidence             | Mitigation                                                                                                                                                                                        |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **R1**  | **The physical model changes the trajectory.** A 5× larger slide at the 90 km/h operating point moves the car's path (a 25.8° slip at 25 m/s is ~12 m/s of lateral velocity). AC-079's 290 km/h ceiling, AC-089's no-regression pin, and the 3-frame `DRIFT_PROXY` gate must be **re-measured**, not preserved.          | **HIGH** | D3, C4               | Re-baseline explicitly in the new slice; sequence after the cosmetic step so there is a working baseline to fall back to.                                                                         |
| **R2**  | **The countersteer authority may be too strong at low speed** (70 m/s² × speedPercent = 19.7 m/s² at 25 m/s) and could feel twitchy below ~40 km/h.                                                                                                                                                                      | MEDIUM   | D3                   | Gate the authority on `                                                                                                                                                                           | slip | > COUNTERSTEER_MIN_SLIP_RAD`(already) and consider a`speedPercent` floor; measure in the RED lane at 25 m/s. |
| **R3**  | **`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` changes `π/2 → 0.45`**, which moves `car.pass.test.ts`'s derived `DRIFT_VISUAL_YAW_LIMIT_RAD` (= `MAX_SLIP · GAIN`, 0.7854 → 0.225) and its monotonicity-sweep endpoints. The test re-derives it, so it self-syncs — but the **envelope value moves by 3.5×**.                   | MEDIUM   | E5, D2               | Re-derive and re-baseline in the same slice; verify the sweep endpoints still bracket the new envelope.                                                                                           |
| **R4**  | **The pose state is a new state type.** It must NOT be added to `DriftModelState` if that type is byte-identity pinned (AC-047D) — a **separate** `DriftPoseState` is required, exactly as the coupling design's D2 already decided for the transient gain.                                                              | MEDIUM   | E5, `research.md` D2 | New separate type + a new numeric field on `PhysicsCarState`; check the exact AC-047D assertion form in the RED slice.                                                                            |
| **R5**  | **The steering-authority fade is unverified** and could make the car un-controllable if the numbers are wrong.                                                                                                                                                                                                           | LOW      | D2                   | Ship it disabled; enable only after a measured pass.                                                                                                                                              |
| **R6**  | **The proxy null at `GAIN = 1.0`** is a live trap for any visual-gain work in parallel (the coupling design's Track A). If Track A and this model are both in flight, they must not both touch the gain.                                                                                                                 | MEDIUM   | E3                   | One owner for the visual channel.                                                                                                                                                                 |
| **R7**  | **`collision.utils.ts:410,417-418` zeroes `speed`/`slipAngle`/`lateralVelocity` on a bounce** but not the drift yaw offset ⇒ a wall impact kills the slide and the pose would ring on with no slip to follow it.                                                                                                         | LOW      | E5                   | With the slip-proportional pose the offset self-heals to 0; still add a RED note.                                                                                                                 |
| **R8**  | **The glass canopy does not follow the drift yaw** (`glass.pass.ts:1048,1086` never reads `driftYawOffsetRad`) ⇒ a bigger slide **amplifies** the visible body-vs-canopy mismatch.                                                                                                                                       | MEDIUM   | `research.md` R7     | Out of scope; record as a carried-forward residual for a later step.                                                                                                                              |
| **R9**  | **The upstream arcade-research draft's model had a sign error in the visible-pose target** (`-committed` under the production slip convention).                                                                                                                                                                          | —        | D1                   | Corrected here; the implementer must use **D1 as written**, not the upstream draft.                                                                                                               |
| **R10** | **This research was produced under a paused/budget-constrained directive** (2026-10-24). The gaps in E4 (OutRun 2, Ridge Racer 4/7, Daytona USA's own ROM, the Mario Kart per-character tables) are **flagged, not filled** — nothing was invented to cover them.                                                        | LOW      | E4                   | The gaps do not affect the constant set: every number in D2 is anchored to a game that _does_ publish the value.                                                                                  |
| **R11** | **A future "tidy-up" of the pose spring can silently destroy the arcade feel.** The pop and the settle are one dial (E6.5): raising ζ 0.35 → 0.45 halves the pop (2.22° → 1.35°), and raising ω 18 → 24 collapses it further (1.37°). The entry overshoot that looks like a defect is the pop's energy reservoir (E6.2). | **HIGH** | D6.1, D6.7           | Pin the pop, the settle AND the max per-tick target step in the RED slice as **named contracts with stated minimums**; record the asymmetric-damping idea as **REFUTED** so it is not re-derived. |
| **R12** | **The declared `DRIFT_VISUAL_YAW_LIMIT_RAD` (0.2250 rad) is exceeded by 2.00 %** by the arcade-refs' own entry overshoot (0.2295 rad) — the envelope is derived from `MAX_SLIP · GAIN` rather than from the measured peak.                                                                                               | MEDIUM   | E6.6, D6.8           | Re-derive the limit from the measured peak (or keep the pose inside it via the slew), and re-verify `physics.step.service.test.ts:156-163` against the overshoot.                                 |
| **R13** | **The 4 Hz steer-chatter stress** drives the far-side body swing to 22.6° and the settle to ~2 s. The slew bounds the per-tick target step but does **not** suppress the committed model's chatter response ⇒ this is a **D1 latch** property, not a pose-channel defect.                                                | MEDIUM   | E6.10                | Out of the game-feel charter's scope; raised as a risk on the latch for Phase B to consider. Not reproduced against production.                                                                   |
| **R14** | **Two competing visual accumulators** (the coupling design's `DriftTransientGainState` and the arcade-refs' pose spring) are the SAME mechanism; shipping both would double the visual gain and fight the single bridge write.                                                                                           | **HIGH** | D6.5                 | Merge into ONE `DriftPoseSpringState` (three scalars) on `PhysicsCarState`; one owner for the visual channel (extends R6).                                                                        |
| **R15** | **The spring's perceived attack is NOT the spring's** (E6.4): the onset is identical across every variant, so a Phase-B author who assumes "crisper = raise ω" would silently trade the pop away for no perceived gain.                                                                                                  | MEDIUM   | E6.4, E6.5           | State the attack's real source (the committed-angle ramp + the slip chase) in the RED slice's comments; raise ω only for the pop, within the R11 guardrails.                                      |

---

#### Certainty statement

| Claim                                                                                   | Confidence                                                                                               | Basis                                                                                                                                                                                 |
| --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C1** — the production centrifugal multiplier is 1.5, not 0.3                          | **0.99**                                                                                                 | direct source read, `physics.step.service.ts:184`                                                                                                                                     |
| **E2 / C4** — the current slide is unbounded at speed                                   | **0.97**                                                                                                 | my own 60 Hz simulation of the reproduced equations, 89 m/s and 25 m/s                                                                                                                |
| **E2 / C5** — the countersteer mirrors the slide instead of catching it                 | **0.97**                                                                                                 | my own simulation (+15.4° / +52.2° inversions) **and** an independent source-level reading of the two causes (yaw-derived push, sign-crossing gate)                                   |
| **E3** — the proxy identity and the null at `GAIN = 1.0`                                | **0.95**                                                                                                 | confirmed to the last digit against the recorded baseline, and independently by two repo Tier-3 scouts                                                                                |
| **E4** — the per-game five-phase table                                                  | **0.88** for the `[SRC]` rows, **0.55** for the `[CONSENSUS]` rows, **0.00** for the `[NOT FOUND]` cells | ROM decompilations / reimplementation sources (see the source index below)                                                                                                            |
| **D2** — the constant set                                                               | **0.86**                                                                                                 | every value is either verified by my own simulation or anchored to a published game number; the _feel_ choice inside the band remains a RED-slice decision                            |
| **D3** — the measured predictions                                                       | **0.93**                                                                                                 | my own simulation; the model is deterministic and the harness is exact, so the residual uncertainty is the production integration order (tick-start vs tick-end speed), not the model |
| **D4** — the sequencing recommendation                                                  | **0.80**                                                                                                 | a judgement call about plan shape, not a measurement                                                                                                                                  |
| **R1** — the trajectory risk                                                            | **0.85**                                                                                                 | the lateral velocity change is arithmetic; the _contract_ consequences depend on the re-baselining that has not happened yet                                                          |
| **D6.1** — the arcade-refs' `K = 324` / `C = 12.6` are correct and must not be re-tuned | **0.97**                                                                                                 | measured; the asymmetric-damping alternative collapses the pop 2.61° → 0.00°, and the harness is validated against this artifact's own D3 numbers to the digit                        |
| **D6.2** — the slew-limited articulation and its per-tick bound                         | **0.93**                                                                                                 | measured (max step 2.13°/tick, speed-independent) and the bound is derived, not fitted                                                                                                |
| **D6.3** — the constant set                                                             | **0.90**                                                                                                 | every value is measured in the harness; the _in-band_ choice (slew 15–30) remains a RED-slice decision                                                                                |
| **D6.5** — the state-ownership merge (one accumulator, three scalars)                   | **0.80**                                                                                                 | a judgement about plan shape, not a measurement; the mechanics are source-verified                                                                                                    |
| **E6.4** — the attack rate is the model's, not the spring's                             | **0.93**                                                                                                 | measured across every variant: the onset is identical to the millisecond                                                                                                              |
| **E6.6** — the 2.00 % envelope overshoot                                                | **0.90**                                                                                                 | measured; the declared limit is derived from `MAX_SLIP · GAIN`, and the overshoot is a direct consequence of the (deliberate) entry overshoot                                         |
| **E6.7** — tick placement and the camera's non-consumption of the offset                | **0.92**                                                                                                 | source-verified by the `boundary-mapper` Tier-3 scout (two read sites, one bridge write, no interpolation, camera yaw trajectory-only)                                                |
| **E6.8** — no second-order precedent in the repo                                        | **0.95**                                                                                                 | source-verified by the `implementation-pattern-scout` Tier-3 scout; the house idiom is first-order TAU                                                                                |
| **E6.9** — the exact zero-straight-line identity and the stability margins              | **0.95**                                                                                                 | measured (0.000000000 rad over 5 s); stability is arithmetic (ω·dt ≤ 0.533 « 2)                                                                                                       |
| **E6.10** — the 4 Hz chatter stress                                                     | **0.85**                                                                                                 | a model-level observation on my reproduction, deliberately reported as a risk, not a defect claim                                                                                     |

##### Source index

**Mario Kart Wii** — `https://github.com/vabold/Kinoko` → `source/game/kart/KartMove.cc`, `KartMove.hh`, `KartParam.hh`
**Mario Kart DS** — `https://github.com/RonyxDumb/mkds` → `src/arm9/0206c5e4_func_0206c5e4.c`, `0206cc60_func_0206cc60.c`, `0206cdd4_DriverContext_ApplyDriftBoost.c`, `0206ceb4_func_0206ceb4.c`; `https://github.com/saavor/mkjs` → `code/entities/kart.js`, `code/formats/kartphysicalparam.js`
**Super Mario Kart** — `https://github.com/carlosbravoa/smk-native` → `docs/NOTES.md`, `src/player.c`, `include/smk.h`
**Mario Kart 64** — `https://github.com/n64decomp/mk64` → `src/player_controller.c`, `src/data/kart_attributes.c`, `include/defines.h`
**OutRun 2** — `https://segaretro.org/OutRun_2` · `https://segaretro.org/images/8/8c/OutRun2_Chihiro_US_Manual_Standard.pdf` · `https://github.com/CrownParkComputing/OutRun2-XBERecomp/blob/main/PORT_STATUS.md`
**Ridge Racer (PS1)** — `https://github.com/kazuyette/rr-decomp/blob/main/src/fn/func_80026E7C.c` · `https://github.com/kazuyette/rr-pc-port/blob/main/src/physics_psx.c` · `http://www.rr-ken.net/RR/r-project/sd.html`
**Ridge Racer Type 4** — `https://ridgeracer.fandom.com/wiki/Drifting` · `https://archive.org/download/r4_ridge_racer_type_4_english/r4_ridge_racer_type_4_english_djvu.txt`
**Ridge Racer 7** — `http://www.rr-ken.net/RR7/system.html` · `https://ridgeracer.fandom.com/wiki/Nitrous`
**Daytona USA / Sega Rally '95** — `https://github.com/xandoxan65/segarally95-recomp/blob/main/src/game/game_start_race_obj_car_{velocity_step.c,frame.c,yaw_step.c,motion.c}` · `https://github.com/alphanu1/daytona-arcade-recomp/blob/main/docs/daytona-usa-recomp-design.md`
**Virtua Racing** — `https://github.com/matiaszanolli/sega-vr-disasm/blob/master/disasm/modules/68k/game/physics/` · `.../analysis/PHYSICS_SYSTEM_ARCHITECTURE.md`
**TrackMania** — `https://github.com/Teero888/tmuf_physics` → `src/reference/car_models.c`, `src/reference/car_forces.c`, `src/reference/car.c`, `src/common/vehicle_tuning.c`, `include/tmuf_physics/state.h`
**This repo (read-only)** — `examples/neatRacing/physics/drift.model.ts:44-121,140-161,179-201,205-226,245-262,280-292,305-330,337-343,360-370` · `examples/neatRacing/physics/physics.step.service.ts:134,140,184,193,373,388,396,406,412,417,473-479` · `examples/neatRacing/physics/drift.model.test.ts` · `examples/neatRacing/engine/passes/car.pass.ts:86,388,391,398,438,559-561,1365,1469,1567` · `examples/neatRacing/engine/passes/glass.pass.ts:1048,1086` · `examples/neatRacing/physics/collision.utils.ts:384-386,410-418` · `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:107-146` · `plans/neatracing-voxel-racer.research.md:66-89,341-352` · `plans/neatracing-voxel-racer.plans.md:2345,8124,8225-8238`

**Game-feel addendum — this repo (read-only)** — `examples/neatRacing/physics/drift.model.ts:363` (the dead `computeDriftYawOffset` export) · `examples/neatRacing/host/render-loop.utils.ts:27,41` (`NEATRACING_FIXED_TIMESTEP_MS = 1000/60`, `NEATRACING_MAX_CATCHUP_STEPS_PER_FRAME = 4`) · `examples/neatRacing/browser-entry/browser-entry.ts:941,955` (the single bridge write; the camera yaw does NOT include the offset) · `examples/neatRacing/editor/editor.camera.ts:48-56,430-447` (the first-order TAU smoothing precedent — the house idiom) · `examples/neatRacing/physics/physics.step.service.test.ts:156-163` (the `|driftYawOffset| < π/2` and spawn-zero pins) · `examples/neatRacing/engine/passes/car.pass.ts:1365,1565-1573,1598` (the two read sites; the tires inherit the body matrix) · `examples/neatRacing/engine/passes/glass.pass.ts:1048,1086` (the canopy never reads the offset)

**Game-feel addendum — the measurement harness** — `tmp/driftfeel-sim2.mjs` … `tmp/driftfeel-final.mjs` (scratch, **gitignored**, not part of the repo): the 60 Hz reproduction of the D1 model, validated against this artifact's own published D3 numbers (entry overshoot 17.6 %, dive −3.67°, pop +2.61°, settle 550 ms, max step −4.65°, peak rate 108 °/s) **to the digit** before any new figure was trusted.

**Files changed by this research pass:** none in `src/` or `examples/`. This artifact and the
plan append are the only writes.
