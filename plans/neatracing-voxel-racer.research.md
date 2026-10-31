# Research artifact — Step 14N drift-feel integration coupling design

- **Plan**: `plans/neatracing-voxel-racer.plans.md` (Step 14N, `[WIP]`)
- **Step**: 14N — Arcade drift FEEL sharpening
- **Author role**: 02-researching (Tier-1 drift-feel scout), read-only research
- **Date**: 2026-10-24 (Step 14N planning round)
- **Sub-orchestrators used**: `boundary-mapper`, `implementation-pattern-scout` (both read-only, both re-verified by the orchestrator against source)
- **Status**: RESEARCH COMPLETE — the feel direction, the floor value, and the AC-088 lever remain RED-slice decisions (deliberately NOT invented here)

---

## Question

Where should a **transient drift-yaw accumulator** live, and how must it feed the car body
matrix so that the Step 14N work does not break:

1. **AC-078** — the tire-follow pin (`TIRE_YAW_FOLLOW_TOLERANCE_RAD = 0.002` rad),
2. **AC-079** — `proportionalTractionLoss` (the 290 km/h ceiling),
3. the **DRIFT_PROXY** fixture gates (`DRIFT_PROXY_MIN_RAD = 0.01`, 3 consecutive frames, 6-frame window),
4. **zero straight-line yaw**.

Plus: state ownership, tick order, and which constants move.

---

## Evidence

### E1 — The drift-yaw production chain (verified in source)

| Stage | Location | Behaviour |
|---|---|---|
| Constant | `examples/neatRacing/physics/drift.model.ts:97` | `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD = 0.5` |
| Producer | `examples/neatRacing/physics/physics.step.service.ts:416-417` | `nextDriftYawOffset = nextSlipAngle * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` |
| Return | `physics.step.service.ts:440` | returned from `stepPhysics` (pipeline at `:365-450`) |
| Field | `physics.step.service.ts:108` | `PhysicsCarState.driftYawOffset` (spawn init `0` at `:312`) |
| Transport | `examples/neatRacing/browser-entry/browser-entry.ts:941` | `carPoseBag.driftYawOffsetRad = car.driftYawOffset` — the **only** production read of the field |
| Shadow | `examples/neatRacing/engine/passes/car.pass.ts:1365` | Euler sum `car.yawRad + (car.driftYawOffsetRad ?? 0)` |
| Body | `car.pass.ts:1565-1573` | matrix product `T · Ry(yaw) · Ry(drift)` |
| Wheels | `car.pass.ts:1598` | wheel chain parented onto `NEATRACING_CAR_BODY_MODEL_MATRIX` |

`drift.model.ts:363` `computeDriftYawOffset()` is a **dead export** — production inlines the
multiplication, so the "shape" of the body animation is literally `slipAngle × gain`.

Because `Ry(yaw)` and `Ry(drift)` share the same world-up axis, `Ry(a)·Ry(b) ≡ Ry(a+b)`; the
shadow's Euler sum and the body's matrix product agree **exactly** today. Any change that keeps
a single scalar offset preserves that identity.

`examples/neatRacing/engine/passes/glass.pass.ts:1048,1086` reads `car.yawRad` /
`steerAngleRad` but **not** `driftYawOffsetRad` — the glass canopy stays on base yaw. This is a
pre-existing inconsistency, but feel work amplifies it (the body will visibly yaw away from the
canopy). Flagged, not owned by Step 14N.

### E2 — The DRIFT_PROXY is measured from the **rendered body matrix** vs the travel chord

`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:789-822`:

- `bodyYawRad = yawFromMatrixColumn3(m)` — the yaw column of the **rendered car body model matrix**;
- the window's oldest/newest positions give a world-XZ **chord** direction;
- `proxyRad = |wrapAngle(endpointMidYawRad - offsetAngle(chordDx, chordDz))|`.

The endpoint-anchored window mean cancels the mid-window lag exactly; the residual is bounded by
`alpha·T²/8` plus a speed-variation term ≤ 0.002 rad. `maxDriftProxyRad` is the **max** over the
run; `driftProxyRunFrames` counts **consecutive** frames ≥ `DRIFT_PROXY_MIN_RAD = 0.01` and
resets to 0 on any frame below it.

### E3 — DECISIVE: the drift proxy is a **V-shaped** function of the visual gain

The physics translates the car along `heading + slipAngle` (`physics.step.service.ts`, step 4c);
the rendered body yaw is `heading + driftYawOffset`. Therefore

```
proxy  = |bodyYaw - travelDirection|
       = |(heading + slip*GAIN_eff) - (heading + slip)|
       = |slipAngle| * |GAIN_eff - 1|
```

The recorded Step 14G baseline is `0.049388550574608114 rad` at `GAIN = 0.5`, which implies
`|slipAngle| = 0.049388550574608114 / 0.5 = 0.098777101149216228 rad` at that operating point.
The baseline is *exactly* `0.5 × |slip|` — the identity above is confirmed against measured data.

| `GAIN_eff` | proxy (rad, at `\|slip\| = 0.09878`) | meaning |
|---|---|---|
| 0.00 | 0.0988 | body on the heading — **no slide at all** |
| **0.50 (today)** | **0.0494** | body halfway to the travel direction |
| 0.75 | 0.0247 | slide *shrinks* |
| **1.00** | **0.0000** | **NULL — the slide becomes invisible** |
| 1.25 | 0.0247 | slide grows again, on the other side |
| 1.50 | 0.0494 | equals today's baseline |
| 2.00 | 0.0988 | 2× baseline |

**Consequences the RED slice must internalise before pinning anything:**

- The proxy is **non-monotonic** in the gain with a **null at `GAIN_eff = +1.0`**. A naive
  "raise the gain to feel more drift" move to 1.0–1.25 makes the slide *weaker*, and at exactly
  1.0 the slide vanishes. Any floor pinned against the gain must record the forbidden null.
- Raising the gain **above** 1.0 is a *different visual branch* from the current one: the body
  rotates past the travel direction instead of sitting short of it.
- The current `GAIN = 0.5` means the body yaw sits at only **half** the physical slip. The crisp
  arcade-drift look (nose further into the turn than the path) is the branch where the body is
  *further* from the travel direction than the heading is — i.e. an **effective gain below the
  current 0.5** (or negative), which is the direction that **increases** the proxy monotonically
  and never crosses the `+1.0` null.

### E4 — State ownership and tick order

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

### E5 — The constraints that pin the design (all verified)

| # | Constraint | Location | Effect on the design |
|---|---|---|---|
| C1 | `spawn.driftYawOffset === 0` | `physics.step.service.test.ts:156-163` | the offset must start at 0 |
| C2 | `\|driftYawOffset\| < Math.PI / 2` (and finite, and `> 0` after 60 ticks) | `physics.step.service.test.ts:162` | the **total** offset must stay under π/2 — this is the hard bound on any gain/accumulator magnitude |
| C3 | determinism = run A vs run B of the **same** code (never a golden literal) | `physics.step.service.test.ts:130-137` | adding a field to `PhysicsCarState` is safe |
| C4 | `serializeDriftState(state) === serializeDriftState(createZeroDriftState())` after 120 straight-line steps | `drift.model.test.ts:309-327` (AC-047D) | `DriftModelState` has exactly **two** fields (`lateralVelocity`, `slipAngle`); **adding a field to it touches this byte-identity pin** |
| C5 | `computeLongitudinalTractionLoss(0, 60, 0.67) === 0` exactly | `drift.model.test.ts:176-183` | straight-line traction must stay exactly 0 |
| C6 | 600 straight-line full-throttle steps → `speed === MAX` and `slipAngle === 0` | `physics.step.service.test.ts:247-252` | the trajectory must not change |
| C7 | `DRIFT_VISUAL_YAW_LIMIT_RAD = NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` (= π/2 × 0.5 ≈ 0.785) and the monotonic sweep uses `[-L, -L/2, 0, L/2, L]` | `car.pass.test.ts:371-373, 479-516` | the limit is **derived from the gain** and auto-scales; the sweep asserts the *pass* is a faithful `Ry(drift)`, so it constrains the pass, not the physics |
| C8 | `tireYawDriftEngaged` needs ≥ 3 **consecutive** frames with proxy ≥ 0.01 | fixture `:876-881` | a transient that drives the proxy below 0.01 for even one frame **resets** the run counter and can break **both** AC-078 and AC-079 |
| C9 | `tireYawFollowsBodyDuringDrift` = `maxRelativeYawError ≤ 0.002 && maxRelativeOffsetError ≤ 0.002` | fixture `:926-928` | the 0.002 rad pin is a **relative** measure (`wheelYaw − bodyYaw == mount.rotationYawRad`) — **invariant to the offset's magnitude** |
| C10 | `proportionalTractionLoss` = `tireYawDriftEngaged && 0 < maxSpeedKmhDrift < 290` | fixture `:935-937` | a **ceiling only**, no ratio; satisfied iff the drift stays engaged and the speed stays under 290 |
| C11 | `NEATRACING_WHEEL_HUB_HEIGHT_M = NEATRACING_WHEEL_WORLD_RADIUS_M = 0.5625 m` | `voxel/wheel.mount.utils.ts:86` | the tire-gradient radius is currently the **hub height**, not a dedicated constant |

**There is NO assertion anywhere that the body yaw/heading is zero on a straight line.** The
"zero straight-line yaw" invariant is currently unenforced and must be closed by a new RED
contract.

### E6 — DECISIVE: the AC-088 tire-gradient measurement is **blind to the shader**

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
(`engine/lighting.ts:62, 75-81`). The shader's gradient axis is therefore *almost* the local Y
axis that the fixture splits on, so the measurement **axis** is roughly right; it is the
**sampling** (albedo instead of shaded output) that is wrong.

**AC-088 as currently worded is therefore not closable by tuning the three named constants.** The
RED contract must either (a) sample the shaded framebuffer instead of the VBO, or (b) be
re-scoped to the jest-level contract that already exists in
`examples/neatRacing/engine/lighting.test.ts` (`describe('fixed-space tire lighting (OBS-20)')`),
which owns the shader-source / uniform-feed contract. The jest owner of the tire gradient is
`engine/lighting.test.ts`, **not** `car.pass.test.ts` — the Step 14N RED packet names the latter.

### E7 — Secondary risks (verified, dormant)

- `examples/neatRacing/physics/collision.utils.ts:410,417-418` zeroes `speed`, `slipAngle` and
  `lateralVelocity` on a bounce but **not** `driftYawOffset` — dormant today, latent the moment a
  transient is wired.
- Spec drift is already live: the spec's illustrative formula is
  `clamp(lateralVelocity/speed · GAIN, ±max)` while the implementation is `atan2(...) · GAIN`
  with **no clamp**.

---

## Decision

### D1 — The model: a **transient gain**, not an additive yaw offset

> **The accumulator is a transient, bounded, dimensionless *gain modulation* on the drift yaw,
> and the drift-yaw offset stays exactly proportional to the slip angle.**

```
offset  = slipAngle * (NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD + transientGain)
proxy   = |slipAngle| * |GAIN_eff - 1|,   GAIN_eff = GAIN + transientGain
```

with `transientGain` integrated per tick by a **pure** step function and carried as its own
small state.

**Why this shape and not `offset = slip*GAIN + accumulatedYaw`:**

1. **Zero straight-line yaw is exact by construction.** With slip ≡ 0 the offset ≡ 0 *whatever*
   the accumulator holds. An additive accumulator would carry a **stale yaw** onto the next
   straight line and would need an explicit reset contract (and a hard-reset rule) to satisfy
   the user's zero-straight-line-yaw requirement.
2. **The "body-animation shape" is retained**, as AC-087 demands: the offset is still
   `slipAngle × gain`; only the *gain* becomes transient. `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`
   keeps owning the body yaw (it becomes the *base* gain of a transient gain).
3. **The proxy stays monotone in the chosen direction.** Driving `GAIN_eff` **down** (0.5 → 0.25
   → −0.5) increases `|GAIN_eff − 1|` monotonically and **never crosses the `+1.0` null**, so
   the 3-consecutive-frame drift gate (C8) and the `tireYawDriftEngaged` precondition of AC-078
   and AC-079 are never endangered. Driving the gain *up* toward 1.0 would approach the null and
   would break C8 at the worst possible moment.
4. **The accumulator is visual-only**: it never touches `x`, `y`, `heading`, `speed`,
   `slipAngle` or `lateralVelocity`. The physics trajectory is therefore byte-identical
   (C5, C6, AC-089) and the `proportionalTractionLoss` 290 km/h ceiling (C10) is untouched.
5. **The pass is not re-authored** (AC-087), so the body matrix composition, the shadow Euler sum
   and the wheel parenting are structurally unchanged.

The **alternative** (additive yaw accumulator) is documented here for completeness but is
**not recommended**: it (a) breaks the zero-straight-line-yaw guarantee unless hard-reset,
(b) has a *null* in the middle of its range — an additive offset equal to `slip·(1 − GAIN)`
puts the body exactly on the travel direction and the proxy collapses to 0 for a frame or two,
resetting the 3-consecutive counter, and (c) decouples the offset from the slip, so the
`car.pass.test.ts` monotonicity contract (C7) would have to be re-authored against a quantity
that is no longer slip-derived.

### D2 — Where the accumulator lives

| Element | Home | Rationale |
|---|---|---|
| The pure step function + its constants | `examples/neatRacing/physics/drift.model.ts` (new export) | pure, side-effect free, no `Math.random` (AC-089); unit-testable in isolation; the drift model is already the owner of the drift *shape* |
| The accumulator **state** | a **new, separate** state type in `drift.model.ts` (e.g. `DriftTransientGainState { transientGain: number }`) — **NOT** a field of `DriftModelState` | C4: `DriftModelState` is byte-identity pinned by AC-047D after 120 straight-line steps; a separate type leaves that pin **untouched** |
| The accumulator's carrier on the car | a new numeric field on `PhysicsCarState` (`physics.step.service.ts:73-112`), init `0` in `createInitialPhysicsCarState` | the physics state is the only per-car, per-tick, JSON-serializable, deterministic carrier (C3); the session replaces the record wholesale each tick, so the field must be carried forward |
| The step call site | `physics.step.service.ts` inside the drift step (next to `:416-417`) | the accumulator must advance in **tick** order, with the same `dt = 1/60` as the rest of the pipeline |
| The render side | **nothing** | `car.pass.ts` and `browser-entry.ts` are untouched: the accumulator's effect is already inside the single `driftYawOffset` scalar they consume |

### D3 — How it feeds the body matrix

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
`wheelOffset − bodyOffset` against `mount.rotationYawRad`. Because the wheels are *parented onto*
the body matrix, the wheel yaw is `bodyYaw + mount.rotationYawRad` for **any** body yaw, and the
wheel offset is likewise identical. The residual is a floating-point difference, independent of
the offset's magnitude. **Enlarging the drift cannot move the 0.002 rad residual** — the only
way it could move is if the offset grew so large that the `Ry` composition lost float precision
(irrelevant at ≤ 0.2 rad), or if the *pass* stopped composing `Ry(drift)` as a matrix product.

The `car.pass.test.ts` monotonicity contract (C7) also survives: the limit is **derived** as
`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`, so it auto-scales
with the base gain, and the sweep passes explicit offsets to the pass — it constrains the *pass*,
not the physics.

### D4 — State ownership

| State | Owner | Lifetime | Notes |
|---|---|---|---|
| `transientGain` | `PhysicsCarState` (new field) | per car, per tick | replaced wholesale each tick; JSON-serializable; carried by `race-session.ts:81-89` |
| `DriftTransientGainState` | `drift.model.ts` (new type) | per car | **separate** from `DriftModelState` (C4) |
| `driftYawOffset` | `PhysicsCarState` (existing, `:108`) | per car, per tick | unchanged field; the accumulator is folded into its value |
| `carPoseBag.driftYawOffsetRad` | `browser-entry.ts:941` | per rendered frame | unchanged; the render never sees the accumulator |

### D5 — Tick order

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

### D6 — Constants that move

**Move (the levers):**

| Constant | Current | Home | Role |
|---|---|---|---|
| `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` | `0.5` | `drift.model.ts:97` | the **base gain**; the magnitude lever. **Its relation to the AC-087 proxy is V-shaped with a null at `1.0`** (E3) — the RED slice must record the null as a forbidden region |
| `NEATRACING_DRIFT_TRANSIENT_GAIN_MAX` | new | `drift.model.ts` | magnitude bound of the transient. **Must be chosen so that `\|GAIN_eff\| · \|slipAngle\| < π/2`** (C2). With `\|slip\| ≤ π/2` in the worst case this argues for `\|GAIN_eff\| ≤ 1`; at the measured operating point (`\|slip\| ≈ 0.0988`) the headroom is far larger |
| `NEATRACING_DRIFT_TRANSIENT_GAIN_RATE_PER_S` | new | `drift.model.ts` | how fast the slide **builds** (the "sustain" half of AC-087) |
| `NEATRACING_DRIFT_TRANSIENT_GAIN_DECAY_PER_S` | new | `drift.model.ts` | how fast the slide **unwinds** on drift exit (the "crisper recovery" half). **These two trade off** — a slow decay sustains the slide but softens the recovery; this is the concrete form of the feel-direction decision |
| `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD` | new (AC-087 floor) | `drift.model.ts` (exported) | the AC-087 floor. **Coupling note:** the browser fixture hardcodes its own constants (`DRIFT_PROXY_MIN_RAD = 0.01` at fixture `:128`), so the floor needs a single source of truth *and* a mirrored literal in the fixture; the plan should name both |
| `NEATRACING_TIRE_GRADIENT_RADIUS_M` | new (candidate) | `car.pass.ts` | AC-088 names the gradient radius, but today `uTireGradientRadius` is fed `NEATRACING_WHEEL_HUB_HEIGHT_M` (`car.pass.ts:1469`) — a **wheel-mount** constant, not a lighting constant. Decoupling it into a lighting-owned named constant is correct hygiene — **but see E6: it cannot by itself close the measured residual** |

**Must NOT move (physics constants — moving them changes the trajectory and breaks AC-079/AC-089):**

`NEATRACING_DRIFT_MIN_SPEED_MPS = 15`, `NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD = 0.25`,
`NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD = 2`,
`NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S = 0.6`,
`NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S = 12`,
`NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S = 4`, `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = π/2`.

**Moves as a side effect (derived, auto-consistent):**
`car.pass.test.ts`'s `DRIFT_VISUAL_YAW_LIMIT_RAD = NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`
— it re-derives itself from the base gain, so the monotonicity sweep stays green.

### D7 — What the RED slice `P1-drift-feel-red` must author (recommended contracts)

1. **AC-087 floor.** Pin `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD` **against the measured baseline
   `0.049388550574608114 rad`** and record the *direction* (bigger slide vs. crisper recovery) plus
   the chosen named lever, per the plan. Record the **`GAIN_eff = +1.0` null** as a forbidden
   region, and state whether the lever drives `GAIN_eff` down (recommended: monotone, null-free)
   or up (crosses the null).
2. **Zero straight-line yaw — a NEW contract.** No assertion exists today (E5). It must assert
   that the **body yaw** (not just the slip) is zero/unchanged on a straight line, and — because
   the recommended model is slip-proportional — that the offset is *exactly* 0 whenever
   `slipAngle === 0`. Under the additive alternative this contract would need a hard-reset rule.
3. **Drift-exit reset / recovery.** Assert the transient returns to its rest value when the drift
   ends, and that the recovery time matches the recorded decay constant.
4. **The proxy never dips below the gate mid-drift.** A regression guard for C8: assert that the
   3-consecutive-frame run is not reset by the transient (i.e. the effective gain never approaches
   `+1.0` during an established drift).
5. **AC-088 — the measurement, not the constant.** Per E6, the RED contract must be authored
   against a measurement that can observe the fragment-shader gradient (shaded-framebuffer
   sampling), or the AC-088 lever must be re-scoped to the existing jest owner
   (`engine/lighting.test.ts`, OBS-20). **A RED contract of the form "the *current* fixture's
   measured gradient is strictly non-zero" is unsatisfiable by the named constants** and would
   strand the tune slice. This is a **plan-wording correction**, not a tuning task.
6. **Owner correction.** The Step 14N RED packet names `examples/neatRacing/engine/passes/car.pass.test.ts`
   as the file for the tire-gradient magnitude assertion, but the tire-gradient contract is owned
   by `examples/neatRacing/engine/lighting.test.ts` (`describe('fixed-space tire lighting (OBS-20)')`).
   The RED slice should correct the file list.

---

## Risks

| # | Risk | Severity | Evidence | Mitigation |
|---|---|---|---|---|
| R1 | **The gain is a V-shaped proxy lever with a null at 1.0.** A tune that raises the gain toward 1.0 makes the slide *weaker*, and at 1.0 the slide is invisible — while the offset magnitude looks *bigger*. | **HIGH** | E3 | Record the null as forbidden; prefer driving `GAIN_eff` down; add a RED regression guard on the proxy floor that would catch a null crossing |
| R2 | **AC-088's named lever set cannot close the measured residual.** The fixture measures vertex albedo, which is uniform by construction; the gradient is a fragment-stage effect. | **HIGH** | E6 | Re-scope AC-088's lever to the *measurement*; correct the file owner to `engine/lighting.test.ts`; treat this as a plan amendment in the RED slice, not a tuning task |
| R3 | **The 3-consecutive-frame gate (C8) is fragile.** Any transient that dips the proxy below `0.01` for one frame resets `driftProxyRunFrames`, which can break `tireYawDriftEngaged` and therefore **both** AC-078 and AC-079 simultaneously. | **HIGH** | E5 (C8), fixture `:876-881` | Visual-only design; drive the gain away from the null; add the RED guard in D7.4 |
| R4 | **The `\|driftYawOffset\| < π/2` pin (C2) bounds the total gain.** A large `GAIN_eff` combined with a large slip could breach it at the `physics.step.service.test.ts` fixture. | MEDIUM | E5 (C2) | Bound `NEATRACING_DRIFT_TRANSIENT_GAIN_MAX` so `\|GAIN_eff\| · \|slip\| < π/2`; the RED slice should measure the slip envelope at the pinned fixture before choosing the bound |
| R5 | **A transient that ramps slowly may be measured before it reaches steady state**, so the RED floor could be red for the wrong reason (the transient had not engaged yet). | MEDIUM | E2 (window mean), E3 | The 6-frame window mean is endpoint-anchored and cancels mid-window lag (residual ≤ 0.002 rad), but the RED contract must sample **after** the ramp completes, and must name that sampling point |
| R6 | **The build-rate / decay-rate trade-off is the feel decision itself** and cannot be tuned independently. | MEDIUM | D6 | The RED slice must record the direction; the tune slice gets one named pair of constants, not four |
| R7 | **The glass canopy does not follow the drift yaw** (`glass.pass.ts:1048,1086` has no `driftYawOffsetRad`). Feel work amplifies the visible mismatch between the body and the canopy. | MEDIUM | E1 | Out of Step 14N's scope (the pass is not re-authored); record as a carried-forward residual for a later step |
| R8 | **The bounce path zeroes `speed`/`slipAngle`/`lateralVelocity` but not the offset** (`collision.utils.ts:410,417-418`) — dormant today, latent once the accumulator is wired. | LOW | E7 | With the slip-proportional model the offset is zeroed automatically by `slipAngle = 0`; still worth a residual note |
| R9 | **Spec drift is already live** — the spec's illustrative formula (`clamp(lateralVelocity/speed · GAIN, ±max)`) does not match the implementation (`atan2(...) · GAIN`, no clamp). | LOW | E7 | Record as a residual; the spec amendment belongs to a documentation slice, not this one |
| R10 | **The car pass is not re-authored (AC-087)**, so a transient larger than the pass's current assumptions would need a pass change — out of scope. | LOW | D3 | Keep the total offset well inside `DRIFT_VISUAL_YAW_LIMIT_RAD`'s order of magnitude; the pass's monotonicity contract is unaffected either way |
| R11 | **The workflow-update-sync hook cannot see letter-SUFFIXED step ids.** `.github/hooks/workflow-update-sync.mjs` parses step headings with `/^#{4,5}\s+(?:Step|Packet)\s+([A-Z]?\d+)\s+(?:—|-)\s+(.+)\s\[([A-Z]+)\]/` — an id pattern that accepts `14` and `E1` but not `14N`. The heading `#### Step 14N — … [WIP]` is therefore never parsed, so the hook reports `currentWipStep: null` / `actionTaken: 'between-steps'` even though Step 14N **is** `[WIP]`, and it will not auto-advance Step 14N → Step 15. The hook still returns `ok: true` / `pass: true` and `plan-sync` still passes, so this is a **silent** degradation. | MEDIUM (workflow) | `.github/hooks/workflow-update-sync.mjs:196-198` | Pre-existing and **not** introduced by this research pass. Outside this slice's edit scope; the step advancement must be driven manually by the orchestrator. Owner: the workflow-hook maintainer |

---

## Certainty statement

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
- **D1 (the transient-gain model)** — confidence **0.85** as a *design recommendation*; it is
  **not** a decision. The feel direction and the floor value are explicitly owned by the RED
  slice `P1-drift-feel-red` and are **not** invented here.
- **R1, R2** — confidence **0.85**. Both are plan-level corrections that the RED slice must
  record before any production edit.
