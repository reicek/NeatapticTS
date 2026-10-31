# AUDIT-PHYSICS — read-only code audit of the drift/traction model

**Artifact id**: `neatracing-voxel-racer.audit-physics`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 3` record, `plans/neatracing-voxel-racer.plans.md` L9988+, `dispatch_plan[0]` = `AUDIT-PHYSICS`
**Subject**: `examples/neatRacing/physics/drift.model.ts` (787 lines) + `examples/neatRacing/physics/physics.step.service.ts` (647 lines)
**Constraints honoured**: READ-ONLY code audit · NO child agents · NO browser runs · NO Chrome MCP · no production byte touched.

---

## 0. Provenance, method and the naming deviation

**Method.** Every claim below is a *static-code* read plus a repo-wide readership grep. Line numbers are cited against the two subject files as they stand on disk (2026-10-27). No runtime measurement was performed (the directive forbids browser runs; a headless physics harness was also not in scope). The numeric sensitivities that are quoted as "derived" are arithmetic from the source constants (0.55 / 0.45 / 89 / 15 / 3.5 / 2.5 / 0.045 / 70) and are marked **DERIVED** where they are not literally printed in the source.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. There is no runtime evidence in this audit, so every finding is `static-code` ranked. Where a source JSDoc disagrees with the code, the **code wins** and the JSDoc drift is reported as a literal gap (G5).

**Naming deviation (disclosed).** The contract names `plans/<PlanName>.research.md`. That exact path is **already occupied** by a different, closed 14N artifact (`plans/neatracing-voxel-racer.research.md`, dated 2026-10-24 — the Step 14N drift-feel *integration coupling* design), and `plans/neatracing-voxel-racer.drift-feel.research.md` is likewise occupied. Clobbering or appending to a closed, dated artifact would fuse two different questions under one date. This artifact therefore follows the repo's own topic-suffix family convention (`neatracing-voxel-racer.14N-red-contracts.md`, `.14N-fixture-gate.md`, `.14N-rebaseline-risk.md`, `.drift-feel.research.md`) as a **new sibling**: `plans/neatracing-voxel-racer.audit-physics.research.md`. It remains a `plans/*.research.md` sibling of the tracker and is **not** under `docs/research/`. The orchestrator may rename or merge it without loss.

**Scope boundary.** This is the PHYSICS half only. The camera aim path, the wheel-spin idiom seam, the `stepPhysics` wrapper in `browser-entry.ts`, and the τ=0.08 follow law are `AUDIT-CAMERA`'s scope and are **not** adjudicated here. The single cross-boundary fact established by this audit is recorded in **F4**.

---

## 1. Question

The `USER VERDICT 3` dispatch asks, of the live physics model:

> what does traction/scrub loss **ACTUALLY** scale with; where speed × turning-angle coupling exists or is missing; the MIN_SPEED gate, the clamps, the latch, the D1.3 deficit form; literal GAPS (TODOs, dead branches, constants defined but unread, channels never wired).

And the verdict opens a NEW SPEC surface (S-2/S-3) asking whether the live model implements speed × steer proportionality, and **where fixed constants defeat the proportionality at game speeds**.

The audit must therefore answer two things: **what the law is**, and **what the contract can and cannot prove about it**.

---

## 2. Evidence

### 2.A — What the traction/scrub loss ACTUALLY scales with  *(confidence 0.95, source: static-code)*

The traction loss in the live model is a single term: the **longitudinal speed scrub**.

`drift.model.ts:632-646`
```ts
export function computeLongitudinalSpeedScrub(
  slipAngle: number,
  speed: number,
  speedPercent: number,
): number {
  return (
    NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S *   // 0.55   (drift.model.ts:282)
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
      state.slipAngle,   // tick-START published slip
      nextSpeed,         // tick-END speed
      speedPercent,      // tick-START speed percent
    ) * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
);
```

With `speedPercent = speed / 89` (`physics.step.service.ts:506-513`, `resolveSpeedPercent`, against `NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S = 89` at `:142`):

> **scrub [m/s²] = 0.55 · |published slip| · speed² / 89**

Four properties follow, and they are the whole answer to "what does it scale with":

| # | Property | Evidence |
|---|----------|----------|
| A1 | The **speed** dependence is **quadratic**, not proportional. The `speed` and `speedPercent` factors are *the same quantity twice* — `speedPercent` is a normalized copy of `speed`. The `speed²` term is a duplication artifact, not a designed second-order term. | `drift.model.ts:632-646` + `physics.step.service.ts:506-513` |
| A2 | The **slip** factor is the **PUBLISHED** slip (`state.slipAngle`, the tick-start `atan2` of the lateral velocity), **not** the committed slip, **not** the steering angle, **not** the yaw rate. | `physics.step.service.ts:422` (`state.slipAngle`) vs `:396` (`stepDriftCommitment` → `nextCommitted.slip`, a different value) |
| A3 | The scrub is **exactly zero on a straight line** (published slip = 0 ⇒ term = 0). Pinned by `[RED-G5-SPEED-SCRUB]`. | `drift.model.test.ts:724-736` |
| A4 | The scrub is applied as a **reduction of the tick-end speed**, floored at 0, and is *not* a separate "traction" channel — the value is computed inside `stepPhysics` and **discarded**. | `physics.step.service.ts:419-430` |

**DERIVED sensitivity table** (saturated committed slip 0.45 rad, so the published slip saturates there too):

| speed | speedPercent | scrub accel [m/s²] |
|---|---|---|
| 89 m/s (320.4 km/h) | 1.0 | 22.03 |
| 80 m/s | 0.899 | 17.80 |
| 30 m/s | 0.337 | 2.50 |
| 20 m/s | 0.225 | 1.11 |
| ≤ 15 m/s | ≤ 0.169 | **0.000 — the gate is closed** (see B1) |

**DERIVED equilibrium**: the engine accel is `89/5 = 17.8` m/s² (`NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2`, `physics.step.service.ts:148`). The scrub balances it at `0.55 · 0.45 · v²/89 = 17.8` ⇒ **v ≈ 80.0 m/s (287.5 km/h)**, matching the documented measured 79.858 m/s / 287.49 km/h deficit of 32.55 km/h at 600 ticks. The source's own JSDoc records the **4/9 crossover** (`0.4444`): the engine wins while the coefficient `c ≤ 4/9`; the landed `c = 0.55` is above the crossover, so the **scrub binds**. The scrub's maximum (22.03 m/s²) exceeds the engine's 17.8 m/s², so the scrub is the limiting term at top speed.

### 2.B — Where speed × turning-angle coupling exists, and where it is missing  *(confidence 0.93)*

**The coupling is MISSING.** There is **no steering-angle term and no yaw-rate term anywhere in the traction law.** The only "turning" quantity that reaches the speed path is the *committed slide*, and that slide's magnitude is destroyed by a clamp.

The full chain, as implemented:

```
steer ──▶ [0.5 threshold] ──▶ latch (turnSign) ──▶ [3.5 rad/s · |steer| · dt]  (SPEED-INDEPENDENT)
      ──▶ committed slip ──▶ [clamp ±0.45 rad]  ──▶ chase ──▶ published slip ──▶ scrub
```

Four independent blocks stand between "turning" and "traction loss". Each must be re-decided for S-3.

**B1 — The MIN_SPEED gate is a hard cliff, not a proportional ramp.** *(confidence 0.95)*
`drift.model.ts:184` — `NEATRACING_DRIFT_MIN_SPEED_MPS = 15` (54 km/h).

- `drift.model.ts:494-501` (stage 0 of `stepDriftCommitment`): below the gate the commitment is **released** (`slip = 0`, `turnSign = 0`) and the **re-arm token is cleared** (`reArmed = false`).
- `drift.model.ts:680-683` (stage 0 of `computeNextLateralVelocity`): the lateral velocity is returned as **exactly 0**.

Consequence: at ≤ 54 km/h the traction loss is **exactly 0.000**, with no fade-in. A car at 55 km/h and a car at 320 km/h differ by a *step*, not by a curve. The verdict's named constant ("the MIN_SPEED gate") is real and is a genuine discontinuity in the speed→traction map.

**B2 — The engage threshold is a fixed steer magnitude, and the engage ramp is speed-independent.** *(confidence 0.93)*
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

The car **rotates** at `yawRatePerTick = steer · 2 · speedPercent · dt` (`physics.step.service.ts:515-522`, `NEATRACING_PHYSICS_STEERING_RATE = 2`) and **loses no speed whatsoever**. At full speed, a half-lock turn is a free turn. This is the sharpest possible demonstration that **S-3 (speed × turning angle → traction loss) is not implemented**: the coupling term does not merely under-scale, it is *absent*.

**B3 — The commit clamp erases the turning angle.** *(confidence 0.92)*
`drift.model.ts:66` — `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45`; applied at `drift.model.ts:551-555`:
```ts
slip = clamp(slip, -NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD, NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD);
```
Once `|committed slip| ≥ 0.45 rad`, **further steering magnitude has zero effect on the traction law** — the clamp is a hard saturation on the only turning-shaped input to the scrub. Above the clamp the scrub reduces to `0.55 · 0.45 · speed²/89`: a function of **speed alone**. **This is the mechanism by which a fixed constant defeats the proportionality: the 0.45 rad commit clamp converts a "speed × turning" law into a "speed-only" law in the entire operating envelope the model actually reaches at speed.**

**B4 — The tire's grip budget is contractually SPEED-INDEPENDENT (the G-2 pin).** *(confidence 0.95)*
`drift.model.ts:567-579`:
```ts
export function computeGripBudgetFactor(
  slipAngle: number,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars -- the law is speed-INDEPENDENT (G-2)
  speedPercent: number,
): number {
  const demand = Math.max(0, Math.abs(slipAngle) - NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD);
  return 1 / (1 + NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD * demand);
}
```
The `speedPercent` parameter is accepted and **ignored** — with an explicit eslint-disable naming the reason. The factor is `1/(1 + 2·max(0, |slip| − 0.2))`: at the 0.45 rad saturation it is **0.667**, a 33 % falloff, and it is **speed-independent by design**.

**This is a RATIFIED RED pin, not an oversight.** `drift.model.test.ts` `[RED-G2]` asserts `computeGripBudgetFactor(0.9, 1) === computeGripBudgetFactor(0.9, 0.2)` — the tire grip budget is contractually a function of the slip only. The model's own JSDoc (`drift.model.ts:558-562`) records the decision: *"The budget is a function of the SLIP only — the speed demand is charged by the SPEED SCRUB (stage 6), not by the tire (G-2)."*

**Consequence for the verdict:** the `USER VERDICT 3` contested set names "the 20 km/h floor, MAX_SLIP_ANGLE_RAD 0.45, the commit clamp, the MIN_SPEED gate" as the fixed constants that defeat the proportionality. **It does not name G-2** — and G-2 is the pin that most directly forbids a speed term in the tire. Any S-2/S-3 implementation that adds speed to the grip law **must re-open the ratified `[RED-G2]` pin**, which the RED lane will otherwise re-pin. G-2 must be added to the contested set.

**B5 — The chase deliberately ignores the yaw rate.** *(confidence 0.90)*
`drift.model.test.ts` `[RED-G6-COMMITTED-CHASE]` pins that *"the 14N model has NO outward push — stage 3 is the committed chase, and the chase tracks the COMMITTED slip, **not the yaw rate**."* The chase target is `Math.tan(slip) * speed` (`drift.model.ts:694`). So the yaw rate — the one quantity that is literally "how much the car is turning" — is deliberately excluded from the lateral dynamics, and (per B4) it is also excluded from the tire.

### 2.C — The clamps, complete  *(confidence 0.94)*

| Clamp / bound | Where | Value | Runtime? | Effect on S-2/S-3 |
|---|---|---|---|---|
| Commit clamp | `drift.model.ts:551-555` (const `:66`) | ± 0.45 rad | **YES** | **Kills the turning term above 0.45 rad** (B3) |
| MIN_SPEED gate | `drift.model.ts:494-501`, `:680-683` (const `:184`) | > 15 m/s | **YES** | **Traction loss is exactly 0 below 54 km/h** (B1) |
| Speed floor | `physics.step.service.ts:420` | 0 m/s | YES | The scrub can only bleed speed to 0 |
| Speed ceiling | `physics.step.service.ts:567-602` (const `:142`) | 89 m/s | YES | The engine limit; the scrub's top-speed bound |
| Off-road floor | `physics.step.service.ts:436-445` (const `:182`) | 22.25 m/s (80.1 km/h) | YES | **Lifts a post-scrub speed back up — masks the scrub off-road** |
| Published-slip bound | `drift.model.ts:779-787` | (−π/2, π/2) | YES | Never reached (the published slip saturates ≈ 0.45) |
| Published-slip **assertion** bounds | `drift.model.ts:94` (0.002 overshoot), `:128` (0.452 max), `:163` (0.493 off-road max) | — | **NO — assertion only** | They never clip; they are read by tests (and mirrored as literals in the browser fixture) |

**The off-road floor is a real measurement hazard.** `isOffRoadSpeedFloorEngaged` (`physics.step.service.ts:543-565`) re-lifts the **post-scrub** speed to 22.25 m/s. On grass, the drift scrub's speed deficit is therefore **partially cancelled by the floor**. Any new proportional-traction contract must be measured and asserted **on road** — the browser fixture already enforces `straightCaptureOnRoad && driftCaptureOnRoad`, so the existing lane is on-road-safe.

### 2.D — The latch (three-state machine)  *(confidence 0.90 for L1, 0.88 for L2, 0.85 for L3)*

State: `DriftCommittedState { slip, turnSign, reArmed }` (`drift.model.ts:385-396`).

- **ENGAGE** (`drift.model.ts:514-525`): requires `reArmed && |steer| >= 0.5 && slip === 0`; sets `turnSign = sign(steer)`.
- **RE-ARM** (`drift.model.ts:526-530`): `|steer| < 0.15` ⇒ `reArmed = true`. **Lives only inside the `turnSign === 0` branch.**
- **RELEASE** (`drift.model.ts:537-548`): at `|slip| <= 0.02` ⇒ `slip = 0, turnSign = 0, reArmed = false`; and at the MIN_SPEED gate (`:494-501`).
- **CONTRACTION** (`drift.model.ts:537-549` + `computeCountersteerAuthority` `:591-618`): an opposite-sign steer unwinds the slide at a flat `EXIT_RATE_RAD_PER_S = 2.5` rad/s and **never flips `turnSign`**.

**L1 — a NEUTRAL wheel does NOT unwind the slide; it FREEZES it.** *(confidence 0.90)*
`drift.model.ts:506-508`:
```ts
const countersteering = turnSign !== 0 && Math.sign(steer) === -turnSign;
const drifting = turnSign !== 0 && !countersteering;
```
With `steer = 0`, `Math.sign(0) = 0 ≠ -turnSign` ⇒ `countersteering = false` ⇒ `drifting = true` ⇒ the **entry** ramp runs with `|steer| = 0`:
```ts
slip -= turnSign * 3.5 * Math.abs(steer) * dt;   // -= turnSign * 3.5 * 0 * dt  →  NO CHANGE
```
The slide is **frozen at its current value indefinitely**. The chase's hard-zero (stage 5, `drift.model.ts:709-714`) is gated on `!drifting`, so it does **not** fire either — the lateral velocity keeps chasing the frozen target and the slide persists. The slide is unwound **only** by an opposite-sign steer (`|steer| ≥ 0.9`, `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER`), or by the speed gate closing. A player who lifts the wheel mid-drift sees the car keep sliding, forever.

**L2 — the neutral re-arm branch is unreachable while committed.** *(confidence 0.88)*
The `|steer| < 0.15` re-arm sits inside `if (turnSign === 0)`. A committed car that returns the wheel to neutral therefore does **not** re-arm (it cannot — it is committed) and does **not** unwind (per L1). The latch is self-consistent, but the "neutral = let go" behaviour a player expects is absent, and the two branches of the latch have disjoint reachable conditions.

**L3 — the ENGAGE gate uses an exact float comparison.** *(confidence 0.85)*
`drift.model.ts:517` — `slip === 0`. It is reachable today (both release paths assign the literal `0`), so it is safe as written, but it is a fragile equality on a float that a future non-zeroing release path would break silently.

**L4 — the countersteer authority is NOT slide-magnitude-proportional.** *(confidence 0.90)*
`drift.model.ts:612-618` returns `turnSign · 70 · |steer| · speedPercent` m/s², gated only by `|slip| > 0.02` (`:231`). Catching a 0.45 rad slide and catching a 0.03 rad slide take the **same** authority. This is the one place in the model where a `speedPercent` factor genuinely enters an acceleration — it is the closest existing precedent for a "speed-proportional" term, and the 01-planning author should note it as such.

**L5 — the chase response rate is speed-independent.** *(confidence 0.85)*
`drift.model.ts:219` — `NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S = 0.045`, a **fixed** time constant, and `gripFactor` is speed-independent by G-2 (B4). So the chase's *response rate* does not scale with speed; only the chase *target* (`tan(slip)·speed`) does. The physics path therefore carries its own fixed τ, parallel to the camera's contested τ = 0.08. The verdict's contested set names only the camera's τ; this one is not in it.

### 2.E — The D1.3 deficit form, and why it cannot prove S-2/S-3  *(confidence 0.94)*

The D1.3 contract is a **single-point, absolute, floor-only** assertion.

**Jest lane** — `physics.step.service.test.ts:697-715`:
```ts
const straight = runSteps(600, 42, STRAIGHT_INPUT, 'road');
const drift  = runStepsFrom(atTopSpeed(), 600, FIXTURE_INPUT, 'road');
const deficitKmh = (straight.speed - drift.speed) * 3.6;
expect(deficitKmh).toBeGreaterThanOrEqual(NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH);
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
3. **It is absolute km/h, not a ratio.** The plan's own wording calls the D1.3 form *"the scale-free scrub-rate / speed-deficit assertion"* (`plans/neatracing-voxel-racer.plans.md` L4024) — but the assertion is **literally an absolute km/h delta**. It is not scale-free, and the plan's description overstates it.
4. **The upper-bound clause is PROVABLY VACUOUS.** `TRACTION_LOSS_MAX_SPEED_KMH = 340` (fixture L201). The service clamps the speed to 89 m/s = **320.4 km/h** (`physics.step.service.ts:142`, `:567-602`), and the scrub only ever **reduces** the speed. Therefore `maxSpeedKmhDrift ≤ 320.4 < 340` **always** — the clause can never fail. It is a dead term inside a check **named** `proportionalTractionLoss`: the name promises proportionality, the form asserts a floor.
5. **The unit-lane speed pin is a MONOTONICITY, not a proportionality.** `drift.model.test.ts` `[RED-G5-SPEED-SCRUB-2]` asserts only `expect(fast).toBeGreaterThan(slow)` (fast = the scrub at 80 m/s, slow = at 20 m/s). **Any strictly increasing function of speed passes this** — a logarithm, a `√v`, a `v^0.1`. The current contract set **cannot distinguish a proportional scrub from a merely monotone one**, and therefore cannot establish S-2.

**Correction to the verdict's premise (load-bearing).** `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH = 20` (`drift.model.ts:341`) is **never read by production code**. A repo-wide grep over `examples/`, `docs/` and `plans/` returns it in exactly two executable places: `physics.step.service.test.ts:85,704,1094` (the jest lane) and the **mirrored literal** `var DRIFT_MIN_SPEED_DEFICIT_KMH = 20;` (`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:227`, used at `:2504` and reported at `:2736`). The browser fixture **mirrors** the constant rather than importing it (the debug seam is fixed at five fields), so the two can drift apart silently.

**Therefore the 20 km/h floor does not "defeat the proportionality at game speeds" — it has no runtime effect at all.** What it defeats is the ability to **prove** proportionality: a floor at a single point constrains only the under-scrub direction and is blind to the shape of the speed→traction curve. The verdict's characterisation of this constant should be corrected in the amended plan.

### 2.F — Literal GAPS

**F1 — Fully dead exports (defined, exported, read by NOBODY).** *(confidence 0.96)*
A repo-wide grep over `examples/neatRacing/**/*.ts` for `COUNTERSTEER_HOLD_TICKS` returns **exactly one match — the declaration itself**:
`drift.model.ts:251` — `export const NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS = 8;`
It is read by **no production code and not even by a test**. Its own JSDoc records it as an open GAP (*"EXPORTED but NOT ENFORCED"*). It is a **hard** gap: a named, documented, contract-surface constant with **no reader at all**.

**F2 — Exports read only by tests (never by production).** *(confidence 0.95)*

| Symbol | Where | Value | Production reader? |
|---|---|---|---|
| `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD` | `drift.model.ts:288` | 0.2 | **NO** — jest only + a mirrored fixture literal |
| `NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD` | `drift.model.ts:294` | 0.01 | **NO** — jest only + a mirrored fixture literal |
| `NEATRACING_DRIFT_PROXY_MIN_FRAMES` | `drift.model.ts:299` | 3 | **NO** — jest only + a mirrored fixture literal |
| `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH` | `drift.model.ts:341` | 20 | **NO** — jest only + a mirrored fixture literal |
| `computeDriftSlideProxyRad()` | `drift.model.ts:655-662` | fn | **NO** — jest only; the fixture mirrors the formula |
| `NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD` | `drift.model.ts:128` | 0.452 | **NO** — jest only + a mirrored fixture literal |
| `NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD` | `drift.model.ts:163` | 0.493 | **NO** — jest only + a mirrored fixture literal |

The last four (the AC-087 prominence-floor family) are the notable ones: **the deterministic unit proxy never reaches production.** The proxy exists in the test lane and as a **hand-copied literal** in the browser fixture — the "one source of truth" is a **duplication** that can drift. The published-slip ceilings are legitimate as *assertion* bounds, but they are asserted in a lane that cannot import them.

**F3 — Dead fields and dead parameters.** *(confidence 0.95)*
- `DriftStepInput.slipAngle` (`drift.model.ts:421-437`) is **passed but never read**: `stepDriftCommitment` destructures `{ steer, speed, dt }` (`:489`); `computeNextLateralVelocity` destructures `{ speed, speedPercent, dt, lateralVelocity }` (`:674`); `computeCountersteerAuthority` destructures `{ steer, speedPercent }` (`:595`). Yet `physics.step.service.ts:402` and `:416` still pass it. **A dead field on the model's own input contract.**
- `computeGripBudgetFactor`'s `speedPercent` parameter — accepted and ignored, eslint-disabled at `drift.model.ts:569`. It encodes the G-2 decision *in the signature*, which is why the pin is easy to miss on a read.

**F4 — Channels published but never wired (the S-4 surface).** *(confidence 0.96)*
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

**F5 — No traction-loss scalar exists on the channel.** *(confidence 0.95)*
There is **no** traction-loss / scrub / grip field on `PhysicsCarState`. The scrub is computed at `physics.step.service.ts:419-430` and **discarded**; only its *effect* (a reduced speed) survives. Therefore:
- The camera **cannot** read "the same drift/traction-loss amount the tires experience" (S-4) from the physics today.
- To satisfy S-4 the view layer would have to **re-derive** the scrub from `car.slipAngle` and `car.speed` by re-implementing `computeLongitudinalSpeedScrub` outside the model — a **duplicated physics law** in the presentation layer, which is the classic drift hazard, and which would silently diverge from the model the moment a constant moves.
- **The physics must publish the shared scalar** for S-4 to be implementable without duplication. Candidate scalars, in order of directness: **(a)** the scrub acceleration `0.55·|slip|·speed·speedPercent` — the literal "traction loss", already speed-proportional, already the quantity the tire actually loses; **(b)** the grip factor `1/(1+2·max(0,|slip|−0.2))`; **(c)** the committed slip.

**F6 — Stale JSDoc describing the retired TP-02 model.** *(confidence 0.90)*
`physics.step.service.ts:97-108`:
- `slipAngle`: *"Zero while the car is not moving; **grows only under lateral push, decays via grip**"* — this describes the **retired TP-02 outward-push** model. The 14N model has **no outward push** (pinned by `[RED-G6-COMMITTED-CHASE]`, B5); the slip is produced by the committed chase.
- `lateralVelocity`: *"**Driven by the TP-02 outward push**, drained by the grip rate"* — the same staleness; the lateral velocity is now driven by the committed chase toward `tan(slip)·speed`.

Both are on the **public state contract** that the camera and the collision code read, so a reader of the interface is actively misinformed about where the lateral motion comes from.

**F7 — A mixed tick read in the scrub.** *(confidence 0.88)*
`physics.step.service.ts:419-427` feeds the scrub the **tick-start** published slip (`state.slipAngle`) and the **tick-start** `speedPercent` (`:374`) but the **tick-end** speed (`nextSpeed`). The inline comment claims *"the scrub reads the tick-start slip, like every other tick-start read in this integrator"* — true for the slip, but the **speed** in the same expression is a tick-end read. The choice is self-consistent (the scrub bleeds the tick-end speed), but the comment overstates the uniformity, and the mixed read makes the feedback loop (scrub → speed ↓ → chase target ↓ → published slip ↓ → scrub ↓) order-dependent in a way that is hard to reason about from the comment alone.

**F8 — The "single source of truth" is a duplication in the fixture.** *(confidence 0.90)*
The browser fixture mirrors `DRIFT_MIN_SPEED_DEFICIT_KMH` (L227), `DRIFT_MIN_SPEED_KMH` (L233), `ACCEL_TARGET_SPEED_KMH` (L236), `DRIFT_PROXY_MIN_RAD` (L256) and the published-slip ceilings as **hand-written literals**, because the debug seam is fixed at five fields and cannot import. Every one of these is a drift hazard on the next constant change, and the mirrored `proportionalTractionLoss` check's dead 340 km/h clause will keep passing regardless of what the physics does.

---

## 3. Decision

**D1 — Gap classification of the four verdict clauses.**

| Clause | Gap type | Confidence | One-line verdict |
|---|---|---|---|
| **S-2** (traction loss scales proportionally with speed) | **partial** | 0.92 | The scrub is **quadratic** in speed (via the duplicated `speedPercent`), but only through a redundancy; the slip factor is speed-independent, the tire's grip budget is **contractually** speed-independent (G-2), and below the MIN_SPEED gate the loss is **exactly zero**. |
| **S-3** (traction loss = speed × turning angle) | **missing** | 0.93 | There is **no steering-angle or yaw-rate term anywhere in the traction law**. The only turning-shaped input is the committed slide, whose magnitude is erased by the 0.45 rad clamp, and a full-speed turn below the 0.5 engage threshold loses **exactly zero** speed. |
| **S-4** (one shared scalar drives both) | **missing** | 0.95 | **No traction-loss scalar is published on the channel.** The camera cannot read the tires' loss amount; the only way to satisfy S-4 today is to duplicate the physics law in the view layer. |
| **S-1** (camera animates proportionally) | **out of scope** | — | `AUDIT-CAMERA`'s surface. The physics-side fact that bears on it: `car.yawRate` and `car.drift.pose.poseRate` are already published and **entirely unwired** (F4). |

**D2 — The four fixed constants that defeat the proportionality, in dependency order.** All four must be re-decided for S-3; the order matters because each one gates the next.

| # | Constant | Where | What it does to the proportionality | Contested by the verdict? |
|---|---|---|---|---|
| **B4** | **G-2** (the speed-independent tire grip budget) | `drift.model.ts:567-579` + `[RED-G2]` | **Forbids a speed term in the tire outright** | **NO — must be added** |
| B1 | `NEATRACING_DRIFT_MIN_SPEED_MPS = 15` | `drift.model.ts:184` | A hard cliff: **zero** traction loss below 54 km/h | YES |
| B2 | `NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD = 0.5` + the speed-independent `ENTRY_RATE = 3.5` | `drift.model.ts:189`, `:201` | A full-speed turn under half-lock is **free**; the slide builds at the same rad/s at any speed | Partially (the ramp is not named) |
| B3 | `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45` | `drift.model.ts:66` | **Erases the turning term** above 0.45 rad ⇒ the law becomes speed-only | YES (as "the commit clamp" / "MAX_SLIP_ANGLE_RAD 0.45") |

**D3 — The contract must be RE-AUTHORED, not merely re-pinned.** The current AC-079 set is floor-only, single-point, and (in one term) provably vacuous. To *establish* S-2/S-3 the contract needs at minimum:

1. A **multi-speed comparison** — the same steer input at ≥ 2 speeds, asserted as a **ratio** or a **normalized deficit**, not a monotonicity. `[RED-G5-SPEED-SCRUB-2]`'s `fast > slow` cannot distinguish proportional from monotone and must be **strengthened**, not merely re-run.
2. A **turn-angle comparison** — the same speed at ≥ 2 steer magnitudes (one above, one below the 0.5 engage threshold), asserting that the loss **increases** with turning. This is the assertion that would have caught the missing coupling.
3. A **clamp-region assertion** — that the loss continues to rise with turning **past** the 0.45 rad commit clamp, or a documented decision that the clamp **is** the saturation and the proportionality is defined only below it.
4. The **dead 340 km/h clause must be removed or made reachable** (it can never fail; see 2.E reason 4). The plan's own B-9 disposition replaced the retained 290 km/h ceiling; a second provably-vacuous ceiling has replaced it and carries the same defect.
5. The mirrored fixture literals (F8) must be re-derived from the model, or the seam must be widened to carry the values.

**D4 — S-4 requires a physics-side decision BEFORE the camera contract can be written.** The shared scalar must be **published**, not re-derived. Recommended: publish the **scrub acceleration** (F5-a) as the canonical "traction-loss amount" and let the camera consume it. The two channels already published and unwired — `car.yawRate` and `car.drift.pose.poseRate` — are the natural *proportional* camera inputs and should be wired in the same wave, so S-1 and S-4 are satisfied by one shared read rather than two parallel re-derivations.

**D5 — The corrections the amended plan must carry forward.**
1. **G-2 joins the contested set** (D2, B4) — it is the pin that forbids a speed term in the tire, and it is not currently listed.
2. **The 20 km/h floor is a test-only constant** (2.E) — it has **no runtime effect**; it defeats the *proof*, not the *law*. The verdict's premise needs correcting.
3. **The D1.3 form is not "scale-free"** (2.E reason 3) — it is an absolute km/h floor at one point.
4. **The camera's τ = 0.08 is not the only fixed τ** — the tire chase has its own fixed `SLIP_RESPONSE_TAU_S = 0.045` (L5) and its response rate is speed-independent by G-2.

**D6 — The literal gaps are, in priority order:** **F1** (a named contract constant with **no reader at all** — `COUNTERSTEER_HOLD_TICKS`), **F5** (no traction-loss scalar — blocks S-4), **F4** (`yawRate`/`poseRate`/`committed.*` unwired), **F2+F8** (the duplicated mirror set), **F3** (the dead `slipAngle` input field), **F6** (the stale TP-02 JSDoc on the public contract), **F7** (the mixed tick read), and **L1** (the neutral-wheel freeze — a *behavioural* gap that plausibly contributes to the user's "the implementation we have now is terrible").

---

## 4. Risks

- **R1 (highest) — the G-2 pin.** Implementing speed in the tire without re-opening `[RED-G2]` breaks a **ratified RED contract**; the RED lane will re-pin it and the GREEN lane will fail. G-2 is currently absent from the verdict's contested set, so the amended plan would be authored against a pin nobody has been told is contested. **Mitigation: add G-2 to the contested set in the 01-planning amendment, explicitly.**
- **R2 — the overshoot tolerance is coupled to the scrub coefficient.** `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD = 0.002` (`drift.model.ts:94`) is **calibrated** to the measured chase overshoot at `c = 0.55`, and the source JSDoc records the calibration envelope as **exhausted** at `c ≈ 0.61` (only 0.00018 rad of headroom). **Any change to the scrub coefficient, the entry rate, or the chase τ forces a re-derivation of the overshoot tolerance and of the published-slip ceilings** (`0.452` / `0.493`), and by extension of the browser fixture's mirrored literals (F8). This is a **coupled blast radius**, not an isolated constant — the 01-planning author must schedule the re-derivation with the change, not after it.
- **R3 — the off-road floor masks the scrub.** On grass, `isOffRoadSpeedFloorEngaged` re-lifts the post-scrub speed to 22.25 m/s, so the speed deficit is partially cancelled. **Any new proportional-traction measurement must be taken on road** (the existing fixture lane already enforces `straightCaptureOnRoad && driftCaptureOnRoad`, so it is on-road-safe — the risk is for *new* lanes, not the retained one).
- **R4 — the mirrors can drift silently.** The proxy floor, the deficit floor and the published-slip ceilings are hand-copied into the fixture. The `proportionalTractionLoss` check's dead 340 km/h clause will keep passing no matter what the physics does, so the fixture will report GREEN while the model is wrong. **Mitigation: remove or make reachable the vacuous clause before the amended contract is authored, and widen the seam or re-derive the mirrors with every constant change.**
- **R5 — the neutral-wheel freeze (L1) is a live behaviour, not a theoretical one.** If the amended contract keeps the latch mechanics (the verdict retains the seam mechanics but re-decides the follow/aim law), a player who releases the wheel mid-drift sees a **frozen slide and a frozen camera**. This is a plausible contributor to the user's "terrible" report and should be an explicit decision, not an accident.
- **R6 — no runtime measurement was performed.** Every claim here is a static-code read; the sensitivities in 2.A are **DERIVED** arithmetically from the source constants. Before the contract is re-authored, a measurement lane should confirm: (i) the scrub at 25 / 50 / 89 m/s at a fixed committed slip; (ii) the **zero-traction** case for `|steer| < 0.5` at top speed; (iii) the scrub at two steer magnitudes at a fixed speed; and (iv) the neutral-wheel freeze (L1). Without (ii) and (iii) the S-3 gap is argued, not measured — and the S-3 RED contract should be *born* from the measurement.
- **R7 — the audit boundary.** This artifact adjudicates the **physics** surface only. The camera aim path, the wheel-spin idiom seam, the `browser-entry.ts` `stepPhysics` wrapper and the τ = 0.08 follow law belong to `AUDIT-CAMERA`; **S-1 and S-4 remain open until that audit returns.** The one cross-boundary fact established here is F4/F5: the physics publishes **no** traction-loss scalar, and the camera's only drift inputs today are `car.drift.pose.pose` and `car.slipAngle`. If `AUDIT-CAMERA` reaches a different conclusion about the available inputs, the merge point is the orchestrator's (per the verdict's `merge_point`), and this artifact's D4 stands only as a *recommendation*.
- **R8 — the "no child agents" constraint.** The dispatch explicitly forbade child agents, overriding the generic Tier-1 delegation mandate. This audit is therefore **single-context** and its findings are **not cross-checked by an independent scout**. Every headline claim is a direct code read with a cited line, and the three highest-stakes claims (the dead constant, the unwired `yawRate`, the test-only deficit floor) were re-verified by independent repo-wide greps in this session (F1, F4, 2.E). Residual risk: a *missed* gap that a second pair of eyes would have found. Mitigation: the 01-planning author re-runs the F-series greps before the contract is frozen.

---

## 5. Evidence index (files read, read-only)

| File | Read | Purpose |
|---|---|---|
| `plans/neatracing-voxel-racer.plans.md` | L9988–10120 (the `USER VERDICT 3` record), L9740–10093 (the tail packets) | the dispatch, the contested set, the plan-family conventions |
| `examples/neatRacing/physics/drift.model.ts` | **complete, 787 lines** | the model under audit |
| `examples/neatRacing/physics/physics.step.service.ts` | **complete, 647 lines** | the tick pipeline |
| `examples/neatRacing/physics/drift.model.test.ts` | L695–760 | `[RED-G2]`, `[RED-G5-SPEED-SCRUB]`, `[RED-G5-SPEED-SCRUB-2]`, `[RED-G6-COMMITTED-CHASE]` |
| `examples/neatRacing/physics/physics.step.service.test.ts` | L690–800, L481–483 | the D1.3 deficit pin; the `yawRate` test-only reads |
| `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html` | L195–275, L2488–2516, L2736 | the mirrored literals and the `proportionalTractionLoss` check |
| `examples/neatRacing/browser-entry/browser-entry.ts` | L972, L981–982 | the only production reader of the drift channel |
| `examples/neatRacing/collision/collision.utils.ts` | L386–389 | the only production reader of `slipAngle` / `lateralVelocity` |
| `plans/neatracing-voxel-racer.research.md` | L1–60 | the pre-existing 14N artifact (to avoid clobbering it) |

**No file was modified by this audit.** The only writes are this artifact and the `AUDIT-PHYSICS` PlanUpdate appended to `plans/neatracing-voxel-racer.plans.md`.
