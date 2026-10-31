# Step 14N Phase-B — Physics-Coupling + Re-Baselining Risk

**Agent:** `02-researching` (Phase-B plan agent **2/4**) · **Slice:** `14N` · **Plan:** `plans/neatracing-voxel-racer.plans.md`
**Charter:** propose the **AC-079 / AC-089 re-baselining**, the **AI steer-seam impact (Phase-2 shared seam)**, and the **fixture threshold updates**.
**Writes:** this file ONLY. No plan edit, no production edit. Read-only reconnaissance of the plan, the two Phase-A research artifacts, the physics lane, the fixture and the two AI lanes.
**Status:** COMPLETE (authored 2026-10-26; 344 lines). This file **is** the deliverable — the chat report is a courtesy summary only. Sibling Phase-B artifacts: `-red-contracts.md`, `-integration.md`, `-fixture-gate.md`.

---

## DELIVERABLE SUMMARY (self-contained — read this if the chat report was lost)

**The one-line answer.** 14N's committed-drift model is a *trajectory* change, so AC-079/AC-089 must be **re-baselined as contracts, not as numbers** — and the re-baselining is blocked on **two unspecified MODEL rules** (not test changes) plus one **plan-level file-set amendment** for the AI steer seam.

### The three mandated deliverables

| # | Deliverable | Where | Headline |
|---|---|---|---|
| **1** | **AC-079 / AC-089 re-baselining** | **D1** (§D1.1–D1.5) | A 27-row pin ledger (D1.2) with a verdict per pin — **PRESERVE / RE-BASELINE / RE-HOME / RE-SCOPE / DELETE-as-dead** — plus the AC-079 **semantic** replacement (D1.3), the proposed **AC-089 re-wording** (D1.4) and **two model requirements** that fall out of the ledger (D1.5). |
| **2** | **AI steer-seam impact (Phase-2 shared seam)** | **D2** (§D2.1–D2.4) | The model turns `sign(steer)` into a **discrete intent bit**; the human axis is **digital** `{−1,0,+1}` but **both** AI lanes emit a **continuous proportional** steer ⇒ an AI heading-error zero-crossing becomes an **accidental countersteer**. A 3-clause **Phase-2-safe intent contract** (D2.2) + a documented fallback with its residual risk (D2.3) + a 9-row impact ledger (D2.4). |
| **3** | **Fixture threshold updates** | **D3** (§D3.1–D3.2) | Full ledger for `p1-turn-drift-green-smoke.html`: **4 re-baselines** (one must change *semantics*, one must be *derived* from the AC-087 authority, one needs a *re-derived residual bound* for the second-order pose), **5 preservations** (structurally justified), **1 new named check**, **1 re-scope** to the `uTireGradient*` uniform seam. |

### The five HIGH-severity findings (details + evidence in §Risks)

1. **RB-3 — AC-079's law is RETIRED, not moved.** The new constant set retires `computeLongitudinalTractionLoss` (0.6) in favour of `SPEED_SCRUB_COEFF_PER_S` (0.25). AC-079's wording ("a proportional law that scales with the slip angle and the speed") had exactly **one** implementation — that export. Unless the law is **re-homed** onto the scrub term, **AC-079 becomes vacuous**. Its 6 jest pins must be **re-authored, not deleted**. *(conf 0.95)*
2. **RB-4 — the 290 km/h ceiling INVERTS; the check FAILS.** The old model scrubbed 24.2 m/s² (above the 17.8 m/s² engine) and bled the drift phase to the recorded 127 km/h; the new model scrubs 10.0 m/s² (**below** the engine), so the car can hold ~320 km/h in a drift ⇒ `maxSpeedKmhDrift` goes "127 < 290" ⇒ "≈320 > 290". **Fix:** replace the speed *ceiling* with a **scrub-rate / speed-deficit assertion** (scale-free, and the actual content of AC-079). *(conf 0.78)*
3. **RB-1 — D1's latch re-arm rule is UNSPECIFIED and is load-bearing.** Only stage 5's hard zero makes the exponential chase reach **exact** 0, and therefore only it keeps **AC-047D's byte-identity** and the straight-line **exact-zero** pins green. **Blocking for `P1-drift-feel-red`.** *(conf 0.95 / 0.92)*
4. **RB-2 — D1 has NO `MIN_SPEED_MPS` gate** in stages (1)/(3) ⇒ the low-speed exact-zero pin (`physics.step.service.test.ts:264-265`: `lateralVelocity === 0` at 10 m/s) **fails**. The gate must live **inside the model**, not in a test. *(conf 0.90)*
5. **RB-5 — the steer-seam intent overload** (deliverable 2) needs a **plan-level file-set amendment**: the intent signal implies a `PhysicsStepInput` field, and `host/input.ts` / `browser-entry.ts` / `ai.autopilot.ts` are **not** in the AC-079/087/089 file sets. **That is `01-planning`'s call**, not this agent's.

### The new public surface this artifact proposes

`NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH` (D1.3) · `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER` + `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS` (D2.3, fallback only) · the **latch re-arm rule** (RB-1) · the **bounce-reset** contract for the committed state **and** the pose state (RB-9) · the new fixture check `driftFeelSlideProminence` (D3.1) · the new fixture constants `DRIFT_PROXY_MIN_HOLD_FRAMES` and the derived `DRIFT_PROXY_MIN_RAD` (D3.1).

### Two corrections to the brief (both verified against the file tree)

* `examples/neatRacing/ai/observation.assembler.ts` **does not exist** — the implemented analog is `examples/racing_curriculum/controller/observation.assembler.ts`.
* The **Phase-2 lane has no lateral-velocity model** (`environment.step.service.ts:808` → `lateralSpeedWorld = 0`) and does **not** import the drift model ⇒ 14N changes the shared steer **seam**, **not** the Phase-2 **dynamics**. The user's *"not sure if this will affect later AIs"* is answered precisely: **the seam yes, the dynamics not yet** (P-4).

### Validation state (all green)

`cortex-index` **PASS** (1998 docs, 9/9 families fresh) · `cortex-first-search` **PASS** (27452 chunks) · `slice-advancement` **PASS** for slice `14N` with `--changed-files=plans/neatracing-voxel-racer.14N-rebaseline-risk.md` (all 4 sub-gates green) · `workflow-update-sync.mjs --dry-run` **PASS but silently degraded** (`currentWipStep: null` while 14N **is** the `[WIP]` step — the step-id regex does not match letter-suffixed ids; recorded for `00-helping`, **not** fixed here, and **out of this agent's edit scope**).

---

## Question

The Phase-A MERGE block (`plans/neatracing-voxel-racer.plans.md:8414-8441`) **LIFTS the MUST-NOT-MOVE PIN**: the committed-drift model replacement is a *trajectory* change, so **AC-079 (the 290 km/h traction-loss ceiling) and AC-089 (the no-regression pin) must be RE-BASELINED, not preserved**. Phase B must produce that re-baselining plan, and must also answer two coupling questions the Phase-A block leaves open:

1. **What exactly is re-baselined, and what must NOT move?** (a pin inventory with a verdict per pin, a value or a semantic replacement, and the file that owns it.)
2. **What does the new model do to the AI steer seam** — the channel that Phase 2 (`AC-027` observation assembler, `AC-028` scripted controller) shares with the Phase-1 autopilot through the **same 2.2 rad⁻¹ gain convention** (`AC-086`)?
3. **Which browser-fixture thresholds must move, which must hold, and what new named contract replaces them?**

---

## Evidence

Provenance: `runtime` > `static-code` > `docs` > `external`. Confidence is per finding. All line numbers were read directly from the working tree in this session (the `docs/browser-tests/scenarios/*.html` family is **not in the Cortex corpus** — `load_document` → *"Document not found"* — so the fixture was read natively; that is a recorded retrieval gap, not a correctness gap).

### E1 — The 7-stage model turns `sign(steer)` from a continuous control into a discrete intent

`plans/neatracing-voxel-racer.drift-feel.research.md:277-341` (arcade-refs D1):

```
(0) LATCH   turnSign = latched at entry = sign(steer that started the drift)
            countersteering = (turnSign != 0) && sign(steer) == -turnSign
            drifting        = (turnSign != 0) && !countersteering
(1) committedSlip -= turnSign * ENTRY_RATE(3.5) * |steer| * dt      (drifting)
    committedSlip  = unwinds at EXIT_RATE(2.0) * dt                 (otherwise)
(4) if (countersteering && |slip| > 0.02)
        vLat += turnSign * COUNTERSTEER_AUTHORITY(70 m/s²) * |steer| * speedPercent * dt
(5) if (!drifting && |slip| < 0.02) { vLat = 0; committedSlip = 0; }   <-- HARD ZERO
```

**Today** `steer` has **no** semantic overload: `computeCountersteerDelta` (`drift.model.ts:311-322`) is a *magnitude* function whose sign gate is `steer * slipSign <= 0` — it reacts to a *crossing*, not to an intent.
**After 14N** the **sign bit of a continuous axis** latches a drift and triggers a 70 m/s² counter-authority. **Confidence 0.95** (source + D1 as written).

**E1a — the latch re-arm rule is NOT specified in D1.** D1 says *"latched at entry"* and never says when `turnSign` returns to 0. This is not cosmetic:
* If `turnSign` is never cleared, then **after any drift, every later opposite-sign steer is permanently a countersteer** and the AI can never re-enter a drift in the other direction.
* Stage 5's hard zero (`!drifting && |slip| < 0.02`) requires `!drifting`, which is satisfiable *during* a countersteer — so the hard zero can fire — but the **latch itself is not released** by it.
**Confidence 0.95** (D1 read as written).

**E1b — the hard zero (stage 5) is LOAD-BEARING for the exact-zero pins.** Stage 3's chase is **exponential** (`vLat += (vTarget - vLat) * (1 - exp(-dt*gripFactor/τ))`) and therefore never reaches exact 0. Only stage 5 snaps to exact 0. The AC-047D byte-identity pin (`drift.model.test.ts:325-327`), the straight-line exact zeros (`physics.step.service.test.ts:250-251`, `:264-265`) and the C5 traction-loss zero (`drift.model.test.ts:181`) **all require stage 5 to fire**. **Confidence 0.92** (mechanism) / **0.85** (that the RED slice will pin it).

**E1c — D1's stage (1) has NO minimum-speed gate.** `MIN_SPEED_MPS = 15` survives in the constant set, but the committed-drift ramp as written has no speed condition, and stage 3's chase drives `vLat` toward `tan(committedSlip) * speed` at **any** speed. The current model passes the low-speed pin (`physics.step.service.test.ts:264-265`, 10 m/s with `steer: 1` for 120 ticks ⇒ `lateralVelocity === 0` and `slipAngle === 0` **exactly**) only because the old slip is tiny at 10 m/s. **Under D1 as written that pin FAILS.** The committed model therefore **must** carry the `MIN_SPEED_MPS` gate into stage (1) *and* stage (3). **Confidence 0.90.**

### E2 — The two `steer` producers have *complementary* noise structure

| Producer | File:line | `steer` values | Nature of a sign flip |
|---|---|---|---|
| **Human** | `examples/neatRacing/host/input.ts:89-91` | `(left?1:0) - (right?1:0)` ⇒ **exactly {−1, 0, +1}** | a key release+press — *is* an intent |
| **AI (Phase 1)** | `examples/neatRacing/physics/ai.autopilot.ts:174` | `clampSteer(headingError * 2.2)` ⇒ **continuous** | a **zero-crossing of a geometric error** — is *not* an intent |
| **AI (Phase 2 scripted)** | `examples/racing_curriculum/controller/scripted.controller.ts:170-175` | `clamp(normalizedHeadingError * 2.2)` ⇒ **continuous** | same |

**Confidence 0.97** (source reads). The 2.2 rad⁻¹ gain is a **duplicated literal** in two files (`ai.autopilot.ts:30` `NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD = 2.2`; `scripted.controller.ts:53` `STEER_PROPORTIONAL_GAIN = 2.2`) with **no shared constant module** — exactly the seam `AC-086` names.

### E3 — The Phase-2 lane has **no lateral-velocity model** and does **not** import the drift model

* `examples/racing_curriculum/environment/environment.step.service.ts:808` → `const lateralSpeedWorld = 0;`
* `examples/racing_curriculum/environment/environment.types.ts:101-102` → *"Signed lateral speed in world units per second; `0` until a lateral-velocity model is added."*
* A whole-tree grep of `examples/racing_curriculum` for `neatRacing|drift\.model|DRIFT_MAX_SLIP|slipAngle|driftYawOffset` returns **21 matches, none of them an import of the neatRacing physics** — only `slipAngleRadians` reads inside the observation assembler and the NGE controller.

⇒ **The 14N drift model does not change the Phase-2 dynamics directly.** What crosses the boundary is the **steer-seam semantics** and the **observation normalisation constants**. **Confidence 0.97.**

### E4 — The Phase-2 observation normalisation is calibrated to the OLD slip range

`examples/racing_curriculum/controller/observation.assembler.ts:945` normalises the slip channel by `Math.PI / 2` (channel 74, documented in `examples/racing_curriculum/README.md:625` as `slipAngleRadians / (π / 2)`); the lateral channel is normalised by `/54` (`:933`). The new model bounds the slip at **`MAX_SLIP_ANGLE_RAD = 0.45 rad`**, so the entire live range compresses to **±0.286 of the nominal channel range** — a **3.5× loss of observation resolution** for every controller trained on channel 74. Both channels are **dead-zero in production today** (E3), so this is a *forward-looking* cost, not a live regression. **Confidence 0.85.**

### E5 — The AI's closed loop changes because the **trajectory** changes

The autopilot steers toward `desiredHeading - car.heading` — the **body yaw**, which the slip does *not* change. But the car **translates** along `heading + slipAngle` (`physics.step.service.ts:425`), so a 5× larger slide displaces the car laterally before the next tick and the 12 m lookahead's bearing shifts. At 25 m/s with a 25.83° committed slip the lateral velocity is `tan(0.451)·25 ≈ 12.0 m/s`; over a 0.5 s drift that is ~6 m of lateral displacement ⇒ **≈0.46 rad of bearing shift to the lookahead target** — past the AI's `FULL_THROTTLE_ERROR_RAD = 0.35` and approaching `FULL_BRAKE_ERROR_RAD = 0.7` (`ai.autopilot.ts:33,:36`). The AI's **binary** throttle/brake therefore modulates far more, which lowers `speedPercent`, which **lowers the countersteer authority** (it is scaled by `speedPercent`) — a **positive feedback loop**. **Confidence 0.55** (first-order estimate; the closed loop is not simulated) ⇒ **must be MEASURED in the RED slice, not asserted.**

### E6 — The AC-079 jest law is **retired** by the new constant set

`drift-feel.research.md` D2: *"Constants that this model RETIRES: `LATERAL_GRIP_RATE_PER_S` (4) → `SLIP_RESPONSE_TAU_S` + the saturation budget; `COUNTERSTEER_RECOVERY_RATE_PER_S` (12) → `COUNTERSTEER_AUTHORITY_MPS`; `LONGITUDINAL_TRACTION_LOSS_RATE_PER_S` (0.6) → `SPEED_SCRUB_COEFF_PER_S` (0.25, and now saturating rather than proportional-to-speed)."*

`computeLongitudinalTractionLoss` is the **only** implementation of AC-079's *"proportional law that scales with the slip angle and the speed"* (`plans.md:474`). It is retired. **AC-079 therefore has no implementation left to bind to, unless the scrub term is re-authored as its successor.** **Confidence 0.95.** This is the single most consequential coupling fact in this artifact: *AC-079 is not merely a number that moves — it is a law that must be re-homed or it silently becomes vacuous.*

### E7 — The 290 km/h ceiling will not merely "drift" — the check **flips**

The old model scrubbed **24.2 m/s²** at 89 m/s (D3) — *more* than the engine's 17.8 m/s² — so the drift phase bled down to the **127 km/h** recorded in the Step 14G green instance (`plans.md:274`). The new model scrubs **10.0 m/s²** (D3) — *below* the engine — so the car can hold the full ~320 km/h through a sustained drift. The fixture check is `tireYawDriftEngaged && 0 < maxSpeedKmhDrift < 290` (`p1-turn-drift-green-smoke.html:935-937`): the **ceiling becomes the binding constraint and the check inverts from "127 < 290" to "≈320 > 290" ⇒ FAILURE.** **Confidence 0.78** (the D3 scrub figure is the arcade scout's own simulation; the fixture's scenario speed must be confirmed).

### E8 — The visible yaw envelope **shrinks 3.5×**

`DRIFT_VISUAL_YAW_LIMIT_RAD = MAX_SLIP · GAIN` moves `0.7854 → 0.2250` rad (R3). The pose spring's own entry overshoot measures **0.2295 rad (2.00 % OVER)** without the slew, and **0.2246 rad (0.19 % inside)** with `ARM 0.01` + `SLEW 20` (D6.8 / the D6 table). ⇒ the envelope and the pose are **coupled at the last 0.0004 rad**. **Confidence 0.93.**

### E9 — The 3-frame DRIFT_PROXY gate inherits a pose-spring linearity break

`DRIFT_PROXY_WINDOW = 6` with an **endpoint-mean chord** residual bound assumes body yaw ≈ **linear** in time (`p1-turn-drift-green-smoke.html:134-146`). The pose spring makes the rendered yaw **second-order** (ζ = 0.35 ⇒ 31 % first overshoot), so the bound must be re-derived. The 3-frame engagement gate itself is trivially met (the hold proxy rises 4.6×, to 0.2266 rad). **Confidence 0.85** (fixture scout, line-verified).

### E10 — The tyre-follow tolerance is **structurally** invariant

`TIRE_YAW_FOLLOW_TOLERANCE_RAD = 0.002` (`:101`, used `:927-928`) measures the **relative** pose between the wheels and the body; the wheels are **parented onto the body matrix** (`car.pass.ts:1583-1598`), so the relative pose does not depend on the body's absolute yaw. The measured margin is `2.220446049250313e-16` — float32 matrix noise. **AC-078 PRESERVES.** **Confidence 0.95.**

### E11 — The `DriftModelState` byte-identity pin survives a field addition — but only with the hard zero

`drift.model.test.ts:325-327` is a **self-comparison** (`serializeDriftState(state) === serializeDriftState(createZeroDriftState())`), not a hardcoded shape. Adding `committedSlip`/`turnSign` to `DriftModelState` keeps the comparison meaningful **iff** the fresh zero state carries them at 0 **and** the straight-line run reaches them at exactly 0 — which is stage 5's job (E1b). **Confidence 0.90.**

### E12 — The collision bounce resets the slide but not the visual state

`collision.utils.ts:410,417-418` zeroes `speed` / `slipAngle` / `lateralVelocity` on a bounce but **not** `driftYawOffset`. Under the new model the pose is a **ringing spring** ⇒ after a bounce the pose would ring on with no slip to follow it, and the committed state would survive a wall hit. **A new bounce-reset contract is required.** **Confidence 0.90.**

### E13 — `computeDriftYawOffset` is a **dead twin**

`drift.model.ts:363` exports a function that production **inlines** at `physics.step.service.ts:417` (`nextDriftYawOffset = nextSlipAngle * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD`). The jest pin `computeDriftYawOffset(10,10) ≈ (π/4)·GAIN` to 10 digits (`drift.model.test.ts:331-334`) therefore pins a function **no production path calls**. **Confidence 0.95.**

### E14 — The `AC-083` "driving feel does not change" clause is falsified by 14N

`AC-083` (`plans.md:479`, and the 14M packet `:2557`): *"the existing 12 m lookahead and the 2.2 rad⁻¹ steer gain are **RETAINED so the driving feel does not change**."* Under 14N the **same** 2.2 gain now produces a **4.6× larger slide** (D3), because the gain now sets the *drift depth* through `committedSlip -= turnSign · ENTRY_RATE · |steer| · dt`. ⇒ the clause is no longer true as written. **Confidence 0.90.** (The *chain order* is already correct — 14N precedes 14M in `plans.md:275` — so this is a **wording/measurement** defect, not a sequencing defect.)

### E15 — The `AC-086` "without a second source of truth" promise is at risk from a *second* source of truth

`AC-086` promises the racing line + curvature speed profile are a **stable track-relative surface** shared by the Phase-2 assembler and scripted controller. 14N adds a **second** track-relative truth: the **curvature ⇒ achievable cornering speed** relation now depends on the drift model's scrub coefficient (10.0 m/s² instead of 24.2 m/s²), so the `AC-084` curvature-speed target tuned against the old scrub becomes **conservative** (the AI will under-corner). Two sources of truth for "how fast can this corner be taken" ⇒ the exact failure mode `AC-086` forbids. **Confidence 0.70.**

### E16 — `docs/browser-tests/scenarios/*.html` is **not in the Cortex corpus**

`cortex:load_document` on the fixture returns *"Document not found"*; the file was read natively (a permitted known-path fallback). **Confidence 0.99.** The 1998-document index is otherwise fresh (both `cortex-index` and `cortex-first-search` gates **PASS**).

---

## Decision

> **D0 — The re-baselining is a SHAPE change, not a VALUE change.**
> `AC-079` (`plans.md:474`) and `AC-089` (`plans.md:484`) are **contract-shaped**: *"the tires lose traction **proportionally**"*, *"the low-speed and straight-line behaviour stays undegraded"*, *"the determinism contract holds"*, *"the retained named checks stay true"*. A trajectory-changing model can satisfy every one of those **words** while breaking every **number**. Therefore the re-baselining must:
> **(a)** re-home the AC-079 *law* onto the new term (E6) — otherwise the AC is vacuous;
> **(b)** split AC-089's pin set into **PRESERVE / RE-BASELINE / RE-SCOPE** and require a **RE-BASELINE LEDGER** row per invalidated pin with a **MEASURED** value;
> **(c)** keep the **exact-zero** identities in the PRESERVE set — they are the only things that make "no regression" mean anything, and they survive **only** via stage 5 (E1b).

### D1 — The AC-079 / AC-089 re-baselining

#### D1.1 — The three-part pin verdict

| # | Pin | Owner | Verdict |
|---|---|---|---|
| **P1** | AC-079's proportional law (`computeLongitudinalTractionLoss`: monotonicity `:165-166`, >30 %-cap `:172-173`, zero-on-straight `:181-183`, speed-scaling `:190`) | `drift.model.test.ts` | **RE-HOME** onto the speed-scrub term. The export is retired (E6); the six pins are **re-authored against the scrub**, not deleted. |
| **P2** | AC-079's browser ceiling (`TRACTION_LOSS_MAX_SPEED_KMH = 290`, `:107`; `proportionalTractionLoss`, `:935-937`) | fixture | **RE-SEMANTIC** — the ceiling is **replaced** by a scrub-rate assertion (D1.3). The check *name* stays. |
| **P3** | AC-089's no-regression list (`AC-047A..D`, `AC-078`, `AC-079`, determinism, fixture global `passed`) | plan + tests | **RE-SPLIT** into the three sets in D1.2. |

#### D1.2 — The re-baselining ledger (jest + service)

Verdicts: **P** = preserve, **R** = re-baseline (value), **RH** = re-home (law), **S** = re-scope, **X** = delete-as-dead (with a re-authored successor).

| ID | Pin (file:line) | Current | Verdict | New baseline |
|---|---|---|---|---|
| J1 | `MAX_SLIP_ANGLE_RAD === π/2` (`drift.model.test.ts:70`) | π/2 | **R** | `0.45` (D2) — also un-breaks `car.pass.test.ts:371-373` (the limit re-derives itself) |
| J2 | `computeSlipAngle(10,10) ≈ π/4` / `\|·(100,10)\| < π/2` (`:147-148`) | atan2 | **P** | the *construction* is unchanged (the chase produces `vLat`; the angle is still `atan2(vLat, speed)`) |
| J3 | build-up: `\|slip\|>0`, `<MAX`, finite (`:206-210`) | — | **R** | `toBeLessThanOrEqual(0.45)` — the committed slip **converges asymptotically** to the clamp, so a strict `<` survives only on a thin margin (E1b) |
| J4 | AC-047A: `\|latVel\|>10`, `\|slip\|>0.2`, `<MAX` (`:271-275`) | — | **P (verify)** | new hold slip 0.4528 rad @89 m/s and lateral ≈12.0 m/s @25 m/s ⇒ **the thresholds still hold** — the one AC-047A pin that survives untouched. **Measure, do not assume.** |
| J5 | service `\|slip\|<π/2` **literal** (`physics.step.service.test.ts:144`) | π/2 literal | **R** | import `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` (already imported at `:44`) instead of the literal |
| J6 | spawn `driftYawOffset === 0` (`:158`) | 0 | **P** | the pose initialises to 0 |
| J7 | `\|driftYawOffset\|>0`, `<π/2` under sustained steering (`:161-163`) | π/2 | **R** | the **derived** `DRIFT_VISUAL_YAW_LIMIT_RAD = 0.2250`; the pose overshoot must be shown to stay inside (0.2246 with the slew) |
| J8 | AC-079 `\|slip\|>0.2`, `<MAX` (`:240-244`) | — | **R** | `<= 0.45` |
| J9 | straight-line `x > spawnX` after 10 ticks (`:211`) | — | **P** | the hard zero keeps `slip` exactly 0 |
| J10 | **straight-line `speed === 89` EXACTLY and `slipAngle === 0` EXACTLY** after 600 ticks (`:250-251`) | exact | **P (highest risk)** | survives **only** via stage 5 (E1b) + the latch re-arm rule (E1a). **The RED slice MUST pin stage 5 explicitly.** |
| J11 | **low-speed `lateralVelocity === 0` EXACTLY and `slipAngle === 0` EXACTLY** (10 m/s, `steer: 1`, 120 ticks) (`:264-265`) | exact | **R — and D1 needs a model change** | D1 has **no `MIN_SPEED_MPS` gate** (E1c) ⇒ **this pin FAILS unless stage (1)/(3) are speed-gated.** The gate is a **model requirement**, not a test change. |
| J12 | determinism twin-run (drift) (`:251`) | — | **P** | the model stays pure |
| J13 | determinism twin-run (service, `PhysicsCarState` JSON) (`:135`) | — | **P** | the new state types ride on `PhysicsCarState` ⇒ covered by the existing twin-run |
| J14 | **AC-047D byte-identity to `createZeroDriftState()`** (`:325-327`) | exact | **P (conditional)** | survives a field addition (E11) **iff** the fresh zero state carries the new fields at 0 **and** stage 5 fires |
| J15 | `computeLongitudinalTractionLoss(0,60,0.67) === 0` (`:181`) | exact | **RH** | re-anchored on the scrub term's zero-on-a-straight identity |
| J16 | traction monotonicity in slip and in speed demand (`:165-166`) | — | **RH** | onto the scrub term |
| J17 | traction beats the old 30 % cap (`:172-173`) | `< 0.7` | **RH** | the new budget `1/(1+2·max(0,\|slip\|−0.20))` gives 0.4545 at slip 0.8 — the *property* survives, the expression changes |
| J18 | `'NEATRACING_DRIFT_MIN_GRIP_FACTOR' in driftModel === false` and `'computeGripFalloff' in driftModel === false` (`:155-156`) | — | **P + extend** | the new `gripFactor` budget must not resurrect a floor; **add the new export names to the forbidden set** |
| J19 | `computeCountersteerDelta` sign contract (`:141-143`) | — | **X** | the export is retired; its sign contract is the **C5 mirroring root cause** (D1 of the arcade model) — it must be **deleted and replaced**, not preserved |
| J20 | `computeDriftYawOffset(10,10) ≈ (π/4)·GAIN` (`:331-334`) and `(10,0) === 0` (`:335`) | — | **X → RH** | the dead twin (E13) is either **wired as the single source of the offset formula** or **deleted** (no deferred cleanup); the pin is re-authored against the pose spring either way |
| J21 | `computeTractionFactor(0,1)===1` / `(THRESHOLD,1)===1` / `(0.75·MAX,1)∈(0,1)` (`:119-122`) | — | **R** | the threshold moves 0.25 → **0.20**; the "= 1 below threshold" property survives in the new budget form |
| J22 | `createZeroDriftState` → slip 0, lateral 0 (`:90-91`) | — | **P + extend** | the new fields must be 0 here (J14) |
| J23 | `computeOutwardPush` zero at standstill / below `MIN_SPEED` (`:103-105`) | — | **P** | the outward push lives in the **service** (`CENTRIFUGAL_MULTIPLIER = 1.5`), not in the retired drift-model terms |
| J24 | `computeSlipAngle(5,0) === 0` (`:149`) | — | **P** | — |
| J25 | `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER === 1.5` (`physics.step.service.test.ts:125`) | 1.5 | **P** | not re-tuned by 14N (its *effect* changes because the slip changes) |
| J26 | countersteer recovery: peak>0, recovered `\|slip\|<peak`, `\|latVel\|<5` (`:225,:235,:306`) | — | **P (strengthened)** | the new model **unwinds and hard-zeros** instead of mirroring ⇒ the recovery is *cleaner* than the current behaviour |
| J27 | `computeSlipAngle` / `computeTractionFactor` import block (`:23-41`) | 8+8 | **R** | the export list changes; the file header (`:1-19`, `:10-15`) documents the green contract and must be updated in the same slice |

#### D1.3 — The AC-079 browser check must change **semantics**, not just value

Today: `proportionalTractionLoss = tireYawDriftEngaged && 0 < maxSpeedKmhDrift < 290` (`:935-937`) — a **ceiling only** (C10). It asserts *nothing about proportionality*; the proportionality lives only in the jest law that 14N retires (E6). Meanwhile the 290 ceiling is about to **invert** (E7).

**Decision:** replace the ceiling with a **scrub-rate / speed-deficit assertion** — the observable that *is* AC-079's content ("a full-speed turn MUST reduce the effective grip proportionally to the drift demand"), and which is **scale-free** (it survives any future change to the scrub coefficient):

```
proportionalTractionLoss =
      tireYawDriftEngaged
  && driftSpeedDeficitKmh >= NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH   // NEW named constant
  && maxSpeedKmhDrift     <= TRACTION_LOSS_MAX_SPEED_KMH              // re-pinned, no longer binding
```
where `driftSpeedDeficitKmh` = (the straight-phase speed − the drift-phase speed) at comparable points, and `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH` is a **new named export** whose **value is the RED slice's decision, measured** (the old model bled 127 km/h at a 90 km/h entry; the new model scrubs 10.0 m/s² against a 17.8 m/s² engine ⇒ the deficit shrinks by ~4×, so the new floor is **substantially lower** than today's accidental margin and must be re-measured, not carried).
`TRACTION_LOSS_MAX_SPEED_KMH` is re-pinned **above the scenario's top speed** (so it stops being the binding constraint) and the *number* is the RED slice's decision.

#### D1.4 — The AC-089 re-baselining text (for `01-planning` to author)

> **AC-089 (re-baselined).** NO REGRESSION, **re-baselined**: the *contracts* stay green — the low-speed and straight-line behaviour is undegraded, the determinism contract holds, the fixture global `passed` stays true — while the *baselines* that a trajectory change necessarily moves are re-measured and **recorded in a RE-BASELINE LEDGER** (one row per invalidated pin: pin, old baseline, new baseline, measured value, owner file). The **PRESERVE SET** (bit-identical, zero delta allowed) is: the AC-047D byte-identity, the determinism twin-runs, the straight-line exact zeros (`speed === MAX`, `slipAngle === 0`, `lateralVelocity === 0`, the zero traction loss), the low-speed gate (no drift below 15 m/s), AC-078's tyre-follow tolerance, and the 24 non-drift fixture checks. The **RE-BASELINED SET** (value may move, delta recorded) is: AC-079's law and ceiling, the DRIFT_PROXY gates, the AC-047A slip band, the AC-047C yaw envelope, and the 5× trajectory metrics. **No silent deletion:** every retired export and every retired test is re-authored in the same slice.

#### D1.5 — Two **model** requirements that fall out of the re-baselining (not test changes)

1. **The `MIN_SPEED_MPS = 15` gate must be inside the committed model** (stages 1 and 3), not merely a constant in the file. Without it the low-speed exact-zero pin J11 fails (E1c) and the arcade behaviour breaks below 54 km/h.
2. **Stage 5's hard zero must be an explicit, separately-named contract** — e.g. `NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD` (already named) — and the RED slice must pin it as *"after the hard zero, `committedSlip`, `vLat` and the pose are **exactly** 0"*. Every exact-zero pin in the PRESERVE set depends on it (E1b).

### D2 — The AI steer-seam impact (the Phase-2 shared seam)

#### D2.1 — The core hazard

`sign(steer)` becomes a **discrete intent bit** (E1), but **both** AI producers are **slip-blind**:
* the Phase-1 autopilot reads only `{x, y, heading, speed}` (`ai.autopilot.ts:57-60`) — it never sees `slipAngle` or `lateralVelocity`;
* the Phase-2 scripted controller runs in a simulator with **no lateral-velocity model at all** (E3);
* the future Phase-2 NEAT controllers will emit a **continuous** neural steer whose sign flips are noise-driven.

⇒ **A routine heading-error zero-crossing in an AI is, under 14N, an accidental countersteer command** that (i) halts the committed ramp, (ii) unwinds the committed slip at 2.0 rad/s, and (iii) applies up to `70 · |steer| · speedPercent · dt` of authority *against* the slide. Item (ii) and (iii) are **not** magnitude-gated in the same way as (i): the unwind is magnitude-**independent**.

#### D2.2 — The proposed **Phase-2-safe steer-seam contract** (3 clauses)

> **S1 — Intent must be explicit, never inferred from the sign of a continuous axis.** The countersteer trigger must read a **dedicated, named intent signal**, not `sign(steer)`. Because the human lane's steer is **digital** (`{−1,0,+1}`, E2), the human's intent is recoverable exactly; because the AI lanes' steer is **continuous**, theirs is not.
> **S2 — The intent signal must be producer-declared.** The human lane sets it from the key state; `stepAutopilotInput` and `computeScriptedControl` set it **explicitly to "no intent"** (never implicitly, by omission). ⇒ the AI's behaviour under 14N changes **only** through the intended trajectory change, never through an accidental intent.
> **S3 — The intent signal must be *held*, not instantaneous.** A minimum hold (a named tick/second constant) is required so a key-mash or a neural-noise flip cannot be read as intent. This also gives the `GF-15` 4 Hz steer-chatter stress (22.6° far-side swing, R13) a defined outcome instead of an undefined one.

**Implementation note for `01-planning`:** S1–S3 imply a **new field on `PhysicsStepInput`** (`physics.step.service.ts:59-65`) and therefore touch `host/input.ts`, `browser-entry.ts` and `ai.autopilot.ts` — **files that are not in the AC-079/AC-087/AC-089 file sets** (`plans.md:474,:482,:484`). ⇒ **a plan-level file-set amendment is required.** That is `01-planning`'s call, not this agent's; this artifact records the requirement.

#### D2.3 — The **fallback** if the plan cannot expand the file set

A **magnitude + duration gate inside the drift model**, with two **new named constants**:
`NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER` (≈0.9 — the human's digital ±1 satisfies it; an AI's zero-crossing does not) and `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS` (a debounce).
* **Trade-off, stated honestly:** a 4 Hz square-wave steer has a 125 ms half-period, so a hold longer than ~125 ms is needed to reject the `GF-15` chatter — but a human countersteer must feel responsive inside ~50 ms. **The window is narrow and the RED slice must MEASURE it.** The gate rejects the AI's *zero-crossing* (|steer| ≈ 0) but **not** the AI's *saturated* correction (|steer| = 1 at ≥ 26° of heading error, `ai.autopilot.ts:174`) — and under the new trajectory the AI saturates **more** often (E5). ⇒ the fallback is **weaker** than S1–S3 and must be recorded as such.
* **The latch re-arm rule is mandatory under either option:** `turnSign` must be released when the hard zero fires, so a later opposite-sign steer re-latches cleanly instead of being permanently classified as a countersteer (E1a). **This is a spec gap in D1 that Phase B must close — it is not optional.**

#### D2.4 — The Phase-2 impact ledger

| # | Impact | Severity | Confidence | Action |
|---|---|---|---|---|
| P-1 | The **steer-seam semantics** are overloaded with drift-intent for a producer that cannot express intent (E1, E2) — and that seam is the **shared** one `AC-086` names | **HIGH** | 0.95 | S1–S3 (or the D2.3 fallback) — **the headline Phase-2 risk** |
| P-2 | The Phase-1 autopilot's closed loop changes: 5× slide ⇒ bearing shift ⇒ binary throttle/brake modulation ⇒ `speedPercent` drop ⇒ **less** countersteer authority (E5) | MEDIUM | 0.55 | **MEASURE** in the RED slice (AI heading-error / throttle-duty statistics over a lap) before any gate constant is chosen |
| P-3 | The Phase-2 **observation normalisation** (channel 74 `slipAngleRadians / (π/2)`, lateral `/54`) is calibrated to the **old** slip range; the new `MAX_SLIP = 0.45` compresses the live signal to **±0.286** of the nominal range (E4) | MEDIUM (forward-looking) | 0.85 | re-pin the normalisers **before** any Phase-2 lateral-velocity model is adopted; record it in the `AC-027`/`AC-028` work items |
| P-4 | The Phase-2 lane has **no** drift model, so 14N does **not** change its dynamics directly (E3) | — (clarifier) | 0.97 | record in the tracker so the user's *"not sure if this will affect later AIs"* is answered precisely: **the seam yes, the dynamics not yet** |
| P-5 | `AC-083`'s clause *"the 2.2 gain is RETAINED so the driving feel does not change"* is **falsified** by 14N (E14) | MEDIUM | 0.90 | `01-planning` re-words the clause to *"the gain is RETAINED so the **steer authority** does not change; the **slide** is deliberately sharpened by 14N"* |
| P-6 | `AC-084`'s curvature-speed target was tuned against the **old** scrub (24.2 m/s²) and becomes **conservative** under the new one (10.0 m/s²) ⇒ the AI will under-corner (E15) | MEDIUM | 0.70 | the 14M RED slice must re-take the corner-entry speed baseline **after** 14N closes |
| P-7 | The 14M **chain order is already correct** (14N precedes 14M, `plans.md:275`) | — (clarifier) | 0.95 | no sequencing change needed — record it so it is not re-litigated |
| P-8 | Step 14I's road-containment work (AC-081) inherits a **worse** AI line from the 5× slide ⇒ 14I's scope grows | LOW | 0.60 | note in the 14I packet |
| P-9 | The `GF-15` 4 Hz chatter stress (R13) is a **latch** property and is **not** fixed by the pose channel ⇒ it lands squarely on the AI seam (D2.2/S3) | MEDIUM | 0.80 | the RED slice must decide the intent-hold constant **with the chatter stress in the acceptance set** |

### D3 — The fixture threshold updates

`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html` (965 lines).

#### D3.1 — The threshold ledger

| Line | Constant / check | Current | Verdict | Proposed |
|---|---|---|---|---|
| 101 | `TIRE_YAW_FOLLOW_TOLERANCE_RAD` | 0.002 | **PRESERVE** | — (E10: structurally invariant; the relative pose is parented) |
| 107 | `TRACTION_LOSS_MAX_SPEED_KMH` | 290 | **RE-BASELINE (semantic + value)** | re-pin **above** the scenario's top speed so it stops being the binding constraint (E7); the *value* is the RED slice's decision. **Add** `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH` (D1.3) |
| 110 | `DRIFT_MIN_SPEED_KMH` | 54 | **PRESERVE** | — (15 m/s, unchanged; the operating point 90 km/h is unchanged) |
| 114 / 117 | `ACCEL_TARGET_SPEED_KMH` / `ACCEL_MAX_FRAMES` | 90 / 600 | **PRESERVE** | — (scenario scaffolding) |
| 121 | `TRACK_LIT_FLOATS_PER_VERTEX` | 7 | **PRESERVE** | — (model-independent) |
| 128 | `DRIFT_PROXY_MIN_RAD` | 0.01 | **RE-BASELINE (relative, not absolute)** | **derive** as `0.25 × NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD` (the AC-087 authority) so the fixture and the unit live in **one** system — the Phase-A MERGE's item 2 applied to the fixture. The *coefficient* is the RED slice's decision. |
| 132 | `DRIFT_PROXY_MIN_FRAMES` | 3 | **RE-BASELINE (split)** | keep **3** for **engagement** (the hold proxy rises 4.6×, so 3 is trivially met) and **add** `DRIFT_PROXY_MIN_HOLD_FRAMES` for the AC-087 **sustain** (the new hold, not the new entry) — value = RED decision |
| 146 | `DRIFT_PROXY_WINDOW` | 6 | **RE-BASELINE the bound, keep the window** | the 6-frame endpoint-mean chord residual (`:134-146`) assumes body yaw ≈ **linear**; the pose spring makes it **second-order** (ζ = 0.35 ⇒ 31 % overshoot) ⇒ the residual bound must be **re-derived** (E9). Keep 6 unless the derivation says otherwise |
| 919 | `wheelTireShading` | shader-source + apply pattern | **RE-SCOPE (AC-088)** | capture the `uTireGradient*` uniforms through the fixture's existing `uniform1f` interception hook and assert a **non-degenerate** `[dark, bright]` factor spread — the VBO vertex-colour read is **blind by construction** (`upperAvg == lowerAvg == 0.28565`) |
| 926-928 | `tireYawFollowsBodyDuringDrift` | tol 0.002 | **PRESERVE** | — |
| 935-937 | `proportionalTractionLoss` | `0 < max < 290` | **RE-SEMANTIC** | D1.3 — a **scrub-rate / speed-deficit** assertion, not a ceiling |
| — | **NEW `driftFeelSlideProminence`** | absent | **ADD** | the AC-087 **floor** is pinned on the **deterministic jest unit proxy**; the browser proxy spread (0.0494 / 0.0603 / 0.3599 rad) is **noise**, not an anchor. The fixture carries only a **looser observable** (a *presence* check), never the floor |
| 922-923 | `speedUnderW`, `driftTurnWA` | — | **PRESERVE** | — (a larger slide only strengthens them) |
| 893-921 | the 24 infra / track / shadow / steer checks | — | **PRESERVE** | — |
| 342-371 | the track-paint constants | — | **PRESERVE** | — (unrelated surface) |

#### D3.2 — The fixture's **own** re-baselining obligations

* **F-1 — the 3-frame engagement gate must be re-derived, not preserved** (`R1` in `drift-feel.research.md:575` names it). Under the new model the *entry* is 4.5× faster (0.10 s vs 0.25 s) **and** the *exit* is abrupt (the hard zero) ⇒ the "sustained drift" detector must tolerate both. The **hold** is the new invariant; the **entry** is no longer a difficulty.
* **F-2 — the residual bound of the 6-frame window** must be re-derived for the second-order pose (E9). This is the **one** fixture threshold whose failure mode is *silent* (a bound that is too loose does not fail; it just stops protecting).
* **F-3 — a new bounce/scatter scenario is needed**: the pose is a ringing spring and `collision.utils.ts:410,417-418` does not reset it (E12) ⇒ the fixture needs a "bounce ⇒ pose resets, no ring-out" case, and the jest suite needs the same contract.
* **F-4 — the fixture must NOT re-pin the AC-087 floor to a browser proxy number** (Phase-A MERGE item 2, `plans.md:8294`). The floor belongs to the **unit** proxy; the browser is a *presence* check only.

---

## Risks

| # | Risk | Severity | Evidence | Mitigation / owner |
|---|---|---|---|---|
| **RB-1** | **The latch re-arm rule is unspecified in D1** (E1a). Every exact-zero pin (J10, J11, J14) and the hard zero itself depend on it. | **HIGH** | D1 stage (0) | **Phase B must specify it**; the RED slice pins `turnSign` release on the hard zero. **Blocking for `P1-drift-feel-red`.** |
| **RB-2** | **D1 has no `MIN_SPEED_MPS` gate** (E1c) ⇒ the low-speed exact-zero pin J11 fails and the arcade breaks below 54 km/h. | **HIGH** | D1 stages (1)/(3) | the gate must be **inside** the model (D1.5); not a test change |
| **RB-3** | **AC-079's law is retired** (E6) ⇒ AC-079 becomes vacuous unless the law is re-homed onto the scrub term. | **HIGH** | D2 "Constants that this model RETIRES" | D1.3's scrub-rate assertion + D1.4's "no silent deletion" clause |
| **RB-4** | **The 290 km/h ceiling inverts** (E7) ⇒ `proportionalTractionLoss` **FAILS**, not drifts. | **HIGH** | D3 (10.0 vs 17.8 m/s²) | D1.3 — replace the semantics; the RED slice re-pins the number |
| **RB-5** | **`sign(steer)` becomes a discrete intent bit** for a slip-blind AI producer (E1, E2). | **HIGH** | `ai.autopilot.ts:174`; `input.ts:89-91` | D2.2 S1–S3; needs a **plan-level file-set amendment** |
| **RB-6** | The **AI's saturated** steer (|steer| = 1 at ≥ 26° error) still satisfies the D2.3 fallback gate ⇒ the fallback does **not** fully protect the AI. | MEDIUM | `ai.autopilot.ts:174` | prefer S1–S3; if the fallback is used, **record the residual explicitly** |
| **RB-7** | The `GF-15` 4 Hz chatter (22.6° far-side swing) is a **latch** property the pose channel cannot fix (R13) — it lands on the AI seam. | MEDIUM | R13 | the intent-hold constant must be decided **with the chatter stress in the acceptance set** |
| **RB-8** | The **yaw envelope** and the pose overshoot are coupled at the last 0.0004 rad (E8) — a 0.19 % margin is not a safety margin. | MEDIUM | D6.8 | the RED slice must pin the **max per-tick pose-target step (≤ 2.13°)** *and* the far-side pop (≈ 2.2°) *and* the settle (550 ms) **together**, never the hold pose alone (R11) |
| **RB-9** | The **collision bounce** does not reset the committed/pose state (E12) ⇒ a wall hit leaves a ringing pose and a surviving latch. | MEDIUM | `collision.utils.ts:410,417-418` | a new bounce-reset contract in both the jest and the fixture lanes (F-3) |
| **RB-10** | The **AI's throttle/brake duty** changes under the new trajectory (E5) — a **positive feedback** through `speedPercent`. | MEDIUM | E5 (first-order) | **MEASURE** in the RED slice; confidence 0.55 ⇒ it is *not* asserted as fact here |
| **RB-11** | `AC-083`'s *"the driving feel does not change"* clause is falsified (E14) and `AC-084`'s corner-speed target becomes conservative (E15). | MEDIUM | E14, E15 | `01-planning` re-words AC-083; the 14M RED slice re-takes the baseline after 14N |
| **RB-12** | The Phase-2 **observation normalisers** (channel 74, `/54`) are mis-calibrated for the new slip range if Phase 2 ever adopts a lateral-velocity model (E4). | MEDIUM (forward-looking) | E3, E4 | record in the `AC-027`/`AC-028` work items **now**, before the model is adopted |
| **RB-13** | The `DRIFT_PROXY_WINDOW` residual bound fails **silently** if it is merely loosened (E9). | MEDIUM | E9 | re-derive it, and **pin the derivation** in the fixture comment |
| **RB-14** | The `docs/browser-tests/scenarios/*.html` family is **not in the Cortex corpus** (E16) — a retrieval blind spot for every future agent on this step. | LOW | `load_document` → "Document not found" | raise with `00-helping` / `repo-cortex-scout` — index the browser-test scenario family |
| **RB-15** | `computeDriftYawOffset` is a **dead twin** whose jest pin looks like a live contract (E13) ⇒ a reader can believe the offset formula is pinned when production inlines it. | LOW | E13 | wire-or-delete in the same slice (no deferred cleanup) |
| **RB-16** | The `computeCountersteerDelta` sign contract (J19) is the **C5 mirroring root cause**; if the export survives "for compatibility", the mirroring returns. | LOW | D1 of the arcade model | delete it; keep the no-resurrected-floor style guard (J18) |

---

## Handoff

* **To `01-planning` (Phase-B merge):** this file is the **re-baselining + coupling-risk input** for the Step 14N slice packets. The two **plan-level amendments** it implies are (a) an **AC-089 re-wording** (D1.4) and (b) a **file-set amendment** for the steer-intent channel (D2.2) or the in-model gate constants (D2.3). Both are `01-planning` decisions; this artifact only records the requirement and the evidence.
* **To `03-red-testing` (`P1-drift-feel-red`):** the ledger in **D1.2** is the RED re-authoring checklist; **D1.5** and **RB-1/RB-2** are the two **model** requirements that must be settled **before** any RED contract can be written; **D3.1** is the fixture-side contract list; **RB-10** is the one measurement this artifact could not make (AI heading-error / throttle-duty statistics over a lap).
* **To `03-red-testing` / `04-implementing`:** the **new named exports** proposed by this artifact — `NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH` (D1.3), `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER` and `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS` (D2.3, fallback only) — and the **latch re-arm** rule (RB-1) are the new public surface.
* **To `04-implementing`:** the **bounce-reset** contract (RB-9) must cover the committed state **and** the pose state, not only the legacy three fields.
* **Unchanged by this artifact:** the Phase-A MERGE's selected mix, the constant set (D2 of the arcade artifact + the D6 game-feel amendment), the one-accumulator rule, the ARM 0.01 / SLEW 20 values, and the file ownership of the visual channel.

### Certainty statement

| Claim | Confidence | Basis |
|---|---|---|
| `sign(steer)` becomes a discrete intent bit under the 7-stage model | **0.95** | D1 as written + the two producers' source reads |
| The latch re-arm rule is unspecified in D1 | **0.95** | D1 read as written |
| The hard zero (stage 5) is load-bearing for the exact-zero pins | **0.92** | stage 3 is exponential; only stage 5 snaps to exact 0 |
| D1 lacks a `MIN_SPEED_MPS` gate ⇒ the low-speed pin fails | **0.90** | D1 stages (1)/(3) + `physics.step.service.test.ts:264-265` |
| AC-079's `computeLongitudinalTractionLoss` is retired ⇒ the AC is vacuous unless re-homed | **0.95** | arcade D2 "Constants that this model RETIRES" |
| The 290 km/h ceiling inverts and the check **fails** | **0.78** | D3 scrub 10.0 m/s² vs engine 17.8 m/s² + the recorded 127 km/h drift-phase peak |
| The Phase-2 lane has no drift model and no lateral-velocity model | **0.97** | `environment.step.service.ts:808`; `environment.types.ts:101-102`; zero cross-imports |
| Phase-2 channel-74 normalisation compresses 3.5× | **0.85** | `observation.assembler.ts:945` vs `MAX_SLIP = 0.45` |
| AC-078's tyre-follow tolerance is structurally invariant | **0.95** | the wheels are parented onto the body matrix |
| The yaw envelope shrinks to 0.2250 and the pose reaches 0.2246 | **0.93** | R3, D6.8, the D6 table |
| `AC-083`'s "driving feel does not change" is falsified by 14N | **0.90** | D3's 4.6× proxy + the AC-083 wording |
| The AI's throttle/brake duty changes under the new trajectory | **0.55** | first-order bearing estimate; **not simulated** ⇒ MEASURE |
| `AC-084`'s corner-speed target becomes conservative | **0.70** | D3's scrub figures vs the old tuning |

### Provenance and method

* **Cortex-first:** `cortex-index` gate **PASS** (1998 docs, all 9 families fresh, corpus alive) and `cortex-first-search` gate **PASS** (27452 chunks, 3 corpus results) — both run before the native reads.
* **Cortex:** `cortex:load_document` for the plan (52.3 KB, truncated to a temp file) and for the two Phase-A research artifacts; `get_slice_context` for slice `14N`; `search_corpus` / `search_context` for the coupling and drift-feel artifacts.
* **Native fallback (permitted, and recorded):** the plan (1.49 MB) and the fixture (`docs/browser-tests/scenarios/*.html`, **not in the corpus** — E16) were read with `grep` / `view` after the corpus returned zero or "Document not found".
* **Delegated (3 read-only scouts, non-overlapping file sets, parallel):** `boundary-mapper` (the AI steer seam — `ai.autopilot.ts`, `scripted.controller.ts`, `observation.assembler.ts`, `drift.model.ts`, `physics.step.service.ts`; it also **corrected** the packet path: `examples/neatRacing/ai/observation.assembler.ts` **does not exist** — the implemented analog is `examples/racing_curriculum/controller/observation.assembler.ts`), `implementation-pattern-scout` (the fixture threshold surface), `implementation-pattern-scout` (the jest RED-contract surface). All three returned `structured-v1` reports with line-verified anchors; **no files were changed by any of them.**
* **Direct primary reads (orchestrator):** `ai.autopilot.ts` (full), `host/input.ts` (the steer axis), `scripted.controller.ts` (the gain and the steer), `environment/environment.step.service.ts:805-820`, `environment/environment.types.ts:98-104`, `physics.step.service.ts:53-65`, `drift-feel.research.md` (D1, D2, D3, D6, the risk table, the certainty table), `neatracing-voxel-racer.plans.md` (`AC-079/087/088/089/090`, `AC-081..086`, the 14M packet, the Phase-A MERGE block).
* **Conflict resolution:** the `observation.assembler.ts` path in the dispatch packet was **contradicted by the file tree** and by the boundary map; the **file tree won** (runtime/static > the packet's assumption). The 0.01 rad engagement threshold's re-baseline status is the one **medium-confidence** fixture marking (R1's literal words name only the 3-frame gate; the 0.01 rad reading is inferred as the same gate family) — flagged in the fixture scout's report and **not** asserted as fact here.

### Validation evidence (this agent)

| Check | Result |
|---|---|
| `run_gate_check(cortex-index)` | **PASS** — 1998 documents, 9/9 families fresh, `corpus_mcp_alive: true` |
| `run_gate_check(cortex-first-search)` | **PASS** — 27452 chunks, index fresh, 3 corpus results |
| `run_gate_check(slice-advancement, slice-id=14N, changed-files=plans/neatracing-voxel-racer.14N-rebaseline-risk.md)` | **PASS** — all 4 sub-gates green: `plan-sync`, `step-packet`, `plan-slice-quality`, `plan-command-lint` |
| `node .github/hooks/workflow-update-sync.mjs --plan=plans/neatracing-voxel-racer.plans.md --json --dry-run` | `ok: true`, `pass: true`, but **`currentWipStep: null`** and `actionTaken: "between-steps"` — **while Step 14N is the single `[WIP]` step** (`plans.md:275`) |

**The hook's degradation is a real defect, not a no-op.** The step-id pattern in `.github/hooks/workflow-update-sync.mjs` does not match a letter-suffixed step id such as `14N`, so the hook cannot resolve the WIP step and reports *"between-steps"* — **silently**, with `pass: true`. The gate still passes (it validates the *plan's* registration, not the hook's resolution), so the failure is invisible to the gate layer. Consequences for this step:

1. The hook **cannot** advance or record the 14N step state ⇒ the step transition must be driven **manually** by the orchestrator.
2. The hook was therefore run in **`--dry-run`** mode: it calls `fs.writeFileSync` (`:164`), and this agent's charter forbids plan writes.
3. **Out of this agent's edit scope** — raised here for `00-helping` / `agent-maintenance-coordinator` as a **workflow-gap** ticket, not fixed here. It affects **every** letter-suffixed step id in the repo, not only 14N.
