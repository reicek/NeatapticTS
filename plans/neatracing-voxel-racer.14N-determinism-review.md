## DRIFT CHANNEL — REPLAY STABILITY / ORDERING-DRIFT REVIEW (read‑only)

**Verdict: REQUEST_CHANGES.** The published‑slip *peak value* is a genuine model property (not a tick‑budget artifact) — but the **browser fixture's drift‑feel lane is not replay‑stable at all**: its frame‑counted phases, its off‑road excursion, and the off‑road speed‑floor break make the *check verdicts* flip with the frame rate, and the browser's own 0.47 bound is violated by the browser's own run.

All numbers below are Node reconstructions of the **production** sources (`drift.model.ts` + `physics.step.service.ts` + `render-loop.utils.ts` + `collision.utils.ts` + `track.generator.ts`) driven through the production `advanceFrameSteps`; harness validated bit‑identically against the jest lane (0.45041978773870417 / 0.4511617369362305@t42 / 79.85792723069594 / 32.55473 km/h ✓). Scripts: `C:\Users\reice\AppData\Local\Temp\drift-probe\probe{,2,3,4,5,6}.ts` → `probe*.cjs` (all **outside** the repo). No repo file was modified; no mutating git command was run.

---

## S1 — TICK‑BUDGET DIVERGENCE

### 1.1 What a 600‑frame phase actually costs in ticks (fixture phase sequence, real track + real surface resolver)

| fps | frame Δ | ticks / 600‑frame phase | simulated time of that phase | whole fixture sequence |
|---|---|---|---|---|
| 60 | 16.67 ms | **600** | 10.00 s | 2342 ticks (39.0 s) |
| 90 | 11.11 ms | **400** | 6.67 s | 1602 ticks (26.7 s) |
| 120 | 8.33 ms | **300** | 5.00 s | 1230 ticks (20.5 s) |
| 144 | 6.94 ms | **250** | 4.17 s | 1044 ticks (17.4 s) |
| 163 | 6.13 ms | **221** | 3.68 s | 936 ticks (15.6 s) |

⇒ a **2.7× spread** in the tick budget (and in the simulated duration) of the *same* fixture phase. Your premise ("a frame‑counted phase has a machine‑dependent tick count") is **correct**.

### 1.2 THE MEASUREMENT TABLE — tick budget → peak |published slip|

**(a) ROAD‑only (the jest‑lane condition): W+A from the top speed, E entry ticks, then 180 neutral ticks**

| E (entry ticks) | peak \|slip\| | peak @ absolute tick | end‑of‑entry \|slip\| | end‑of‑hold \|slip\| | speed end |
|---|---|---|---|---|---|
| 1 | 0.0583333333 | 97 | 0.0180739570 | 0.0583333333 | 88.95 |
| 2 | 0.1166666667 | 98 | 0.0487260879 | 0.1166666667 | 88.90 |
| 3 | 0.1750000000 | 99 | 0.0881925069 | 0.1750000000 | 88.86 |
| 4 | 0.2333333333 | 105 | 0.1314865848 | 0.2333333333 | 88.81 |
| 5 | 0.2916666667 | 117 | 0.1756345272 | 0.2916666667 | 88.76 |
| 6 | 0.3500000000 | 126 | 0.2204129951 | 0.3500000000 | 88.72 |
| **8 … 600** | **0.4511617369** | **42** (always) | 0.3083 → 0.4500 | 0.45039 → 0.45000 | 82.15 → 79.88 |

Warm prefix (W ticks of straight) shifts **only the timestamp** of the peak, never the value: W = 0/30/120/288/600 → peak @ tick 42/72/162/330/**642**, value 0.4511617369 in every case.

**(b) The browser lane (real track, grass live) — the fixture’s own phase sequence**

| fps | P1‑WARM ticks | P1 ticks | P2 ticks | P1‑WARM peak | P1 peak | P1 last slip | P1 capture km/h | P2 peak | `moveSlip` (frame) | `moveSlip` (tick) | commitment release tick |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60 | 120 | 600 | 600 | 0.5039547685 | 0 | 0 | 0 | 0 | **0.0000000000** | 0 | 280 |
| 90 | 80 | 400 | 400 | 0.4811532814 | 0.5038379751 | 0 | 0 | 0 | **0.5038379751** | 0.5038379751 | 307 |
| 120 | 60 | 300 | 300 | 0.4605269008 | 0.5038098516 | 0 | 0 | 0 | **0.5038098516** | 0.5038098516 | 461 |
| 144 | 50 | 250 | 250 | 0.4608263200 | 0.5039560400 | 0 | 6.8978 | 0 | **0.5039560400** | 0.5039560400 | 437 |
| 163 | 44 | 221 | 221 | 0.4614021774 | 0.4984153760 | −0.4984153760 | 59.954448 | 0.5039413437 | **0.5039413437** | 0.5039413437 | 430 |

**(c) ROAD‑only contrast (same sequence, road resolver)** — `moveSlip` = **0.4511617369 at every frame rate** ✓ (the published peak *is* frame‑rate independent on the road), but the capture pair is not:

| fps | P1 capture km/h | P2 capture km/h | AC‑079 deficit km/h (floor 20) | verdict |
|---|---|---|---|---|
| 60 | 286.751 | 287.477 | −0.726 | **FAIL** |
| 90 | 281.935 | 287.212 | −5.277 | **FAIL** |
| 120 | 272.327 | 285.876 | −13.549 | **FAIL** |
| 144 | 262.566 | 283.571 | −21.005 | PASS |
| 163 | 254.079 | 280.878 | −26.799 | PASS |

### 1.3 ANSWER to S1

* **The published slip does NOT overshoot its clamp differently at different tick budgets on the road.** For every entry budget ≥ 8 ticks the peak is **exactly 0.4511617369 rad**, always at absolute tick 42 — a model property (the chase lag vs. the shrinking target: at t42 `vLat/target = 1.00296592`, `tan(peak)/tan(0.45) = 1.00296783`), **not** a replay‑ordering artifact. The budget only (i) truncates the transient for E ≤ 6 (the peak is then monotone in E: 0.0583·E) and (ii) shifts the peak's timestamp.
* **But the browser lane is nevertheless not replay‑stable**, for two *different* reasons than the one you suspected:
  1. **Off‑road pollution + a broken off‑road floor.** The car leaves the road, and `integrateLongitudinalSpeed` (`physics.step.service.ts:510‑544`) applies the grass floor **before** the scrub (`physics.step.service.ts:415‑427`) while gating it on the *tick‑start* speed. The scrub therefore pushes the tick‑end speed under 22.25 and the floor never re‑arms: **measured — the floor breaks at t=116, the car is at 0 m/s by t≈181, and the MIN_SPEED gate releases the commitment at t=133.** With *no* slide (no scrub) the floor holds at exactly 22.25 for 400 ticks — which is why the existing jest test (`physics.step.service.test.ts:352`, `throttle: 0`) never sees the bug.
  2. **Consequence: the car DIES at a frame‑rate‑dependent tick** (release tick 280/307/461/437/430), so the peak lands in a **different phase** at each rate (P1‑WARM @60, P1 @90–144, P2 @163) and the later phases measure a stopped car. The browser's own checks therefore **flip verdict with the frame rate**:
     * `driftFeelModelIdentity` (fixture `:1798‑1800`, `moveSlip.maxRad ≤ 0.45+0.02 = 0.47`): **PASS at 60 fps** (reads 0.0 — the car died before P1) → **FAIL at 90/120/144/163 fps** (0.5038–0.5040).
     * `driftFeelZeroStraightLineYaw` (fixture `:1750`, `straightPose.maxRad === 0`): **PASS at 60–144 fps** (the pose snapped to 0 *because the commitment was released by the MIN_SPEED gate on a dead car*) → **FAIL at 163 fps** (the car is still alive, pose ≈ 0.25 rad).
     * `proportionalTractionLoss` (fixture `:1703‑1706`, deficit ≥ 20 km/h): **FAIL at 60/90/120/144 fps** (0 → 6.90 km/h) → **PASS at 163 fps** (59.95 km/h). In the road‑only lane it flips the other way (FAIL at 60–120, PASS at 144–163).
* **Your 0.5038 is reproduced exactly** (0.503837975136 @ 90 fps) — but it is an **off‑road artifact**, not a tick‑budget artifact: it is present at *every* frame rate (0.4984–0.5040, a 1.1 % spread) because the published slip = `atan2(vLat, speed_post_scrub)` has **no clamp of its own** and the off‑road speed collapse outruns the lateral‑velocity decay (ratio 1.1372 vs 1.0030 on the road).

---

## S2 — DECLARED vs MEASURED ENVELOPE

**True worst case over the reachable input space** (surface × entry speed 15.5–89 × steer 0.5–1.0 × entry budget 1–600, 630 runs/regime):

| regime | worst published \|slip\| | where | declared 0.452 | browser 0.47 |
|---|---|---|---|---|
| **road only** | **0.4511617369** | 89 m/s, steer 1, E ≥ 8 (peak @ t42) | **holds** (headroom 0.00084) | holds |
| **grass (live in the browser)** | **0.5039742512** | 89 m/s, steer 0.75, E=12 (peak @ t132) | **VIOLATED** (+0.0520) | **VIOLATED** (+0.0340) |

* **Verdict: the 0.002 tolerance does NOT hold over the reachable space.** It holds *only* for the on‑road, top‑speed W+A scenario it was calibrated on. The whole 30–89 m/s grass band gives 0.5038–0.5040, i.e. the browser's measurement is **reproducible and stable in magnitude** (0.5038 @90 fps reproduces your number to 10 digits) — it is the *envelope* that is wrong, not the measurement.
* **Mechanism at the peak (t132, grass):** tick‑start speed 15.280 m/s (just above the MIN_SPEED gate 15), post‑scrub speed 14.824, `vLat = 8.394`, `target = tan(0.45)·15.28 = 7.381` → `vLat/target = 1.1372`, `tan(peak)/tan(0.45) = 1.1413` → published = 0.5038. The MIN_SPEED gate releases the commitment on the *next* tick, which is what caps the excursion at ≈0.504 instead of letting it run toward π/2.
* **Coupling to the scrub coefficient — YES, hard‑coupled, and I reproduced the source’s own numbers.** (Control: c = 0.55 → 0.45116173693623052, bit‑identical to production.)

| c | published peak | headroom vs 0.452 | 600‑tick deficit (floor 20 km/h) | verdict |
|---|---|---|---|---|
| 0.4444 (=4/9) | 0.4500000000 | 0.00200 | 1.06 km/h | D1.3 FAIL |
| 0.5000 | 0.4506057892 | 0.00139 | 18.60 km/h | D1.3 FAIL |
| **0.5500** | **0.4511617369** | **0.00084** | 32.55 km/h | ok |
| 0.6000 | 0.4517131890 | 0.00029 | 44.78 km/h | ok (the JSDoc’s “0.00029”) |
| 0.6100 | 0.4518228714 | 0.00018 | 47.05 km/h | ok (the JSDoc’s 0.45182287) |
| 0.6260 | 0.4519979733 | 0.0000020 | 50.57 km/h | ok (the JSDoc’s “crossing ≈0.626”) |
| 0.6500 | 0.4522611403 | −0.00026 | 55.60 km/h | **bound VIOLATED** |
| 1.0000 | 0.4559891215 | −0.00399 | 107.14 km/h | **bound VIOLATED** |

⇒ the admissible band for the 0.002 tolerance is **c ∈ [≈0.504, ≈0.626]**; 0.55 sits inside it with 0.00084 of headroom. **Yes — any coefficient change outside that band invalidates the tolerance** (and the source says so itself: `drift.model.ts:83‑91` “ANY coefficient change FORCES a re‑derivation of this tolerance”). Note the JSDoc’s claim that the overshoot is “INTRINSIC for every c > 4/9” is confirmed: at c = 4/9 the peak is exactly 0.45 (no overshoot), at c = 0.45 it is 0.4500457751.

---

## S3 — THE REPLAY CONTRACT TO PROPOSE (not implemented)

**Mechanism chosen: a deterministic TICK COUNTER on the 14N seam + TICK‑BASED phase budgets, plus (per the existing two‑tier rule) the exact peak assertions move to the jest lane.**

**Production (1 file, 3 edits) — `examples/neatRacing/browser-entry/browser-entry.ts`**
1. `:127‑140` — add `readonly tick: number;` to `interface NeatRacingDriftState` and update the “EXACTLY four‑field” doc (`:127‑129`; the same wording is mirrored in the fixture at `:836‑838` and in `plans/neatracing-voxel-racer.14N-fixture-gate.md:96‑102`).
2. `:963‑968` — add `tick: car.tick,` to the seam write (`car.tick` is already in scope and used at `:975` and `:1037`; it is the deterministic 60 Hz counter `state.tick + 1` from `physics.step.service.ts:449/456`).
3. `:1250‑1259` — no change (the `Window` declaration references the interface).

**Fixture (1 file) — `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html`**
4. `:839‑855` `sampleDriftSeam()` — read `seam.tick` defensively and return it (this is what makes every phase tick‑exact).
5. `:209‑323` — convert the phase budgets to ticks (values unchanged, 1 frame @60 Hz = 1 tick): `ACCEL_MAX_FRAMES:209` (keep as a watchdog; the phase is speed‑gated), `SPEED_CAPTURE_FRAME:301 → SPEED_CAPTURE_TICKS = 600`, `STRAIGHT_WARMUP_FRAMES:308 → 120`, `CATCH_FRAMES:310 → 30`, `MIRROR_HOLD_FRAMES:315 → 300`, `SETTLE_HOLD_FRAMES:318 → 30`, `SETTLE_FRAMES:319 → 120`, `STUCK_OFFSET_TAIL_FRAMES:321 → 30`, `RECHECK_FRAMES:323 → 120`.
6. `:947‑1000` `observeFeelPhase()` — drive the phase on a **tick delta**, not a frame count: take `startTick` from the first seam sample, end the phase on the first frame with `seam.tick − startTick ≥ tickBudget` (the frame count stays as a watchdog), and move the named capture point (`:989‑999`, `captureLabel && frameIndex === frameCount - 1`) to the first frame whose tick delta **equals** the budget, so the P1/P2 capture pair is tick‑comparable. `observeUntil(predicate, maxFrames, cb)` (`:789`) already exists for exactly this “run until a predicate” shape; `observeFrames` (`:766`) is the alternative insertion point.
7. `:1585‑1588` `FEEL_MOVE_PHASES` unchanged; `:1862‑1866` report `speedCaptureTicks` (and the per‑phase tick deltas) so a failure is diagnosable.
8. **Coverage awareness (mandatory):** the seam is written **once per frame** (`browser-entry.ts:957‑959`), so at 163 fps the fixture sees a *decimated* tick trace (1 sample per 0.37 ticks) and a 1‑tick spike can fall between samples. The browser may therefore only carry **lower‑bound (presence) / upper‑bound‑on‑bounded‑quantity** assertions, exactly as `plans/neatracing-voxel-racer.14N-fixture-gate.md:176‑179` demands; the exact published‑slip peak and envelope belong to the jest lane. The `tick` field also lets the fixture *report* the coverage (samples/ticks) instead of silently aliasing.

**RED contracts to add in the jest lane**
* **R1 (the replay contract — pins the S1 answer).** `examples/neatRacing/physics/physics.step.service.test.ts`, inside `describe('physics step service slip and drift envelopes (LD-15)')` (`:278`): `it('the published-slip peak is a MODEL property, not a tick-budget artifact (REPLAY-STABILITY)')` — for E ∈ {8, 30, 60, 120, 600} entry ticks (W+A from the top speed, road, then 180 neutral ticks) assert the peak |slip| is **exactly** `0.4511617369362305` for every E and that the peak tick is the same across the budgets. This is the number the browser fixture must **not** re‑derive from frames.
* **R2 (RED today — the off‑road envelope gap).** Same file: `it('the published slip stays inside the declared envelope on GRASS (OFF-ROAD-GAP)')` — the peak published slip on grass must be ≤ `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD + NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` (0.452). **Measured today: 0.5038468334 → RED.** Either the model is fixed (clamp the published slip, or fix the off‑road collapse so `vLat/speed` never exceeds `tan(0.45)`) or a *separate* off‑road published‑slip ceiling is declared and exported.
* **R3 (RED today — the off‑road floor gap, the root cause).** Same file, next to the existing floor test (`:352`): `it('the off-road speed floor holds with the throttle down and a slide engaged (OFF-ROAD-FLOOR-GAP)')` — on grass with throttle 1 and steer 1 the speed must stay ≥ `NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S`. **Measured today: breaks at t=116, decays to 0 → RED.** The existing test at `:352` uses `throttle: 0`, which is why it never sees the bug.
* **R4 (pins the commitment semantics you rely on).** `examples/neatRacing/physics/drift.model.test.ts`, inside `describe('drift model commitment and countersteer (14N G-4 / D1 model gaps)')` (`:689`): `it('a LATCHED commitment is HELD under a NEUTRAL steer and released by the MIN_SPEED gate (COMMITMENT-HOLD)')` — after latching, 240 neutral ticks keep `committed.slip` exactly at the latched value with `turnSign ≠ 0`; a tick with `speed ≤ 15` releases it (slip 0, turnSign 0, reArmed false). Measured: 0.4500000000 held for 240 ticks; release zeroes the slip and clears the token.
* **R5 (guards the coupling).** `drift.model.test.ts` (constants describe at `:524`): assert the shipped `NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S` lies in the band where the measured published peak stays ≤ 0.452 **and** the D1.3 deficit floor still holds (`0.504 ≤ c ≤ 0.626`), so a re‑pin of the coefficient without a re‑derivation of the tolerance fails loudly.

**Geometry blocker that R‑conversion alone does not fix (must be stated in the same slice):** the browser lane drives a closed ~240 m‑radius track; a 600‑tick (10 s) run at 89 m/s covers ~890 m ≈ 3.7 laps, so the car is *guaranteed* to leave the road. Tick‑exact phases make the lane replay‑stable but it will keep measuring a **dead car** unless either (i) the capture horizon is shortened to a tick budget the track can hold (the road‑only emulation is clean at 120–300 ticks) and the deficit floor is re‑pinned for that horizon, or (ii) the deficit/peak assertions move to the jest lane (the 14N two‑tier rule, `14N-fixture-gate.md:169‑179`). Also note the fixture already *records* the frame‑rate dependence as “pre‑existing and out of this contract’s scope” (`:285‑288`) and sizes the 600‑frame budget on the jest lane’s 600‑**tick** horizon (`:295‑300`) — that mismatch is the root of the whole drift.

---

## (d) COMMITMENT‑RELEASE CONFIRMATION — your claim is CORRECT

`examples/neatRacing/physics/drift.model.ts:416` `stepDriftCommitment`:

```ts
416: export function stepDriftCommitment(prev, input): DriftCommittedState {
420:   let { slip, turnSign, reArmed } = prev;
426:   if (!(speed > NEATRACING_DRIFT_MIN_SPEED_MPS)) {     // MIN_SPEED gate (15 m/s, :115)
427:     if (turnSign !== 0) {
428:       slip = 0; turnSign = 0; reArmed = false;          // HARD RELEASE
432:     return { slip, turnSign, reArmed };
433:   }
437:   const countersteering = turnSign !== 0 && Math.sign(steer) === -turnSign;
438:   const drifting = turnSign !== 0 && !countersteering;
440:   if (turnSign === 0) { /* ENGAGE … */ }
457:   } else if (drifting) {
458:     // Stage 1 IN - the entry ramp.
459:     slip -= turnSign * NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S * Math.abs(steer) * dt;
466:   } else {
467:     // Stage 1 OUT - the countersteer unwinds the slide.
468:     if (Math.abs(slip) <= NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD) {
469:       slip = 0; turnSign = 0; reArmed = false;            // RELEASE on the countersteer
470:     } else { slip = sign(slip) * max(0.02, |slip| - 2.5 * dt); }
482:   }
479:   slip = clamp(slip, -0.45, 0.45);
```

* With a **latched** `turnSign ≠ 0` and a **neutral** `steer = 0`: `Math.sign(0) = 0 ≠ −turnSign` ⇒ `countersteering = false` (`:437`) ⇒ `drifting = true` (`:438`) ⇒ the **Stage 1 IN** branch (`:457‑461`) is taken ⇒ `slip -= turnSign · 3.5 · |steer| · dt` = `slip − 0` ⇒ **the committed slip is HELD, not unwound.** ✔
* **Measured (G1)**: after 60 W+A ticks, `committed.slip = −0.4500000000`, `turnSign = 1`, `reArmed = true`; a **240‑tick neutral hold keeps `committed.slip = −0.4500000000` exactly** (the *published* slip decays 0.4510360759 → 0.4501692147 only because the scrub bleeds speed; it never returns under 0.45 while the latch holds).
* **Release paths — BOTH, and only these two**: (a) the **MIN_SPEED gate** (`:426‑433`): at `speed ≤ 15` the latch is zeroed and `reArmed` cleared — measured: `speed = 14.5` ⇒ `slip 0, turnSign 0, reArmed false, published 0`; (b) the **countersteer unwind** (`:466‑482`): the slip unwinds at 2.5 rad/s toward the 0.02 snap band and the latch releases on entry — measured: released after **12 ticks**, `slip 0, turnSign 0, reArmed false, vLat 0`. A **neutral steer never releases** the latch. After a release the re‑arm token is cleared, so a fresh steer does **not** re‑latch (measured G4) until a neutral tick re‑latches it (G5).
* Consequence for the fixture: this is exactly why the fixture’s “straight‑line” P1 phase is **not** a straight line in the drift channel — the retained W+A commitment is still latched through P1, which is why the P1 capture speed is a *slide* speed (the AC‑079 deficit is negative in the road‑only lane) and why the P1 pose is 0.225 rad on the road (the `driftFeelZeroStraightLineYaw` exact‑zero check only passes in the browser because the car dies).

---

## (e) WHERE YOUR CLAIMS ARE WRONG

1. **“|slip| = 0.4488 after 288 ticks of a straight line” — WRONG (impossible).** A straight line never engages the drift (`NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD = 0.5`, `drift.model.ts:120`): 288 straight ticks from the top speed give published slip **0.000000000000**, `committed.slip = 0`, `turnSign = 0`, speed 89.0. The value **0.4488 is real but is a W+A reading**: it is the browser’s `PRE-WA` phase at 60 fps on the real track = **0.448846746116** (my emulation, exact match). A straight line *after* a W+A (latch held) reads 0.450185299165; 288 W+A ticks read 0.450185299165; 288 W+A ticks on grass read 0.
2. **“|slip| = 0.5009 in a 30‑tick W+A entry” — WRONG for the model you quoted.** On the road, 30 W+A ticks from the top speed = **0.4506537271** (committed −0.45, speed 87.68). 0.5009 is only reproducible **off‑road**: a 30‑tick W+A on grass gives 0.4981890460 (entry 25 m/s), 0.5002685350 (26), 0.5024603014 (28), 0.5033677670 (30), 0.5039368510 (40+). Your 0.5009 ≈ an off‑road entry at ~27 m/s.
3. **“the browser measured 0.5038 … 12 % over the 0.45 clamp” — the number is right (reproduced: 0.503837975136 @ 90 fps) but the attribution is wrong**: it is **not** a tick‑budget artifact; it is the **off‑road** regime, and it is present at every frame rate (0.4984–0.5040). The frame rate only moves *which phase* it lands in and *how much* (0.0056 rad). The tick budget never changes the peak value on the road (0.4511617369 for every E ≥ 8).
4. **“does the published slip overshoot its clamp differently at different tick budgets?” — WRONG on the road.** The overshoot is 0.00116 rad (0.45116 vs 0.45) for **every** entry budget ≥ 8 ticks, always at absolute tick 42; a warm prefix shifts only the timestamp. Below 8 ticks the transient is merely **truncated** (0.0583·E for E ≤ 6). The real budget sensitivity is in **phase composition** (which phase the car dies in) and in the **capture pair** (the AC‑079 deficit flips 0 ↔ 59.95 km/h across 60→163 fps).
5. **“the 0.002 tolerance holds” (implicit) — WRONG over the reachable space.** It holds only on‑road (0.45116 ≤ 0.452); the grass regime reaches 0.5039742512, violating both 0.452 and the browser’s own 0.47.
6. **“is the tolerance coupled to c = 0.55?” — CORRECT**, and tighter than the JSDoc states: the admissible band is **c ∈ [≈0.504, ≈0.626]**; the D1.3 deficit floor kills c < 0.504 and the 0.452 ceiling kills c > 0.626.
7. **“the browser releases 0..4 fixed simulation ticks per animation frame” — CORRECT** (`render-loop.utils.ts:27‑64`, `:195‑218`); 600 frames = 600/400/300/250/221 ticks at 60/90/120/144/163 fps.
8. **“the browser fixture pins its own bound at 0.45 + 0.02 = 0.47” — CORRECT** (fixture `:379‑380`, asserted at `:1798‑1800`) — and my emulation shows the browser’s own lane **violates its own bound** (0.5038) at 90–163 fps while **passing vacuously** at 60 fps.
9. **“the commitment is released ONLY by a countersteer or by the MIN_SPEED gate closing” — CORRECT (both), and note there is no third path**: a neutral steer neither releases nor unwinds it.

---

### Risks / gaps I did not close
* The browser‑lane numbers are a **faithful Node reconstruction** of the fixture’s phase sequence (production track + production surface resolver + production `advanceFrameSteps` + production `stepPhysics`), **not** a real WebGL run: the real run adds frame‑timing jitter, which can shift the *frame‑sampled* peak by ±1 frame but cannot change the tick‑level conclusions (the tick budgets, the release ticks and the tick peaks are exact).
* The **body‑to‑travel proxy** path (`sampleFeelProxy`, the `tireYawDriftEngaged` input to the AC‑079 check) was not emulated; the deficit/verdict flips I report are on the *measured capture speeds*, which are exact.
* I did not verify the served bundle (`examples/neatRacing/assets/neatracing.bundle.js`) against the current source byte‑for‑byte beyond the earlier check that it contains the 14N model.

```structured-v1
OUTPUT_CONTRACT: structured-v1
TASK_STATUS: SUCCESS
TIER: 3
ROLE: deterministic-reviewer
TASK_RECEIVED: Replay-stability / ordering-drift review of the neatRacing drift channel: (S1) tick-budget divergence of the browser fixture's frame-counted drift phases vs the 60 Hz tick contracts, (S2) declared vs measured published-slip envelope, (S3) the smallest replay-stable fixture contract, plus a commitment-release semantics verification.
FILES_READ:
- examples/neatRacing/physics/drift.model.ts (66, 94, 115, 120, 213, 416-497, 601-656, 658-700, 710-718)
- examples/neatRacing/physics/physics.step.service.ts (362-470, 490-544)
- examples/neatRacing/physics/collision.utils.ts (165-169, 529-544)
- examples/neatRacing/host/render-loop.utils.ts (27-64, 195-218)
- examples/neatRacing/browser-entry/browser-entry.ts (120-145, 940-985, 1249-1259)
- examples/neatRacing/physics/physics.step.service.test.ts (218-640)
- examples/neatRacing/physics/drift.model.test.ts (524-1094)
- examples/racing_curriculum/track/track.generator.ts (14-33, 69-119)
- docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html (200-400, 760-1010, 1470-1620, 1690-1900)
- plans/neatracing-voxel-racer.14N-fixture-gate.md (76-106, 167-235, 398-410, 553-565)
- plans/neatracing-voxel-racer.14N-red-contracts.md (837-851)
FILES_CHANGED:
- NONE (read-only review; all probe scripts are outside the repo, in C:\Users\reice\AppData\Local\Temp\drift-probe\)
KEY_FINDINGS:
- [non-deterministic-order / uncontrolled-async] The browser fixture's frame-counted phase budgets give a 2.7x tick-budget spread (600 frames = 600/400/300/250/221 ticks at 60/90/120/144/163 fps), and the fixture's check VERDICTS flip with the frame rate: driftFeelModelIdentity PASS@60 / FAIL@90-163 (0.5038>0.47); driftFeelZeroStraightLineYaw PASS@60-144 / FAIL@163; AC-079 deficit FAIL@60-144 (0-6.9 km/h) / PASS@163 (59.95 km/h). The road-only lane flips too (deficit -0.73 -> -26.80 km/h).
- [seed-not-propagated (analog: the tick clock is not propagated to the fixture)] No tick counter exists on the 14N seam (browser-entry.ts:127-140, 963-968), so the fixture cannot make its phases tick-exact; car.tick is already available at the write site (:975, :1037).
- [checkpoint-state-omission (analog: the off-road surface state is omitted from the envelope derivation)] The declared published-slip ceiling 0.452 (drift.model.ts:66+94) is calibrated ONLY on-road; the reachable grass regime peaks at 0.5039742512 (worst over surface x entry speed 15.5-89 x steer 0.5-1 x budget 1-600) - violating both 0.452 and the browser's own 0.47. The published slip has no clamp of its own; the excursion is bounded only by the MIN_SPEED gate.
- [floating-point-reorder / model gap] The off-road speed floor is broken by the post-floor scrub (physics.step.service.ts:510-544 applies the 22.25 floor before the 415-427 scrub, gated on the tick-start speed): measured break at t=116, car at 0 m/s by t~181, commitment released at t=133. The existing jest floor test (:352) uses throttle 0 and never sees it.
- [non-deterministic-order] S1 ANSWER: the published-slip peak is a model property, not a budget artifact - 0.4511617369 for EVERY entry budget >= 8 ticks (always at absolute tick 42); a warm prefix shifts only the timestamp (42/72/162/330/642); budgets <= 6 ticks merely truncate the transient. The frame-rate sensitivity lives in the phase composition, the capture pair and the off-road excursion, not in the peak value.
- [shared-RNG-state (analog: shared mutable commitment state)] The latched commitment survives a neutral steer (drift.model.ts:437-438, 457-461) and is released ONLY by the MIN_SPEED gate (:426-433) or the countersteer unwind (:466-482) - so the fixture's "straight-line" P1 phase is a slide, which is why the P1 exact-zero check passes only vacuously (the car dies).
- [RISK] The 0.002 published-slip tolerance is hard-coupled to the scrub coefficient: admissible band c in [~0.504, ~0.626] (D1.3 deficit floor kills c<0.504; the 0.452 ceiling kills c>0.626); reproduced the source's own numbers (c=0.6 -> 0.00029 headroom, c=0.61 -> 0.45182287, crossing ~0.626).
ACTIONS_TAKEN:
- Built and validated an out-of-repo Node harness (esbuild bundle) reproducing the production model; control run bit-identical to the jest lane.
- Ran the entry-budget sweep (E=1..600, road/grass), the warm-prefix sweep, the browser phase-sequence emulation at 60/90/120/144/163 fps on the real track + real surface resolver and on a road-only resolver, the reachable-space worst-case sweep (630 runs/regime), the scrub-coefficient sweep via out-of-repo patched copies, the off-road floor-break trace, and the commitment-hold/release trace.
VALIDATION_EVIDENCE:
- Harness control: 180-tick slip -0.45041978773870417; run peak 0.45116173652... = 0.4511617369362305 @t42; 1200-tick speed 79.85792723069594; 600-tick straight 89.0; deficit 32.55473 km/h (all match the jest lane exactly).
- Off-road floor: grass + throttle + slide -> floor 22.25 breaks at t=116, speed 0 by t~181, MIN_SPEED release at t=133; grass + throttle + no slide -> holds at 22.25 for 400 ticks.
- Peak mechanism: road t42 vLat/target = 1.00296592 (tan ratio 1.00296783); grass t132 vLat/target = 1.13719928 (tan ratio 1.14129419), speed 15.280 -> 14.824.
- Browser lane (real track): moveSlip 0.0000000000 @60 fps vs 0.5038379751 / 0.5038098516 / 0.5039560400 / 0.5039413437 @90/120/144/163; release ticks 280/307/461/437/430; total ticks 2342/1602/1230/1044/936.
- Scrub sweep control at c=0.55 bit-identical (0.45116173693623052); c=0.6 -> 0.4517131890, c=0.61 -> 0.4518228714, c=0.626 -> 0.4519979733, c=0.65 -> 0.4522611403 (VIOLATED).
HANDOFF: Route to the 14N plan/RED-contract author: (1) add the seam tick field + tick-based phase budgets to the fixture and the browser-entry seam; (2) add RED R1 (peak is budget-independent) in physics.step.service.test.ts:278+, RED R2 (off-road published-slip envelope) and R3 (off-road floor holds under throttle+slide) as RED contracts for the model gap, RED R4 (commitment hold/release) in drift.model.test.ts:689+, RED R5 (scrub-coefficient band); (3) decide the browser capture horizon (the 600-tick horizon cannot be held on a 240 m-radius track) and re-pin the AC-079 deficit floor accordingly.
BLOCKERS: NONE
RISKS_OR_GAPS:
- The browser-lane table is a production-source reconstruction, not a real WebGL run; per-frame sampling jitter can move the frame-sampled peak by +/-1 frame (the tick-level conclusions are unaffected).
- The fixture's body-to-travel proxy path (tireYawDriftEngaged) was not emulated; the reported verdict flips are on the exact capture speeds.
- The published-slip peak may also be reachable in the curb surface or during a surface transition (not swept) - the grass finding is a lower bound on the envelope gap.
LEARNING_EVENT_NEEDED: true
SUGGESTED_NEXT_AGENT: deterministic-reviewer (re-review after the fixture tick-budget conversion) + api-contract-reviewer (the seam becomes a 5-field surface - the "EXACTLY four-field" contract at browser-entry.ts:127-129, fixture:836-838 and 14N-fixture-gate.md:96-102 must be amended together)
SUMMARY: The published-slip peak is a genuine model property (0.4511617369 on-road for every entry tick budget >= 8, always at absolute tick 42) - NOT a tick-budget artifact - so the on-road envelope 0.452 holds; but the reachable OFF-ROAD regime peaks at 0.5039742512 (violating both 0.452 and the browser's own 0.47), because the off-road speed floor is broken by the post-floor scrub, the car decays to a stop, and the published slip (unclamped) spikes as the speed collapses just above the MIN_SPEED gate. Meanwhile the browser fixture is frame-counted, so its 600-frame phase is 600..221 ticks across 60..163 fps, its car dies at a frame-rate-dependent tick, and three of its checks flip verdict with the frame rate; the fixture's own 0.47 bound is violated by its own run at 90-163 fps and passes vacuously at 60 fps. The 0.002 tolerance is hard-coupled to the scrub coefficient (admissible c in [~0.504, ~0.626]). The commitment is HELD under a neutral steer and released only by the MIN_SPEED gate or the countersteer - the user's commitment claim is correct; their 0.4488-on-a-straight-line and 0.5009-on-road claims are not.
```