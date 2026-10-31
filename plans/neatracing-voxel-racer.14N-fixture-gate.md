# Step 14N — Phase-B proposal 4/4: browser fixture check set + visible-window user-verdict gate protocol

- **Plan**: `plans/neatracing-voxel-racer.plans.md` (Phase 1, Step 14N `[WIP]`)
- **Step**: 14N — Arcade drift FEEL sharpening (opens from `user-drift-verdict-2026-10-24`)
- **Author role**: 02-researching (Tier-1 drift-feel scout 4/4), read-only research + proposal
- **Status**: PROPOSAL COMPLETE — the fixture check set and the user-gate protocol are specified;
  the numeric FLOORS that the RED slice must pin are named as RED-slice decisions, not invented here
- **Sub-orchestrators used**: `implementation-pattern-scout` (fixture harness map),
  `boundary-mapper` (pose + tire-gradient uniform seam map). Both read-only; both re-verified by the
  orchestrator against the source before merge.
- **Inputs**: Phase-A MERGE block (`plans.md:8414-8440`), the drift-feel research artifact
  (`plans/neatracing-voxel-racer.drift-feel.research.md` D1/D3/D6 + Risks R1-R15), the coupling
  research artifact (`plans/neatracing-voxel-racer.research.md`), the plan's AC-087..AC-090 rows.
- **Scope of this file**: the BROWSER half of Step 14N and the USER gate. The deterministic jest
  contracts (the pop / settle / step-bound / envelope NUMBERS) belong to the RED-contracts proposal
  (`plans/neatracing-voxel-racer.14N-red-contracts.md`); this file states the **interface** the
  browser half needs from that lane and does not restate the jest contracts.
- **Gates run**: `cortex-index` → **PASS** (1,998 documents, 27,452 chunks, index fresh, all 10
  document families fresh, corpus + workflow MCP alive); `cortex-first-search` → **PASS**
  (3 corpus search results, index fresh). **Documented native-tool fallback**: the Cortex corpus
  does **not** index `docs/browser-tests/**/*.html`, so the fixture was read with `view`/`grep`.
  This is the fallback the research-methodology skill allows for a known file path, and it is
  recorded here so the gate has the trail.

### 0. The recommendation in one page

1. **The browser fixture cannot own the pop / settle / step-bound / envelope numbers**, because the
   page exposes no `heading`, no `slip` and no `pose` — every browser-computable angle is
   `pose - slip` (§1). **The deterministic jest lane owns the numbers; the fixture owns the shape,
   the wiring, the AC-088 uniform seam and the regression guards** (§3.0).
2. **The single highest-leverage change** is a narrow read-only observability seam in
   `browser-entry.ts` exposing the raw `pose` / `slip` scalars (§1). It converts the browser half
   from approximate to exact, and it is a **required file-set amendment** for 14N. Without it, the
   **Option B** fallback (§3.8) applies and must be recorded as a deliberate downgrade.
3. **The fixture needs a new scenario phase** — drift entry → hold → **countersteer catch** →
   release → settle — because the current scenario has **no countersteer at all** and therefore
   cannot observe the pop, the ring or the settle (§4.1). **W3 re-base**: the sequence is now
   **three controlled runs** (the straight capture, the drift capture with the **7b-hold /
   7b-release** split, and the mirror), the budgets are **physics ticks** rather than frames, and
   the capture window is **uncapped in both runs** so the two capture points are comparable by
   construction (§4.1, §4.5).
4. **The proposed check set** is 7 new named checks (§3.1-§3.7), each with a stated derivation
   from a Tier-1 measured value and a stated headroom factor, plus the retained AC-078/AC-079
   checks and the 3-frame drift gate.
5. **AC-088 becomes a ~6-line fixture-only change**: widen `recordUniform` (fixture L151-159) to
   capture the four `uTireGradient*` floats, then assert the factor spread is non-degenerate. This
   is the narrowest possible fix for the exact residual Step 14G recorded (§3.7).
6. **The user gate is a six-question scripted protocol with a deterministic verdict rule**
   (§5.3-§5.4), a **seven-item precondition list** (§5.1), a **three-surface recording format**
   (§5.5), and a **re-open rule** that keeps a REJECTED verdict from closing the step (§5.6).
7. **Three plan-text corrections are required before the slices are authored** (§8) — AC-087 pins
   the floor on the wrong quantity, AC-089 says "AC-079 stays GREEN" against a LIFTED pin, and
   AC-088 names the wrong test-file owner. **These are contradictions, not wording nits, and they
   are the highest-priority ask of this proposal.**

---

## 1. The one finding that shapes everything below

**The browser fixture CANNOT separate the drift pose from the physics heading.** The page exposes
exactly one gameplay global (`window.neatRacingStart`, `browser-entry.ts:1229`) and no physics
state. Everything the fixture can observe about the car's heading is a **WebGL model matrix** or
the **HUD DOM**:

| Observable | Source | What it actually is |
|---|---|---|
| body model matrix, index 0 of `carLitModelsThisFrame` | `uniformMatrix4fv` hook (fixture L205-214) | `heading + pose` (car.pass.ts:1563-1572) |
| wheel matrices, indices 1..4 | same | `heading + pose` (they inherit the body yaw, car.pass.ts:1589-1627) |
| shadow model matrix | `uniformMatrix4fv` hook | `heading + pose` (car.pass.ts:1365) |
| travel direction | chord across a rolling 6-frame window (fixture L806-807) | `heading + slip` |
| speed | HUD text (`readSpeed(speedLabel)`) | scalar, no pose |

So every browser-computable angle is a **difference of two unobserved quantities**:

```
bodyToTravel = (heading + pose) - (heading + slip) = pose - slip
```

`pose` alone is **not recoverable** from `bodyToTravel` without `heading` or `slip`. Consequences:

1. **The browser CANNOT own the envelope contract** (which is on `|pose|`), because `|pose|` is not
   observable. `|bodyToTravel|` is a *different* quantity — at the modelled hold it is
   `|(-0.1773) - (-0.4431)| = 0.2658 rad`, which is **above** the 0.2250 rad envelope even though
   `|pose| = 0.1773 rad` is well inside it. A fixture check written as
   `|bodyToTravel| <= 0.2250` would **fail a correct implementation**.
2. **The browser CANNOT own the per-tick pose-target step**, because the target is internal.
3. **The browser CANNOT own the pop or the settle** as exact quantities, because both are defined
   on the pose trace.
4. The old proxy identity `proxy = |slip| * |GAIN_eff - 1|` (research E3) **does not survive the
   new model**: the pose is no longer `slip * GAIN`, so the proxy is now `|pose - slip|` and is a
   mix of the physical slide and the visual spring. It is still a valid "is something sliding?"
   detector; it is **not** a contract quantity.

**Recommendation (the required amendment to the Step 14N file set).** Add a **narrow read-only
observability seam** so the browser half measures the pose instead of inferring it:

```ts
// examples/neatRacing/browser-entry/browser-entry.ts  (NEW in the Step 14N file set)
// One named, read-only debug surface next to the EXISTING bridge write (:941).
//
// IT IS A STABLE BAG, NOT A FRESH OBJECT LITERAL PER FRAME (Step 14Q2, the
// SEAM WRITER slice P1-drift-traction-seam): ONE object is allocated ONCE
// and its SEVEN fields are re-assigned in place, once per frame. The
// zero-per-frame-allocation rule forbids a fresh literal per frame.
const driftState = {          // ONE object, allocated ONCE
  poseRad: 0,         // car.drift.pose.pose  (the rendered pose scalar)
  poseRateRadPerS: 0, // car.drift.pose.poseRate (the PUBLISHED pose rate)
  slipAngleRad: 0,   // the physics slip
  speedMps: 0,        // the physics speed
  tickHz: 60,
  tick: 0,            // car.tick — the PHYSICS TICK COUNTER
  followOffsetRad: 0, // the chase-camera follow offset (Step 14P)
  tractionSeverity: 0 // the PUBLISHED traction severity (Step 14Q2)
};
// ... once per frame, in the same closure that builds the composer frame:
driftState.poseRad = ...;
driftState.poseRateRadPerS = car.drift.pose.poseRate; // READ, never re-derived
driftState.slipAngleRad = ...;
driftState.speedMps = ...;
driftState.tickHz = ...;
driftState.tick = ...;
driftState.followOffsetRad = ...;
driftState.tractionSeverity = car.tractionSeverity; // READ, never re-derived
window.neatRacingDriftState = driftState;
```

- **The seam is EXACTLY EIGHT FIELDS** (W3 contract, extended by Step 14P, extended by Step 14Q2,
  extended by Step 14R):
  `poseRad`, `poseRateRadPerS`, `slipAngleRad`, `speedMps`, `tickHz`, `tick`, `followOffsetRad`,
  `tractionSeverity`.
  The fifth field was added by
  the W3 re-base because the fixture's tick
  budgets are only reproducible if the page exposes the **physics tick**, not just the tick RATE:
  without `tick` the fixture can only count wall-clock frames, so a frame that carries N catch-up
  ticks is indistinguishable from N frames that each carry one tick — and the per-tick input
  alignment (§4.1) becomes unobservable. `tick` is `car.tick` verbatim (the production counter),
  and it is **read-only** (the fixture never writes it). The **sixth** field was added by Step
  14P: `followOffsetRad` is the chase camera's own share of the shared slip scalar (the camera's
  aim offset from the nose heading), so the browser half can MEASURE how far the 3D view actually
  swung instead of inferring it. It is written from the ONE aim law
  (`resolveChaseCameraAimRad` + `stepChaseCameraFollow`), **once per frame**, in the same closure
  that steps the follow, so the published offset and the published slip are from the SAME tick.
  The **seventh** field was added by Step 14Q2 (`P1-drift-traction-seam`, the SEAM WRITER):
  `tractionSeverity` is the **PUBLISHED traction severity** — the shared scalar the physics state
  publishes (`car.tractionSeverity`, computed by `computeTractionSeverity` in
  `examples/neatRacing/physics/drift.model.ts`). It is **READ** from the state and **NEVER
  re-derived** in the browser, so the fixture observes the ONE number the simulation produced.
  The **eighth** field was added by Step 14R (the SEAM WRITER slice
  `P1-drift-presentation-car`, the READER slice `P1-drift-presentation-green`):
  `poseRateRadPerS` is the **PUBLISHED pose-spring RATE** in rad/s, READ from
  `car.drift.pose.poseRate` and **NEVER re-derived** in the browser. It is a **NAMED
  CHANNEL and it is NOT the traction severity**: the severity stays the **seventh**
  field (published by Step 14Q) and it is **NOT** re-published as the eighth, so no
  consumer re-derives it. **WHY THIS CHANNEL**: the browser half is RENDER-RATE
  dependent (the host releases 0-4 simulation ticks per rendered frame), so a frame
  difference measures the CAMERA, not the car - the published rate is the crab's own
  per-tick quantity, so the fixture can measure the ring-down and the far-side pop in
  SIMULATION time next to the pose it belongs to. The seam stays a **STABLE BAG**
  (ONE allocation per boot, refreshed IN PLACE, EXACTLY ONE
  `window.neatRacingDriftState` assignment per driven frame), so the
  zero-per-frame-allocation rule holds.
- **A READER MUST COPY THE FIELDS (the stable-bag consequence).** `window.neatRacingDriftState`
  points at ONE object for the whole session, so a reader that keeps a REFERENCE instead of
  copying the fields would observe later frames. The fixture's `sampleDriftSeam()` copies the
  EIGHT fields into a NEW object per sample, so the current reader is safe; a future reader must
  copy too.
- **The per-tick input reality (why `tick` is not optional)**: the host samples the keyboard
  **once per rAF frame** and **replays the same sample into every catch-up tick** of that frame
  (`examples/neatRacing/host/race-session.ts:200-256`). The fixture therefore **cannot** apply a
  steer per tick; the honest implementation is to flip the keys at **tick-ALIGNED** boundaries
  (the fixture waits for `tick` to advance past a target before/after each key event) and to
  **report the residual alignment** in the evidence notes. This is a documented RISK, not a
  silent approximation: without `tick` even the residual could not be reported.
- **What the alignment log records (and what it does NOT)**: the tick the flip was **issued** on
  (`tickAtFlip`), the tick the wait started from (`tickBefore`), whether the simulation clock
  advanced during the wait (`advanced`), and how many frames the wait took (`alignFrames`). It
  does **not** record the tick the input **lands** on — the host replays the *next* frame's
  keyboard sample into that frame's ticks, so the input lands **1 to 4 ticks** after `tickAtFlip`.
  The recorded alignment is therefore an **upper bound** on the misalignment, and it is reported
  rather than hidden.

- **Cost**: one file added to the Step 14N file set (`examples/neatRacing/browser-entry/browser-entry.ts`),
  one assignment per frame, **no new public gameplay API** (it is a `window.` debug surface, not
  an exported module symbol), and **no change** to the existing bridge contract
  (`carPoseBag.driftYawOffsetRad` stays exactly as it is).
- **Payoff**: it converts the browser half from *shape guesser* into *exact verifier*. It enables
  the model-identity check (§4.2), the exact envelope check (§3.4), the exact settle check (§3.2),
  and the zero-straight-line identity (§3.5).
- **If the seam is rejected** (01-planning declines the file-set amendment): fall back to
  **Option B** (§3.8) — the browser carries only the AC-088 uniform check, the loose slide-prominence
  check, the no-mirrored-inversion bound, and the retained regression checks; the pop / settle /
  step-bound / envelope live in jest only, and the user gate carries the feel verdict. **Option B is
  strictly weaker and MUST be recorded as such in the plan.**

---

## 2. The existing fixture — what we are extending

`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html` (994 lines, 44,738 bytes) is the
**only** drift scenario in the repo. It is a self-contained ES5 browser page (not a `src/` module —
**ES5 only**: `var`, `function`, no arrow functions, no `const`/`let`, no optional chaining).

| Fact | Anchor |
|---|---|
| **31** named checks in `result.checks` (a plain `key: expr` object literal, L893-938) | L893-937 (keys), L938 (`};`) |
| `result.passed = Object.values(result.checks).every(Boolean)` | L939 |
| evidence payload `result.notes = JSON.stringify({...})` | L940-982 |
| public result `window.p1TurnDriftGreenSmokeResult` | L32, re-assigned L987 |
| `uniform1f` wrapper — calls `recordUniform(loc, value)` for **every** float uniform | L189-197 |
| **`recordUniform(loc, value)` — the narrow capture point; today it records ONLY `uSpacing`** | **L151-159** |
| `getUniformLocation` hook (builds a COMPLETE name map — any uniform is resolvable) | L184-188 |
| 4 per-frame capture arrays, cleared by `resetFrameCapture()` | L56/57/66/72, L598-603 |
| drift-proxy measurement block (rolling 6-frame endpoint-anchored window) | L780-825 |
| W+A drift phase | `observeFrames(100, cb)` L773, callback L774-848 |
| key dispatch helper | `dispatchKey(code, down)` L289-293 (window-level `KeyboardEvent`) |
| retained AC-078 check `tireYawFollowsBodyDuringDrift` | L926-928 |
| retained AC-079 check `proportionalTractionLoss` | L935-937 |

**Insertion anchors for this step** (all confirmed against the file):

| What | Where |
|---|---|
| new named check keys | after **L937** (`proportionalTractionLoss`), before **L938** (`};`) — grouped with the AC-078/AC-079 drift checks |
| new gate constants | after **L146** (the last constant is `DRIFT_PROXY_WINDOW`) |
| new measurement state | the state block **L83-96** |
| new per-frame capture | inside the W+A callback **L774-848**; frame-scoped state must be flushed by `resetFrameCapture()` at **L848** |
| new evidence | a new sub-object in the notes payload, after **L981** |
| new intermediate predicates | the **L872-890** band (beside `tireYawDriftEngaged` L875-880) |
| new phases | after the W+A phase closes (after L848-849), before the checks object is built |
| **the `uTireGradient*` value capture** | inside `recordUniform`, **L151-159** |

**Convention to mirror**: every new check gets a `// AC-xxx:` rationale comment block carrying the
AC id, the WHY, and the numeric rationale; every new gate is a top-level `SCREAMING_SNAKE` var
with the same three-part comment.

---

## 3. The proposed check set

### 3.0 The two-tier rule (this is the load-bearing design decision)

The Phase-A MERGE item 7(c) already says: *"AC-087 floor re-pinned on the DETERMINISTIC unit proxy
(browser-proxy spread 0.0494 / 0.0603 / 0.3599 rad is noise, not a floor anchor)"*, and the
research D5-2 says: *"Pin the floor in the deterministic jest lane; let the fixture carry a looser
observable."*

So the browser half is a **two-tier** structure:

| Tier | Owns | Why |
|---|---|---|
| **Tier 1 — deterministic jest lane** | the exact NUMBERS: pop angle, settle time, max per-tick pose-target step, the `|pose|` envelope, the AC-087 floor, the zero-straight-line identity, the re-baselined AC-079 ceiling | 60 Hz fixed tick, direct state access, no render-rate dependence, no frame jitter |
| **Tier 2 — browser fixture** | (a) the **wiring** — the new pose reaches the renderer and the model identity is correct; (b) the **shape** — a pop-and-ring, a settle, a bounded envelope are OBSERVABLE in the served build; (c) the **AC-088 uniform seam** — fully deterministic in the browser; (d) the **regression guards** — the retained named checks and the 3-frame drift gate | the served bundle is what the user actually drives; a jest-green build can still be wired wrong on the page |

**Rule the browser half MUST obey: every browser threshold is a LOWER BOUND (or an UPPER bound on a
*bounded* quantity) derived from the Tier-1 measured value with a STATED headroom factor. No
browser check may assert an equality on a render-rate-dependent quantity.** This is the specific
failure mode that made the Step 14G fixture flaky (three measured staleness defects, fix-packet
iteration 12).

**The D1.3 speed deficit is a TWO-TIER quantity (W3 re-base).** The AC-079 deficit is
`straightSpeed - driftSpeed` measured over a **comparable pair** of runs. The browser half can
only produce a **loose, same-regime witness**; the exact number belongs to the Tier-1 jest lane:

| Tier | Owns | Why |
|---|---|---|
| **Tier 1 — jest lane** | the exact D1.3 deficit number and the 20 km/h floor as a hard contract | 60 Hz fixed tick, controlled entry speed, no render-rate dependence |
| **Tier 2 — browser fixture (W3)** | a **same-regime witness** (both capture points on the ribbon, so the pair is comparable), the deficit **>= the 20 km/h floor** (the retained contract, now ACHIEVABLE on-road), and a **ceiling** on the deficit | the browser cannot control the entry speed exactly, and the 14N re-base showed the deficit is carried by the **slide LENGTH**, not by the surface regime |

**The W3 re-base overturned the old "regime split" premise.** The old fixture expected the drift
capture to be **off-road** while the straight capture was on-road (a "different regime ⇒ a
different speed" story). The production simulation over the REAL track (seed 42, `large`) shows
the opposite: a drifting car's **world-frame lateral offset is only 0.05-3.2 m** at the capture
point (the body yaws INTO the slide), so **both capture points are on-road** and the old
`captureRegimeSplit` assertion is **UNSATISFIABLE** — it could never pass. The browser witness is
therefore `straightCaptureOnRoad && driftCaptureOnRoad && !captureRegimeSplit` (the **same**-regime
witness) plus the deficit floor and the ceiling.

### 3.1 `driftFeelFarSidePop` — the pop is present and is a POP, not a mirror

- **Browser meaning**: during the countersteer catch, the body-to-travel angle `bodyToTravel`
  **crosses to the far side** and then **rings back** (≥ 2 direction reversals with decaying
  amplitude), and the far-side excursion is **bounded** (not a mirrored inversion).
- **Why the two halves are needed together**: the OLD model mirrored the slide — the slip went
  −65.8° → −54.4° → **+10.6° → +28.1°** (research D3) — so `bodyToTravel` crossed zero and grew
  to **+14°**. The NEW model catches in 0.30 s with a **2.22°** far-side pop. A sign-crossing
  ALONE therefore cannot discriminate the two models; the **bounded amplitude** can. Hence:
  `popDeg >= POP_MIN_DEG` (a pop happened) **AND** `popDeg <= POP_MAX_DEG` (it was a pop, not a
  mirror) **AND** `ringReversals >= 2` (it rang — the second-order signature).
- **Fixtures (Tier-1 numbers, for the browser's loose bounds)**:
  - Tier-1 measured pop: **2.22°** (research D6.9, recommended variant; 2.61° for the arcade-refs
    variant). Tier-1 contract floor: **≥ 2.0°** (the Phase-A MERGE item 7(a) value).
  - Browser lower bound: **1.0°** (≈ 45 % of the Tier-1 floor — loose enough to survive render
    jitter, tight enough to reject "no pop at all").
  - Browser upper bound: **6.0°** (≈ 2.7× the Tier-1 value, but far below the old model's 14°
    mirror). The Phase-A MERGE calls the mirrored inversion the *root cause*; this bound is the
    browser's proof that the root cause is gone.
- **The RED slice pins the exact numbers**; the values above are the *proposed* browser bounds.

### 3.2 `driftFeelSettles` — the catch settles, and it settles FAST

- **Browser meaning**: after the countersteer is released, the pose trace returns to the
  pre-drift baseline and **stays** there (no sustained ring, no stuck offset).
- **Measurement**: with the §1 seam, `settleMs` = the time from the catch instant until
  `|pose - poseBaseline| < 0.5°` **and stays** below it for 250 ms. Without the seam, use
  `bodyToTravel` and the same shape.
- **Fixtures**: Tier-1 measured settle: **550 ms** (invariant across the whole slew band 15-30 /s,
  research D6.3 — the ring-out is the spring's, not the articulation's). Tier-1 contract: ≤ 550 ms.
  Browser bound: **≤ 900 ms** (a 1.6× headroom for render-rate sampling, since the browser
  samples the settle at frame rate and the last frame may straddle the 550 ms crossing).
- **Companion check** `driftFeelNoStuckOffset`: after the settle, `|pose| <= 1.0°` (or
  `|bodyToTravel| <= 3.0°` without the seam) for the remainder of the phase — this catches the
  "the pose never returns" failure, which a settle-time check alone can miss if the trace is
  sampled too coarsely.
- **The W3 re-based measurement (the settle tail).** The settle is measured from the **release**
  instant (the countersteer keydown), not from a "catch" that the fixture cannot name: the pose
  must come back inside `0.5°` of the baseline **and stay** there for 250 ms. The W3 re-base adds
  the matching **tail** bound — `|pose| <= 0.005 rad` over the last 30 ticks of the settle
  (`STUCK_OFFSET_TAIL_TICKS = 30`) — because the simulation shows the settle still carries the
  far-side pop's residual ringing (**0.00304 rad** over the last 30 ticks) while the straight
  phase that follows has already decayed to **0.000076 rad**. The settle **time** bound
  (≤ 900 ms) is unchanged; the simulation measures the settle at ≈ 417 ms.

### 3.3 `driftFeelPoseRateBounded` — the per-tick step bound, in a frame-rate-independent form

- **The quantity the RED contract owns** is the **per-tick pose-TARGET step** (≤ 2.13°). The
  browser cannot see the target and cannot see ticks. What it CAN see is the **pose rate**.
- **Browser formulation (frame-rate independent)**:
  `max over frames of ( |pose[i] - pose[i-1]| / dtFrame ) <= POSE_RATE_LIMIT_DEG_PER_S`.
- **The bound**: research D6.2 gives the provable per-tick target bound
  `max |ΔposeTarget| <= ART * SLEW * dt + 0.5 * |slip_at_hard_zero|` = `0.08 * 20 * (1/60) + 0.01`
  = **0.0367 rad = 2.10°/tick = 126 °/s**; the measured per-tick step is **2.13°** and the measured
  peak pose rate is **92 °/s** (research D6.9). The proposed browser limit is **128 °/s** (the
  target-step rate) — which is 1.39× the measured peak and **4.3× below** the OLD raw-steer
  artifact the slew exists to kill (9.17°/tick at 89 m/s = **550 °/s**, research D6.2).
- **Why the frame-rate-independent form matters**: at 144 Hz a per-frame step check would be
  measuring 2.4 ticks per frame and would read ~0.5°/frame; a per-frame degree bound would be
  meaningless. Dividing by the measured frame delta makes the check valid at any refresh rate.
- **This is a NEW-code regression guard, not an old-vs-new comparison**: the old production model
  had no articulation term and therefore no rate spike; the slew is what keeps the NEW term bounded.

### 3.4 `driftFeelPoseEnvelope` — the pose stays inside the declared envelope

- **Browser meaning**: `max |pose| <= POSE_ENVELOPE_RAD` over the whole drift phase.
- **The envelope value is a RED-slice decision** (Phase-A MERGE item 7(e): *"envelope re-derived
  from the measured peak (0.2295 rad) or the pose held inside 0.2250 (the recommended variant does
  the latter)"*). The recommended variant holds the pose at **0.2246 rad** inside the declared
  **0.2250 rad** limit (`MAX_SLIP 0.45 * GAIN 0.5`).
- **Browser bound**: the RED slice pins the declared envelope; the browser asserts
  `max |pose| <= envelope + 0.002 rad` (a 0.11° render-jitter headroom). **This check REQUIRES the
  §1 seam** — without it, the only computable angle is `bodyToTravel` (0.2658 rad at the modelled
  hold), which is *not* the envelope quantity and would produce a false failure.
- **The existing unit pin** at `physics.step.service.test.ts:156-163` (`|driftYawOffset| < π/2`,
  spawn offset `=== 0`) must be **re-verified against the overshoot** (research D6.8, risk R12) —
  that is Tier-1 work, but the browser check is the independent cross-check that the *rendered*
  pose also respects it.

### 3.5 `driftFeelZeroStraightLineYaw` — the exact identity, browser-side — **RE-BASED by W3**

- **The Tier-1 contract (unchanged, jest lane)**: the pose is **exactly zero** on a straight line
  (research D6.9 measured `0.000000000 rad`; the Tier-1 contract is an EXACT identity to 1e-9,
  **not a tolerance**).
- **The browser half CANNOT assert an exact zero, and the W3 re-base says why.** The pose spring
  has **no snap**: it decays asymptotically towards its hard zero, so a browser sample can never
  be *structurally* zero — and the strict-equality form (`max |pose| === 0`) would be a
  **render-rate-dependent equality**, which §3.0's rule forbids. The production simulation over
  the real track measures, after the countersteer release: the **settle tail (the last 30 ticks of
  the ring-down) = 0.00304 rad** and the **straight phase that follows = 0.000076 rad**. The
  browser therefore asserts the **re-based** expectation:
  - the **straight phase AFTER the release**: `max |pose| <= 0.002 rad` (the declared
    render-jitter headroom — the same 0.002 rad tolerance the other 14N bounds use);
  - the **settle tail**: `max |pose| <= 0.005 rad` over the last 30 ticks of the ring-down (the
    far-side pop's residual ringing — a SEPARATE bound, because the settle still carries it while
    the straight phase no longer does).
  - and the **slide-holds** witness, so the pass cannot be vacuous: the slip at the end of the
    hold is `>= 0.2 rad` (the commit is still alive) and the slip in the last 10 ticks of the
    release is `<= 0.02 rad` (the countersteer really did release it).
- **Where the straight line is measured — the D5 re-base.** The old form measured the straight line
  **before** the drift (the pre-drift acceleration phase). The W3 re-base measures it **AFTER a
  countersteer release**, because that is the only configuration in which the "the pose returns to
  zero" identity is meaningful: it is the *return*, not the *absence*, that the D5 model change
  (a neutral steer HOLDS the commit) put at risk.
- **The 7b split (D5).** The single "7b" window in the old design is now **7b-hold** and
  **7b-release**:
  - **7b-hold** — the **NEUTRAL** steer (the steering key is released and **nothing else** is
    pressed): the committed slide must **still be there** (`|slip| >= 0.2 rad` and
    `|pose| >= 0.05 rad`). This is the D5 contract: a neutral steer does **not** release the
    commitment;
  - **7b-release** — the **COUNTERSTEER** (the release-then-opposite-press): the slide must unwind
    (`|slip| <= 0.02 rad` in the last 10 ticks of the window) and the pose must return
    (the straight phase that follows). The keyup must come **first**; a simultaneous press never
    arms the countersteer, so the fixture records the key order and the check reads it.
- **Why the browser should still carry it**: the straight-line phase is the one place the browser
  can observe the pose *in isolation* (no slip, no articulation), so the check is meaningful and
  cheap. Without the seam the browser can only assert `max |bodyToTravel| <= 0.5°` on the straight
  — a weaker but still useful guard against a raw-steer residual (the ungated articulation left a
  −4.58° residual per research D3).

### 3.6 `driftFeelSlideProminence` — the browser-side, LOOSE AC-087 observable

- **NOT the AC-087 floor.** AC-087's floor is re-pinned on the deterministic unit proxy
  (Phase-A MERGE 7(c)). This browser check is the loose "it reads as a drift on the served page"
  observable.
- **Browser measurement**: `maxBodyToTravelRad` (the existing proxy, maximum over the drift phase)
  `>= SLIDE_PROMINENCE_MIN_RAD`.
- **Proposed bound**: the 14G baseline was **0.0494 rad**; the modelled new hold is **0.2658 rad**
  (≈ 5.4×). A browser floor of **0.10 rad** (≈ 2× the old baseline) is loose enough to survive the
  recorded render-rate spread (0.0494 / 0.0603 / **0.3599** rad) and still fails a build where the
  slide did not get more prominent.
- **The branch hazard (the implementation-pattern-scout's critical flag)**: a bare magnitude floor
  on the proxy can be satisfied by a **vanished** slide on the wrong visual branch. The Phase-A
  MERGE item 7(g) supersedes the `GAIN_eff = +1.0` forbidden-null contract because the gain stays
  at 0.5 and the full model replaces it — but the fixture should still assert the **sign** of the
  body-to-travel angle so a mirrored/inverted branch cannot pass. Concretely: the body-to-travel
  angle must keep a **consistent sign** for the whole drift phase (no sign flips mid-drift) — a
  sign flip is the mirror, and §3.1's pop check is the *one* sanctioned sign flip, at the catch.

### 3.7 `tireGradientMagnitudeIsNonDegenerate` — AC-088, re-scoped to the fragment seam

- **The Step 14G residual**: the fixture asserted only the gradient **SHADER SOURCE** and the
  per-frame apply pattern `[0,1,1,1,1]`, while the **measured gradient was 0**
  (`upperAvg == lowerAvg == 0.28565`). The measurement is **blind BY CONSTRUCTION**: the mesh
  builder writes per-quad constant colours (car.pass.ts:541-544), so the VBO vertex colours can
  never show a gradient.
- **The correct seam**: the fragment stage. The five uniform names live in
  `CAR_LIT_UNIFORM_NAMES` (car.pass.ts:237-241) and are fed by `feedTireGradientUniforms()`
  (car.pass.ts:1464-1484), called from exactly two sites: **disabled** before the body draw
  (car.pass.ts:1579) and **enabled** before the 4-wheel group (car.pass.ts:1583). The fragment
  shader computes `factor = mix(dark, bright, t)` (car.pass.ts:324-327).
- **The narrowest fix (a ~6-line fixture-only change)**: extend the fixture's `recordUniform(loc,
  value)` function (**L151-159**) to also capture the four `uTireGradient*` floats. The
  `uniform1f` wrapper (**L189-197**) already calls `recordUniform` for **every** float uniform, and
  `recordUniform` already resolves the name through the COMPLETE `getUniformLocation` map
  (**L184-188**) — so the capture is nearly free; only the four extra assignments are new. The
  existing `uApplyTireGradient` special case at L193-196 stays as it is.
- **This complements, and does not replace, the existing `wheelTireShading` check** (fixture L919),
  which asserts `wheelShading.ok && fixedSpaceShader && tireGradientPatternOk` — i.e. the shader
  SOURCE and the per-car apply pattern `[0,1,1,1,1]`. That is exactly the residual Step 14G
  recorded: the *pattern* is asserted, the *magnitude* is not. The two checks together close
  AC-088.
- **The check**:
  - `uTireGradientBrightFactor !== uTireGradientDarkFactor` — a **non-degenerate** factor spread.
    Today: bright **1.7**, dark **0.35** (car.pass.ts:388/:391) — a spread of **1.35**.
  - `uTireGradientRadius > 0` — today `NEATRACING_WHEEL_HUB_HEIGHT_M` = **0.5625 m**.
  - `uTireGradientThreshold` in (0, 1) — today `80/255 ≈ 0.31373`.
  - the apply sequence is still `[0,1,1,1,1]` per car (the existing `carLitApplySequence`).
- **The proposed contract**: the factor spread `bright - dark >= 1.0` (today 1.35) — a **named
  minimum** so a future "tidy-up" that equalises them fails loudly instead of silently flattening
  the tire. **The RED slice pins the exact minimum.**
- **Important**: this check is **fully deterministic in the browser** (uniforms are set from
  constants every frame), so it is a legitimate Tier-2 check and does not need the §1 seam.

### 3.8 The fallback check set (Option B — no debug seam)

If the §1 observability seam is not adopted, the browser half reduces to:

| Kept | Dropped (and why) |
|---|---|
| `tireGradientMagnitudeIsNonDegenerate` (deterministic) | `driftFeelPoseEnvelope` — `|pose|` is not observable |
| `driftFeelSlideProminence` (loose) | `driftFeelFarSidePop` / `driftFeelSettles` as **exact** contracts — only the loose shape survives, and the shape is ambiguous between the old mirrored model and the new one |
| `driftFeelNoStuckOffset` (loose, on `bodyToTravel`) | `driftFeelPoseRateBounded` — a per-frame rate bound is not the per-tick contract and is refresh-rate dependent |
| `driftFeelZeroStraightLineYaw` (loose: `|bodyToTravel| <= 0.5°`) | the exact zero identity (only jest can assert it) |
| the retained checks + the 3-frame drift gate | — |

**Option B MUST be recorded in the plan as a deliberate downgrade**, and the user gate (§5) then
carries the whole feel verdict. It is strictly weaker and must not be presented as equivalent.

---

## 4. The scenario changes the fixture needs

### 4.1 The re-based phase sequence (W3) — tick budgets, not frame counts

The current scenario is **A (baseline) → D (baseline) → W (accelerate) → W+A (100-frame drift)**.
There is **no countersteer phase**, so the pop, the ring and the settle are **not observable at
all** today. A new phase is required — and the W3 re-base changes its shape.

**Three controlled runs.** The two NAMED capture points (the AC-079 pair) are measured in **two
runs that each start from the SAME fresh spawn state and each run the SAME tick budget**, so the
two capture points are comparable by construction (the same elapsed ticks from the same state) —
not by hope. A **third** run (the mirror) follows on its own fresh spawn state; it is not a
capture pair, it only carries the OTHER steer sign.

**The ribbon re-entry (the AC-033 restart carrier).** The drift manoeuvre needs road, and the
spawn pose alone does not stay on the ribbon for the whole accel window. The production session
re-seeds `state.car = createCar()` at the spawn pose — **ON** the ribbon — on a mode switch
(`examples/neatRacing/host/race-session.ts:219-225`), so the fixture uses that carrier to return
the car to the ribbon before each run. The fixture therefore performs **three** re-entries (one per
run) and **records each one's outcome** in the evidence notes.

**The tick budgets (physics ticks, not frames).** The budgets are derived from a simulation over
the REAL track (seed 42, `large`) that runs the production physics module, the production track
generator and the production surface resolver — so the numbers below are the production
behaviour, not a guess:

| # | Run | Phase | Keys | Ticks | Purpose |
|---|---|---|---|---|---|
| P0 | — | boot + mode switch + **ribbon re-entry #1** | — | — | the fresh spawn state on the ribbon |
| P1 | 1 | **P1-STRAIGHT** | `W` only | **210** | the straight-line capture: the throttle only, the speed **UNCAPPED**; the NAMED `'straight'` capture point is the end of this phase |
| P2 | 2 | **ribbon re-entry #2** | — | — | the fresh spawn state again |
| P3 | 2 | **P2-ACCEL** | `W` only | **140** | the throttle down, the car above the drift engagement gate |
| P4 | 2 | **P2-COMMIT** | `W` + `A` | **30** | the commit: the slip ramps to the committed 0.45 rad |
| P5 | 2 | **P2-HOLD** | `W` only (the **neutral** steer) | **40** | **7b-hold**: the committed slide must STILL be there; the NAMED `'drift'` capture point is the end of this phase (the same elapsed tick as `'straight'`) |
| P6 | 2 | **P2-RELEASE** | `W` + `D` (release `A` **first**) | **30** | **7b-release**: the countersteer; the slip unwinds at 2.5 rad/s toward the 0.02 rad snap band (0.45 rad needs ~12 ticks, so the window is 2.5× that) |
| P7 | 2 | **P2-SETTLE** | `W` only (release `D`) | **40** | the ring-down (the settle tail = the last 30 ticks) |
| P8 | 2 | **P2-STRAIGHT** | `W` only | **20** | the straight line **AFTER** the release: the pose must be back at the hard zero |
| P9 | 3 | **ribbon re-entry #3** | — | — | the fresh spawn state for the mirror |
| P10 | 3 | **P3-ACCEL** | `W` only | **120** | the mirror's accel budget (the mirror is not a capture pair, so it does not have to match the P2 tick count, but it has to clear the drift gate and stay on the ribbon) |
| P11 | 3 | **P3-MIRROR-COMMIT** | `W` + `D` | **30** | the mirror's commit (the other steer sign) |
| P12 | 3 | **P3-MIRROR-HOLD** | `W` only (the **neutral** steer) | **40** | the mirror's 7b-hold |
| P13 | 3 | **P3-MIRROR-CATCH** | `W` + `A` (release `D` **first**) | **30** | the mirror's 7b-release (the same genuine release-then-opposite-press) |
| P14 | 3 | **P3-MIRROR-SETTLE** | `W` only (release `A`) | **40** | the mirror's ring-down |
| P15 | 3 | **P4-RECHECK** | `W` only | **120** | the straight-line re-check: the pose must be back at the hard zero again |

**The capture window is NOT speed-governed; the release/settle/straight window IS.** The old
fixture capped the straight run at 120 km/h and left the drift run uncapped, so the pair compared
a **capped** run against an **uncapped** one — the measured "deficit" was the cap, not the
scrub. In the re-based fixture the cap (`FEEL_STRAIGHT_MAX_SPEED_KMH = 120`) only acts from
`P2-RELEASE` onwards, in **both** runs, so the two capture points are read at the same
free-acceleration speed.

**The throttle STAYS DOWN through the straight phases, on purpose**: a coasting car drops below
the drift gate, the drift disengages and the pose snaps to 0 for the **wrong reason** — the
vacuous pass that the re-based expectation forbids.

**The countersteer must be a genuine release-then-opposite-press**, not a simultaneous press: the
new model latches `turnSign` at drift entry (research D1 stage 0) and treats an opposite steer as
the countersteer. A `keyup('KeyA')` followed by a `keydown('KeyD')` is the honest input, and the
fixture **records the key order** so the check reads the recorded order rather than assuming it.

**The per-tick input is IMPOSSIBLE, and the fixture says so.** The host samples the keyboard
**once per rAF frame** and replays the same sample into every catch-up tick
(`examples/neatRacing/host/race-session.ts:200-256`), so the fixture **cannot** apply a steer per
tick. The honest implementation is: flip the keys at **tick-ALIGNED** boundaries (wait for the
seam's `tick` to advance past a target before and after each key event) and **report the residual
alignment** in the evidence notes. The `FEEL_FRAME_WATCHDOG_PER_TICK` / `FEEL_TICK_ALIGN_MAX_FRAMES`
guards turn a stalled tick into a loud failure instead of a silently short phase.

**The existing `observeFrames` / `observeUntil` helpers and `dispatchKey` are sufficient** — the
new `observeFeelPhase` helper is the tick-budgeted version of `observeUntil` (it waits on the
seam's `tick`, not on wall-clock frames).

**The 3-frame drift gate regression guard** (Phase-A MERGE 7(f)) — `DRIFT_PROXY_MIN_FRAMES = 3`
with `DRIFT_PROXY_MIN_RAD = 0.01` (fixture L128 / L132) — is **retained unchanged** and is
re-measured under the new model. It is the cheapest guard that the slide is still a *sustained*
slide and not a one-frame twitch.

### 4.2 A model-identity check (recommended, requires the §1 seam) — **RE-SHAPED by W3**

**This is the check that makes the browser half trustworthy.** Without it, a green fixture could be
measuring the OLD model (a stale bundle, or a build where the new model is not wired into the
served page) and would still pass every loose shape check.

**The discriminator**: the new model **bounds** the slip at `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD =
0.45 rad`; the OLD model grew the slip **without bound** (research D3: **−65.8° = −1.15 rad** at
89 m/s, still growing). The W3 re-base pins that discriminator as a **REGIME-AWARE** bound on the
**PUBLISHED** slip, not as the old flat `0.45 + 0.02 = 0.47` (an UNRE-declared approximation that
let the 0.5038 rad off-road artifact through).

**The two regime-aware mirrors (D6, ratified as the FALLBACK):**

| Regime | Bound | Kind | Derivation |
|---|---|---|---|
| **ON-ROAD** | **0.452 rad** | **DERIVED** | the committed budget `0.45` + the chase-lag overshoot allowance `0.002` (the published slip peaks slightly above the committed budget because the chase lags a target that shrinks as the scrub bleeds speed). The jest lane's measured on-road peak is **0.4511617369362305 rad** ⇒ **0.00084 rad** of headroom. |
| **OFF-ROAD** | **0.493 rad** | **DECLARED** | the smallest 0.001-grid value at or above the W2a post-RC-1 raw off-road peak **0.4872167352892035 rad** + the **0.005 rad** margin, i.e. the FORMULA `Math.ceil((0.4872167352892035 + 0.005) * 1000) / 1000`. |

- **The mirrors are FORMULAS, not literals** (the W2b gap report's rule: "the fixture mirrors must
  be formulas, not literals"). The browser fixture cannot import the TypeScript exports, so it
  re-derives each bound from the same two named components the export is derived from. A
  re-declaration of the named export only needs the measured peak and the margin to be updated in
  the fixture.
- **The 0.45 COMMITTED BUDGET IS NOT RELAXED.** The mirrors are the **published-slip assertion
  bounds**; the committed `0.45 rad` budget is untouched, and the 0.002 rad overshoot allowance
  is the ONLY slack above it.
- **The regime is decided by the phase that carried the published-slip peak**: the ON-ROAD mirror
  when that phase was on the ribbon, the OFF-ROAD mirror when it was not, and the **widest
  declared bound** when the regime could not be measured (a missing measurement must never
  *tighten* the assertion). The classifier is the **RENDERED** ribbon (the distance from the
  phase position to the road mesh's own teal edge and fill vertices), so the classification reads
  the road the browser actually drew; the seam still carries no surface field, so the
  classification is a **geometric measurement**, not a model read.
- **The D6 publish-site CLAMP is DROPPED.** There is no clamp in the production publish path, so
  the fixture must not assert a **clamp-inert equality** (an equality that a clamp would make
  trivially true). The equality died with the clamp; what remains is a **one-sided** assertion
  (`max |published slip| <= the regime-aware bound`) plus the diagnostic evidence.

```
driftFeelModelIdentity:
  publishedSlipPeak !== null                                   // the peak was MEASURED
  AND moveSlip.maxRad <= publishedSlipBoundRad                 // the REGIME-AWARE bound
  // where publishedSlipBoundRad is chosen by the peak's phase regime:
  //   on-road -> 0.452 (DERIVED)   off-road -> 0.493 (DECLARED)
  //   unmeasurable -> 0.493 (the widest DECLARED bound)
```

**A second, independent identity signal**: the new model's **countersteer CATCHES** — the slip
returns toward zero within ~0.30 s of the countersteer (research D3: −25.94° → −18.96° → −7.01°
→ **0.00°**), whereas the old model **mirrored** it (−65.8° → −54.4° → **+10.6° → +28.1°**). The
discriminator in the browser is: **the slip magnitude after the catch must be SMALLER than before
the catch** (a catch), not larger (a mirror). That is measurable without the seam *if* the seam
exposes `slipAngleRad`; without the seam, only the loose `bodyToTravel` bound of §3.1 is available.

**Recommendation**: adopt the seam. It is one assignment per frame and it converts the entire
browser half from approximate to exact.

### 4.3 Fixture staleness audit (the Step 14G lesson — MANDATORY)

Step 14G needed **fix-packet iteration 12** to close **three measured fixture staleness defects**
(the stale track-program literal, the stale 6-float track-VBO stride, the stale wheel-VBO capture
heuristic) plus two latent and two newly surfaced defects. The 14N change touches the very seams
the fixture depends on:

| Fixture dependency | 14N change | Staleness risk | Required check |
|---|---|---|---|
| the `uniform1f` hook's `currentTireGradientApply` is **sticky** and is read inside the `drawArrays` hook (L259) | the tire-gradient feed may be **extracted** into `car.pass.tire-gradient.utils.ts` (boundary-mapper proposal) and the setter ORDER may change | a reordered feed would mis-attribute the apply flag to the wrong draw | re-verify the per-car apply sequence is still `[0,1,1,1,1]` and that the body draw precedes the wheel group |
| the wheel-VBO capture heuristic (`pickWheelBufferIdThisFrame`, L587-595) | the car mesh is unchanged, but the drift model changes the **wheel matrix chain** (the wheels inherit the body pose) | the capture still works, but the **relative yaw** it measures is now pose-dominated | re-measure `tireYawFollowsBodyDuringDrift` against the new pose |
| `carLitModelsThisFrame` positional layout (body = 0, wheels = 1..4) | unchanged, but the **glass pass** may start reading the drift yaw (the canopy decision) | a glass draw would add a 6th matrix with a different program | the capture is car-only by program source (boundary-mapper confirmed), but re-verify the array length assumption after any car.pass refactor |
| the drift-proxy window constants (L128 / L132 / L146) | the new model changes the proxy's MEANING (`|pose - slip|` instead of `|slip| * |GAIN - 1|`) | the retained 3-frame gate could flip red for a reason unrelated to the slide | re-baseline the gate against the new model and record the new spread |

**Rule**: the 14N fixture change is NOT accepted until the fixture has been audited against the
FINAL bytes of the served bundle, with the audit recorded in the plan.

### 4.4 The re-baselined AC-079 check — **RE-PINNED by W3**

The Phase-A MERGE item 5 lifts the MUST-NOT-MOVE pin: the new model changes the trajectory **by
design**, so `proportionalTractionLoss` (fixture L935-937, ceiling `TRACTION_LOSS_MAX_SPEED_KMH =
290`) must be **re-baselined**, not preserved. Research D3 predicts the new speed scrub at
**10.0 m/s²** at 89 m/s (vs **24.2 m/s²** today) with the engine still winning at 17.8 m/s², so
the car should NOT reach the 290 km/h ceiling during the drift phase. The check **stays** (the
ceiling contract is still valuable) but the RED slice must re-pin the expected peak and the
ceiling value.

**The W3 re-pin (the browser lane).** The browser's D1.3 witness is now a **same-regime**
witness plus a **deficit floor** plus a **ceiling**:

```
proportionalTractionLoss (browser lane, re-pinned):
  tireYawDriftEngaged                                          // the retained witness
  AND straightCaptureOnRoad AND driftCaptureOnRoad             // the SAME regime
  AND !captureRegimeSplit                                      // and NOT a regime split
  AND driftSpeedDeficitKmh !== null
  AND driftSpeedDeficitKmh >= DRIFT_MIN_SPEED_DEFICIT_KMH      // the retained 20 km/h FLOOR
  AND maxSpeedKmhDrift <= TRACTION_LOSS_MAX_SPEED_KMH          // the re-pinned absolute ceiling
```

- **The 20 km/h floor is ACHIEVABLE on-road** under the re-based budgets: the simulation over
  the real track measures the straight capture at **224.28 km/h** and the drift capture at
  **198.66 km/h** ⇒ a deficit of **25.62 km/h (11.42 %)** with a jitter range of
  **[24.10, 27.19] km/h** and **0 violations** across the jitter sweep. The floor is a
  **retained contract**, and the re-based fixture finally produces the evidence for it.
- **The deficit is carried by the SLIDE LENGTH, not by the surface regime.** The old story ("the
  drift capture is off-road, so the off-road decel makes the deficit") is **false** for this
  manoeuvre: the drift capture is **on-road** (the world-frame lateral offset at the capture is
  only 0.05 m), and the deficit is the longitudinal **scrub** of the committed slide
  (`computeLongitudinalSpeedScrub` reads the tick-start published slip and the tick-start speed
  percentile, and the floor applies to the POST-scrub speed).
- **The ceiling is the retained ABSOLUTE-speed ceiling, re-baselined** (the old `290 km/h` was
  lifted by the Phase-A MERGE item 5; the re-pinned `TRACTION_LOSS_MAX_SPEED_KMH` is **340 km/h**,
  a loose bound that clears the measured 320.4 km/h straight-line top speed by ~6 %). It bounds
  the drift speed's **absolute** value, not the deficit: a ceiling-only form would judge
  proportionality for a reason that has nothing to do with it, which is exactly why the D1.3
  re-semantic added the deficit floor.
- **The pair MUST be comparable**: the old fixture capped the straight run at 120 km/h and left
  the drift run uncapped, so the "deficit" it measured was the **cap** (a 100+ km/h artifact),
  not the scrub. In the re-based fixture the capture window is **uncapped in both runs** and the
  cap only acts from the release onwards (§4.1).

### 4.5 The W3 re-base record (the fixture's authoritative contract)

The W3 slice re-bases the browser half of Step 14N. Four contract items, each with the place it
lands:

| # | W3 contract item | Where it lands | The change |
|---|---|---|---|
| 1 | **the `tick` field, in all 3 places together** (the seam, the fixture, this doc) | `browser-entry.ts` (the seam interface + the per-frame assignment + the `Window` declaration), the fixture's `sampleDriftSeam`, this doc §1 | the seam is **EXACTLY five fields**: `poseRad`, `slipAngleRad`, `speedMps`, `tickHz`, **`tick`** (`car.tick`, read-only). The `api-contract-reviewer` co-signs the 3-place change. **Step 14P extends this to SIX fields** (the sixth is `followOffsetRad`, §4.6) and **Step 14Q2 extends it to SEVEN** (the seventh is `tractionSeverity`, §4.7) — the W3 row is left as the W3 record. |
| 2 | **the D5 re-base** (commit → countersteer release → measure the straight line; **7b-hold / 7b-release**) | the fixture's phase block, this doc §3.5 and §4.1 | a **neutral** steer **HOLDS** the commit (7b-hold: the slide must still be there); the **countersteer** releases it (7b-release: the slip unwinds to the snap band); the straight line is measured **AFTER** the release. The single old "7b" window is now two windows. |
| 3 | **regime-aware re-derived mirrors** | the fixture's mirror constants, this doc §4.2 | the on-road mirror **0.452** (DERIVED) and the off-road mirror **0.493** (DECLARED), both as **formulas** over the named components; the regime is chosen by the peak's phase and classified against the **rendered** ribbon; the D6 **clamp is dropped**, so the fixture asserts a **one-sided** bound and **no clamp-inert equality**. The 0.45 committed budget is **not** relaxed. |
| 4 | **the tick budgets + the per-tick input** | the fixture's phase budgets, this doc §4.1 | the budgets are **physics ticks** (210 / 140 / 30 / 40 / 30 / 40 / 20 for the two capture runs, and 120 / 30 / 40 / 30 / 40 / 120 for the mirror), derived from a simulation over the **real** track; the per-tick input is **impossible** (the keyboard is sampled once per rAF frame and replayed into every catch-up tick), so the fixture flips the keys at **tick-ALIGNED** boundaries and **reports the residual alignment**. |

**The simulation that produced the budgets.** The budgets and the measured values are not
estimates: the production `physics.step.service.ts`, the production `drift.model.ts`, the
production track generator and the production surface resolver were compiled and run against the
**real** track (seed 42, `large`) — the same track the browser lane drives. The authoritative
measurements:

| Measurement | Value |
|---|---|
| the straight capture (210 ticks, uncapped) | **224.28 km/h**, lateral **0.00 m**, on-road |
| the drift capture (140 + 30 + 40 ticks) | **198.66 km/h**, the pose **0.2247 rad**, the slip **0.4448 rad**, lateral **−0.05 m**, on-road |
| the D1.3 deficit | **25.62 km/h (11.42 %)**, the jitter range **[24.10, 27.19] km/h**, **0 violations** of the 20 km/h floor |
| the far-side pop | **2.245°** (the left slide) / **2.212°** (the mirror) — inside the [1.0°, 6.0°] browser bound |
| the settle tail (the last 30 ticks) | **0.00304 rad** (the left slide) / **0.00295 rad** (the mirror) — inside the 0.005 rad bound |
| the straight phase after the release | **0.000076 rad** (the left slide) / **0.000075 rad** (the mirror) — inside the 0.002 rad bound |
| the release's slip tail (the last 10 ticks) | **0** — inside the 0.02 rad bound |
| the published-slip peak | **0.44479 rad** in the hold (on-road) — inside the 0.452 rad on-road mirror |
| the mirror's published-slip peak | **0.44353 rad** — inside the 0.452 rad on-road mirror |

**The W3 build (the seam has to be LIVE in the served bytes, not only in the source).** The drift
seam is a **runtime** contract: the browser fixture loads the **built** bundle
(`docs/assets/neatracing.bundle.js`), so a source-only change would leave the served page with
the OLD four-field seam, `seam.tick === undefined`, the tick budgets silently degraded to frame
budgets, and `driftFeelPoseRateBounded` reporting false. W3 therefore rebuilds the artifact
(`npm run build:neatRacing`) and refreshes the tracked copy — the bundle is a **build output**
(never hand-edited), and the green slice's own **G2** ("`npm run build:neatRacing` exits 0") and
**G3** ("the bundle identity — the byte length and the sha256 recorded before and after") criteria
own it.

| Artifact | Bytes | sha256 |
|---|---|---|
| `docs/assets/neatracing.bundle.js` (the fixture's load target) — **W3 (BEFORE Step 14P)** | **398,699** | `3C255A7F91A99AA8B3C7172B2BADCD23426DC63746A286E81778A16E33FFD9D8` |
| `examples/neatRacing/assets/neatracing.bundle.js` (the tracked copy) — **W3 (BEFORE Step 14P)** | **398,699** | `3C255A7F91A99AA8B3C7172B2BADCD23426DC63746A286E81778A16E33FFD9D8` |
| `docs/assets/neatracing.bundle.js` (the fixture's load target) — **Step 14P (AFTER)** | **400,394** | `A18DAD2B1A947F6DB3FA83FDF11D5D5848118BB047FF10C97CBC55CB561BE222` |
| `examples/neatRacing/assets/neatracing.bundle.js` (the tracked copy) — **Step 14P (AFTER)** | **400,394** | `A18DAD2B1A947F6DB3FA83FDF11D5D5848118BB047FF10C97CBC55CB561BE222` |
| `docs/assets/neatracing.bundle.js` (the fixture's load target) — **Step 14Q2 (BEFORE)** | **400,941** | `5CEDAB2746CECEC7AEBE63EF7CD4B379D16995B1E36056F950860453D95A2A93` |
| `examples/neatRacing/assets/neatracing.bundle.js` (the tracked copy) — **Step 14Q2 (BEFORE)** | **400,747** | `662D1689D0769A1837160EF386056890A3B898BB2F4212AEA67A045937A3EF65` |
| `docs/assets/neatracing.bundle.js` (the fixture's load target) — **Step 14Q2 (AFTER)** | **400,941** | `5CEDAB2746CECEC7AEBE63EF7CD4B379D16995B1E36056F950860453D95A2A93` |
| `examples/neatRacing/assets/neatracing.bundle.js` (the tracked copy) — **Step 14Q2 (AFTER)** | **400,941** | `5CEDAB2746CECEC7AEBE63EF7CD4B379D16995B1E36056F950860453D95A2A93` |
| `docs/assets/neatracing.bundle.js` (the fixture's load target) — **Step 14R (BEFORE)** | **402,024** | `0C968DB7337A67E3B827C124F202375C8C7321656D04CF8C703D9803A2BBCA68` |
| `examples/neatRacing/assets/neatracing.bundle.js` (the tracked copy) — **Step 14R (BEFORE)** | **402,024** | `6187B2A8EC450B3D55198668FF83B9FE13C06A15376BB79E6C7696346E4219BE3` |
| `docs/assets/neatracing.bundle.js` (the fixture's load target) — **Step 14R (AFTER)** | **402,024** | `0C968DB7337A67E3B827C124F202375C8C7321656D04CF8C703D9803A2BBCA68` |
| `examples/neatRacing/assets/neatracing.bundle.js` (the tracked copy) — **Step 14R (AFTER)** | **402,024** | `0C968DB7337A67E3B827C124F202375C8C7321656D04CF8C703D9803A2BBCA68` |

**The Step 14Q2 rebuild record (the 14P precedent, followed again).** The bundle is a **build
output** (never hand-edited): `npm run build:neatRacing` (exit 0) rebuilt the **docs** copy, and a
**SECOND, IDENTICAL** esbuild run with `--outfile=examples/neatRacing/assets/neatracing.bundle.js`
refreshed the **tracked** copy. The two copies are now **byte-identical**
(400,941 B / `5CEDAB…2A93`).

**The Step 14R rebuild record (the same 14P / 14Q2 precedent, followed again).** The bundle is
a **build output** (never hand-edited): `npm run build:neatRacing` (exit 0) rebuilt the **docs**
copy, and a **SECOND, IDENTICAL** esbuild run with
`--outfile=examples/neatRacing/assets/neatracing.bundle.js` refreshed the **tracked** copy. The
**docs** copy was already current (**402,024 B / `0C968DB7…CA68`** — it carried the Step 14R
live-collision wiring) and is **UNCHANGED** by the rebuild; the **tracked** copy was **STALE**
(**402,024 B / `6187B2A8…9BE3`** — the Step 14R HUD slice's build, taken BEFORE the live-collision
wiring landed in `physics.step.service.ts`) and now matches the docs copy **byte for byte**
(**402,024 B / `0C968DB7…CA68`**). The two `.map` copies differ ONLY in the relative `sources`
path depth (2,188,935 B vs 2,182,145 B); their `mappings` and `sourcesContent` are **IDENTICAL**,
so it is a path artefact, not a build divergence.

**The tracked copy was STALE before this slice's rebuild, and that is a RECORDED observation, not
a silent repair.** The 14P row above is a **different** identity (400,394 B /
`A18DAD2…E222`), so the tree had already moved on: the **docs** copy had been rebuilt by the
Step 14Q / 14Q2 code slices (it carried the **seven-field** seam — 5 occurrences of
`tractionSeverity`), while the **tracked** copy still carried the **six-field** seam (**0**
occurrences of `tractionSeverity`, 400,747 B). A fixture served from the tracked copy would have
observed the OLD six-field seam, and the new 45th check would have had nothing to read. The
rebuild is what closed that gap, and the AFTER row is what the gate reads.

**The built seam literal is now the seven-field** form
(`neatRacingDriftState={poseRad:…, slipAngleRad:…, speedMps:…, tickHz:…, tick:…,
followOffsetRad:…, tractionSeverity:…}`) — the seventh field is the Step 14Q2 PUBLISHED traction
severity, READ from the physics state field (`car.tractionSeverity`) and **never re-derived** in
the browser, and the `neatRacingDriftState` name appears **EXACTLY ONCE** in the built bytes. The
write is a **stable bag** (ONE object, re-pointed at the window each frame), so the
zero-per-frame-allocation rule holds. The 14G precedent's 397,364 B is a **reference, not a
target** (§9) — W4 records the **final** identity after the gate build.

**The Step 14R built seam literal is the EIGHT-field** form — the built bytes read
`…g.poseRad=C.drift.pose.pose,g.poseRateRadPerS=C.drift.pose.poseRate,g.slipAngleRad=C.slipAngle,g.speedMps=C.speed,g.tickHz=1/W0,g.tick=C.tick,g.followOffsetRad=M.offsetRad,g.tractionSeverity=C.tractionSeverity,window.neatRacingDriftState=g`.
The **eighth** field is the **PUBLISHED pose-spring RATE** (`C.drift.pose.poseRate`), READ from the
physics state and **never re-derived** in the browser; the **seventh** field is still the
**PUBLISHED traction severity** (`C.tractionSeverity`), so the severity is **NOT** re-published as
the eighth. The `neatRacingDriftState` name appears **EXACTLY ONCE** in the built bytes and the
write is still a **stable bag** (ONE object allocated once, refreshed IN PLACE, ONE assignment per
driven frame), so the zero-per-frame-allocation rule holds.

**The plan-side amendment (the seam count in the plan text).** The five-field seam also required
the PLAN TEXT to move, or the plan and the fixture would have stated different field counts for
the same seam: the live tune-state criterion (plan L2423), the live RED-slice criterion (L2375),
the AC-087 coverage-backlog row (L482) and the blocker-table B-3 row (L8984) now all state the
**five-field** seam; the PATCH B-3 entry's "the 4-field browser debug seam stays exactly the
fixture-gate fields … the debug seam does NOT grow a fifth field" clause is **WITHDRAWN** by a
dated SUPERSEDED bullet (L4004) that does **not** reopen the `reArmed` amendment. The DATED
VERIFICATION records that still carry the 4-field claim are **LEFT INTACT** for the audit trail
and are **MARKED, not rewritten**, by that same bullet.

**The dead legacy seam branch is REMOVED (No Deferred Cleanup).** The fixture's `readDriftSeam()`
read `window.neatRacingDebug.drift` **first** — a path that is written **NOWHERE** in the repo. It
is removed in the **same** step that introduced the fifth field: had that 4-field shape ever been
re-introduced it would have been read *instead of* the five-field seam, `tick` would have been
`undefined`, and the tick budgets would have silently degraded to frame budgets — exactly the
failure mode the W3 re-base exists to prevent. The seam is now read from **ONE** place.

**The W3 residual risks (carried forward, not closed by W3).**

1. **The per-tick input is not achievable** (item 4 above): the residual alignment is reported,
   not eliminated. W4 records the observed alignment in the evidence.
2. **The mirror run's re-check tail leaves the ribbon** (the simulation: the re-check ends
   off-road). No assertion depends on that phase's regime, and the regime-aware mirror only
   follows the **peak's** phase — but W4 should read the per-phase regime table before trusting
   a red.
3. **The `drift.model.test.ts` export coverage** for the two named published-slip exports is
   **outside the W2b write boundary** and is **not** closed by W3. The fixture re-derives both
   mirrors as formulas, so the fixture does not depend on it — but the jest lane's export-level
   coverage of the named constants is still owed.
4. **No browser run happened in W3** (W4 owns it): every number above is from the production
   simulation, not from the served page. The fixture's first served-page run is the first
   measurement of the *rendered* path, and it may legitimately differ inside the declared
   headroom.

### 4.6 The Step 14P re-base record (the SIXTH seam field and the 44th named check)

Step 14P (the travel-camera slice, `P1-drift-rewire-travel-camera`, and this green slice,
`P1-drift-rewire-green`) re-bases the browser half of the 14N wave the same way W3 re-based it: the
seam grows by **EXACTLY ONE** field, the reader grows with it, and this document moves with both.

| # | 14P contract item | Where it lands | The change |
|---|---|---|---|
| 1 | **the `followOffsetRad` field, in all 3 places together** (the seam, the fixture, this doc) | `browser-entry.ts` (the per-frame assignment), the fixture's `sampleDriftSeam`, this doc §1 and §4.5 | the seam is **EXACTLY six fields**: `poseRad`, `slipAngleRad`, `speedMps`, `tickHz`, `tick`, **`followOffsetRad`**. The field is the chase camera's own share of the shared slip scalar (the camera's aim offset from the nose heading), written **ONCE per frame** from the ONE aim law, in the same closure that steps the follow — so the published offset and the published slip are from the **same tick**. |
| 2 | **the reader grows to the six fields** | the fixture's `sampleDriftSeam` | the reader reads all six, treats a non-number field as `undefined`, returns `null` only when **all six** are undefined, and publishes `followOffsetRad` on every sample. The seam is still read from **ONE** place. |
| 3 | **the ONE new named check** | the fixture's `result.checks` | `driftFeelCameraFollowOffset` — the named-check count grows by **EXACTLY ONE** (**43 → 44**). The count is **ASSERTED against the ACTUAL `result.checks` length** and is never hard-coded; `result.checkNames` is published next to the count so the gate can assert the exact **SET**, not only the total. |
| 4 | **the bundle identity** | §4.5's table | `npm run build:neatRacing` (exit 0) plus the tracked copy's refresh: **398,699 B / `3C255A…D9D8`** (BEFORE) → **400,394 B / `A18DAD2…E222`** (AFTER), **byte-identical in both copies**. The six-field seam literal appears **EXACTLY ONCE** in the built bytes. |

**The new check's contract (why it is not vacuous).** The chase camera aims along the PUBLISHED
travel direction, so its follow offset (a) **leaves the nose** by a measurable amount (the floor,
0.05 rad ≈ 2.9°), (b) **never swings past** the published slip it tracks (the ceiling: the follow
is a convex combination of the target history, so `|follow| ≤ max|published slip|` always, plus the
0.002 rad render-jitter headroom), and (c) **closes the aim error** — the published slip minus the
follow offset, which is exactly the camera's remaining divergence from the travel direction — to
at or below **0.1 rad** (≈ 5.7°) at least once during the drift, where the camera points the
**same way** as the direction the car is travelling. A camera pinned to the nose (the frozen-camera
defect) reports a zero offset: the floor fails, the convergence count stays at zero and there is
no sign to compare. A camera that over-rotated past the travel direction breaks the ceiling. A
camera that never tracks keeps the aim error open. The comparison floors are the published slide
(≥ 0.2 rad — well over the ≤ 6.0° far-side pop the catch is allowed to produce) and the follow
offset (≥ 0.01 rad — a camera that has not started to move carries no sign).

**What Step 14P does NOT claim.** The browser lanes of this wave were **NOT RUN** under the
code-only mandate of the green slice: the served-page run (§5.1's preconditions) and the
visible-window fixture run with `passed === true` are **DEFERRED**, not passed. The named check is
**authored and syntax-checked**, not **measured**; the first measurement of the 44th check is the
first served-page run. The AC-090 user verdict is likewise **deferred** and is not claimed here.

**The audit constraint that is still OPEN (a carried RISK, not a closure).**
`plans/neatracing-voxel-racer.audit-visuals.research.md` (D5 / R5) requires that the sixth seam
field carry the **shared traction scalar** and that the 44th check be **RATIO-shaped**, because "a
6th field carrying only the follow offset would let the fixture report GREEN while the tires show
nothing". The 14P plan decision pins the sixth field to the **follow offset** and defers the
traction severity to Step 14Q (the seam grows 6 → 7 there). The 44th check is therefore
**camera-shaped, not ratio-shaped**, and this record does **not** claim the D5 / R5 constraint is
closed. The traction-severity ratio check is owed by the slice that grows the seam to seven fields.

### 4.7 The Step 14Q2 re-base record (the SEVENTH seam field, the 45th named check and the browser envelope re-pin)

Step 14Q2 (the SEAM WRITER slice `P1-drift-traction-seam` and this green slice,
`P1-drift-law-green`) re-bases the browser half of the 14N wave the same way W3 and 14P re-based
it: the seam grows by **EXACTLY ONE** field, the reader grows with it, and this document moves with
both. The seam WRITER and the seam READER live in **different slices** (the 3-code-files cap
forbids them in one slice — the 2026-10-27 VERIFICATION 4 blocker B-1), so this record covers the
**READER** half and the **fixture** half.

| # | 14Q2 contract item | Where it lands | The change |
|---|---|---|---|
| 1 | **the `tractionSeverity` field, in all 3 places together** (the seam, the fixture, this doc) | `browser-entry.ts` (the per-frame assignment — the SEAM WRITER, slice `P1-drift-traction-seam`), the fixture's `sampleDriftSeam`, this doc §1 and §4.5 | the seam is **EXACTLY seven fields**: `poseRad`, `slipAngleRad`, `speedMps`, `tickHz`, `tick`, `followOffsetRad`, **`tractionSeverity`**. The field carries the **PUBLISHED** traction severity: the value is **READ** from the physics state field `car.tractionSeverity` (computed by `computeTractionSeverity`, `examples/neatRacing/physics/drift.model.ts`) and is **NEVER re-derived** in the browser. The write is a **stable bag** (ONE object, re-pointed at the window each frame), so the **zero-per-frame-allocation** rule holds, and the seam name appears **EXACTLY ONCE** in the built bytes. |
| 2 | **the reader grows to the seven fields** | the fixture's `sampleDriftSeam` | the reader reads all seven, treats a non-number field as `undefined`, returns `null` only when **all seven** are undefined, and publishes `tractionSeverity` on every sample. The seam is still read from **ONE** place. (The stable-bag consequence: the reader **copies** the fields into a new object per sample, so it never observes a later frame through a retained reference.) |
| 3 | **the ONE new named check** | the fixture's `result.checks` | `driftFeelTractionSeverityRatio` — the named-check count grows by **EXACTLY ONE** (**44 → 45**). The count is **ASSERTED against the ACTUAL `result.checks` length** and is never hard-coded; `result.checkNames` is published next to the count so the gate can assert the exact **SET**, not only the total. |
| 4 | **the browser DECLARED POSE ENVELOPE re-pin** | the fixture's `DRIFT_FEEL_POSE_ENVELOPE_RAD` | the browser literal moves **0.305 → 0.30185** on the **PUBLISHED** basis (`0.493 * 0.45 + 0.08` — the widest published-slip ceiling the production now publishes). The **2 SERVICE-TEST** envelope pins (`0.2025` and `0.2825`) stay on the **COMMITTED** basis — they were re-pinned in Step 14Q and are **NOT** touched here, and the two bases are **NEVER conflated**. The **FIVE** gain-coupled test-file literals are **NOT** taken here (they belong to the Step 14Q slice `P1-drift-law-repin`). |
| 5 | **the bundle identity** | §4.5's table | `npm run build:neatRacing` (exit 0) plus a **second, identical** esbuild run into the tracked copy: the **docs** copy **400,941 B / `5CEDAB…2A93`** (BEFORE and AFTER — it was already current) and the **tracked** copy **400,747 B / `662D16…F65` → 400,941 B / `5CEDAB…2A93`** (it was **STALE**, still carrying the six-field seam). The two copies are now **byte-identical**. |

**The new check's contract (why it is not vacuous, and why it is a RATIO).**
`plans/neatracing-voxel-racer.audit-visuals.research.md` (D5 / R5) requires that the traction check
be **RATIO-shaped, not a floor**, because "a seam field carrying only the follow offset would let
the fixture report GREEN while the tires show nothing". This slice pays that constraint:

- the **published** severity is the **shared law applied to the two published numbers** — the
  service computes it in the same tick from the **post-traction speed** and the **published slip**
  (`physics.step.service.ts`), the very two numbers the seam publishes — so the RATIO
  `published / oracle` is **1** for every sample of a correct build;
- the oracle is the **fixture's VERIFICATION MIRROR** of the shared law
  (`min(1, max(0, |slip| / 0.45 * speed / 89))`, the same "mirror the FORMULA over the named
  components, never hard-code a literal" rule the other 14N mirrors follow). It is **not** a
  consumer: no production value is derived from it and it is **never substituted** for the
  published severity — the published number is the number under test;
- the check also asserts that the severity is a **share** (inside `[0, 1]`), that it **responds to
  the slide** (never zero at a published slide of ≥ 0.2 rad — the same floor the camera check uses)
  and that at least one sample was actually ratio-checked (so the check cannot pass vacuously).

A severity that was always 1, always 0, stale, clamped to the wrong bound or re-derived in the
browser breaks the RATIO. A **floor** would have passed all of them — which is exactly why D5 / R5
forbids the floor form.

**What Step 14Q2 does NOT claim.** The browser lanes of this wave were **NOT RUN** under the
user's NO-browser / NO-CDP mandate: the served-page run (§5.1's preconditions) and the
visible-window fixture run with `passed === true` are **DEFERRED**, not passed. The 45th check is
**authored and syntax-checked** (the inline script parses; the static count of the
`result.checks` literal is **45**, with no duplicate names), **not measured**; the first
measurement of the 45th check is the first served-page run. The AC-090 user verdict is likewise
**deferred** and is not claimed here. The **equilibrium deficit** recorded in this wave is the
**MEASURED 32.911 km/h equilibrium** (asymptote 32.91154453692649 km/h, bit-identical at 3000 and
3600 ticks, 0.00054 km/h from the pinned 32.911 km/h inside the 0.002 km/h band) — the 600-tick
transient reading (32.556920153495916) is **never** the recorded value. The law evidence is
**MANUAL-mode** and **ROAD-ONLY**.

**The audit constraint that is now PAID (D5 / R5).** §4.6 left the D5 / R5 constraint **OPEN**:
the sixth field carried the follow offset and the 44th check was camera-shaped, not ratio-shaped,
so the traction-severity ratio check was owed by the slice that grows the seam to seven fields.
**This is that slice**: the seam carries the shared traction scalar as its **seventh** field and
the 45th check is **RATIO-shaped**. The constraint is **closed by this record** (subject to the
deferred browser lanes: the check is authored and syntax-checked, and its first **measurement**
belongs to the visible-window run).

### 4.8 The Step 14R re-base record (the EIGHTH seam field, the 46th named check and the HUD traction readout)

Step 14R (the SEAM WRITER slice `P1-drift-presentation-car` and this green slice,
`P1-drift-presentation-green`) re-bases the browser half of the 14N wave the same way W3, 14P and
14Q2 re-based it: the seam grows by **EXACTLY ONE** field, the reader grows with it **in the SAME
edit**, and this document moves with both. The seam WRITER and the seam READER live in **different
slices** (the 3-code-files cap forbids them in one slice — the 2026-10-27 VERIFICATION 4 blocker
B-1), so this record covers the **READER** half and the **fixture** half. The 46th check reads the
**RENDERED** HUD readout, so this record also carries the **overlay** half of the presentation wave.

| # | 14R contract item | Where it lands | The change |
|---|---|---|---|
| 1 | **the `poseRateRadPerS` field, in all 3 places together** (the seam, the fixture, this doc) | `browser-entry.ts` (the per-frame assignment — the SEAM WRITER, slice `P1-drift-presentation-car`), the fixture's `sampleDriftSeam`, this doc §1 and §4.5 | the seam is **EXACTLY eight fields**: `poseRad`, **`poseRateRadPerS`**, `slipAngleRad`, `speedMps`, `tickHz`, `tick`, `followOffsetRad`, `tractionSeverity`. The eighth field carries the **PUBLISHED pose-spring RATE** (`car.drift.pose.poseRate`, rad/s), **READ** from the physics state and **NEVER re-derived** in the browser. It is a **NAMED CHANNEL and it is NOT the traction severity**: the severity stays the **seventh** field and is **NOT** re-published. The write is a **stable bag** (ONE allocation, refreshed IN PLACE), the zero-per-frame-allocation rule holds and the seam name appears **EXACTLY ONCE** in the built bytes. |
| 2 | **the reader grows to the eight fields** + **the FIELD-COUNT ASSERTION** (the 2026-10-27 VERIFICATION 4 blocker B-3) | the fixture's `sampleDriftSeam` and `countSeamFields` | the reader reads all eight, treats a non-number field as `undefined`, returns `null` only when **all eight** are undefined, and publishes `poseRateRadPerS` on every sample. It **COUNTS the fields the seam actually publishes** (`Object.keys(seam).length`) and the run only passes when the count is **EXACTLY eight** — a seam with seven means the writer never landed, a seam with nine means a field grew without a re-base record. The count is published as `result.seamFieldCount` / `result.seamFieldCountSamples` / `result.seamFieldCountMatches` and is a **SEPARATE condition of `result.passed`** — it is deliberately **NOT** a named check, so the named-check count grows by **EXACTLY ONE** (45 → 46) and the gate can assert the exact field **SET** and not only the total. |
| 3 | **the ONE new named check** | the fixture's `result.checks` | `driftFeelHudTractionReadoutRatio` — the named-check count grows by **EXACTLY ONE** (**45 → 46**). The count is **ASSERTED against the ACTUAL `result.checks` length** (`result.checkCount`) and is never hard-coded; `result.checkNames` is published next to it so the gate can assert the exact **SET**, not only the total. |
| 4 | **the HUD traction readout read** (the overlay half of the presentation wave) | the fixture's `readHudTractionText` / `parseHudTractionPercent` / `measureHudTractionReadout` and the same-instant sampler | the fixture reads the **RENDERED** `Traction NN%` text out of the HUD DOM (`div > div > div:nth-child(4)` — the container's direct child is the HUD **wrapper** div, so the traction label is the wrapper's **fourth** child) and compares it with the **PUBLISHED** `tractionSeverity` of the **same frame** (the DOM read and the seam sample are taken in ONE rendered frame of the same car state). The fixture never re-derives the severity: it compares what the overlay **painted** against the number the simulation **published**. |
| 5 | **the bundle identity** | §4.5's table | `npm run build:neatRacing` (exit 0) plus a **second, identical** esbuild run into the tracked copy: the **docs** copy **402,024 B / `0C968DB7…CA68`** (BEFORE and AFTER — it was already current) and the **tracked** copy **402,024 B / `6187B2A8…9BE3` → 402,024 B / `0C968DB7…CA68`** (it was **STALE**). The two copies are now **byte-identical**. |

**The new check's contract (why it is not vacuous, and why it is an EQUALITY BAND).** The HUD
shows the **PUBLISHED** severity, so the rendered readout and the published share are the **same
number in two representations**:

- the **rendered** side is the whole percent the overlay painted (`Traction 42%` for a published
  `0.42`), parsed with the named pattern `/Traction\s+(-?\d+)\s*%/` — a text that does not match is
  a **PARSE FAILURE**, and a parse failure is a **failure of the check**, never a quiet fallback to
  another number in the overlay;
- the **published** side is `seam.tractionSeverity` of the **same frame** — the drive session hands
  `state.car.tractionSeverity` to the overlay and the **same** state's `car.tractionSeverity` is
  written into the seam bag in the same frame, so the pair is **same-instant** by construction;
- the **assertion is the absolute ±0.5 point display-quantization band**, NOT `ratio == 1`: the
  readout is a **ROUNDED** percent, so a published share of `0.004` renders as `0%` (a ratio of
  `0`) — that is the display quantization and not a defect. The band absorbs exactly that
  quantization and nothing else; a readout that is **stale**, **zeroed**, **clamped to the wrong
  bound** or **re-derived in the browser** is off by far more than half a point and **FAILS**;
- the check is **NON-VACUOUS**: it requires at least one **parsed** sample, **zero** parse
  failures, **zero** mismatches, at least one **ratio** sample (a published share above zero), a
  **non-zero** published severity at some point and a **non-zero** rendered percent at some point —
  so a run in which the overlay never showed the traction can never pass.

A **floor** ("the readout is above zero") would have passed every one of those defects — exactly
the D5 / R5 failure mode ("the fixture reports GREEN while the tires show nothing"). The check is
therefore **equality-shaped**, like the 14Q2 ratio check, and it compares the **one** published
number against its **one** rendered representation.

**What Step 14R does NOT claim.** The browser lanes of this wave were **NOT RUN** under the
user's NO-browser / NO-Chrome-DevTools-MCP / NO-CDP mandate: the served-page run (§5.1's
preconditions) and the visible-window fixture run with `passed === true` are **DEFERRED**, not
passed. The 46th check is **authored and syntax-checked** (the inline script parses; the static
count of the `result.checks` literal is **46**, with **no duplicate name**), **not measured**; the
first measurement of the 46th check is the first served-page run. The **seam field-count
assertion** is likewise authored and syntax-checked, not measured. The **AC-090 user verdict** is
likewise **deferred** and is **not** claimed here — the 2026-10-27 verdict stays **REJECT** and the
six questions are **re-armed** in §5.3. The evidence is **MANUAL-mode** and **ROAD-ONLY**.

**The retained named-check set is intact (the proof that the growth is EXACTLY ONE).** The current
fixture carries **46** named checks (**13** `driftFeel*` + **33** retained). The independent
**runtime** baseline (the published `checkNames` of the three real browser runs of 2026-10-27, 43
names) is a **subset** of the current set: **no name is missing** and there is **no duplicate**. The
three names added across 14Q2 + 14R are `driftFeelCameraFollowOffset` (14P),
`driftFeelTractionSeverityRatio` (14Q2) and `driftFeelHudTractionReadoutRatio` (14R) — **exactly
ONE per re-base**.

**THE PLAN'S "24 non-drift fixture checks" IS A STALE NUMBER (a RECORDED observation, never a
silent re-base).** The AC-089 PRESERVE SET text still names "the 24 non-drift fixture checks"; that
figure belongs to the 14N-era arithmetic (the 31-check pre-14N fixture minus the 7 proposed 14N
drift-feel checks). The fixture of today carries **33** non-`driftFeel` named checks, and **all 33
are retained** (0 removed, 0 renamed). The **PRESERVE SET itself is intact** — the number in the
plan text is stale, the checks it names are not. The plan text is **NOT** re-based by this slice
(a green slice does not re-write an AC row); the correction is recorded here and in the plan's
validation evidence for the next planning pass.

---

AC-090 is explicit: *"the automated contracts are necessary but NOT sufficient, because the
2026-10-24 verdict is a FEEL verdict; the answer is recorded in ## Latest validation evidence and a
REJECTED verdict RE-OPENS the step instead of closing it."*

### 5.1 The preconditions (the gate may not start until ALL hold)

| # | Precondition | Why |
|---|---|---|
| G1 | the fixture global `window.p1TurnDriftGreenSmokeResult.passed === true` on the **FINAL** bytes | the automated half is necessary, not sufficient |
| G2 | `npm run build:neatRacing` exits 0 | the served bundle is the artifact under test |
| G3 | **bundle identity**: the bundle the fixture loaded and the bundle the user drives are the **same bytes** (record the byte size and, ideally, a hash before and after) | the exact Step 14G failure mode was a **stale bundle**; a user gate on a different bundle than the fixture tested is worthless |
| G4 | the allow-listed lanes are green on the final bytes: `jest:base` physics\|car\|host, `jest:base` engine/passes/car, `jest:base` engine/lighting, `npx tsc -p tsconfig.neatRacing.json --noEmit` (the allow-list authority is the pointer note below) | same reason |
| G5 | the dev server is serving the **rebuilt** bundle (the plan records the dev server at `http://127.0.0.1:8787`) | a cached bundle is the same defect as G3 |
| G6 | a **REAL visible foreground** browser with a **hardware GPU** | the Step 14G precedent: *"a REAL visible foreground Chrome 155.0.8059.85 with a hardware GPU (ANGLE NVIDIA GeForce RTX 4070 SUPER, hardwareAccelerated true), 0 console errors and 0 network failures"*. A headless or software-rendered run is NOT a valid gate. |
| G7 | 0 console errors and 0 network failures during the gate run | same precedent |

**The allow-list pointer (where the lanes above come from).** The authoritative allow-list is the one
the gate publishes: `neataptic-validation-mcp get_active_validation_allowlist`. Its
`allowlistAuthority` is **`active-step.validation`** — the **active STEP's** `validation:` list (the
Step 14N packet), **NEVER** a slice packet. The runnable lanes it publishes today are the **THREE**
`jest:base` lanes — `examples/neatRacing/physics|examples/neatRacing/car|examples/neatRacing/host`,
`examples/neatRacing/engine/passes/car` and `examples/neatRacing/engine/lighting` — plus
`npx tsc -p tsconfig.neatRacing.json --noEmit`, with the **FOUR** manual items (the build + the
visible-window browser smoke, the fixture smoke, the G1..G7 precondition check and the mandatory
staleness audit) behind them. G4 above quotes that published list; the earlier **two-lane** rendering
(the `physics|car|host` lane and the `car.pass` lane only) was the 14N-era rendering and is
**superseded** by the `engine/lighting` lane the current allow-list carries.

**THE `activeSlice` POINTER THAT TOOLING PUBLISHES IS STALE (a RECORDED observation, never a silent
re-base).** The same call still reports `activeSlice.slice_id: 'P1-drift-feel-green'` (status
`[PLANNED]`, goal `green-testing`) — the **RETIRED** 14N green packet, superseded by the W3 / 14P /
14Q / 14Q2 / 14R re-bases. The resolver takes the first WIP slice, else the first PLANNED slice, so on
a multi-wave step it names a superseded slice; the allow-listed **COMMANDS** are unaffected, because
they come from the STEP and not from the slice packet. The green slice that owns the **CURRENT** gate
is `P1-drift-presentation-green` (Step 14R) — the slice that re-armed the six questions in §5.3.1 and
that §5.6's re-open rule below now names. The residual itself (the resolver's slice choice, and the
plan's own close-out rows, which still carry the same stale observation) is **RECORDED here** for the
next planning / maintenance pass; this artifact does **not** re-base the plan.

### 5.2 The human procedure (fixed, repeatable, scripted)

The user must be given an **exact input script** — not a free-form "try it out" — so the verdict
is reproducible and comparable against the 2026-10-24 baseline.

```
1. Open the served game page (manual mode).
2. Hold W until the speed reads ~90 km/h (the drift gate is 54 km/h; 90 is the modelled operating point).
3. Hold A for ~1.5 s          -> LEFT drift entry + hold   (left turn = NEGATIVE slip, D1)
4. Release A, hold D ~0.4 s   -> the CATCH: the far-side pop
5. Release D, keep W for ~2 s -> the SETTLE
6. Repeat mirrored: hold D ~1.5 s, then A ~0.4 s, then release
7. Straighten and keep W ~2 s -> the residual check
```

The user should perform the sequence **at least twice** (once per direction) before answering.

### 5.3 The question set (six fixed questions)

Each question maps to a measurable predicate so the verdict is not a matter of taste alone. The
answers are recorded verbatim.

| # | Question | Allowed answers | Maps to |
|---|---|---|---|
| **Q1** | *"When you countersteer, does the car snap back with a visible flick — the body swinging past the line and settling?"* | `YES` / `NO` / `TOO MUCH` | §3.1 (pop) |
| **Q2** | *"Is the slide CRISP or MUSHY? (crisp = the car points decisively into the turn and the path is clearly different from where the car points; mushy = the car looks like it is just crabbing sideways)"* | `CRISP` / `MUSHY` | the primary criterion — the 2026-10-24 verdict |
| **Q3** | *"Does the countersteer CATCH the slide, or does it MIRROR it (the car snaps to the other side and keeps going)?"* | `CATCHES` / `MIRRORS` | §4.2 (the root cause) |
| **Q4** | *"Does the car ever twist so far that it looks broken, or does the body look detached from the path?"* | `NO` / `YES` | §3.4 (envelope) + the canopy risk (R8) |
| **Q5** | *"While the car is sliding, can you SEE the tires differentiated (the tire surface visibly shaded across its width)?"* | `YES` / `NO` | §3.7 (AC-088) |
| **Q6** | *"Is the car still normal to drive on a straight line, and does it still turn normally at low speed?"* | `YES` / `NO` | AC-089 (no regression) |

### 5.3.1 The Step 14R re-arm (the six questions are RE-ASKED; the answer slots are PENDING)

**NO AGENT RECORDS AN AC-090 VERDICT.** The 2026-10-27 verdict stays **REJECT**. The NEXT
AC-090 verdict belongs to the **USER** (or to a fresh verification instance that is explicitly
authorised to record the user verdict). A green-testing slice that records an AC-090 pass is a plan
violation — and the Step 14R green slice `P1-drift-presentation-green` records **no verdict**.

**The six questions of §5.3 are RE-ASKED, verbatim, against the re-wired Step 14R build** — the
build whose tires, HUD and debug seam now all read the **ONE** published traction scalar, and
whose car-to-car collision resolution is called from the **live** tick. The re-arm changes the
**build under the gate**; it does **not** change the **questions**, the **input script** (§5.2), the
**preconditions** (§5.1) or the **verdict rule** (§5.4).

| # | Question (verbatim, §5.3) | Allowed answers | Status on the Step 14R re-arm |
|---|---|---|---|
| **Q1** | the countersteer flick | `YES` / `NO` / `TOO MUCH` | **PENDING** — the USER's MANUAL road run |
| **Q2** | the slide crisp or mushy | `CRISP` / `MUSHY` | **PENDING** — the USER's MANUAL road run |
| **Q3** | the catch or the mirror | `CATCHES` / `MIRRORS` | **PENDING** — the USER's MANUAL road run |
| **Q4** | the twisted body | `NO` / `YES` | **PENDING** — the USER's MANUAL road run |
| **Q5** | the differentiated tires | `YES` / `NO` | **PENDING** — the USER's MANUAL road run |
| **Q6** | the straight line and the low-speed turn | `YES` / `NO` | **PENDING** — the USER's MANUAL road run |

**The precondition state on the re-arm (which of §5.1's G1..G7 hold on the FINAL bytes).**

| # | Precondition | State on the Step 14R final bytes |
|---|---|---|
| G1 | the fixture global `passed === true` on the FINAL bytes | **NOT ESTABLISHED** — the visible-window run is **DEFERRED** under the NO-browser mandate. The fixture is **authored and syntax-checked**; the run is not a passing claim. |
| G2 | `npm run build:neatRacing` exits 0 | **MET** — exit 0 on the FINAL bytes. |
| G3 | the bundle identity (byte size + hash before and after) | **MET** — both copies recorded BEFORE and AFTER in §4.5; the two copies are now **byte-identical** at **402,024 B / `0C968DB7…CA68`**. |
| G4 | the allow-listed lanes are green on the final bytes | **MET** — the `physics\|car\|host` lane and the `browser-entry` lane pass, `tsc` is clean (§4.8). The `browser-entry` lane was run as an **EXTRA** lane under the Step 14R close-out; the published allow-list (see the §5.1 pointer note) names `physics\|car\|host`, `engine/passes/car` and `engine/lighting` plus `tsc`. |
| G5 | the dev server serves the **rebuilt** bundle | **DEFERRED** with the browser lanes (the served-page run was not performed). |
| G6 | a **REAL visible foreground** browser with a hardware GPU | **DEFERRED** with the browser lanes. |
| G7 | 0 console errors and 0 network failures | **DEFERRED** with the browser lanes. |

**The re-arm is NOT a verdict, NOT a pass and NOT a waiver of G1/G5/G6/G7.** It is the statement
that the six questions are **open against the re-wired build** and that the answer slots stay
**empty** until the user drives the road in MANUAL mode.

### 5.4 The verdict rule (deterministic, not negotiable)

```
ACCEPT  <=>  Q2 == CRISP
           AND Q3 == CATCHES
           AND Q4 == NO
           AND Q5 == YES
           AND Q6 == YES
           AND Q1 in {YES, TOO MUCH}
           AND (Q1 == TOO MUCH  =>  the envelope check §3.4 is green)
```

- **Q2 / Q3 / Q6 failing ⇒ REJECT** (these are the load-bearing criteria; a mushy or mirroring
  drift is the exact defect the step exists to fix).
- **Q1 == NO** ⇒ REJECT *unless* the automated pop contract is green **and** the user is asked
  again with the pop demonstrated on screen (the "show, do not tell" rule: a user who has not seen
  the pop cannot judge it).
- **Q5 == NO** ⇒ REJECT (AC-088 is a named acceptance criterion, not a nicety).
- **Any REJECT ⇒ the step RE-OPENS.** It does not close, it does not advance to Step 14I, and it
  does not get a "close with a note".

### 5.5 The recording format (THREE surfaces, per AC-090's file list)

AC-090's file list is `plans/neatracing-voxel-racer.plans.md + plans/README.md +
plans/Roadmap.md`, so the verdict is recorded in **all three** (not only in the plan):

1. **`plans/neatracing-voxel-racer.plans.md`** — the `## Latest validation evidence` section, as a
   new dated bullet (the plan's convention: *"gate runs are recorded in `## Latest validation
   evidence`"*, plans.md:52).
2. **`plans/README.md`** — the active-workstream row for Step 14N.
3. **`plans/Roadmap.md`** — the corresponding status line.

The YAML block that goes into (1):

```yaml
user-visual-gate:
  step: '14N'
  verdict: ACCEPT | REJECT          # required
  answered_by: <the user's words, quoted verbatim>
  build:
    bundle: docs/assets/neatracing.bundle.js
    bytes: <byte size>
    sha256: <hash>
    built_at: <timestamp>
  environment:
    browser: <name + version>
    gpu: <renderer string>
    hardware_accelerated: true
    console_errors: 0
    network_failures: 0
    window_state: visible foreground
  scripted_run:
    left:  { q1: <...>, q2: <...>, q3: <...>, q4: <...>, q5: <...>, q6: <...> }
    right: { q1: <...>, q2: <...>, q3: <...>, q4: <...>, q5: <...>, q6: <...> }
  automated_precondition:
    fixture_passed: true
    checks_true: <n> / <n>
  on_reject:
    reopens_step: true
    redaction: <what the user said, verbatim, for the next RED slice>
```

### 5.5.1 The PENDING-answer recording format (the re-armed state, Step 14R)

While the six answers are **PENDING**, the three surfaces of §5.5 carry a **PENDING** record
instead of a verdict. The shape is fixed so the later answer can be dropped in without a rewrite:

```yaml
user-visual-gate:
  step: '14R'
  verdict: PENDING               # PENDING | ACCEPT | REJECT - only the USER (or an
                                 # explicitly authorised verification instance) may set
                                 # ACCEPT or REJECT; a green-testing slice NEVER may
  rearmed_at: '2026-10-28'
  rearm_reason: 'the Step 14R presentation wave (the traction-consistent tires, the HUD
                 readout and the live collision) re-wired the build under the gate'
  answered_by: PENDING           # the user's words, quoted verbatim
  build:
    bundle: docs/assets/neatracing.bundle.js
    bytes: 402024
    sha256: 0C968DB7337A67E3B827C124F202375C8C7321656D04CF8C703D9803A2BBCA68
    built_at: PENDING
  environment:
    browser: PENDING
    gpu: PENDING
    hardware_accelerated: PENDING
    console_errors: PENDING
    network_failures: PENDING
    window_state: PENDING
  scripted_run:
    left:  { q1: PENDING, q2: PENDING, q3: PENDING, q4: PENDING, q5: PENDING, q6: PENDING }
    right: { q1: PENDING, q2: PENDING, q3: PENDING, q4: PENDING, q5: PENDING, q6: PENDING }
  automated_precondition:
    fixture_passed: PENDING       # the visible-window run is DEFERRED under the
                                  # NO-browser / NO-CDP mandate - it is NOT a pass
    checks_true: PENDING
  on_reject:
    reopens_step: true
    redaction: PENDING
```

**The rule that keeps a PENDING from becoming a silent pass.** A `PENDING` verdict is **not** an
`ACCEPT` and **not** a `REJECT`: the step stays open, the 2026-10-27 **REJECT** stays the last
recorded verdict, and the six answer slots stay **empty** until the user drives the road. The
`checks_true` slot is `PENDING` and never `46/46` while the visible-window run has not happened —
the fixture's 46 named checks are **authored and syntax-checked**, not **measured**.

### 5.6 The re-open rule (what a REJECT triggers)

1. **The step stays `[WIP]`.** It is not closed and not compressed.
2. The **verbatim user words** are recorded (they are the spec input for the next RED slice).
3. A **fix packet** is appended to the plan under the deterministic ID
   `fix-packet-P1-drift-presentation-green-iteration-<n>` (per the execute skill §11, whose ID is
   `fix-packet-<slice_id>-iteration-<n>` with `slice_id` identifying the **failing** slice), carrying
   the rejected criterion (Q1..Q6) and the observed symptom. The failing slice is the green slice that
   owns the **CURRENT** gate — `P1-drift-presentation-green` (Step 14R) — and **NOT** the retired 14N
   green packet `P1-drift-feel-green`, which the W3 / 14P / 14Q / 14Q2 / 14R re-bases superseded.
   **THE PLAN'S AC-090 ROW STILL NAMES THE RETIRED ID `fix-packet-P1-drift-feel-green-iteration-<n>`**
   — a stale pointer, **RECORDED here and never a silent re-base**: the plan owns that row and this
   pass is DOC-ONLY on this artifact, so the next planning pass repoints it.
4. The next RED slice decides whether the failure is a **model** problem (the physical half), a
   **visual** problem (the accumulator), or a **presentation** problem (the tire gradient, the
   canopy) — and the decision is recorded in the plan BEFORE any production edit, exactly as the
   original 14N packet requires for the feel DIRECTION.
5. A REJECT is **not** grounds for relaxing the automated contracts. The contracts stay; the
   implementation moves.

---

## 6. What this proposal asks of the other Phase-B proposals and of 01-planning

| Ask | To | Why |
|---|---|---|
| **correct the AC-087 / AC-088 / AC-089 / AC-090 ROW TEXT** per §8.1-§8.4 before authoring the 14N slices | 01-planning (author) | **two of the four are direct contradictions** with the Phase-A MERGE (AC-087, AC-089) and one is a known file-owner error (AC-088); a slice authored on the current text would be incoherent |
| add `examples/neatRacing/browser-entry/browser-entry.ts` to the Step 14N **file set**, with the one-line debug surface of §1 | 01-planning (author) | the browser half cannot measure `|pose|`, the pop, the settle or the model identity without it |
| expose `poseRad` + `slipAngleRad` + `speedMps` in the debug surface | 01-planning / the implement slice | §3.1-§3.5, §4.2 |
| name a **deterministic unit proxy** in `drift.model.ts` for AC-087 and forbid the fixture from comparing the browser proxy against it | the RED-contracts proposal | §8.1 (plans.md:8294) |
| pin the **Tier-1** numbers (pop floor, settle ceiling, per-tick step ceiling, envelope value, AC-087 floor, the re-baselined AC-079 ceiling, the AC-088 factor-spread minimum) | the RED-contracts proposal | the browser half only carries loose derivations of them (§3.0) |
| name the **fixture** loose bounds as RED-slice decisions with a stated headroom factor | the RED-contracts proposal | §3.1-§3.7 |
| re-baseline the **3-frame drift gate** and the **proxy constants** against the new model | the re-baselining proposal | §4.3, §4.4 |
| record the **canopy decision** (glass.pass.ts has no drift field) as a user-visible risk in the gate script | the integration proposal | research R8; Q4 asks about it directly |
| record the **Option B fallback** as a deliberate, weaker alternative | the integration proposal | §3.8 |
| state the **new** bundle byte size / hash in the 14N gate record (the 14G value 397,364 B is a reference, not a target) | 05-green-testing | §9 |

---

## 7. Risks and open items in THIS proposal

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| F1 | The **debug seam** is rejected (it touches a public page surface) | MEDIUM | Option B (§3.8) is specified and is a coherent (weaker) alternative; the plan must record the downgrade |
| F2 | The browser's loose bounds are set too tight and flake on a slow machine | MEDIUM | every bound is derived with a stated headroom; a bound is never an equality; the RED slice may relax a bound ONCE, in the plan, with a recorded reason |
| F3 | The fixture is run against a **stale bundle** and measures the old model | HIGH (precedent) | §4.2's model-identity check + §5.1 G3 (bundle identity) + §4.3's mandatory staleness audit |
| F4 | The 200-frame added scenario makes the fixture slow / flaky on a weak GPU | MEDIUM | `observeUntil` is used for the speed gate and the settle window; the phase count is bounded; the G6 hardware-GPU requirement protects the measurement quality |
| F5 | The user's verdict is a **taste** judgement that cannot be reproduced | MEDIUM | the six fixed questions + the fixed input script + verbatim recording make the verdict a repeatable, comparable record rather than a vibe |
| F6 | The user gate is skipped (the Step 14C/14G precedent: the visual gate was deferred **five** times) | HIGH | the gate is a **named acceptance criterion** (AC-090) with a deterministic verdict rule; a deferral must be recorded as a deferral in the plan, NOT as a pass |
| F7 | The re-baselined AC-079 ceiling is set too loose and the 290 km/h contract is lost by accident | MEDIUM | §4.4: the check stays; the RED slice re-pins both the expected peak and the ceiling and records the delta. **W3 update**: the re-pinned absolute ceiling is **340 km/h** (a loose bound over the measured 320.4 km/h top speed) and the **binding** contract is now the **20 km/h deficit floor**, which the W3 re-base makes **achievable on-road** (25.62 km/h measured) — so the risk is now about the *floor*, not the ceiling |
| F8 | The **canopy mismatch** (R8) makes Q4 read `YES` for a reason unrelated to the drift feel | MEDIUM | the gate script asks Q4 as a *twist* question, and the integration proposal must record the canopy decision; the plan should note that a `YES` on Q4 may be the canopy, not the pose |
| F9 | The **AC row text is authored as-is** (AC-087 on the browser proxy, AC-089 "AC-079 stays GREEN", AC-088 on `car.pass.test.ts`) and the slices are written against the contradictory text | **HIGH** | §8.1-§8.4: the corrections must land **before** the slices are authored. The RED lane cannot satisfy "AC-079 stays GREEN" and "AC-079 is re-baselined" at the same time. **This is the single highest-priority ask of this proposal.** |
| F10 | The fixture's **check count** drifts (31 today; +N new) and the gate record says "ALL 31 checks true" by copy-paste from 14G | MEDIUM | §9: the 14N record must state the new count explicitly and list the new check names; the count is asserted, not assumed. **W3 update**: the re-based fixture carries **43** named checks — the **31** retained ones (one of which, `proportionalTractionLoss`, is **re-pinned in place** and keeps its name) plus the **12** drift-feel ones (`driftFeelFarSidePop`, `driftFeelSettles`, `driftFeelNoStuckOffset`, `driftFeelPoseRateBounded`, `driftFeelPoseEnvelope`, `driftFeelZeroStraightLineYaw`, `driftFeelSlideHoldsOnNeutralSteer`, `driftFeelCountersteerReleasesSlide`, `driftFeelSlideProminence`, `tireGradientMagnitudeIsNonDegenerate`, `driftFeelModelIdentity`, `driftGateSurvivesThreeConsecutiveFrames`) — the last **three** of which are the packet's named trio, and the **two** `driftFeelSlideHoldsOnNeutralSteer` / `driftFeelCountersteerReleasesSlide` entries are the W3 **7b-hold / 7b-release** split of the old single 7b window. The count is **43** (31 + 12), and it is ASSERTED against the ACTUAL `result.checks` length in the fixture, never assumed. W4's record must state **43**, not 31. **14P update**: the count grew by **EXACTLY ONE** to **44** (43 + `driftFeelCameraFollowOffset`, the chase-camera follow offset — see §4.6); the count is still ASSERTED against the ACTUAL `result.checks` length and is never hard-coded, and `result.checkNames` is published next to it so the gate can assert the exact SET. **14Q2 update**: the count grew by **EXACTLY ONE** again to **45** (44 + `driftFeelTractionSeverityRatio`, the traction-severity RATIO — see §4.7); the count is still ASSERTED against the ACTUAL `result.checks` length and is never hard-coded. **14R update**: the count grew by **EXACTLY ONE** again to **46** (45 + `driftFeelHudTractionReadoutRatio`, the RENDERED HUD traction readout against the PUBLISHED severity — see §4.8); the count is still ASSERTED against the ACTUAL `result.checks` length and is never hard-coded, `result.checkNames` is published next to it, and the **retained set is proven against the 2026-10-27 runtime baseline** (43 published names, **0 missing**, **0 duplicate**). The **seam field count** (7 → 8) is a **SEPARATE** published condition of `result.passed` (`result.seamFieldCountMatches`) and is deliberately **NOT** a named check, so the named-check count grows by EXACTLY ONE per re-base |
| F11 | The `recordUniform` widening (L151-159) is scoped to the CAR lit program only, but the `uTireGradient*` names may also be fed by a non-car program | LOW | the capture must be guarded by `sourceOf(currentProgram).indexOf('shadeLitSurface') !== -1` exactly as the existing `uApplyTireGradient` case does (fixture L193-196), so a track/prop program cannot pollute the captured values |

---

## 8. Plan-text corrections this proposal requires (for `01-planning`)

Reading the AC rows against the Phase-A MERGE surfaced **three plan-text conflicts** and **one
already-flagged correction**. They are not cosmetic: two of them put the AC table in direct
contradiction with the MERGE, and 01-planning cannot author a coherent slice until they are fixed.

### 8.1 AC-087 — the floor is pinned on the WRONG quantity (CONFIRMED CONFLICT)

- **The AC row says** (plans.md:482): *"at a sustained full-speed full-steer turn the **MEASURED
  drift proxy** reaches at least the NAMED exported constant `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD`
  … which the RED slice pins against the MEASURED baseline of the 2026-10-24 final run (**the drift
  proxy 0.049388550574608114 rad over 81 frames**)"* — i.e. the **BROWSER** proxy.
- **The Phase-A MERGE item 7(c) says**: *"AC-087 floor re-pinned on the **DETERMINISTIC unit proxy**
  (browser-proxy spread 0.0494 / 0.0603 / 0.3599 rad is **noise**, not a floor anchor)"*.
- **The plan's own research already says it** (plans.md:8294): *"The AC-087 floor must be pinned in
  **ONE unit system**. The browser proxy and the model `slipAngle` are **DIFFERENT quantities** and
  are **not numerically comparable**."*
- **Resolution**: AC-087's text must be corrected to name the **deterministic unit proxy** (which is
  consistent with the AC's own file list — `drift.model.ts + drift.model.test.ts` — the constant
  lives in the model, not in the fixture). The browser proxy becomes a **loose observable only**
  (§3.6) and **must not** be compared numerically against
  `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD`.
- **Consequence for the model**: the new model must expose a **named deterministic unit proxy**
  in `drift.model.ts` that the RED lane can assert against. Its name and value are RED-slice
  decisions; this proposal does not invent them.

### 8.2 AC-088 — the file owner is wrong (CONFIRMED, and independently re-verified)

- **The AC row says** (plans.md:483): files `car.pass.ts + car.pass.test.ts +
  docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html`.
- **The plan's own research already flags it** (plans.md:8192): *"AC-088 needs a plan-level lever
  correction (measurement) and a **file-owner correction (`engine/lighting.test.ts`, not
  `car.pass.test.ts`)**"*.
- **Independently re-verified by the orchestrator**: the tire-gradient **shader-source** and
  **uniform-feed** contracts live in `examples/neatRacing/engine/lighting.test.ts` —
  `CAR_LIT_FRAGMENT_SHADER_SOURCE` contains `uApplyTireGradient` / `uTireGradientRadius` /
  `uTireGradientThreshold` (L1026-1029), the five uniform names are enumerated (L1034-1038), the
  double-feed comment is at L1061, and the per-feed assertions are at L1059 and L1067-1079. The
  **lever constants** (`NEATRACING_WHEEL_HUB_HEIGHT_M`, `NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR`,
  `NEATRACING_TIRE_GRADIENT_DARK_FACTOR`) live in `car.pass.ts`; the **magnitude assertion** lives
  in the fixture (§3.7).
- **Resolution**: AC-088's file list becomes
  `car.pass.ts + engine/lighting.test.ts + docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html`.
  The AC's **measurement** clause must also be corrected from *"the MEASURED gradient is 0"* to
  *"the gradient is a **fragment-stage** effect; the VBO vertex-colour probe is **blind by
  construction** (`upperAvg == lowerAvg == 0.28565`); the contract is the `uTireGradient*` uniform
  seam"*.

### 8.3 AC-089 — "stay GREEN" contradicts the LIFTED pin (CONFIRMED CONFLICT)

- **The AC row says** (plans.md:484): *"**AC-078** (tire-proportional-follow) and **AC-079**
  (proportional-traction-loss) **stay GREEN**"* — i.e. preserved verbatim.
- **The Phase-A MERGE item 5 says**: *"**MUST-NOT-MOVE PIN: LIFTED** for this slice. The model
  replacement changes the trajectory **BY DESIGN**; **AC-079 (290 km/h ceiling) and AC-089 must be
  RE-BASELINED, not preserved.**"*
- **Resolution**: AC-089's text must be corrected to *"AC-078 stays GREEN as written; **AC-079 is
  RE-BASELINED** (the ceiling and the expected peak are re-pinned by the RED slice and the delta is
  recorded); the low-speed and straight-line behaviour is undegraded; the determinism contract
  holds; the fixture global `passed` stays true and **the retained named checks stay true — and the
  NEW named checks added in this step are part of that set**."*

### 8.4 AC-090 — the verdict has THREE recording surfaces, not one

- **The AC row's file list** (plans.md:485) is `plans/neatracing-voxel-racer.plans.md +
  plans/README.md + plans/Roadmap.md`. The verdict therefore must be recorded in **all three**, not
  only in the plan's `## Latest validation evidence` (§5.5 corrected accordingly).

### 8.5 The new check set must be added to AC-089's "retained named checks"

AC-089 says *"the retained named checks stay true"*. The checks proposed in §3 must be listed
explicitly in the amended AC-089 text, or the RED lane will not know they are in scope.

---

## 9. The Step 14G precedent — the exact values to reuse for the 14N gate

The 14N user gate must reuse the 14G precedent's harness and evidence shape, not invent a new one.

| Item | The 14G recorded value (plans.md:274, :2119, :2250) |
|---|---|
| fixture global | `window.p1TurnDriftGreenSmokeResult.passed === TRUE` with **ALL 31 checks true** |
| repeatability | **3/3 captured visible-window runs** |
| browser | **REAL visible foreground Chrome 155.0.8059.85** |
| GPU | **ANGLE NVIDIA GeForce RTX 4070 SUPER**, `hardwareAccelerated: true` |
| console / network | **0 console errors, 0 network failures** |
| allow-listed lanes | `physics|car|host` **11 suites / 166 tests PASS**; the `car.pass` lane **1 suite / 4 tests PASS**; `tsc -p tsconfig.neatRacing.json --noEmit` exit 0; `build:neatRacing` exit 0 |
| bundle | `docs/assets/neatracing.bundle.js` — **388.1 kB / 397,364 bytes** |
| AC-078 measurement | `maxRelativeYawError` **2.220446049250313e-16**, `maxRelativeOffsetError` **3.477524998629633e-06** vs the 0.002 rad tolerance |
| AC-079 measurement | the drift phase peaked at **127 km/h** under the 290 km/h ceiling; `driftEngaged` true; the drift proxy **0.049388550574608114 rad over 81 frames** |

**Two consequences for 14N**:

1. **The bundle size WILL change** (new model, new constants, new accumulator). The 14N gate must
   record the **new** byte size / hash — it must NOT compare against 397,364. What the 14G precedent
   gives 14N is the **shape** of the identity check (record the served bytes; prove the fixture and
   the user drove the same bytes), not the value.
2. **The check count is 46** — the 31 retained checks plus the 12 drift-feel checks (see the **F10 row**
   in §8, which carries the 43-count and the named check list) plus the **ONE** 14P camera
   follow offset check (`driftFeelCameraFollowOffset`, see §4.6) plus the **ONE** 14Q2
   traction-severity ratio check (`driftFeelTractionSeverityRatio`, see §4.7) plus the **ONE** 14R
   HUD traction readout check (`driftFeelHudTractionReadoutRatio`, see §4.8). The 14N record must
   state the count explicitly (the 14G record stated "ALL 31 checks true"; the 14N record must say
   "ALL 46 checks true" with the 46 named). The count is **ASSERTED against the ACTUAL
   `result.checks` length** in the fixture, never assumed.
3. **The fixture's own comment already records the render-rate dependence as pre-existing**
   (fixture L141-145: *"The scenario itself (the fixed-frame drift phase) stays render-rate
   dependent — that is pre-existing and out of this contract's scope"*). The two-tier rule of §3.0
   is therefore **not an invention of this proposal** — it is the same boundary the fixture's own
   author already drew. The browser half must stay on the *shape* side of that line.

---

## 10. Certainty

| Claim | Confidence | Basis |
|---|---|---|
| The browser fixture **cannot** separate `pose` from `heading` (the §1 finding) | **0.93** | the observable set was enumerated exhaustively (WebGL matrices + HUD DOM + the single `window.neatRacingStart` global); the body, wheel and shadow matrices all carry `heading + pose`; the only other angle is `heading + slip` from the chord |
| The `uTireGradient*` uniform seam is the correct and only measurable AC-088 surface, and the VBO vertex-colour probe is blind by construction | **0.95** | boundary-mapper read of car.pass.ts:541-544, :237-241, :1464-1484, :1579, :1583 and the fragment block :324-327; **independently re-verified by the orchestrator**: `engine/lighting.test.ts:1026-1079` owns the shader-source + uniform-feed assertions, and the fixture's `recordUniform` (L151-159) is a ~6-line widening that the existing `uniform1f` wrapper already feeds for every float uniform |
| The existing fixture's structure, the **31** named checks, the drift-proxy block, and the insertion anchors | **0.95** | the orchestrator re-read the check object literal (L893-937) and counted the keys itself — **31**, matching the 14G record (the earlier scout count of 32 was off by one and is corrected here) |
| The **AC-087 / AC-089 conflicts** with the Phase-A MERGE are real and blocking | **0.94** | both AC rows were read verbatim (plans.md:482, :484) against the MERGE items 7(c) and 5, and against the plan's own prior research (plans.md:8294, :8192) |
| The far-side pop, the settle and the step bound are the correct browser-visible *shape* signals for the new model, and the "bounded + ringing" combination discriminates the new model from the old mirrored one | **0.80** | the model predictions are measured (research D3/D6.9), but the discrimination has NOT been re-measured against the OLD build in a browser — the old-build numbers are simulation-derived. **This is the single item that most needs a RED-lane measurement.** |
| The proposed browser bounds (1.0° / 6.0° / 900 ms / 128 °/s / 0.10 rad / 1.35) are reasonable | **0.72** | each is derived from a Tier-1 measured value with a stated headroom, but the headroom factors are proposals, not measurements |
| The user-gate procedure, the six questions and the verdict rule are a sound protocol | **0.85** | they are derived from the plan's AC-090 text, the Step 14G precedent (plans.md:274, :2119, :2250), and the execute skill's fix-packet / re-open conventions |
| The debug seam is the right way to close the browser-observability gap | **0.80** | it is the narrowest seam that yields exact measurements; the alternative (a wider production surface) is worse, and Option B is a coherent but weaker fallback |

**Recommended next step**: this proposal is one of four Phase-B inputs. The orchestrator merges all
four; `01-planning` (author) then patches the Step 14N slices — **including the file-set amendment
in §6** — and a FRESH `01-planning` (verification) must confirm `green-light: true` before any
`03-red-testing` dispatch. The 14G-era green light does **not** cover 14N.
