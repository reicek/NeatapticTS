# Research artifact — Step 14N arcade drift references (drift transient tuning)

- **Plan**: `plans/neatracing-voxel-racer.plans.md` (Step 14N, `[WIP]`)
- **Step**: 14N — Arcade drift FEEL sharpening
- **Scout charter**: *"Mario Kart / OutRun 2 / Ridge Racer / Daytona / Virtua Racing / TrackMania drift transients — entry attack, hold, countersteer snap-back, overshoot, numeric rates."*
- **Author role**: 02-researching (Tier-1 drift-feel scout), read-only research
- **Date**: 2026-10-24
- **Primary sibling artifact**: `plans/neatracing-voxel-racer.research.md` (integration-coupling design — unchanged by this file)
- **Sub-orchestrators used**: `research` (harness built-in, web/ROM-decomp source sweep), `implementation-pattern-scout` (repo Tier-3), `boundary-mapper` (repo Tier-3)
- **Status**: RESEARCH COMPLETE — the *feel direction* and the *floor value* remain RED-slice decisions; the **model** and the **constant set** below are settled and numerically verified.
- **Addendum (2026-10-25, game-feel scout)**: a **second pass** on this artifact adds the
  **spring-damper response layer** — *attack rates, damping ratio, overshoot, anticipation* —
  as **section E6** (evidence) and **D6** (decision) below, plus risks **R11–R15** and the
  matching certainty rows. **E1–E5, D1–D5 and R1–R10 above are UNCHANGED** (the arcade-refs
  brief stands); D6 **amends** D1's stage 7 and D2's constants only. The decisive findings
  are appended to `plans/neatracing-voxel-racer.plans.md` as
  `## PlanUpdate - Step 14N SCOUT game-feel`.

---

## Question

How do the classic arcade racers tune the **drift transient**, decomposed into
**entry attack → hold → countersteer snap-back → overshoot → numeric rates**, and what
**concrete constant pattern** should `examples/neatRacing/physics/drift.model.ts` adopt to
answer the user verdict *"tilt too slow/icy; bounce-back must be sharper; want exciting
arcade drift"*?

**Addendum question (game-feel, 2026-10-25).** Given the committed-drift model and its
second-order **pose spring**, what **spring-damper response** delivers the arcade feel —
specifically (a) the **attack rate**, (b) the **damping ratio**, (c) the **overshoot**
behaviour, and (d) the **anticipation** (the body leading the slide) — and what are the
**concrete accumulator parameters**? The sub-questions that decide it: *is the entry
overshoot a defect to be damped, or the energy reservoir that produces the far-side pop?*
*is the pop caused by the pose target's discontinuity, or by the spring?* *does the spring
set the perceived attack at all?* — answered in **E6** and **D6**.

---

## Evidence

### E1 — The current model, reproduced from source (verified)

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

### E2 — FIRST-HAND SIMULATION of the current model (reproduced equations, 60 Hz)

| Scenario | Result |
|---|---|
| **C, 25 m/s (the fixture's 90 km/h operating point), full left steer** | slip plateaus at **-4.50°** (0.0785 rad); visual proxy plateaus at **0.0392 rad**; takes **~0.25 s** to reach 63 % and **~0.75 s** to reach 95 % |
| **A, accelerating from rest, full left steer at t=1 s** | slip grows monotonically with speed: -1.51° (1.25 s) → -6.69° (2 s) → -10.2° (2.5 s) → -17.2° (3.5 s) → -23.4° (4 s) — **never plateaus** while the car is still accelerating |
| **E, 89 m/s (top speed), full left steer** | -21.97° (0.25 s) → -37.47° (0.5 s) → -56.69° (1 s) → **-65.82° (1.5 s) and still growing**; effective damping collapses **4.0/s → 0.51/s** as the slip grows |
| **D, 25 m/s, full COUNTERSTEER at t=2.5 s** | slip goes **-4.49° → +1.58° (+0.25 s) → +4.48° (settled)** — it crosses zero and **parks in a mirror-image equilibrium of the same size** |
| **E, 89 m/s, full COUNTERSTEER at t=2.5 s** | **-65.82° → -27.16° (+0.5 s) → +10.55° (+0.75 s) → +52.15° (+1.5 s)** — a full inversion, never recovered |

> **C4 — the slide is not "too small", it is UNBOUNDED.** The damping term is
> `4 · traction`, and `traction` *falls* as the slip grows ⇒ more slip ⇒ less damping ⇒
> more slip. That is **positive feedback, not a grip budget**. The "icy" read is a
> **first-0.5-seconds** problem; after that the slide runs away.

> **C5 — THE COUNTERSTEER CANNOT CATCH THE SLIDE. IT MIRRORS IT.** Two independent
> causes, both verified in source:
> 1. **The outward push is a function of the yaw rate, and the yaw rate is a function of
>    the steer.** Countersteering therefore *reverses* the "centrifugal" push, and the two
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

> **C7 — `outwardPush` is dimensionally inconsistent.** `yawRatePerTick` is *radians per
> tick*; `rad/tick · m/s` is not m/s. The correct centripetal term is
> `yawRate[rad/s] · speed[m/s] · dt`. The current form is a per-tick velocity delta
> (`-222 m/s` per tick at 89 m/s), which is why the top-speed slide explodes.

> **C8 — the longitudinal scrub can out-accelerate the engine.**
> `0.6 · |slip| · speed · speedPercent` = **24.2 m/s²** at 26°/89 m/s, versus the
> engine's **17.8 m/s²**. The car cannot hold speed in a deep slide.

### E3 — The proxy identity (independently confirmed by three agents)

`plans/neatracing-voxel-racer.research.md:66-89` (E3) and, separately, the repo
`boundary-mapper` (RISK-1) and the repo `implementation-pattern-scout` (§3) all agree:

```
rendered body yaw = heading + slipAngle · GAIN
travel direction  = heading + slipAngle
proxy = |bodyYaw - travelDir| = |slipAngle| · |GAIN - 1|
```

* The recorded browser baseline **0.049388550574608114 rad** ⇒ `|slip| ≈ 0.0988 rad = 5.66°`
  (confirmed to the last digit: `0.049388550574608114 = 0.5 × 0.098777101149216228`).
* **The proxy is V-shaped in the gain with a NULL at `GAIN = 1.0`.** Raising the gain
  toward 1.0 makes the slide *weaker*. The only monotone direction is **down**.
* The browser proxy is a **render-level, render-rate-dependent** quantity (recorded spread
  0.0494 / 0.0603 / 0.3599 rad). The deterministic twin is the model's own `slipAngle`.

### E4 — Arcade references: the five phases, per title

Labels: `[SRC]` = value recovered from source/decompilation · `[CONSENSUS]` = community
measurement, not from source · `[SIM]` = derived by simulation · `[NOT FOUND]` = no public
data; **nothing invented**.

| Game | Entry attack | Hold | Countersteer snap-back | Overshoot | Speed scrub |
|---|---|---|---|---|---|
| **Mario Kart Wii** | commanded drift angle ramps **+150 °/frame** toward target; auto ±30 °/frame; steering lag `reactivity` `[SRC]` | clamp **±60°**; charge tiers 270/300/75; `stabilizationFactor` 0.5/0.22 `[SRC]` | **linear −2.0 °/frame = −120 °/s**; velocity filter `dirDiff *= 0.1` ⇒ **τ ≈ 167 ms** `[SRC]` | prevented by sign-flip guard + turn fade | accel swaps to the drift curve `[SRC]` |
| **Mario Kart DS** | **rate-limited ramp ±200 units/tick** to a hard clamp `[SRC]` | 2 tiers, 10-frame alternation lockout `[SRC]` | **linear ±200 units/tick**, hard clamp at 0 `[SRC]` | impossible (clamped) | `speed *= 0.95/frame` above cap `[SRC]` |
| **Super Mario Kart** | two independent ramps: velocity lag **13.5°/18°**, pose offset **46°/57°**; hop 17 frames `[SRC]` | no mini-turbo (wiki "3 s" claim conflicts with the ROM) | linear per-frame decay **with a residual** (−64/−96) `[SRC]` | spin-out at ±$480/frame `[SRC]` | ~30 % of pace lost `[SRC]` |
| **Mario Kart 64** | first-order lag **τ = 12 frames**, seeded `unk_078 · 0.14` `[SRC]` | clamp **±40.08°** (0x1C70); mini-turbo tiers 10/18 frames `[SRC]` | **exponential τ = 12 frames ≈ 0.4 s @30 Hz**; stick remap eats 13/53 countersteer units `[SRC]` | impossible (+0.33° deadband) | `decel += |angle_deg| · 0.004/frame` — **2.5× cheaper than a hard turn** `[SRC]` |
| **OutRun 2** | **NOT FOUND** — no assist; 4-step manual technique (lift→brake→steer→throttle) | NOT FOUND; steering-motor power 100/90/80/60 % | NOT FOUND | catch-the-slide is designed in, unquantified | NOT FOUND |
| **Ridge Racer (PS1)** | wheel ±0x500 of ±0x1000 per frame (**0→full lock 3.2 frames ≈ 107 ms**); **velocity direction chases heading at 1/5 per frame ⇒ τ = 5 frames = 166.7 ms** `[SRC]` | grip = **hard projection** (lateral velocity discarded each frame); drift gate `speed > 320` raw; per-car `lim` 9 100–21 000 `[SRC]` | **exponential, τ = 5 frames = 166.7 ms**; wheel to neutral in 2 frames (τ ≈ 48 ms) `[SRC]` | **no anti-overshoot clamp**; spin cap 0x249 BAM12 ≈ 51.4° | `speed · cos ψ` per frame; coast drag ×996/1000 `[SRC]` |
| **Ridge Racer Type 4** | ~0.5 s accelerator release `[CONSENSUS]` | 3 drift stages, no seconds | NOT FOUND | rotation clamp provably exists, angle NOT FOUND | NOT FOUND |
| **Ridge Racer 7** | accel-drift / brake-drift; minimum brake-pulse length exists, value NOT FOUND | scrub saturates **≈30°** in the driver's view | assist-mediated, setup variable; **fails open** (mis-countersteer ⇒ the car drifts the *other* way) — a **relay / re-latch** `[CONSENSUS]` | 360° drifts possible ⇒ no hard rotation clamp | per-car tiers only |
| **Daytona USA** *(Sega Rally '95 sibling proxy — Daytona USA's own ROM physics is not public)* | yaw gated on `|steer| > 0.0349` and `speed > 0.3704`; slide blends toward the road tangent `[SRC]` | road-relative; self-limiting | **`error *= cos Δθ` per frame** ⇒ quadratic convergence: 30° → ×0.866 (τ 125 ms), 60° → ×0.500 (τ 33 ms), 90° → instant, 15° → ×0.966 (τ 489 ms) `[SRC]` | **none** (monotone) | speed converges to the corner target (τ 2.2–3.6 s) `[SRC]` |
| **Virtua Racing (32X)** — *the grip-based CONTRAST* | steering accel 24 units/frame, EMA α=0.5 (τ = 24 ms); **drift threshold `|slip| > 0.215 rad = 12.3°`** `[SRC]` | saturated grip budget (max 1.504× slip); spin-out at `|lateral_vel| ≥ $C0F2` | **the reference arcade countersteer: steering accel HALVED and `drift -= (drift >> 3)` ⇒ ×7/8 per frame ⇒ τ = 7.49 frames = 124.8 ms @60 Hz**; 90 % gone in 284 ms; **hard snap to zero on sign reversal; hard zero when `|v| ≤ 15`** `[SRC]` | structurally excluded | `final_force = (net_force >> 1) · grip >> 7` `[SRC]` |
| **TrackMania** | tire-saturation event (DIRECT→CAPTURED on `slip_seen`); steering slew 20.0 ⇒ 0→full lock in **50 ms @100 Hz** `[SRC]` | **captured-angle rail**, clamp **π/4 = 45°**; grip clamped to `min(requested, μN)` `[SRC]` | **hard state reset to zero** on captured-angle sign cross (`capture_exit_side_speed_max = 1.0`) `[SRC]` | none | proportional to over-limit demand, saturating `[SRC]` |

**Cross-game synthesis.** All eleven titles use one of three architectures:
(a) a **velocity-direction lag filter**, (b) a **saturated grip budget**, or
(c) a **clamped design angle + a separate pose channel**. In *every* case the slide is a
**commanded, clamped quantity that the body chases** — never an emergent force-balance
equilibrium. The shared grammar is:

> **entry = a rate-limited ramp to a hard clamp; exit = a linear unwind; the countersteer
> has its own authority applied *against* the slide; the sign is latched at entry.**

Only two real overshoot sources exist in the whole corpus: RR7's **failing-open countersteer
relay**, and the SMK/MKWii **pose-vs-velocity split** (a dedicated second channel that the
body swings on).

**Anchor band** (for picking a clamp): Ridge Racer PS1 steady slip **14.1°**, Virtua Racing
drift threshold **12.3°**, Mario Kart 64 clamp **40.1°**, Mario Kart Wii clamp **60°**,
TrackMania clamp **45°**, Ridge Racer 7 saturation **≈30°**.

### E5 — Local repo constraints (repo Tier-3 scouts)

* **Naming grammar** (`implementation-pattern-scout`): one `export const` block at the top
  of the file; `NEATRACING_DRIFT_<SUBJECT>_<UNIT>`; unit suffixes `_RAD`, `_MPS`, `_PER_S`,
  `_PER_RAD`; floor/ceiling infixes `MIN_` / `MAX_`; JSDoc = one-line summary, blank line,
  then a 2–6-line "why" naming the AC id and the measured rationale; **no** `@param`/`@returns`
  on constants.
* **Only one production importer**: `physics.step.service.ts` (calls `computeNextLateralVelocity`
  at `:388`, `computeLongitudinalTractionLoss` at `:406`, `computeSlipAngle` at `:412`, and
  inlines `nextSlipAngle · GAIN` at `:417`). **The car pass (`engine/passes/car.pass.ts`)
  imports nothing from the drift model** — it consumes one opaque radian scalar
  (`car.driftYawOffsetRad` at `:438`, `:1365`, `:1567`). ⇒ **no drift lever requires
  re-authoring the car pass.**
* **Drift levers that live *outside* the declared AC-087 file set**:
  `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER` and `NEATRACING_PHYSICS_STEERING_RATE` are in
  `physics.step.service.ts` — a slice file-set amendment is required to touch them.
* **The visual-yaw formula exists twice**: `drift.model.ts:363 computeDriftYawOffset()` is
  **mathematically identical** to `physics.step.service.ts:417` but has **zero production
  consumers** (dead export / divergent twin).
* **AC-088 is unmeasurable through the named levers** (both repo scouts agree): the tire
  gradient is a **fragment-stage** effect, while the fixture's `analyzeWheelShading()` reads
  the **wheel VBO vertex colours**, which the mesh builder writes *uniform* ("Vertex colors
  stay uniform rubber", `car.pass.ts:559-561`) ⇒ `upperAvg == lowerAvg == 0.28565` **by
  construction**. No value of `NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR`/`_DARK_FACTOR`/
  `NEATRACING_WHEEL_HUB_HEIGHT_M` can make the measured gradient non-zero. This is a
  **plan-wording correction**, not a tuning task.
* **Spec §5.2 is stale** against the code (`MIN_GRIP_FACTOR = 0.25` deleted, countersteer 4.0
  vs 12, lateral grip 6.0 vs 4, and the **countersteer sign rule is stated backwards**).

### E6 — Game-feel response design (the spring-damper layer) — SECOND-HAND SIMULATION

*Added by the game-feel scout (2026-10-25). E1–E5 and D1–D5 above are UNCHANGED.*

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

## Decision

### D1 — THE MODEL (stated explicitly)

**Committed-drift-angle arcade model.** The drift becomes a *commanded, clamped state*; the
body chases it; the countersteer gets its **own** authority applied **against** the slide.

**Sign convention (pinned to production, verified):**
`lateralVelocity > 0` = the velocity vector lies to the **LEFT** of the heading;
`slipAngle = atan2(lateralVelocity, speed)`, so a **LEFT** turn (steer > 0) produces a
**NEGATIVE** slip (the nose leads, the velocity lags right). The visible yaw offset is
`slipAngle · GAIN`, hence **negative** in a left turn. *(The upstream draft of this model
wrote the visible-pose target as `-committed`; with this convention that is a sign error and
would mirror the whole drift. Corrected here to `+committedSlip`.)*

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

1. *It is the only shape that can produce the requested behaviour.* A first-order
   lateral-velocity model is **monotone** — it cannot overshoot (verified: variant (b) below
   produced **0.00°** of swing-past). The overshoot must live in a **second-order** channel.
2. *It fixes C5 at the root.* The countersteer no longer reaches the slide through
   `sign(slip)`; it has its own authority, and the drift sign is **latched** so the
   "reversal" of C5 becomes a controlled **unwind**, not a mirror.
3. *It fixes C4 at the root.* The slide is bounded by `MAX_SLIP_ANGLE_RAD`, not by a force
   balance.
4. *The slip-gated steer-yaw term in (7) is what makes the swing visible* **and** keeps the
   **zero-straight-line-yaw** guarantee (the term vanishes when `slip = 0`).

### D2 — THE CONSTANT SET (for `examples/neatRacing/physics/drift.model.ts`)

Repo grammar `NEATRACING_DRIFT_<SUBJECT>_<UNIT>`, one `export const` block at the top of
the file. Values marked **[SIM]** are **verified by my own 60 Hz simulation** of the
equations above; **[REF]** values are the arcade anchor the number was chosen against.

```ts
// ---- drift envelope ----------------------------------------------------------
export const NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD = 0.45;   // rad (25.8°)  [SIM][REF: TrackMania 45°, MK64 40.1°]
export const NEATRACING_DRIFT_MIN_SPEED_MPS        = 15;    // m/s          (UNCHANGED)

// ---- entry attack / hold -----------------------------------------------------
export const NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S = 3.5;   // rad/s (201°/s) [SIM][REF: MKWii 150°/frame]
export const NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S  = 2.0;   // rad/s (115°/s) [SIM][REF: MKWii -2.0°/frame]
export const NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S  = 0.045; // s  (τ of the chase) [SIM][REF: VR 24 ms, RR 167 ms]

// ---- tire saturation (a budget, not a damping multiplier) ---------------------
export const NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD = 0.20; // rad (11.5°) [REF: VR threshold 12.3°]
export const NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD  = 2;    // 1/rad (UNCHANGED)

// ---- countersteer: its own authority, applied against the slide ---------------
export const NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS   = 70;  // m/s²  [SIM][REF: VR τ=125 ms]
export const NEATRACING_DRIFT_COUNTERSTEER_MIN_SLIP_RAD    = 0.02;// rad (1.15°) [REF: MK64 0.33° deadband]
export const NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD      = 0.02;// rad (1.15°) [REF: VR "|v|<=15 => 0"]

// ---- visible pose (second order: the spring-damper) --------------------------
export const NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2   = 324;  // rad/s² (ω = 18 rad/s, f = 2.86 Hz)
export const NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S      = 12.6; // rad/s  (ζ = 0.35 ⇒ 31 % first overshoot)
export const NEATRACING_DRIFT_POSE_STEER_YAW_RAD          = 0.08; // rad (4.6°)
export const NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD = 0.10;// rad (5.7°) — slip at which the term is full

// ---- speed scrub --------------------------------------------------------------
export const NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S     = 0.25; // 1/s (down from 0.6) [REF: MK64 2.5x cheaper]

// ---- retained -----------------------------------------------------------------
export const NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD = 0.5;  // rad (UNCHANGED — the base gain)
```

**Constants that this model RETIRES** (their jobs are taken over):
`NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S` (4) → becomes `SLIP_RESPONSE_TAU_S` + the
saturation budget; `NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S` (12) → becomes
`COUNTERSTEER_AUTHORITY_MPS`; `NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S`
(0.6) → becomes `SPEED_SCRUB_COEFF_PER_S` (0.25, and now *saturating* rather than
proportional-to-speed).
`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` changes from `π/2` to `0.45` (it is currently
**unused inside its own module** — `atan2` bounds it by construction — so it is an
envelope pin, not a lever).

**Optional refinement (NOT verified — decide in the RED slice):** a steering-authority fade
so a big slide stays catchable — `NEATRACING_DRIFT_STEER_AUTHORITY_FADE_START_RAD = 0.52`
(30°) and `..._FADE_SPAN_RAD = 0.52` (30°), i.e. zero steering authority at 60° of slip.
Modelled on MKWii's `1 - (angle - 60)/40`.

### D3 — MEASURED PREDICTIONS of the new model [SIM, my own run, 60 Hz]

Full left steer at t = 1.0 s, full countersteer at t = 3.0 s:

| Metric | Current model | **New model** |
|---|---|---|
| Entry: 63 % of the hold slide | ~0.25 s @25 m/s, ~0.5 s @89 m/s | **0.10 s** @both |
| Entry: 95 % of the hold slide | ~0.75 s @25 m/s | **0.20 s** |
| Hold slide (89 m/s) | -65.8° and **growing** | **-25.94° and BOUNDED** |
| Hold slide (25 m/s) | -4.50° (too small to see) | **-25.83°** |
| Visual proxy at hold | 0.0392 rad @25 m/s | **0.2266 rad = 4.6× the 0.0494 baseline** |
| Countersteer: slip after +0.1 s | -65.8° → -54.4° (still negative) | **-25.94° → -18.96°** |
| Countersteer: slip after +0.2 s | +10.6° (**mirrored**) | **-7.01°** |
| Countersteer: slip after +0.3 s | +28.1° (mirrored) | **0.00° (hard zero)** |
| Visible swing-past | **none** (0.00°) | **-14.02° at +0.1 s = 4.83° past the hold pose** |
| Visible ring after the snap | none | +2.61° → -0.51° → -0.53° → +0.23° → 0 (ζ = 0.35) |
| Residual pose on a straight line | 0 (exact) | **-0.07° ≈ 0** (slip-gated steer-yaw) |
| Speed scrub @89 m/s | 24.2 m/s² (**> engine 17.8**) | **10.0 m/s²** (engine still wins) |

**Variant check — where the swing-past comes from.** With the steer-yaw term removed from
(7) (`poseTarget = slip · GAIN` only) the model produces **0.00°** of swing-past: the
physical recovery is monotone and the visible channel simply follows it. The steer-yaw term
is the swing source; the **slip-gate is what keeps the zero-straight-line guarantee**
(ungated it leaves a -4.58° residual body yaw while driving straight).

### D4 — Sequencing recommendation for the orchestrator (criterion: fun / exciting)

**Adopt the physical model — but recognize that the already-recorded coupling design's
"Track A" is a strict subset of it.**

* The coupling design's **transient gain** (`offset = slip · (GAIN + transientGain)`) and this
  model's **pose spring-damper** are *the same mechanism* (a second-order visual channel).
  This model simply **sources it from the physical slip** instead of a synthetic gain, so
  the visual swing and the physical catch become one coherent thing.
* **The visual-only route (Track A) cannot satisfy the user's verdict.** The verdict's
  operative clause is *"bounce-back must be sharper"*, and the bounce-back is **physical**
  (C5). A cosmetic swing on a car that is still sliding the wrong way is not a fix.
* **Track A's gain-down move is still the cheapest first step** (proxy
  `0.0988 × 1.5 = 0.148 rad` = 3.0× baseline, byte-identical trajectory) and is a
  reasonable *interim* if the budget cannot absorb the re-baselining. But it must be
  recorded as **cosmetic-only**, and it must be driven **down** (0.5 → 0.25 → -0.5) — never
  up, because the proxy is V-shaped with a null at `GAIN = 1.0` (E3).

**⇒ Recommended mix: (1) the physical model as a NEW slice with re-baselining; (2) the pose
spring-damper as the visual half of it (this *is* the transient the plan already names);
(3) the AC-087 floor re-pinned on the *deterministic* unit proxy, not the noisy browser
proxy.**

### D5 — Plan items this research requires to be corrected

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

### D6 — THE GAME-FEEL ACCUMULATOR (the spring-damper layer)

*Added by the game-feel scout (2026-10-25). D1–D5 above are UNCHANGED. D6 **AMENDS D1's
stage 7 and D2's constants**; the committed-drift model, the grip budget, the countersteer
authority and the hard zero all stand exactly as written.*

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

| Constant | Value | Status |
|---|---|---|
| `NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2` | **324** (ω = 18) | **UNCHANGED** (arcade-refs) |
| `NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S` | **12.6** (ζ = 0.35) | **UNCHANGED** (arcade-refs) |
| `NEATRACING_DRIFT_POSE_STEER_YAW_RAD` (articulation amplitude) | **0.08** | **UNCHANGED** (arcade-refs) |
| `NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD` (arm threshold) | **0.10 → 0.01** | **CHANGED** |
| `NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S` | **20** (band 15–30) | **NEW** |

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

| Metric | Arcade-refs (D3) | Recommended (D6) |
|---|---|---|
| Hold slip / pose / δ | −25.39° / −10.16° / 15.23° | **identical** (the spring is unchanged) |
| Entry overshoot | 17.63 % (1.79°) | 18.35 % (1.86°) — **kept deliberately** |
| Catch dive | −3.67° | −4.40° |
| **Far-side pop** | **2.61° (17 %)** | **2.22° (15 %) — the pop is PRESERVED** |
| Settle (\|δ\| < 0.5°) | 550 ms | 550 ms |
| Peak pose rate | 108 °/s | 92 °/s |
| Onset to 25 % / 50 % of δ | 33.3 / 66.7 ms | **33.3 / 66.7 ms (unchanged)** |
| **Max pose-target step per tick** | **−4.65° (9.17° @ 89 m/s)** | **+2.13°, speed-independent** |
| Max \|pose\| vs the 0.2250 rad envelope | **13.15° = 0.2295 rad — 2.00 % OVER** | **12.87° = 0.2246 rad — 0.19 % inside** |
| Residual pose at rest | −0.000° | −0.000° |
| Zero-straight-line max \|pose\| | 0 (exact) | **0.000000000 rad (exact)** |

**D6.10 — WHAT THIS RESEARCH REQUIRES OF PHASE B.** (1) Record the asymmetric-damping idea as
**REFUTED** so it is not re-derived. (2) **AMEND** the arcade-refs D1 stage 7 + D2 constants;
the rest of the arcade-refs model stands. (3) **MERGE** the two visual channels into ONE
accumulator (D6.5). (4) The RED slice pins **three** numbers + the exact identity (D6.7).
(5) Re-derive the yaw limit (D6.8). (6) The pop is a **corpus contract** (D6.7a).

---

## Risks

| # | Risk | Severity | Evidence | Mitigation |
|---|---|---|---|---|
| **R1** | **The physical model changes the trajectory.** A 5× larger slide at the 90 km/h operating point moves the car's path (a 25.8° slip at 25 m/s is ~12 m/s of lateral velocity). AC-079's 290 km/h ceiling, AC-089's no-regression pin, and the 3-frame `DRIFT_PROXY` gate must be **re-measured**, not preserved. | **HIGH** | D3, C4 | Re-baseline explicitly in the new slice; sequence after the cosmetic step so there is a working baseline to fall back to. |
| **R2** | **The countersteer authority may be too strong at low speed** (70 m/s² × speedPercent = 19.7 m/s² at 25 m/s) and could feel twitchy below ~40 km/h. | MEDIUM | D3 | Gate the authority on `|slip| > COUNTERSTEER_MIN_SLIP_RAD` (already) and consider a `speedPercent` floor; measure in the RED lane at 25 m/s. |
| **R3** | **`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` changes `π/2 → 0.45`**, which moves `car.pass.test.ts`'s derived `DRIFT_VISUAL_YAW_LIMIT_RAD` (= `MAX_SLIP · GAIN`, 0.7854 → 0.225) and its monotonicity-sweep endpoints. The test re-derives it, so it self-syncs — but the **envelope value moves by 3.5×**. | MEDIUM | E5, D2 | Re-derive and re-baseline in the same slice; verify the sweep endpoints still bracket the new envelope. |
| **R4** | **The pose state is a new state type.** It must NOT be added to `DriftModelState` if that type is byte-identity pinned (AC-047D) — a **separate** `DriftPoseState` is required, exactly as the coupling design's D2 already decided for the transient gain. | MEDIUM | E5, `research.md` D2 | New separate type + a new numeric field on `PhysicsCarState`; check the exact AC-047D assertion form in the RED slice. |
| **R5** | **The steering-authority fade is unverified** and could make the car un-controllable if the numbers are wrong. | LOW | D2 | Ship it disabled; enable only after a measured pass. |
| **R6** | **The proxy null at `GAIN = 1.0`** is a live trap for any visual-gain work in parallel (the coupling design's Track A). If Track A and this model are both in flight, they must not both touch the gain. | MEDIUM | E3 | One owner for the visual channel. |
| **R7** | **`collision.utils.ts:410,417-418` zeroes `speed`/`slipAngle`/`lateralVelocity` on a bounce** but not the drift yaw offset ⇒ a wall impact kills the slide and the pose would ring on with no slip to follow it. | LOW | E5 | With the slip-proportional pose the offset self-heals to 0; still add a RED note. |
| **R8** | **The glass canopy does not follow the drift yaw** (`glass.pass.ts:1048,1086` never reads `driftYawOffsetRad`) ⇒ a bigger slide **amplifies** the visible body-vs-canopy mismatch. | MEDIUM | `research.md` R7 | Out of scope; record as a carried-forward residual for a later step. |
| **R9** | **The upstream arcade-research draft's model had a sign error in the visible-pose target** (`-committed` under the production slip convention). | — | D1 | Corrected here; the implementer must use **D1 as written**, not the upstream draft. |
| **R10** | **This research was produced under a paused/budget-constrained directive** (2026-10-24). The gaps in E4 (OutRun 2, Ridge Racer 4/7, Daytona USA's own ROM, the Mario Kart per-character tables) are **flagged, not filled** — nothing was invented to cover them. | LOW | E4 | The gaps do not affect the constant set: every number in D2 is anchored to a game that *does* publish the value. |
| **R11** | **A future "tidy-up" of the pose spring can silently destroy the arcade feel.** The pop and the settle are one dial (E6.5): raising ζ 0.35 → 0.45 halves the pop (2.22° → 1.35°), and raising ω 18 → 24 collapses it further (1.37°). The entry overshoot that looks like a defect is the pop's energy reservoir (E6.2). | **HIGH** | D6.1, D6.7 | Pin the pop, the settle AND the max per-tick target step in the RED slice as **named contracts with stated minimums**; record the asymmetric-damping idea as **REFUTED** so it is not re-derived. |
| **R12** | **The declared `DRIFT_VISUAL_YAW_LIMIT_RAD` (0.2250 rad) is exceeded by 2.00 %** by the arcade-refs' own entry overshoot (0.2295 rad) — the envelope is derived from `MAX_SLIP · GAIN` rather than from the measured peak. | MEDIUM | E6.6, D6.8 | Re-derive the limit from the measured peak (or keep the pose inside it via the slew), and re-verify `physics.step.service.test.ts:156-163` against the overshoot. |
| **R13** | **The 4 Hz steer-chatter stress** drives the far-side body swing to 22.6° and the settle to ~2 s. The slew bounds the per-tick target step but does **not** suppress the committed model's chatter response ⇒ this is a **D1 latch** property, not a pose-channel defect. | MEDIUM | E6.10 | Out of the game-feel charter's scope; raised as a risk on the latch for Phase B to consider. Not reproduced against production. |
| **R14** | **Two competing visual accumulators** (the coupling design's `DriftTransientGainState` and the arcade-refs' pose spring) are the SAME mechanism; shipping both would double the visual gain and fight the single bridge write. | **HIGH** | D6.5 | Merge into ONE `DriftPoseSpringState` (three scalars) on `PhysicsCarState`; one owner for the visual channel (extends R6). |
| **R15** | **The spring's perceived attack is NOT the spring's** (E6.4): the onset is identical across every variant, so a Phase-B author who assumes "crisper = raise ω" would silently trade the pop away for no perceived gain. | MEDIUM | E6.4, E6.5 | State the attack's real source (the committed-angle ramp + the slip chase) in the RED slice's comments; raise ω only for the pop, within the R11 guardrails. |

---

## Certainty statement

| Claim | Confidence | Basis |
|---|---|---|
| **C1** — the production centrifugal multiplier is 1.5, not 0.3 | **0.99** | direct source read, `physics.step.service.ts:184` |
| **E2 / C4** — the current slide is unbounded at speed | **0.97** | my own 60 Hz simulation of the reproduced equations, 89 m/s and 25 m/s |
| **E2 / C5** — the countersteer mirrors the slide instead of catching it | **0.97** | my own simulation (+15.4° / +52.2° inversions) **and** an independent source-level reading of the two causes (yaw-derived push, sign-crossing gate) |
| **E3** — the proxy identity and the null at `GAIN = 1.0` | **0.95** | confirmed to the last digit against the recorded baseline, and independently by two repo Tier-3 scouts |
| **E4** — the per-game five-phase table | **0.88** for the `[SRC]` rows, **0.55** for the `[CONSENSUS]` rows, **0.00** for the `[NOT FOUND]` cells | ROM decompilations / reimplementation sources (see the source index below) |
| **D2** — the constant set | **0.86** | every value is either verified by my own simulation or anchored to a published game number; the *feel* choice inside the band remains a RED-slice decision |
| **D3** — the measured predictions | **0.93** | my own simulation; the model is deterministic and the harness is exact, so the residual uncertainty is the production integration order (tick-start vs tick-end speed), not the model |
| **D4** — the sequencing recommendation | **0.80** | a judgement call about plan shape, not a measurement |
| **R1** — the trajectory risk | **0.85** | the lateral velocity change is arithmetic; the *contract* consequences depend on the re-baselining that has not happened yet |
| **D6.1** — the arcade-refs' `K = 324` / `C = 12.6` are correct and must not be re-tuned | **0.97** | measured; the asymmetric-damping alternative collapses the pop 2.61° → 0.00°, and the harness is validated against this artifact's own D3 numbers to the digit |
| **D6.2** — the slew-limited articulation and its per-tick bound | **0.93** | measured (max step 2.13°/tick, speed-independent) and the bound is derived, not fitted |
| **D6.3** — the constant set | **0.90** | every value is measured in the harness; the *in-band* choice (slew 15–30) remains a RED-slice decision |
| **D6.5** — the state-ownership merge (one accumulator, three scalars) | **0.80** | a judgement about plan shape, not a measurement; the mechanics are source-verified |
| **E6.4** — the attack rate is the model's, not the spring's | **0.93** | measured across every variant: the onset is identical to the millisecond |
| **E6.6** — the 2.00 % envelope overshoot | **0.90** | measured; the declared limit is derived from `MAX_SLIP · GAIN`, and the overshoot is a direct consequence of the (deliberate) entry overshoot |
| **E6.7** — tick placement and the camera's non-consumption of the offset | **0.92** | source-verified by the `boundary-mapper` Tier-3 scout (two read sites, one bridge write, no interpolation, camera yaw trajectory-only) |
| **E6.8** — no second-order precedent in the repo | **0.95** | source-verified by the `implementation-pattern-scout` Tier-3 scout; the house idiom is first-order TAU |
| **E6.9** — the exact zero-straight-line identity and the stability margins | **0.95** | measured (0.000000000 rad over 5 s); stability is arithmetic (ω·dt ≤ 0.533 « 2) |
| **E6.10** — the 4 Hz chatter stress | **0.85** | a model-level observation on my reproduction, deliberately reported as a risk, not a defect claim |

### Source index

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
