# AUDIT-ENVELOPE — numeric re-derivation of the drift engage envelope

**Artifact id**: `neatracing-voxel-racer.audit-envelope`
**Author**: `02-researching` (Tier 1), 2026-10-27
**Dispatch origin**: the `USER VERDICT 5` record, `plans/neatracing-voxel-racer.plans.md` L10267–10291, wave id `14N-V5-audit-envelope`
**Subject**: `examples/neatRacing/physics/drift.model.ts` (787 lines), `examples/neatRacing/physics/physics.step.service.ts` (647 lines), `examples/neatRacing/host/input.ts`, `examples/neatRacing/host/mode-selector.ts`, `examples/neatRacing/host/race-session.ts`, `examples/neatRacing/physics/ai.autopilot.ts`, `examples/neatRacing/browser-entry/browser-entry.ts`
**Constraints honoured**: READ-ONLY code audit · NO child agents (caller order overrides the generic Tier-1 delegation mandate) · NO browser runs · NO Chrome MCP · no production byte touched · no git mutation.
**Sibling artifacts**: `neatracing-voxel-racer.audit-physics.research.md` (constant inventory F-series, L-series, R6), `neatracing-voxel-racer.audit-visuals.research.md` (presentation surface). This artifact is the ENVELOPE half and does not duplicate either.

---

## 0. Provenance, method and the evidence class used

**Method — MEASURED, not derived.** The caller's objective was a *numeric* re-derivation, so arithmetic-by-hand was not sufficient. The real production modules were bundled headlessly and executed in Node:

```
$probe | node_modules/.bin/esbuild.cmd --bundle --platform=node --format=cjs --loader=ts \
        --outfile=$env:TEMP/drift-probe.cjs
node $env:TEMP/drift-probe.cjs
```

The probe imports the **live** `physics.step.service.ts`, `drift.model.ts`, `ai.autopilot.ts`, `collision.utils.ts` and `racing_curriculum/track/track.generator.ts`. It is a *headless Node* run — **not** a browser run, so the NO-browser / NO-Chrome-MCP order is honoured. Bundle output and probe sources were written **outside the repository** (`%TEMP%`); no repository file other than this artifact was created or modified, and no git command was run.

**Evidence ranking applied**: `runtime/validation > static code > comments/docs > external`. This wave is the only one of the four with **runtime** evidence, so its findings outrank the static-code findings of `AUDIT-PHYSICS` wherever they touch the same constant. Where the two disagree, the disagreement is recorded explicitly in **R6**.

**Reconciliation (why the numbers may be trusted).** The measured per-tick scrub was reconstructed by inverting the *real* integrator (`physics.step.service.ts:567-601` → `min(89, v + 17.8·dt)`) and compared against the model's own law `computeLongitudinalSpeedScrub` (`drift.model.ts:632-646`):

```
scrub_measured − scrub_model = 0.000000000   at every sampled tick (t = 1 … 1200)
```

The reconstruction is therefore **exact**, not a fit. All figures below are measured, not derived.

**Scope boundary.** This artifact adjudicates the *envelope* (what magnitudes of slip and traction loss are reachable, and at what speed/steer). It does **not** adjudicate the camera aim path (`AUDIT-VISUALS`), the wheel-spin idiom seam, or whether the live rAF loop really steps the drift channel (`AUDIT-MOTION`). It does adjudicate the **input** surface, because the engage threshold is only meaningful against the axis the host actually supplies.

---

## 1. Question

> Re-derive **numerically** the drift engage envelope (the `drift.model.ts` constants) for a **full-speed held steer turn**; show **why the user sees ZERO drift**; and determine whether **proportional traction loss (S-1 / S-2 / S-3) is possible under the B1–B4 fixed constants** — or prove the model cannot lose traction at full speed.

Three sub-questions, three answers, in **D1** (envelope), **D2** (zero drift) and **D3** (S-1/S-2/S-3).

---

## 2. Evidence

### 2.A The committed ramp reaches its clamp in 8 ticks, at any speed

`stepDriftCommitment` (`drift.model.ts:489-560`) ramps the committed slip by
`slip −= turnSign · 3.5 · |steer| · dt` (`:520-524` on engage, `:534-536` while drifting)
and clamps it to ±0.45 (`:66`, `:551-555`).

Measured at |steer| = 1 (dt = 1/60, ramp = 0.0583333 rad/tick):

| tick | committed slip | published slip | note |
|---|---|---|---|
| 1 | −0.058333 | −0.018074 | first ramp step |
| 5 | −0.291667 | −0.175635 | |
| **8** | **−0.45 (clamped)** | −0.308278 | clamp hit at tick 8 (0.133 s) |
| 20 | −0.45 | −0.444018 | |
| 45 | −0.45 | **−0.451153** | chase overshoot (published max 0.452, `:94`/`:128`) |

The ramp is **speed-independent** (B2). Measured at every entry speed 5 … 89 m/s, the committed slip reaches exactly −0.45 (S3 sweep). ⇒ **The slide depth is a constant 0.45 rad (25.8°) at 5 m/s and at 89 m/s alike.**

### 2.B The traction loss IS real at full speed (the "no traction" premise is FALSE)

Road, throttle = 1, |steer| = 1, entry 89 m/s, paired against a steer-0 baseline of the same throttle:

| tick | v (m/s) | v (km/h) | published slip | scrub (m/s²) | deficit (km/h) |
|---|---|---|---|---|---|
| 1 | 89.0000 | 320.40 | −0.018074 | 0.00000 | 0.000 |
| 8 | 88.7837 | 319.62 | −0.308278 | 12.98067 | 0.779 |
| 12 | 88.6866 | 319.27 | −0.399347 | 18.76633 | 1.128 |
| 20 | 88.3010 | 317.88 | −0.444018 | 21.39787 | 2.516 |
| 30 | 87.6788 | 315.64 | −0.450654 | **21.50395** | 4.756 |
| 45 | 86.8045 | 312.50 | −0.451153 | 21.10655 | 7.904 |
| 60 | 86.0298 | 309.71 | −0.451044 | 20.72444 | 10.693 |
| 120 | 83.7221 | 301.40 | −0.450662 | 19.60453 | 19.000 |
| 300 | 80.8299 | 290.99 | −0.450169 | 18.24589 | 29.412 |
| 600 | 79.9570 | 287.85 | −0.450018 | 17.84564 | 32.555 |
| 900 | 79.8672 | 287.52 | −0.450000 | 17.80472 | 32.878 |
| **1200** | **79.8579** | **287.49** | −0.450000 | 17.80049 | **32.911** |

The scrub peaks at **21.50 m/s²** and relaxes to **17.80 m/s²**, which is exactly the engine accel (`NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2 = 89/5 = 17.8`, `physics.step.service.ts:148`). The terminal state is an **equilibrium at 79.858 m/s = 287.49 km/h**, i.e. a **32.91 km/h** deficit against the 320.40 km/h top speed.

Closed-form check: `0.55·0.45·(v + 17.8/60)·(v/89) = 17.8` ⇒ v = 79.858 m/s. **Measured = 79.8579.** The law and the measurement agree to 4 significant figures.

⇒ **The model DOES lose traction at full speed. "The model cannot lose traction at full speed" is FALSE.** The magnitude is ~10 % of top speed, and it is a *terminal* shift reached over 15–20 s, not a rate the player feels in a 1–3 s corner (at t = 30 the deficit is 4.76 km/h; at t = 60, 10.69 km/h).

### 2.C The traction loss is a STEP FUNCTION of |steer| — not a proportion

Road, throttle = 1, entry 89 m/s, 600 ticks, measured deficit against the paired steer-0 baseline:

| \|steer\| | published slip | committed slip | turnSign | deficit (km/h) |
|---|---|---|---|---|
| 0.00 | 0 | 0 | 0 | **0.000000** |
| 0.10 | 0 | 0 | 0 | **0.000000** |
| 0.15 | 0 | 0 | 0 | **0.000000** |
| 0.20 | 0 | 0 | 0 | **0.000000** |
| 0.25 | 0 | 0 | 0 | **0.000000** |
| 0.40 | 0 | 0 | 0 | **0.000000** |
| **0.49** | 0 | 0 | 0 | **0.000000** |
| **0.50** | −0.450018 | −0.45 | 1 | **32.5392** |
| 0.51 | −0.450018 | −0.45 | 1 | 32.5399 |
| 0.60 | −0.450018 | −0.45 | 1 | 32.5448 |
| 0.75 | −0.450018 | −0.45 | 1 | 32.5500 |
| 0.90 | −0.450018 | −0.45 | 1 | 32.5532 |
| 0.95 | −0.450018 | −0.45 | 1 | 32.5541 |
| **1.00** | −0.450018 | −0.45 | 1 | **32.5547** |

Two facts, both load-bearing:

1. **Below 0.5 the loss is EXACTLY ZERO** — not small, *bit-exact* zero (the latch never opens, `drift.model.ts:514-525`). B2 is a **hard cliff**, not a fade.
2. **Above 0.5 the loss is essentially CONSTANT.** The whole 0.50 → 1.00 span is 32.5392 → 32.5547 km/h, a spread of **0.0155 km/h = 0.048 %**.

⇒ **There is no severity gradient in the traction model.** A gentle turn and a violent turn at the same speed produce the *same* 0.45 rad slide and the *same* terminal speed. This is the physical meaning of the user's "the tires still never lose traction".

### 2.D The traction loss is also entry-speed-INDEPENDENT

Road, throttle = 1, |steer| = 1, 600 ticks. The committed slip is −0.45 at **every** entry speed:

| entry (m/s) | terminal v (m/s) | published slip | deficit (km/h) |
|---|---|---|---|
| 89 | 79.9570 | −0.450018 | 32.555 |
| 80 | 79.8831 | −0.450005 | 32.821 |
| 70 | 79.7656 | −0.449984 | 33.244 |
| 60 | 79.6312 | −0.449960 | 33.728 |
| 50 | 79.4759 | −0.449933 | 34.287 |
| 40 | 79.2948 | −0.449901 | 34.939 |
| 30 | 79.0807 | −0.449864 | 35.709 |
| 20 | 78.8241 | −0.449818 | 36.633 |
| 16 | 78.7066 | −0.449798 | 37.056 |
| **15.1** | 78.6788 | −0.449793 | 37.156 |
| **15.0** | 78.6761 | −0.449792 | 37.166 |
| **14.0** | 78.6457 | −0.449787 | 37.275 |
| 10 | 78.5167 | −0.449764 | 37.740 |
| 5 | 78.3364 | −0.449732 | 38.389 |

The slide depth is *identical* at 5 m/s and 89 m/s (B2 × B3), and the deficit is a **near-constant 32.6 – 38.4 km/h** that *grows* as the entry speed falls — i.e. it tracks the **equilibrium shift**, not the speed at which the turn began. **A real friction circle would show the opposite: the faster the entry, the harder the bite.** The tire's grip budget is speed-blind by the G-2 pin (`drift.model.ts:567-579`), so nothing in the model can express that.

### 2.E The host supplies a TWO-STATE steer axis — the continuous envelope is unreachable

`host/input.ts:51` `PRESSED_AXIS_RATIO = 1`, `:54` `NEUTRAL_AXIS_RATIO = 0`, `:87-91`:

```
throttle: presses.up    ? 1 : 0
brake:    presses.down  ? 1 : 0
steer:    (presses.left ? 1 : 0) − (presses.right ? 1 : 0)
```

⇒ the live steer axis takes **only the values {−1, 0, +1}**. The engage threshold 0.5 (`drift.model.ts:189`) is therefore **always met** whenever a steer key is held, and the sub-threshold regime of 2.C is **unreachable on the keyboard path**.

This **falsifies the standing hypothesis** carried over from `AUDIT-PHYSICS` F-B2 *as applied to the live path* ("a full-speed turn with |steer| < 0.5 loses exactly zero speed"). That statement remains true **as a model property** but is **not a candidate explanation for the user's report**, because the host never produces |steer| < 0.5 except 0.

Measured, with a hypothetical analog ramp (steer 0 → 1 over 60 ticks): the latch opens at the first tick where |steer| ≥ 0.5 (t = 30, committed slip −0.02917 — the ramp is seeded on the engage tick), and by t = 60 the slip is at the 0.45 clamp. **The model is written for a continuous axis that the host does not provide.**

### 2.F The page BOOTS in AI mode, and the AI almost never crosses the threshold

* `host/mode-selector.ts:62-63` — `INITIAL_RACE_MODE = MODE_OPTION_AI` ("the page opens as a running AI-driven race").
* `host/race-session.ts:239-242` — in `ai` mode the session samples `sampleAutoInput(state.car)` **instead of** the keyboard sample.
* `physics/ai.autopilot.ts:30, :174` — `steer = clampSteer(headingError · 2.2)`.

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

### 2.G On the real demo track the OFF-ROAD model dominates the drift model

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

⇒ The claim "the tires never lose traction" is, on the live track, *almost* literally true: the surface, not the tire, is the whole story.

### 2.H The neutral wheel FREEZES the slide; only a countersteer unwinds it

`stepDriftCommitment` computes `countersteering = turnSign !== 0 && Math.sign(steer) === −turnSign` (`drift.model.ts:511-512`). `Math.sign(0) = 0 ≠ −turnSign`, so **releasing the key is not countersteering** — the "drifting" branch runs the entry ramp with `|steer| = 0` (`drift.model.ts:534-536`), i.e. `slip −= turnSign·3.5·0·dt`, i.e. **no change**. Measured (steer = 1 for 60 ticks, then steer = 0):

```
t = 60   com −0.45   slip −0.451044   scrub 2.92   deficit 10.693 km/h
t = 120  com −0.45   slip −0.450662   scrub 1.80   deficit 19.000 km/h
t = 300  com −0.45   slip −0.450169   scrub 0.45   deficit 29.412 km/h
t = 600  com −0.45   slip −0.450018   scrub 0.05   deficit 32.555 km/h
```

⇒ **Releasing the steering key leaves a permanent 0.45 rad slide that keeps scrubbing the car down to the 287.5 km/h terminal speed, forever.** The player cannot straighten out of a slide by lifting the wheel.

The **countersteer** does unwind it — and, measured, it unwinds it *identically* for every magnitude tested:

| countersteer | com @63 | com @66 | com @69 | com @72 | lat @63 | lat @66 | lat @69 |
|---|---|---|---|---|---|---|---|
| −0.40 | −0.325 | −0.200 | −0.075 | **0** | −36.273 | −25.916 | −14.715 |
| −0.50 | — | — | — | **0** (t=90) | | | |
| −0.89 | — | — | — | **0** (t=90) | | | |
| −0.90 | −0.325 | −0.200 | −0.075 | **0** | −33.933 | −22.875 | −11.518 |
| −1.00 | — | — | — | **0** (t=90) | | | |

The committed-slip unwind is **entirely** the `EXIT_RATE = 2.5 rad/s` ramp (`:537-549`): 0.45 / 2.5 = 0.18 s = 10.8 ticks, identical for all magnitudes. The 70 m/s² `COUNTERSTEER_AUTHORITY` (`:225`) only changes the *lateral-velocity* decay, by 2–3 m/s out of 15–40. **The authority is decorative in the recovery path.** And `NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS = 8` (`:251`) has **exactly one occurrence in the whole repository — its own declaration** (repo-wide grep over `*.ts`): it is dead.

After the unwind the latch is **released with `reArmed = false`** (`:541-544`), so the car **cannot re-engage a drift until the wheel is returned to neutral** (|steer| < 0.15, `:526-530`).

### 2.I The reachable slip set is EIGHT discrete values, and the only control is tap length

Because the entry ramp is 0.0583333 rad/tick and the clamp is 0.45, and because the neutral wheel freezes the slide (2.H), the *reachable* committed slips are exactly:

| key held (ticks) | committed slip | terminal v | note |
|---|---|---|---|
| 1 | −0.058333 | 88.952 | |
| 2 | −0.116667 | 88.905 | |
| 3 | −0.175000 | 88.857 | |
| 5 | −0.291667 | 88.763 | |
| 7 | −0.408333 | 83.910 | |
| **8 and up** | **−0.45** | **79.957** | clamp |
| released / 0 | 0 | 89.000 | never engaged |

⇒ the slip domain is **{0} ∪ {0.0583333·k : k = 1…7} ∪ {0.45}** — eight values, quantised to 1/60 s of key-hold time. This is an **accidental** control channel (a by-product of the L1 freeze), not a designed envelope, and it is not humanly modulable.

### 2.J The visual surface under-reports the physical slide by 2×

* `computeDriftSlideProxyRad` (`:655-659`) = `|slip| · |GAIN − 1|` = `0.45 · 0.5` = **0.225 rad (12.9°)** against a physical slide of **0.45 rad (25.8°)**.
* The body yaw the renderer draws is the pose spring `slip·0.5 + 0.08·articulation` (`:727-777`, GAIN = 0.5 at `:332`). Measured: **−0.153 rad (8.8°)** on road at full lock; **−0.205 rad (11.7°)** on grass at the floor.
* The camera rigid-mirrors the body (`browser-entry.ts:972` → `carPoseBag.driftYawOffsetRad = car.drift.pose.pose`).

⇒ the "odd camera angle" the user names is the **8.8° pose spring**, and the body-vs-travel angle is 25.8° − 8.8° ≈ 17° on road / ≈ 14° on grass. **The physical slide IS wired into the travel direction** (`physics.step.service.ts:470-486` integrates x, y along `velocityHeading = nextHeading + nextSlipAngle`) — so it is *not* a camera trick in the sense of "the physics is absent"; it is a camera trick in the sense that **the rendered body yaw under-reports the physical slide by 2×**. Adjudicated in `AUDIT-VISUALS`; recorded here only as the envelope-side cause of the perception.

---

## 3. Decision

### D1 — The envelope, re-derived numerically

| quantity | value | source |
|---|---|---|
| engage threshold | 0.5 | `drift.model.ts:189` |
| entry ramp | 3.5 rad/s × \|steer\|, **speed-independent** | `:201`, `:520-536` |
| ticks to the 0.45 clamp at \|steer\| = 1 | **8** (0.133 s) | measured |
| reachable committed slips | **{0} ∪ {0.0583333·k, k=1…7} ∪ {0.45}** | measured (2.I) |
| chase time constant | 0.045 s (published slip reaches −0.4511 in ~45 ticks) | `:219`, measured |
| scrub law | `0.55·\|published slip\|·v·(v/89)` m/s² | `:632-646` |
| peak scrub at full lock | **21.50 m/s²** (at t = 30) | measured |
| terminal drift speed (engine-balanced) | **79.858 m/s = 287.49 km/h** | measured, analytic-identical |
| terminal speed deficit | **32.91 km/h** (10.3 % of the 320.40 km/h cap) | measured |
| traction loss for \|steer\| < 0.5 | **exactly 0.000000** | measured (2.C) |
| traction loss spread over \|steer\| 0.5 → 1.0 | **0.0155 km/h (0.048 %)** | measured (2.C) |
| traction loss spread over entry speed 5 → 89 m/s | 32.56 → 38.39 km/h (equilibrium shift, not a bite) | measured (2.D) |
| slide depth at 5 m/s vs 89 m/s | **identical (−0.45)** | measured (2.D) |

### D2 — Why the user sees ZERO drift (ranked by measured contribution)

1. **The page boots in AI mode** (`mode-selector.ts:63`, `race-session.ts:239-242`) and **the AI engages a drift on 2.444 % of ticks** (2.F). For 97.6 % of the run the tires are *bit-exactly* not slipping. **This is the strongest and most likely proximate cause.**
2. **On the real demo track the car is on grass 921/1200 ticks and pinned at the 80.1 km/h off-road floor** (2.G). There, the off-road decel (−44.5 m/s²) is **32 : 1** larger than the drift scrub (1.39 m/s²). Whatever the player feels is surface drag, not tire slip.
3. **The model has no severity gradient** (2.C): every turn above the threshold yields the same 0.45 rad slide and the same terminal speed. There is nothing to "feel" as a deeper or a harder drift.
4. **The traction loss is a terminal shift, not a rate** (2.B): a 1–3 s corner yields 4.8 – 19.0 km/h, which reads as "the car slowed a bit", not as "the tires let go".
5. **The rendered body yaw under-reports the physical slide by 2×** (2.J) — the perception of "just a camera angle" (cross-ref `AUDIT-VISUALS`).
6. **Residual, not adjudicated here**: whether the live rAF loop actually steps the drift channel (`AUDIT-MOTION`).

**Correction to the dispatch premise.** "The model cannot lose traction at full speed" is **FALSE**: the model loses 32.91 km/h at full lock on road, and the scrub reaches 21.50 m/s² — larger than the engine's 17.8 m/s². The premise survives only in the *live-track* sense (2.G), where the surface dominates, and in the *AI-mode* sense (2.F), where the threshold is never crossed.

### D3 — S-1 / S-2 / S-3 under the frozen B1–B4 constants

| proposal | verdict | proof |
|---|---|---|
| **S-1** traction loss proportional to **speed** (a real friction circle) | **IMPOSSIBLE without re-opening G-2 (B4) and B2** | the grip budget is speed-blind by contract (`:567-579`); the slide depth is 0.45 rad at every speed (2.D); the observable deficit is a constant equilibrium shift, not a bite (2.D) |
| **S-2** traction loss proportional to the **slide magnitude** | **FORMALLY PRESENT, DEGENERATE IN PRACTICE** | the law is already `∝ \|published slip\|` (`:632-646`) — but the reachable slip set is **8 discrete values** quantised to 1/60 s of tap length (2.I), because B2 gates it, B3 clamps it, and the L1 freeze is the only (accidental) way to land on an intermediate value |
| **S-3** traction loss proportional to the **steering effort / turn severity** | **IMPOSSIBLE** | measured: 0.50 → 32.5392 km/h, 1.00 → 32.5547 km/h — a **0.048 %** spread (2.C). The scrub reads the *clamped published* slip (B3), not the steer and not the yaw rate (`:632-646`) |

⇒ **Under B1–B4 the drift model can emit exactly ONE traction-loss magnitude: a step function at |steer| ≥ 0.5.** There is no continuous, steer-proportional, speed-proportional traction loss available. Delivering S-1/S-2/S-3 requires re-opening **B2** (the speed-independent entry rate), **B3** (the 0.45 clamp) **and G-2 (B4)** — and **G-2 is NOT in the verdict's contested set and must be added to it.** G-2 is the binding constraint: without it no friction circle can exist at any speed.

### D4 — Numbered gaps (file:line)

| # | gap | file:line | confidence | evidence |
|---|---|---|---|---|
| **1** | The traction loss is a **step function** of \|steer\|, not a proportion: 0.49 → exactly 0 km/h, 0.50 → 32.5392 km/h, 1.00 → 32.5547 km/h (0.048 % spread). No severity gradient exists for the player to feel. | `drift.model.ts:189`, `:632-646`, `:66` | 0.95 | measured (2.C) |
| **2** | **S-3 is impossible** under the frozen constants. The scrub reads the *clamped published* slip, so above the 0.45 clamp the law is speed-only. | `drift.model.ts:66`, `:632-646`, `:551-555` | 0.93 | measured (2.C) + source |
| **3** | **G-2 (the speed-independent grip budget) is the binding constraint on S-1 and is NOT in the verdict's contested set** — it must be added, or S-1 can never be delivered. | `drift.model.ts:567-579` | 0.90 | static + measured (2.D) |
| **4** | The **engage threshold is a bit-exact cliff**: below 0.5 the traction loss is not "small", it is **0.000000** (the latch never opens). No fade-in exists. | `drift.model.ts:189`, `:514-525` | 0.95 | measured (2.C) |
| **5** | The **host supplies a two-state steer axis** {−1, 0, +1}; the model's continuous-control envelope is unreachable from the keyboard, so the sub-threshold regime is dead code in practice and every turn is a full-lock 0.45 rad slide. | `host/input.ts:51`, `:54`, `:87-91` | 0.95 | static (2.E) |
| **6** | The **page boots in AI mode** and the **AI engages a drift on 2.444 % of ticks** (heading error must exceed 13.02°). The zero-drift report is fully explained for the default boot. | `host/mode-selector.ts:62-63`, `host/race-session.ts:239-242`, `physics/ai.autopilot.ts:30`, `:174`, `drift.model.ts:189` | 0.90 | static + measured (2.F) |
| **7** | The **off-road model dominates the drift model on the demo track**: grass 921/1200 ticks, the 22.25 m/s floor pins the car at 80.1 km/h, and the off-road decel is **32 : 1** larger than the drift scrub at that speed. The "full-speed turn" is an 80 km/h turn on grass. | `physics.step.service.ts:173`, `:182`, `browser-entry.ts:512-516`, `:733` | 0.92 | measured (2.G) |
| **8** | The **neutral wheel FREEZES the slide** (L1 confirmed): releasing the key leaves a permanent 0.45 rad slide that keeps scrubbing to the terminal speed. The player cannot straighten out by lifting the wheel. | `drift.model.ts:511-512`, `:534-536` | 0.93 | measured (2.H) |
| **9** | The **countersteer authority is decorative** in the recovery path: the 2.5 rad/s exit ramp unwinds the slide identically at \|steer\| = 0.4 and 1.0; the 70 m/s² authority moves the lateral velocity by only 2–3 m/s of 15–40. And `COUNTERSTEER_HOLD_TICKS` has **no reader anywhere in the repo**. | `drift.model.ts:214`, `:225`, `:237`, `:251`, `:537-549` | 0.88 | measured (2.H) + repo-wide grep |
| **10** | The **reachable slip domain is 8 discrete values** quantised to 1/60 s of key-hold time — an accidental by-product of the L1 freeze, not a designed control surface. S-2 is therefore formally present but unusable. | `drift.model.ts:66`, `:201`, `:520-536`, `:551-555` | 0.92 | measured (2.I) |
| **11** | The **rendered body yaw under-reports the physical slide by 2×** (proxy 0.225 rad vs physical 0.45 rad; pose 0.153 rad). The physical slide *is* wired into the travel direction, so this is a *presentation* under-report, not an absent physics. | `drift.model.ts:332`, `:655-659`, `:727-777`, `browser-entry.ts:972` | 0.90 | static + measured (2.J) → `AUDIT-VISUALS` |
| **12** | The **traction loss has no published channel**: the scrub is the only tire-load term and is never surfaced on the drift channel, so neither the HUD nor the composer can express "the tires let go". (Inherited F5, now with a measured magnitude.) | `physics.step.service.ts:419-427`, `drift.model.ts:282` | 0.80 | static |

### D5 — Confidence summary

| finding | confidence | provenance | act |
|---|---|---|---|
| The model loses 32.91 km/h at full lock on road | **0.95** | runtime (measured, reconciled to 1e-9) | ACT |
| The loss is steer-independent above 0.5 | **0.95** | runtime | ACT |
| The loss is exactly 0 below 0.5 | **0.95** | runtime | ACT |
| The slide depth is speed-independent (0.45 at 5 and 89 m/s) | **0.95** | runtime | ACT |
| The AI engages a drift 2.444 % of the time; the page boots in AI mode | **0.90** | runtime + static | ACT (see D2.6 for the residual) |
| The off-road model dominates on the demo track | **0.92** | runtime | ACT (cross-ref `AUDIT-MOTION`) |
| The neutral-wheel freeze (L1) | **0.93** | runtime | ACT |
| `COUNTERSTEER_HOLD_TICKS` is dead | **0.90** | static (repo-wide grep, 1 hit) | ACT |
| The countersteer authority is decorative | **0.88** | runtime | DELEGATE — a reviewer should confirm the intended feel contract before the constant is deleted |
| The 2× visual under-report | **0.90** | static + runtime | DELEGATE — `AUDIT-VISUALS` owns the decision |
| "The user sees zero drift" — full causal attribution | **0.78** | runtime (parts 1–5) + residual | DELEGATE — `AUDIT-MOTION` (does the live loop step the drift?) and `AUDIT-VISUALS` (is the slide visible?) must close the last 22 % |

---

## 4. Risks

- **R1 — The zero-drift attribution is 78 % confident, not higher.** D2.1 (AI mode) and D2.2 (off-road dominance) are each sufficient to produce the report; D2.3–D2.5 are contributory. The residual 22 % sits in the motion loop (`AUDIT-MOTION`) and the presentation (`AUDIT-VISUALS`). **Do not close VERDICT 5's O_3 on this artifact alone.**
- **R2 — G-2 is a ratified contract, not a free constant.** `[RED-G2]` asserts `computeGripBudgetFactor(0.9, 1) === computeGripBudgetFactor(0.9, 0.2)`. Delivering S-1 requires **re-opening a ratified pin** and re-writing that test. The verdict's contested set (B1–B4) does not include it; **the orchestrator must add it**, or S-1 is undeliverable by construction.
- **R3 — The measured terminal speed (79.8579 m/s) is the equilibrium, not a transient.** A fixture that asserts the deficit at a fixed 600 ticks (as the D1.3 pin does) will read 32.555 km/h, not 32.911 km/h. Any new RED contract must pin the *equilibrium*, or it will be a vacuous floor assertion (the inherited F-E criticism).
- **R4 — The off-road floor can MASK a drift defect.** A car pinned at 22.25 m/s shows a scrub of 1.39 m/s²; a traction-loss regression would be invisible on the demo track. New traction contracts must run on a **road-only** surface resolver, or they will be vacuous.
- **R5 — The neutral-wheel freeze makes the drift STICKY, which is the opposite failure mode from the user's report and could mask a fix.** A fix that only touches the ramp will not change the reported symptom, because the symptom is dominated by D2.1/D2.2. **Order the fixes: mode/AI → off-road → envelope.**
- **R6 — `AUDIT-PHYSICS` F-B2 must be re-read as a *model* property, not a *live-path* explanation.** This wave's 2.E falsifies the live-path reading of it (the host never produces |steer| < 0.5). Rank: runtime (this wave) > static (`AUDIT-PHYSICS`).
- **R7 — The off-road published-slip ceiling (0.493, `:163`) is reachable on the demo track** (the AI measured max 0.466686) and the grass run reached −0.467015, so the grass path has its own envelope, not a rescaled road envelope. Any traction contract must state the surface.
- **R8 — Measurement hygiene.** The probe wrote only to `%TEMP%`; no repository byte and no git ref was touched. Reproduce with: `esbuild --bundle --platform=node --format=cjs --loader=ts` over a stdin probe importing `examples/neatRacing/physics/physics.step.service.ts` + `examples/racing_curriculum/track/track.generator.ts`, then `node` the bundle. **The reconciliation invariant to re-check is `scrub_measured − scrub_model == 0` at every tick** — if it is non-zero, the integrator assumption (`min(89, v + 17.8·dt)`, `physics.step.service.ts:567-601`) has drifted and every deficit figure here is void.

---

## 5. Evidence index (file:line)

| file:line | fact used |
|---|---|
| `examples/neatRacing/physics/drift.model.ts:66` | `MAX_SLIP_ANGLE_RAD = 0.45` (B3) |
| `examples/neatRacing/physics/drift.model.ts:94`, `:128`, `:163` | published-slip overshoot 0.002 / max 0.452 / off-road max 0.493 |
| `examples/neatRacing/physics/drift.model.ts:169`, `:177` | grip falloff threshold 0.2, rate 2 |
| `examples/neatRacing/physics/drift.model.ts:184` | `MIN_SPEED_MPS = 15` (B1) |
| `examples/neatRacing/physics/drift.model.ts:189` | `STEER_ENGAGE_THRESHOLD = 0.5` (B2) |
| `examples/neatRacing/physics/drift.model.ts:195` | `STEER_NEUTRAL_THRESHOLD = 0.15` |
| `examples/neatRacing/physics/drift.model.ts:201`, `:214` | `ENTRY_RATE 3.5`, `EXIT_RATE 2.5` rad/s |
| `examples/neatRacing/physics/drift.model.ts:219` | `SLIP_RESPONSE_TAU_S = 0.045` |
| `examples/neatRacing/physics/drift.model.ts:225`, `:237`, `:251` | countersteer 70 m/s² / min steer 0.9 / **HOLD_TICKS 8 (dead)** |
| `examples/neatRacing/physics/drift.model.ts:282` | `SPEED_SCRUB_COEFF_PER_S = 0.55` |
| `examples/neatRacing/physics/drift.model.ts:314`, `:332` | `POSE_STEER_YAW_RAD 0.08`, `VISUAL_YAW_GAIN_RAD 0.5` |
| `examples/neatRacing/physics/drift.model.ts:464-472` | `createZeroDriftChannelState` — `reArmed: true` at spawn |
| `examples/neatRacing/physics/drift.model.ts:489-560` | `stepDriftCommitment`: MIN_SPEED gate `:494-501`; ENGAGE `:514-525`; RE-ARM `:526-530`; ramp `:531-536`; unwind `:537-549`; clamp `:551-555` |
| `examples/neatRacing/physics/drift.model.ts:567-579` | `computeGripBudgetFactor` — **`speedPercent` accepted and ignored (G-2)** |
| `examples/neatRacing/physics/drift.model.ts:591-618` | `computeCountersteerAuthority` |
| `examples/neatRacing/physics/drift.model.ts:632-646` | `computeLongitudinalSpeedScrub` — `0.55·\|slip\|·v·speedPercent` |
| `examples/neatRacing/physics/drift.model.ts:655-659` | `computeDriftSlideProxyRad` = `\|slip\|·\|GAIN−1\|` |
| `examples/neatRacing/physics/drift.model.ts:670-725` | `computeNextLateralVelocity` — MIN_SPEED hard zero, chase, countersteer, hard zero |
| `examples/neatRacing/physics/drift.model.ts:727-777` | `stepDriftPoseSpring` — pose target `slip·0.5 + 0.08·articulation`, **unclamped** |
| `examples/neatRacing/physics/physics.step.service.ts:133` | fixed timestep 1/60 |
| `examples/neatRacing/physics/physics.step.service.ts:142`, `:148` | max speed 89; accel 89/5 = 17.8 |
| `examples/neatRacing/physics/physics.step.service.ts:173`, `:182` | off-road decel −89/2 = −44.5; off-road floor 89/4 = 22.25 |
| `examples/neatRacing/physics/physics.step.service.ts:192` | `STEERING_RATE = 2` rad/s |
| `examples/neatRacing/physics/physics.step.service.ts:419-427` | the scrub call (tick-start slip, tick-end speed, tick-start speedPercent) |
| `examples/neatRacing/physics/physics.step.service.ts:455-461` | the pose spring step |
| `examples/neatRacing/physics/physics.step.service.ts:470-486` | motion integrated along `velocityHeading = nextHeading + nextSlipAngle` |
| `examples/neatRacing/physics/physics.step.service.ts:506-513` | `resolveSpeedPercent = clamp(speed/89)` |
| `examples/neatRacing/physics/physics.step.service.ts:567-601` | `integrateLongitudinalSpeed` — `min(89, v + 17.8·dt)` on road |
| `examples/neatRacing/host/input.ts:51`, `:54`, `:87-91` | `PRESSED_AXIS_RATIO = 1`; the two-state steer axis |
| `examples/neatRacing/host/mode-selector.ts:57-63` | `INITIAL_RACE_MODE = 'ai'` |
| `examples/neatRacing/host/race-session.ts:239-242` | ai mode samples the autopilot instead of the keyboard |
| `examples/neatRacing/physics/ai.autopilot.ts:30`, `:174` | `STEER_GAIN_PER_RAD = 2.2`; `steer = clampSteer(headingError · 2.2)` |
| `examples/neatRacing/browser-entry/browser-entry.ts:512-516` | the demo track generation spec |
| `examples/neatRacing/browser-entry/browser-entry.ts:733` | `createTrackSurfaceResolver(trackSpec.segments)` |
| `examples/neatRacing/browser-entry/browser-entry.ts:972` | `driftYawOffsetRad = car.drift.pose.pose` |
| `examples/neatRacing/browser-entry/browser-entry.ts:1147-1149` | the keyboard / autopilot / `stepPhysics` seams |

---

## 6. Merge notes for Agent Zero

1. **The dispatch premise is corrected**: the model *does* lose traction at full speed (32.91 km/h, 21.50 m/s² peak scrub). The "no traction" experience is explained by **AI-mode boot (2.444 % engagement)**, **off-road dominance (32 : 1)**, **the absent severity gradient**, and the **2× visual under-report** — not by an empty envelope.
2. **G-2 must be added to the contested set.** Without re-opening the speed-independent grip budget, S-1 is undeliverable. S-3 is undeliverable under B1–B4 **at all**, and S-2 exists only as an 8-value accidental domain.
3. **Fix order**: mode/AI → off-road → envelope. A pure envelope fix will not move the reported symptom (R5).
4. **The shared plan tracker was deliberately NOT edited** by this wave — four audit waves are running in parallel against the same plan tail and a concurrent append risks a merge conflict. This sibling artifact is the durable record; the orchestrator owns the `PlanUpdate` row.
5. **No child agents were dispatched** — the caller's "NO child agents" order overrides the generic Tier-1 delegation mandate, consistent with `AUDIT-PHYSICS` R8.
6. **No browser and no Chrome MCP were used.** The measurement was a headless Node run of the production modules.
