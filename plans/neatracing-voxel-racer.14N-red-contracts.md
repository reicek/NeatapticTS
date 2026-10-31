# Step 14N — `P1-drift-feel-red` — RED CONTRACT SET

> **Phase-B plan agent 1/4 (RED contracts).** Author: `02-researching` (Tier-1).
> Date: 2026-10-26. Active plan: `plans/neatracing-voxel-racer.plans.md` (Step 14N, [WIP]).
> Active slice: `P1-drift-feel-red` (red-testing, tdd_sequence red→green).
>
> **Authority.** This file is a *proposal*, not a plan edit. It specifies the RED
> contract set that merge items **7a–7g** demand, plus the physical umbrella
> contracts the merged design cannot be delivered without, the re-baselined green
> pins that the re-baseline breaks, and the decision ledger that the merge left to
> the RED slice. `01-planning` must fold it into the plan + the slice packet
> BEFORE any production edit.
>
> **Write scope honoured:** this file only. No plan edit, no production edit, no
> git mutation.

> **WRITE STATUS: COMPLETE (2026-10-26).** 1047 lines / ~73.9 kB / 15 sections.
> §0 scope · §1 evidence base · §2 unit-system ledger · §3 `SCN-DRIFT-01` ·
> §4 merge items 7a–7g · §5 the P0 physical umbrella contracts · §6 the
> re-baselined green-pin ledger · §7 the GONE-pin ledger · §8 the decision
> ledger · §9 the file set · §10 the browser-lane changes · §11 the open items
> for `01-planning` · §12 the risks · §13 the certainty table · §14 the handoff ·
> **§15 the purge-proof structured output contract + validation evidence.**
>
> **If you are reading this file after an agent purge:** everything the Phase-B
> plan agent 1/4 produced is IN this file. The chat report is a courtesy only.
> §15 carries the machine-readable contract, the gate result, the write-scope
> attestation and the exact resume instructions.

---

## 0. What this file is / is not

| | |
|---|---|
| **IS** | The exact RED contract set for `P1-drift-feel-red`: test homes, unit systems, floors/ceilings, verbatim assertion text, per merge item 7a–7g. |
| **IS** | The re-baselining ledger (the 14 green pins the merge's constant change breaks) and the GONE ledger (the retired symbols). |
| **IS** | The decision ledger for the values the merge explicitly delegated to the RED slice (the in-band slew, the AC-087 floor, the measurement conventions). |
| **IS NOT** | A plan edit. The plan's Step-14N packet, the AC-087..AC-090 rows and the slice file list are **stale** (see §11 O-1) and `01-planning` owns that patch. |
| **IS NOT** | An implementation. No production file is touched here. |
| **IS NOT** | A new test file. All four jest homes and the browser fixture already exist — the RED phase **updates** them, it does not create them. |

---

## 1. Evidence base

| # | Source | What it fixes | Certainty |
|---|---|---|---|
| E1 | `plans/neatracing-voxel-racer.plans.md:8414-8460` (Phase-A MERGE block, Agent Zero central merge, 2026-10-26) | the adopted design, the constant set, merge items 7a–7g, the file set, the risk list | 0.97 |
| E2 | `plans/neatracing-voxel-racer.drift-feel.research.md` (54.9 kB) D1 (7-stage model), D2 (constants), D3 (measured predictions), D6.1–D6.10 (game-feel amendment) | the model semantics, the measured pop/settle/step, the envelope re-derivation, the V-shaped gain analysis | 0.93 |
| E3 | `plans/neatracing-voxel-racer.research.md` (30.8 kB) E1–E7 / D1–D7 / R1–R11 | the proxy identity, the AC-088 blindness proof, the AC-047D byte-pin, the constraint table | 0.90 |
| E4 | `examples/neatRacing/physics/drift.model.ts` (read directly, 372 lines) | the current exports/const values that the re-baseline replaces | 0.99 |
| E5 | `examples/neatRacing/physics/physics.step.service.ts` (read directly, 570 lines) | `PhysicsCarState` L73-111, the single pose insertion point L416-417, the tick owner | 0.99 |
| E6 | `examples/neatRacing/physics/{drift.model,physics.step.service,collision.utils}.test.ts` (read directly) | the exact current green assertion text that §6 re-pins | 0.99 |
| E7 | `examples/neatRacing/engine/lighting.test.ts` (read directly, OBS-20 describe L1024-1095) + `engine/passes/car.pass.ts` (read directly) | the `uTireGradient*` uniform seam, the OBS-20 home, the exported lever constants | 0.98 |
| E8 | `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html` (read directly) | the `uniform1f` hook L189-197, the 3-frame gate L128-132 / L815-880, the named-check list | 0.96 |
| E9 | Tier-3 `implementation-pattern-scout` (read-only dispatch, `ok:true`) | the four RED homes, the house idioms, the reusable helpers, the four green pins that break | 0.90 |
| E10 | Tier-3 `boundary-mapper` (read-only dispatch, `ok:true`) | the source anchors, the insertion point, the collision-bounce gap, the 10 stale-vs-merge deltas | 0.90 |

**Direct-read note.** Cortex was consulted first (`get_slice_context`, `load_document`).
The plan is 1.9 MB and the research artifacts 30–55 kB, and the Cortex index does not
cover `examples/neatRacing` sources, so the file:line anchors were taken from direct
worktree reads of known paths (the permitted degraded-Cortex fallback).

---

## 2. Unit-system ledger — ONE unit per quantity, no exceptions

| Quantity | Unit | Where it is produced | Where it is consumed |
|---|---|---|---|
| slip angle | **rad** (signed; a LEFT turn is NEGATIVE) | `drift.model.ts` `computeSlipAngle` / the model state | the pose target, the proxy, the envelope |
| visible yaw offset (pose) | **rad** | the pose spring, on the 60 Hz tick | `PhysicsCarState.driftYawOffset` → `browser-entry.ts:941` → `car.pass.ts:1564-1567` |
| drift **slide proxy** | **rad** = `|slipAngle| · |GAIN − 1|` | `computeDriftSlideProxyRad(slipAngle)` (NEW, §8 D-3) | the AC-087 prominence floor |
| the *browser* proxy | **rad** (render-level, chord-measured, render-rate dependent) | the fixture, `analyzeWheelShading`/chord math | the fixture's engagement gate + the loose prominence mirror |
| pop / settle / step | **deg** for the pop and the step, **ticks (60 Hz)** for the settle | the pose-spring measurement | the three 7a contracts |
| tire-gradient spread | **dimensionless factor** (`bright − dark`) | `car.pass.ts` `feedTireGradientUniforms` | the AC-088 magnitude floor |
| speed | **m/s** (`units/s`; `MAX_SPEED_UNITS_PER_S = 89`) | `physics.step.service.ts` | `speedPercent = speed / 89` |
| time | **s** (`dt = 1/60` exactly) | the fixed timestep | every rate constant |

**Unit traps this file closes.**
1. `NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD` is a **PROXY** (rad of *apparent* slide), NOT a
   slip angle. A slip floor of `0.20` rad would be a different (and wrong) contract.
2. The **browser** proxy and the **deterministic** proxy are the same formula but different
   measurements (render-level chord vs. model state). The recorded browser spread
   `0.0494 / 0.0603 / 0.3599 rad` is a **7× spread** → it is noise, never a floor anchor
   (merge 7c). The deterministic unit proxy is the floor anchor.
3. The fixture's `DRIFT_PROXY_MIN_RAD = 0.01` is the **engagement** gate ("is the drift
   engaged?"), NOT the **prominence** floor ("is the slide big enough?"). They must never
   be conflated, and `DRIFT_PROXY_MIN_RAD` is **not** renamed by this slice.

---

## 3. `SCN-DRIFT-01` — the ONE named scenario every contract uses

```ts
/**
 * SCN-DRIFT-01 - the single named drift scenario for Step 14N.
 *
 * 60 Hz, dt = 1/60 exactly. Full throttle, no brake, top speed.
 *   Phase A (BUILD)      : 180 ticks (3.000 s), steer = +1  (full LEFT)
 *   Phase B (COUNTER)    : 120 ticks (2.000 s), steer = -1  (full RIGHT)
 *
 * Sign convention (production-pinned): a LEFT turn produces a NEGATIVE slip.
 * The speed is held at 89 m/s (speedPercent = 1) by the throttle: the new speed
 * scrub is 0.25 * 0.453 * 89 = 10.1 m/s^2 against the engine's 17.8 m/s^2, so
 * the car stays at the speed clamp for the whole scenario.
 *
 * The pose reference for the pop and the settle is
 *   poseFinal = the pose at the LAST tick of Phase B.
 */
const SCN_DRIFT_01_DT_S = 1 / 60;
const SCN_DRIFT_01_BUILD_TICKS = 180;
const SCN_DRIFT_01_COUNTER_TICKS = 120;
const SCN_DRIFT_01_MAX_SPEED_M_PER_S = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S; // 89
```

`SCN-DRIFT-01` is the scenario the arcade-refs/gaming research simulated (D3: "full left
steer at t = 1.0 s, full countersteer at t = 3.0 s"), so the measured design values
(pop 2.22°, settle 550 ms, step 2.13°) are the values this scenario must reproduce.

**Why one scenario.** Merge 7a pins three numbers **together at one
(ω, ζ, art, arm, slew) point** — never the hold pose alone. Splitting them across
scenarios would let a future tune satisfy one and break another.

---

## 4. The contract set — merge items 7a–7g (REQUIRED)

Classification legend: **RED** = fails today (the red-phase artifact) · **GUARD** = passes
today and must still pass at green (a regression guard, never a red).

### 7a — the far-side pop floor + settle + max per-tick pose-target step, pinned TOGETHER

> Merge text: *"(a) far-side pop floor (~2.0 deg at the recommended point) + settle (550 ms)
> + max per-tick pose-target step (2.13 deg) pinned TOGETHER at one (omega, zeta, art, arm,
> slew) point - never the hold pose alone"*

**Home:** `examples/neatRacing/physics/drift.model.test.ts` (new describe
`'drift pose spring feel contract (AC-087, merge 7a)'`).
**Home (integration twin):** `examples/neatRacing/physics/physics.step.service.test.ts`
(the pose that `stepPhysics` writes must equal the pose the pure spring produces).

**The pinned point (one point, all three numbers):**

```ts
const POSE_POINT = {
  stiffnessRadPerS2: NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2,   // 324   (omega = 18 rad/s)
  dampingRadPerS: NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S,         // 12.6  (zeta  = 0.35)
  articulationRad: NEATRACING_DRIFT_POSE_STEER_YAW_RAD,            // 0.08
  armSlipRad: NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD,       // 0.01
  slewPerS: NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S,         // 20    (band 15-30)
};
```

**The measurement convention (a contract, not an implementation detail — §8 D-5/D-6):**

```ts
// poseTarget(t) = slipAngle(t) * VISUAL_YAW_GAIN_RAD + POSE_STEER_YAW_RAD * articulation(t)
//   articulation(t) is the POST-tick value carried in the state  (contract, see D-8)
// poseFinal    = the pose at the last tick of Phase B
// popDeg       = max |pose(t) - poseFinal| over the Phase B ticks          -> 2.22 deg
// settleTicks  = the first Phase-B tick after which |pose - poseFinal| < 0.5 deg
//                 for EVERY remaining tick                                 -> 33 ticks
// maxStepDeg   = max |poseTarget(t) - poseTarget(t-1)| over ALL ticks      -> 2.13 deg
```

**The exact assertions:**

```ts
it('AC-087: the far-side pop is preserved at the pinned pose point (merge 7a)', () => {
  const run = runDriftScenario(SCN_DRIFT_01);

  // The pop is a CORPUS contract with a stated MINIMUM (D6.7a): a future
  // "tidy-up" that raises zeta to 0.45 or omega to 24 must not silently
  // halve it. Measured 2.22 deg at slew 20; the floor is 2.0 deg.
  expect(run.popDeg).toBeGreaterThanOrEqual(2.0);
  expect(Number.isFinite(run.popDeg)).toBe(true);
});

it('AC-087: the pose settles within the named window after the countersteer (merge 7a)', () => {
  const run = runDriftScenario(SCN_DRIFT_01);

  // Measured 33 ticks (550.0 ms). The band is 30..34 ticks (500..566.7 ms):
  // the upper edge is the merge's 550 ms + one tick of measurement slack,
  // the lower edge stops a "tidy-up" from killing the ring-out outright.
  expect(run.settleTicks).toBeGreaterThanOrEqual(30);
  expect(run.settleTicks).toBeLessThanOrEqual(34);
});

it('AC-087: the per-tick pose-target step stays inside the slew bound (merge 7a)', () => {
  const run = runDriftScenario(SCN_DRIFT_01);

  // Measured 2.13 deg/tick; the arcade's own raw-steer term steps
  // -4.65 deg/tick at 49.8 m/s and 9.17 deg/tick at 89 m/s, so the slew
  // bound is the whole point of the amendment. Ceiling 2.20 deg (3.3% headroom).
  expect(run.maxStepDeg).toBeLessThanOrEqual(2.2);

  // The SLEW-INDEPENDENT half, exactly: the articulation term can never step
  // the target by more than ARTICULATION_RAD * SLEW * dt, at ANY speed.
  expect(run.maxArticulationStepDeg).toBeLessThanOrEqual(
    (NEATRACING_DRIFT_POSE_STEER_YAW_RAD *
      NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S *
      SCN_DRIFT_01_DT_S) * (180 / Math.PI),
  );
});

it('AC-087: the three numbers are pinned at ONE point, not independently (merge 7a)', () => {
  // The anti-teardown guard: the pop, the settle and the step are read from a
  // SINGLE run of the SINGLE pinned point. A green pop bought with a different
  // (omega, zeta, art, arm, slew) is a contract violation.
  const run = runDriftScenario(SCN_DRIFT_01, POSE_POINT);

  expect(run.point).toEqual(POSE_POINT);
  expect(run.popDeg).toBeGreaterThanOrEqual(2.0);
  expect(run.settleTicks).toBeLessThanOrEqual(34);
  expect(run.maxStepDeg).toBeLessThanOrEqual(2.2);
});
```

**Status: RED.** Today `drift.model.ts` exports none of `POSE_STIFFNESS`,
`POSE_DAMPING`, `POSE_ARTICULATION_SLEW_PER_S`, `POSE_STEER_YAW_FULL_SLIP_RAD` and has no
pose spring at all → the import fails → the suite is red for the right reason
("module has no exported member"), not for a fixture reason.

**Envelope coupling (7e, cross-reference).** The pop and the envelope are the SAME
trajectory: the measured peak `|pose| = 0.2246 rad` must stay inside the
`MAX_SLIP · GAIN = 0.2250 rad` limit. A tune that raises the pop above the floor must not
push the peak past the limit. That is why 7e pins the pose-held-inside variant.

---

### 7b — zero-straight-line yaw as an EXACT identity (1e-9; measured 0.000000000)

> Merge text: *"(b) zero-straight-line yaw as an EXACT identity (1e-9, measured
> 0.000000000)"* — D6.7: *"not a tolerance"*.

**Home:** `drift.model.test.ts` (pure) + `physics.step.service.test.ts` (integration).
**Unit:** rad.

```ts
it('AC-087: a straight line carries EXACTLY zero visible yaw (merge 7b)', () => {
  // Pure accumulator: with slipAngle = 0 the articulation target is
  // steer * speedPercent * min(1, 0 / ARM) = 0, so the articulation never
  // leaves 0, the target never leaves 0, and the pose never leaves 0.
  let spring = createInitialDriftPoseSpringState();
  for (let tick = 0; tick < 600; tick += 1) {
    spring = stepDriftPoseSpring(spring, {
      slipAngle: 0,
      steer: 1,              // a steering input must not create a yaw either
      speedPercent: 1,
      dt: SCN_DRIFT_01_DT_S,
    });
  }

  expect(spring.pose).toBe(0);          // EXACT identity
  expect(spring.poseRate).toBe(0);
  expect(spring.articulation).toBe(0);
});

it('AC-087: the whole physics step keeps a zero straight-line yaw (merge 7b)', () => {
  const state = runSteps(600, 42, { throttle: 1, brake: 0, steer: 0 });

  expect(state.driftYawOffset).toBe(0);                 // EXACT
  expect(state.slipAngle).toBe(0);                      // EXACT
  expect(Math.abs(state.driftYawOffset)).toBeLessThan(1e-9);  // the 1e-9 contract
});
```

**Status: RED** (no pose spring exists) for the pure form; the integration form is
**GUARD** for the `slipAngle === 0` leg and **RED** for the pose leg.

**Why it is an identity, not a tolerance.** The articulation is *slip-gated*
(`min(1, |slip| / ARM)`), so on a straight line every factor is exactly `0` and the
float arithmetic is exact. A tolerance-based contract would let a future un-gating of the
articulation term (the −4.58° residual of D3's variant check) pass unnoticed. The
`toBe(0)` form is therefore mandatory, and the `1e-9` form is the *outer* bound that
survives a future refactor of the gate.

---

### 7c — the AC-087 floor re-pinned on the DETERMINISTIC unit proxy

> Merge text: *"(c) AC-087 floor re-pinned on the DETERMINISTIC unit proxy (browser-proxy
> spread 0.0494/0.0603/0.3599 rad is noise, not a floor anchor)"*

**Home:** `drift.model.test.ts` (the deterministic lane — the floor anchor) +
the browser fixture (the loose visible-window mirror, §10).
**Unit:** rad of proxy = `|slipAngle| · |GAIN − 1|`.

```ts
it('AC-087: the drift slide prominence reaches the named floor at a sustained full-steer turn', () => {
  // The floor is a NAMED EXPORTED CONSTANT (the AC-087 wording).
  expect(NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD).toBe(0.2);

  const run = runDriftScenario(SCN_DRIFT_01);

  // The MEASURED proxy (the deterministic twin of the browser proxy).
  expect(run.maxProxyRad).toBeGreaterThanOrEqual(
    NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD,
  );
});

it('AC-087: the slide is SUSTAINED, not a spike (the AC "magnitude AND sustain" clause)', () => {
  const run = runDriftScenario(SCN_DRIFT_01);

  // The proxy must stay at or above the floor for a sustained run inside the
  // 180-tick build phase, not for a single tick.
  expect(run.longestRunAboveFloorTicks).toBeGreaterThanOrEqual(120); // 2.0 s
});

it('AC-087: the floor is a proxy floor, not a slip floor', () => {
  // The proxy identity, pinned so the two quantities can never be confused:
  //   proxy = |slipAngle| * |GAIN - 1|
  expect(computeDriftSlideProxyRad(0.098777101149216228)).toBeCloseTo(
    0.049388550574608114,
    12,
  );
  expect(computeDriftSlideProxyRad(0)).toBe(0);
});
```

**Status: RED.** Today the model's hold slip at 25 m/s is `0.0785 rad` → the proxy is
`0.0392 rad` → the floor `0.20` is **5.1× above the current behaviour**, so the
assertion is a real change, not a no-op. The named constant does not exist yet.

**Floor value rationale (this slice's decision, §8 D-2):**

| candidate | vs. old browser baseline `0.0494` | vs. the new measured hold `0.2256` | verdict |
|---|---|---|---|
| `0.10` | 2.0× | 44 % below the hold | too loose — the old model would fail it, but barely |
| **`0.20`** | **4.05×** | **11.4 % below the hold** | **chosen** — a real 4× step up, with tune headroom |
| `0.2256` | 4.6× | 0 % | no headroom; the tune slice could not move anything |
| `0.25` | 5.1× | 11 % ABOVE the hold | unreachable — the floor could never be met |

**Coupled constraint the floor implies:** the proxy saturates at
`MAX_SLIP · GAIN · |1 − GAIN|` → with `GAIN = 0.5` the floor `0.20` requires
`MAX_SLIP_ANGLE_RAD ≥ 0.40 rad`. The merged `0.45` gives a `0.2256` ceiling — an 11 %
margin. This coupling must be stated in the plan so the tune slice cannot raise the gain
toward `1.0` (see 7g) without breaking the floor.

---

### 7d — AC-088 re-scoped to the fragment-stage `uTireGradient*` uniform seam

> Merge text: *"(d) AC-088 re-scoped to the fragment-stage `uTireGradient*` uniform seam
> (the VBO vertex-colour measurement is blind BY CONSTRUCTION: upperAvg == lowerAvg ==
> 0.28565)"*

**Home:** `examples/neatRacing/engine/lighting.test.ts` — the **OBS-20 describe**
(`'fixed-space tire lighting (OBS-20)'`, L1024) — **not** `car.pass.test.ts`.
**Rationale for the move:** the OBS-20 describe already owns the `uTireGradient*` seam and
holds the richer harness (`createCarMockWebGlHarness` with `uniform1fFeeds`); `car.pass.test.ts`
uses the lighter `createMockGl`. The AC-088 row's file list (`car.pass.ts + car.pass.test.ts
+ the fixture`) must therefore be **amended** by `01-planning` (§11 O-4).

**Unit:** dimensionless factor spread (`bright − dark`).

```ts
it('AC-088: the tire-gradient magnitude is non-degenerate at the uniform seam', () => {
  // The NAMED magnitude floor (the AC-088 lever, exported from car.pass.ts).
  expect(NEATRACING_TIRE_GRADIENT_MIN_SPREAD).toBe(0.5);

  const harness = createCarMockWebGlHarness();
  const pass = createCarPass(harness.gl, { body: CAR_FIXTURE_BIN, wheel: CAR_FIXTURE_BIN });
  pass.draw(makeDrawFrame([
    { worldX: 0, worldZ: 0, yawRad: 0, teamColorId: 1, wheelSpinRad: 0 },
  ]));

  // The SEAM the fragment stage actually reads (car.pass.ts:324-327).
  const brightFeeds = harness.uniform1fFeeds.filter(
    (feed) => feed.name === 'uTireGradientBrightFactor',
  );
  const darkFeeds = harness.uniform1fFeeds.filter(
    (feed) => feed.name === 'uTireGradientDarkFactor',
  );
  expect(brightFeeds.length).toBeGreaterThan(0);
  expect(darkFeeds.length).toBe(brightFeeds.length);

  // The fed values ARE the named exported levers (no hidden literal in the feed).
  expect(brightFeeds.every((f) => f.value === NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR)).toBe(true);
  expect(darkFeeds.every((f) => f.value === NEATRACING_TIRE_GRADIENT_DARK_FACTOR)).toBe(true);

  // The MAGNITUDE: the fragment stage can only differentiate the tire if the
  // two ends of the gradient differ.
  expect(
    NEATRACING_TIRE_GRADIENT_BRIGHT_FACTOR - NEATRACING_TIRE_GRADIENT_DARK_FACTOR,
  ).toBeGreaterThanOrEqual(NEATRACING_TIRE_GRADIENT_MIN_SPREAD);

  // The radius and the threshold are part of the same seam.
  const radiusFeeds = harness.uniform1fFeeds.filter(
    (feed) => feed.name === 'uTireGradientRadius',
  );
  expect(radiusFeeds.every((f) => f.value > 0)).toBe(true);
  const thresholdFeeds = harness.uniform1fFeeds.filter(
    (feed) => feed.name === 'uTireGradientThreshold',
  );
  expect(thresholdFeeds.every((f) => f.value > 0 && f.value < 1)).toBe(true);
});
```

**Status: RED** — `NEATRACING_TIRE_GRADIENT_MIN_SPREAD` does not exist today → the import
fails. The **value** leg (1.7 − 0.35 = 1.35 ≥ 0.5) is a GUARD that passes today; the
**export** leg is the red.

**Honest limit of this contract (must be recorded in the plan).** The AC-088 *rendered*
outcome ("the sliding tire is visually differentiated") is **not provable from JavaScript**:
the gradient is computed in the fragment shader, and the VBO vertex-colour probe is blind
by construction (`upperAvg == lowerAvg == 0.28565` — the wheel's vertex colours carry no
vertical gradient, because the gradient is a *fragment-stage* effect). This contract
therefore pins the **inputs** of the gradient and its **non-degeneracy**; the visible
gradient itself is carried by **AC-090 (the user visual gate)**. Writing a JS assertion
that pretends to measure the rendered gradient would be a false contract.

---

### 7e — the envelope re-derived from the measured peak, or the pose held inside 0.2250

> Merge text: *"(e) envelope re-derived from the measured peak (0.2295 rad) or the pose held
> inside 0.2250 (the recommended variant does the latter)"*

**Decision (this slice, §8 D-10): the RECOMMENDED variant** — the pose is held **inside**
the natural limit `MAX_SLIP · GAIN = 0.2250 rad`; the limit does **not** move.

**Home:** `physics.step.service.test.ts` (the envelope pin) + `engine/passes/car.pass.test.ts`
(the derived limit constant).

```ts
it('AC-087: the visible yaw stays inside the declared envelope (merge 7e)', () => {
  // The envelope is DERIVED from the constants - it is not a magic number.
  expect(
    NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
  ).toBeCloseTo(0.225, 10);   // 0.45 * 0.5

  const run = runDriftScenario(SCN_DRIFT_01);

  // The measured peak is 0.2246 rad (12.87 deg) - 0.19 % inside the limit.
  expect(run.maxAbsPoseRad).toBeLessThanOrEqual(0.225);
  expect(run.maxAbsPoseRad).toBeGreaterThan(0.22);   // the pose really does reach the envelope
});
```

**The re-pinned green pin (replaces `physics.step.service.test.ts:156-163`):**

```ts
// BEFORE (green today, must be re-pinned in the SAME red step):
//   expect(Math.abs(state.driftYawOffset)).toBeLessThan(Math.PI / 2);
// AFTER:
expect(Math.abs(state.driftYawOffset)).toBeLessThanOrEqual(
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
);
```

**Status:** the derivation assertion is **RED** (today `MAX_SLIP = π/2` → the product is
`0.7854`); the envelope assertion is a **RE-PIN + GUARD** (today's offset is tiny, so the
`≤ 0.225` form passes today and must keep passing after the re-baseline).

**Fallback rule (must be written into the plan).** If the tune slice cannot keep the peak
inside `0.2250`, the limit moves to the **measured peak** and **both** pins move in the
same step (`physics.step.service.test.ts` **and** `car.pass.test.ts`'s
`DRIFT_VISUAL_YAW_LIMIT_RAD`, which is derived from the same constants) — the two must
never diverge, because `car.pass.test.ts` uses the limit as the *input range* of its
monotonic-mapping contract.

**Why ω = 18 is load-bearing.** D6.8: ω = 18 is the **largest** ω that keeps the pose
inside the declared envelope at ζ = 0.35. A second, independent reason to keep it — the
tune slice must not raise ω above 18 without re-deriving the envelope.

---

### 7f — the 3-frame drift-gate regression guard

> Merge text: *"(f) the 3-frame drift-gate regression guard"*

**Home:** the browser fixture (the gate lives there) + a deterministic jest twin.
**Unit:** frames (browser) / ticks (jest).

The fixture already owns the gate — `DRIFT_PROXY_MIN_RAD = 0.01` (L128),
`DRIFT_PROXY_MIN_FRAMES = 3` (L132) and the consecutive-run accounting
(`driftProxyRunMaxFrames`, L815-880). What is **missing** is a *named* check: the run
length is currently folded into `tireYawDriftEngaged`, so a regression that shortened the
gate to a single frame would be invisible.

```ts
// JEST TWIN (deterministic, in drift.model.test.ts)
it('AC-087: the drift engagement gate survives three consecutive frames (merge 7f)', () => {
  const run = runDriftScenario(SCN_DRIFT_01);

  // The engagement gate threshold is UNCHANGED (0.01 rad of proxy).
  expect(NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD).toBe(0.01);
  expect(NEATRACING_DRIFT_PROXY_MIN_FRAMES).toBe(3);

  // The gate counts CONSECUTIVE frames, not total frames: a single-frame
  // spike can never satisfy it.
  expect(run.longestRunAboveEngagementGateTicks).toBeGreaterThanOrEqual(
    NEATRACING_DRIFT_PROXY_MIN_FRAMES,
  );
});
```

```js
// BROWSER FIXTURE (ES5 style, the fixture's own idiom)
var DRIFT_PROXY_MIN_RAD = 0.01;          // RETAINED VERBATIM
var DRIFT_PROXY_MIN_FRAMES = 3;          // RETAINED VERBATIM

// NEW named check (added to result.checks):
driftGateSurvivesThreeConsecutiveFrames:
  driftProxyRunMaxFrames >= DRIFT_PROXY_MIN_FRAMES,
```

**Status: GUARD** (the gate passes today — the current model *is* engaged for many
frames). The **red artifact** is the new *named* check (it does not exist → the fixture's
check list changes 31 → 32, which the plan's acceptance text must record) and the jest
twin's two new named exports.

**Disambiguation (must be in the plan):** the **engagement** gate (`0.01` rad, 3 frames)
and the **prominence** floor (`NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD = 0.20` rad) are
different contracts and must not share a name or a threshold.

---

### 7g — the `GAIN_eff = +1.0` forbidden-null contract is SUPERSEDED

> Merge text: *"(g) the GAIN_eff=+1.0 forbidden-null contract is SUPERSEDED by the full
> model (gain stays 0.5; the V-null knowledge is retained for the record)"*

**Home:** `drift.model.test.ts`.

**DO NOT AUTHOR (an explicit prohibition, not a contract):** no test may pin, require or
simulate `GAIN_eff ≈ 1.0`, and no test may assert that the drift is *absent* at a gain of
1.0. The V-null analysis (`proxy = |slip| · |GAIN − 1|` → the proxy has a **null** at
`GAIN = 1.0`) stays in `plans/neatracing-voxel-racer.drift-feel.research.md` **as a
record**, not as a test.

**The replacement contract (the anti-null guard):**

```ts
it('AC-087: the visual yaw gain stays on the monotone branch (merge 7g)', () => {
  // The gain is UNCHANGED at 0.5 - the body yaw shape is retained
  // (AC-087: "NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD keeps owning the body yaw").
  expect(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD).toBe(0.5);

  // The anti-null guard: the proxy scales with |GAIN - 1|, which has a NULL
  // at GAIN = 1.0. The gain must never approach that null.
  expect(Math.abs(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD - 1)).toBeGreaterThanOrEqual(0.5);

  // And the sign: the visible yaw must NOT mirror the slide. Under the
  // production convention (LEFT = negative slip) the pose target is
  // +committedSlip * GAIN, so the pose and the slip share a sign.
  const run = runDriftScenario(SCN_DRIFT_01);
  expect(Math.sign(run.poseAtHold)).toBe(Math.sign(run.slipAtHold));
  expect(run.poseAtHold).toBeLessThan(0);   // a LEFT turn yaws the body LEFT
});
```

**Status: GUARD** (all three legs pass today; they exist to stop a future "tune the gain
up" from silently deleting the slide — the exact failure the V-null analysis predicts).

**The 7g + 7c coupling:** raising the gain toward 1.0 would *shrink* the proxy and break
the 7c floor (`0.20` rad). The two contracts together make the gain's monotone branch
explicit and machine-checked.

---

## 5. The physical umbrella contracts (P0) — the merge's physical half

> Merge item 1: *"the arcade-refs 7-stage committed-drift model ADOPTED (countersteer
> currently MIRRORS the slide; visual-only transient gain REJECTED as primary fix)"*

Merge items 7a–7g are the **visual/contract** list. Without the following, the merged
design would deliver a *bigger* slide that still **mirrors** on the countersteer — i.e. the
user's actual complaint ("bounce-back must be sharper") would be untouched. These are
therefore part of the same RED set.

### P0-1 — the countersteer UNWINDS the slide; it never MIRRORS it

**Home:** `physics.step.service.test.ts` (replaces the AC-047B pin) + `drift.model.test.ts`.
**Unit:** rad.

```ts
it('AC-047B: the countersteer unwinds the slide to the hard zero and never mirrors it', () => {
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  };

  // Build the drift.
  const drifted = runStepsFrom(start, 180, { throttle: 1, brake: 0, steer: 1 });
  expect(Math.abs(drifted.slipAngle)).toBeGreaterThan(0.4);   // the slide is real
  expect(drifted.slipAngle).toBeLessThan(0);                  // LEFT = negative slip

  // Full countersteer for 20 ticks (333 ms).
  const recovered = runStepsFrom(drifted, 20, { throttle: 1, brake: 0, steer: -1 });

  // THE HARD ZERO: the slide is snapped to exactly zero, not parked in a
  // mirror-image equilibrium of the same size.
  expect(recovered.slipAngle).toBe(0);
  expect(recovered.lateralVelocity).toBe(0);

  // THE ANTI-MIRROR CLAUSE: at no point during the unwind did the slip cross
  // to the opposite sign. (Today it reaches +28.1 deg - a full inversion.)
  expect(runDriftScenario(SCN_DRIFT_01).maxSlipAfterCounterOnset).toBeLessThanOrEqual(0);
});
```

**Status: RED.** Today, 20 ticks of full countersteer at 89 m/s leaves the car in a
**mirrored** equilibrium (`+4.48°` at 25 m/s, `+52.15°` at 89 m/s per E2 scenario D/E).
This is the C5 defect the merge's physical half exists to fix, and it is the one
contract that makes "bounce-back must be sharper" true.

**Re-pinned green pin:** the existing AC-047B assertion
(`expect(Math.abs(state.lateralVelocity)).toBeLessThan(5)` after 15 ticks) is **too weak
and mis-timed** for the new model (the hard zero lands at ~18 ticks) → it becomes the
`toBe(0)` form above.

### P0-2 — the slide is BOUNDED (the C4 fix)

```ts
it('AC-079: the sustained high-speed slide stays inside the declared envelope', () => {
  // The re-pinned bound: MAX_SLIP_ANGLE_RAD is 0.45 rad (25.8 deg), not pi/2.
  // The model's committed angle is CLAMPED, so the slide is a budget, not a
  // force balance. Today the slide reaches -65.8 deg (1.148 rad) and keeps
  // growing - this assertion is the red.
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  };
  const state = runStepsFrom(start, 1200);

  expect(Math.abs(state.slipAngle)).toBeLessThanOrEqual(
    NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
  );
  expect(Math.abs(state.slipAngle)).toBeGreaterThan(0.2);
});
```

**Status: RED after the re-pin.** The *existing* text
(`toBeLessThan(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD)`) is **green today only because the
constant is `π/2`** — re-pinning the constant to `0.45` turns this existing green test
**red**. That is the re-baseline mechanism working as intended, and it is why §6 forbids
deferring the constant re-pin to the green phase.

### P0-3 — the collision bounce resets the pose state (merge risk item 8)

**Home:** `examples/neatRacing/physics/collision.utils.test.ts` (the file exists).
**Unit:** rad.

```ts
it('AC-079: a bounced car carries no drift pose', () => {
  // The bounce recomposes the car around the reflected world velocity and
  // zeroes the slip channels - the VISUAL pose must be zeroed with them, or
  // the car keeps the old slide's yaw while driving straight.
  const sliding: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    x: 0,
    z: 0,
    heading: 0,
    speed: 40,
    slipAngle: -0.3,
    lateralVelocity: -12,
    driftYawOffset: -0.15,
    driftPose: { pose: -0.15, poseRate: 3.2, articulation: -1 },
  };

  const bounced = resolveTrackBoundaryCar(sliding, /* the fixture's resolver args */);

  expect(bounced.driftYawOffset).toBe(0);
  expect(bounced.driftPose.pose).toBe(0);
  expect(bounced.driftPose.poseRate).toBe(0);
  expect(bounced.driftPose.articulation).toBe(0);
});
```

**Status: RED.** `recomposeCarWithVelocity` (collision.utils.ts:402-420) zeroes
`speed` / `lateralVelocity` / `heading` / `slipAngle` and **never touches
`driftYawOffset`** — and today that is invisible only because the value happens to be 0
in the existing fixture. The RED uses a **non-zero** pose, which is what makes it fail
today.

**Fixture note:** the pose state is a *new* field on `PhysicsCarState`; the bounce
contract must name the field the implementer chooses (this slice's proposal: a nested
`driftPose: DriftPoseSpringState` on `PhysicsCarState`, so the three scalars travel
together and cannot be half-reset).

### P0-4 (OPTIONAL — needs a 01-planning decision) — the entry attack

D3 measured the new entry at 63 % of the hold slide in **0.10 s** (today: 0.25 s at
25 m/s, 0.5 s at 89 m/s) and 95 % in **0.20 s** (today 0.75 s). This is the "crisper
recovery" half of the user's un-named direction. It is **not** in merge items 7a–7g, so it
is flagged OPTIONAL:

```ts
it('AC-087: the slide reaches 63% of its hold within 150 ms', () => {
  const run = runDriftScenario(SCN_DRIFT_01);
  expect(run.ticksTo63PctOfHold).toBeLessThanOrEqual(9);   // 150 ms
});
```

**Recommendation:** include it — it is the cheapest way to make the "crisper" half of the
direction *measurable* rather than a matter of opinion. But it is the RED slice's call to
mark it optional, and `01-planning` must either adopt it or record the omission.

---

## 6. Re-baselined green pins — MUST move in the SAME red step (no deferred cleanup)

> Merge item 5: *"MUST-NOT-MOVE PIN: LIFTED for this slice. AC-079 and AC-089 are
> RE-BASELINED, not preserved."*

Each row is a **currently green** assertion that the merged constant set breaks. Leaving
any of them in place produces a green-phase edit (a contract moved after the fact), which
this plan forbids.

| # | Home | Current text | Re-baselined text | Why it breaks |
|---|---|---|---|---|
| **G-1** | `drift.model.test.ts` (constants describe) | `expect(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD).toBe(Math.PI / 2)` | `expect(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD).toBe(0.45)` | the constant becomes `0.45` |
| **G-2** | `drift.model.test.ts` (AC-079, traction-vs-speed) | `expect(highSpeed).toBeLessThan(lowSpeed)` (the grip is speed-dependent) | `expect(highSpeed).toBe(lowSpeed)` + a NEW `it`: *"the grip budget is a function of the slip only; the speed demand is charged by the speed scrub"* | the new grip budget `1/(1 + RATE·max(0,|slip|−THRESHOLD))` is **speed-independent** |
| **G-3** | `drift.model.test.ts` (AC-079, 30 %-cap) | `computeTractionFactor(0.8, 1) < 0.7` | re-base the argument to `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` (0.45): `1/(1+2·0.25) = 0.667 < 0.7` | `0.8 rad` is no longer a legal slip (the max is 0.45); keep the `< 0.7` intent (5 % margin — flag it) |
| **G-4** | `drift.model.test.ts` (countersteer helper) | `computeCountersteerDelta` "only when steer matches the slip sign" | re-authored: *"the countersteer authority is applied AGAINST the slide, gated on the LATCHED turn sign, and only above `COUNTERSTEER_MIN_SLIP_RAD`"* + the new signature | the gate inverts (the old gate is the C5 defect) |
| **G-5** | `drift.model.test.ts` (AC-079, longitudinal traction loss ×2) | the two `computeLongitudinalTractionLoss` `it`s | **GONE pins** (§7) + a NEW speed-scrub `it` | the helper is retired → `SPEED_SCRUB_COEFF_PER_S` |
| **G-6** | `drift.model.test.ts` (outward push) | the `computeOutwardPush` `it` | **GONE pin** + a NEW chase `it` (the physical chase replaces the outward push) | the 7-stage model has no outward push |
| **G-7** | `drift.model.test.ts` (the live dead-twin `it`) | `expect(computeDriftYawOffset(10, 10)).toBeCloseTo((π/4)·GAIN, 10)` | **GONE pin** (`'computeDriftYawOffset' in driftModel === false`) | the pose spring is the single visible-yaw source (merge risk item 8) |
| **G-8** | `physics.step.service.test.ts` (envelope pin, L156-163) | `expect(Math.abs(state.driftYawOffset)).toBeLessThan(Math.PI / 2)` | `...toBeLessThanOrEqual(MAX_SLIP·GAIN)` (0.2250) | 7e — the envelope is re-derived |
| **G-9** | `physics.step.service.test.ts` (slip envelope) | `expect(Math.abs(state.slipAngle)).toBeLessThan(Math.PI / 2)` | `...toBeLessThanOrEqual(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD)` | the same constant re-pin |
| **G-10** | `physics.step.service.test.ts` (AC-079 bounded slide) | `...toBeLessThan(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD)` (green only because the constant is `π/2`) | same text, **re-pinned constant** → this test becomes **RED** | P0-2 — this is the intended red |
| **G-11** | `physics.step.service.test.ts` (AC-079 speed scrub) | `expect(state.speed).toBeLessThan(MAX − 20)` | **UNCHANGED — it survives.** The new scrub is `0.25·0.453·89 = 10.1 m/s²` against the engine's `17.8 m/s²` → the speed still lands far below the ceiling | it does **not** break — do not "re-baseline" a pin that holds (the merge's 290 km/h ceiling risk was measured, not assumed) |
| **G-12** | `physics.step.service.test.ts` (AC-079 straight line) | `expect(state.slipAngle).toBe(0)` and `expect(state.speed).toBe(MAX)` | **UNCHANGED** (the hard zero keeps the straight line exact) | it holds |
| **G-13** | `physics.step.service.test.ts` (AC-079 low speed) | `expect(state.lateralVelocity).toBe(0)` / `expect(state.slipAngle).toBe(0)` at 10 m/s | **UNCHANGED** (below `MIN_SPEED_MPS` the drift is off) | it holds |
| **G-14** | `physics.step.service.test.ts` (AC-047B) | `expect(Math.abs(state.lateralVelocity)).toBeLessThan(5)` after 15 countersteer ticks | `expect(state.lateralVelocity).toBe(0)` after **20** ticks (the hard zero) | P0-1 — the new model snaps to exactly 0 at ~18 ticks; the 15-tick window is too early |
| **G-15** | `car.pass.test.ts` (the drift-yaw describe) | the fixtures use `driftYawOffsetRad: 0.35` and `0.5` | re-base to values **inside** the new envelope: `0.12` and `0.2` (the limit is now 0.2250, and 0.35/0.5 are **out of envelope**) | the fixtures must not use out-of-envelope offsets — the derived `DRIFT_VISUAL_YAW_LIMIT_RAD` changes value, so the fixtures must change with it |
| **G-16** | `drift.model.test.ts` (AC-047A) | `expect(Math.abs(state.slipAngle)).toBeGreaterThan(0.2)` | **UNCHANGED** (the new hold is 0.45) | it holds |
| **G-17** | `drift.model.test.ts` (AC-047D byte-identity) | `serializeDriftState(state) === serializeDriftState(createZeroDriftState())` | **UNCHANGED** — the new state's extra fields (`committedSlip`, `turnSign`) are all exactly 0 on a straight line at 10 m/s | it holds, *provided* the pose state is NOT on `DriftModelState` (merge item 2) |

**Count: 17 re-pinned / re-examined green pins, of which 10 actually change.** All 10 must
move in the SAME red step as the new contracts.

---

## 7. GONE pins (the retired symbols must not come back)

House idiom (from the existing AC-079 "the hard grip floor is gone" `it`):

```ts
it('AC-079: the retired drift-rate constants are gone', () => {
  expect('NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S' in driftModel).toBe(false);
  expect('NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S' in driftModel).toBe(false);
  expect('NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S' in driftModel).toBe(false);
  expect('computeDriftYawOffset' in driftModel).toBe(false);            // G-7
  expect('computeOutwardPush' in driftModel).toBe(false);                // G-6
  expect('computeLongitudinalTractionLoss' in driftModel).toBe(false);   // G-5
});
```

**The merge retired exactly three CONSTANTS** (LATERAL_GRIP_RATE, COUNTERSTEER_RECOVERY,
LONGITUDINAL_TRACTION_LOSS). The **helpers** that the 7-stage model replaces
(`computeOutwardPush`, `computeLongitudinalTractionLoss`, `computeDriftYawOffset`) are
**not** in the merge's retire list, but each of them has a live green `it` and each is
**incompatible** with the new model:

| helper | new-model fate | recommendation | confidence |
|---|---|---|---|
| `computeOutwardPush` | the 7-stage model has **no outward push** (stage 3 is the chase) | **RETIRE** (GONE) | 0.85 |
| `computeLongitudinalTractionLoss` | replaced by the saturating **speed scrub** | **RETIRE** (GONE) | 0.88 |
| `computeDriftYawOffset` | the visible pose is now the **spring** state, not `slip·GAIN` | **RETIRE** (GONE) — the merge's risk item 8 offers "wired or deleted"; deleting is the only option that leaves ONE visible-yaw formula | 0.90 |
| `computeTractionFactor` | survives as the **grip budget** (speed-independent) | **KEEP**, re-specified: `1/(1 + RATE·max(0,|slip|−THRESHOLD))` | 0.90 |
| `computeCountersteerDelta` | survives with a **new gate** (the latched turn sign) | **KEEP**, re-specified | 0.85 |
| `computeSlipAngle` | unchanged | **KEEP** | 0.95 |

**Open item O-7 (§11):** `01-planning` must ratify this helper retire list in the plan
before the RED phase authors the GONE pins, because the merge text only retired the
constants.

---

## 8. Decision ledger — the values THIS slice owns

The merge delegated exactly two decisions to the RED slice (*"RED decides in-band"*, and
the AC-087 floor *"whose VALUE is decided by the RED slice"*). This slice also has to
own the measurement conventions, otherwise the numbers are not reproducible.

| # | Decision | Value | Rationale | Certainty |
|---|---|---|---|---|
| **D-1** | the in-band articulation slew | **`NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S = 20`** (band 15–30) | the band's pop/step table: 15 → 2.10°/1.75°, **20 → 2.22°/2.13°**, 30 → 2.30°/2.89°. 20 is the merge's recommendation and the only value whose measured step (2.13°) matches the merge's stated 7a target exactly | 0.92 |
| **D-2** | the AC-087 deterministic proxy floor | **`NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD = 0.20` rad** | 4.05× the old browser baseline (0.0494), 11.4 % below the new measured hold (0.2256) → a real step up with tune headroom (§7c table) | 0.88 |
| **D-3** | the deterministic proxy helper | **`computeDriftSlideProxyRad(slipAngleRad): number` = `|slip|·|GAIN−1|`**, exported from `drift.model.ts` | one definition of the proxy for both lanes; the browser lane mirrors the formula in the fixture because the fixture cannot import a module | 0.90 |
| **D-4** | the browser prominence mirror | **`0.10` rad** (fixture-local `var`, ES5 style) | the browser proxy is render-rate dependent with a 7× observed spread → the visible lane carries HALF the strict floor so a render artifact cannot fail the AC; the deterministic lane carries the strict floor | 0.85 |
| **D-5** | the AC-088 magnitude floor | **`NEATRACING_TIRE_GRADIENT_MIN_SPREAD = 0.5`** (exported from `car.pass.ts`) | the current spread is `1.7 − 0.35 = 1.35`; a 0.5 floor keeps the factor ratio ≥ 2× and is far below the current value, so the tune slice can move either end | 0.84 |
| **D-6** | the settle measurement | `|pose − poseFinal| < 0.5°`, counted in **ticks**, band **[30, 34]** (500–566.7 ms), measured 33 (550 ms) | the merge's 550 ms is a simulation number; a tick band is exact, reproducible, and survives the ±1-tick convention difference | 0.86 |
| **D-7** | the pop measurement | `max |pose(t) − poseFinal|` over the **Phase B** ticks, floor **2.0°**, measured 2.22° | the merge's "~2.0 deg" is read as a **stated minimum** (D6.7a: "a named corpus contract with a stated minimum") | 0.90 |
| **D-8** | the per-tick step measurement | `max |poseTarget(t) − poseTarget(t−1)|` over ALL ticks, ceiling **2.20°**, measured 2.13°; the articulation-only sub-bound is exactly `ART·SLEW·dt = 1.52778°` | the 2.13° target has no tick-grid slack, so the ceiling carries 3.3 %; the exact analytic sub-bound is the speed-independent half of the claim | 0.88 |
| **D-9** | the articulation observation convention | the state carries the **POST-tick** articulation; `poseTarget = slip·GAIN + ART·articulation` | without this, the observation is off-by-one ambiguous and the step measurement is not reproducible | 0.86 |
| **D-10** | the envelope variant | the **pose held inside 0.2250** (the merge's recommended variant); the limit does **not** move | keeps ONE derivable limit (`MAX_SLIP·GAIN`), and ω = 18 is already the largest ω that satisfies it | 0.90 |
| **D-11** | the feel DIRECTION (the AC-087 open decision) | **a BIGGER SLIDE** (the physical model), recorded in the plan **BEFORE any production edit** | the merge adopted the physical half; the AC-087 wording requires the direction to be recorded before the production edit; "a crisper recovery" is delivered *by* the same model (P0-1) and is additionally measured by P0-4 (optional) | 0.92 |
| **D-12** | `computeDriftYawOffset` | **DELETE** (GONE pin) | the merge's risk item 8 offers "wired or deleted"; the pose spring is the single visible-yaw source, and keeping a `slip·GAIN` helper next to it is the "two divergent visual-yaw formulas" the merge forbids | 0.88 |

**Recorded for the plan (not a test):** the asymmetric-damping idea is **REFUTED** (it
collapses the far-side pop 2.61° → 0.00°) — it must be recorded so it is not re-derived
(D6.10.1).

---

## 9. The file set — the RED homes and the widened packet

**The current `P1-drift-feel-red` packet's `files_to_change` is STALE.** It lists
`drift.model.test.ts`, `car.pass.test.ts`, `p1-turn-drift-green-smoke.html` and the plan.
The merge's own file set (§9 of the MERGE block) and the contracts above need **five**
test/fixture homes.

| # | File | What the RED phase changes there | In the current packet? |
|---|---|---|---|
| 1 | `examples/neatRacing/physics/drift.model.test.ts` | the constants describe (G-1), the GONE pins, 7a, 7b, 7c, 7f-twin, 7g, P0-1/2 (unit) | ✅ |
| 2 | `examples/neatRacing/physics/physics.step.service.test.ts` | 7e, G-8/9/10/14, P0-1 (integration), the 7b integration form | ❌ **MISSING** |
| 3 | `examples/neatRacing/engine/lighting.test.ts` | 7d (the OBS-20 describe owns the uniform seam) | ❌ **MISSING** |
| 4 | `examples/neatRacing/engine/passes/car.pass.test.ts` | G-15 (the out-of-envelope fixtures) | ✅ |
| 5 | `examples/neatRacing/physics/collision.utils.test.ts` | P0-3 (the bounce pose reset) | ❌ **MISSING** |
| 6 | `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html` | 7d's uniform capture + the named check, 7f's named check, 7c's loose prominence mirror | ✅ |
| 7 | `plans/neatracing-voxel-racer.plans.md` | the direction decision (D-11), the AC row amendments, the slice packet | ✅ |

**No new test file may be created** — all four jest homes and the fixture already exist.
`engine/passes/glass.pass.test.ts` does **not** exist (see O-8).

**Slice-atomicity flag (O-3).** The slice-validator enforces *atomic intent, ≤ 3 files*.
This RED set touches **5 test/fixture files + the plan = 6**. `01-planning` must either
(a) split `P1-drift-feel-red` into 2–3 sub-slices (e.g. *red-model* → *red-pose* →
*red-observable*), or (b) record an explicit, reasoned exception. This is a **gate**
obstruction, not a style note.

### 9.1 — the J1–J27 → RED-home mapping (added by the RED slice, `P1-drift-feel-red`)

The D1.2 ledger (`14N-rebaseline-risk.md` §D1.2, L184-210) owns 27 pins. **This table
records where each pin is now WRITTEN** — the ledger's verdict column stays the authority,
this table only says which RED home carries the contract and under which contract id. The
**Step 14N packet overrides** two D1.2 verdicts: **J23 and J25 are OVERRIDDEN to DELETE**
(their exports are in the RATIFIED retire list, so a PRESERVE contract would be
un-greenable) — the packet wins over the ledger on those two rows.

| J | D1.2 verdict (the authority) | the packet's override | RED home | the contract id(s) that now carry it |
|---|---|---|---|---|
| J1 | R → 0.45 | — | 1 | `RED-G1` |
| J2 | P (the `atan2` construction survives) | — | 1 | `RED-J24-PRESERVE` |
| J3 | R → `toBeLessThanOrEqual(0.45)` | — | 1, 2 | `RED-G1` · `RED-G9-SLIP-ENVELOPE` · `RED-G10-BOUNDED-SLIDE` |
| J4 | P (verify) | — | 1, 2 | `RED-G16-AC047A` · `RED-P0-1-BUILD-SLIDE` |
| J5 | R (the π/2 literal → the import) | — | 2 | `RED-G9-SLIP-ENVELOPE` |
| J6 | P (the pose initialises to 0) | — | 2 | `RED-D1-SPAWN-ZEROS` |
| J7 | R (the DERIVED `0.2250` → `0.3050`) | the packet's derived envelope **0.3050 rad** wins over the artifact's 0.2250 | 2, 5 | `RED-G8-ENVELOPE` · `RED-POSE-FINITE` · `RED-POSE-NONZERO` · `RED-G15-POSE-ENVELOPE` |
| J8 | R → `<= 0.45` | — | 1, 2 | `RED-P0-2-UNIT` · `RED-G10-BOUNDED-SLIDE` · `RED-P0-2-BOUNDED-SLIDE` |
| J9 | P (the hard zero) | — | 1, 2 | `RED-7B-UNIT` · `RED-7B-INTEGRATION` |
| J10 | P (highest risk — the stage-5 pin) | — | 1, 2 | `RED-7B-UNIT` · `RED-PRESERVE-STRAIGHT-TOP-SPEED` · `RED-7B-CHANNEL-ZEROS` |
| J11 | R (a MODEL change, not a test change) | — | 1, 2 | `RED-D1-MODEL-GAP-MIN-SPEED` · `RED-D1-MIN-SPEED-GATE` · `RED-D1-MIN-SPEED-GATE-UNCOMMITTED` |
| J12 | P (the model stays pure) | — | 1, 2 | `RED-J12-DETERMINISM` · `RED-PRESERVE-DETERMINISM` |
| J13 | P | — | 2 | `RED-PRESERVE-DETERMINISM` |
| J14 | P (conditional — the byte identity) | — | 1, 2 | `RED-G17-AC047D` (both homes) |
| J15 | RH (re-anchored on the scrub zero identity) | — | 1 | `RED-G5-SPEED-SCRUB` |
| J16 | RH (onto the scrub term) | — | 1 | `RED-G5-SPEED-SCRUB-2` · `RED-G2` |
| J17 | RH (the property survives, the expression changes) | — | 1 | `RED-G3` |
| J18 | P + extend (the forbidden set grows) | — | 1 | `RED-J18-PRESERVE` · `RED-GONE-CONSTANTS` |
| J19 | X (deleted and replaced) | — | 1 | `RED-G4` · `RED-G4-HELPER-RENAME` · `RED-GONE-HELPERS` |
| J20 | X → RH (the dead twin) | — | 1 | `RED-GONE-HELPERS-2` · `RED-GONE-HELPERS-3` |
| J21 | R (the threshold 0.25 → 0.20) | — | 1 | `RED-G2` · `RED-G1-GRIP-THRESHOLD` |
| J22 | P + extend (the new fields are 0) | — | 1, 2 | `RED-GONE-HELPERS-3` · `RED-7B-CHANNEL-ZEROS` · `RED-7B-COMMITTED-ZEROS` |
| J23 | P | **OVERRIDDEN to DELETE** | 1 | `RED-G6-COMMITTED-CHASE` · `RED-GONE-HELPERS` |
| J24 | P | — | 1 | `RED-J24-PRESERVE` |
| J25 | P | **OVERRIDDEN to DELETE** | 2 | `RED-GONE-CENTRIFUGAL` |
| J26 | P (strengthened) | — | 1, 2 | `RED-P0-1-UNIT` · `RED-P0-1-UNIT-SLIDE` · `RED-P0-1-BUILD-SLIDE` · `RED-P0-1-HARD-ZERO` · `RED-P0-1-ANTI-MIRROR` |
| J27 | R (the export list changes) | — | 1 | the import-fails block itself · `RED-GONE-CONSTANTS` · `RED-GONE-HELPERS*` |

**The 7a-7g and P0-1..P0-4 contracts, mapped to their homes.**

| contract | home 1 (`drift.model.test.ts`) | home 2 (`physics.step.service.test.ts`) | home 3 (`collision.utils.test.ts`) | home 4 (`lighting.test.ts`) | home 5 (`car.pass.test.ts`) | home 6 (the fixture) |
|---|---|---|---|---|---|---|
| **7a** (pop / settle / step) | `RED-7A-POP` · `RED-7A-SETTLE` · `RED-7A-STEP` · `RED-7A-POINT` | — | — | — | — | `driftFeelFarSidePop` · `driftFeelSettles` |
| **7b** (the straight line is EXACT) | `RED-7B-UNIT` | `RED-7B-INTEGRATION` · `RED-7B-CHANNEL-ZEROS` · `RED-7B-COMMITTED-ZEROS` | — | — | — | `driftFeelZeroStraightLineYaw` |
| **7c** (the prominence floor) | `RED-7C-FLOOR` · `RED-7C-SUSTAIN` · `RED-7C-PROXY` | — | — | — | — | `driftFeelSlideProminence` |
| **7d** (the tire gradient) | — | — | — | `RED-7D-MIN-SPREAD-EXPORT` · `RED-7D-FACTOR-FEEDS-PRESENT` · `RED-7D-FACTOR-FEEDS-LEVERS` · `RED-7D-SPREAD-NON-DEGENERATE` · `RED-7D-RADIUS-POSITIVE` · `RED-7D-THRESHOLD-IN-UNIT-INTERVAL` · `RED-7D-NO-VERTEX-PROBE` | — | `tireGradientMagnitudeIsNonDegenerate` |
| **7e** (the envelope derivation) | — | `RED-7E-ENVELOPE-DERIVATION` · `RED-7E-ENVELOPE-DERIVATION-POSE` | — | — | `RED-G15-SLIP-ONLY-ENVELOPE` · `RED-G15-POSE-ENVELOPE` | `driftFeelPoseEnvelope` · `driftFeelNoStuckOffset` · `driftFeelPoseRateBounded` |
| **7f** (the 3-frame gate) | `RED-7F-ENGAGEMENT` | — | — | — | — | `driftGateSurvivesThreeConsecutiveFrames` |
| **7g** (the anti-null gain) | `RED-7G-ANTI-NULL` | — | — | — | — | `driftFeelModelIdentity` |
| **P0-1** (the countersteer catch) | `RED-P0-1-UNIT` · `RED-P0-1-UNIT-SLIDE` | `RED-P0-1-BUILD-SLIDE` · `RED-P0-1-HARD-ZERO` · `RED-P0-1-ANTI-MIRROR` | — | — | — | — |
| **P0-2** (the bounded slide) | `RED-P0-2-UNIT` | `RED-G10-BOUNDED-SLIDE` · `RED-P0-2-BOUNDED-SLIDE` | — | — | — | — |
| **P0-3** (the bounce reset) | — | — | `RED-P0-3-CAR-CAR-ZERO` · `RED-P0-3-CAR-CAR-COMMITTED-ZERO` · `RED-P0-3-CAR-CAR-POSE-RATE` · `RED-P0-3-BOUNDARY-ZERO` · `RED-P0-3-LATENT-SLIP-DEFECT` · `RED-P0-3-GUARD-LATERAL` · `RED-P0-3-UNCOMMITTED` | — | — | — |
| **P0-4** (the entry attack window) | `RED-P0-4` | — | — | — | — | — |

**The 4 G-pin rows that are NOT in the J-ledger** (they are the §6 G-ledger rows that are
not D1.2 J-numbers): G-1 → home 1 · G-2 → home 1 · G-3 → home 1 · G-4 → home 1 ·
G-5 → home 1 · G-6 → home 1 · G-7 → home 1 · G-8 → home 2 · G-9 → home 2 · G-10 → home 2 ·
G-14 → home 2 · G-15 → home 5 · G-16 → home 1 · G-17 → homes 1 and 2.
**G-11, G-12, G-13, G-16, G-17 are UNCHANGED** per §6; the two homes that carry a G-16/G-17
contract keep the pre-existing assertion bodies and only add the `// [RED-…]` tag, so a
PRESERVE row never has its body re-authored.

---

## 10. Browser-lane (fixture) contract changes

The fixture is ES5-style (`var` / `function`). The RED phase changes it in four places.

| # | Change | Detail | Status |
|---|---|---|---|
| B-1 | **capture the tire-gradient magnitude** | extend the `uniform1f` interception hook (L189-197) to record `uTireGradientBrightFactor` / `uTireGradientDarkFactor` / `uTireGradientRadius` / `uTireGradientThreshold` (today it records only `uSpacing` and `uApplyTireGradient`) | the narrowest possible fix; the seam is already there |
| B-2 | **the AC-088 named check** | `tireGradientMagnitudeIsNonDegenerate: (maxBright − minDark) >= NEATRACING_TIRE_GRADIENT_MIN_SPREAD && radius > 0 && threshold ∈ (0,1)` over the captured frames | the check does not exist today → the fixture's check count changes **31 → 32** (the plan's acceptance text must be updated) |
| B-3 | **the 7f named check** | `driftGateSurvivesThreeConsecutiveFrames: driftProxyRunMaxFrames >= DRIFT_PROXY_MIN_FRAMES` (the gate constants are retained verbatim) | the run length is currently folded into `tireYawDriftEngaged` → not independently visible |
| B-4 | **the 7c prominence mirror** | a local `var DRIFT_MIN_SLIDE_PROXY_RAD = 0.10;` (loose mirror of the exported `0.20`) + a named check `driftSlideProminenceReachesFloor` over the measured `maxDriftProxyRad` | the fixture has the engagement gate (`0.01`) but no prominence floor |

**Fixture invariants that must NOT move:** `DRIFT_PROXY_MIN_RAD = 0.01` (L128),
`DRIFT_PROXY_MIN_FRAMES = 3` (L132), `DRIFT_PROXY_WINDOW = 6` (L146),
`TIRE_YAW_FOLLOW_TOLERANCE_RAD = 0.002` (L101),
`TRACTION_LOSS_MAX_SPEED_KMH = 290` (L107).

---

## 11. Open items for `01-planning` (blockers, not suggestions)

| # | Open item | Severity |
|---|---|---|
| **O-1** | **The `P1-drift-feel-red` slice packet is STALE vs. the Phase-A merge.** The packet predates the merge and names a 3-file test set; the merge's own file set is 9 files and the contract set above needs 5 test/fixture homes. The packet must be re-patched *before* the RED phase starts. | **BLOCKER** |
| **O-2** | **A fresh `01-planning` `green-light: true` for Step 14N is required and does not exist** (the 2026-10-23 green-light does not cover 14N; `plans.md:5664` demands a fresh one). | **BLOCKER** |
| **O-3** | **The ≤ 3-file slice rule vs. the 6-file RED set** (§9). Split the slice or record a reasoned exception. | **BLOCKER (gate)** |
| **O-4** | **The AC-088 row's file list must be amended** (`car.pass.test.ts` → `lighting.test.ts`, the OBS-20 describe), because the uniform seam's owner is the OBS-20 describe with the richer harness. The AC row also says the lever lives in `car.pass.ts` — that stays true, but the *red_test* home moves. | HIGH |
| **O-5** | **The plan's "ALL 31 checks true" acceptance language** (Step 14G's record) must become **32** with the two new named checks (`tireGradientMagnitudeIsNonDegenerate`, `driftGateSurvivesThreeConsecutiveFrames`) — and, if B-4 is adopted, **33**. The check names must be recorded in the plan. | HIGH |
| **O-6** | The slice packet's `research_artifact` must point at **`plans/neatracing-voxel-racer.drift-feel.research.md`** (the arcade-refs/game-feel brief) and at **this file**. | MEDIUM |
| **O-7** | **Ratify the helper retire list** (§7): `computeOutwardPush`, `computeLongitudinalTractionLoss`, `computeDriftYawOffset` → GONE; `computeTractionFactor` / `computeCountersteerDelta` → re-specified. The merge retired only the three constants; each retired helper carries a live green `it` that becomes a GONE pin. | HIGH |
| **O-8** | **The canopy contract has no test home.** `engine/passes/glass.pass.ts` exists and **never reads `driftYawOffsetRad`** (confirmed by direct grep) → the canopy does not follow the body's drift yaw, and the new slide is 4.6× bigger, so the mismatch is amplified. There is **no `glass.pass.test.ts`** and the RED phase may not create one. Decision needed: **(a)** authorize `engine/passes/glass.pass.test.ts` as a new RED home (one file, one contract: *the canopy's model matrix must carry the same drift yaw as the car body*), or **(b)** defer the canopy to AC-090 and record the mismatch as accepted. **Recommendation: (a)** — the merge already lists `glass.pass.ts` in the file set. | HIGH |
| **O-9** | **The `DriftModelState` shape grows** (the 7-stage model needs `committedSlip` + the latched `turnSign`). The AC-047D byte-identity pin compares two serialisations, so it survives — but the plan's AC-047D wording should be re-read against the new state shape before the RED phase writes the pin. | MEDIUM |
| **O-10** | **The `workflow-update-sync.mjs` step-id regex rejects letter-suffixed ids** (`14N`) → the hook reports `currentWipStep: null`. The 14N advancement must be driven manually by the orchestrator. | LOW (operational) |

---

## 12. Risks carried by this contract set

| # | Risk | Mitigation | Certainty |
|---|---|---|---|
| R-1 | **The 7a numbers are re-derived by the implementer, not reproduced.** The pop/settle/step come from the research's own 60 Hz simulation of the model *equations*, not from a production run. If the production implementation differs (e.g. the articulation's post-tick convention), the measured values shift. | The contracts are stated as **floors/ceilings with a stated margin** (pop ≥ 2.0° vs 2.22° measured; settle ≤ 34 ticks vs 33; step ≤ 2.20° vs 2.13°), not as exact values. The implementer must report the measured triple in the green phase; if a value misses its floor, the **contract** is re-litigated, not the measurement. | 0.80 |
| R-2 | **The envelope margin is 0.19 %** (0.2246 vs 0.2250). It is exact and deterministic, so it is not flaky — but it leaves the tune slice almost no headroom in ω. | The ω ≤ 18 constraint is written into the plan as a **hard** constraint of the tune slice, and the fallback rule (move the limit to the measured peak, in BOTH pins) is specified. | 0.85 |
| R-3 | **The settle band [30, 34] ticks was chosen by this slice**, not measured across the whole slew band by the implementer. D6.3 reports the settle is invariant at 550 ms across the band 15–30 — if the implementer measures 29 or 35 ticks at some point, the band is wrong. | The band is stated as a **RED-slice decision (D-6)** and the implementer reports the measured settle at the pinned point; a miss re-opens D-6 explicitly. | 0.78 |
| R-4 | **The AC-087 floor couples to `MAX_SLIP` and `GAIN`** (§7c). A later tune that raises the gain toward 1.0 (the V-null branch) breaks the floor — which is the *point* — but a tune that lowers `MAX_SLIP` below 0.40 also breaks it unexpectedly. | The coupling (`MAX_SLIP ≥ 0.40` at `GAIN = 0.5`) is written into the plan next to the constant set, and the floor is a **named exported constant** so the coupling is machine-checked from both sides. | 0.88 |
| R-5 | **GF-15 steer chatter** (a 22.6° far-side swing at a 4 Hz square-wave steer) is a property of the LATCH design and is **not** covered by any contract here. | Flagged to `01-planning` as a known uncovered behaviour; the P0-4 (optional) entry contract and the user's AC-090 gate are the observation points. **This is a residual risk, not a solved one.** | 0.80 |
| R-6 | **The canopy mismatch** (O-8) is amplified by the 4.6× slide and is not covered by any contract in this set. | O-8's decision. | 0.85 |
| R-7 | **The 7d contract pins the gradient INPUTS, not the rendered gradient.** The AC-088 wording ("the magnitude is ASSERTED and becomes STRICTLY NON-ZERO") is satisfied in a weaker sense than the original intent. | Stated explicitly in §7d and in the plan: the rendered gradient is carried by **AC-090 (the user visual gate)**; the JS contract pins non-degeneracy of the seam. | 0.90 |
| R-8 | **The RED set is larger than the slice-atomicity rule allows** (O-3). If the slice is not split, the gate will fail the red step. | O-3. | 0.90 |

---

## 13. Certainty table (the gate the orchestrator applies)

| Finding | Confidence | Action |
|---|---|---|
| The merge items 7a–7g are the required RED contract set, and this set covers all seven | 0.95 | ACT |
| The AC-087 floor must live on the deterministic proxy, not the browser proxy | 0.93 | ACT |
| The AC-088 measurement is blind by construction and must move to the uniform seam | 0.90 | ACT |
| The three 7a numbers are reproducible only as floors/ceilings with margins | 0.80 | DELEGATE — the implementer measures the triple and reports it; the floors stand |
| The settle band [30, 34] ticks | 0.78 | DELEGATE — confirm against the implementer's measurement |
| The pose peak 0.2246 rad stays inside 0.2250 | 0.85 | ACT (with the fallback rule) |
| The helper retire list (§7) | 0.85 | DELEGATE — `01-planning` ratifies (O-7) |
| The 17 green pins / 10 actual re-pins | 0.92 | ACT |
| The RED file set is 6 files vs. the ≤ 3-file rule | 0.90 | ACT (escalate to 01-planning) |
| The canopy has no test home | 0.92 | DELEGATE — 01-planning decides (O-8) |
| The 7f engagement gate passes today (GUARD, not RED) | 0.88 | ACT |
| The browser prominence mirror at 0.10 rad | 0.85 | ACT (D-4) |

**Conflicts recorded.** The only evidence conflict in this set is the **scenario speed**
for the 7a measurements: the research reports the hold slide at both 25 m/s and 89 m/s,
and the pop/step numbers are only reproducible at `speedPercent = 1` (the articulation
amplitude is speed-scaled, so a 25 m/s run cannot reach a 2.22° pop). Resolution: the
tie-break order `runtime/validation > static code > comments/docs` does not apply (both
sources are simulation artefacts) → the **merge text wins** (it states the pop/step as
slew-band properties and the D6.2 bound as **speed-independent**) → the scenario is pinned
at 89 m/s. Residual risk: if the implementer's simulation cannot reproduce 2.22° at
89 m/s, the floor (2.0°) is what is re-litigated, not the scenario.

---

## 14. Handoff

**To `01-planning` (author).** This file is a *proposal*; the plan + the slice packet must
be patched before any RED authoring:

1. Re-patch the `P1-drift-feel-red` packet (O-1): the file set (§9), the `red_test` homes,
   the `research_artifact` (O-6), the AC-087..AC-090 row amendments (O-4, O-5).
2. Record the **feel direction** (D-11, "a bigger slide") in the plan **before any
   production edit** — the AC-087 wording requires it.
3. Record the **decision ledger** (§8) and the **helper retire list** (§7, O-7).
4. Obtain the **fresh 14N green-light** (O-2) and resolve the **slice atomicity** (O-3).
5. Decide the **canopy** (O-8) and the **optional entry contract** (P0-4).

**To `03-red-testing` (red author).** After the packet is patched and green-lit: author
the contracts exactly as specified here (§4, §5, §6, §7, §10), then run the focused jest
lanes and record the failing output. Expected red failures: the missing exports (the
constant set, the pose spring, the two named floor constants), the re-pinned envelope
(0.2250), the re-pinned slip bound (0.45 → P0-2 red), the AC-047B hard zero (P0-1 red),
and the bounce pose reset (P0-3 red).

**To `04-implementing` (tune).** The merged constant set (E1 item 4) is the single source;

---

## 15. Purge-proof output contract (the machine-readable record of this task)

> This section exists because the agent registry may purge the author mid-run and
> the chat report can be lost. **This file is the deliverable.** Everything below
> is the authoritative record of what Phase-B plan agent 1/4 did and what it needs
> next. Re-running the chat report is unnecessary.

```structured-v1
OUTPUT_CONTRACT: structured-v1
TASK_STATUS: SUCCESS
TIER: 1
ROLE: 02-researching
TASK_RECEIVED: Step 14N Phase-B plan agent 1/4 — propose the P1-drift-feel-red contract set (test files, floors, unit systems, exact assertions) for merge items 7a-7g; write ONLY to plans/neatracing-voxel-racer.14N-red-contracts.md; no plan or production edits.

FILES_READ:
- plans/neatracing-voxel-racer.plans.md (Step 14N packet ~L2320-2400; AC-087..AC-090 rows L482-485; Phase-A MERGE block L8414-8460; green-light clause L5664)
- plans/neatracing-voxel-racer.drift-feel.research.md (E1-E6, D1-D6.10 — the 7-stage model, the constant set, the measured predictions, D6.7/D6.8)
- plans/neatracing-voxel-racer.research.md (coupling design E1-E7 / D1-D7 / R1-R11)
- examples/neatRacing/physics/drift.model.ts (L44-L363: exports, constants, the dead twin)
- examples/neatRacing/physics/drift.model.test.ts (full, 337 lines)
- examples/neatRacing/physics/physics.step.service.ts (L73-111, L364-442, L416-417)
- examples/neatRacing/physics/physics.step.service.test.ts (full, 267 lines)
- examples/neatRacing/physics/collision.utils.ts (L380-430, export list)
- examples/neatRacing/physics/collision.utils.test.ts (describe/it map)
- examples/neatRacing/engine/lighting.test.ts (OBS-20 describe L1024-1095)
- examples/neatRacing/engine/passes/car.pass.ts (L230-250, L285-400, L1450-1495)
- examples/neatRacing/engine/passes/car.pass.test.ts (drift-yaw describe L360-521)
- examples/neatRacing/engine/passes/glass.pass.ts (the canopy — no test home, never reads driftYawOffsetRad)
- docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html (L87-215, L784-973)

FILES_CHANGED:
- plans/neatracing-voxel-racer.14N-red-contracts.md (NEW — the only write of this task)

KEY_FINDINGS:
- 7a: three numbers pinned TOGETHER at ONE (omega=18, zeta=0.35, art=0.08, arm=0.01, slew=20) point via ONE named scenario (SCN-DRIFT-01: 180 build ticks + 120 countersteer ticks at 89 m/s, 60 Hz) — pop floor 2.0 deg (measured 2.22), settle band [30,34] ticks = 500-566.7 ms (measured 33 = 550), per-tick pose-TARGET step ceiling 2.20 deg (measured 2.13) + the exact speed-independent sub-bound ART*SLEW*dt = 1.52778 deg. Stated as floors/ceilings with margins, never the hold pose alone. [0.80 reproducibility, 0.95 contract shape]
- 7b: zero-straight-line yaw as an EXACT identity — toBe(0) on pose/poseRate/articulation over 600 ticks (the articulation is slip-gated so the float math is exact), with 1e-9 as the outer bound; plus the integration form on PhysicsCarState. [0.92]
- 7c: AC-087 re-pinned on the DETERMINISTIC proxy |slip|*|GAIN-1|; new exported floor NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD = 0.20 rad (4.05x the old browser baseline 0.0494; 11.4% under the measured 0.2256 hold); the CURRENT model measures 0.0392 -> a genuine RED. Coupled constraint: MAX_SLIP >= 0.40 at GAIN=0.5. The browser lane gets a LOOSE 0.10 mirror (the 0.0494/0.0603/0.3599 spread is noise). [0.88]
- 7d: AC-088 moved to the fragment-stage uTireGradient* uniform seam, owned by lighting.test.ts's OBS-20 describe (NOT car.pass.test.ts — the richer harness lives there); new exported NEATRACING_TIRE_GRADIENT_MIN_SPREAD = 0.5 (current spread 1.35). The VBO probe is blind BY CONSTRUCTION (upperAvg == lowerAvg == 0.28565), so the RENDERED gradient is carried by AC-090, not by a false JS assertion. [0.90]
- 7e: the RECOMMENDED variant — the pose is held inside MAX_SLIP*GAIN = 0.2250 rad (measured peak 0.2246, 0.19% inside); the derivation assertion is RED today (0.45*0.5 vs pi/2*0.5); fallback rule = move the limit to the measured peak in BOTH pins. [0.85]
- 7f: the fixture's 3-frame CONSECUTIVE gate (DRIFT_PROXY_MIN_RAD 0.01 / DRIFT_PROXY_MIN_FRAMES 3) is retained verbatim; the red artifact is the missing NAMED check (the run length is currently folded into tireYawDriftEngaged) + a deterministic jest twin. [0.88]
- 7g: SUPERSEDED — an explicit DO-NOT-AUTHOR on any GAIN_eff ~ 1.0 contract, replaced by the anti-null guard (gain pinned at 0.5, |GAIN-1| >= 0.5, the pose and the slip share a sign). [0.90]
- P0 (the merge's physical half, NOT in 7a-7g, but required for the merge to mean anything): the countersteer must UNWIND to the hard zero and never MIRROR (today it parks at +52.15 deg at 89 m/s); the slide must stay inside the re-pinned 0.45 bound (today it reaches 1.148 rad); a bounce must reset the pose state (collision.utils.ts:402-420 never touches driftYawOffset). All three are genuine REDs.
- 17 green pins examined / 10 actually re-pinned (MAX_SLIP pi/2 -> 0.45; the grip budget loses its speed dependence; the countersteer gate inverts; 3 helpers retired; the envelope pi/2 -> 0.2250; AC-047B 15 ticks -> 20 ticks + toBe(0); car.pass.test.ts fixtures 0.35/0.5 are now OUT of envelope). The re-pinned 0.45 bound turns the existing green "sustained high-speed slide stays bounded" test RED ON PURPOSE.
- The current P1-drift-feel-red packet is STALE: 3 of the 5 required RED homes are missing from files_to_change (physics.step.service.test.ts, lighting.test.ts, collision.utils.test.ts); AC-087's floor moves off the browser proxy; AC-088's red_test moves house; research_artifact must point at the drift-feel brief; a fresh 14N green-light does not exist.

ACTIONS_TAKEN:
- RAG-loaded the workflow snapshot, the 14N slice context, the plan Step 14N packet, the Phase-A MERGE block and both research artifacts (Cortex load_document; targeted view/grep on the 1.9 MB plan).
- Dispatched 2 read-only Tier-3 scouts in parallel after build_dispatch_packet returned ok:true for both (caller_tier 1, complexity moderate).
- Authored this file (new file only).
- Ran the consolidated slice-advancement gate (14N, changed-files = this artifact): PASS 4/4.

VALIDATION_EVIDENCE:
- neataptic-gate-mcp run_gate_check(gate=slice-advancement, slice-id=14N, changed-files=plans/neatracing-voxel-racer.14N-red-contracts.md) -> pass:true, 4/4 sub-gates (plan-sync, step-packet, plan-slice-quality, plan-command-lint), fixHint "All 4 gates passed for slice 14N (TRIVIAL)".
- git status --short -> the only write attributable to this task is this file. NOTE: plans/neatracing-voxel-racer.plans.md shows as modified in the working tree, but that is the PRE-EXISTING 2026-10-25 game-feel scout PlanUpdate already in the tree BEFORE this task (121 insertions), NOT an edit by this agent. The sibling untracked files (14N-integration.md, 14N-rebaseline-risk.md, *.research.md, right-world.specs.md) belong to the other Phase-B agents and to earlier scouts.
- No jest lane was run: the RED contracts are PROPOSED, not authored (authoring is 03-red-testing's job, and the slice packet must be re-patched first).

BLOCKERS:
- O-1 the P1-drift-feel-red slice packet is stale vs. the Phase-A merge -> 01-planning must re-patch.
- O-2 a fresh 01-planning green-light:true for 14N is required (plans.md:5664) and does not exist.
- O-3 the RED set is 6 files (5 test/fixture + the plan) vs. the slice-validator's <=3-file atomicity rule -> split the slice or record a reasoned exception.
- O-7 the merge retired only 3 constants, but 3 more helpers (computeOutwardPush, computeLongitudinalTractionLoss, computeDriftYawOffset) are incompatible with the 7-stage model and each owns a live green `it` -> the retire list must be ratified before the GONE pins are authored.
- O-8 the canopy contract has NO test home (no engine/passes/glass.pass.test.ts exists and the RED phase may not create one); glass.pass.ts never reads driftYawOffsetRad -> authorize a new home or defer to AC-090.

RISKS_OR_GAPS:
- R-1 the 7a numbers come from the research's own simulation, not a production run -> stated as floors/ceilings with margins; a miss re-opens the contract, not the measurement. [0.80]
- R-2 the envelope margin is 0.19% -> omega <= 18 becomes a hard constraint of the tune slice, with the both-pins fallback rule. [0.85]
- R-3 the settle band [30,34] ticks is THIS slice's decision, not an implementer measurement -> DELEGATE for confirmation. [0.78]
- R-5 GF-15 steer chatter (22.6 deg far-side swing at 4 Hz square-wave steer) is a property of the LATCH design and is covered by NO contract here -> residual, surfaced to the user gate.
- Unresolved conflict recorded: the 7a scenario speed (the pop is only reproducible at speedPercent=1, i.e. 89 m/s) -> the merge text wins (the step bound is stated speed-independent); the FLOOR, not the scenario, is the re-ligitable number.

LEARNING_EVENT_NEEDED: false

SUGGESTED_NEXT_AGENT: 01-planning (author)

PHASE_COMPLETE: true

SUB_ORCHESTRATORS_USED:
- implementation-pattern-scout (read-only; the four RED homes, the house idioms, the reusable helpers, the green pins that break)
- boundary-mapper (read-only; the source anchors, the single pose insertion point, the collision-bounce gap, the stale-vs-merge deltas)

SUMMARY:
Proposed the complete P1-drift-feel-red contract set for Step 14N in the single permitted new file. All seven merge items 7a-7g are specified with an exact test home, an exact unit, a named floor/ceiling, verbatim assertion text and a RED-vs-GUARD classification; 7a pins the pop/settle/per-tick-step TOGETHER at one (omega, zeta, art, arm, slew) point via one named scenario rather than the hold pose alone. Beyond the required list I added the P0 physical umbrella contracts (the countersteer must unwind, not mirror; the slide must be bounded at the re-pinned 0.45; a bounce must reset the pose) because without them the merge would deliver a bigger slide that still MIRRORS on the countersteer, i.e. the user's "bounce-back must be sharper" would be untouched. The set also carries the 17-pin re-baselining ledger (10 real changes, including the one that deliberately turns an existing green test RED), the GONE ledger, the decision ledger for the values the merge delegated to the RED slice (slew 20/s, AC-087 floor 0.20 rad on the deterministic proxy, AC-088 spread floor 0.5, the measurement conventions), the widened file set, and 10 open items for 01-planning. The gate passes 4/4. No plan or production file was edited.
```

### 15.1 Write-scope attestation (re-verify after any resume)

| Rule | Status | Evidence |
|---|---|---|
| Only the designated NEW artifact was written | HELD | `git status --short` — only `plans/neatracing-voxel-racer.14N-red-contracts.md` is attributable to this task. |
| No plan edit | HELD | `plans/neatracing-voxel-racer.plans.md` untouched by this agent (the pre-existing 2026-10-25 game-feel PlanUpdate is in the tree, not mine). |
| No production edit | HELD | no file under `examples/`, `src/`, `docs/` written or modified. |
| No git mutation | HELD | read-only inspection only; no add/commit/checkout/reset/fetch. |
| The RED phase may not CREATE a test file | HELD | §9 names only EXISTING homes; all four jest homes + the browser fixture exist. |
| The merged accumulator is NOT on `DriftModelState` | HELD | §4/§5 place the 3 scalars on `PhysicsCarState` (AC-047D byte-pin preserved). |
| The spring steps on the 60 Hz tick, never the render path | HELD | §4/§9/§10 keep the pose off the render path; the browser lane only READS the pose. |

### 15.2 Resume instructions (if this task is re-dispatched after a purge)

1. **Do NOT re-author.** This file is complete and IS the deliverable; re-writing it wastes the budget and risks drift.
2. Read §0 (scope) and §11 (open items) first, then §4/§5 (the contracts) — §1-§3 and §6-§10 are the supporting ledgers.
3. The only remaining work is `01-planning`: re-patch the slice packet, record the feel direction (D-11) BEFORE any production edit, record the decision ledger (§8) + the helper retire list (§7, O-7), obtain the fresh 14N green-light (O-2), and resolve the slice atomicity (O-3), the canopy (O-8) and the optional entry contract (P0-4).
4. Then `03-red-testing` authors the RED contracts verbatim, and `04-implementing` (tune) applies the constant set.
5. If a question about a NUMBER arises, do not re-derive it: §2 (units), §3 (scenario), §8 (decisions) and §13 (certainty) already carry the provenance, the tie-break rule and the residual risk.

### 15.3 Change log of this file (author-side, for the audit trail)

| When | Change | Why |
|---|---|---|
| 2026-10-26 (initial) | Created with §0-§14 — the full contract set, the ledgers, the open items, the handoff. | The task deliverable. |
| 2026-10-26 (post-gate) | Added the top `WRITE STATUS: COMPLETE` marker + §15 (the structured output contract, the write-scope attestation, the resume instructions, this change log). | The registry may purge the author mid-run and lose the chat report; the FILE must carry the whole record. |

### 15.4 The `04-implementing` (tune) handoff, continued from §14 above

the pose state is **three scalars on `PhysicsCarState`** (never on `DriftModelState`); the
spring steps on the **60 Hz tick** (never the render path); the sign convention is
**LEFT = negative slip**; the pose target is **`+committedSlip·GAIN`** (never `−`).
