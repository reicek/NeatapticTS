# Step 14N — Phase-B Proposal 3/4: INTEGRATION WIRING

> **Role:** `02-researching` (Tier 1) — Phase-B plan agent 3/4 of 4.
> **Scope:** propose the **integration wiring** for the merged `DriftPoseSpringState` accumulator:
> state types, the `drift.model.ts` restructure, the 60 Hz tick wiring in `physics.step.service.ts`,
> the collision reset, the dead-export disposition, and the naming grammar.
> **NOT in scope (owned by the sibling proposals):** the RED contract list (proposal 1/4),
> the physics-coupling re-baseline risk (proposal 2/4), the browser fixture + user gate (proposal 4/4).
> **Hard constraint honoured:** this proposal writes **only this file**. No plan edit, no production edit.

| Field | Value |
| --- | --- |
| `step` | `14N` (Arcade drift FEEL sharpening) |
| `proposal` | 3/4 — integration |
| `date` | 2026-10-26 |
| `inputs` (RAG) | plan Step 14N entry + **Phase-A MERGE block** (`plans/neatracing-voxel-racer.plans.md:8414-8480`, the design authority) + `plans/neatracing-voxel-racer.drift-feel.research.md` (D1/D2/D3/D6) + `plans/neatracing-voxel-racer.research.md` (E1-E7, D1-D7, R1-R11) |
| `production files read` (read-only) | `examples/neatRacing/physics/drift.model.ts` (372 L), `examples/neatRacing/physics/physics.step.service.ts` (543 L), `examples/neatRacing/physics/collision.utils.ts`, `examples/neatRacing/physics/drift.model.test.ts`, `examples/neatRacing/physics/physics.step.service.test.ts` |
| `delegation` | 1 × Tier-3 `boundary-mapper` (line-anchored surface map) + 1 × Tier-3 `plan-scout`-grade read of the plan |
| `status` | **COMPLETE** — 2026-10-26T08:03 ET. All 13 sections below are final; this file is the sole deliverable and is **self-contained** (no dependency on the agent's chat report). |

> **FOR THE RESUMING AGENT / PLAN AUTHOR (read this first).**
> This document is complete as written — nothing was left in a scratch buffer. Three things in it
> need a decision *outside* this proposal, and they are flagged inline where they occur:
> **(1)** §9.1 lists 11 numbered hand-off items for the 4-way merge (5 of them are single-assertion
> re-homes in the three shared test files — assign one owner per test file);
> **(2)** two judgement calls are deliberately escalated rather than settled — the pose target's slip
> source (§4.4, 0.85) and the nested-vs-flat state shape (§8.2, 0.80);
> **(3)** two file-set additions are required — `browser-entry/browser-entry.ts` (1 line, §5.5) and a
> NEW `physics/collision.utils.test.ts` (§6, §9.1-9).
> The one-line summary of the proposal is §0; the per-finding certainty table is §12.

---

## 0. The one-line answer

Replace the **stateless** visual offset (`driftYawOffset = slip × GAIN`, recomputed from scratch
every tick) with a **stateful** pose spring (`pose`, `poseRate`, `articulation`) that is integrated
**once per 60 Hz physics tick**, carried inside **one** nested per-car drift channel on
`PhysicsCarState`, **reset atomically** on a bounce, and published to the renderer as the **same
single scalar** the render bridge already consumes. The dead `computeDriftYawOffset` export is
**deleted** (not wired), so exactly one visual-yaw formula survives in the codebase.

---

## 1. Scope, non-goals, and the resulting file set

### 1.1 Non-goals

- No re-design of the physical 7-stage model — that is proposal 2/4's domain; this document only
  fixes the **interfaces** the two halves compose through.
- No new RED contracts — proposal 1/4 owns the assertions; this document names the *observable
  quantities* each contract must be able to read.
- No canopy/glass fix — recorded as a follow-up slice (§10, R-04).
- No change to the render contract (`car.pass.ts` consumes a single scalar from the render pose
  bag; that contract is **unchanged**).

### 1.2 The file set this proposal implies

| File | Change | Why | Collides with |
| --- | --- | --- | --- |
| `examples/neatRacing/physics/drift.model.ts` | **Major** — constants, types, 3 new step functions, 4 deletions | The model + constant owner (MERGE item 4) | proposal 1/4 (RED homes), 2/4 |
| `examples/neatRacing/physics/physics.step.service.ts` | **Major** — state field, tick wiring, 1 constant deleted | The tick owner | proposal 2/4 (re-baseline) |
| `examples/neatRacing/physics/collision.utils.ts` | **Small** — 2 reset writes | Bounce reset (MERGE risk 8) | — |
| `examples/neatRacing/browser-entry/browser-entry.ts` | **1 line** (`:941`) | The single render-bridge read of the published pose | — |
| `examples/neatRacing/physics/drift.model.test.ts` | Delete/re-home the test-only mirror | Dead-export cleanup | **proposal 1/4 — HARD COLLISION** |
| `examples/neatRacing/physics/physics.step.service.test.ts` | Re-home the drift pins | Dead-export cleanup + new pins | **proposal 1/4 — HARD COLLISION** |
| `examples/neatRacing/engine/passes/car.pass.test.ts` | Envelope limit re-derivation | `MAX_SLIP` change (§8.4) | **proposal 1/4 — HARD COLLISION** |
| `examples/neatRacing/engine/passes/car.pass.ts` | **NONE** | The render contract is unchanged (§5.5) | — |
| `examples/neatRacing/engine/passes/glass.pass.ts` | **NONE in 14N** | Canopy desync → follow-up slice (§10, R-04) | — |

> **Cross-proposal collision warning for the 4-way merge:** the three test files are the shared
> surface between proposal 1/4 (RED contracts) and this one. The merge must assign **one** owner
> per test file. The hand-off in §9.1 lists the exact test-file items this proposal expects to be
> re-homed so proposal 1/4 can write them directly instead of round-tripping.

---

## 2. Sign and unit invariants — the contract the two halves compose through

These are the invariants that must hold across **all four** proposals. They are stated here because
every one of them is currently a live sign-error hazard, and the MERGE already flagged one
(`-committed` in the pose target) as a SIGN ERROR.

**INV-1 — SLIP convention (already pinned to production, verified here).**
`slipAngle = atan2(lateralVelocity, speed)`; `lateralVelocity` is *positive = left of the heading*.
A **LEFT turn therefore produces a NEGATIVE slip** — the car's velocity lags to the right of a
counter-clockwise heading. `steer > 0` = counter-clockwise = **LEFT**
(`resolveYawRatePerTick`, `physics.step.service.ts:473`, `STEERING_RATE = 2`).
*Verified against the live code:* `computeOutwardPush` returns
`-centrifugalMultiplier · yawRatePerTick · (speed − minSpeed)` (`drift.model.ts:225`), so a positive
yaw rate pushes `lateralVelocity` negative ⇒ negative slip. The convention is self-consistent
today and must not change.

**INV-2 — `turnSign` is the TURN direction, `committedSlip` is in the SLIP convention.**
`turnSign = +1` for a LEFT turn (steer > 0 at commitment), `-1` for a RIGHT turn, `0` = not
committed. The committed ramp therefore **decreases** `committedSlip` for a LEFT turn:
`committedSlip −= turnSign · ENTRY_RATE · |steer| · dt`.
*Why this matters:* if `turnSign` is instead defined as `sign(slip)` (i.e. `-1` for a left turn),
the same ramp formula produces a **positive** `committedSlip` for a LEFT turn and the entire
committed chase drives the car the wrong way. The MERGE's sign item only pins the pose target;
**the ramp sign is a second, independent hazard** and this proposal pins it explicitly.

**INV-3 — the pose is in the SLIP convention and uses `+committedSlip`.**
`poseTarget = committedSlip · GAIN + ART_RAD · articulation`. A LEFT turn ⇒ `committedSlip < 0` ⇒
`pose < 0` ⇒ the body rotates clockwise, i.e. *into* the slide. This matches the **live** formula
`nextSlipAngle · NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` (`physics.step.service.ts:416-417`), which
must keep the same sign. The MERGE item 6 (`use +committedSlip`, not `-committed`) is consistent
with this.

**INV-4 — the countersteer term must REDUCE `|slip|`.**
The countersteer authority adds `turnSign · AUTHORITY · |steer| · speedPercent · dt` to
`lateralVelocity`. For a LEFT turn (`turnSign = +1`, `lateralVelocity < 0`) that pushes `lateralVelocity`
**toward zero** ⇒ the slide is caught, not deepened. (The old model's `computeCountersteerDelta`
used `-slipSign · …` for the same purpose — the sign convention carries over, the *authority*
mechanism does not.)

**INV-5 — units.**
`COUNTERSTEER_AUTHORITY_MPS = 70` is an **acceleration** (m/s²), because it is multiplied by `dt`
to yield a velocity delta in m/s. The `_MPS` suffix in the MERGE's constant list is therefore
**unit-mislabeled**. See §8.3 (REC-4) — this needs a single decision applied consistently across
all four proposals.

**INV-6 — the pose spring is integrated ONLY at the physics tick rate.**
`dt = NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS` (`physics.step.service.ts:125`, `1/60`). The spring
is never integrated in the render path. A per-frame spring would settle **2.4× faster at 144 Hz**
than at 60 Hz (MERGE risk 8).

---

## 3. The state types

### 3.1 The three new types (all in `drift.model.ts`)

```ts
/**
 * The latched physical drift commitment (arcade-refs D1, stages 0-1).
 *
 * `slip` is the COMMITTED slide angle in the SLIP convention
 * (LEFT turn => negative, INV-1/INV-2). `turnSign` latches the turn
 * direction at commitment time so the countersteer gate can distinguish
 * "steering into the slide" from "catching the slide".
 */
export interface DriftCommittedState {
  /** Committed slide angle in rad, SLIP convention. Bounded by MAX_SLIP_ANGLE_RAD. */
  slip: number;
  /** Latched turn direction: +1 = LEFT, -1 = RIGHT, 0 = not committed. */
  turnSign: number;
}

/**
 * The merged VISUAL accumulator (MERGE item 2 + game-feel D6).
 *
 * ONE accumulator only: the coupling design's transient-gain state and
 * the arcade-refs pose spring are the same mechanism and must not ship
 * as two competing accumulators.
 */
export interface DriftPoseSpringState {
  /** Published visual yaw offset in rad, SLIP convention. */
  pose: number;
  /** Spring velocity in rad/s. */
  poseRate: number;
  /** Slew-limited articulation accumulator, dimensionless, in [-1, 1]. */
  articulation: number;
}

/**
 * The whole per-car drift channel: the physical commitment plus the
 * merged visual pose spring. ONE carrier so a bounce reset is atomic
 * (a single assignment) and the spring's three scalars can never be
 * updated out of sync with each other.
 */
export interface DriftChannelState {
  /** The latched physical commitment. */
  committed: DriftCommittedState;
  /** The merged visual pose spring. */
  pose: DriftPoseSpringState;
}
```

### 3.2 Where the state lives — and why it is *not* on `DriftModelState`

**DECISION: the drift channel is a single new nested field on `PhysicsCarState`.**

```ts
// physics.step.service.ts — inside `export interface PhysicsCarState` (currently :73-114)
  /**
   * The per-car drift channel: the latched physical commitment plus the
   * merged visual pose spring. Carried as ONE nested value so the bounce
   * reset is atomic and the spring's internal scalars cannot drift apart.
   */
  drift: DriftChannelState;
```

**Why one nested field rather than five flat fields.** The pose spring is a 3-scalar integrator
(`pose`, `poseRate`, `articulation`). If those were flat fields on `PhysicsCarState`, the bounce
reset in `collision.utils.ts` would have to zero five fields in two branches, and a partial write
(a pose advanced while its rate was not) is a silent spring corruption. One nested carrier makes
the reset a single `drift: createZeroDriftChannelState()` assignment, and makes the reset
**structurally incapable** of being partial. The same argument applies to `committed` (a latched
pair that must not be half-reset).

**Why it is NOT on `DriftModelState` — and why `DriftModelState` should be deleted entirely.**
The MERGE forbids the visual accumulator on `DriftModelState` because of the **AC-047D byte-identity
pin**. The pin is stronger than the MERGE records, and it decides the whole question:

* `drift.model.test.ts:44-45` — `const serializeDriftState = (state: DriftModelState): string => JSON.stringify(state);`
* `drift.model.test.ts:325-328` — after 120 straight-line steps, `serializeDriftState(state)` must
  equal `serializeDriftState(createZeroDriftState())` **byte for byte**.

Because the serializer is a whole-object `JSON.stringify`, the pin is **shape-sensitive**: any field
added to `DriftModelState` changes the serialized bytes, and the zero-state and the straight-line
state must keep the same key set *and* the same key order. Any new accumulator that is not exactly
zero after a straight line (a spring's `poseRate` is zero; a latch is zero) would still pass — but
the pin becomes a fragile accident of key order rather than a designed contract.

**Better still: `DriftModelState` has no production consumer at all.** Verified:

| Export | Production consumer | Test-only? |
| --- | --- | --- |
| `DriftModelState` (`:100`) | **none** — `PhysicsCarState` *flattens* `lateralVelocity`/`slipAngle` (`physics.step.service.ts:101`, `:94`) | yes |
| `createZeroDriftState` (`:124`) | **none** (spawn uses `createInitialPhysicsCarState`) | yes |
| `stepDriftModel` (`:140`) | **none** (the live seam is `computeNextLateralVelocity`) | yes |
| `computeDriftYawOffset` (`:363`) | **none** | yes |
| `computeOutwardPush` (`:217`) | internal only (via `stepDriftModel` / `computeNextLateralVelocity`) | exported for tests |
| `computeTractionFactor` (`:250`) | internal only | exported for tests |
| `computeCountersteerDelta` (`:310`) | internal only | exported for tests |
| `computeLongitudinalTractionLoss` (`:283`) | **live** (`physics.step.service.ts:405`) | no |
| `computeSlipAngle` (`:339`) | **live** (`physics.step.service.ts:412`) | no |
| `computeNextLateralVelocity` (`:179`) | **live** (`physics.step.service.ts:388`) | no |

`DriftModelState` / `createZeroDriftState` / `stepDriftModel` are a **test-only mirror** of two
production fields. Under the new model the drift integration becomes *stateful* (the committed
angle accumulates), so a stateless `DriftModelState` mirror can no longer represent it — keeping it
would create a **second implementation of the model** (a dual-path code smell that the MERGE
explicitly forbids for the visual half, and that the execute skill forbids generally).

> **DECISION (recommended): delete `DriftModelState`, `createZeroDriftState` and `stepDriftModel`
> together with their tests, and re-anchor AC-047D on the PRODUCTION path:**
> `JSON.stringify(createInitialPhysicsCarState(seed))` after 120 straight-line ticks must equal
> `JSON.stringify(createInitialPhysicsCarState(seed))` byte for byte.
> `physics.step.service.test.ts` already has the exact helper it needs —
> `const serializeState = (state: PhysicsCarState): string => JSON.stringify(state);`
> (`:72-73`) — and the determinism test at `:133-137` already compares two runs. This makes
> AC-047D a **stronger** contract (it pins the shipped path, not a shadow) and removes the
> shape-sensitivity entirely, because the straight-line tick now provably leaves every field of the
> nested channel at its zero value.
>
> **If the plan author prefers to keep AC-047D textually bound to `DriftModelState`:** the fallback
> is to keep the 2-field `DriftModelState` *unchanged* and carry the committed state in a separate
> type — but then `stepDriftModel` can no longer compute a correct lateral velocity without the
> commitment, so it must be deleted anyway and AC-047D must be re-anchored regardless. **There is
> no design in which `stepDriftModel` survives as a meaningful test.**

### 3.3 The `PhysicsCarState` shape after the change

```ts
export interface PhysicsCarState {
  x: number;            // :75   (unchanged)
  y: number;            // :77   (unchanged)
  heading: number;      // :79   (unchanged)
  speed: number;        // :81   (unchanged)
  yawRate: number;      // :88   (unchanged)
  slipAngle: number;    // :94   (unchanged — the PUBLISHED tick-end slip)
  lateralVelocity: number; // :101 (unchanged — the PUBLISHED tick-end lateral velocity)
  drift: DriftChannelState;  // NEW — replaces `driftYawOffset` (:108)
  tick: number;         // :110  (unchanged)
}
```

**`driftYawOffset` is RETIRED from `PhysicsCarState`.** The published visual yaw offset is now
`state.drift.pose.pose` — one home, one write, no mirror to fall out of sync.

*Why retire rather than keep it as a flat mirror:* today the bounce path (`collision.utils.ts`)
zeroes `lateralVelocity` but **leaves `driftYawOffset` stale** (§6) — a latent bug. With the pose
living inside the channel, the reset writes the whole channel and the stale-pose class of bug is
**eliminated by construction** rather than by discipline.

*Blast radius of the retirement (verified):*

| Consumer | Impact |
| --- | --- |
| `browser-entry/browser-entry.ts:941` — `carPoseBag.driftYawOffsetRad = car.driftYawOffset;` | **1 line** → `= car.drift.pose.pose;` |
| `physics.step.service.ts:108` (decl), `:312` (spawn), `:416-417` (compute), `:440` (carry) | replaced by the channel (§5) |
| `physics.step.service.test.ts:156-163` | re-homed to the pose (proposal 1/4) |
| `engine/passes/car.pass.ts:438`, `:1365`, `:1567` | **NO CHANGE** — these read the *render pose bag* field `driftYawOffsetRad`, not the car state. The bag contract is untouched. |
| `engine/passes/car.pass.test.ts:7` (a doc comment naming `PhysicsCarState.driftYawOffset`) | comment refresh |
| `src/**` | **NO MATCHES** for `driftYawOffset` or `PhysicsCarState` — the state record is entirely internal to the example, so the retirement is contained. |

---

## 4. `drift.model.ts` restructure

### 4.1 Constant table (single source, per MERGE item 4)

Current → proposed. All current line numbers are **pre-change**.

| Current name | Current line | Current value | Proposed name | Proposed value | Disposition |
| --- | --- | --- | --- | --- | --- |
| `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` | `:44` | `Math.PI / 2` | same | **`0.45`** | **VALUE CHANGE** (MERGE) — see §8.4, it silently moves the car-pass envelope limit |
| `NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD` | `:50` | `0.25` | same | **`0.20`** | VALUE CHANGE (MERGE) |
| `NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD` | `:59` | `2` | same | `2` | unchanged |
| `NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S` | `:68` | `0.6` | — | — | **RETIRE** (MERGE) → superseded by `SPEED_SCRUB_COEFF_PER_S` |
| `NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S` | `:75` | `12` | — | — | **RETIRE** (MERGE) → superseded by `COUNTERSTEER_AUTHORITY_MPS` |
| `NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S` | `:82` | `4` | — | — | **RETIRE** (MERGE) → superseded by the committed chase + `SLIP_RESPONSE_TAU_S` |
| `NEATRACING_DRIFT_MIN_SPEED_MPS` | `:90` | `15` | same | `15` | unchanged |
| `NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` | `:97` | `0.5` | same | `0.5` | unchanged — becomes the pose target's base gain |
| — | — | — | `NEATRACING_DRIFT_ENTRY_RATE_RAD_PER_S` | `3.5` | **NEW** (MERGE) |
| — | — | — | `NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S` | `2.0` | **NEW** (MERGE) |
| — | — | — | `NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S` | `0.045` | **NEW** (MERGE) — the committed chase's exponential response |
| — | — | — | `NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS` | `70` | **NEW** (MERGE) — unit label caveat, §8.3 |
| — | — | — | `NEATRACING_DRIFT_COUNTERSTEER_MIN_SLIP_RAD` | `0.02` | **NEW** (MERGE) — the countersteer gate's minimum slip |
| — | — | — | `NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD` | `0.02` | **NEW** (MERGE) — the zero-crossing snap-back threshold |
| — | — | — | `NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2` | `324` | **NEW** (MERGE) |
| — | — | — | `NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S` | `12.6` | **NEW** (MERGE) |
| — | — | — | `NEATRACING_DRIFT_POSE_STEER_YAW_RAD` | `0.08` | **NEW** (MERGE) — the articulation amplitude |
| — | — | — | `NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD` | `0.01` | **NEW** (MERGE) — the articulation **arm** (was 0.10) |
| — | — | — | `NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S` | `20` | **NEW** (MERGE) — the ONE genuinely new constant of the visual half |
| — | — | — | `NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S` | `0.25` | **NEW** (MERGE) — replaces the retired traction-loss constant |

**No deferred cleanup.** The three retired constants are **deleted**, not aliased and not left at
their old values behind a new name. `drift.model.test.ts:66-82` currently pins all three
(`LONGITUDINAL_TRACTION_LOSS_RATE_PER_S < LATERAL_GRIP_RATE_PER_S`, etc.) — those assertions are
removed, not rewritten.

### 4.2 Export surface after the restructure

| Export | Disposition | Note |
| --- | --- | --- |
| `DriftModelState` (`:100`) | **DELETE** | test-only mirror; the byte-pin moves to the production path (§3.2) |
| `DriftStepInput` (`:108`) | **REPLACE** | `yawRatePerTick` and `centrifugalMultiplier` become dead under the committed model; the tick-start `lateralVelocity` and `slipAngle` become inputs |
| `createZeroDriftState` (`:124`) | **DELETE** | replaced by `createZeroDriftChannelState` |
| `stepDriftModel` (`:140`) | **DELETE** | the committed model is stateful; a stateless mirror cannot express it |
| `computeOutwardPush` (`:217`) | **DELETE** | the committed chase replaces the centrifugal push |
| `computeTractionFactor` (`:250`) | **REPLACE** | new law + name (see below) |
| `computeLongitudinalTractionLoss` (`:283`) | **RENAME** | `computeLongitudinalSpeedScrub` — same form, new coefficient |
| `computeCountersteerDelta` (`:310`) | **REPLACE** | new law + name (see below) |
| `computeSlipAngle` (`:339`) | **KEEP UNCHANGED** | still the slip law; `atan2` form, zero at standstill |
| `computeDriftYawOffset` (`:363`) | **DELETE** | dead export — §7 |
| `computeNextLateralVelocity` (`:179`) | **REPLACE** | becomes the committed chase; signature changes (§4.4) |
| `DriftCommittedState`, `DriftPoseSpringState`, `DriftChannelState` | **NEW** | §3.1 |
| `DriftPoseStepInput` | **NEW** | §4.4 |
| `createZeroDriftChannelState` | **NEW** | spawn + bounce reset factory |
| `stepDriftCommitment` | **NEW** | D1 stages 0-1 |
| `stepDriftPoseSpring` | **NEW** | D1 stage 7 + D6 |

**Two renames, with their rationale (naming decisions, §8.2):**

- `computeTractionFactor` → **`computeGripBudgetFactor`**. The old law is a saturating
  `1/(1+demand)²` scaled by `speedPercent`; the new law is a **budget**
  `1/(1 + RATE · max(0, |slip| − THRESHOLD))` that is **not** scaled by `speedPercent`. The old name
  hides a genuine semantic change (the grip loss no longer depends on speed), so the rename is
  warranted, not cosmetic.
- `computeCountersteerDelta` → **`computeCountersteerAuthority`**. The old function *removed*
  lateral velocity (`-slipSign · …`); the new one *adds* authority toward zero with a
  **turn-sign gate** instead of a `steer·sign(slip)` gate, and its magnitude is an acceleration
  (m/s²) rather than a recovery rate (1/s). Same words, opposite mechanism — a rename prevents the
  next reader from "fixing" it back.

### 4.3 The model body (ordering contract for proposal 2/4)

`drift.model.ts` owns the **stage order**; `physics.step.service.ts` owns the **tick order**.
The 7 stages (arcade-refs D1) map onto three exported functions plus the service:

| D1 stage | Where it runs | Exposed as |
| --- | --- | --- |
| 0 — gate (speed > `MIN_SPEED_MPS`, steer engaged) | `drift.model.ts` | inside `stepDriftCommitment` |
| 1 — commitment ramp (`ENTRY_RATE` / `EXIT_RATE`) | `drift.model.ts` | `stepDriftCommitment` |
| 2 — grip budget factor | `drift.model.ts` | `computeGripBudgetFactor` |
| 3 — committed chase (`vTarget = tan(committedSlip)·speed`, exponential response) | `drift.model.ts` | `computeNextLateralVelocity` |
| 4 — countersteer authority | `drift.model.ts` | `computeCountersteerAuthority` |
| 5 — snap-back at the zero crossing | `drift.model.ts` | inside `computeNextLateralVelocity` |
| 6 — longitudinal speed scrub | `physics.step.service.ts` (it owns `speed`) | `computeLongitudinalSpeedScrub` |
| 7 — the merged pose spring | `drift.model.ts` | `stepDriftPoseSpring` |

Stage 6 is the one stage the service owns, because the service owns `speed` and applies the scrub
to the tick-end speed. The service's existing tick order (scrub **before** the slip computation)
is preserved, so the published slip remains computed at the post-scrub speed.

### 4.4 The three new function signatures

```ts
/** One physical drift tick: the tick-start sample the model reads. */
export interface DriftStepInput {
  /** Steering axis in [-1, 1]; positive steers counter-clockwise (LEFT). */
  steer: number;
  /** Longitudinal speed in m/s at tick start. */
  speed: number;
  /** Normalized speed in [0, 1]. */
  speedPercent: number;
  /** Fixed timestep in seconds (1/60). */
  dt: number;
  /** Lateral velocity at tick start, m/s. */
  lateralVelocity: number;
  /** Slip angle at tick start, rad, SLIP convention. */
  slipAngle: number;
}

/** One visual pose tick: the tick-END sample the spring reads. */
export interface DriftPoseStepInput {
  /** Steering axis in [-1, 1]; positive = LEFT. */
  steer: number;
  /** Normalized speed in [0, 1]. */
  speedPercent: number;
  /** The slip that feeds the pose target, rad, SLIP convention. */
  slipAngle: number;
  /** Fixed timestep in seconds (1/60). */
  dt: number;
}

/** Creates the zeroed drift channel — used by spawn AND by the bounce reset. */
export function createZeroDriftChannelState(): DriftChannelState;

/** D1 stages 0-1: the gate and the commitment ramp. */
export function stepDriftCommitment(
  prev: DriftCommittedState,
  input: DriftStepInput,
): DriftCommittedState;

/** D1 stages 3-5: the committed chase, the countersteer authority, the snap-back. */
export function computeNextLateralVelocity(
  committed: DriftCommittedState,
  input: DriftStepInput,
): number;

/** D1 stage 7 + game-feel D6: the merged visual pose spring. */
export function stepDriftPoseSpring(
  prev: DriftPoseSpringState,
  input: DriftPoseStepInput,
): DriftPoseSpringState;
```

**Note the deliberate asymmetry in the two slip inputs.** `DriftStepInput.slipAngle` is the
**tick-start** slip (the physical model reads the world as it was); `DriftPoseStepInput.slipAngle`
is the **tick-end** slip (the visual layer reacts to the frame that just happened). This mirrors the
existing doctrine in the service — "the loss reads the tick-start slip, like every other tick-start
read in this integrator" (`physics.step.service.ts:399-401`).

**Which slip feeds the pose target — recommendation and its consequence.** The MERGE item 6 refers
to "the pose target" and to `committed`, so the pose target uses the **committed** slip, not the
measured slip. This proposal recommends that, for three reasons: (i) the committed slip is smooth
and monotone, so the pose target has no physics jitter in it; (ii) the zero-straight-line identity
(MERGE RED contract (b), `1e-9`) then holds **exactly by construction** — committed slip `= 0` and
`|slip| < ARM` ⇒ `artTarget = 0` ⇒ `poseTarget = 0` ⇒ the spring never leaves zero; (iii) it makes
the body *lead* the measured slide, which is the arcade feel the user asked for.
**This is a decision for proposal 1/4 to pin as a RED contract** — the observable difference is that
with the committed slip the pose is smooth and monotonic, with the measured slip it inherits the
45 ms response lag and the measured envelope peak of 0.2295 rad.

---

## 5. The 60 Hz tick wiring in `physics.step.service.ts`

### 5.1 Tick order (proposed)

```
stepPhysics(state, input, surfaceResolver?)            // :362
 ├─ 1  surface (tick-start read)                        // :370  (unchanged)
 ├─ 2  yawRatePerTick, nextHeading                      // :373-377 (unchanged)
 ├─ 3  nextSpeed (longitudinal)                         // :380-383 (unchanged)
 │
 ├─ 4  DRIFT CHANNEL — physical
 │    4a  nextCommitted = stepDriftCommitment(
 │           state.drift.committed,
 │           { steer, speed: state.speed, speedPercent,
 │             dt, lateralVelocity: state.lateralVelocity,
 │             slipAngle: state.slipAngle })            // :386-397  (REWRITTEN)
 │    4b  nextLateralVelocity = computeNextLateralVelocity(
 │           nextCommitted, { …same input… })            // :386-397  (REWRITTEN)
 │
 ├─ 4c SPEED SCRUB (tick-start slip)                    // :398-411 (RENAMED call)
 │    nextSpeedAfterTraction = max(0, nextSpeed -
 │           computeLongitudinalSpeedScrub(state.slipAngle, nextSpeed, speedPercent) * dt)
 │
 ├─ 4d nextSlipAngle = computeSlipAngle(
 │           nextLateralVelocity, nextSpeedAfterTraction)  // :412-415 (unchanged)
 │
 ├─ 4e POSE SPRING  ★ THE NEW TICK
 │    nextPose = stepDriftPoseSpring(
 │           state.drift.pose,
 │           { steer, speedPercent,
 │             slipAngle: nextCommitted.slip,   // committed slip, INV-3
 │             dt })
 │
 ├─ 5  translate along (nextHeading + nextSlipAngle)     // :419-421 (unchanged)
 └─ 6  rebuild:
        slipAngle: nextSlipAngle,
        lateralVelocity: nextLateralVelocity,
        drift: { committed: nextCommitted, pose: nextPose },   // ★ replaces driftYawOffset
        tick: nextTick
```

**The ordering is load-bearing and is the whole integration contract:**

- The pose spring must run **after** the slip exists (4d) — it is a pure function of the slip.
- The pose spring must run **after** the commitment exists (4a) — its target is the committed slip.
- The pose spring must run **before** the rebuild (6) — the rebuild publishes the pose.
- The pose spring must run **inside** `stepPhysics`, i.e. at 60 Hz, never in the render path (INV-6).
- The pose spring must **not** run before the scrub (4c) — the published slip is the post-scrub slip
  and the pose must follow the published slip, not an intermediate one.

### 5.2 Line-anchored change list for `physics.step.service.ts`

| # | Anchor (pre-change) | Change |
| --- | --- | --- |
| 1 | import block `:37-43` | Replace the four imported symbols with the new surface: `computeLongitudinalSpeedScrub`, `computeNextLateralVelocity`, `computeSlipAngle`, `stepDriftCommitment`, `stepDriftPoseSpring`, `createZeroDriftChannelState`, `type DriftChannelState`. **`NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD` leaves this import** — the gain moves *inside* the pose spring, so the service no longer multiplies by it. |
| 2 | `PhysicsCarState` `:108` | Delete `driftYawOffset: number;`. Add `drift: DriftChannelState;` (with the JSDoc from §3.2). |
| 3 | `createInitialPhysicsCarState` literal `:312` | Replace `driftYawOffset: 0,` with `drift: createZeroDriftChannelState(),`. |
| 4 | `stepPhysics` step 4 `:386-397` | Rewrite the single `computeNextLateralVelocity` call into the two-call commitment→chase sequence (§5.1, 4a/4b). Drop `yawRatePerTick` and `centrifugalMultiplier` from the drift input; add `lateralVelocity: state.lateralVelocity` and `slipAngle: state.slipAngle`. |
| 5 | `stepPhysics` step 4b `:405` | Rename the call `computeLongitudinalTractionLoss` → `computeLongitudinalSpeedScrub`. The **position** in the tick order and the tick-start-slip read are unchanged. |
| 6 | `stepPhysics` `:416-417` | **DELETE** `const nextDriftYawOffset = nextSlipAngle * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD;` — this is the line the whole slice exists to replace. |
| 7 | `stepPhysics` new `:416-417` | **INSERT** the pose-spring call (§5.1, 4e). |
| 8 | rebuild literal `:440` | Replace `driftYawOffset: nextDriftYawOffset,` with `drift: { committed: nextCommitted, pose: nextPose },`. |
| 9 | `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER` `:184` | **DELETE the constant** — the committed chase replaces the centrifugal push entirely; the constant becomes vestigial. (See §7.2 and the hand-off item §9.1-4: `physics.step.service.test.ts:125` currently pins it to `1.5`.) |
| 10 | file header `:19` and the constant's own JSDoc `:180` | Refresh the prose — the header says "the `CENTRIFUGAL_MULTIPLIER = 0.3` outward push (TP-02)" while the constant is `1.5` (a pre-existing doc drift that the deletion resolves anyway). |

### 5.3 `race-session.ts` needs NO change

`host/race-session.ts:252` is `state.car = deps.stepPhysics(state.car, input);` — a **whole-record
swap**. Because `stepPhysics` returns the new `drift` channel, the state is carried automatically.
The spawn path (`:210`, `:220`) routes through `createInitialPhysicsCarState`, which now creates
the zero channel. **No change, no risk.** This is the reason the nested-channel design is safe:
the state is never field-by-field copied anywhere.

### 5.4 Why the spring must not run in the render path

`browser-entry.ts` renders at the display rate. If the spring were integrated per rendered frame,
its time constant would scale with the refresh rate: the 550 ms settle (MERGE contract (a)) would
become ~229 ms at 144 Hz — a **2.4× difference in feel between a 60 Hz and a 144 Hz display**. The
spring is therefore integrated exactly once per physics tick and the renderer only **reads** it.

### 5.5 The render bridge — one line, one scalar, unchanged contract

```ts
// browser-entry/browser-entry.ts:941
- carPoseBag.driftYawOffsetRad = car.driftYawOffset;
+ carPoseBag.driftYawOffsetRad = car.drift.pose.pose;
```

Everything downstream is untouched:

- the pose bag literal (`:855-863`, zero-init at `:862`) is unchanged — the bag still carries one
  scalar;
- `fillCarRenderPose` (`:929-937`) does not set the field, so **the write at `:941` must stay after
  it** (unchanged ordering);
- `car.pass.ts:438` (`readonly driftYawOffsetRad?: number`), `:1365` (the shadow Euler sum
  `yaw + driftYawOffset`) and `:1563-1568` (the body matrix `T · Ry(yaw) · Ry(drift)`) all consume
  the **bag** field and need no change;
- `Ry(a)·Ry(b) ≡ Ry(a+b)`, so the shadow's Euler sum and the body's matrix product remain exactly
  consistent, and AC-078 (tire-follow, 0.002 rad) is **invariant to the offset's magnitude** because
  the wheels are parented onto the body matrix (`:1586-1600`).

**Therefore `car.pass.ts` needs NO change in 14N.** The MERGE's file list included it under
"uniform feed"; on inspection the uniform feed is already correct and unchanged. The only
`car.pass` work in 14N is in **`car.pass.test.ts`** — the derived envelope limit (§8.4).

---

## 6. Collision reset (`collision.utils.ts`)

The bounce reset is `recomposeCarWithVelocity` (`:402-419`):

```ts
  if (speedMagnitude <= STANDING_VELOCITY_TOLERANCE_M_PER_S) {
    return { ...car, speed: 0, lateralVelocity: 0 };            // :409-411  (standing)
  }
  return {
    ...car,
    heading: wrapHeading(Math.atan2(velocityY, velocityX)),      // :415
    speed: speedMagnitude,                                       // :416
    slipAngle: 0,                                               // :417
    lateralVelocity: 0,                                         // :418
  };                                                             // :413-419  (moving)
```

**Both branches must reset the drift channel.** The proposal:

```ts
  if (speedMagnitude <= STANDING_VELOCITY_TOLERANCE_M_PER_S) {
    return {
      ...car,
      speed: 0,
      slipAngle: 0,          // ★ NEW — see below
      lateralVelocity: 0,
      drift: createZeroDriftChannelState(),   // ★ NEW — the pose + the commitment
    };
  }
  return {
    ...car,
    heading: wrapHeading(Math.atan2(velocityY, velocityX)),
    speed: speedMagnitude,
    slipAngle: 0,
    lateralVelocity: 0,
    drift: createZeroDriftChannelState(),     // ★ NEW — the pose + the commitment
  };
```

Three points, in order of importance:

1. **The channel reset covers the commitment as well as the pose.** A bounce that left
   `committedSlip` latched would re-enter the committed chase after the bounce, with the car
   pointing somewhere else — a "phantom slide" out of a wall. Resetting the whole channel is
   atomic and cannot half-reset.
2. **`slipAngle: 0` is added to the STANDING branch** (`:409-411`). It is currently missing: the
   standing branch zeroes `speed` and `lateralVelocity` but leaves `slipAngle` stale. The next tick
   reads the **tick-start** slip (`state.slipAngle`) to drive the speed scrub, so a stale non-zero
   slip survives into the next tick's scrub. This is a **pre-existing latent defect**, and the new
   model makes it worse, because the tick-start slip also feeds the commitment gate and the
   countersteer gate. Fixing it here is in scope: the branch is already being edited.
3. **`driftYawOffset` is not zeroed today** (neither branch) — that latent bug **disappears** by
   construction once the pose lives inside the channel and the channel is zeroed wholesale.

**Import safety:** `collision.utils.ts` currently imports only *types* from `./physics.step.service`
(`import type { PhysicsCarState, PhysicsSurfaceType }`, `:32-35`) — type-only, erased at compile
time, so there is no runtime cycle today. The new import is
`import { createZeroDriftChannelState } from './drift.model';`. **`drift.model.ts` imports
nothing at all**, so this introduces **no cycle** in either direction. (The factory lives in
`drift.model.ts`, not in `physics.step.service.ts`, specifically to keep it that way.)

**Testability note (for proposal 1/4):** the bounce path is **not wired into the live tick** —
`browser-entry.ts:83` imports only `createTrackSurfaceResolver` from `collision.utils.ts`, and the
bounce is only reachable through `resolveCarCollisions` (`:195-200`) / `resolveTrackBoundaryCar`
(`:232-248`) via the call sites at `:304-312` and `:457-460`. So the reset can only be exercised by
a **unit-level** test in `collision.utils.test.ts` — a new file that is **not** in the MERGE's file
set. **The plan author must add `examples/neatRacing/physics/collision.utils.test.ts` to the 14N
file set**, or the bounce reset ships untested. (This is a real gap in the MERGE's file list, not a
discretionary addition.)

**Reconciliation note (OBS-4 / GAP-N2 — the ratified bounce reset CLEARS the re-arm token):** the
`createZeroDriftChannelState()` wording above is the SPAWN contract, where the re-arm token starts
`true` (A-4). The landed bounce reset is deliberately NOT that channel: `collision.utils.ts` derives
it with `zeroBouncedDriftChannel()` (`:406-412`), which zeroes the pose and the committed slip and
sets `committed.reArmed = false` on BOTH branches of `recomposeCarWithVelocity` (`:434-451`). A
bounce is a RELEASE, not a spawn — the token is cleared, so the next legitimate drift has to
re-latch from scratch (the model re-arms the token when the wheel is seen neutral) and no phantom
mirrored slide can leave a wall. That is what AC-03 (ZERO-the-drift-channel) and the PASSING
`[RED-P0-3-UNCOMMITTED]` contract (`collision.utils.test.ts:585-586` — the turn sign AND the re-arm
token are cleared after a bounce) require. Where this artifact sentence and the ratified plan/RED
contract disagree, the plan and the RED test win over the artifact: the only production deviation
is that one token value, and the `createZeroDriftChannelState()` import stays cycle-free.

---

## 7. The dead export

### 7.1 `computeDriftYawOffset` — **DELETE**

`drift.model.ts:363-372`. Verified consumers: **none in production.** Its only callers are its own
test at `drift.model.test.ts:331` and `:335`. The live path has always been the inline multiply at
`physics.step.service.ts:416-417`.

**Disposition: delete the function and its test.** It is not "wired up" — there is nothing to wire it
to:

- It is a **stateless** function of `(lateralVelocity, speed, gain)`. The new pose is a **stateful**
  spring over `(pose, poseRate, articulation)`. A stateless function cannot produce it.
- It computes `atan2(lateralVelocity, speed) · gain` — i.e. exactly the old inline multiply.
  Wiring it up would *reintroduce* the old formula and defeat the slice.
- It has no third parameter that could carry the spring state, and adding one would make it a
  different function under the old name.

**The MERGE's phrasing ("wired or deleted in the same step — the natural home for the pose state")
is ambiguous between the two options; the integration evidence resolves it: the *pose state* is the
natural home for the pose state, and that is `DriftPoseSpringState` in `drift.model.ts` (§3.1). The
old function is simply obsolete.** Deleting it satisfies the MERGE's "no deferred cleanup" rule and
leaves exactly **one** visual-yaw formula in the codebase: `stepDriftPoseSpring`.

### 7.2 The wider dead-export family (the "no dual-path code" sweep)

The same sweep should dispose of the other test-only exports listed in §3.2, because leaving
`stepDriftModel` alive next to the new stateful model would be **two implementations of the drift
model** — the exact dual-path hazard the MERGE forbids for the visual half:

| Export | Disposition |
| --- | --- |
| `computeDriftYawOffset` (`:363`) | **DELETE** (§7.1) |
| `stepDriftModel` (`:140`) | **DELETE** — superseded by `stepDriftCommitment` + `computeNextLateralVelocity` |
| `createZeroDriftState` (`:124`) | **DELETE** — superseded by `createZeroDriftChannelState` |
| `DriftModelState` (`:100`) | **DELETE** — the AC-047D pin moves to the production path (§3.2) |
| `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER` (`physics.step.service.ts:184`) | **DELETE** — vestigial after the committed chase |

---

## 8. Naming

### 8.1 Grammar

`NEATRACING_<DOMAIN>_<SUBJECT>_<UNIT>` — the existing drift grammar
(`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD`, `NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD`) is kept.
Unit suffixes in use: `_RAD`, `_RAD_PER_S`, `_RAD_PER_S2`, `_PER_RAD`, `_PER_S`, `_MPS`, `_MPS2`,
`_S`, `_S2`, `_RATIO`. The new constants follow the same grammar with the `NEATRACING_DRIFT_`
prefix.

### 8.2 State and function names

| Symbol | Name | Rationale |
| --- | --- | --- |
| Physical commitment type | `DriftCommittedState` | "Committed" is the arcade-refs vocabulary; `DriftCommittedState` sorts next to `DriftPoseSpringState` alphabetically. |
| Commitment field | `committed.slip` | `committedSlip` at the call site would stutter (`state.drift.committed.committedSlip`). `state.drift.committed.slip` reads cleanly. |
| Commitment field | `committed.turnSign` | `turnSign` is self-describing with the JSDoc (`+1 = LEFT`). |
| Visual accumulator type | `DriftPoseSpringState` | **Fixed by MERGE item 2** — do not rename. |
| Visual fields | `pose`, `poseRate`, `articulation` | **Fixed by MERGE item 2.** |
| Channel type | `DriftChannelState` | "Channel" distinguishes the carrier (physical + visual) from its two members. |
| `PhysicsCarState` field | `drift` | One nested carrier. `state.drift.pose.pose` stutters once; the alternative (`driftPose` + `driftCommitted` as two flat fields) buys one less stutter at the cost of a **non-atomic reset** (§3.2). **Recommend `drift`; the plan author may overrule with a one-line change.** |
| Factory | `createZeroDriftChannelState` | Matches the existing `createZeroDriftState` naming. |
| Commitment step | `stepDriftCommitment` | Parallel to `stepDriftModel` / `stepDriftPoseSpring`. |
| Pose step | `stepDriftPoseSpring` | Parallel; the type is `DriftPoseSpringState`. |
| Grip law | `computeGripBudgetFactor` | The law changed from a saturating `1/(1+d)²` to a budget `1/(1+R·max(0,\|slip\|−T))` that is **not** speed-scaled (§4.2). |
| Countersteer law | `computeCountersteerAuthority` | The law changed from "remove velocity" to "add authority toward zero", with a turn-sign gate and m/s² units (§4.2). |
| Scrub law | `computeLongitudinalSpeedScrub` | The constant it uses is renamed to `SPEED_SCRUB_COEFF_PER_S`; the old name referenced the retired constant. |

### 8.3 REC-4 — the unit label on the countersteer authority (**cross-proposal decision**)

`COUNTERSTEER_AUTHORITY_MPS = 70` is an **acceleration** (m/s²): it is multiplied by `dt` to produce
a velocity delta in m/s. The `_MPS` suffix denotes a *speed* in this codebase
(`NEATRACING_DRIFT_MIN_SPEED_MPS = 15` is m/s), so the MERGE's name is **unit-mislabeled**.

> **Recommendation:** rename to **`NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2`** (matching the
> existing `_S2` suffix, e.g. `NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2` at `physics.step.service.ts:140`).

**This must be decided ONCE and applied in all four proposals**, or the merged plan will carry two
spellings. The same question applies to `NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2` — the MERGE's
own spelling, which uses `_S2` (stiffness is a rad/s² quantity) and is therefore **already
correct**. Recording this so the plan author applies one convention.

### 8.4 The derived envelope limit — a silent consequence of the `MAX_SLIP` change

`car.pass.test.ts:371-372` derives the envelope limit from two drift constants:

```ts
const DRIFT_VISUAL_YAW_LIMIT_RAD =
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD;
```

With `MAX_SLIP_ANGLE_RAD` changing from `π/2` to **`0.45`**, this limit changes from
`0.7854 rad` to **`0.2250 rad`** — which is *exactly* the 0.2250 limit the MERGE quotes
("max |pose| 0.2246 rad, inside the 0.2250 limit"). **This confirms the MERGE's envelope story and
ties the two proposals together** — but it also exposes a hole:

> **The derived limit bounds the SLIP term only.** The pose target also carries the articulation
> term: `poseTarget = committedSlip · GAIN + ART_RAD · articulation`, with
> `ART_RAD = POSE_STEER_YAW_RAD = 0.08`. During a **countersteer** the two terms have the **same**
> sign (see §2 INV-3/INV-4: a left turn has `committedSlip < 0`, and a countersteer drives
> `articulation < 0`), so `|poseTarget|` can reach
> `0.45 · 0.5 + 0.08 = 0.3050 rad` — **0.08 rad (4.6°) above the derived limit**.
>
> This is precisely the far-side-pop scenario the whole slice targets, so the envelope is most
> likely to be violated *at the moment the feature is working*.

**Recommendation:** re-derive the limit to include the articulation term —
`DRIFT_VISUAL_YAW_LIMIT_RAD = MAX_SLIP_ANGLE_RAD · VISUAL_YAW_GAIN_RAD + POSE_STEER_YAW_RAD`
(= `0.3050 rad`) — **rather than clamping the pose**, because clamping the pose to ±0.2250 would
fight the spring (the spring's whole job is to overshoot and ring; a clamp would flatten the pop
that is the feature). This is a **joint decision for proposals 1/4 and 2/4** — flagging it here so
the merge resolves it once, with the envelope limit derived from the same constant set that the
model uses.

---

## 9. Hand-off items for the 4-way merge

### 9.1 What this proposal expects the other proposals (and the plan author) to absorb

1. **To proposal 1/4 (RED contracts):** the AC-047D byte-pin moves from `drift.model.test.ts`
   (the `DriftModelState` mirror) to `physics.step.service.test.ts` (the production path), using the
   existing `serializeState` helper at `:72-73`. The straight-line identity must hold for the whole
   nested channel, not just for two scalars.
2. **To proposal 1/4:** `physics.step.service.test.ts:125` pins
   `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER === 1.5` — **that constant is being deleted**; the
   assertion must go, not be re-pinned.
3. **To proposal 1/4:** `drift.model.test.ts:66-82` pins the three retired constants — those
   assertions are removed, and the new constant set is pinned instead.
4. **To proposal 1/4:** `drift.model.test.ts:70` pins `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD === Math.PI/2`
   — **the value changes to 0.45**; the assertion must change.
5. **To proposal 1/4:** `physics.step.service.test.ts:156-163` (the `driftYawOffset` spawn/bounds
   test) must be re-homed to the pose (`state.drift.pose.pose`).
6. **To proposal 1/4:** the pose target's slip source (committed vs measured) must be pinned as an
   explicit RED contract — §4.4 recommends **committed** and gives the reason.
7. **To proposal 1/4 / 2/4:** the envelope limit must include the articulation term (§8.4) —
   one decision, applied once.
8. **To proposal 2/4 (re-baseline risk):** the trajectory changes by design (MERGE item 5); the
   integration-visible consequence is that the tick-start slip now feeds the **commitment gate** and
   the **countersteer gate** as well as the speed scrub, so the §6.2 stale-`slipAngle` fix in the
   standing branch is load-bearing, not cosmetic.
9. **To the plan author:** add `examples/neatRacing/physics/collision.utils.test.ts` to the 14N
   file set — the bounce reset is otherwise untested (§6).
10. **To the plan author:** `browser-entry/browser-entry.ts` enters the 14N file set for **one
    line** (§5.5) — the MERGE's file list did not include it.
11. **To the plan author:** `car.pass.ts` and `glass.pass.ts` need **no change** in 14N; only
    `car.pass.test.ts` does (§8.4). The canopy desync is a follow-up slice (§10, R-04).

### 9.2 What this proposal does NOT need from the others

The physical stage bodies (proposal 2/4), the RED assertions (proposal 1/4) and the browser fixture
(proposal 4/4) are all independent of the interfaces fixed here, **except** for the five items
above, all of which are single-line or single-assertion adjustments.

---

## 10. Risks and gaps

| ID | Risk | Impact | Mitigation / status |
| --- | --- | --- | --- |
| R-01 | **GF-15 steer chatter.** A 4 Hz square-wave steer produces a 22.6° far-side swing — a property of the *latch* design, not of the spring. | The user-visible feel can oscillate. | The MERGE assigns the mitigation decision to Phase B. The **integration** exposure is that the articulation term is what amplifies the chatter (`art` slews toward `steer`), so the slew constant (20/s) is the natural mitigation lever. **Proposal 2/4 + 1/4 own the decision; this proposal flags that the mitigation must be expressible in the constants above.** |
| R-02 | **Envelope limit vs the articulation term** (§8.4). | A 0.08 rad over-limit at the pop — the feature's best moment. | Re-derive the limit to `MAX·GAIN + POSE_STEER_YAW_RAD`. Decision needed in the merge. |
| R-03 | **Canopy desync** — `glass.pass.ts:1048` reads `car.yawRad` only, never `driftYawOffsetRad`. A 0.2250 rad (12.9°) body slide is a **12.9° canopy error**. | Visibly wrong canopy on every slide. | **Deferred to a follow-up slice (suggested 14O)**: the fix needs a new render-bag field (a render-contract change touching `car.pass.ts` + `glass.pass.ts` + the bag type), which is a different concern from the drift model. **This proposal recommends the plan author create 14O rather than widen 14N**, and that the USER visual gate (proposal 4/4) explicitly check the canopy so the gap is not shipped unnoticed. |
| R-04 | **The bounce path is not in the live tick** (§6) — `browser-entry.ts:83` imports only the surface resolver. | The reset can only be unit-tested. | New `collision.utils.test.ts` in the 14N file set (§9.1-9). |
| R-05 | **The `MAX_SLIP_ANGLE_RAD` change also changes the *published* slip bound** — the slip is now clamped to ±0.45 rad instead of `±π/2`. | `physics.step.service.test.ts:145` asserts `|slipAngle| < π/2` (still true) but the **behaviour** is materially different (the slide can no longer run away to 90°). | Intentional (MERGE). It is the mechanism that bounds the drift. Flagged for proposal 2/4's re-baseline. |
| R-06 | **`PhysicsCarState` is JSON-serialized whole** (`physics.step.service.test.ts:72-73`) and the record is compared byte-for-byte across runs. | A nested object changes the serialized shape (and the key order). | **Safe**: the determinism contract is run-A vs run-B, never a golden literal, and the new field is plain numbers in a plain object — deterministic and JSON-safe. The *shape* change only affects the AC-047D byte-pin, which is re-anchored in §3.2. |
| R-07 | **A stale `slipAngle` survives the standing bounce** (§6.2). | One tick of wrong speed scrub + a wrong commitment gate. | Fixed in the same edit (§6). |
| R-08 | **The pose spring's `dt` must be the fixed 1/60**, not a measured frame delta. | A display-rate-dependent feel (2.4× at 144 Hz). | INV-6; the spring is inside `stepPhysics` only. |
| R-09 | **`car/generate-car.ts:22-36` (`MutableNeatRacingCarRenderPose`) does not declare `driftYawOffsetRad`**, although the live bag (`browser-entry.ts:862`) does. | A pre-existing type drift: the bag is already out of sync with its declared type. | **Out of scope** — no change is proposed to the bag, so the drift is unchanged by 14N. Noted for the docs/type follow-up. |
| R-10 | **`.github/hooks/workflow-update-sync.mjs` cannot parse letter-suffixed step ids** like `14N` and silently degrades. | Plan/artifact sync may not run for 14N. | **Operational, not integration**: the orchestrator must drive the 14N slice advancement manually. Flagged to the orchestrator, not to a proposal. |
| R-11 | **The examples tree has no README** at any level and sits outside the docs generator. | No generated doc surface to refresh. | The follow-up is JSDoc-only; no docs artifact is produced by 14N. |

---

## 11. Validation evidence

| Check | Command / source | Result |
| --- | --- | --- |
| Cortex index freshness | `neataptic-gate-mcp-run_gate_check` → `cortex-index` | **PASS** |
| Folder quality | gate `folder-quality` | 4 pre-existing failures, **unrelated** to this slice |
| RAG slice context | `get_slice_context` (workflow + gate MCP) for `14N` | loaded (Step 14N entry + 3 planned slices) |
| Dead-export proof | `rg` over `examples/neatRacing/**` for `computeDriftYawOffset`, `stepDriftModel`, `createZeroDriftState` | **zero production consumers**; all test-only (§3.2 table) |
| State-record blast radius | `rg` for `driftYawOffset` and `PhysicsCarState` over `src/**` | **no matches** outside the example ⇒ the retirement is contained (§3.3) |
| Import-cycle check | read of `collision.utils.ts:32-35` | `import type` only (erased) + `drift.model.ts` has **no imports** ⇒ no cycle (§6) |
| Sign convention | read of `computeOutwardPush` (`drift.model.ts:225`), `computeSlipAngle` (`:339`), `resolveYawRatePerTick` (`physics.step.service.ts:473`) | INV-1 confirmed against the live code |
| Envelope cross-check | `car.pass.test.ts:371-372` derived limit | `0.45 × 0.5 = 0.2250 rad` — **matches the MERGE's 0.2250 limit exactly** (§8.4) |
| Tier-3 boundary map | `boundary-mapper` scout (`drift-pose-surfaces`) | line-anchored surface map returned; all anchors re-verified directly against the source before use |
| Build / type check | **NOT RUN** | this proposal changes no production file, so there is nothing to compile. The type-level risk of the proposed shape (a nested `DriftChannelState` on a JSON-serialized record) is covered by reading the two serializers (`JSON.stringify` over plain values) rather than by a compile. |

---

## 12. Certainty

| Finding | Certainty | Basis |
| --- | --- | --- |
| The dead-export family is test-only and must be deleted, not wired | **0.97** | exhaustive grep over the example + a full read of `drift.model.ts` |
| The pose spring must be integrated at the 60 Hz tick, not the render path | **0.97** | MERGE risk 8 + the tick's own `dt` constant; no ambiguity |
| The AC-047D pin is shape-sensitive and must be re-anchored on the production path | **0.93** | the serializer is a whole-object `JSON.stringify` (read at `drift.model.test.ts:44-45`); the "must re-anchor" conclusion follows from the committed model being stateful |
| The bounce reset must zero the whole channel in **both** branches, and the standing branch's missing `slipAngle: 0` is a latent defect | **0.92** | direct read of `collision.utils.ts:402-419` |
| `car.pass.ts` / `glass.pass.ts` need no change in 14N | **0.90** | the bag contract is unchanged and the glass path is a follow-up (R-03) — a judgement about slice scope, with the code path fully read |
| The `MAX_SLIP` change moves the derived envelope to 0.2250 and the articulation term breaks the derivation | **0.90** | arithmetic on the two constants + the derived limit at `car.pass.test.ts:371-372`; the "breaks the derivation" step depends on the sign of the articulation during a countersteer, which is derived from the D6 formula rather than measured |
| The committed slip (not the measured slip) should feed the pose target | **0.85** | follows from MERGE item 6's wording plus the exactness argument for the zero-straight-line identity; the alternative is defensible, so it is escalated to proposal 1/4 as a RED decision rather than settled here |
| The single nested `drift` channel over two flat fields | **0.80** | the atomic-reset argument is strong, but the `state.drift.pose.pose` stutter is a legitimate counter-argument — flagged for the plan author (§8.2) |
| The `COUNTERSTEER_AUTHORITY_MPS` unit-label defect | **0.95** | the value is multiplied by `dt` to yield m/s ⇒ it is m/s²; the naming inconsistency is objective |

**Overall certainty on this proposal: 0.92.** The interfaces, the tick order, the reset and the
dead-export disposition are read directly from the source and are certain. The two judgement calls
(the pose target's slip source and the nested-vs-flat state shape) are explicitly escalated rather
than silently decided, and the three test-file hand-offs (§9.1) are stated so proposal 1/4 can
write them without a round-trip.

**Residual gaps this proposal does not close:** the physical stage bodies (proposal 2/4), the RED
assertions (proposal 1/4), the browser fixture (proposal 4/4), and the canopy follow-up (R-03).
