# AUDIT-INPUT — live keyboard steering chain vs the fixture's scripted steer injection

**Artifact id:** `audit-input` (VERDICT-5 wave, plan record `14N-V5-audit-input`)
**Author:** `02-researching` (Tier 1)
**Dispatch origin:** Agent Zero (USER VERDICT 5, 2026-10-27) — audit wave `14N-V5-audit-input`
**Subject:** `examples/neatRacing` live input path — browser key handlers → steer value → drift engage/hold/commit gates — compared against `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html`
**Constraints honoured:** READ-ONLY code only (no source, fixture, plan or bundle was modified); NO browser; NO Chrome MCP; NO child agents; no implementation.

---

## 0. Provenance & method

| Item | Value |
|---|---|
| Plan | `plans/neatracing-voxel-racer.plans.md` (active phase 1 "Graphics engine + manual keyboard driving" [WIP], active step **14N** [WIP], active slice `P1-drift-feel-green`) |
| Context load | Cortex MCP `freshness_check` → `{"fresh":true,"stale":[]}`; `search_corpus` (RRF + DiskANN, `dense_state: warm`); `load_chunk` on chunk_index 604/605 = the USER VERDICT 5 record and the `14N-V5-audit-input` row |
| Plan tail | 120-line tail of `plans/neatracing-voxel-racer.plans.md` (sibling AUDIT-PHYSICS rows F-F0..F-F8, L1, R-2, R-3, R-8, MERGED AUTHOR BRIEF D_A..D_N, USER VERDICT 4, USER VERDICT 5) — read to avoid duplicating sibling rows |
| Native fallback | `view` / `grep` on known file paths only (documented degradation: Cortex `load_chunk` returns plan prose, not whole source files). All findings below were cross-checked against a second read of the same region. |
| Bundle check | `examples/neatRacing/assets/neatracing.bundle.js` and `docs/assets/neatracing.bundle.js` — 398 699 B each, written 2026-10-27 09:56, i.e. **after** the drift sources (2026-10-26 23:42 / 23:58) → the served bundle on disk IS current. The bundle source map contains the RE-ARM token and the engage-threshold JSDoc. |

**Source-of-truth order applied:** runtime/validation > static code > comments/docs > external. Sibling findings already in the plan are **cited, not re-proved** (F-F0..F-F8, L1, R-2, R-3, R-8).

---

## 1. Question

1. What does an **interactive, full-speed, held turn** actually feed into the drift model on the live path — threshold value, ramp ticks, MIN_SPEED gate direction, RE-ARM / hold semantics?
2. Do the engage gates **ever** open on the live path?
3. How does the fixture's "scripted steer injection" differ from the live path — is it a separate injection point, or the same chain under different preconditions?

---

## 2. Evidence

### 2.1 The live chain (read end to end)

```
window 'keydown'/'keyup'  (browser-entry.ts:775-798 / :801-825)
  → presses.left / presses.right                      (browser-entry.ts:775-798)
    → sampleKeyboardInput(deps.presses)              (race-session.ts:242)
      → steer = left?1:0 − right?1:0                 (input.ts:89-91, PRESSED_AXIS_RATIO=1 at :51)
        → stepPhysics(steer, throttle, brake, …)      (race-session.ts:252)
          → updateDriftChannel(...)                   (physics.step.service.ts:396-460)
            → committed.slip / committed.reArmed        (drift.model.ts:495-545)
```

### 2.2 The fixture's injection point

`docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:818-822` — `dispatchKey()` builds a **real** `KeyboardEvent{code}` and dispatches it on `window`. There is no private steer setter, no direct model poke, no mocked input module. The fixture therefore drives the **identical** live chain; only its *preconditions and phase sequence* differ (see G-7).

### 2.3 The engage gate, evaluated for a held key

| Gate | Source | Value | Held-key result |
|---|---|---|---|
| `\|steer\| ≥ STEER_ENGAGE_THRESHOLD` | `drift.model.ts:189`, tested at `:513` | 0.5 | **PASS** (live \|steer\| = 1) |
| `speed > MIN_SPEED_MPS` | `drift.model.ts:184`, tested at `:495` | 15 m/s (54 km/h) | **PASS** above 54 km/h; **FAIL** below |
| RE-ARM token `reArmed` | `drift.model.ts:464-470` (spawn), cleared at `:499`, re-latched at `:522-524` | TRUE at spawn | **PASS** on the first tick |
| `committed.slip === 0` | `drift.model.ts:509-512` | 0 at spawn | **PASS** on the first tick |

**Conclusion (2.3):** the four gates are all satisfied **on the first tick** of a held key above 54 km/h. The gates are *not* the break.

---

## 3. Numbered gaps

> Severity is my own ranking (CRITICAL = explains the user's symptom outright; HIGH = materially changes the drive feel or hides evidence; MEDIUM/LOW = latent or hygiene).
> Every row is `file:line`. Rows marked *(cited)* are already in the plan via a sibling audit and are **not** re-proved.

### G-1 — CRITICAL — the live path never feeds a non-zero steer: the boot mode is `ai`, so the keyboard is dead

- `examples/neatRacing/host/mode-selector.ts:70` — `INITIAL_RACE_MODE = MODE_OPTION_AI` → the page boots in **`ai`** mode.
- `examples/neatRacing/host/mode-selector.ts:263-267` — `select.value = mode` mirrors `ai` into the DOM select; the user *sees* a mode dropdown set to AI.
- `examples/neatRacing/host/race-session.ts:239-242` — the frame branch prefers the autopilot sample whenever `mode === 'ai'`:
  `deps.modeSelector.mode === 'ai' && deps.sampleAutoInput !== undefined ? deps.sampleAutoInput(state.car) : deps.sampleKeyboardInput(deps.presses)`
  → `sampleKeyboardInput` is **never called** in the boot mode.
- `examples/neatRacing/browser-entry/browser-entry.ts:1148` — `sampleAutoInput` is always supplied, so the AI branch is always the one taken in `ai`.
- `examples/neatRacing/browser-entry/browser-entry.ts:775-798` — the keydown handler still latches `presses.left/right`; those flags are **dead** in the boot mode.
- `examples/neatRacing/browser-entry/browser-entry.ts:323-330` — `createBootShellOverlays` never calls `setMode`, so nothing flips the mode at boot.

**Proof that the engage gates never open on the default live path:** the live steer is *exactly* `0` for every tick (G-2), so the `|steer| ≥ 0.5` gate at `drift.model.ts:513` can never be satisfied. The gates are not closed — **they are never fed**.

### G-2 — CRITICAL — the live keyboard axis is binary ±1; there is no analog ramp, so the 0.5 threshold cannot discriminate

- `examples/neatRacing/host/input.ts:51` — `PRESSED_AXIS_RATIO = 1`.
- `examples/neatRacing/host/input.ts:89-91` — `steer: (presses.left ? 1 : 0) - (presses.right ? 1 : 0)`.
- `examples/neatRacing/physics/drift.model.ts:189` — `STEER_ENGAGE_THRESHOLD = 0.5`.

The live steer can only be `0`, `+1` or `−1`. The nominal "engage band 0.4–0.6" is unreachable from a keyboard, so the threshold is a constant, not a control: a key press either fully engages or does nothing.

### G-3 — HIGH — the commit ramp is **speed-independent** and saturates in 8 ticks at any speed above 15 m/s

- `examples/neatRacing/physics/drift.model.ts:201` — `ENTRY_RATE_RAD_PER_S = 3.5`.
- `examples/neatRacing/physics/drift.model.ts:66` — `MAX_SLIP_ANGLE_RAD = 0.45` (the clamp).
- `examples/neatRacing/physics/drift.model.ts:169` — `GRIP_FALLOFF_THRESHOLD_RAD = 0.2`.
- `examples/neatRacing/physics/drift.model.ts:517-521` (engage ramp) and `:528-530` (drift ramp) — `slip -= turnSign · 3.5 · |steer| · dt`, `dt = 1/60`.

**Derived:** per tick `|Δslip| = 3.5/60 = 0.0583333 rad`. Grip falloff begins at **tick 4** (0.2 rad); the 0.45 rad clamp is reached at **tick 8** (0.133 s) — identically at 16 m/s and at 89 m/s. The live driver has no speed-dependent build-up: the slide is a step response to the key, not a curve.

### G-4 — HIGH — the MIN_SPEED gate is **not** inverted, but it is a cliff, not a taper

- `examples/neatRacing/physics/drift.model.ts:184` — `MIN_SPEED_MPS = 15` (54 km/h).
- `examples/neatRacing/physics/drift.model.ts:495-501` — `if (!(speed > 15)) { … slip = 0; turnSign = 0; reArmed = false; return …; }`.

**Derived:** the gate **opens** as the speed rises (correct direction — a slow car cannot drift) and **closes** below 15 m/s, where it also *clears the RE-ARM token* (`:499`). There is no taper: at 15.0 m/s the lateral velocity is exactly `0` (`drift.model.ts:679-681`) and one tick later it is full. The live driver therefore experiences a binary "drift on/off at 54 km/h" boundary.

### G-5 — HIGH — the RE-ARM token can only be re-latched while **no slide is latched**, and only by a fully released key

- `examples/neatRacing/physics/drift.model.ts:464-470` — the spawn channel: `reArmed: true` (wired at `physics.step.service.ts:311`).
- `examples/neatRacing/physics/drift.model.ts:509-524` — the re-latch sits **inside** `if (turnSign === 0)` and requires `|steer| < 0.15` (`:195`).
- `examples/neatRacing/physics/drift.model.ts:522-524` — the re-latch itself.

**Consequence:** a live keyboard can re-arm (steer `0 < 0.15`) but only in the window where no slide is latched. Once a slide is latched, releasing the key neither re-arms nor unwinds (G-6). A second drift therefore requires: countersteer to unwind past `SNAP_BACK_ZERO_SLIP_RAD = 0.02` (`drift.model.ts:257`) **and then** a ≥1-tick neutral-steer window.

### G-6 — HIGH — a held *or released* key never releases the slide: the only live unwind is an exact opposite-key countersteer *(cited: plan row L1 — from the input side)*

- `examples/neatRacing/physics/drift.model.ts:506` — `countersteering = turnSign !== 0 && Math.sign(steer) === -turnSign`. With `steer = 0`, `Math.sign(0) = 0 ≠ -turnSign` → `countersteering = false` → `drifting = true` (`:507`) → the ramp **keeps growing**.
- `examples/neatRacing/physics/drift.model.ts:526-530` (ramp branch) vs `:531-545` (unwind branch).
- `examples/neatRacing/physics/drift.model.ts:237` — `COUNTERSTEER_MIN_STEER = 0.9`; the keyboard's ±1 clears the magnitude bar, but the **direction** must be exactly opposite.

**Live-specific consequence:** the interactive driver's natural "lift the wheel to straighten up" gesture **keeps the slide at 0.45 rad** with its latched turn direction, and the ramp continues to push it. The only unwind is pressing the opposite key (the fixture's P2-RELEASE idiom, `p1-turn-drift-green-smoke.html:2187`).

### G-7 — CRITICAL — the fixture's "scripted steer injection" is **not a separate path**; the divergence is preconditions and phase sequence

- `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:818-822` — `dispatchKey` → `window.dispatchEvent(new KeyboardEvent('keydown', { code, bubbles: true, cancelable: true, composed: true }))` → the **live** `keydownListener` (`browser-entry.ts:775-798`) → `presses` → `sampleKeyboardInput` (`input.ts:83-92`) → steer ±1.

⇒ **The fixture exercises the identical chain.** Its green drift capture (committed slip ≈ 0.452 rad, a real speed deficit) therefore *proves the model and the chain work*. The live failure is a **precondition** failure, not a wiring failure.

The live path does **not** reproduce the fixture's preconditions:

| # | Fixture precondition | Fixture site | Live path |
|---|---|---|---|
| a | `setManualMode()` — bypasses the mode gate | `:1895`, `:2106` | user must find the dropdown (`mode-selector.ts:246-262`); boot is `ai` (G-1) |
| b | `reentryCarOnRibbon()` — two-step `ai`→`manual` toggle so the restart carrier re-seeds the car **on the ribbon** | `:2148`, `:2167`, `:2226`; impl `:1240-1250` | no equivalent: the user commits the steer **wherever the car happens to be** (typically mid-corner, often off-road) |
| c | `setFeelThrottle(true)` — throttle explicitly held | `:2149`; impl `:1212-1216` | throttle depends on the user holding **W** *at the same time* as the steer |
| d | **Uncapped** 210-tick straight accel before the commit | `:2154` (P1-STRAIGHT), `:2168-2177` (P2-ACCEL 140 + P2-COMMIT 30 + P2-HOLD 40 = 210) | no warm-up; the user's key press lands at an arbitrary speed |
| e | Capture pair taken at the **end** of a 40-tick **neutral-steer** hold (the key is released at `:2175-2177`) | `:2175-2177` | the user typically **holds** the steer into the capture |

**Derived for (d):** 210 ticks ÷ 60 × `ACCEL = 17.8` m/s² (`physics.step.service.ts:148`) → **62.3 m/s (224 km/h) on-ribbon**; the capture is therefore taken **on-road at speed**, on the axis. The live turn is taken at whatever speed the user was already doing, usually mid-corner.

**Derived for (e):** the fixture **never holds the steer** through the capture — the committed window is 30 ticks (0.5 s) and the drift capture follows 40 ticks of *released* steer. There is no fixture evidence for "steer held through the corner", which is precisely the live case.

### G-8 — HIGH — no road containment on the live path: a held full-lock turn leaves the 30 m ribbon in ≈ 0.46 s, and the off-road floor then masks the only traction-loss evidence

- `examples/neatRacing/physics/collision.utils.ts:234-256` — `resolveTrackBoundaryCar` (the boundary bounce) is referenced **only** by `collision.utils.ts` and `collision.utils.test.ts`; it is **never called** by `browser-entry.ts` or `race-session.ts`.
- `examples/neatRacing/physics/collision.utils.ts:197` — `resolveCarCollisions` (car-car) is likewise unwired.
- `examples/neatRacing/browser-entry/browser-entry.ts:733`, `:1149` — only `createTrackSurfaceResolver` (`collision.utils.ts:165-169`) is wired, i.e. the surface *type* is known but no containment is applied.
- `examples/neatRacing/physics/physics.step.service.ts:173` — `OFF_ROAD_DECEL_UNITS_PER_S = -44.5`; `:182` — `OFF_ROAD_SPEED_FLOOR_UNITS_PER_S = 22.25`; `:436-445` — the floor re-lifts the **post-scrub** speed.
- `examples/racing_curriculum/track/track.generator.ts:14-18`, `:30` — the large bucket is radius 240 m (±18 %) with a 30 m road (±10 %).

**Derived:** a full-lock turn has a **speed-independent radius of 44.5 m** (yaw ∝ `speedPercent`, speed ∝ `speedPercent`, `physics.step.service.ts:192` `STEERING_RATE = 2`) against a ~240 m corner radius and a 15 m half-width — so a held full lock at top speed leaves the ribbon in ≈ **28 ticks (0.46 s)**. Off-road the speed settles at the 22.25 m/s floor (80.1 km/h), where the scrub `0.55·|slip|·v·(v/89)` (`drift.model.ts:264`, applied at `physics.step.service.ts:419-427`) collapses to ≈ **1.4 m/s²** instead of ≈ 20 m/s² on-road, and is then re-lifted by the floor. *(Cross-ref sibling R-3.)*

The fixture documents this itself as the reason it needed the re-entry: `p1-turn-drift-green-smoke.html:1218-1239` — "the recorded browser lane enters the 14N block **OFF-ROAD** … an off-road car can never produce the ON-ROAD straight capture the D1.3 capture pair needs — off-road the speed settles at the 22.25 m/s floor (80.1 km/h)".

### G-9 — MEDIUM — the bounce reset of the drift channel is never wired, so the "a bounce releases the slide" rule does not exist at runtime

- `examples/neatRacing/physics/collision.utils.ts:406-450` — `zeroBouncedDriftChannel()` clears the channel and the RE-ARM token; referenced **only** inside `collision.utils.ts`.
- `examples/neatRacing/physics/drift.model.ts:455-462` — the model documents the bounce release that the runtime never performs.

Not the live break (the live path never bounces, because the boundary bounce is unwired — G-8), but it is an unwired published contract, and it means the live RE-ARM token can only ever be cleared by the MIN_SPEED cliff (G-4).

### G-10 — MEDIUM — input latching hazards that the fixture structurally cannot catch

- `examples/neatRacing/host/race-session.ts:236-243` — the input is sampled **once per rAF frame** and replayed to every catch-up tick of that frame; `examples/neatRacing/host/input.ts:83-92` reads the latched booleans.
- `examples/neatRacing/browser-entry/browser-entry.ts:775-798` / `:801-825` — the booleans are latched on keydown/keyup only: **no `blur` handler, no `visibilitychange` handler**, no `event.repeat` handling.

**Consequences:** (a) a press-and-release inside one frame (≤ ~16 ms, or up to 100 ms of catch-up per `host/render-loop.utils.ts:48`) is **invisible** to the physics; (b) if the window loses focus while a key is held, the keyup is never delivered and `presses.left` stays TRUE → a **permanent full-lock steer** with no way to recover except clicking back into the page. The fixture always dispatches and releases keys in order, so neither hazard is exercised.

### G-11 — MEDIUM — the HUD advertises a control the boot mode ignores, and there is no traction readout at all

- `examples/neatRacing/host/hud.ts:132` — the controls hint is `'WASD / ARROWS to drive'`, mounted as a static label (`hud.ts:392-393`).
- `examples/neatRacing/host/mode-selector.ts:70` — the boot mode is `ai`, so the hint is **false on load** (G-1).
- `examples/neatRacing/host/hud.ts:383-393` — the HUD mounts exactly four labels (speed, lap, time, hint). There is **no slip / traction / grip indicator**, so the only user-visible traction evidence is the raw speed number, and the user cannot correlate it with the slide.

### G-12 — MEDIUM — in the default `ai` mode the autopilot's own steer barely clears the engage threshold, and its throttle/brake schedule fights the slide

- `examples/neatRacing/physics/ai.autopilot.ts:167` — `steer: clampSteer(headingError * NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD)`; gain `2.2` at `:31`.
- `examples/neatRacing/physics/ai.autopilot.ts:160-162` (throttle cut above 0.35 rad), `:163-165` (brake above 0.7 rad).

**Derived:** `|steer| ≥ 0.5` requires `|headingError| ≥ 0.2273 rad`; the AI cuts the throttle above 0.35 rad and brakes above 0.7 rad. In the boot mode the drift gate therefore opens only in a narrow error band in which the car is simultaneously coasting or braking — any slide seen in `ai` is a transient artifact of the AI's own cornering error, not a driver-visible drift.

### G-13 — LOW — the served bundle's cache-buster predates the entire drift wave

- `examples/neatRacing/index.html:95-99` — `bundlePath` is built with the frozen query `?v=20260928-1`; `:119` — `script.src = bundlePath`.

The file on disk is current (398 699 B, 2026-10-27 09:56, after the 2026-10-26 sources; the source map contains the RE-ARM token and the engage-threshold JSDoc), so this is **not** the break — but any browser holding the pre-drift response for the same URL keeps serving the old bundle. Confidence that this *contributed* to the user's observation: **0.40** (no browser was run; cache state is unverifiable by read-only code).

### G-14 — LOW — the fixture could not read the HUD speed and fell back to the debug seam

- `docs/browser-tests/scenarios/p1-turn-drift-green-smoke.html:1465-1467` — "the HUD text read returned 0 at the capture points in the recorded runs" → the fixture fell back to the `window.neatRacingDriftState` seam.
- `examples/neatRacing/browser-entry/browser-entry.ts:980-986` — the 5-field debug seam (`slip`, `speed`, `reArmed`, `phase`, …).
- `examples/neatRacing/host/hud.ts:335-342` — the HUD labels carry **no DOM id**, so the fixture had to reach them by structural query.

This corroborates G-11: the live HUD is not a reliable traction observation surface, so "the tires never lose traction" is unobservable in the live build without the debug seam.

---

## 4. Decision

**What an interactive full-speed held turn feeds:** `steer = ±1` (binary, `input.ts:51`, `:89-91`), sampled once per frame (`race-session.ts:236-243`) and replayed to each catch-up tick. Above 15 m/s the engage gates are all satisfied on the **first tick**; the slip then ramps at **0.0583333 rad/tick** (3.5 rad/s × 1/60, `drift.model.ts:201`, `:517-521`, `:528-530`) to the 0.45 rad clamp in **8 ticks / 0.133 s**, independent of speed; the RE-ARM token is **not** re-latchable while the slide is latched (`:509-524`); and the slide is unwound **only** by an exact opposite-key press (`:506`, `:237`).

**Do the engage gates ever open on the live path?** **No — not on the default path, and not because a gate is closed.** The page boots in `ai` (`mode-selector.ts:70`) and the frame branch prefers the autopilot sample (`race-session.ts:239-242`), so `sampleKeyboardInput` is never called and the live steer is exactly `0` for every tick — the `|steer| ≥ 0.5` gate at `drift.model.ts:513` is never fed. In `manual` mode the gates **do** open on the first tick above 54 km/h; the live failure then becomes a *feel + evidence* failure (speed-independent 8-tick snap to 0.45 rad, no re-arm while latched, release only by opposite key, and an off-road floor that masks the speed scrub).

**Live vs fixture:** the fixture's steer injection is **not** a separate path — `dispatchKey` (`p1-turn-drift-green-smoke.html:818-822`) dispatches a real `KeyboardEvent` into the live listener. The divergence is entirely in **preconditions and phase sequence** (G-7 a–e): manual mode, an on-ribbon re-seed, an explicit throttle, a 210-tick uncapped straight to 62.3 m/s, and a **neutral-steer** capture window. The live user has none of these, and the live user **holds** the steer through the corner — a case the fixture never covers.

**Merge routing:** the plan file is **not** edited by this audit (Agent Zero owns the merge). This artifact is the deliverable.

---

## 5. Risks / gaps in this audit

- **R-A (single-context, no cross-check).** The user's "NO child agents" order overrides the generic Tier-1 delegation mandate, so this audit was not independently re-checked by a second scout. A missed gap in the live input path is possible. *(Same disclosure shape as the sibling's R-8.)*
- **R-B (no runtime evidence).** No browser was run, so every claim here is static. The "gates never open" conclusion is a control-flow proof (mode → branch → steer value), not a captured frame.
- **R-C (cache state unverifiable).** G-13's contribution to the user's observation is confidence 0.40; the on-disk bundle is current, so a stale cached bundle is *possible* but unproven.
- **R-D (boundary respected).** The camera aim law, the wheel-spin idiom, the `browser-entry` `stepPhysics` wrapper, the numeric envelope and the presentation inventory are owned by AUDIT-CAMERA / AUDIT-MOTION / AUDIT-ENVELOPE / AUDIT-VISUALS and are **not** re-litigated here.
- **R-E (no duplication).** F-F0..F-F8, L1, R-2, R-3 and the MERGED AUTHOR BRIEF D_A..D_N were read from the plan tail and are cited rather than re-proved; G-6 and G-8 are the input-side views of L1 and R-3 respectively.
