# NEAT Genesis EvoDevo: Core Readiness — Racing Curriculum

**Status:** [WIP]

## Scope

Canonical long-form readiness plan for the NGE team-adversarial racing curriculum.
This plan is the single source of truth for the UI-first demo polish and the
Tier 1—6 ladder defined in `examples/racing_curriculum/reference.plans.md`.
This workstream is downstream of:

- `plans/completed/NEAT_Genesis_EvoDevo.md`
- `plans/completed/Memory_Optimization.md`
- `plans/completed/NEAT_Genesis_EvoDevo_Core_Readiness.logs.md` (prior NGE core audit archive)

If any upstream plan conflicts with this one, the upstream plan wins.

## POC framing

The current racing-curriculum browser demo, worker slices, and track physics are a
starting point and a proving ground, not a shippable benchmark. Every phase below
advances the POC toward the reference design in `examples/racing_curriculum/reference.plans.md`,
with each tier green-gated before the next tier begins. Claims of completion are
valid only when the phase's focused tests, build, and quality gates pass.

## Reference design

- `examples/racing_curriculum/reference.plans.md` defines the Tier 1—6 ladder,
  team structure, radio semantics, tire/pit design, promotion rules, carry/reset
  policy, and acceptance criteria used below.
- `examples/flappy_bird/` is the UI parity baseline for the Phase 1 demo polish.

## Current state

Claim: 01-planning @ 2026-06-24T19:53:57-04:00
Claim: 07-logging @ 2026-06-25T06:35:45-04:00 — Step 04 compressed to logs, Step 05 opened as [WIP] for user visual confirmation.
Claim: 04-implementing @ 2026-06-25T06:44:11-04:00 — Fixing Tier 1 race-pack layout to render two cars (Team A cyan / Team B magenta).
Claim: 01-planning @ 2026-06-25T07:05:05-04:00 — Step 05 marked [DONE] after browser-ui-specialist visual confirmation; Step 06 opened as [WIP] for Tier 1 contract docs.
Claim: 06-documenting — Step 06 Tier 1 contract documentation complete; validation evidence recorded; Step 07 opened as [WIP].
Claim: 05-green-testing @ 2026-06-25T18:56:48-04:00 — Step 05 green validation passed; Step 06 opened as [WIP] for Tier 2 contract documentation.
Claim: 07-logging — Step 07 compressed Phase 3 into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`, advanced Phase 4 Step 01 to [WIP], and all required gates passed.
Claim: 04-implementing @ 2026-06-25T20:00:00-04:00 — Implementing Step 09 Tier 1/Tier 2 racing baseline rules in `examples/racing_curriculum/renderer`, `browser-entry`, `track`, and `environment`.
Claim: 04-implementing @ 2026-06-25T20:44:00-04:00 — Step 09 implementation complete; all four focused Jest slices pass, build and plan validators pass; handoff to 05-green-testing for Step 10.
Claim: 05-green-testing @ 2026-06-25T20:18:31-04:00 — Step 10 green validation and regression triage passed; all focused and regression Jest slices, build, lint, and plan validators green; handoff to 06-documenting for Step 11.
Claim: 04-implementing @ 2026-08-13T22:45:00Z — Step 13 implementation slice `p3-s13-impl-renderer` complete; per-car guide lines and team-color tire trails fixed; all renderer/browser-entry tests, build, lint, tsc, and plan validators pass; handoff to 05-green-testing for `p3-s13-green-renderer`.
Claim: 04-implementing @ 2026-08-14T02:40:00Z — Step 14 slices `p3-s14-impl-physics` and `p3-s14-green-physics` complete; off-track penalty, wrong-direction detection, and car-vs-car pushing implemented in browser step and worker race-pack; all focused tests, build, lint, tsc, and plan validators pass; handoff to Step 15 planning.
Claim: 01-planning @ 2026-06-25T22:02:22-04:00 — Authoring Step 15 packet: Tier 1 default start + minimal Tier 3 4-car fallback; decision to start at Tier 1 and use `[0, 0, 1, 1]` layout recorded.
Claim: 04-implementing @ 2026-06-25T22:09:23-04:00 — Implementing slice p3-s15-impl-tier-layout: change ACTIVE_CURRICULUM_TIER default to 1 and add Tier 3 [0, 0, 1, 1] layout branch.
Claim: 04-implementing @ 2026-06-25T22:18:00-04:00 — Slice p3-s15-impl-tier-layout complete; all focused and regression Jest slices pass (69 + 249 tests), build, lint, tsc, and plan validators green; handoff to 05-green-testing for p3-s15-green-tier-layout.
Claim: 04-implementing @ 2026-06-26T15:24:25-04:00 — Slice p3-s19-impl-obs-team-offset complete; team-aware optimal-line offset implemented in observation.assembler.ts; focused Jest slice passes (8/8), tsc and lint green; handoff to 05-green-testing for p3-s19-green-obs-team-offset.
Claim: 05-green-testing @ 2026-06-26T15:27:32-04:00 — Slice p3-s19-green-obs-team-offset passed; focused observation.assembler tests (8/8), browser-entry regression (69/69), tsc (tsconfig.json + tsconfig.test.json), lint, plan-sync, and plan-phase-packets validators all green; slice marked [DONE]; handoff to p3-s19-red-browser-per-car.
Claim: 04-implementing @ 2026-06-26T16:15:37-04:00 — Slice p3-s19-impl-per-car-observation complete; exported `derivePerCarObservationState` in observation.assembler.ts; focused Jest slice 12/12, tsc, lint, plan-sync, and plan-phase-packets validators green; handoff to 05-green-testing for p3-s19-green-per-car-observation.

- **Phase 1 is [DONE].** Step 01-04 and all slices passed green validation. User confirmed the right-side network panel live-value refresh and the inner-track guidance overlay. Archive is in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- **Phase 2 — Tier 1: Single agent on simple track** is [DONE]. Step 01-07 all passed; Phase 2 history is compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- **Phase 3 — Tier 2: 1v1 with radio (one car per team)** is [DONE]. All steps (Step 08 through Step 19) passed green validation. Step 19 pivoted from shared-controller fan-out to independent per-car NEAT agents (DR-2026-06-26-01); 22 red-green slices all [DONE]; 348 tests pass, lint clean, tsc clean. Phase 3 step/slice details are compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- **Phase 4 — Tier 3: 2v2 no pits** is [PLANNED]. Step 01 — Plan Tier 3 boundary waits for user browser-demo confirmation of the Phase 3 independent-agent baseline.

- **Step 05 visual confirmation:** Browser-ui-specialist confirmed two cars render with cyan (Team A) and magenta (Team B) guiding lines, no Phase 1 regressions, and only minor viewport/alpha observations (see Step 05 evidence block).
- Tier 1—6 ladder, promotion rules, and carry/reset policy are defined in this plan and sourced from `examples/racing_curriculum/reference.plans.md`.
- Upper-tier features still depend on NGE primitives that may be experimental or missing (`ModulatorBroadcaster`, `EpisodicSlot`, `GatingRouter`, polyandric reproduction wiring). Those are routed to `nge-core-algorithm`, not compensated for locally.

### User-reported Tier 2 demo defects — 02-research findings (2026-06-25)

User observed three symptoms in the browser demo:

1. Demo loads straight into Tier 2.
2. Tier 2 shows 2 cars instead of 1.
3. Second car leaves the road without resistance and does not recover.

**02-researching investigation summary (read-only; no source edits):**

- `examples/racing_curriculum/reference.plans.md` defines Tier 2 as "2 cars (one per team), team radio active" (lines 326-334); Tier 1 is also 2 cars total (1 per team) but radio off. Therefore 2 cars in Tier 2 is correct per the reference design; the user's expectation of 1 car is not aligned with the reference plan.
- `examples/racing_curriculum/README.md` line 367 states: "The browser demo runs Tier 2 by default." `browser-entry.ts:132` hard-codes `const ACTIVE_CURRICULUM_TIER = 2;`. Loading into Tier 2 is therefore the documented demo default.
- The page subtitle "Tier 2 solo NGE harness with a spline-smoothed visual circuit and removed optimal-line guidance." is generated by `resolveStageNarrativeForTier` in `browser-entry.ts:2781-2798`. The copy incorrectly calls Tier 2 a "solo harness" and says "one car", contradicting the 1v1 reference design. This is stale/misleading UI copy, not a hard-coded tier number.
- The second car drives off track because the browser harness computes **one** controller output from the primary car's observation vector and fans the same `{throttle, steer}` to every car via `resolveControlFanOut` (`browser-entry.ts:645, 649, 2821-2829`). The controller's observation assembler reads the primary car's `carX/carY/carHeading` only (`observation.assembler.ts:700-728`), so car 1 receives the same control that was optimal for car 0's lateral position but wrong for car 1's opposite-lane start. There is no off-track resistance or recovery in the browser's local `stepEnvironment` path; `environment.step.service.ts` advances cars blindly without boundary checks (`stepCarKinematics` at `:421-452`). Off-track enforcement only exists in the worker-side `simulation-worker.race-pack.service.ts` (60-tick grace, 500-point fitness penalty, no respawn).
- `environment.step.service.ts` default initializer is a 3v3 six-car shape (`TEAM_LAYOUT = [0,0,0,1,1,1]`, `:30`), but the browser shell overrides this at runtime with `resolveCurriculumRacePackLayout`/`resolveCurriculumRacePackCars` (`browser-entry.ts:2460-2535`).
- Track geometry has 2 lanes; spawn logic places Team 0 and Team 1 on opposite lane centerlines (`browser-entry.ts:2526-2527`), but "inward/outward" assignment depends on the random loop rotation and is not deterministic.
- Pit architecture also drifts from reference: the plan says one pit per team, but `environment.step.service.ts:20-22` uses `PIT_SLOTS_PER_TEAM = 3` and `track.generator.ts` builds six pit boxes.

**Recommended fixes (targeted, no scope expansion):**

1. Correct the misleading subtitle/tooltip copy in `browser-entry.ts:2788-2798` so Tier 1/2 are described as "1v1" (two cars, one per team) rather than "solo".
2. In the browser local demo path, either (a) step each car with its own controller/observation vector, or (b) temporarily reduce the demo to a single active car and render the opponent as a non-physics placeholder until per-car controller wiring is ready. Do not paper over the missing 1v1 wiring with duplicated controls.
3. Add track-boundary resistance/penalty to `environment.step.service.ts` so cars cannot drive indefinitely off track in the local demo path; align behavior with the worker runner or the reference plan's implied constraint.
4. Fix the inward/outward lane assignment in `resolveCurriculumRacePackCars` to be rotation-invariant (e.g., use a deterministic lane index based on team rather than a fixed normal sign).
5. Reconcile pit slot count with `reference.plans.md` (1 pit per team) as part of the ongoing Tier 2/Tier 3 boundary work.

**Next agent:** 03-red-testing should write failing tests for the subtitle copy, per-car control wiring, and off-track boundary enforcement before 04-implementing changes any code.

### NGE shared-controller architectural audit — 02-research findings (2026-06-26)

**Scope:** Audit every NGE (NEAT Genesis EvoDeVo / continuous-evolution) demo in the repository for violations of the independent-agent architecture: a single shared neural-network controller fanned out to multiple visual agents. This applies to all NGE demos, not only the racing demo.

**Method:** Cortex-first search was attempted, but the Cortex MCP tools are not present in the current toolset, so the inventory used native PowerShell/view searches and three read-only specialist scouts (`boundary-mapper`, `implementation-pattern-scout`, `docs-scout`) in parallel. No production files were changed.

**Demos audited:**

| Demo                                              | Shared controller?     | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | NGE?                           | Recommended fix                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `examples/racing_curriculum`                      | **Yes**                | `browser-entry/browser-entry.ts:644-648` calls `controller.computeControlWithEvidence(envState, trackSpec)` once per tick; `:653-664` passes the single output through `resolveControlFanOut(lastControlOutput, envState.cars?.length ?? 1)` into `stepEnvironment`/`requestRacingWorkerStep`; `:2840-2848` defines `resolveControlFanOut` cloning `{throttle, steer}` to every car. Worker-side `simulation-worker.race-pack.service.ts` already evaluates a distinct `networks[carIndex].activate(...)` per car. | Yes                            | Replace the single shared `controller` in `browser-entry.ts` with one controller/network instance per visual car; build per-car observations; remove `resolveControlFanOut` with no dual path. Step 19 slices `p3-s19-red-browser-per-car` through `p3-s19-green-browser-per-car` already cover this. |
| `plans/NEAT_Genesis_EvoDeVo_AntHive_Demo.md`      | N/A (no runnable code) | No example directory exists under `examples/`                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Yes (planned only)             | Enforce the independent-agent rule in the design phase: one network/controller per ant/agent, no shared controller fan-out.                                                                                                                                                                           |
| `plans/NEAT_Genesis_EvoDeVo_PredatorPrey_Demo.md` | N/A (no runnable code) | No example directory exists under `examples/`                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Yes (planned only)             | Enforce the independent-agent rule in the design phase: one network/controller per predator/prey agent, no shared controller fan-out.                                                                                                                                                                 |
| `examples/asciiMaze`                              | No                     | One network/agent per maze episode; no multi-agent simulation                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No (classic NEAT curriculum)   | None — not an NGE demo.                                                                                                                                                                                                                                                                               |
| `examples/evolveXor`                              | No                     | One network per XOR evaluation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | No (starter NEAT)              | None — not an NGE demo.                                                                                                                                                                                                                                                                               |
| `examples/flappy_bird`                            | No                     | README/trainer state "each genome controls a bird"                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | No (classic single-agent NEAT) | None — not an NGE demo.                                                                                                                                                                                                                                                                               |
| `examples/helloNetwork`                           | No                     | Single forward-pass walkthrough                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | No (starter, no evolution)     | None — not an NGE demo.                                                                                                                                                                                                                                                                               |
| `examples/neatChat`                               | No                     | Single recurrent network per chat session                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | No (sequence-learning chat)    | None — not an NGE demo.                                                                                                                                                                                                                                                                               |
| `examples/sequenceReset`                          | No                     | Single LSTM, no agents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | No (LSTM state demo)           | None — not an NGE demo.                                                                                                                                                                                                                                                                               |

**Common shared-controller patterns found:**

- A single `controller.computeControlWithEvidence(...)` call once per tick from an aggregated environment state.
- A `resolveControlFanOut(controlOutput, carCount)` helper that clones one control output into `carCount` copies.
- Passing the cloned array to a step/evaluation function as if it were per-agent controls.
- A single mutable `Network` instance mutated by runtime adaptation in-place, shared by all visual agents.
- A single `radioChannel`/`SingleCarRadioChannel` buffer written once and implicitly shared by all cars.

**Planned-demo gaps:**

- Ant Hive and Predator/Prey NGE demos are documented in plans but have no runnable example code. When implementation starts, the first architectural gate should be a one-network-per-agent invariant.

**Next steps for 01-planning:**

1. Decide whether the current Step 19 per-car control work is sufficient to close the racing shared-controller violation, or whether a broader architectural gate (e.g., an `nge-benchmark-scout` pre-check) should be added before any future NGE demo is declared ready.
2. Update `examples/racing_curriculum/reference.plans.md` and the Ant Hive/Predator-Prey plan files to explicitly state the one-network-per-agent invariant and forbid `resolveControlFanOut`-style fan-out helpers in NGE demos.
3. When Ant Hive and Predator/Prey are implemented, make their first red tests assert independent controllers (distinct outputs for distinct agents from the start).

**Validation run during this audit:**

- `npx tsc --noEmit -p tsconfig.json` ? PASS
- `npm run lint` ? PASS
- No production files changed.

## Non-goals

- Do not hand-code queen, blocker, pacer, or pit-strategy roles.
- Do not claim Tier 4—6 completion before the required NGE primitives are confirmed
  available.
- Do not move browser rendering authority into workers; workers own simulation and
  evolution, the host owns DOM/canvas presentation.
- Do not patch missing NGE primitives locally inside the demo; record them as
  blockers and route to NGE Core.
- Do not edit generated docs/examples output directly.

## No-deferred-cleanup policy

Any migration, refactor, or API replacement in this workstream removes old code
in the same step that introduces the new code. In particular, the migration from
`src/visualization/network-view` to a local racing network visualizer must delete
the old import path and any dead host wiring in the same implementation step; no
backward-compatibility wrappers or dual-path code are permitted.

## Tier ladder summary (from reference.plans.md)

| Tier | Cars per team | Radio | Tires/pits | Track                                | Purpose                             |
| ---- | ------------- | ----- | ---------- | ------------------------------------ | ----------------------------------- |
| 1    | 1             | off   | off        | simple oval/flowing circuit          | single-car NGE learns to drive      |
| 2    | 1             | on    | off        | simple circuit with one tight corner | self-monitoring radio signal        |
| 3    | 2             | on    | off        | intermediate with overtaking zones   | first role differentiation          |
| 4    | 2             | on    | on         | intermediate with pit tradeoffs      | tire budget + pit blocking          |
| 5    | 3             | on    | on         | full competition circuit             | full NGE team racing                |
| 6    | 3             | on    | on         | full circuit, multi-window strategy  | sustained co-evolutionary arms race |

## Promotion and carry/reset semantics (from reference.plans.md)

### Promotion rules

- The ladder advances only when a team completes the current tier reliably over a
  small deterministic pack of race variants, not a single lucky race.
- **Within-team refill** (after promotion): the best-performing car becomes the
  "queen" for the next generation's polyandric reproduction; newborns receive
  queen DNA as primary template with non-overlapping drone patches from the other
  cars; newborns may receive a short driving-school warm-start.
- **Cross-team promotion**: both teams must reach promotion-threshold performance
  to advance together. A team that is far ahead holds at the current tier until the
  opponent catches up within a threshold, or until a maximum wait generation is
  reached.

### Carry-state (persisted across tier promotion)

- current weights and biases
- `GatedRecurrentCell` hidden state (slow lifetime adaptation)
- `EpisodicSlot` contents (opponent pit timing, teammate radio calibration, track danger profiles)
- developmental stage and module focus history
- `ModulatorBroadcaster` gain calibration
- team radio read/write calibration (learned radio semantic)
- other category-independent policy state

### Reset-state (at every new race start)

- world position, heading, and speed
- tire states (restored to `[1.0, 1.0, 1.0, 1.0]`)
- collision cooldowns and off-track timers
- short-horizon observation buffers (episode-scoped recurrent state)
- recent action-history buffers
- current team radio field (cleared at race start)
- other race-local episode state

## Implementation phases

### Phase 1 — Racing UI/behavior completion to Flappy Bird parity and inner-track centerline [DONE]

[DONE] Phase 1 Step 01-04 completed and validated. Detailed step/slice content, validation evidence, and PlanUpdate blocks are archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md` under "Phase 1 — Final step/slice archive (Step 01-04)".

- User confirmed the right-side network panel live-value refresh and the inner-track guidance overlay.
- All focused tests passed, bundle rebuilt, folder-quality gate passed.
- Phase 2 remains [PLANNED] and will be advanced separately by 01-planning.

### Phase 2 — Tier 1: Single agent on simple track [DONE]

[DONE] Phase 2 Step 01-07 completed and validated. Detailed step/slice content, validation evidence, and PlanUpdate blocks are archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md` under "Phase 2 — Tier 1: Single agent on simple track [DONE] — Final archive".

- Tier 1 single-agent benchmark: deterministic 2-car 1v1 pack on inner-lane centerline, worker-authoritative race episode runner, lap detection, lap-time fitness, and per-agent cyan/magenta guiding lines all passed.
- Browser-ui-specialist confirmed two cars render with cyan (Team A) and magenta (Team B) guiding lines, no Phase 1 regressions.
- Tier 1 usage contract documented in `examples/racing_curriculum/README.md`.
- Phase 3 advanced to [WIP]; Step 01 — Plan Tier 2 boundary is the active frontier.

### Phase 3 — Tier 2: 1v1 with radio (one car per team) [DONE]

```yaml
phase: 3
title: 'Tier 2: 1v1 with radio (one car per team)'
status: '[DONE]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Phase 4 — Tier 3: 2v2 no pits'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Tier 2 green gate passes: two cars (one per team) run a full episode; the self-radio field is wired as a 77-dim observation and 9-dim action seam; no pits, no tire degradation.'
  - 'Phase 2 is [DONE] before Phase 3 starts.'
placeholder_steps:
  - 'Step 09 — Implement Tier 1/Tier 2 racing baseline rules'
  - 'Step 10 — Green validation and regression triage'
  - 'Step 11 — Document Tier 1/Tier 2 baseline contract'
  - 'Step 12 — Reconcile user-reported Tier 2 demo defects and plan hardening steps'
  - 'Step 13 — Renderer hardening: guide lines + trails + header text'
  - 'Step 14 — Physics hardening: off-track penalty + wrong direction + car pushing'
  - 'Step 15 — Tier layout/start: Tier 1 default + Tier 3 fallback'
  - 'Step 16 — Document updated Tier 1/Tier 2 demo contract'
  - 'Step 17 — Logging and tracker handoff'
```

**Phase objective:** Add the team radio field with one car per team (two cars total).
Both cars learn to write and read radio as a self-monitoring signal (e.g., pace
intent, threat level). No pit/tire complexity yet.

**Stop conditions:**

- **Done:** Phase 2 is [DONE] and Tier 2 green gate passes.
- **Hold:** user must confirm radio-field dimensions and self-signal semantics.
- **Blocked:** upstream NGE primitive missing; route to `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 2 green gate (defined when phase is expanded).

#### Step 01 — Plan Tier 2 1v1 radio boundary [DONE]

- Recorded one-car-per-team/two-cars-total, radio-on/no-pits/no-tires baseline, 7-channel self-signal semantics, and authored Step 02-07 packets. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 02 — Research Tier 2 observation/action and radio contracts [DONE]

- Documented 77-in (70 base + 7 self-radio tail `[70..76]`) / 9-out (2 control + 7 radio-write) contract, traced `prepareObservationState` self-radio write/read wiring, and deferred worker-authoritative evolution protocol wiring to a later phase. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 03 — Red tests for Tier 2 1v1 pack and 77-dim observation [DONE]

- Added focused red tests covering Tier 2 pack layout (`[0, 1]`), 77-input/9-output network shape, and self-radio write split; failures were honest missing-implementation gaps. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 04 — Implement Tier 2 1v1 radio loop [DONE]

- Wired Tier 2 1v1 pack layout (`TIER_TWO_TEAM_LAYOUT = [0, 1]`), activated `ACTIVE_CURRICULUM_TIER = 2`, built 77-input/9-output MLP, split outputs into throttle/steer + 7-channel self-radio write; kept Tier 1 intact; all targeted tests passed, bundle built (732.1kb). Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 05 — Green validation and regression triage [DONE]

- Confirmed Step 04 implementation satisfies all red tests; browser-entry, controller, observation assembler, and simulation-worker race-pack tests passed; Tier 1 paths remained green; lint, build, and plan validators passed. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 06 — Document Tier 2 contract [DONE]

- Updated `examples/racing_curriculum/README.md` with Tier 2 pack layout, 77-in/9-out network shape, self-radio semantics, activation instructions, runnable TypeScript example, and feedback-loop Mermaid diagram; `npm run docs`, `npm run lint`, and plan validators passed; example validated with `tsx`. Residual tooling/gate gaps recorded as carry-forward risks in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 07 — Logging and tracker handoff [DONE]

- Compressed Phase 3 step/slice details into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`; Phase 3 marked [DONE]; Phase 4 Step 01 advanced to [WIP].


#### Step 08 — Red tests for Tier 1/Tier 2 racing baseline rules [DONE]

- Added focused red tests for renderer colors, per-car guides, lane constants, alternating pits, and boundary walls. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 09 — Implement Tier 1/Tier 2 racing baseline rules [DONE]

- Implemented the five baseline rules (renderer color/guidance, lane constants, alternating pits, boundary clamping); all four focused Jest slices passed. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 10 — Green validation and regression triage [DONE]

- Confirmed Step 09 implementation did not break existing focused tests, race-pack regressions, controller tests, or quality gates. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 11 — Document Tier 1/Tier 2 baseline contract [DONE]

- Updated `examples/racing_curriculum/README.md` with the Tier 1/Tier 2 baseline contract. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 12 — Reconcile user-reported Tier 2 demo defects and plan hardening steps [DONE]

- Scope reconciliation: seven user-reported defects assigned to Phase 3 hardening or deferred to Phase 4; Step 13–17 packets authored with red-green slices. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 13 — Renderer hardening: guide lines + trails + header text [DONE]

- Renderer hardening implemented; green validation slice `p3-s13-green-renderer` [DONE] (7 suites, 85 tests passed). Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 14 — Physics hardening: off-track penalty + wrong direction + car pushing [DONE]

- Physics hardening (off-track penalty, wrong-direction detection, car-vs-car pushing) implemented; green validation slice `p3-s14-green-physics` [DONE]. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 15 — Tier layout/start: Tier 1 default + Tier 3 fallback [DONE]

- Tier layout implemented; implementation slice `p3-s15-impl-tier-layout` [DONE] (29 suites, 249 tests), green validation slice `p3-s15-green-tier-layout` [DONE]. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 16 — Document updated Tier 1/Tier 2 demo contract [DONE]

- Updated `examples/racing_curriculum/README.md` with the updated demo contract; `npm run docs` and lint passed. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 18 — Tier 1 demo defect investigation [DONE]

- Source-grounded alignment brief identified four Tier 1 demo defects (red guide-line ignored, car overlap, yellow guide-line, cyan center divider) with file:line evidence. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 19 — Tier 1 independent-agent architecture pivot [DONE]

- Pivoted from shared-controller fan-out to independent per-car NEAT agents. All 22 red-green slices [DONE]; 348 tests pass, lint clean, tsc clean. Decision Record DR-2026-06-26-01 recorded. Detailed content archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.

#### Step 17 — Logging and tracker handoff [DONE]

- Compressed Phase 3 step/slice details into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`; Phase 3 marked [DONE]; `plans/README.md` and `plans/Roadmap.md` updated. Phase 4 remains [PLANNED] pending user browser-demo confirmation.

### Phase 4 — Tier 3: 2v2 no pits [PLANNED]

```yaml
phase: 4
title: 'Tier 3: 2v2 no pits'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Phase 5 — Tier 4: 2v2 tires and pits'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Tier 3 green gate passes: two independently-evolving teammates per team (four distinct NEAT genomes/networks) develop distinct behavioral specializations through experience; each car has its own observation, controller, and adaptation state; no pits.'
  - 'Phase 3 is [DONE] before Phase 4 starts.'
placeholder_steps:
  - 'Step 01 — Plan Tier 3 boundary'
  - 'Step 02 — Research 2v2 coevolution and role-divergence contracts'
  - 'Step 03 — Red tests for two-car team runtime'
  - 'Step 04 — Implement 2v2 worker evaluation loop'
  - 'Step 05 — Green validation and regression triage'
  - 'Step 06 — Document Tier 3 contract'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** First appearance of role differentiation. Two independently-evolving teammates per team, each with its own NEAT genome-derived network, observation, controller, and adaptation state, share only radio/team observations and a team-scoped fitness signal. They must develop distinct behavioral specializations through experience on an intermediate track with genuine overtaking zones. No pit timing complexity.

**Stop conditions:**

- **Done:** Phase 3 is [DONE] and Tier 3 green gate passes.
- **Hold:** user must confirm team-fitness semantics for 2v2.
- **Blocked:** upstream NGE primitive missing; route to `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 3 green gate (defined when phase is expanded).

#### Step 01 — Plan Tier 3 boundary [PLANNED]

```yaml
phase: 4
step: 1
title: 'Plan Tier 3 boundary'
status: '[PLANNED]'
goal: 'planning'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 02 — Research 2v2 coevolution and role-divergence contracts'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
specialists:
  - '01-planning'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Phase 4 Step 01 marked [DONE] and Step 02-07 packets exist with Tier 3 boundary decisions recorded.'
  - 'Plan-sync and plan-phase-packet validators pass after Phase 4 expansion.'
```

**User instruction:**

Expand the Phase 4 placeholder into concrete Step 02-07 packets and record the Tier 3 2v2 boundary decisions (team layout, per-car observation/action shape, role-divergence seam, team fitness semantics, and carry-forward risks). Preserve the Step 19 independent-agent contract: every car has its own NEAT genome-derived network, observation, controller, and adaptation state; teammates share only radio/team observations and a team-scoped fitness signal. Do not implement code in this step; only author the plan packets that the researching/implementing steps will execute.

**Step objective:**

Capture the first role-differentiation tier boundary: two independently-evolving teammates per team (four distinct NEAT genomes/networks), each with its own observation, controller, and adaptation state, sharing radio/team observations and a team-scoped fitness signal. No pits, intermediate track with genuine overtaking zones. Author the downstream Step 02-07 packets so the phase can advance without re-planning.

**Stop conditions:**

- **Done:** Step 02-07 packets exist, the Tier 3 boundary decisions are recorded, and plan validators pass.
- **Hold:** user must confirm team-fitness semantics for 2v2 (e.g., whether fitness is shared equally, role-weighted, or front-car biased).
- **Blocked:** upstream NGE primitive missing (`ModulatorBroadcaster`, `EpisodicSlot`, `GatingRouter`, polyandric reproduction wiring); route to `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

### Phase 5 — Tier 4: 2v2 tires and pits [PLANNED]

```yaml
phase: 5
title: 'Tier 4: 2v2 tires and pits'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Phase 6 — Tier 5: 3v3 full'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
  - 'nge-core-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Tier 4 green gate passes: tire degradation, one pit per team, pit-entrance blocking, and EpisodicSlot pit-timing motivation are observable and deterministic.'
  - 'Missing NGE primitives (EpisodicSlot, GatingRouter) are either confirmed available or escalated to nge-core-algorithm with a recorded blocker.'
  - 'Phase 4 is [DONE] before Phase 5 starts.'
placeholder_steps:
  - 'Step 01 — Plan Tier 4 boundary'
  - 'Step 02 — Research tire/pit mechanics and NGE primitive dependencies'
  - 'Step 03 — Red tests for tire/pit contracts'
  - 'Step 04 — Implement tire/pit layer or route NGE blockers'
  - 'Step 05 — Green validation and regression triage'
  - 'Step 06 — Document Tier 4 contract'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** Introduce tire degradation as metabolic budget, one pit per
team, pit-entrance blocking, and the first `EpisodicSlot` motivation (opponent
pit timing patterns). This tier may require upstream NGE primitives
(`EpisodicSlot`, `GatingRouter`); if unavailable, record the blocker and hold
rather than compensate locally.

**Stop conditions:**

- **Done:** Phase 4 is [DONE] and Tier 4 green gate passes.
- **Hold:** required NGE primitives are not yet available.
- **Blocked:** upstream NGE primitive missing and escalation unresolved; route to
  `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 4 green gate (defined when phase is expanded).

### Phase 6 — Tier 5: 3v3 full [PLANNED]

```yaml
phase: 6
title: 'Tier 5: 3v3 full'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Phase 7 — Tier 6: 3v3 advanced strategy'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Tier 5 green gate passes: three cars per team, full radio, tire/pit mechanics, queen/blocker/pacer role divergence, and polyandric reproduction active.'
  - 'Phase 5 is [DONE] before Phase 6 starts.'
placeholder_steps:
  - 'Step 01 — Plan Tier 5 boundary'
  - 'Step 02 — Research 3v3 full-team contracts'
  - 'Step 03 — Red tests for three-car team runtime'
  - 'Step 04 — Implement 3v3 full evaluation loop'
  - 'Step 05 — Green validation and regression triage'
  - 'Step 06 — Document Tier 5 contract'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** Full NGE team racing: three cars per team, full team radio,
tire degradation, one pit per team, pit-entrance blocking legal, queen/blocker/
pacer roles must emerge from experience, polyandric reproduction active.

**Stop conditions:**

- **Done:** Phase 5 is [DONE] and Tier 5 green gate passes.
- **Hold:** user must confirm reproduction policy and role-divergence observables.
- **Blocked:** upstream NGE primitive missing; route to `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 5 green gate (defined when phase is expanded).

### Phase 7 — Tier 6: 3v3 advanced strategy [PLANNED]

```yaml
phase: 7
title: 'Tier 6: 3v3 advanced strategy'
status: '[PLANNED]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Workstream closure or next benchmark'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
  - 'neatchat-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Tier 6 green gate passes: multi-generation hall-of-fame opponent snapshots, modeIsEvolvable engagement, and sustained co-evolutionary arms race observable.'
  - 'Missing NGE primitives (ModulatorBroadcaster, polyandric reproduction wiring) are either confirmed available or escalated to nge-core-algorithm with a recorded blocker.'
  - 'Phase 6 is [DONE] before Phase 7 starts.'
placeholder_steps:
  - 'Step 01 — Plan Tier 6 boundary'
  - 'Step 02 — Research analytics, radio, and NGE dependencies'
  - 'Step 03 — Red tests for analytics contracts'
  - 'Step 04 — Implement benchmark analytics'
  - 'Step 05 — Green validation and regression triage'
  - 'Step 06 — Document Tier 6 contract'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** Sustained co-evolutionary arms race: multi-generation
hall-of-fame opponent snapshots, `reproductionPolicy.modeIsEvolvable` fully
engaged, team-level fitness telemetry, and strategy-divergence analytics. This
tier may require upstream NGE primitives (`ModulatorBroadcaster`, polyandric
reproduction wiring); if unavailable, record the blocker and hold rather than
compensate locally.

**Stop conditions:**

- **Done:** Phase 6 is [DONE] and Tier 6 green gate passes.
- **Hold:** required NGE primitives are not yet available.
- **Blocked:** upstream NGE primitive missing and escalation unresolved; route to
  `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 6 green gate (defined when phase is expanded).

## Validation gates

- `plan-sync`: confirms the active [WIP] plan is registered in plan indexes.
- `step-packet`: confirms Phase 2 Step 01-07 packets are copy-pasteable and
  MCP-readable, and placeholder phases conform to schema.
- `phase-compression`: used when marking a phase [DONE] before advancing.
- `routing-table-freshness`: confirms agent/skill routing metadata is current.
- `stale-wip-plans`: used before closure or archival handoff.

## Latest validation evidence

- Phase 1 is [DONE]; archive is in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- Phase 2 is [WIP] with Step 06 (document Tier 1 contract) as the active boundary. Step 05 is [DONE] with browser-ui-specialist visual confirmation recorded; Step 04 is [DONE] and compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- Plan index (`plans/README.md`) and roadmap (`plans/Roadmap.md`) status text
  updated to reflect Phase 1 [DONE] / Phase 2 [WIP].
- `validate-plan-sync` (script): PASS — plan registered in README/Roadmap, status
  WIP, 0 errors, 0 warnings.
- `validate-plan-phase-packets` (script): PASS — Phase 2 Step 06 [WIP] (document Tier 1
  contract) conforms to schema, 0 errors, 0 warnings.
- 2026-06-25: Step 05 marked [DONE] and Step 06 opened [WIP] after browser-ui-specialist
  visual confirmation. `validate-plan-sync`: PASS (0 errors, 0 warnings).
  `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- Previous Phase 1 UI parity tranche and network-panel revisit evidence remains
  in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- 2026-06-24: Slice-fix for `workflow-update-sync.mjs` — a plan with a `[WIP]`
  phase but no step-level `[WIP]` now returns `ok: true`, `pass: true`,
  `actionTaken: between-steps`, and populated `downstreamTrackers`. Focused
  `plan-workflow` tests pass (10/10). Type check and lint pass. The sync hook
  no longer auto-advances the plan in this state.
- 2026-06-25: Step 07 completed — Phase 2 marked [DONE], compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`, Phase 3 Step 01 advanced to [WIP]. `workflow-update-sync`: pass (phase boundary detected, next boundary Phase 3 Step 01). `phase-compression.gate`: pass. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- 2026-06-26: Step 18 research completed and Step 19 implementation packet authored. Step 18 expanded to cover four Tier 1 demo defects (red guide-line ignored, car overlap, yellow guide-line, cyan center divider) with file:line evidence. Step 19 added 12 red-green slices across observation assembler, browser per-car control, environment/worker separation + grid, and renderer visual cleanup. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings). `workflow-update-sync`: ok/pass, current WIP correctly identified as Phase 3 Step 19; next PLANNED step is deferred to Step 17 closure which appears after Step 19 in the tracker.

```yaml
PlanUpdate:
  slice_id: workflow-sync-between-steps-fix
  plan: plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md
  changed_files:
    - .github/hooks/workflow-update-sync.mjs
    - scripts/agent-customization/customization-utils.mjs
    - scripts/agent-customization/plan-workflow.test.ts
    - plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
  validation:
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='scripts/agent-customization/plan-workflow'"
      expected_exit: 0
      result: '10 passed, 10 total'
    - command: 'node .github/hooks/workflow-update-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'ok: true, pass: true, actionTaken: between-steps, nextPlannedStep: Phase 1 Step 3'
    - command: 'node .github/hooks/workflow-update-sync.mjs --json --hook-check --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'ok: true, pass: true, actionTaken: between-steps'
  next: 'Hand off to 05-green-testing for any additional repo-wide validation if required.'
```

### Latest validation evidence

- 2026-06-25 (05-green-testing, `p3-s13-green-renderer`): Focused Jest slice `examples/racing_curriculum/(renderer|browser-entry)` passed — 7 suites, 85 tests. `npm run lint`, `npx tsc` (tsconfig.json + tsconfig.test.json), `npm run build:racing-curriculum` (732.9 kb bundle), `validate-plan-sync`, and `validate-plan-phase-packets` all returned PASS (0 errors/warnings). The literal `docs:folders:racing-curriculum` alias is absent from the rebuilt bundle and from `examples/racing_curriculum/index.html`. Live browser visual confirmation was attempted via `browser-ui-specialist`, but the session does not expose Chrome DevTools MCP tools, so the agent could not navigate/render the page. A headless Chrome screenshot was captured at `tmp/racing-curriculum.png` but cannot be interpreted by the orchestrator without vision tools. Step 13 remains [WIP] at the `p3-s13-green-renderer` slice pending live visual verification or resolution of the browser-tooling gap.
- 2026-06-26: Erroneous workflow sync advance (Step 13 ? [DONE], Step 14 ? [WIP]) reverted; plan restored to Step 13 [WIP], Step 14 [PLANNED]. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- 2026-06-26: Workflow sync: Advanced Phase 3 Step 12 ? [DONE]; Phase 3 Step 13 ? [WIP].
- 2026-06-25: Step 12 scope reconciliation complete — seven user-reported Tier 2 demo defects assigned to Phase 3 hardening (renderer, physics, tier layout/start) or deferred to Phase 4; `ACTIVE_CURRICULUM_TIER` default decision (Tier 1) and Tier 3 4-car fallback decision recorded; Step 13-17 packets authored with red-green slices. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings). Phase 3 remains [WIP] at Step 12; Phase 4 remains [PLANNED].
- 2026-06-25: Step 07 completed — Phase 3 marked [DONE], compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`; Phase 4 Step 01 — Plan Tier 3 boundary opened as [WIP]; `plans/README.md` and `plans/Roadmap.md` updated to Phase 4 [WIP]. `workflow-update-sync`: pass (phase boundary, Phase 4 Step 1 active). `phase-compression.gate`: pass. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- 2026-06-25: Workflow sync: Advanced Phase 3 Step 1 ? [DONE]; Phase 3 Step 2 ? [WIP]
- 2026-06-25: Step 07 completed — Phase 2 marked [DONE], compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`, Phase 3 Step 01 advanced to [WIP]. `workflow-update-sync`: pass (phase boundary detected, next boundary Phase 3 Step 01). `phase-compression.gate`: pass. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- 2026-06-25: Step 05 visual fix applied — `resolveCurriculumRacePackLayout` returns `[0, 1]` for Tier 1, producing a two-car 1v1 pack. Added regression test in `browser-entry.progression.test.ts`. Rebuilt `docs/assets/racing-curriculum.bundle.js`. Focused racing-curriculum tests: 38 suites / 247 tests PASS. Type check (`tsconfig.json`, `tsconfig.test.json`) PASS. Lint PASS.
- Claim: 04-implementing @ 2026-06-25T21:49:00-04:00 — Implementing slice `p3-s14-impl-physics` (off-track penalty, wrong-direction detection, car-vs-car pushing).
- 2026-06-25: Step 04 compressed to `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`;
  Step 05 opened as [WIP] for user visual confirmation of Tier 1 guiding lines.
  `validate-plan-sync`: PASS — 0 errors, 0 warnings.
  `validate-plan-phase-packets`: PASS — 0 errors, 0 warnings.
  `phase-compression.gate`: PASS.
- 2026-06-25: Plan sync validator pass: 0 errors, 0 warnings (after Roadmap + README update).
- 2026-06-25: Plan phase-packet validator pass: 0 errors, 0 warnings (Phase 2 [WIP], Step 04 red-green slices conform to schema).
- 2026-06-25: Workflow sync attempted to advance Phase 2 Step 4 ? [DONE]; Phase 2 Step 5 ? [WIP]. This advance was incorrect because Step 04 still has planned `p2-04-red-guiding-lines` / `p2-04-impl-guiding-lines` / `p2-04-green` slices. Statuses reverted to Step 04 [WIP], Step 05 [PLANNED].
- 2026-06-25: Workflow sync: Advanced Phase 2 Step 3 ? [DONE]; Phase 2 Step 4 ? [WIP]
- 2026-06-25: Workflow sync: Advanced Phase 2 Step 1 ? [DONE]; Phase 2 Step 2 ? [WIP]

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Context: NGE racing-curriculum workstream — Phase 1 [DONE], Phase 2 Tier 1 single-agent benchmark [DONE], Phase 3 Tier 2 1v1-with-radio [DONE], Phase 4 Tier 3 2v2 no pits [PLANNED].
Current boundary: Phase 4 Step 01 — Plan Tier 3 boundary is [PLANNED] and requires user browser-demo confirmation of the Phase 3 independent-agent baseline before advancing.

What is already covered:
- Phase 1 UI/behavior completion and inner-track centerline are [DONE], archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- Phase 2 Tier 1 single-agent benchmark is [DONE], archived in the logs file.
- Phase 3 is [DONE]: all steps (Step 08 through Step 19) passed green validation. Step 19 pivoted from shared-controller fan-out to independent per-car NEAT agents (DR-2026-06-26-01); 22 red-green slices all [DONE]; 348 tests pass, lint clean, tsc clean. Phase 3 step/slice details compressed into the logs file.
- `plans/README.md` and `plans/Roadmap.md` updated to Phase 3 [DONE] / Phase 4 [PLANNED].
- `validate-plan-sync` and `validate-plan-phase-packets` both PASS (0 errors, 0 warnings) after the planning edits.

Next narrow task: Await user browser-demo confirmation of the Phase 3 independent-agent baseline (two cars, each controlled by its own NEAT network). On confirmation, advance Phase 4 Step 01 to [WIP] and plan the Tier 3 2v2 boundary.

Known worktree cautions:
- All Phase 3 production changes are under `examples/racing_curriculum/`; no `src/` files changed.
- Cross-demo architecture note: when `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md` and `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` become active, they should import the same independent-agent contract rather than reintroduce a shared controller.
```

---
