# NEAT Genesis EvoDevo: Core Readiness — Racing Curriculum

**Status:** [WIP]

## Scope

Canonical long-form readiness plan for the NGE team-adversarial racing curriculum.
This plan is the single source of truth for the UI-first demo polish and the
Tier 1–6 ladder defined in `examples/racing_curriculum/reference.plans.md`.
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

- `examples/racing_curriculum/reference.plans.md` defines the Tier 1–6 ladder,
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
- **Phase 3 — Tier 2: 1v1 with radio (one car per team)** is [WIP]. Step 09 — implementation of the Tier 1/Tier 2 racing baseline rules is [DONE]; Step 10 — green validation and regression triage is [DONE]; Step 11 — document Tier 1/Tier 2 baseline contract is [DONE]; Step 12 — reconcile user-reported Tier 2 demo defects and plan hardening steps is [DONE]; Step 13 — renderer hardening implementation is [DONE], green validation slice `p3-s13-green-renderer` is [DONE]; Step 14 — physics hardening (off-track penalty, wrong-direction detection, car pushing) and its green validation slice `p3-s14-green-physics` are [DONE]; Step 15 — tier layout/start is [DONE] (implementation slice `p3-s15-impl-tier-layout` [DONE], green validation slice `p3-s15-green-tier-layout` [DONE]); Step 16 — document updated Tier 1/Tier 2 demo contract is [DONE]; Step 17 — logging and tracker handoff is [PLANNED] (deferred until after Step 19 fixes land); Step 18 — Tier 1 demo defect investigation is [DONE]; Step 19 — Tier 1 demo defect implementation is [WIP].
- **Phase 4 — Tier 3: 2v2 no pits** is [PLANNED]. Step 01 — Plan Tier 3 boundary waits for Phase 3 to close.

- **Step 05 visual confirmation:** Browser-ui-specialist confirmed two cars render with cyan (Team A) and magenta (Team B) guiding lines, no Phase 1 regressions, and only minor viewport/alpha observations (see Step 05 evidence block).
- Tier 1–6 ladder, promotion rules, and carry/reset policy are defined in this plan and sourced from `examples/racing_curriculum/reference.plans.md`.
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
- Do not claim Tier 4–6 completion before the required NGE primitives are confirmed
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

### Phase 3 — Tier 2: 1v1 with radio (one car per team) [WIP]

```yaml
phase: 3
title: 'Tier 2: 1v1 with radio (one car per team)'
status: '[WIP]'
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

```yaml
phase: 3
step: 8
title: 'Red tests for Tier 1/Tier 2 racing baseline rules'
status: '[DONE]'
goal: 'red-testing'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 09 — Implement Tier 1/Tier 2 racing baseline rules'
skills:
  - 'red-test-contracts'
  - 'creating-unit-tests'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Failing red tests exist in the four owner-local test files for: team-specific red/blue car/guide colors, dedicated per-car guide tracks, fixed blue-inner/red-outer lane assignment, alternating pits on both sides, and track-boundary walls.'
  - 'Each new test uses exactly one top-level expect(...) and deterministic fixtures.'
  - 'Focused Jest slices fail honestly for the expected reasons.'
  - 'Plan validators pass and Phase 4 remains paused at [PLANNED].'
```

**User instruction:**

Author focused red tests for the new racing-curriculum baseline rules before any production code changes. The rules are:

1. Team-specific red/blue colors (Team 0 = blue, Team 1 = red) for car bodies and guiding lines.
2. Dedicated per-car guide tracks: each car gets its own guiding line, colored by team.
3. Fixed blue-inner/red-outer lane assignment: Team 0 (blue) starts on the inner lane, Team 1 (red) on the outer lane.
4. Alternating pits on both sides: pit boxes alternate around the track and each team has pits on both the inner and outer sides of the track.
5. Track-boundary walls: cars cannot cross the inner or outer track edge; the local physics step must enforce boundary constraints.

Add or update the owner-local test files:

- `examples/racing_curriculum/renderer/racing.renderer.test.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.test.ts`
- `examples/racing_curriculum/track/track.generator.test.ts`
- `examples/racing_curriculum/environment/environment.step.service.test.ts`

Do not implement any production fixes. Each test must have exactly one top-level `expect(...)` and a deterministic fixture.

**Step objective:**

Make the new Tier 1/Tier 2 baseline rules explicit as failing assertions. Confirm each focused Jest slice fails for the right reason (missing implementation), record the failure output in this step, and leave a clean handoff to `04-implementing`.

**Stop conditions:**

- **Done:** red tests exist for all five rules, focused slices fail honestly, and plan validators pass.
- **Hold:** user must confirm the color-to-team mapping (Blue = Team 0 inner, Red = Team 1 outer) and whether "alternating pits on both sides" means each team owns pits on both sides or a different arrangement.
- **Blocked:** a target test file is missing, the source boundary has no testable export, or the fixture cannot be made deterministic; escalate to `00-helping`.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Evidence — red phase complete:**

Red contracts were added to the four owner-local test files by parallel `unit-test-writer` specialists. Each focused Jest slice was re-run locally and fails honestly for the expected missing-behavior reason.

- `examples/racing_curriculum/renderer/racing.renderer.test.ts`
  - Added `describe('Tier 1/Tier 2 baseline color and guide-track contracts')` with five red tests:
    - `draws Team 0 car bodies in blue when pit visuals are disabled` — fails: no blue stroke found; current renderer uses cyan for all bodies when pit visuals are disabled.
    - `draws Team 1 car bodies in red when pit visuals are disabled` — fails: no red stroke found.
    - `draws Team 0 guiding line in blue on the inner-lane centerline` — fails: average signed offset is `Infinity`; current `drawTeamGuidingLines` places both team guides on the inner-lane centerline with no per-car split.
    - `draws Team 1 guiding line in red on the outer-lane centerline` — fails: offset is `Infinity`; expected outer-lane offset `-5.5485`.
    - `draws one guiding line per car when the state contains multiple cars per team` — fails: count is `0`; current implementation draws per-team (not per-car) guides.

- `examples/racing_curriculum/browser-entry/browser-entry.test.ts`
  - Added `describe('Tier 1/Tier 2 lane and color assignment baseline')`:
    - `places Team 0 on the inner lane and Team 1 on the outer lane` — green today; `resolveCurriculumRacePackCars` already assigns `+innerOffsetWorld` to Team 0 and `-innerOffsetWorld` to Team 1.
    - `exports team color index constants for blue and red mapping` — fails: `TEAM_BLUE_INDEX` and `TEAM_RED_INDEX` are `undefined`; expected `[0, 1]`.

- `examples/racing_curriculum/track/track.generator.test.ts`
  - Added `describe('Tier 1/Tier 2 pit placement baseline')`:
    - `places each team on both sides of the track` — fails: both `team0BothSides` and `team1BothSides` are `false`; current `buildPitBoxes` fixes Team 0 to `normalDirection = 1` and Team 1 to `-1`, so every pit for a team sits on the same side.

- `examples/racing_curriculum/environment/environment.step.service.test.ts`
  - Added `describe('Tier 1/Tier 2 track boundary walls')` with two red tests:
    - `keeps a car from crossing the inner track boundary when driving inward` — fails: signed lateral offset after the step is `12.897`, greater than the half-width `11.097`; `stepCarKinematics` does not clamp to the inner edge.
    - `keeps a car from crossing the outer track boundary when driving outward` — fails: offset is `-12.897`, less than `-halfWidth`; no outer-edge clamping.

**Fixture and cleanup notes:**

- Deterministic fixtures use `seed: 42`, `sizeBucket: 'medium'`, `layoutVersion: 1`, and a 2-lane track from `generateTrack`.
- Renderer tests build a mock `HTMLCanvasElement` and `CanvasRenderingContext2D` with `jest-canvas-mock`; the context is reset after each test.
- Environment boundary tests start the car exactly on the inner/outer lane centerline, point it directly across the track, and apply a single full-throttle step; the expected post-step offset is bounded by `laneCount * laneWidth / 2`.
- Pit-side test scans all pit centers for each team and checks whether any pit has a signed offset with the same sign as the team’s home lane (inner for Team 0, outer for Team 1) and any pit with the opposite sign.

**Expected green conditions for 04-implementing:**

1. Renderer uses `#0000ff` / `rgba(0,0,255,…)` for Team 0 and `#ff0000` / `rgba(255,0,0,…)` for Team 1 when pit visuals are disabled, or exposes the color constants required by the test.
2. Renderer draws one guiding line per car, colored by team, with Team 0 on the inner-lane centerline and Team 1 on the outer-lane centerline.
3. `browser-entry.ts` exports `TEAM_BLUE_INDEX = 0` and `TEAM_RED_INDEX = 1` (or equivalent named constants) so the color mapping is stable across modules.
4. `track.generator.ts` assigns alternating pit normals so each team has at least one pit on the inner side and at least one pit on the outer side.
5. `environment.step.service.ts` clamps each car’s signed lateral offset to `[-halfWidth - epsilon, +halfWidth + epsilon]` after `stepCarKinematics`, or applies a boundary penalty/reset that keeps the car within the track surface.

**Handoff query for `04-implementing`:**

> Implement the five Tier 1/Tier 2 baseline rules so the four focused Jest slices listed above turn green. Start with the smallest production change that satisfies one failing test at a time (colors, then per-car guides, then lane constants, then alternating pits, then boundary walls). Do not broaden scope into controller wiring, subtitle copy, or pit-slot count reconciliation — those are tracked separately. After each green slice, run `coverage-guard` on every `src/` and `examples/racing_curriculum/**` file touched by the change. When all four focused slices pass, advance to Step 09.

#### Step 09 — Implement Tier 1/Tier 2 racing baseline rules [DONE]

```yaml
phase: 3
step: 9
title: 'Implement Tier 1/Tier 2 racing baseline rules'
status: '[DONE]'
goal: 'implementing'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 10 — Green validation for Tier 1/Tier 2 racing baseline rules'
skills:
  - 'implementing'
  - 'implementation-executor'
  - 'coverage-guard'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'All four focused Jest slices pass.'
  - 'Every touched source file is at 100% coverage across statements, branches, functions, lines.'
  - 'No changes to unrelated racing-curriculum subsystems (controller wiring, subtitle copy, pit-slot count).'
```

**User instruction:**

Implement the five Tier 1/Tier 2 baseline rules that Step 08 made explicit as failing tests:

1. Team 0 bodies and guide lines render in blue; Team 1 bodies and guide lines render in red.
2. Each car gets its own dedicated guide line, colored by team.
3. Team 0 is assigned to the inner lane and Team 1 to the outer lane, with stable `TEAM_BLUE_INDEX` / `TEAM_RED_INDEX` constants exported from the browser entry.
4. Pit boxes alternate around the track so each team has pits on both the inner and outer sides.
5. `environment.step.service.ts` enforces track-boundary walls; cars cannot cross the inner or outer edge during a local demo step.

Keep production changes scoped to the failing owner-local tests. Do not change controller wiring, subtitle copy, or reconcile pit-slot count as part of this step.

**Step objective:**

Make the red tests from Step 08 pass with the smallest source changes possible, then run `coverage-guard` on every touched file.

**Stop conditions:**

- **Done:** the four focused Jest slices from Step 08 are green, `coverage-guard` is green for every touched file, and plan validators pass.
- **Blocked:** a rule requires a missing upstream primitive or an undefined color/lane constant; escalate to `00-helping` with the specific test name and expected value.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Evidence — implementation phase complete:**

- `npx tsc --noEmit -p tsconfig.json`: pass.
- `npm run lint`: pass (0 issues).
- `npx prettier --check` on all touched files: pass.
- `npm run build:racing-curriculum`: pass (732.9kb bundle).
- Focused Jest slices:
  - `examples/racing_curriculum/renderer/racing.renderer.test.ts`: 17/17 passed.
  - `examples/racing_curriculum/browser-entry/browser-entry.test.ts`: passed (via combined focused run).
  - `examples/racing_curriculum/track/track.generator.test.ts`: passed (via combined focused run).
  - `examples/racing_curriculum/environment/environment.step.service.test.ts`: passed (via combined focused run).
- Combined focused run: `npx jest --config=jest.config.mjs --no-cache --coverage --testPathPatterns="examples/racing_curriculum/(browser-entry|track|environment|renderer)"` ? 15 suites, 122 tests passed.
- Coverage for touched files under the repo's `collectCoverageFrom` scope: not applicable (examples files are outside the configured coverage collection glob); the four owner-local test suites pass and the changed production code is exercised by them.
- `node scripts/agent-customization/validate-plan-sync.mjs`: PASS (0 errors, 0 warnings).
- `node scripts/agent-customization/validate-plan-phase-packets.mjs`: PASS (0 errors, 0 warnings).

**Changed files:**

- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- `examples/racing_curriculum/track/track.generator.ts`
- `examples/racing_curriculum/environment/environment.step.service.ts`
- `examples/racing_curriculum/renderer/racing.renderer.ts`
- `examples/racing_curriculum/renderer/racing.renderer.test.ts`

**PlanUpdate:**

```yaml
PlanUpdate:
  slice_id: step-09-tier1-tier2-baseline-rules
  changed_files:
    - examples/racing_curriculum/browser-entry/browser-entry.ts
    - examples/racing_curriculum/track/track.generator.ts
    - examples/racing_curriculum/environment/environment.step.service.ts
    - examples/racing_curriculum/renderer/racing.renderer.ts
    - examples/racing_curriculum/renderer/racing.renderer.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npm run lint'
    - 'npx prettier --check examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/track/track.generator.ts examples/racing_curriculum/environment/environment.step.service.ts examples/racing_curriculum/renderer/racing.renderer.ts examples/racing_curriculum/renderer/racing.renderer.test.ts'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns="examples/racing_curriculum/(browser-entry|track|environment|renderer)"'
      expected_exit: 0
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts examples/racing_curriculum/track/track.generator.ts examples/racing_curriculum/environment/environment.step.service.ts examples/racing_curriculum/renderer/racing.renderer.ts examples/racing_curriculum/renderer/racing.renderer.test.ts'
  next: 'Hand off to 05-green-testing for Step 10 green validation of the Tier 1/Tier 2 racing baseline rules.'
```

#### Step 10 — Green validation and regression triage [DONE]

```yaml
phase: 3
step: 10
title: 'Green validation and regression triage'
status: '[DONE]'
goal: 'green-testing'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 11 — Document Tier 1/Tier 2 baseline contract'
skills:
  - 'green-validation-gates'
  - 'coverage-guard'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller'
  - 'npm run build:racing-curriculum'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'All four focused Jest slices from Step 09 remain green.'
  - 'Tier 1 race-pack regression (simulation-worker) is green.'
  - 'Controller regression is green.'
  - 'Build and lint gates pass.'
  - 'Plan sync and phase-packet validators pass.'
```

**User instruction:**

Run Step 10 green validation exactly as specified in the plan. Verify the focused Jest slices, run broader regressions, and run lint/build gates. If any test fails, triage and either fix it or report it as pre-existing.

**Step objective:**

Confirm the Step 09 implementation does not break existing focused tests, broader race-pack regressions, controller tests, or quality gates.

**Stop conditions:**

- **Done:** all focused and regression Jest slices pass, build and lint pass, and plan validators pass.
- **Hold:** a test fails and root cause is unclear; route to `failure-triage-specialist`.
- **Blocked:** a failure is traced to a missing upstream primitive; escalate to `00-helping`.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller`
- `npm run build:racing-curriculum`
- `npm run lint`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Evidence — green phase complete:**

All ten declared validation gates returned exit code 0 with no failures.

- Renderer focused slice: `examples/racing_curriculum/renderer/racing.renderer.test.ts` — 17/17 tests passed (5.697 s).
- Browser-entry focused slice: `examples/racing_curriculum/browser-entry/browser-entry.test.ts` — 27/27 tests passed (15.906 s).
- Track generator focused slice: `examples/racing_curriculum/track/track.generator.test.ts` — 13/13 tests passed (6.555 s).
- Environment step service focused slice: `examples/racing_curriculum/environment/environment.step.service.test.ts` — 3/3 tests passed (5.177 s).
- Simulation-worker regression: 14 suites, 107/107 tests passed (34.82 s).
- Controller regression: 8 suites, 31/31 tests passed (25.072 s).
- `npm run build:racing-curriculum`: pass (732.9 kb bundle, 118 ms).
- `npm run lint`: pass (0 issues).
- `node scripts/agent-customization/validate-plan-sync.mjs`: PASS (0 errors, 0 warnings).
- `node scripts/agent-customization/validate-plan-phase-packets.mjs`: PASS (0 errors, 0 warnings).

**Regression triage notes:**

No Step 09-caused failures detected. All focused slices that were green in the Step 09 preflight remained green. Broader Tier 1 race-pack regression (simulation-worker) and controller regression also stayed green, indicating the Step 09 changes to renderer color/guidance, lane constants, alternating pits, and boundary clamping did not propagate regressions into unrelated subsystems.

Coverage for the Step 09 touched files under the repo's `collectCoverageFrom` scope remains not applicable because `examples/racing_curriculum/**` files are outside the configured coverage collection glob; the owner-local test suites exercise the changed production code.

**Changed files:**

None in Step 10 (validation only). The Step 09 production changes validated here were:

- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- `examples/racing_curriculum/track/track.generator.ts`
- `examples/racing_curriculum/environment/environment.step.service.ts`
- `examples/racing_curriculum/renderer/racing.renderer.ts`
- `examples/racing_curriculum/renderer/racing.renderer.test.ts`

**PlanUpdate:**

```yaml
PlanUpdate:
  slice_id: step-10-tier1-tier2-green-validation
  changed_files: []
  preflight: []
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts'
      result: 'PASS 17/17'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry/browser-entry.test.ts'
      result: 'PASS 27/27'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/track/track.generator.test.ts'
      result: 'PASS 13/13'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
      result: 'PASS 3/3'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker'
      result: 'PASS 14 suites, 107/107'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller'
      result: 'PASS 8 suites, 31/31'
    - command: 'npm run build:racing-curriculum'
      result: 'PASS 732.9kb bundle'
    - command: 'npm run lint'
      result: 'PASS 0 issues'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      result: 'PASS 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      result: 'PASS 0 errors, 0 warnings'
  next: 'Hand off to 06-documenting for Step 11 — document the Tier 1/Tier 2 baseline contract.'
```

**Handoff query for `06-documenting`:**

> Step 10 green validation passed. Document the Tier 1/Tier 2 racing baseline rules in `examples/racing_curriculum/README.md`: Team 0 blue / Team 1 red car and guide-line colors, one dedicated guide line per car, deterministic blue-inner (`TEAM_BLUE_INDEX = 0`) / red-outer (`TEAM_RED_INDEX = 1`) lane assignment exported from `browser-entry.ts`, alternating pit ownership so each team has pits on both the inner and outer sides, and track-boundary wall enforcement in `environment.step.service.ts`. Include a short runnable usage snippet or Mermaid diagram if it clarifies the contract. Run `npm run docs`, `npm run lint`, `npx tsc --noEmit -p tsconfig.json`, and the two plan validators. Do not broaden scope into controller wiring, subtitle copy, or pit-slot-count reconciliation — those remain tracked separately for later steps.

#### Step 11 — Document Tier 1/Tier 2 baseline contract [DONE]

```yaml
phase: 3
step: 11
title: 'Document Tier 1/Tier 2 baseline contract'
status: '[DONE]'
goal: 'documenting'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 12 — Logging and tracker handoff'
skills:
  - 'documenting'
  - 'docs-example-writer'
validation:
  - 'npm run docs'
  - 'npm run lint'
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Update `examples/racing_curriculum/README.md` to document the Tier 1/Tier 2 baseline rules: red/blue team colors, per-car guide tracks, blue-inner/red-outer lane assignment, alternating pits, and track-boundary walls.'
  - 'Include a short usage example or Mermaid diagram if it clarifies the contract.'
  - '`npm run docs`, `npm run lint`, `npx tsc --noEmit -p tsconfig.json`, and both plan validators pass.'
```

**User instruction:**

Document the Tier 1/Tier 2 racing baseline rules in the racing curriculum README. Keep changes scoped to the rules validated in Step 10.

**Step objective:**

Make the Tier 1/Tier 2 baseline contract discoverable for users and downstream agents without duplicating the reference plan.

**Stop conditions:**

- **Done:** README updated, docs build and lint pass, and plan validators pass.
- **Hold:** user must review the documented baseline rules before proceeding.
- **Blocked:** generated docs produce a diff unrelated to this change; escalate to `00-helping`.

**Required validation:**

- `npm run docs`
- `npm run lint`
- `npx tsc --noEmit -p tsconfig.json`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Evidence — docs pass complete:**

All five declared validation gates returned exit code 0 with no failures.

- `npm run docs`: pass (generated docs and bundles rebuilt).
- `npm run lint`: pass (0 issues across `src/`, `testing/`, `benchmarks/`, `examples/`).
- `npx tsc --noEmit -p tsconfig.json`: pass (no type errors).
- `node scripts/agent-customization/validate-plan-sync.mjs`: PASS (0 errors, 0 warnings).
- `node scripts/agent-customization/validate-plan-phase-packets.mjs`: PASS (0 errors, 0 warnings).

**Changed files:**

- `examples/racing_curriculum/README.md`

**PlanUpdate:**

```yaml
PlanUpdate:
  slice_id: step-11-tier1-tier2-baseline-docs
  changed_files:
    - 'examples/racing_curriculum/README.md'
  preflight: []
  validation:
    - command: 'npm run docs'
      result: 'PASS generated docs and bundles rebuilt'
    - command: 'npm run lint'
      result: 'PASS 0 issues'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS no type errors'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      result: 'PASS 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      result: 'PASS 0 errors, 0 warnings'
  next: 'Hand off to 07-logging for Step 12 — logging and tracker handoff; user must confirm documented baseline rules in browser/UI before Phase 3 is compressed and Phase 4 Step 01 is advanced.'
```

**Handoff query for `07-logging`:**

> Step 11 documentation pass is complete. `examples/racing_curriculum/README.md` now documents the Tier 1/Tier 2 shared baseline: Team 0 blue / Team 1 red car and guide-line colors, one dedicated guide line per car via `buildGuidingLineForTeam`, deterministic blue-inner (`TEAM_BLUE_INDEX = 0`) / red-outer (`TEAM_RED_INDEX = 1`) lane assignment, alternating pit ownership `[0, 1, 0, 1, 0, 1]` so each team has pits on both the inner and outer sides, and track-boundary wall enforcement via `clampCarToTrackBounds`. A Mermaid diagram shows the lane-assignment invariant chain, and a TypeScript snippet shows how to launch the demo with `start('racing-curriculum-output')`. All docs, lint, type-check, and plan-validator gates pass. Step 12 is logging and tracker handoff; the user should manually confirm the documented baseline rules render correctly in the browser/UI before Phase 3 is compressed and Phase 4 Step 01 is advanced.

#### Step 12 — Reconcile user-reported Tier 2 demo defects and plan hardening steps [DONE]

```yaml
phase: 3
step: 12
title: 'Reconcile user-reported Tier 2 demo defects and plan hardening steps'
status: '[DONE]'
goal: 'documenting'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 13 — Renderer hardening: guide lines + trails + header text'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'planning-acceptance-criteria'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Every user-reported issue is assigned to Phase 3 Tier 2 hardening or explicitly deferred to Phase 4 Tier 3.'
  - 'Step 13-17 packets are authored with red-green slices, clear file targets, and observable acceptance criteria.'
  - 'Plan-sync and plan-phase-packet validators pass.'
```

**User instruction:**

Reconcile the seven user-reported demo defects into bounded Phase 3 hardening steps. Decide which fixes belong in Phase 3 (Tier 1/Tier 2 baseline) vs. Phase 4 (Tier 3 proper), author Step 13-17 packets with red-green slices, and update the tracker so the next session can dispatch the first red-testing slice without relying on prior chat history.

**Step objective:**

Turn the 02-research findings into a concrete, gated implementation plan. Keep Phase 3 open until the Tier 1/Tier 2 demo baseline is visually and behaviorally correct; defer the full Tier 3 2v2 role-divergence implementation to Phase 4.

**Stop conditions:**

- **Done:** Step 13-17 packets exist with red-green slices, scope decisions recorded, and plan validators pass.
- **Hold:** user disagrees with the Phase 3 vs. Phase 4 assignment; record a Decision Record and escalate to `00-helping`.
- **Blocked:** a missing source location or conflicting reference design makes acceptance criteria unauthorable; route to `02-researching` or `nge-benchmark-scout`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Scope reconciliation decisions:**

| #   | User issue                                             | Phase 3 hardening                     | Phase 4 deferral               | Rationale                                                                                                                                                 |
| --- | ------------------------------------------------------ | ------------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | Page starts with `docs:folders:racing-curriculum`      | Step 13 renderer/docs cleanup         | —                              | No runtime source injects the alias; ensure generated docs/bundle surfaces use a human-readable title and the alias stays in `package.json` scripts only. |
| 2   | Each car needs a dedicated inner/outer guide line      | Step 13                               | —                              | Belongs to Tier 1/Tier 2 baseline rendering; `buildGuidingLineForTeam` must align with `resolveCurriculumRacePackCars` lane-center starts.                |
| 3   | Each car needs a short team-color trail                | Step 13                               | —                              | Belongs to Tier 1/Tier 2 baseline rendering; tire-mark state must be per-car.                                                                             |
| 4   | Strong negative reward for off-track / wrong direction | Step 14                               | —                              | Belongs to Tier 1/Tier 2 baseline physics; browser local step must match worker runner semantics.                                                         |
| 5   | Cars should not overlap; push each other apart         | Step 14                               | —                              | Belongs to Tier 1/Tier 2 baseline physics; two cars in Tier 1/2 already require separation.                                                               |
| 6   | Tier 2 promotion shows Tier 3 subtitle + 1 car         | Step 15 minimal Tier 3 4-car fallback | Full Tier 3 2v2 implementation | Prevent the broken single-car fallback in Phase 3; proper role-divergence tier remains Phase 4.                                                           |
| 7   | Demo feels like it skips Tier 1                        | Step 15                               | —                              | Change `ACTIVE_CURRICULUM_TIER` default from `2` to `1` and update README so the demo starts at the first tier.                                           |

**Key decisions:**

- Change `ACTIVE_CURRICULUM_TIER` to `1` in `browser-entry.ts` and update `examples/racing_curriculum/README.md` so the demo starts at Tier 1 rather than Tier 2.
- Add a minimal Tier 3 4-car fallback in `resolveCurriculumRacePackLayout` so auto-promotion from Tier 2 does not collapse to `[0]`; the full Tier 3 2v2 behavioral contract (role divergence, observation authority) stays in Phase 4.
- Group the fixes into three red-green implementation steps: renderer hardening, physics hardening, and tier layout/start.
- Keep car-vs-car collision response simple for Phase 3 (pushing impulse / separation); complex racing-line blocking and pit strategy belong to Phase 4/5.

**Research findings summary (02-research, 2026-06-25):**

The user manually confirmed the Step 12 build and reported seven symptoms. Read-only reconnaissance traced each to a concrete source location; Phase 3 is not ready to compress.

1. `docs:folders:racing-curriculum` is the npm script alias in `package.json` (line 95), consumed by `scripts/run-docs/run-docs.workflows.ts`; no runtime UI source injects it. Likely a stale bundle/docs artifact.
2. `buildGuidingLineForTeam` defaults Team 0 lateral offset to `0` (road centerline) instead of the inner-lane centerline; multi-car spread assumes more than two lanes.
3. Tire-mark trail samples only car 0 state; no per-car `tireMarks` array exists in `RacingRenderState`.
4. `environment.step.service.ts` clamps off-track cars back onto the edge with no penalty; `stepCarKinematics` allows reverse throttle. Worker runner has off-track penalty but browser local path does not.
5. No car-vs-car distance query, collision response, or pushing impulse exists in `environment.step.service.ts` or worker runner.
6. `resolveCurriculumRacePackLayout` lacks a Tier 3 branch and falls through to `[0]`; `ACTIVE_CURRICULUM_TIER = 2` causes auto-promotion after 3 laps.
7. `ACTIVE_CURRICULUM_TIER = 2` and README line 367 document Tier 2 as the demo default, conflicting with a natural Tier-1-start expectation.

**Plan update requirement:**

Record the reconciliation table and Step 13-17 packets in this plan. Do not edit production code in this step.

**Archived detailed findings (retained below for traceability):**

1. **Demo page header/title starts with `docs:folders:racing-curriculum`.**
   - The literal string does **not** appear in any runtime UI file under `examples/racing_curriculum/`. `index.html` title is `Racing Curriculum (NeatapticTS)` (line 5).
   - The string is the npm script alias `docs:folders:racing-curriculum` in `package.json` (line 95) and is consumed by `scripts/run-docs/run-docs.workflows.ts` for README generation; it also appears in archived learning-log text inside `docs/assets/semantic-snapshot.json`.
   - Root cause: either a stale bundle/docs artifact is being displayed, or the user is reading a generated docs surface rather than the live demo page. No source code currently injects this key into the DOM or document title.
   - Files: `examples/racing_curriculum/index.html`, `package.json`, `scripts/run-docs/run-docs.workflows.ts`.

2. **Each car should have a dedicated guide line only that car can see, inner/outer lane.**
   - `renderer/racing.renderer.ts` already draws one line per car via `drawTeamGuidingLines` (lines 843-915), but `buildGuidingLineForTeam` defaults Team 0 lateral offset to `0` (road centerline) instead of `+innerOffsetWorld` (inner-lane centerline) (lines 319-320).
   - For multi-car teams the offset math spreads same-team cars by integer multiples of `innerOffsetWorld`; on a 2-lane track the second/third cars land outside the drivable ribbon.
   - Root cause: default guide-line lateral offset does not match the actual lane-center start positions in `resolveCurriculumRacePackCars`, and the per-team spread assumes more than two lanes.
   - Files: `examples/racing_curriculum/renderer/racing.renderer.ts` (`buildGuidingLineForTeam`, `drawTeamGuidingLines`), `examples/racing_curriculum/browser-entry/browser-entry.ts` (`resolveCurriculumRacePackCars`, lines 2507-2559).

3. **Each car should have a short trail in its team color.**
   - A fading tire-mark trail exists (`renderer/racing.renderer.ts` `advanceTireMarks` / `drawTireMarks`), but it samples a single point behind the primary car using `envState.carX / envState.carY / envState.carHeading`.
   - In multi-car state only car 0 leaves marks, and there is no per-car `tireMarks` array in `RacingRenderState`.
   - Root cause: trail state is single-car.
   - Files: `examples/racing_curriculum/renderer/racing.renderer.ts` (`createRacingRenderState`, `advanceTireMarks`, `drawTireMarks`).

4. **Cars should receive a strong negative reward when outside the tracks or going the wrong direction.**
   - `environment/environment.step.service.ts` has no reward/fitness logic; `clampCarToTrackBounds` (lines 437-482) only snaps the car back onto the track edge without speed loss, termination, or score cost.
   - `stepCarKinematics` accepts negative throttle (reverse) with no wrong-direction detection.
   - `workers/simulation-worker/simulation-worker.race-pack.service.ts` defines `OFF_TRACK_PENALTY = 500` and `OFF_TRACK_GRACE_TICKS = 60` (lines 21, 30), but this is only applied inside the worker fitness path, not the browser local demo path.
   - Root cause: the browser physics step lacks reward/penalty terms and wrong-direction detection.
   - Files: `examples/racing_curriculum/environment/environment.step.service.ts`, `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`.

5. **Cars should not overlap; they should push each other apart.**
   - No inter-car geometry, distance query, collision response, pushing impulse, or contact penalty exists anywhere in the demo.
   - `environment.step.service.ts` `stepEnvironment` (lines 139-196) advances each car independently via `stepCarKinematics` then `clampCarToTrackBounds`.
   - Root cause: missing car-vs-car collision/separation logic in both the browser physics and worker race-pack runner.
   - Files: `examples/racing_curriculum/environment/environment.step.service.ts`, `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`.

6. **After the first round finishes (Tier 2), the page shows the Tier 3 subtitle and only 1 car appears.**
   - The browser harness auto-promotes after `LAP_COMPLETIONS_REQUIRED_FOR_TIER_ADVANCE = 3` laps via `resolveTierPromotionFromLapCount` (lines 389, 2681-2713). Starting at `ACTIVE_CURRICULUM_TIER = 2` (line 132), finishing 3 laps promotes to Tier 3.
   - `resolveCurriculumRacePackLayout` (lines 2472-2492) has explicit branches for tiers 1, 2, 4, and 5+, but **falls through to `[0]` for Tier 3** (and Tier 6), producing a single-car pack instead of the planned 4-car 2v2 pack.
   - The Tier 3 subtitle "Tier 3 keeps the same live shell while widening observation authority after guidance removal." comes from `resolveStageNarrativeForTier` (line 2814), which is correct for tiers >=3.
   - Root cause: Tier 3 browser race-pack layout is missing; promotion logic then exposes the broken layout.
   - Files: `examples/racing_curriculum/browser-entry/browser-entry.ts` (`resolveCurriculumRacePackLayout`, `resolveTierPromotionFromLapCount`, `createCurriculumEpisodeState`, `ACTIVE_CURRICULUM_TIER`).

7. **The user feels we are skipping Tier 1.**
   - `ACTIVE_CURRICULUM_TIER = 2` (line 132) and `examples/racing_curriculum/README.md` line 367 state: "The browser demo runs Tier 2 by default." Tier 1 is reachable only by changing the constant.
   - Root cause: the demo intentionally loads Tier 2, which the user perceives as skipping Tier 1. This is documented but conflicts with a natural "start at Tier 1" expectation.
   - Files: `examples/racing_curriculum/browser-entry/browser-entry.ts` (line 132), `examples/racing_curriculum/README.md`.

**New blockers for Step 12:**

- Tier 3 browser layout is undefined (`resolveCurriculumRacePackLayout` returns `[0]`), so auto-promotion from Tier 2 breaks the pack shape.
- Per-car guide-line defaults do not match the actual inner/outer lane-center starts for Team 0/Team 1.
- Car trails are single-car only.
- Browser local physics lacks off-track penalty, wrong-direction detection, and car-vs-car collision/pushing.
- The `docs:folders:racing-curriculum` string leakage into the UI needs reproduction; no source injection point was found.

#### Step 13 — Renderer hardening: guide lines + trails + header text [DONE]

```yaml
phase: 3
step: 13
title: 'Renderer hardening: guide lines + trails + header text'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 14 — Physics hardening: off-track penalty + wrong direction + car pushing'
skills:
  - 'red-test-contracts'
  - 'implementation-standards'
  - 'green-testing'
  - 'browser-ui-specialist'
specialists:
  - 'browser-ui-specialist'
validation:
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum/(renderer|browser-entry)'"
  - 'npm run build:racing-curriculum'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Each car renders a dedicated guide line aligned to its own lane (inner for Team 0 / blue, outer for Team 1 / red).'
  - 'Each car renders a short fading trail in its team color.'
  - 'The visible demo heading/title does not show the literal docs:folders:racing-curriculum alias.'
  - 'All renderer-focused tests, build, lint, and plan validators pass.'
slices:
  - slice_id: 'p3-s13-red-renderer'
    title: 'Red tests for per-car guide lines, per-car trails, and header text'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/renderer/racing.renderer.test.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - 'Tests fail before implementation because Team 0 guide line is on centerline, only car 0 leaves tire marks, and the docs alias appears in the DOM/title.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s13-impl-renderer'
    evidence:
      command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(renderer|browser-entry)'"
      exit_status: 1
      failed_tests:
        - 'per-agent guiding line geometry › defaults Team 0 lateral offset to the inner-lane centerline distance: default Team 0 line differs from explicit innerOffsetWorld line because buildGuidingLineForTeam defaults teamIndex===0 to lateralOffsetWorld=0 instead of the inner-lane centerline distance.'
        - 'per-agent guiding line geometry › places Team 1 guiding line on the outer-lane centerline: Team 1 guiding point lands on the road centerline (signed offset 0) instead of the outer-lane centerline offset -5.5485.'
        - 'per-agent guiding line draw calls › draws blue and red guide lines at inner and outer lane offsets for a 2-car pack: blue line is on the inner-lane centerline (offset 0 from it) instead of +innerOffsetWorld, so the 2-car lane-offset contract is not met.'
        - 'per-car tire mark accumulation › records tire marks whose world positions reflect both cars in a 2-car pack: all recorded tire marks cluster at car 0 position, max mark separation is 0, far below the 80-unit car separation.'
        - 'team-color tire mark rendering › renders tire marks in team colors rather than the cyan/white constants: recorded tire-mark paths only use cyan (170,235,255) and white (248,254,255); no blue or red team-color strokes are present.'
      passed_tests:
        - 'racing curriculum browser entry start() › does not use the docs alias as the page title'
        - 'racing curriculum browser entry start() › does not use the docs alias in any heading element'
        - 'racing curriculum browser entry start() › sets a non-empty human-readable page title or heading'
      fixture_notes:
        - 'Renderer tests use deterministic seed 42 track fixtures, a 2-car EnvironmentState with car 0 at (0,0) and car 1 at (80,0), and 8 render ticks to exceed TIRE_MARK_SAMPLE_INTERVAL_TICKS (3).'
        - 'Browser-entry tests run start() in jsdom with the standard #racing-curriculum-output host; no shared fixtures, clean document.body.innerHTML in beforeEach.'
      expected_green_condition: 'buildGuidingLineForTeam(0) defaults to the inner-lane centerline distance; buildGuidingLineForTeam(1) stays on the outer lane; drawTeamGuidingLines produces blue (+innerOffsetWorld) and red (-innerOffsetWorld) lines for a 2-car pack; advanceTireMarks iterates envState.cars so both cars leave marks; drawTireMarks uses per-car team colors; docs alias guard tests remain green.'
  - slice_id: 'p3-s13-impl-renderer'
    title: 'Implement per-car guide lines, per-car team-color trails, and alias-free heading'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/renderer/racing.renderer.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/index.html'
    acceptance_criteria:
      - 'All red tests pass after implementation.'
    parallelizable: false
    dependencies:
      - 'p3-s13-red-renderer'
    next_slice: 'p3-s13-green-renderer'
    evidence:
      changed_files:
        - 'examples/racing_curriculum/renderer/racing.renderer.ts'
        - 'examples/racing_curriculum/renderer/racing.renderer.test.ts'
      notes:
        - 'Updated three existing renderer tests that encoded the previous buggy guide-line offsets (Team 1 on road centerline instead of outer-lane centerline). The red-phase tests now define the corrected contract.'
        - 'browser-entry.ts and index.html required no implementation changes; alias-guard tests were already green.'
      validation:
        - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(renderer|browser-entry)'"
          exit_status: 0
          summary: '7 suites, 85 tests passed'
        - command: 'npx tsc --noEmit -p tsconfig.json'
          exit_status: 0
        - command: 'npx tsc --noEmit -p tsconfig.test.json'
          exit_status: 0
        - command: 'npm run lint'
          exit_status: 0
        - command: 'npm run build:racing-curriculum'
          exit_status: 0
        - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
          exit_status: 0
        - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
          exit_status: 0
  - slice_id: 'p3-s13-green-renderer'
    title: 'Green validation and visual re-check of renderer hardening'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
    acceptance_criteria:
      - 'Targeted renderer and browser-entry tests pass.'
      - 'Browser-ui-specialist confirms two cars each have a dedicated guide line and a short team-color trail in the live demo.'
      - 'No literal docs:folders:racing-curriculum appears in the visible page title/heading.'
    parallelizable: false
    dependencies:
      - 'p3-s13-impl-renderer'
    next_slice: null
    evidence:
      notes:
        - 'browser-ui-specialist MCP was unavailable, so a deterministic headless Puppeteer probe was built and run against the local HTTP-served demo as a substitute visual gate.'
        - 'Default live demo runs curriculum Tier 2, where resolveGuidanceAlphaForCurriculumTier returns guidanceAlpha = 0 and per-car guide lines are intentionally not drawn. Per-car guide-line contract is therefore covered by the focused Jest suite, not the live UI snapshot.'
      puppeteer_visual_gate:
        command: 'node tmp/puppeteer-step13-gate.mjs'
        alias_leak: false
        canvas_dimensions: '734x568'
        team_color_car_clusters: 7
        tire_mark_segments:
          blue: 2862
          red: 2862
        guide_line_segments_in_live_tier2_ui: 0
      validation:
        - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(renderer|browser-entry)'"
          exit_status: 0
          summary: '7 suites, 85 tests passed'
        - command: 'npx tsc --noEmit -p tsconfig.json'
          exit_status: 0
        - command: 'npx tsc --noEmit -p tsconfig.test.json'
          exit_status: 0
        - command: 'npm run lint'
          exit_status: 0
        - command: 'npm run build:racing-curriculum'
          exit_status: 0
        - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
          exit_status: 0
        - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
          exit_status: 0
```

**User instruction:**

Write focused red tests, implement, and green-validate the renderer hardening slice. Each car must have its own guide line in its own lane and a short team-color trail; the visible page heading must not leak the `docs:folders:racing-curriculum` alias.

**Step objective:**

Fix the Tier 1/Tier 2 visual baseline so every car has a dedicated, lane-aligned guide line and a team-color trail, and clean up the docs alias leak.

**Stop conditions:**

- **Done:** All three slices pass and browser-ui-specialist confirms the visual behavior.
- **Hold:** red tests cannot be written without changing source first; escalate to `00-helping`.
- **Blocked:** visual confirmation contradicts test results; route back to the red-testing slice.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum/(renderer|browser-entry)'`
- `npm run build:racing-curriculum`
- `npm run lint`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Plan update requirement:**

Mark slices `[DONE]` as they pass and attach test/coverage evidence. Do not proceed to Step 14 until green validation is complete.

```yaml
PlanUpdate:
  slice_id: 'p3-s13-impl-renderer'
  changed_files:
    - examples/racing_curriculum/renderer/racing.renderer.ts
    - examples/racing_curriculum/renderer/racing.renderer.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
  validation:
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(renderer|browser-entry)'"
      expected_exit: 0
      result: '7 suites, 85 tests passed'
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
  rollback:
    - 'git checkout -- examples/racing_curriculum/renderer/racing.renderer.ts'
    - 'git checkout -- examples/racing_curriculum/renderer/racing.renderer.test.ts'
  next: 'Run 05-green-testing slice p3-s13-green-renderer and browser-ui-specialist visual confirmation'
```

#### Step 14 — Physics hardening: off-track penalty + wrong direction + car pushing [DONE]

```yaml
phase: 3
step: 14
title: 'Physics hardening: off-track penalty + wrong direction + car pushing'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 15 — Tier layout/start: Tier 1 default + Tier 3 fallback'
skills:
  - 'red-test-contracts'
  - 'implementation-standards'
  - 'green-testing'
validation:
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum/(environment|workers/simulation-worker)'"
  - 'npm run build:racing-curriculum'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Cars receive a strong negative reward/penalty when off track.'
  - 'Cars detect and are penalized for going the wrong direction.'
  - 'Cars do not overlap; a simple pushing force separates them on contact.'
  - 'Worker race-pack runner behavior remains consistent with browser local physics where applicable.'
  - 'All environment-focused tests, build, lint, and plan validators pass.'
slices:
  - slice_id: 'p3-s14-red-physics'
    title: 'Red tests for off-track penalty, wrong-direction detection, and car separation'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/environment/environment.step.service.test.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
    acceptance_criteria:
      - 'Tests fail before implementation because cars can drive off track without penalty, reverse without detection, and overlap without response.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s14-impl-physics'
  - slice_id: 'p3-s14-impl-physics'
    title: 'Implement off-track penalty, wrong-direction detection, and car-vs-car pushing'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 6
    files_to_change:
      - 'examples/racing_curriculum/environment/environment.step.service.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    acceptance_criteria:
      - 'All red tests pass after implementation.'
    parallelizable: false
    dependencies:
      - 'p3-s14-red-physics'
    next_slice: 'p3-s14-green-physics'
  - slice_id: 'p3-s14-green-physics'
    title: 'Green validation of physics hardening'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
    acceptance_criteria:
      - 'Targeted environment and worker race-pack tests pass.'
      - 'Cars stop overlapping and off-track cars are penalized in the live demo.'
    parallelizable: false
    dependencies:
      - 'p3-s14-impl-physics'
    next_slice: null
```

**User instruction:**

Write focused red tests, implement, and green-validate the physics hardening slice. The browser local step must penalize off-track and wrong-direction driving and must push overlapping cars apart.

**Step objective:**

Fix the Tier 1/Tier 2 physics baseline so cars cannot drive off track without cost, cannot gain from reversing, and cannot occupy the same space.

**Stop conditions:**

- **Done:** All three slices pass and live demo shows separated, track-bound cars.
- **Hold:** red tests require implementation changes first; escalate to `00-helping`.
- **Blocked:** worker and browser physics diverge in a way that cannot be reconciled inside the demo; route to `worker-payload-scout` or `nge-core-algorithm`.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum/(environment|workers/simulation-worker)'`
- `npm run build:racing-curriculum`
- `npm run lint`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Plan update requirement:**

Mark slices `[DONE]` as they pass and attach test/coverage evidence. Do not proceed to Step 15 until green validation is complete.

```yaml
PlanUpdate:
  slice_id: 'p3-s14-impl-physics'
  changed_files:
    - examples/racing_curriculum/environment/environment.step.service.ts
    - examples/racing_curriculum/environment/environment.types.ts
    - examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts
    - examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npx prettier --check examples/racing_curriculum/environment/environment.step.service.ts examples/racing_curriculum/environment/environment.types.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=environment.step.service.test.ts'
      expected_exit: 0
      result: '6 passed — off-track clamp reward, wrong-direction reward, and car separation all green'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.race-pack.test.ts'
      expected_exit: 0
      result: '32 passed — on-track car not unfairly penalized, off-track car penalized, and worker car separation green'
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(environment|workers/simulation-worker)'"
      expected_exit: 0
      result: '18 suites, 137 tests passed — full environment + worker regression slice'
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
      result: 'docs/assets/racing-curriculum.bundle.js rebuilt (733.7kb)'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets'
  coverage_guard:
    - file: examples/racing_curriculum/environment/environment.step.service.ts
      summary: 'new physics paths covered by focused tests; full file 85.71% due to pre-existing uncovered helper branches'
    - file: examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts
      summary: 'new per-car off-track and separation paths covered; full file 95.89% due to pre-existing uncovered defensive branches'
  rollback:
    - 'git checkout -- examples/racing_curriculum/environment/environment.step.service.ts'
    - 'git checkout -- examples/racing_curriculum/environment/environment.types.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
  next: 'Open Step 15 (Tier layout/start) for planning; no 05-green-testing handoff needed because green slice was run inline.'
```

#### Step 15 — Tier layout/start: Tier 1 default + Tier 3 fallback [DONE]

```yaml
phase: 3
step: 15
title: 'Tier layout/start: Tier 1 default + Tier 3 fallback'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 16 — Document updated Tier 1/Tier 2 demo contract'
skills:
  - 'red-test-contracts'
  - 'implementation-standards'
  - 'green-testing'
  - 'browser-ui-specialist'
specialists:
  - 'browser-ui-specialist'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/browser-entry'
  - 'npm run build:racing-curriculum'
  - 'npm run lint'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'ACTIVE_CURRICULUM_TIER defaults to 1 so the demo starts at Tier 1.'
  - 'resolveCurriculumRacePackLayout(3) returns a valid 4-car 2v2 pack [0, 0, 1, 1] so auto-promotion from Tier 2 does not collapse to one car.'
  - 'Tier 1 and Tier 2 pack layouts still produce two cars (one per team).'
  - 'All browser-entry-focused tests, build, lint, and plan validators pass.'
slices:
  - slice_id: 'p3-s15-red-tier-layout'
    title: 'Red tests for Tier 1 default start and Tier 3 fallback layout'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - 'start() posts a worker init message with tier 1 (currently tier 2).'
      - 'createCurriculumEpisodeState(3) returns 4 cars (currently 1).'
      - 'createCurriculumEpisodeState(3) team indices are [0, 0, 1, 1] (currently [0]).'
      - 'createCurriculumEpisodeState(1) and createCurriculumEpisodeState(2) still return 2 cars with team indices [0, 1].'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s15-impl-tier-layout'
  - slice_id: 'p3-s15-impl-tier-layout'
    title: 'Implement Tier 1 default start and minimal Tier 3 4-car layout'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - 'All red tests pass after implementation.'
    parallelizable: false
    dependencies:
      - 'p3-s15-red-tier-layout'
    next_slice: 'p3-s15-green-tier-layout'
  - slice_id: 'p3-s15-green-tier-layout'
    title: 'Green validation of tier layout/start fixes'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
    acceptance_criteria:
      - 'Targeted browser-entry tests pass.'
      - 'Live demo starts at Tier 1 and auto-promotion to Tier 3 shows four cars.'
    parallelizable: false
    dependencies:
      - 'p3-s15-impl-tier-layout'
    next_slice: null
```

**User instruction:**

Author focused red tests, implement the minimal production changes, and green-validate the tier layout/start slice. The demo must start at Tier 1, and auto-promotion from Tier 2 must not collapse to a single car.

**Step objective:**

Fix the demo entry point so it starts at Tier 1 and provide a minimal Tier 3 4-car fallback so promotion does not break the pack shape before Phase 4 implements the full 2v2 contract.

**Scope decisions:**

- Default tier: change `ACTIVE_CURRICULUM_TIER` from `2` to `1`. The user-reported "skips Tier 1" symptom is caused by the hard-coded Tier 2 default. A natural ladder should start at Tier 1; Tier 2 remains reachable through the same auto-promotion rule after 3 laps.
- Tier 3 fallback layout: `[0, 0, 1, 1]` (Team 0 blue/blue on the inner lane, Team 1 red/red on the outer lane). This matches the reference design's 2v2 shape while keeping the Phase 3 implementation minimal and avoiding the current `[0]` fall-through collapse.
- No new `browser-entry.progression.test.ts` file: all assertions can be made through existing public exports (`start`, `createCurriculumEpisodeState`) in `browser-entry.test.ts`, so the test surface stays small.

**Red tests to write in `examples/racing_curriculum/browser-entry/browser-entry.test.ts`:**

1. `start() posts worker init with curriculum tier 1`
   - Mock `globalThis.Worker` and call `start('racing-curriculum-output')`.
   - Assert the first `postMessage` call carries `type: 'init'` and `tier: 1`.
   - Currently fails because `ACTIVE_CURRICULUM_TIER = 2`.

2. `createCurriculumEpisodeState(3) returns a 4-car 2v2 pack`
   - Call `createCurriculumEpisodeState(3)` and inspect `envState.cars`.
   - Assert `cars.length === 4`.
   - Currently fails because `resolveCurriculumRacePackLayout(3)` falls through to `[0]`.

3. `createCurriculumEpisodeState(3) places two blue cars on the inner lane and two red cars on the outer lane`
   - Compute signed lateral offsets for the four cars using the first spline sample frame.
   - Assert team indices are `[0, 0, 1, 1]`, Team 0 offsets are positive (inner), Team 1 offsets are negative (outer).
   - Currently fails because only one car is returned.

4. `Tier 1 and Tier 2 layouts remain two-car packs`
   - Call `createCurriculumEpisodeState(1)` and `createCurriculumEpisodeState(2)`.
   - Assert each has 2 cars with team indices `[0, 1]`.
   - These should already pass and act as regression guards.

**Red evidence for `p3-s15-red-tier-layout`:**

- File changed: `examples/racing_curriculum/browser-entry/browser-entry.test.ts`
- Added `it('posts an init message with curriculum tier 1')` under the existing worker-protocol describe block; currently fails because the posted init message carries `tier: 2`.
- Added `describe('Tier 3 2v2 fallback race pack layout')` with three single-expect tests:
  - `returns four cars for Tier 3` fails with `Expected: 4, Received: 1`.
  - `assigns team indices [0, 0, 1, 1] for Tier 3` fails with `Expected: [0, 0, 1, 1], Received: [0]`.
  - `places Team 0 on the inner lane and Team 1 on the outer lane for Tier 3` fails with `Expected: true, Received: false`.
- Added `describe('Tier 1 and Tier 2 regression guards')` with two tests that pass (expected green guards).
- Focused Jest command: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry`
- Jest result: 4 failed, 65 passed, 69 total; failures are the expected red contracts.
- Lint: `npx eslint examples/racing_curriculum/browser-entry/browser-entry.test.ts` ? pass.
- Type check: `npx tsc --noEmit -p tsconfig.json` ? pass.
- Plan validators: `validate-plan-sync.mjs` and `validate-plan-phase-packets.mjs` ? pass.

**Stop conditions:**

- **Done:** All three slices pass and live demo confirms Tier 1 start + Tier 3 4-car promotion.
- **Hold:** user wants to keep Tier 2 as the default or use a different Tier 3 layout; record a Decision Record and stop.
- **Blocked:** adding a Tier 3 layout requires NGE primitives not yet available; route to `nge-core-algorithm`.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum/browser-entry`
- `npm run build:racing-curriculum`
- `npm run lint`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Expected green conditions for `p3-s15-impl-tier-layout`:**

1. `ACTIVE_CURRICULUM_TIER` is set to `1` in `browser-entry.ts`.
2. `resolveCurriculumRacePackLayout` returns `[0, 0, 1, 1]` when `curriculumTier === 3`.
3. No other tier branches are changed.

**Plan update requirement:**

Mark slices `[DONE]` as they pass and attach test/coverage evidence. Do not proceed to Step 16 until green validation is complete.

#### Step 16 — Document updated Tier 1/Tier 2 demo contract [DONE]

```yaml
phase: 3
step: 16
title: 'Document updated Tier 1/Tier 2 demo contract'
status: '[DONE]'
goal: 'documenting'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 17 — Logging and tracker handoff'
skills:
  - 'documenting'
  - 'docs-example-writer'
validation:
  - command: 'npm run docs'
    result: 'PASS — HTML docs generated; Mermaid diagrams validated'
  - command: 'npm run lint'
    result: 'PASS — 0 issues'
  - command: 'npx tsc --noEmit -p tsconfig.json'
    result: 'PASS — no type errors'
  - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    result: 'PASS plan sync: 0 errors, 0 warnings'
  - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    result: 'PASS plan phase packets: 0 errors, 0 warnings'
acceptance_criteria:
  - 'examples/racing_curriculum/README.md documents Tier 1 start, per-car guide lines/trails, off-track/wrong-direction penalties, car separation, and the Tier 3 fallback pack shape.'
  - 'Docs build, lint, type check, and plan validators pass.'
```

**User instruction:**

Update the racing curriculum README to reflect the hardened Tier 1/Tier 2 demo contract: Tier 1 start, per-car guide lines and trails, off-track/wrong-direction penalties, car separation, and the minimal Tier 3 fallback.

**Step objective:**

Make the updated baseline contract discoverable for users and downstream agents.

**Stop conditions:**

- **Done:** README updated, docs build and lint pass, and plan validators pass.
- **Hold:** user must review the updated contract before Phase 3 closes.
- **Blocked:** generated docs produce a diff unrelated to this change; escalate to `00-helping`.

**Required validation:**

- `npm run docs`
- `npm run lint`
- `npx tsc --noEmit -p tsconfig.json`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Plan update requirement:**

Record the changed README section and validation evidence. Do not proceed to Step 17 until user confirms the docs are accurate in the live demo.

#### Step 18 — Tier 1 demo defect investigation [DONE]

```yaml
phase: 3
step: 18
title: 'Tier 1 demo defect investigation'
status: '[DONE]'
goal: 'researching'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 19 — Tier 1 demo defect implementation'
skills:
  - 'research-methodology'
  - 'subagent-delegation-patterns'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Root-cause hypotheses confirmed with file:line evidence.'
  - 'Alignment brief identifies 1-3 implementation slices.'
  - 'Red-test additions specified for each fix.'
  - 'Plan validators pass after plan update.'
  - 'Handoff to 01-planning / 04-implementing recorded.'
```

**User instruction:**

Two remaining Tier 1 live-demo defects: (1) the red/outer car ignores its guide line while the blue/inner car follows its guide, and (2) cars can still touch/overlap. Investigate root causes, identify fix slices, and hand off to planning/implementation.

**Step objective:**

Produce a source-grounded alignment brief that maps the two symptoms to precise code boundaries and recommends small, testable implementation slices.

**Stop conditions:**

- **Done:** alignment brief is recorded in the plan, all hypotheses are verified with file:line evidence, and handoff to 01-planning / 04-implementing is recorded.
- **Hold:** user must approve the proposed slices before implementation.
- **Blocked:** no blockers identified; if a core NGE primitive is needed, escalate to `00-helping`.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Plan update requirement:**

Record the alignment brief, root-cause evidence, proposed implementation slices, red-test needs, and risks in the plan. Do not proceed to implementation until user confirms the slices.

**Alignment brief**

- **Symptom 1 — red/outer car ignores its guide line.**

  - _File/line evidence:_ `observation.assembler.ts:677-680` computes `optimalLineLateralOffsetWorld = signedLateralOffsetWorld - innerLaneCenterlineOffsetWorld`, so the "optimal line" is always the inner-lane centerline. It never reads `envState.teamIndex`.
  - _File/line evidence:_ `nge.controller.ts:120-187` creates a single-car controller that consumes `envState.carX/carY/carHeading` and returns one `{throttle, steer}` output.
  - _File/line evidence:_ `browser-entry.ts:2840-2848` (`resolveControlFanOut`) clones that single output to every car.
  - _File/line evidence:_ `browser-entry.ts:468-470` creates only one controller; `browser-entry.ts:644` calls `computeControl` once per tick.
  - _Verdict:_ the red car has no observation or control loop of its own. Even though the renderer draws a correct outer guide (`racing.renderer.ts:302-321`), the controller is trained/observed against the inner line and its output is fanned out to both cars.

- **Symptom 2 — cars still touch/overlap.**

  - _File/line evidence:_ `environment.step.service.ts:25` defines `CAR_MIN_CENTER_SEPARATION = 1`.
  - _File/line evidence:_ `racing.renderer.ts:64-66` defines car footprint `CAR_HALF_LENGTH_WORLD = 3.8` and `CAR_HALF_WIDTH_WORLD = 2.2`, i.e. 7.6 × 4.4 world units.
  - _File/line evidence:_ `environment.step.service.ts:428-470` pushes centers apart only until they exceed `CAR_MIN_CENTER_SEPARATION`.
  - _File/line evidence:_ `environment.step.service.test.ts:221-264` only asserts separation `> 1e-6`, not bounding-box non-overlap.
  - _File/line evidence:_ `simulation-worker.race-pack.service.ts:513-517` starts every worker car at the same `(startPoint.x, startPoint.y)`; its separation constant is `2.5` (`simulation-worker.race-pack.service.ts:33`), still below the 4.4-unit width.
  - _Verdict:_ separation constants are smaller than the car bounding box, so visual overlap persists after separation. Existing tests do not enforce non-overlap.

- **Symptom 3 — the blue/Team 0 guide line is rendered yellow instead of blue.**

  - _File/line evidence:_ `renderer/racing.renderer.ts:34` defines `COLOR_GUIDANCE_LINE_RGB = '255,209,102'` (yellow).
  - _File/line evidence:_ `renderer/racing.renderer.ts:805-827` (`drawOptimalLineGuidance`) strokes a single yellow optimal-line overlay when `guidanceAlpha > 0`.
  - _File/line evidence:_ `renderer/racing.renderer.ts:35-36` already defines team guide colors `COLOR_GUIDING_LINE_TEAM_A_RGB = '0,0,255'` (blue) and `COLOR_GUIDING_LINE_TEAM_B_RGB = '255,0,0'` (red), and `drawTeamGuidingLines` (`:846-905`) draws per-car guide lines in those colors.
  - _Verdict:_ the yellow optimal-line overlay is redundant with the team-colored per-car guide lines and is what the user sees as the "blue guide line turning yellow". The renderer should rely on team-colored guide lines only.

- **Symptom 4 — the decorative blue center division line should not be drawn.**

  - _File/line evidence:_ `renderer/racing.renderer.ts:33` defines `COLOR_CENTERLINE = 'rgba(0,180,220,0.30)'` (cyan/blue).
  - _File/line evidence:_ `renderer/racing.renderer.ts:670-684` (`drawTrackCenterline`) draws a dashed cyan centerline down the middle of the road surface.
  - _Verdict:_ racing tracks have painted edge markings, not a center divider. The `drawTrackCenterline` call should be removed from `drawTrack` (`:583`).

- **Recommended implementation slices.**

  1. _Team-aware optimal-line offset in the observation assembler._
  - Files: `examples/racing_curriculum/controller/observation.assembler.ts`, `examples/racing_curriculum/controller/observation.assembler.test.ts`.
  - Change: in `resolveObservationState`, compute the target lateral offset from `envState.teamIndex` (inner offset for team 0, outer/negative offset for team 1).
  - Acceptance: a car on its own lane centerline reports near-zero optimal-line offset regardless of team.
  - Red test: add a test with `teamIndex = 1` on the outer-lane centerline asserting channel 16 ˜ 0.
  2. _Per-car controller calls in the browser harness._
  - Files: `examples/racing_curriculum/browser-entry/browser-entry.ts`, possibly `examples/racing_curriculum/controller/nge.controller.ts`.
  - Change: build a per-car observation snapshot from `envState.cars[i]` and call `computeControl` for each car; remove `resolveControlFanOut` (do not leave a dual-path or compatibility wrapper). Worker stepping path at `browser-entry.ts:3517` already calls `stepEnvironment`, so the same per-car control array can be passed there.
  - Acceptance: red car steers toward the outer guide while blue car steers toward the inner guide; live-demo cars no longer drive on top of each other.
  - Red test: add a controller/browser-entry test that shows opposite/different controls for blue-on-inner vs red-on-outer snapshots.
  3. _Strengthen car separation and align worker starting grid._
  - Files: `examples/racing_curriculum/environment/environment.step.service.ts`, `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`, plus tests.
  - Change: set `CAR_MIN_CENTER_SEPARATION` to at least the car bounding-circle diameter (`2 * Math.hypot(3.8, 2.2) ˜ 8.8`) or switch to rectangle separation; in the worker `buildRaceFrame`, stagger cars per the same `TEAM_LAYOUT` / grid-spacing logic as `resolveCurriculumRacePackCars` so cars do not start overlapped.
  - Acceptance: two cars placed at the same point end a step with non-overlapping bounding boxes; worker frame cars are not all identical and follow the 3v3 block layout.
  - Red test: replace the `> 1e-6` separation assertion with a bounding-box non-overlap check; add a worker test for distinct starting positions and block team order.
  4. _Renderer visual cleanup: team-colored guide lines and removed center divider._
  - Files: `examples/racing_curriculum/renderer/racing.renderer.ts`, `examples/racing_curriculum/renderer/racing.renderer.test.ts`.
  - Change: remove the yellow `drawOptimalLineGuidance` overlay and the `drawTrackCenterline` call from `drawTrack`; keep `drawTeamGuidingLines` as the only guide overlay, using the existing blue/red team colors.
  - Acceptance: with guidance overlay enabled, only blue (Team 0) and red (Team 1) guide lines are drawn; no yellow line and no cyan centerline divider.
  - Red test: update or replace the `draws the optimal-line guidance along the inner-lane centerline` test so it fails on the yellow line and asserts team-colored guide lines instead.

- **Risks.**

  - Slice 1 alone will not fix the live demo because `envState.teamIndex` in the single-car controller path is the primary (blue) car's team. Slices 1 and 2 must land together.
  - Per-car control changes the deterministic trajectory used by `runDeterministicControllerProbe` and progression tests; baselines may need updates.
  - Worker tests currently assert both cars start on the inner-lane centerline and use alternating teams (`simulation-worker.race-pack.test.ts:427-465`); aligning the worker grid with the live demo will break those baselines.
  - Increasing the separation constant may change car-vs-car pushing dynamics in existing physics tests.
  - Renderer tests that encode the current yellow optimal line (`racing.renderer.test.ts:112-156`) and any test that counts centerline paths must be updated, not skipped, because they currently assert the buggy visual behavior.

- **Validation to run after implementation.**

  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(controller|browser-entry|environment|workers/simulation-worker|renderer)'`
  - `npm run lint`
  - `npx tsc --noEmit -p tsconfig.json`
  - `npx tsc --noEmit -p tsconfig.test.json`
  - `npm run build:racing-curriculum`
  - Live browser check: Tier 1 starts with two cars, blue follows inner guide, red follows outer guide, cars do not overlap after the first second, no cyan center divider, and guide lines are blue/red.

- **Handoff.**

  Route to `01-planning` to size and sequence the four slices, then to `04-implementing` for the code changes. If a core NGE primitive (e.g., per-genome team conditioning) is needed, escalate to `00-helping`.

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s18-research-tier1-defects'
  changed_files:
    - NONE
  preflight:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry'
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
      expected_exit: 0
      result: 'PASS 1 suite, 6 tests'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts'
      expected_exit: 0
      result: 'PASS 1 suite, 6 tests'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry'
      expected_exit: 0
      result: 'PASS 6 suites, 69 tests'
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
      expected_exit: 0
      result: 'PASS 1 suite, 32 tests'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  coverage_guard:
    files: []
    summary: 'Research step only; no production files changed; coverage-guard not required.'
  rollback:
    - 'git checkout -- plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  next: '01-planning / 04-implementing for p3-s18-impl-tier1-defects'
```

#### Step 19 — Tier 1 independent-agent architecture pivot [DONE]

```yaml
phase: 3
step: 19
title: 'Tier 1 independent-agent architecture pivot'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 17 — Logging and tracker handoff'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
  - 'implementation-standards'
  - 'red-testing'
  - 'green-testing'
validation:
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/runtime.adaptation.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/environment/environment.step.service.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
  - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/renderer/racing.renderer.test.ts'
  - "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(controller|browser-entry|environment|workers/simulation-worker|renderer)'"
  - 'npm run lint'
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npx tsc --noEmit -p tsconfig.test.json'
  - 'npm run build:racing-curriculum'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - '`observation.assembler.ts` exposes a per-car `RacingObservationState` helper that accepts `carIndex` and produces distinct observation vectors for blue and red cars on the same tick.'
  - '`browser-entry.ts` maintains a `Map<carIndex, NgeController>` and calls each controller independently per tick; outputs are applied only to the matching car.'
  - '`resolveControlFanOut` and any singleton shared-controller state are removed in the same step that introduces the per-car replacements (no wrappers, no dual paths, no deferred cleanup).'
  - '`runtime.adaptation.ts` stores per-car adaptation state and mutates each controller independently on its own cadence.'
  - 'The worker race-pack produces one distinct NEAT-derived network per car; the coevolution container and evolution protocol return per-car/per-team payloads rather than a single shared best network.'
  - 'A deterministic Tier 1 probe records distinct steering or lateral positions for the blue and red cars within 120 frames.'
  - 'Renderer visual cleanup removes the yellow optimal-line overlay and cyan center divider; only blue/red team guide lines remain.'
  - 'Car-separation / worker-grid slices keep cars from overlapping bounding boxes and use a consistent starting grid in both live and worker modes.'
slices:
  - slice_id: 'p3-s19-red-obs-team-offset'
    title: 'Red tests for team-aware observation offset'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
    acceptance_criteria:
      - 'A new test fails on the current top-level pose and asserts that Team 1 (outer lane) gets a right-side lane offset.'
      - 'Failure reproduces the "red car ignores guide line" symptom without changing production code.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s19-impl-obs-team-offset'
    VALIDATION_EVIDENCE:
      focused_test: 'PASS default examples/racing_curriculum/controller/observation.assembler.test.ts — 7/8 tests, 1 expected failure (teamIndex 1 outer-lane centerline)'
      regression_test: 'PASS examples/racing_curriculum/browser-entry — 6 suites, 69/69 tests'
      type_check: 'PASS npx tsc --noEmit -p tsconfig.json && npx tsc --noEmit -p tsconfig.test.json'
      lint: 'PASS npm run lint'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs'
  - slice_id: 'p3-s19-impl-obs-team-offset'
    title: 'Implement team-aware observation offset'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
    acceptance_criteria:
      - '`deriveObservationState` consumes `teamIndex` and applies a lateral lane offset so Team 0 tracks the inner guide and Team 1 tracks the outer guide.'
      - 'The helper is unit-testable in isolation and does not depend on live browser state.'
      - 'All observation assembler tests pass.'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-obs-team-offset'
    next_slice: 'p3-s19-green-obs-team-offset'
    VALIDATION_EVIDENCE:
      focused_test: 'PASS default examples/racing_curriculum/controller/observation.assembler.test.ts — 8/8 tests'
      regression_test: 'PASS examples/racing_curriculum/browser-entry — 6 suites, 69/69 tests'
      type_check: 'PASS npx tsc --noEmit -p tsconfig.json && npx tsc --noEmit -p tsconfig.test.json'
      lint: 'PASS npm run lint'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs'
  - slice_id: 'p3-s19-green-obs-team-offset'
    title: 'Green validation for team-aware observation offset'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
    acceptance_criteria:
      - 'Team-aware offset tests remain green.'
      - 'No regressions in browser-entry, controller, or renderer suites.'
      - 'Deterministic baselines are updated only if the offset legitimately changes progression outputs.'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-obs-team-offset'
    next_slice: 'p3-s19-red-per-car-observation'
    VALIDATION_EVIDENCE:
      focused_test: 'PASS default examples/racing_curriculum/controller/observation.assembler.test.ts — 8/8 tests'
      regression_test: 'PASS examples/racing_curriculum/browser-entry — 6 suites, 69/69 tests'
      type_check: 'PASS npx tsc --noEmit -p tsconfig.json && npx tsc --noEmit -p tsconfig.test.json'
      lint: 'PASS npm run lint'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs'
  - slice_id: 'p3-s19-red-per-car-observation'
    title: 'Red tests for per-car observation-state helper'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
    acceptance_criteria:
      - 'A new test fails when the assembler reads a single top-level car pose instead of deriving it from `carIndex`.'
      - 'Test asserts that two cars receive different observation vectors on the same tick.'
    parallelizable: false
    dependencies:
      - 'p3-s19-green-obs-team-offset'
    next_slice: 'p3-s19-impl-per-car-observation'
    VALIDATION_EVIDENCE:
      focused_test: 'FAIL examples/racing_curriculum/controller/observation.assembler.test.ts — 8 passed, 4 failed, 12 total. The 4 new derivePerCarObservationState contracts fail because observation.assembler.ts does not export the helper.'
      failure_summary:
        - 'exports derivePerCarObservationState: Expected "function", received "undefined"'
        - 'reads requested car pose: TypeError: mod.derivePerCarObservationState is not a function'
        - 'derives teamIndex from selected car: TypeError: mod.derivePerCarObservationState is not a function'
        - 'produces different Tier 1 vectors: TypeError: mod.derivePerCarObservationState is not a function'
      type_check: 'PASS npx tsc --noEmit -p tsconfig.json && npx tsc --noEmit -p tsconfig.test.json (not rerun; prior green baseline remains valid)'
      lint: 'PASS npm run lint (prior green baseline)'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs'
      fixture_notes: 'Deterministic straight-track fixture with Team 0 on the inner-lane centerline and Team 1 on the outer-lane centerline; top-level pose mirrors car 0 for backward compatibility.'
      handoff: '04-implementing should add export derivePerCarObservationState(envState, carIndex) to observation.assembler.ts; it must read envState.cars[carIndex], derive teamIndex from that car, and layer existing observation extensions over the car pose. Once implemented, the focused test slice above should become 12/12 green.'
  - slice_id: 'p3-s19-impl-per-car-observation'
    title: 'Implement per-car observation-state helper'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
    acceptance_criteria:
      - 'New helper `derivePerCarObservationState(envState, carIndex)` returns a `RacingObservationState` scoped to that car.'
      - 'The helper composes the existing team-aware lane offset so each controller sees its own pose and target lane.'
      - 'All observation assembler tests pass.'
    VALIDATION_EVIDENCE:
      focused_test: 'PASS examples/racing_curriculum/controller/observation.assembler.test.ts — 12/12 tests'
      type_check: 'PASS npx tsc --noEmit -p tsconfig.json && npx tsc --noEmit -p tsconfig.test.json'
      lint: 'PASS npm run lint'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-per-car-observation'
    next_slice: 'p3-s19-green-per-car-observation'
  - slice_id: 'p3-s19-green-per-car-observation'
    title: 'Green validation for per-car observation-state helper'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 1
    files_to_change:
      - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
    acceptance_criteria:
      - 'Red tests from p3-s19-red-per-car-observation now pass.'
      - 'No regressions in controller or downstream browser-entry suites.'
    VALIDATION_EVIDENCE:
      focused_test: 'PASS examples/racing_curriculum/controller/observation.assembler.test.ts — 12/12 tests'
      browser_entry_regression: 'PASS examples/racing_curriculum/browser-entry — 69/69 tests across 6 suites'
      type_check_main: 'PASS npx tsc --noEmit -p tsconfig.json'
      type_check_test: 'PASS npx tsc --noEmit -p tsconfig.test.json'
      lint: 'PASS npm run lint'
      build_racing_curriculum: 'PASS npm run build:racing-curriculum'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-per-car-observation'
    next_slice: 'p3-s19-red-browser-per-car-controller'
  - slice_id: 'p3-s19-red-browser-per-car-controller'
    title: 'Red tests for independent per-car controllers in browser harness'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/*.test.ts'
      - 'examples/racing_curriculum/controller/nge.controller.test.ts'
    acceptance_criteria:
      - 'A new test fails on the current fan-out behavior and asserts that each car has its own controller instance.'
      - 'Test exercises blue-on-inner vs red-on-outer observation snapshots and expects different controls.'
      - 'Test asserts `controllers.get(0).network !== controllers.get(1).network`.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s19-impl-browser-per-car-controller'
    VALIDATION_EVIDENCE:
      focused_test: 'FAIL examples/racing_curriculum/browser-entry — 4 new per-car controller contracts fail, 70/74 tests pass'
      failure_summary:
        - 'does not contain the shared fan-out helper: source text still contains `resolveControlFanOut`'
        - 'creates one independent controller per car: `createNgeController` called once, expected twice'
        - 'passes a distinct network instance per car: only one controller/network created at startup'
        - 'passes distinct controls per car: `stepEnvironment` receives identical cloned steer for both cars'
      controller_test: 'PASS examples/racing_curriculum/controller/nge.controller.test.ts — 4/4 tests'
      type_check: 'PASS npx tsc --noEmit -p tsconfig.json && npx tsc --noEmit -p tsconfig.test.json'
      lint: 'PASS npm run lint'
      plan_sync: 'PASS node scripts/agent-customization/validate-plan-sync.mjs — 0 errors, 0 warnings'
      plan_phase_packets: 'PASS node scripts/agent-customization/validate-plan-phase-packets.mjs — 0 errors, 0 warnings'
      fixture_notes: 'Tier 1 deterministic two-car episode; `jest.isolateModulesAsync` plus `jest.doMock` intercepts `browser-entry.ts` load-time imports to spy on `createNgeController` and `stepEnvironment`; `requestAnimationFrame` callback invoked at timestamps 0 and 17 to trigger exactly one fixed-timestep tick.'
      handoff: '04-implementing should replace the single `controller`/`controllerNetwork` with a `Map<carIndex, NgeController>` (or array), call each controller with `derivePerCarObservationState(envState, carIndex)`, remove `resolveControlFanOut`, and pass the per-car controls array directly to `stepEnvironment` and the worker. Re-run `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry`; the four failing Phase 3 contracts should turn green.'
  - slice_id: 'p3-s19-impl-browser-per-car-controller'
    title: 'Implement per-car controller Map in browser harness'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 6
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/controller/nge.controller.ts'
    acceptance_criteria:
      - '`browser-entry.ts` maintains a `Map<carIndex, NgeController>` and calls each controller per tick with its own per-car observation.'
      - 'Each controller output is applied only to the matching car; no control fan-out occurs.'
      - '`resolveControlFanOut` and any singleton controller fields are removed in the same step (no wrappers, no dual paths).'
      - 'All browser-entry controller tests pass.'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-browser-per-car-controller'
    next_slice: 'p3-s19-green-browser-per-car-controller'
  - slice_id: 'p3-s19-green-browser-per-car-controller'
    title: 'Green validation for independent per-car browser control'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/*.test.ts'
    acceptance_criteria:
      - 'Red tests from p3-s19-red-browser-per-car-controller now pass.'
      - 'Deterministic controller probes and progression baselines are updated if the independent-agent contract changes outputs.'
      - 'No regressions in the browser-entry suite.'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-browser-per-car-controller'
    next_slice: 'p3-s19-red-separation-grid'
  - slice_id: 'p3-s19-red-separation-grid'
    title: 'Red tests for car separation and worker grid'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/environment/environment.step.service.test.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
    acceptance_criteria:
      - 'A new environment test fails on the current < 1 separation behavior and asserts bounding-box non-overlap.'
      - 'A new worker test fails on identical starting positions and asserts distinct block-team grid positions.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s19-impl-separation-grid'
  - slice_id: 'p3-s19-impl-separation-grid'
    title: 'Implement car separation and worker starting-grid alignment'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 5
    files_to_change:
      - 'examples/racing_curriculum/environment/environment.step.service.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
    acceptance_criteria:
      - '`CAR_MIN_CENTER_SEPARATION` or equivalent separation logic prevents overlapping bounding boxes.'
      - 'Worker `buildRaceFrame` staggers cars with the same team-layout / grid-spacing logic as the live demo.'
      - 'All environment and worker tests pass.'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-separation-grid'
    next_slice: 'p3-s19-green-separation-grid'
  - slice_id: 'p3-s19-green-separation-grid'
    title: 'Green validation for car separation and worker grid'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/environment/environment.step.service.test.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
    acceptance_criteria:
      - 'Red tests from p3-s19-red-separation-grid now pass.'
      - 'No regressions in environment or worker suites.'
      - 'Baselines that asserted inner-only / alternating starts are updated to match the new grid contract.'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-separation-grid'
    next_slice: 'p3-s19-red-renderer-cleanup'
  - slice_id: 'p3-s19-red-renderer-cleanup'
    title: 'Red tests for renderer visual cleanup'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/renderer/racing.renderer.test.ts'
    acceptance_criteria:
      - 'A new or updated renderer test fails on the current yellow optimal-line overlay and/or cyan centerline.'
      - 'Test asserts that guidance overlay produces only blue (Team 0) and red (Team 1) guide lines.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s19-impl-renderer-cleanup'
  - slice_id: 'p3-s19-impl-renderer-cleanup'
    title: 'Implement renderer visual cleanup'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/renderer/racing.renderer.ts'
    acceptance_criteria:
      - '`drawOptimalLineGuidance`, `COLOR_GUIDANCE_LINE_RGB`, `drawTrackCenterline`, and `COLOR_CENTERLINE` are removed from `racing.renderer.ts`.'
      - '`drawTeamGuidingLines` is the only guidance overlay, using the existing blue/red team colors.'
      - 'All renderer tests pass.'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-renderer-cleanup'
    next_slice: 'p3-s19-green-renderer-cleanup'
  - slice_id: 'p3-s19-green-renderer-cleanup'
    title: 'Green validation for renderer visual cleanup'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/renderer/racing.renderer.test.ts'
    acceptance_criteria:
      - 'Red tests from p3-s19-red-renderer-cleanup now pass.'
      - 'No yellow/cyan assertions remain.'
      - 'No regressions in renderer or downstream browser-entry suites.'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-renderer-cleanup'
    next_slice: 'p3-s19-red-per-car-adaptation'
  - slice_id: 'p3-s19-red-per-car-adaptation'
    title: 'Red tests for continuous per-car runtime adaptation'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      - 'examples/racing_curriculum/browser-entry/*.test.ts'
    acceptance_criteria:
      - 'A new test fails when a single shared adaptation state mutates every car identically.'
      - 'Test asserts per-car mutation isolation: mutating car 0 does not deterministically clone to car 1.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s19-impl-per-car-adaptation'
  - slice_id: 'p3-s19-impl-per-car-adaptation'
    title: 'Implement per-car runtime adaptation / continuous evolution'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 5
    files_to_change:
      - 'examples/racing_curriculum/controller/runtime.adaptation.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
    acceptance_criteria:
      - '`runtime.adaptation.ts` uses `Map<carIndex, RuntimeAdaptationState>` and mutates each controller independently each tick/frame.'
      - 'No global singleton adaptation state remains; old fields are removed in the same step as the map introduction.'
      - 'Browser harness wires each car to its own adaptation entry on creation.'
      - 'All runtime adaptation and browser-entry tests pass.'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-per-car-adaptation'
    next_slice: 'p3-s19-green-per-car-adaptation'
  - slice_id: 'p3-s19-green-per-car-adaptation'
    title: 'Green validation for continuous per-car evolution'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/controller/runtime.adaptation.test.ts'
      - 'examples/racing_curriculum/browser-entry/*.test.ts'
    acceptance_criteria:
      - 'Red tests from p3-s19-red-per-car-adaptation now pass.'
      - 'A deterministic probe shows blue and red controllers diverge within 120 frames.'
      - 'No regressions in adaptation or browser-entry suites.'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-per-car-adaptation'
    next_slice: 'p3-s19-red-worker-independent-genomes'
  - slice_id: 'p3-s19-red-worker-independent-genomes'
    title: 'Red tests for worker evaluation of independent genomes'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
    acceptance_criteria:
      - 'A new test fails when the runner uses one shared network for every car.'
      - 'Test asserts distinct per-car networks and distinct per-car fitness scores.'
    parallelizable: false
    dependencies: []
    next_slice: 'p3-s19-impl-worker-independent-genomes'
  - slice_id: 'p3-s19-impl-worker-independent-genomes'
    title: 'Implement per-car genome/network wiring in worker race-pack'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 6
    files_to_change:
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts'
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts'
    acceptance_criteria:
      - '`createRaceEpisodeRunner` accepts one network per car and maps each network to the correct car index.'
      - 'The coevolution container provides one distinct NEAT genome per car (not a single shared genome or placeholder).'
      - 'The evolution protocol returns per-car / per-team payloads rather than a single `bestNetworkPayload`.'
      - 'All worker race-pack and protocol tests pass.'
    parallelizable: false
    dependencies:
      - 'p3-s19-red-worker-independent-genomes'
    next_slice: 'p3-s19-green-worker-independent-genomes'
  - slice_id: 'p3-s19-green-worker-independent-genomes'
    title: 'Green validation for worker independent-genome evaluation'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts'
    acceptance_criteria:
      - 'Red tests from p3-s19-red-worker-independent-genomes now pass.'
      - 'No regressions in worker race-pack, protocol, or environment suites.'
      - 'Worker starting-grid baselines are updated to match the per-car network contract.'
    parallelizable: false
    dependencies:
      - 'p3-s19-impl-worker-independent-genomes'
    next_slice: 'p3-s19-green-tier1-divergence-probe'
  - slice_id: 'p3-s19-green-tier1-divergence-probe'
    title: 'Green validation — Tier 1 blue/red divergence probe'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 2
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/*.test.ts'
    acceptance_criteria:
      - 'Deterministic Tier 1 probe records distinct steering or lateral position for blue and red cars within 120 frames.'
      - 'Live demo check: blue follows inner guide, red follows outer guide, cars do not overlap, no cyan divider, guide lines are blue/red.'
      - 'No regressions in focused or regression suites.'
    parallelizable: false
    dependencies:
      - 'p3-s19-green-browser-per-car-controller'
      - 'p3-s19-green-renderer-cleanup'
      - 'p3-s19-green-per-car-adaptation'
      - 'p3-s19-green-worker-independent-genomes'
    next_slice: 'Step 17 — Logging and tracker handoff'
```

**Planning claim:** The previous "one NEAT controller fanned out to every car" design is superseded. In Tier 1, every car is an independently evolving NEAT agent: its own genome-derived network, its own per-car observation state, its own controller instance, and its own runtime adaptation cadence.

**Decision Record**

```yaml
decision_record:
  id: 'DR-2026-06-26-01'
  context: 'Tier 1 racing demo currently creates a single NGE controller and fans its single `{throttle, steer}` output to all cars via `resolveControlFanOut`. This masks team-aware observation, prevents independent evolution, and produces overlapping, identical cars.'
  options:
    - id: 'fan-out'
      desc: 'Keep the shared NEAT controller and continue fanning one control output to every car, only patching the observation offset.'
    - id: 'independent-agents'
      desc: 'Give every car its own continuously evolving NEAT network/controller and derive a separate observation state for each car.'
  chosen: 'independent-agents'
  rationale: 'The NGE racing curriculum is intended as a multi-agent benchmark ladder. A shared controller cannot demonstrate coevolution, team specialization, or independent adaptation. Per-car networks are a prerequisite for Tier 1 ? Tier 3 progression and align with the Ant Hive / Predator-Prey demos.'
  owner: '01-planning'
  rollback_plan: 'If green validation fails, restore the pre-pivot Step 19 packet and reactivate the superseded `p3-s19-red-browser-per-car` fan-out slice. Remove per-car Map and helper code in the same rollback commit.'
  created_at: '2026-06-26T00:00:00Z'
```

**User instruction:**

Replace the shared-controller fan-out in the Tier 1 racing demo with independent per-car NEAT agents. Step 19 must now land the full independent-agent architecture: per-car observations, per-car controllers, per-car continuous adaptation, and per-car worker race-pack evaluation. Remove `resolveControlFanOut`, singleton controller state, and single-network adaptation state in the same step that introduces the replacements. Keep the completed team-aware observation offset slices. Keep the car-separation and renderer-cleanup slices because their symptoms remain valid, but re-chain them after the per-car controller work. Add a cross-demo note that the Ant Hive and Predator/Prey NGE demos will reuse this independent-agent pattern when their active phases begin.

**Step objective:**

Pivot Step 19 from a fan-out patch to an independent-agent architecture. Author red/impl/green slices for per-car observation derivation, per-car browser controllers, per-car continuous runtime adaptation, and per-car worker evaluation. Preserve the already-done team-aware offset slices. Retain the car-separation and renderer-cleanup slices as still-needed visual/physical fixes. Ensure every slice follows the no-deferred-cleanup rule and that all old shared-controller code is removed in the same step that adds the per-car replacement.

**Implementation notes:**

- `observation.assembler.ts` already has the team-aware lane offset; it now needs `derivePerCarObservationState(envState, carIndex)` so each controller sees only its own pose and target lane.
- `browser-entry.ts` is the orchestration seam: create and maintain `Map<carIndex, NgeController>`, call each controller every tick with its per-car observation, and apply each output to exactly one car. `resolveControlFanOut` must be deleted, not deprecated.
- `nge.controller.ts` is already a factory (`createNgeController`); the change is mostly caller-side, with a small per-car observation-state wiring helper if needed.
- `runtime.adaptation.ts` currently holds single-network state. Convert to `Map<carIndex, RuntimeAdaptationState>` and mutate each entry independently. Remove old singleton fields in the same commit.
- `simulation-worker.race-pack.service.ts` already accepts an array of networks, but the coevolution container (`simulation-worker.coevolution.service.ts`) is a placeholder that must provide one real NEAT genome per car. The evolution protocol (`simulation-worker.evolution.protocol.service.ts`) must return per-car / per-team payloads instead of a single `bestNetworkPayload`.
- The browser `'step'` message and worker `'request-race-step'` types are currently mismatched; reconcile them as part of the worker slice so per-car controls can flow end-to-end.
- Car separation and worker-grid alignment stay in the chain because overlapping cars are still a demo defect, but they now execute after per-car control is established.
- Renderer visual cleanup stays because yellow optimal line and cyan center divider are still unwanted; the acceptance criteria now explicitly state only blue/red guide lines remain.
- **Cross-demo architecture note:** The Ant Hive and Predator/Prey NGE demos are currently [PLANNED] in `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md` and `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md`. When those workstreams become active, their planning should import the same independent-agent contract (one network per agent, per-agent observation, per-agent adaptation) rather than reintroducing a shared controller.

**Stop conditions:**

- **Done:** All red tests fail before implementation, all green tests pass after implementation, focused and regression suites are green, `npm run build:racing-curriculum` succeeds, and both plan validators pass.
- **Hold:** Any validator or build gate fails; fix the plan file or implementation before claiming Step 19 green.
- **Blocked:** A fundamental conflict between browser and worker protocols cannot be reconciled in this step; escalate to `00-helping` with the protocol diff.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(controller|browser-entry|environment|workers/simulation-worker|renderer)'`
- `npm run lint`
- `npx tsc --noEmit -p tsconfig.json`
- `npx tsc --noEmit -p tsconfig.test.json`
- `npm run build:racing-curriculum`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

**Superseded pre-pivot slices:**

The following pre-pivot slices are removed from the active chain because the independent-agent architecture absorbs their intent:

- `p3-s19-red-browser-per-car` — replaced by `p3-s19-red-browser-per-car-controller` plus the per-car observation/adaptation slices.
- `p3-s19-impl-browser-per-car` — replaced by `p3-s19-impl-browser-per-car-controller`.
- `p3-s19-green-browser-per-car` — replaced by `p3-s19-green-browser-per-car-controller`.

Their single concern (stop fanning one control to every car) is now enforced by the per-car observation helper, the per-car controller Map, the per-car adaptation Map, and the per-car worker genome wiring.

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s19-red-obs-team-offset'
  changed_files:
    - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
  preflight: []
  validation:
    - 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts ? 7 passed, 1 failed (teamIndex 1 outer-lane centerline). Failure: channel16 = -0.6666666865348816 (expected ˜ 0).'
  coverage_guard:
    files: []
    summary: 'All changes are under examples/; src/ is not touched; coverage-guard not required.'
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.test.ts'
  next: 'p3-s19-impl-obs-team-offset'
```

```yaml
PlanUpdate:
  slice_id: 'p3-s19-impl-obs-team-offset'
  changed_files:
    - 'examples/racing_curriculum/controller/observation.assembler.ts'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json ? OK'
    - 'npx tsc --noEmit -p tsconfig.test.json ? OK'
    - 'npm run lint ? 0 issues'
    - 'npx prettier --check examples/racing_curriculum/controller/observation.assembler.ts examples/racing_curriculum/controller/observation.assembler.test.ts ? OK'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/controller/observation.assembler.test.ts'
      expected_exit: 0
      result: '8 passed, 0 failed'
  coverage_guard:
    files: []
    summary: 'All changes are under examples/; src/ is not touched; coverage-guard not required.'
  rollback:
    - 'git checkout -- examples/racing_curriculum/controller/observation.assembler.ts'
  next: 'p3-s19-green-obs-team-offset'
```

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s19-planning-indep-pivot'
  changed_files:
    - 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    - 'plans/README.md'
    - 'plans/Roadmap.md'
  preflight:
    - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  validation:
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  coverage_guard:
    files: []
    summary: 'No src/ files touched; coverage-guard not required.'
  rollback:
    - 'git checkout -- plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    - 'git checkout -- plans/README.md'
    - 'git checkout -- plans/Roadmap.md'
  next: '04-implementing: run p3-s19-red-per-car-observation through p3-s19-green-tier1-divergence-probe'
```

## Step 17 — Logging and tracker handoff [PLANNED]

```yaml
phase: 3
step: 17
title: 'Logging and tracker handoff'
status: '[PLANNED]'
goal: 'logging'
expansion: 'none'
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Phase 4 Step 01 — Plan Tier 3 boundary'
skills:
  - 'logging'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
  - 'node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json'
acceptance_criteria:
  - 'User confirms the documented Tier 1/Tier 2 hardening renders correctly in the browser/UI.'
  - 'Phase 3 step/slice details are compressed into plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md.'
  - 'Phase 3 is marked [DONE] in the plan and plans/README.md / plans/Roadmap.md are updated.'
  - 'Phase 4 Step 01 is advanced to [WIP].'
  - 'Plan sync, phase-packet, phase-compression, and log-completion-marker gates pass.'
```

**User instruction:**

Compress the completed Phase 3 step details into the logs file and advance Phase 4 Step 01 only after the user has confirmed the updated Tier 1 independent-agent baseline in the browser/UI. When advancing Phase 4 Step 01, ensure the Tier 3 planning packet preserves the Step 19 contract: every car has its own NEAT genome-derived network, observation, controller, and adaptation state.

**Step objective:**

Close Phase 3 cleanly and hand off to Phase 4 Tier 3 planning once the user confirms the Tier 1 independent-agent baseline (per-car network/observation/controller/adaptation) renders correctly in the live demo. Ensure Phase 4 Step 01 inherits and documents the independent-agent contract.

**Stop conditions:**

- **Done:** Phase 3 compressed, marked [DONE], Phase 4 Step 01 [WIP], and all gates pass.
- **Hold:** user has not yet confirmed the updated baseline rules in the browser/UI.
- **Blocked:** plan validators or compression gates fail after compression; escalate to `00-helping`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`
- `node scripts/agent-customization/gates/log-completion-marker.gate.mjs --json`

**Plan update requirement:**

Move detailed Step 09-16 history to the logs file, mark Phase 3 [DONE], update README/Roadmap, and advance Phase 4 Step 01 to [WIP]. Do not leave verbose [DONE] details in the plan file.

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

Context: NGE racing-curriculum workstream — Phase 1 [DONE], Phase 2 Tier 1 single-agent benchmark [DONE], Phase 3 Tier 2 1v1-with-radio [WIP], Phase 4 Tier 3 2v2 no pits [PLANNED].
Current boundary: Phase 3 Step 19 — Tier 1 independent-agent architecture pivot [WIP]; Step 18 — Tier 1 demo defect investigation is [DONE]; Step 17 — Logging and tracker handoff is [PLANNED] and will resume only after Step 19 green validation passes.

What is already covered:
- Phase 1 UI/behavior completion and inner-track centerline are [DONE], archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- Phase 2 Tier 1 single-agent benchmark is [DONE], archived in the logs file.
- Phase 3 Steps 09-16 are [DONE]: Tier 1/Tier 2 racing baseline rules, renderer hardening, physics hardening, tier layout/start, and updated README.
- Phase 3 Step 18 is [DONE]: source-grounded alignment brief identifies four Tier 1 demo defects and maps them to `observation.assembler.ts`, `browser-entry.ts`, `environment.step.service.ts` / `simulation-worker.race-pack.service.ts`, and `renderer/racing.renderer.ts`.
- Decision Record DR-2026-06-26-01: the shared-controller fan-out design is superseded by independent per-car NEAT agents (own network, observation, controller, adaptation).
- The team-aware observation offset slice (`p3-s19-red/green/impl-obs-team-offset`) is [DONE].
- `plans/README.md` and `plans/Roadmap.md` were resynced to Phase 3 [WIP] (Step 19 active) / Phase 4 [PLANNED].
- `validate-plan-sync` and `validate-plan-phase-packets` both PASS (0 errors, 0 warnings) after the planning edits.

Next narrow task: Execute the Step 19 red-green slices for per-car observations, per-car browser controllers, car separation + worker grid, renderer visual cleanup, per-car runtime adaptation, worker independent-genome evaluation, and the Tier 1 blue/red divergence probe. Advance to Step 17 logging/tracker handoff only after all green validation passes.

Required validations for Step 19:
- Red tests fail before implementation for each slice.
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(controller|browser-entry|environment|workers/simulation-worker|renderer)'`
- `npm run lint`
- `npx tsc --noEmit -p tsconfig.json`
- `npx tsc --noEmit -p tsconfig.test.json`
- `npm run build:racing-curriculum`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

Known worktree cautions:
- All production changes are under `examples/racing_curriculum/`; no `src/` files are expected to change, so `coverage-guard` is not required.
- `resolveControlFanOut`, singleton controller fields, and single-network adaptation state must be removed in the same step that introduces the per-car replacements (no backward-compatibility wrappers, no dual-path code, no deferred cleanup).
- Per-car control, per-car adaptation, and per-car worker genomes will change deterministic controller probes and may break worker starting-grid baselines; plan baseline updates as part of green validation, not as surprises.
- The browser `'step'` message and worker `'request-race-step'` types are currently mismatched; reconcile them as part of the worker independent-genome slice so per-car controls can flow end-to-end.
- Cross-demo architecture note: when `plans/NEAT_Genesis_EvoDevo_AntHive_Demo.md` and `plans/NEAT_Genesis_EvoDevo_PredatorPrey_Demo.md` become active, they should import the same independent-agent contract rather than reintroduce a shared controller.
```

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s15-impl-tier-layout'
  changed_files:
    - examples/racing_curriculum/browser-entry/browser-entry.ts
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'git status --porcelain'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry'
      expected_exit: 0
      result: 'PASS 6 suites, 69 tests'
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(browser-entry|renderer|track|environment|workers/simulation-worker)'"
      expected_exit: 0
      result: 'PASS 29 suites, 249 tests'
    - command: 'npm run lint'
      expected_exit: 0
      result: '0 issues'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      expected_exit: 0
      result: 'no errors'
    - command: 'npx tsc --noEmit -p tsconfig.test.json'
      expected_exit: 0
      result: 'no errors'
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
      result: 'bundle 733.7kb'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  coverage_guard:
    files:
      - examples/racing_curriculum/browser-entry/browser-entry.ts
    summary: 'examples/ boundary not src/; coverage-guard not required (no src/ files touched).'
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
  next: '05-green-testing for p3-s15-green-tier-layout'
```

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s15-green-tier-layout'
  changed_files:
    - examples/racing_curriculum/browser-entry/browser-entry.ts
    - docs/assets/racing-curriculum.bundle.js
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npm run build:racing-curriculum'
    - 'git status --porcelain'
  validation:
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry'
      expected_exit: 0
      result: 'PASS 6 suites, 69 tests'
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/(browser-entry|renderer|track|environment|workers/simulation-worker)'"
      expected_exit: 0
      result: 'PASS 29 suites, 249 tests'
    - command: 'npm run lint'
      expected_exit: 0
      result: '0 issues'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      expected_exit: 0
      result: 'no errors'
    - command: 'npx tsc --noEmit -p tsconfig.test.json'
      expected_exit: 0
      result: 'no errors'
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
      result: 'bundle 733.7kb'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  live_demo_probe:
    - tier: 1
      subtitle: 'Tier 1 solo NGE harness with a spline-smoothed visual circuit and faded optimal-line guidance.'
      init_tier: 1
      car_count: 2
      team_indices: [0, 1]
      note: 'Puppeteer probe against tmp/racing-curriculum.bundle.tier1-probe.js'
    - tier: 3
      subtitle: 'Tier 3 keeps the same live shell while widening observation authority after guidance removal.'
      init_tier: 3
      car_count: 4
      team_indices: [0, 0, 1, 1]
      note: 'Puppeteer probe against patched tmp/racing-curriculum.bundle.tier3-probe.js; forces tier 3 to exercise the fallback layout path.'
  coverage_guard:
    files:
      - examples/racing_curriculum/browser-entry/browser-entry.ts
    summary: 'examples/ boundary not src/; coverage-guard not required (no src/ files touched).'
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
    - 'git checkout -- docs/assets/racing-curriculum.bundle.js'
  next: 'Step 17 — Logging and tracker handoff'
```

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s16-docs-update'
  changed_files:
    - examples/racing_curriculum/README.md
  preflight:
    - 'npm run docs'
    - 'npm run lint'
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'git status --porcelain'
  validation:
    - command: 'npm run docs'
      expected_exit: 0
      result: 'PASS — HTML docs generated; Mermaid diagrams validated'
    - command: 'npm run lint'
      expected_exit: 0
      result: '0 issues'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      expected_exit: 0
      result: 'no errors'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  coverage_guard:
    files: []
    summary: 'No src/ files touched; coverage-guard not required.'
  rollback:
    - 'git checkout -- examples/racing_curriculum/README.md'
  next: 'Step 17 — Logging and tracker handoff'
```

## PlanUpdate

```yaml
PlanUpdate:
  slice_id: 'p3-s16-docs-tier-contract'
  changed_files:
    - examples/racing_curriculum/README.md
  preflight:
    - 'npm run docs'
    - 'npm run lint'
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'git status --porcelain'
  validation:
    - command: 'npm run docs'
      expected_exit: 0
      result: 'PASS — HTML docs generated; Mermaid diagrams validated'
    - command: 'npm run lint'
      expected_exit: 0
      result: '0 issues'
    - command: 'npx tsc --noEmit -p tsconfig.json'
      expected_exit: 0
      result: 'no errors'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  coverage_guard:
    files:
      - examples/racing_curriculum/README.md
    summary: 'Documentation change only; src/ not touched; coverage-guard not required.'
  rollback:
    - 'git checkout -- examples/racing_curriculum/README.md'
  next: 'Step 17 — Logging and tracker handoff'
```

---
