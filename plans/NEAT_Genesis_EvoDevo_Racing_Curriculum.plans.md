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

- **Phase 1 is [DONE].** Step 01-04 and all slices passed green validation. User confirmed the right-side network panel live-value refresh and the inner-track guidance overlay. Archive is in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- **Phase 2 — Tier 1: Single agent on simple track** is [DONE]. Step 01-07 all passed; Phase 2 history is compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- **Phase 3 — Tier 2: Single agent with radio** is [WIP]. Step 01 — Plan Tier 2 boundary is now the active frontier.
- **Step 05 visual confirmation:** Browser-ui-specialist confirmed two cars render with cyan (Team A) and magenta (Team B) guiding lines, no Phase 1 regressions, and only minor viewport/alpha observations (see Step 05 evidence block).
- Tier 1–6 ladder, promotion rules, and carry/reset policy are defined in this plan and sourced from `examples/racing_curriculum/reference.plans.md`.
- Upper-tier features still depend on NGE primitives that may be experimental or missing (`ModulatorBroadcaster`, `EpisodicSlot`, `GatingRouter`, polyandric reproduction wiring). Those are routed to `nge-core-algorithm`, not compensated for locally.

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

### Phase 3 — Tier 2: Single agent with radio [WIP]

```yaml
phase: 3
title: 'Tier 2: Single agent with radio'
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
  - 'Tier 2 green gate passes: a single car per team uses the radio field as a self-monitoring signal; no pits, no tire degradation.'
  - 'Phase 2 is [DONE] before Phase 3 starts.'
placeholder_steps:
  - 'Step 01 — Plan Tier 2 boundary'
  - 'Step 02 — Research radio self-signal contracts'
  - 'Step 03 — Red tests for radio field wiring'
  - 'Step 04 — Implement single-car radio loop'
  - 'Step 05 — Green validation and regression triage'
  - 'Step 06 — Document Tier 2 contract'
  - 'Step 07 — Logging and tracker handoff'
```

**Phase objective:** Add the team radio field with one car per team. The car
learns to read and write radio as a self-monitoring signal (pace intent, threat
level). No pit/tire complexity yet.

**Stop conditions:**

- **Done:** Phase 2 is [DONE] and Tier 2 green gate passes.
- **Hold:** user must confirm radio-field dimensions and self-signal semantics.
- **Blocked:** upstream NGE primitive missing; route to `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 2 green gate (defined when phase is expanded).

#### Step 01 — Plan Tier 2 boundary [WIP]

```yaml
phase: 3
step: 1
title: 'Plan Tier 2 boundary'
status: '[WIP]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 02 — Research radio self-signal contracts'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Tier 2 boundary is sequenced after Phase 2 [DONE].'
  - 'Single-car-per-team, radio-on, no-pits/no-tires baseline is recorded.'
  - 'Radio field dimensions and self-signal semantics are captured or marked user-hold.'
  - 'Phase 3 Step 02-07 packets are authored and conform to schema.'
```

**Step objective:** Close the Tier 2 planning boundary: confirm Phase 2 completion, record the single-car-per-team radio-on/no-pits baseline, capture radio-field dimensions and self-signal semantics, and author Step 02-07 packets before implementation begins.

**Context the agent must know:**

- Phase 2 is [DONE]; Tier 1 single-agent benchmark passed with worker-authoritative race episode runner and per-agent guiding lines.
- Tier 2 reference design is in `examples/racing_curriculum/reference.plans.md`.
- Phase 3 adds the team radio field as a self-monitoring signal; no pits or tire degradation yet.
- Do not patch missing NGE primitives locally; record blockers and escalate to `nge-core-algorithm`.

**Execution steps:**

1. Read `examples/racing_curriculum/reference.plans.md` Tier 2 section.
2. Inspect current radio-field wiring in `examples/racing_curriculum/`.
3. Confirm single-car-per-team pack shape, radio-on/no-pits/no-tires feature flags, and observation/action boundaries.
4. Define radio field dimensions and self-signal semantics (pace intent, threat level).
5. Check for missing NGE primitives and route to `nge-core-algorithm` if found.
6. Author Step 02-07 packets with acceptance criteria and validation commands.

**Stop conditions:**

- **Done:** assumptions recorded, no unresolved blockers, Step 02-07 packets authored, plan validators pass.
- **Hold:** user must confirm radio-field dimensions or self-signal semantics before Step 02 starts.
- **Blocked:** upstream NGE primitive missing and not yet escalated; route to `nge-core-algorithm`.
- **Route-back:** return to `01-planning` if packet schema validation fails.

**User instruction:** Plan the Tier 2 single-car-with-radio boundary. Confirm Phase 2 completion, record the radio-on/no-pits baseline and self-signal semantics, check for upstream NGE blockers, author Step 02-07 packets, and run plan validators before handing off to Step 02.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

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
  - 'Tier 3 green gate passes: two identical-DNA teammates per team develop distinct behavioral specializations through experience; no pits.'
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

**Phase objective:** First appearance of role differentiation. Two identical-DNA
teammates per team must develop different behavioral specializations through
experience on an intermediate track with genuine overtaking zones. No pit timing
complexity.

**Stop conditions:**

- **Done:** Phase 3 is [DONE] and Tier 3 green gate passes.
- **Hold:** user must confirm team-fitness semantics for 2v2.
- **Blocked:** upstream NGE primitive missing; route to `nge-core-algorithm`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Tier 3 green gate (defined when phase is expanded).

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
- 2026-06-25: Step 07 completed — Phase 2 marked [DONE], compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`, Phase 3 Step 01 advanced to [WIP]. `workflow-update-sync`: pass (phase boundary detected, next boundary Phase 3 Step 01). `phase-compression.gate`: pass. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).

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

- 2026-06-25: Step 07 completed — Phase 2 marked [DONE], compressed into `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`, Phase 3 Step 01 advanced to [WIP]. `workflow-update-sync`: pass (phase boundary detected, next boundary Phase 3 Step 01). `phase-compression.gate`: pass. `validate-plan-sync`: PASS (0 errors, 0 warnings). `validate-plan-phase-packets`: PASS (0 errors, 0 warnings).
- 2026-06-25: Step 05 visual fix applied — `resolveCurriculumRacePackLayout` returns `[0, 1]` for Tier 1, producing a two-car 1v1 pack. Added regression test in `browser-entry.progression.test.ts`. Rebuilt `docs/assets/racing-curriculum.bundle.js`. Focused racing-curriculum tests: 38 suites / 247 tests PASS. Type check (`tsconfig.json`, `tsconfig.test.json`) PASS. Lint PASS.
- 2026-06-25: Step 04 compressed to `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`;
  Step 05 opened as [WIP] for user visual confirmation of Tier 1 guiding lines.
  `validate-plan-sync`: PASS — 0 errors, 0 warnings.
  `validate-plan-phase-packets`: PASS — 0 errors, 0 warnings.
  `phase-compression.gate`: PASS.
- 2026-06-25: Plan sync validator pass: 0 errors, 0 warnings (after Roadmap + README update).
- 2026-06-25: Plan phase-packet validator pass: 0 errors, 0 warnings (Phase 2 [WIP], Step 04 red-green slices conform to schema).
- 2026-06-25: Workflow sync attempted to advance Phase 2 Step 4 → [DONE]; Phase 2 Step 5 → [WIP]. This advance was incorrect because Step 04 still has planned `p2-04-red-guiding-lines` / `p2-04-impl-guiding-lines` / `p2-04-green` slices. Statuses reverted to Step 04 [WIP], Step 05 [PLANNED].
- 2026-06-25: Workflow sync: Advanced Phase 2 Step 3 → [DONE]; Phase 2 Step 4 → [WIP]
- 2026-06-25: Workflow sync: Advanced Phase 2 Step 1 → [DONE]; Phase 2 Step 2 → [WIP]

## Handoff query

```text
Continue from the current repo state only. Do not rely on prior chat history.

Context: NGE racing-curriculum workstream — Phase 1 [DONE], Phase 2 Tier 1 single-agent benchmark [DONE], Phase 3 Tier 2 single-car-with-radio [WIP].
Current boundary: Phase 3 Step 01 — Plan Tier 2 boundary [WIP].

What is already covered:
- Phase 1 UI/behavior completion and inner-track centerline are [DONE], archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- Phase 2 Tier 1 single-agent benchmark is [DONE]: worker-authoritative 2-car race-pack service (33 tests), per-agent cyan/magenta guiding lines (47 focused tests), browser-ui-specialist visual confirmation, Tier 1 README contract, all compressed to the logs file.
- Phase 2 Step 07 (logging and tracker handoff) is [DONE]; plan-sync and phase-packet gates passed.

Next narrow task: Plan the Tier 2 single-car-with-radio boundary. Confirm Phase 2 completion, record the radio-on/no-pits/no-tires baseline and self-signal semantics, check for upstream NGE blockers, author Phase 3 Step 02-07 packets, and run plan validators before handing off to Step 02.

Required validations before claiming Phase 3 Step 01 ready to close:
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

Known worktree cautions:
- Phase 1 and Phase 2 must not regress during Phase 3 work.
- Do not advance to Step 02 until Step 01 packets pass plan validators and any user holds are cleared.
- Missing NGE primitives must be escalated to nge-core-algorithm, not patched locally in the demo.
```

## PlanUpdate

```yaml
PlanUpdate:
  changed_files:
    - examples/racing_curriculum/browser-entry/browser-entry.ts
    - examples/racing_curriculum/browser-entry/browser-entry.progression.test.ts
    - docs/assets/racing-curriculum.bundle.js
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
  validation:
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='(racing\\.renderer|simulation-worker\\.race-pack|browser-entry)' --runInBand"
      expected_exit: 0
    - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum --runInBand'
      expected_exit: 0
  rollback:
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.ts'
    - 'git checkout -- examples/racing_curriculum/browser-entry/browser-entry.progression.test.ts'
    - 'git checkout -- docs/assets/racing-curriculum.bundle.js'
  next: 'browser-ui-specialist visual re-verification of two-car cyan/magenta guiding lines in live demo'
```
