# NEAT Genesis EvoDevo: Core Readiness â€” Racing Curriculum

**Status:** [DONE]

## Scope

Canonical long-form readiness plan for the NGE team-adversarial racing curriculum.
This plan is the single source of truth for the UI-first demo polish and the
Tier 1â€”6 ladder defined in `examples/racing_curriculum/reference.plans.md`.
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

- `examples/racing_curriculum/reference.plans.md` defines the Tier 1â€”6 ladder,
  team structure, radio semantics, tire/pit design, promotion rules, carry/reset
  policy, and acceptance criteria used below.
- `examples/flappy_bird/` is the UI parity baseline for the Phase 1 demo polish.

## Current state

- **Phase 9 - NGE Core Extraction + Driving Improvement + Growth Acceleration [DONE].** All 7 steps [DONE] and green-validated. 341 tests pass, 100% coverage on 6 src/ files, tsc/lint/build pass, browser smoke pass. All step details archived in logs.
- **Phases 1-9 [DONE]** and compressed in plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md.
- **Browser demo path:** examples/racing_curriculum/index.html exercises live per-car runtime adaptation via createPerCarAdaptationEngines + runNgeGrowStabilizeCycle (core module extracted to src/neat/nge-juvenile/).

All phases complete. Plan ready for archival to plans/completed/.

### Research findings summary

- User-reported Tier 2 demo defects investigated (02-research): 2 cars in Tier 2 is correct per reference design; subtitle copy corrected; per-car control wiring and off-track enforcement addressed in Phase 3 Steps 13-19.
- NGE shared-controller architectural audit: racing demo had shared controller fan-out (resolveControlFanOut); fixed in Phase 3 Step 19 (independent per-car NEAT agents). Other NGE demos (AntHive, PredatorPrey) are planned-only with no runnable code.
- Full research evidence in docs/research/racing-curriculum-tier1-demo-defects.md and plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md.

## Non-goals

- Do not hand-code queen, blocker, pacer, or pit-strategy roles.
- Do not claim Tier 4â€”6 completion before the required NGE primitives are confirmed
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

| Tier | Cars per team | Radio | Tires/pits | Sensory leap                         | N_floor (median hidden nodes) | Est. duration (generations) | Track                                | Purpose                             |
| ---- | ------------- | ----- | ---------- | ------------------------------------ | ----------------------------- | --------------------------- | ------------------------------------ | ----------------------------------- |
| 1    | 1             | off   | off        | 70-in/2-out baseline                 | 90 â†’ 1000               | ~10â€“15               | simple oval/flowing circuit          | single-car NGE learns to drive      |
| 2    | 1             | on    | off        | +radio self-signal                   | 2,000                         | ~20â€“30               | simple circuit with one tight corner | self-monitoring radio signal        |
| 3    | 2             | on    | off        | +teammate awareness, role divergence | 8,000                         | ~35â€“50               | intermediate with overtaking zones   | first role differentiation          |
| 4    | 2             | on    | on         | +tire/pit episodic memory            | 20,000                        | ~50â€“70               | intermediate with pit tradeoffs      | tire budget + pit blocking          |
| 5    | 3             | on    | on         | +full team coordination, polyandric  | 40,000                        | ~70â€“90               | full competition circuit             | full NGE team racing                |
| 6    | 3             | on    | on         | +hall-of-fame arms race              | 75,000                        | ~90â€“120              | full circuit, multi-window strategy  | sustained co-evolutionary arms race |

Beyond racing, ant-hive demo continues 75K â†’ 150K â†’ 250K under headless/offline
evaluation. The 250k-node aspirational target is the ant-brain anchor; practical
racing milestones climb 90â†’1kâ†’2kâ†’8kâ†’20kâ†’40kâ†’75k. A 250k-node browser racing sim
at 30fps is infeasible, so the architecture is scale-agnostic. Density target
band: 800â€“3,000 synapses/neuron.

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
- **Capacity floor gate** (DR-003): in addition to reliability, the
  team's median hidden-node count must meet or exceed the tier's `N_floor` before
  promotion is granted. This makes structural growth a necessary condition for
  advancement, not merely a side effect. A team that is reliable but undersized
  holds at the current tier until its median node count reaches the floor.
- **Growth-velocity gate** (DR-003): advancement also requires a
  minimum growth-velocity floor (median nodes gained per generation over the tier
  window). When growth stalls â€” velocity drops below the floor while the team is
  still below `N_floor` â€” the tier duration auto-extends and the per-tier
  growth-morph budget is boosted to reignite structural expansion. Adaptive tier
  duration replaces fixed generation counts: a team that grows steadily promotes
  faster; a team that stalls gets more time and a bigger morph budget before
  timing out.

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

## Growth-drive policy (DR-003)

NGE networks start at ~90 nodes and need a strong drive to grow toward ant-brain
complexity (~250k neurons). Current tier advancement is reliability-only with no
growth, neuron-count, or growth-velocity metric. The following composite policy
addresses this gap. It is entirely a config/knob layer over existing plumbing
(`planGrowthMorphs`, `computeFocusScores`, advancement gates) â€” no new structural
code is required, and rollback is reverting the knobs to defaults.

### Focus-weight retune (Approach B â€” the engine)

Only NGE-idiomatic mechanism: growth belongs to lifecycle policy, uses existing
`planGrowthMorphs`/`computeFocusScores`, is opt-in and deterministic.

- **Lower `wiringCost`** weight from the default to reduce the anti-growth bias
  in the focus score.
- **Raise `novelty`** weight to reward structural exploration.
- **Add a `capacity` term** to the focus-score composition, rewarding networks
  that productively use additional hidden nodes.
- **Raise the reward-delta floor** above 0.0 so that small improvements are not
  discarded; this encourages incremental structural growth.
- **Raise the tier-scaled juvenile growth-morph budget** so that younger networks
  in a new tier receive more aggressive morph allocation, front-loading growth.
- The `computeFocusScores` weights must be externally configurable (confirmed in
  Step 02 research) so that per-tier knob overrides are possible without code
  changes.

Default focus weights (to be retuned): `{ w_u: 0.25, w_r: 0.3, w_n: 0.2, w_s: 0.15,
w_c: 0.1 }`. Rollback: revert to these defaults.

### Fitness complexity bonus (Approach A â€” the accelerator)

Performance-gated complexity bonus in the fitness function. Selection pressure
alone selects for useful capacity:

- A **parsimony density pressure** term keeps nets ant-brain-efficient and
  prevents bloat: networks are rewarded for productive node use, not raw size.
- The complexity bonus is gated on performance â€” a network must demonstrate
  improved racing behavior (lap time, obstacle avoidance, team coordination) to
  earn the bonus, preventing pure bloat strategies.
- This selects for networks that **use** their capacity effectively, not merely
  networks that grow.

### Milestone ladder

The growth-drive milestone ladder is embedded in the tier ladder summary table
above (N_floor and Est. duration columns). Practical racing milestones climb
90â†’1kâ†’2kâ†’8kâ†’20kâ†’40kâ†’75k. Beyond racing, ant-hive demo continues 75Kâ†’150Kâ†’250K
under headless/offline evaluation. The 250k target is aspirational; browser racing
at 250k nodes/30fps is infeasible, so the architecture is scale-agnostic.

**Browser performance caps:** 8,000 hidden nodes is the practical browser racing
ceiling at 30fps. If performance degrades, cap at 2,000. Tiers requiring more than
8k nodes run headless/offline. See "User vision clarification" above.

### Density band

Target density band: **800â€“3,000 synapses/neuron**. Networks significantly below
this band (too sparse) or above it (too dense/bloated) are penalized by the
parsimony density pressure. The density band is a soft target, not a hard
constraint â€” it guides the fitness complexity bonus without blocking advancement.

### Approach D â€” deferred

Structural depth motifs (rewarding multi-module depth architectures or adding a
new add-layer mutation) are deferred. Let topology search discover depth
naturally under the new growth pressure. Revisit D1 (depth-motif reward) only if
width saturates before reaching a tier's `N_floor`. D2 (add-layer mutation) is
premature and risks determinism contracts.

## Worker integration policy (DR-002)

The racing curriculum browser demo runs continuous per-tick host-side Network
mutation via `RuntimeAdaptationEngine.adaptOnTick`. The user explicitly wants
continuous real-time evolution per agent, not discrete NEAT generations. The
decision is to **relocate** `RuntimeAdaptationEngine` into the worker (not convert
to generational). The worker owns the network and runs continuous adaptation
internally, mirroring how Flappy Bird's evolution worker owns everything while the
main thread handles UI/render only.

### Four-axis worker migration

1. **Protocol-shape**: replace the POC step-only bootstrap with the FSM router;
   fix the silently-dropped `{type:'init'}` initialization message so the worker
   receives the full race configuration before stepping.
2. **Controller-authority**: move per-car `Network` references and
   `computeControlWithEvidence` into the worker. The worker owns all per-car
   networks; the main thread no longer holds Network instances or calls
   `computeControlWithEvidence` directly.
3. **Render-path**: consume packed `RacingRenderFrame` typed-array frames
   produced by the worker. The main thread reads render frames and draws to
   canvas; it does not compute simulation state.
4. **Adaptation-authority**: relocate `RuntimeAdaptationEngine` into the worker
   alongside the networks. Continuous adaptation runs inside the worker per tick,
   not on the host main thread. This eliminates the NGE anti-pattern concern
   (host-side per-tick in-place mutation) while preserving the real-time
   evolution experience.

### Rollback

Revert to host-main-thread synchronous activation + host-side adaptation. Remove
worker protocol wiring. The POC physics-only worker path remains as fallback.

## User vision clarification

The user has clarified the fundamental NGE vision. This section is authoritative
for all downstream implementation and must not be contradicted by prior NGE core
design assumptions.

### Organic growth from a seed

NGE networks are NOT static. They start small from a seed network and grow
organically through continuous real-time adaptation. The inspiration is an ant's
brain at a much smaller scale â€” 3D neural networks that mimic the structure of
an ant's brain with a strong capacity to adapt to a changing environment. The
real-time adaptation adds or prunes layers as needed. This is the core mechanism,
not a side effect.

### Continuous adaptation is primary; generations are secondary

- **Continuous real-time adaptation** is the primary evolution mechanism. Agents
  adjust their own values in real time during simulation. No manual controller
  panel â€” agents self-regulate automatically.
- **Generations are optional**, not mandatory for an agent to evolve. When used,
  they serve as a way for agents to multiply and fuse successful networks so they
  can evolve positive traits. One generation per lap is the suggested cadence.
- A session should require roughly 10â€“15 laps/generations (or whatever number it
  takes) to achieve sufficient growth to safely advance to the next tier.
- Static networks are NOT the user's vision. The NGE system must embody continuous
  growth and adaptation.

### NEAT may be refactored

If the original NEAT implementation assumed generational-only evolution, that
was a misunderstanding of the user's purpose. NEAT/NGE core code may be refactored
as needed to achieve this vision. Continuous adaptation and organic growth are
first-class requirements, not anti-patterns to be avoided.

### Browser performance cap

- 8,000 hidden nodes is a reasonable ceiling for browser racing at 30fps.
- If performance becomes a problem, cap at 2,000 hidden nodes.
- The milestone ladder and capacity-gate N_floor values should respect these
  browser caps. Tiers beyond what the browser can sustain run headless/offline.

## Implementation phases

### Phase 1 â€” Racing UI/behavior completion to Flappy Bird parity and inner-track centerline [DONE]

[DONE] Phase 1 Step 01-04 completed and validated. Detailed step/slice content, validation evidence, and PlanUpdate blocks are archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md` under "Phase 1 â€” Final step/slice archive (Step 01-04)".

- User confirmed the right-side network panel live-value refresh and the inner-track guidance overlay.
- All focused tests passed, bundle rebuilt, folder-quality gate passed.
- Phase 2 remains [PLANNED] and will be advanced separately by 01-planning.

### Phase 2 â€” Tier 1: Single agent on simple track [DONE]

[DONE] Phase 2 Step 01-07 completed and validated. Detailed step/slice content, validation evidence, and PlanUpdate blocks are archived in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md` under "Phase 2 â€” Tier 1: Single agent on simple track [DONE] â€” Final archive".

- Tier 1 single-agent benchmark: deterministic 2-car 1v1 pack on inner-lane centerline, worker-authoritative race episode runner, lap detection, lap-time fitness, and per-agent cyan/magenta guiding lines all passed.
- Browser-ui-specialist confirmed two cars render with cyan (Team A) and magenta (Team B) guiding lines, no Phase 1 regressions.
- Tier 1 usage contract documented in `examples/racing_curriculum/README.md`.
- Phase 3 advanced to [WIP]; Step 01 â€” Plan Tier 2 boundary is the active frontier.

### Phase 3 -- Tier 2: 1v1 with radio (one car per team) [DONE]

[DONE] Steps 01-19 completed and validated. Tier 2 1v1 radio, racing baseline rules, renderer/physics hardening, tier layout, demo defect investigation, and independent-agent architecture pivot (DR-011) all green-gated. Detailed step/slice content archived in plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md.

### Phase 4 -- Tier 3: 2v2 no pits [DONE]

[DONE] Steps 01-07 completed and validated. 4-car 2v2 coevolution with role divergence, shared-equal team fitness (DR-001), and worker-side adaptation. 45 suites / 385 tests pass. Detailed content archived in logs.

### Phase 5 -- Tier 4: 2v2 tires and pits [DONE]

[DONE] Steps 01-07 completed and validated. Tire degradation, pit-stop mechanics, 95-channel observation, grip multiplier. 45 suites / 385 tests pass. Detailed content archived in logs.

### Phase 6 -- Tier 5: 3v3 full [DONE]

[DONE] 6-car coevolution, full 3-row radio, role-divergence observables, pit-overlay fix. 46 suites / 394 tests pass. Polyandric reproduction DEFERRED (P1/P2 blockers). DR-006/DR-007 recorded. Detailed content archived in logs.

### Phase 7 -- Tier 6: 3v3 advanced strategy [DONE]

[DONE] Steps 01-07 completed. FSM 5-bug fix, hall-of-fame wiring (OpponentSnapshotPool), strategy-divergence analytics. modeIsEvolvable BLOCKED (DR-008). Carry-forward blockers P1-P5 documented. Detailed content archived in logs.

### Phase 8 — Racing Curriculum v2 [DONE]

**Phase objective:** Continue v2 hardening. Steps 01-17 archived in logs. Steps 18-19 fixed worker-authoritative demo evolution and Tier 1 follow-up defects. Step 20 fixed the network growth blocker. Step 21 fixed the driving improvement blocker (forward-pass evaluation, physics rewards, tier promotion gates). Step 22 fixed adaptation stabilization and reward shaping (13 fixes). Step 23 optimized growth rate (adaptive hysteresis, time-boxed stabilization, TIER_N_FLOOR[1]=1000, lap time display). All 23 steps [DONE] and green-validated. All step details archived in logs.

[DONE] Phase 8 Steps 01-17: all step/slice details and validation evidence archived in plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md.

[DONE] Step 18: Worker-authoritative demo evolution and pit-trap fix. All slices [DONE]. Green: 37 suites / 296 tests, build 760.1kb, lint/tsc clean, browser smoke pass. See logs section Phase 8 Steps 18-19 -- Detailed archive.

[DONE] Step 19: Tier 1 demo follow-up defect hardening. All slices [DONE]. All five ACs pass. Green: 88+32+42 tests, build 760.1kb, lint/tsc clean, browser smoke pass. See logs section Phase 8 Steps 18-19 -- Detailed archive.

[DONE] Step 20: Fix network growth blocker -- network-aware adaptation evaluation. All 5 fixes implemented (network-aware evaluator, tier promotion structure preservation, composite RacingQualitySignal, episodic slots, explicit config). Green: 168 tests across 14 suites, build 760.2kb, lint/tsc clean, browser smoke N109/C420 -> N523/C1524 at ~60 FPS, 0 console errors. See logs section Phase 8 Step 20 -- Detailed archive.

[DONE] Step 21: Driving improvement blocker fix. All 8 ACs pass. Forward-pass evaluation, performance-gated complexityBonus, physics rewards, per-car RacingQualitySignal, tier promotion gates, agent selection, behavioral diversity. See logs section Phase 8 Steps 21-22 -- Detailed archive.

[DONE] Step 22: Adaptation stabilization and reward shaping. 13 fixes across 3 slices + fix slices: stabilization tuning (hysteresis 0->5, cooldown 5->40, improvement threshold 0->0.01, MAX_EPISODIC_SLOTS 100->15, plateau detector), reward shaping (OFF_TRACK -1->-5, WRONG_DIR -1->-5, physics weight 0.1->0.3, guide-following reward, guide divergence penalty, escalating border penalties), evaluator architecture (separate baseline/candidate score windows). 129/129 tests, tsc/lint clean, browser smoke N76->N82 growth confirmed. Commit 737e4f49. See logs section Phase 8 Steps 21-22 -- Detailed archive.


[DONE] Step 23: Growth rate optimization and Tier 1 completion criteria. Adaptive hysteresis (2/3/5 based on node count), time-boxed stabilization (min 5, max 25 ticks), PLATEAU_WINDOW_SIZE=5, PLATEAU_VARIANCE_THRESHOLD=0.1, TIER_N_FLOOR[1]=1000, lap time display. Green: 144/144 tests, tsc/lint clean, browser smoke N82->N85 growth within 60s, lap time displayed. See logs section Phase 8 Step 23 -- Detailed archive.


### Phase 9 - NGE Core Extraction + Driving Improvement + Growth Acceleration [DONE]

**Phase objective:** Extract the NGE grow-stabilize cycle from the racing demo's app layer into `src/neat/nge-juvenile/` as reusable core library behavior. Fix all-cars methodology, driving quality improvement, growth speed (dead knob + batch growth), and pre-existing test defects. The NGE lifecycle should expose a `growStabilizeCycle` mode with sensible defaults and overridable parameters. The app layer must be thinner after this refactor - it should call NGE core, not implement NGE logic.

[DONE] Phase 9 Steps 01-07: All steps green-validated. 341 tests pass, 100% coverage on 6 src/ files, tsc/lint/build pass, browser smoke pass. See logs section "Phase 9 - NGE Core Extraction + Driving Improvement + Growth Acceleration [DONE] - Detailed archive".

- [DONE] Step 01 - Plan Phase 9: boundary map completed, step packets 02-07 authored, gates PASS (plan-sync, step-packet, plan-slice-quality, agent-graph, plan-readiness green-light: true).
- [DONE] Step 02 - Research: boundary map confirmed, new file targets and cycle-break plan documented, tsc clean.
- [DONE] Step 03 - Red tests: 24 red test contracts across 3 files, all fail for right reasons (missing implementation).
- [DONE] Step 04 - Implementation: all 7 slices (04a-04g) completed. Core module created, app layer thinned (no deferred cleanup), all-cars methodology, driving improvement, growth speed, test fixes.
- [DONE] Step 05 - Green validation: 341 tests pass, 100% coverage on 6 src/ files (statements/branches/functions/lines), 6 iterations to green. Browser smoke N79->N85 growth, 0 console errors.
- [DONE] Step 06 - Documentation: JSDoc complete on all new exports, docs PASS, folder quality gates pass (pre-existing gaps documented as risks).
- [DONE] Step 07 - Logging/compression: Phase 9 compressed to logs, plan marked [DONE].
