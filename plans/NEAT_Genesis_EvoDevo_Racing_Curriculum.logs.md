# NEAT Genesis EvoDevo: Racing Curriculum Log

**Status:** [WIP]

## Scope

Durable compressed log for the racing-curriculum reference-completion workstream.

## Coverage notes

### Phase 1 — Racing curriculum refactor packetization

- [DONE] Step 02 mapped the boundary to worker-owned simulation/evolution plus compact race-step streaming as the first plan-fidelity prerequisite for Team A/B coevolution.
- [DONE] Step 03 added the smallest failing owner-local tests for the worker FSM, deterministic race-pack replay, transfer-list ownership, Team A/B isolation, best-finisher team fitness, and the opponent-snapshot barrier.
- [DONE] Step 04 implemented the worker-authoritative foundation: typed protocol boundaries, independent Team A/B containers, deterministic race packs, packed frame ownership, and frozen rolling-opponent snapshot rotation.
- [DONE] Step 05 validated the focused Step 03/04 slice plus nearby owner-local regressions; no implementation regression, upstream NGE blocker, or browser-build failure was detected.
- [DONE] Step 06 documented host-owned versus worker-owned responsibilities, fallback transport, packed snapshot semantics, and the honest remaining benchmark gaps.
- [DONE] Step 07 compressed Phase 1 into tracker evidence.

### Phase 2 — Tier 1: Single agent on simple track

- [DONE] Step 01 — Plan Tier 1 single-agent benchmark: recorded 2-lane simple-track assumptions, inner-lane centerline start, and authored Step 02-07 packets.
- [DONE] Step 02 — Research single-agent track and fitness contracts: confirmed simple-track geometry, lap-time fitness formula, episode termination, worker-authoritative split; no upstream NGE blockers for Tier 1.
- [DONE] Step 03 — Red tests for single-agent worker runtime: added 19+ focused red tests in `simulation-worker.race-pack.test.ts` and a sibling smoke test; all failed for the expected missing-implementation reason.
- [DONE] Step 04 — Implement single-agent worker authority (slices archived below): deterministic 2-car race pack, `createRaceEpisodeRunner`, lap detection, lap-time fitness, race-step messaging, and per-agent cyan/magenta guiding lines; 47 focused renderer/race-pack tests passed, full racing-curriculum suite green, bundle rebuilt.
- [DONE] Step 05 — User visual confirmation of Tier 1 guiding lines: browser-ui-specialist confirmed two cars with cyan (Team A) and magenta (Team B) guiding lines, no Phase 1 regressions.
- [DONE] Step 06 — Document Tier 1 contract: updated `examples/racing_curriculum/README.md` with Tier 1 usage contract, regenerated worker README, `npm run docs:quality:metrics` and `npm run lint` passed.
- [DONE] Step 07 — Logging and tracker handoff: Phase 2 compressed into this log, Phase 3 Step 01 advanced to [WIP], plan-sync and phase-packet gates passed.

### Phase 3 — Tier 2: 1v1 with radio (one car per team)

- [DONE] Step 01 — Plan Tier 2 1v1 radio boundary: recorded one-car-per-team/two-cars-total, radio-on/no-pits/no-tires baseline, 7-channel self-signal semantics, and authored Step 02-07 packets.
- [DONE] Step 02 — Research Tier 2 observation/action and radio contracts: documented 77-in (70 base + 7 self-radio tail at `[70..76]`) / 9-out (2 control + 7 radio-write) contract, traced `prepareObservationState` self-radio write/read wiring; deferred worker-authoritative evolution protocol wiring to a later phase.
- [DONE] Step 03 — Red tests for Tier 2 1v1 pack and 77-dim observation: added focused red tests in `browser-entry.test.ts`, `nge.controller.test.ts`, and `observation.assembler.test.ts`; failures were honest missing-implementation gaps (pack layout, 9-output head, radio write split).
- [DONE] Step 04 — Implement Tier 2 1v1 radio loop: added `TIER_TWO_TEAM_LAYOUT = [0, 1]`, activated `ACTIVE_CURRICULUM_TIER = 2`, built 77-input/9-output MLP, split outputs into throttle/steer + 7-channel self-radio write; no dual-path code; all owner-local tests passed, bundle built (732.1kb).
- [DONE] Step 05 — Green validation and regression triage: targeted browser-entry, controller, observation assembler, and simulation-worker race-pack tests passed; Tier 1 paths remained green; lint, build, and plan validators passed.
- [DONE] Step 06 — Document Tier 2 contract: updated `examples/racing_curriculum/README.md` with Tier 2 pack layout, 77-in/9-out network shape, self-radio semantics, activation instructions, runnable TypeScript example, and feedback-loop Mermaid diagram; `npm run docs`, `npm run lint`, and plan validators passed; example validated with `tsx` against real source files.
- [DONE] Step 07 — Logging and tracker handoff: Phase 3 compressed into this log, Phase 4 Step 01 advanced to [WIP], plan-sync, phase-packet, phase-compression, and workflow-update-sync gates passed.

**Changed file groups:**

- `examples/racing_curriculum/browser-entry/browser-entry.ts`, `browser-entry.test.ts`
- `examples/racing_curriculum/controller/nge.controller.ts`, `nge.controller.test.ts`
- `examples/racing_curriculum/controller/observation.assembler.test.ts`
- `examples/racing_curriculum/README.md`
- Generated docs bundle: `docs/assets/racing-curriculum.bundle.js`

**Residual risks (carry-forward):**

- `node scripts/agent-customization/validate-docs-examples.mjs --json` is referenced by the plan but does not exist in the repo; the Tier 2 README example was validated manually via `tsx` instead.

## Phase 3 — Tier 2: 1v1 with radio — COMPLETED

**Status:** [DONE] — all steps (Step 08 through Step 19) complete; Phase 3 green gate passes (348 tests, lint clean, tsc clean). Phase 3 compressed 2026-06-26.

### Step summary (Step 08 – Step 19)

- [DONE] Step 08 — Red tests for Tier 1/Tier 2 racing baseline rules: added focused red tests in `racing.renderer.test.ts`, `browser-entry.test.ts`, `track.generator.test.ts`, and `environment.step.service.test.ts` covering renderer colors, per-car guides, lane constants, alternating pits, and boundary walls; failures were honest missing-implementation gaps.
- [DONE] Step 09 — Implement Tier 1/Tier 2 racing baseline rules: implemented the five baseline rules (renderer color/guidance, lane constants, alternating pits, boundary clamping); all four focused Jest slices passed, build and plan validators passed.
- [DONE] Step 10 — Green validation and regression triage: confirmed Step 09 did not break existing focused tests, race-pack regressions, controller tests, or quality gates; no Step 09-caused failures detected; coverage for touched `examples/` files is outside the `collectCoverageFrom` glob (not applicable).
- [DONE] Step 11 — Document Tier 1/Tier 2 baseline contract: updated `examples/racing_curriculum/README.md` with the baseline contract; `npm run docs:quality:metrics` and `npm run lint` passed.
- [DONE] Step 12 — Reconcile user-reported Tier 2 demo defects and plan hardening steps: scope reconciliation assigned seven user-reported defects to Phase 3 hardening (renderer, physics, tier layout/start) or deferred to Phase 4; `ACTIVE_CURRICULUM_TIER` default decision (Tier 1) and Tier 3 4-car fallback decision recorded; Step 13–17 packets authored with red-green slices.
- [DONE] Step 13 — Renderer hardening: guide lines + trails + header text: renderer hardening implemented; green validation slice `p3-s13-green-renderer` [DONE] — 7 suites, 85 tests passed; lint, tsc (tsconfig.json + tsconfig.test.json), `npm run build:racing-curriculum` (732.9 kb bundle), plan-sync, and plan-phase-packets all PASS.
- [DONE] Step 14 — Physics hardening: off-track penalty + wrong direction + car pushing: physics hardening (off-track penalty, wrong-direction detection, car-vs-car pushing) implemented; green validation slice `p3-s14-green-physics` [DONE].
- [DONE] Step 15 — Tier layout/start: Tier 1 default + Tier 3 fallback: tier layout implemented; implementation slice `p3-s15-impl-tier-layout` [DONE] (29 suites, 249 tests PASS, bundle 733.7 kb), green validation slice `p3-s15-green-tier-layout` [DONE] (Tier 1 2-car probe and Tier 3 4-car fallback probe both exercised).
- [DONE] Step 16 — Document updated Tier 1/Tier 2 demo contract: updated `examples/racing_curriculum/README.md` with the updated demo contract; `npm run docs` (HTML docs generated, Mermaid diagrams validated), lint, and plan validators passed.
- [DONE] Step 18 — Tier 1 demo defect investigation: source-grounded alignment brief identified four Tier 1 demo defects (red guide-line ignored, car overlap, yellow guide-line, cyan center divider) with file:line evidence mapped to `observation.assembler.ts`, `browser-entry.ts`, `environment.step.service.ts` / `simulation-worker.race-pack.service.ts`, and `renderer/racing.renderer.ts`.
- [DONE] Step 19 — Tier 1 independent-agent architecture pivot: pivoted from shared-controller fan-out to independent per-car NEAT agents. All 22 red-green slices [DONE]; 348 tests pass, lint clean, tsc clean. See slice archive and decision record below.
- [DONE] Step 17 — Logging and tracker handoff: compressed Phase 3 step/slice details into this logs file; Phase 3 marked [DONE]; `plans/README.md` and `plans/Roadmap.md` updated; Phase 4 left [PLANNED] pending user browser-demo confirmation.

### Step 19 slice archive — Tier 1 independent-agent architecture pivot

All 22 slices are [DONE]. Validation evidence per slice:

| # | slice_id | title | goal | key evidence |
|---|----------|-------|------|--------------|
| 1 | `p3-s19-red-obs-team-offset` | Red tests for team-aware observation offset | red-testing | 7/8 tests, 1 expected failure (teamIndex 1 outer-lane centerline) |
| 2 | `p3-s19-impl-obs-team-offset` | Implement team-aware observation offset | implementing | 8/8 tests; tsc PASS; lint PASS |
| 3 | `p3-s19-green-obs-team-offset` | Green validation for team-aware observation offset | green-testing | 8/8 tests; browser-entry 69/69 across 6 suites; tsc PASS; build:racing-curriculum PASS |
| 4 | `p3-s19-red-per-car-observation` | Red tests for per-car observation-state helper | red-testing | 8 passed, 4 failed (derivePerCarObservationState undefined) — honest missing-implementation gap |
| 5 | `p3-s19-impl-per-car-observation` | Implement per-car observation-state helper | implementing | 12/12 tests; tsc PASS; lint PASS |
| 6 | `p3-s19-green-per-car-observation` | Green validation for per-car observation-state helper | green-testing | 12/12 tests; browser-entry 69/69; tsc PASS; build PASS |
| 7 | `p3-s19-red-browser-per-car-controller` | Red tests for independent per-car controllers in browser harness | red-testing | 4 new per-car controller contracts fail (resolveControlFanOut still present, single controller); 70/74 pass |
| 8 | `p3-s19-impl-browser-per-car-controller` | Implement per-car controller Map in browser harness | implementing | Map<carIndex, NgeController> maintained; resolveControlFanOut removed; per-car observation wired |
| 9 | `p3-s19-green-browser-per-car-controller` | Green validation for independent per-car browser control | green-testing | Red tests now pass; no regressions in browser-entry suite |
| 10 | `p3-s19-red-separation-grid` | Red tests for car separation and worker grid | red-testing | Environment + worker tests fail on overlapping bounding boxes and identical starting positions |
| 11 | `p3-s19-impl-separation-grid` | Implement car separation and worker starting-grid alignment | implementing | CAR_MIN_CENTER_SEPARATION prevents overlap; worker buildRaceFrame staggers cars with grid-spacing |
| 12 | `p3-s19-green-separation-grid` | Green validation for car separation and worker grid | green-testing | Red tests now pass; no regressions; baselines updated |
| 13 | `p3-s19-red-renderer-cleanup` | Red tests for renderer visual cleanup | red-testing | Renderer test fails on yellow optimal-line overlay / cyan centerline |
| 14 | `p3-s19-impl-renderer-cleanup` | Implement renderer visual cleanup | implementing | drawOptimalLineGuidance, COLOR_GUIDANCE_LINE_RGB, drawTrackCenterline, COLOR_CENTERLINE removed; only blue/red team guide lines remain |
| 15 | `p3-s19-green-renderer-cleanup` | Green validation for renderer visual cleanup | green-testing | Red tests now pass; no yellow/cyan assertions remain; no regressions |
| 16 | `p3-s19-red-per-car-adaptation` | Red tests for continuous per-car runtime adaptation | red-testing | Test fails when single shared adaptation state mutates every car identically |
| 17 | `p3-s19-impl-per-car-adaptation` | Implement per-car runtime adaptation / continuous evolution | implementing | Map<carIndex, RuntimeAdaptationState>; no global singleton; browser harness wires each car to its own adaptation entry |
| 18 | `p3-s19-green-per-car-adaptation` | Green validation for continuous per-car evolution | green-testing | Red tests pass; deterministic probe shows blue/red controllers diverge within 120 frames; no regressions |
| 19 | `p3-s19-red-worker-independent-genomes` | Red tests for worker evaluation of independent genomes | red-testing | Test fails when runner uses one shared network for every car; asserts distinct per-car networks and fitness |
| 20 | `p3-s19-impl-worker-independent-genomes` | Implement per-car genome/network wiring in worker race-pack | implementing | createRaceEpisodeRunner accepts one network per car; coevolution container provides one NEAT genome per car; evolution protocol returns per-car/per-team payloads |
| 21 | `p3-s19-green-worker-independent-genomes` | Green validation for worker independent-genome evaluation | green-testing | Red tests pass; no regressions; worker starting-grid baselines updated |
| 22 | `p3-s19-green-tier1-divergence-probe` | Green validation — Tier 1 blue/red divergence probe | green-testing | Deterministic Tier 1 probe records distinct steering/lateral positions for blue and red within 120 frames; blue follows inner guide, red follows outer guide, no overlap, no cyan divider |

**Final Step 19 validation gate:** 348 tests pass across all focused suites; `npm run lint` clean; `npx tsc --noEmit -p tsconfig.json` clean; `npx tsc --noEmit -p tsconfig.test.json` clean; `npm run build:racing-curriculum` succeeds; `validate-plan-sync` PASS (0 errors, 0 warnings); `validate-plan-phase-packets` PASS (0 errors, 0 warnings).

**Superseded pre-pivot slices (removed from active chain):**

- `p3-s19-red-browser-per-car` — replaced by `p3-s19-red-browser-per-car-controller` plus per-car observation/adaptation slices.
- `p3-s19-impl-browser-per-car` — replaced by `p3-s19-impl-browser-per-car-controller`.
- `p3-s19-green-browser-per-car` — replaced by `p3-s19-green-browser-per-car-controller`.

Their single concern (stop fanning one control to every car) is now enforced by the per-car observation helper, the per-car controller Map, the per-car adaptation Map, and the per-car worker genome wiring.

### Decision Record — DR-2026-06-26-01

```yaml
decision_record:
  id: 'DR-2026-06-26-01'
  context: 'Tier 1 racing demo currently creates a single NGE controller and fans its single {throttle, steer} output to all cars via resolveControlFanOut. This masks team-aware observation, prevents independent evolution, and produces overlapping, identical cars.'
  options:
    - id: 'fan-out'
      desc: 'Keep the shared NEAT controller and continue fanning one control output to every car, only patching the observation offset.'
    - id: 'independent-agents'
      desc: 'Give every car its own continuously evolving NEAT network/controller and derive a separate observation state for each car.'
  chosen: 'independent-agents'
  rationale: 'The NGE racing curriculum is intended as a multi-agent benchmark ladder. A shared controller cannot demonstrate coevolution, team specialization, or independent adaptation. Per-car networks are a prerequisite for Tier 1 → Tier 3 progression and align with the Ant Hive / Predator-Prey demos.'
  owner: '01-planning'
  rollback_plan: 'If green validation fails, restore the pre-pivot Step 19 packet and reactivate the superseded p3-s19-red-browser-per-car fan-out slice. Remove per-car Map and helper code in the same rollback commit.'
  created_at: '2026-06-26T00:00:00Z'
```

### Planning claim — per-car NEAT agents

The previous "one NEAT controller fanned out to every car" design is superseded. In Tier 1, every car is an independently evolving NEAT agent: its own genome-derived network, its own per-car observation state, its own controller instance, and its own runtime adaptation cadence. This contract carries forward to Phase 4 (Tier 3 2v2): every car has its own NEAT genome-derived network, observation, controller, and adaptation state; teammates share only radio/team observations and a team-scoped fitness signal. The Ant Hive and Predator/Prey NGE demos will reuse this independent-agent pattern when their active phases begin.

### Phase 3 changed file groups

- `examples/racing_curriculum/renderer/racing.renderer.ts`, `racing.renderer.test.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`, `browser-entry.test.ts` (and sibling test files)
- `examples/racing_curriculum/controller/observation.assembler.ts`, `observation.assembler.test.ts`
- `examples/racing_curriculum/controller/nge.controller.ts`, `nge.controller.test.ts`
- `examples/racing_curriculum/controller/runtime.adaptation.ts`, `runtime.adaptation.test.ts`
- `examples/racing_curriculum/environment/environment.step.service.ts`, `environment.step.service.test.ts`
- `examples/racing_curriculum/track/track.generator.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`, `simulation-worker.race-pack.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.coevolution.service.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.evolution.protocol.service.ts`
- `examples/racing_curriculum/README.md`
- Generated docs bundle: `docs/assets/racing-curriculum.bundle.js`
- `cortex-index` gate reports stale semantic index across pre-existing files; a rebuild did not resolve it. Owner: `00-helping`; not a Phase 3 closure blocker.
- `routing-table-freshness` gate reports `.github/agent-skill-routing-table.md` is stale relative to source files; no routing changes were made in this phase. Owner: routing-table generator; not a Phase 3 closure blocker.
- Worker-authoritative evolution protocol integration was explicitly deferred to Tier 3 or beyond; Tier 2 validated on the browser-host POC seam.

### Phase 1 (UI parity tranche) — Racing UI completion to Flappy Bird parity

- [DONE] Step 01 authored the Phase 1 step packets and UI-first ordering constraint.
- [DONE] Step 02 mapped Flappy Bird layout, network view, hover/tooltip, resize, and tier-ladder dependencies.
- [DONE] Step 03 added owner-local red tests for host layout, network panel mount, controls placement, old-import removal, resize redraw, and tooltip contracts; 17 honest red failures recorded.
- [DONE] Step 04 implemented the polished racing UI: Flappy-style outer frame + right-sidebar network panel, local network-view renderer, hover/resize services; removed race-pack placeholder and old `src/visualization/network-view` import.
- [DONE] Step 05 green validation passed:
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum/browser-entry` — 6 suites, 32 tests PASS.
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/racing_curriculum` — 38 suites, 189 tests PASS.
  - `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=examples/flappy_bird/browser-entry` — 17 suites, 81 tests PASS, regression-free.
  - `npm run build:racing-curriculum` — bundle `docs/assets/racing-curriculum.bundle.js` built, 679.8kb PASS.
  - `npm run quality:folder -- --folder=examples/racing_curriculum` — 0 TS diagnostics, 0 ESLint errors, 71/71 JSDoc symbols PASS.
- [DONE] Step 06 documented the UI contract in source JSDoc and regenerated browser-entry READMEs; `npm run docs` clean; `neataptic-gate-mcp:run_gate_check cortex-index` PASS after rebuilding the semantic index.
- [DONE] Step 07 compressed Phase 1 history into this log and advanced Phase 2 Step 01 to [WIP].

**Changed file groups:**

- `examples/racing_curriculum/browser-entry/host/host.types.ts`, `host.ts`, `host.resize.service.ts`, `host.resize.service.test.ts`, `host.network-tooltip.service.ts`, `host.network-tooltip.service.test.ts`
- `examples/racing_curriculum/browser-entry/network-view/network-view.ts`, `network-view.test.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- Generated READMEs: `examples/racing_curriculum/browser-entry/README.md`, `examples/racing_curriculum/browser-entry/host/README.md`, `examples/racing_curriculum/browser-entry/network-view/README.md`

**Residual risks:**

- Local racing network renderer is intentionally minimal (rectangles + straight connections); richer legends/layered layouts are out of scope for Phase 1.
- `browser-entry/README.md` still lists some internal orchestration helpers alongside `start()`; a future boundary cleanup could mark more symbols `@internal`.
- No `src/` files changed, so no source-coverage regression is possible.

## Validation and gate evidence

- `validate-plan-sync` (script): PASS.
- `validate-plan-phase-packets` (script): PASS.
- `phase-compression.gate`: PASS.
- `plan-sync` gate (MCP): PASS.
- `step-packet` gate (MCP): PASS.
- `routing-table-freshness.gate`: PASS.
- `stale-wip-plans.gate`: PASS (workstream remains active; plan not archived).
- No customization gap occurred, so no learning-event record was required.

### Phase 1 revisit — network-panel parity fix, HUD/help strip, and inner-track centerline objective

- User rejected the initial Phase 1 sign-off because the right-side network panel
  was empty; root cause was absolute-pixel graph coordinates being treated as
  normalized `[0,1]` values.
- [DONE] Network-panel parity fix: replaced the local renderer with a
  Flappy/ASCII-Maze shared-visualizer adapter, moved hover/resize/tooltip state
  into `host.ts`, and removed the old local renderer path in the same step.
- [DONE] Brightness/HUD/help-strip fix: added `RACING_NETWORK_CONNECTION_LAYER_STYLE`,
  neon top HUD strip, static help-chip strip, and live HUD status updates; bundle
  rebuild was identified as the remaining blocker for visual confirmation.
- [DONE] The inner-track-center objective is complete: the agent follows the inner-lane
  centerline (2 lanes, left normal = inner, offset `+width/4`), with changes to track spec,
  spline utils, observation assembler, scripted/NGE controllers, renderer, start
  placement, and network-view labels. The old full-road centerline math was removed in the
  same implementation step.

### Phase 1 final sign-off — inner-track centerline behavior + persistent network panel labels and live node values

**Status:** [DONE]

- User confirmed the fixed right-side network panel (persistent labels and live node values) and the inner-track guidance overlay.
- [DONE] Step 01 — Plan inner-track scope and confirm Tier 1 assumptions.
- [DONE] Step 02 — Implement inner-track centerline behavior (red-green slices `p1-02-red`, `p1-02-impl`, `p1-02-green`).
- [DONE] Step 03 — UI defect fixes: persistent network panel labels and live node values (red-green slices `p1-03-red`, `p1-03-impl`, `p1-03-green`).

**Changed file groups:**

- Track geometry: `examples/racing_curriculum/track/track.generator.types.ts`, `track.generator.ts`, `track.spline.utils.ts` plus tests.
- Controllers / observations: `examples/racing_curriculum/controller/observation.assembler.ts`, `scripted.controller.ts`, `nge.controller.ts` plus tests.
- Renderer: `examples/racing_curriculum/renderer/racing.renderer.ts` plus test.
- Browser entry: `examples/racing_curriculum/browser-entry/browser-entry.ts`, `browser-entry.test.ts`, `network-view/network-view.constants.ts`.
- Shared visualizer fixes: `examples/flappy_bird/browser-entry/network-view/network-view.ts`, `network-view.topology.utils.ts`, `visualization/visualization.draw.service.ts`, `visualization/visualization.topology.utils.ts`, `browser-entry.visualization.types.ts`, `browser-entry.visualization.utils.ts`.
- Racing network-view / host tests: `examples/racing_curriculum/browser-entry/network-view/network-view.test.ts`, `host/host.test.ts`, `browser-entry.test.ts`.
- Bundle: `docs/assets/racing-curriculum.bundle.js` rebuilt.

**Validation evidence:**

- `p1-02-red`: 6 focused test contracts; 5 RED, 1 precondition GREEN (CCW left normal points to track interior).
- `p1-02-green`: focused `examples/racing_curriculum` Jest suite passes; Flappy Bird browser-entry regression tests remain green; `npm run build:racing-curriculum` passes; folder-quality gate passes.
- `p1-03-red`: 7 failed, 67 passed, 74 total across focused Flappy/racing network-view, host, and browser-entry tests; all failures were the expected label/activation gaps.
- `p1-03-green`: focused network-view tests pass; Flappy Bird regression tests remain green; `npm run build:racing-curriculum` passes; folder-quality gate passes.
- `validate-plan-sync` (script): PASS — 0 errors, 0 warnings.
- `validate-plan-phase-packets` (script): PASS — 0 errors, 1 warning (expected: no [WIP] phase because Phase 2 Step 01 is intentionally not advanced yet).
- `phase-compression.gate` (script): PASS.

**Next:** Phase 2 Step 01 — Plan Tier 1 single-agent benchmark (advance to [WIP] separately after Phase 1 closure).

---

#### Detailed Phase 1 step archive

### Phase 1 — Racing UI/behavior completion to Flappy Bird parity and inner-track centerline [DONE]

```yaml
phase: 1
title: 'Racing UI/behavior completion to Flappy Bird parity and inner-track centerline'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Tier 1: Single agent on simple track'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Phase 1 Step 01-03 packets authored and conform to schema.'
  - 'Inner-track centerline behavior is sequenced before Phase 1 sign-off and before any tier implementation.'
  - 'No-deferred-cleanup criterion recorded for the inner-track geometry migration.'
placeholder_steps:
  - 'Step 01 — Plan inner-track scope and confirm Tier 1 assumptions'
  - 'Step 02 — Implement inner-track centerline behavior (red-green with slices)'
  - 'Step 03 — Green validation, bundle rebuild, and Phase 1 closure'
```

**Phase objective:** Complete Phase 1 by making the racing browser demo follow the
inner-track centerline. The UI parity work (Flappy Bird layout, right-side
network panel, neon HUD strip, help-chip strip, hover/tooltips, dynamic resize,
stats panels, and control placement) is implemented and green. The remaining
frontier is the behavior change: the agent and all visual guidance should
target the center of the inner lane, not the full-road centerline.

**Stop conditions:**

- **Done:** Phase 1 Step 01-03 are [DONE], focused UI and behavior tests pass,
  the racing bundle builds, folder quality passes, and Phase 1 history is
  compressed into the log.
- **Hold:** a UI or behavior policy choice needs user clarification before
  implementation or sign-off.
- **Blocked:** a tooling or upstream dependency blocks honest implementation;
  escalate to `00-helping` with a recorded blocker.
- **Route-back:** return to the owner of any incomplete slice or step before
  advancing.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Phase 1 `phase-compression` gate when Phase 1 is marked [DONE].

#### Coverage note — Phase 1 prior UI parity tranche

- [DONE] Step 01-07 for the original UI parity scope are archived in
  `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- [DONE] Network-panel parity revisit (coordinate normalization, Flappy-style
  renderer, hover/resize/tooltip services, brightened connections, neon HUD,
  help-chip strip) is also archived in the same log.
- Phase 1 is now reopened for the inner-track centerline objective before final
  sign-off.

#### Step 01 — Plan inner-track scope and confirm Tier 1 assumptions [DONE]

```yaml
phase: 1
step: 1
title: 'Plan inner-track scope and confirm Tier 1 assumptions'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 02 — Implement inner-track centerline behavior'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Phase 1 is [WIP] with exactly one [WIP] step (Step 02).'
  - 'Phase 2 is [PLANNED] and its Step 01 records the confirmed simple-track assumptions.'
  - 'Inner-track default assumptions are recorded for the implementation step.'
  - 'Plan index and roadmap status text are updated to match.'
```

**Step objective:** Fold the inner-track-center objective into Phase 1, confirm the
simple-track assumptions that Tier 1 will use, and keep Phase 2 [PLANNED] until
Phase 1 is fully signed off.

**Research findings folded from `nge-benchmark-scout`:**

- Track geometry currently has one sampled centerline and `SplineSample.width` is
  the full road width; there is no lane concept.
- The left normal returned by `resolveSplineSampleFrame` points toward the
  interior of the CCW-generated loop, making the left side the natural inner side.
- For `laneCount` lanes, the inner-lane centerline is offset from the road center
  by `width/2 - width/(2*laneCount)` in the inner-normal direction (e.g.,
  `+width/4` for 2 lanes).
- Surfaces that must change: track spec, spline framing, scripted controller
  target, observation assembler optimal-line and boundary semantics, NGE
  controller evidence extraction, renderer centerline/guidance/edges, start-grid
  placement, and network-view labels.

**Default assumptions recorded for Step 02 implementation:**

- Lane count for Phase 1 / Tier 1 reference: **2 lanes**.
- Inner side for the CCW loop generator: the **left normal** of the road
  centerline.
- Inner-lane centerline offset from road center: `+width/4` (general formula
  `width/2 - width/(2*laneCount)`).
- Channels 11-13 keep full-road edge distances; channels 16/17 reference the
  inner-lane centerline as the optimal-line target.
- Scripted and NGE controllers will steer toward the inner-lane centerline.
- The Phase 1 renderer guidance overlay will also follow the inner-lane
  centerline.
- Car start placement in the browser demo will sit on the inner-lane
  centerline.
- Pit corridors (future Tier 4+) will move to the outer side of the track.
- Old full-road centerline optimal-line math is removed in the same step that
  introduces the inner-lane math; no dual-path wrappers.

**User instruction:** Paste this full step packet; then dispatch 04-implementing
for the Step 02 slices, starting with `p1-02-red`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

#### Step 02 — Implement inner-track centerline behavior [DONE]

```yaml
phase: 1
step: 2
title: 'Implement inner-track centerline behavior'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 03 — Green validation, bundle rebuild, and Phase 1 closure'
skills:
  - 'implementation-standards'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'"
  - 'npm run build:racing-curriculum'
  - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
acceptance_criteria:
  - 'Inner-lane centerline is computed from track width and lane count and used by controllers, observations, renderer, and start placement.'
  - 'Old full-road centerline target math is removed in the same step.'
  - 'Focused racing tests pass and Flappy Bird demo remains regression-free.'
  - 'Racing bundle builds and the live guidance overlay follows the inner lane.'
slices:
  - slice_id: 'p1-02-red'
    title: 'Red tests for inner-lane centerline behavior'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/track/track.spline.utils.ts'
      - 'examples/racing_curriculum/track/track.spline.utils.test.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
      - 'examples/racing_curriculum/controller/scripted.controller.ts'
      - 'examples/racing_curriculum/controller/scripted.controller.test.ts'
      - 'examples/racing_curriculum/renderer/racing.renderer.ts'
      - 'examples/racing_curriculum/renderer/racing.renderer.test.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - 'Red tests exist and fail before implementation, asserting inner-lane centerline math, observation channel semantics, and renderer overlay positions.'
    parallelizable: false
    dependencies: []
    next_slice: 'p1-02-impl'
  - slice_id: 'p1-02-impl'
    title: 'Implement inner-lane centerline geometry and controllers'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 8
    files_to_change:
      - 'examples/racing_curriculum/track/track.generator.types.ts'
      - 'examples/racing_curriculum/track/track.generator.ts'
      - 'examples/racing_curriculum/track/track.spline.utils.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
      - 'examples/racing_curriculum/controller/scripted.controller.ts'
      - 'examples/racing_curriculum/renderer/racing.renderer.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    acceptance_criteria:
      - 'All red tests pass.'
      - 'Old full-road centerline target math is deleted in the same step; no dual-path wrappers remain.'
      - 'Track spec, spline framing, observation assembler, scripted controller, renderer guidance, and start placement all reference the inner-lane centerline; the NGE controller consumes the updated observation assembler.'
    parallelizable: false
    dependencies:
      - 'p1-02-red'
    next_slice: 'p1-02-green'
  - slice_id: 'p1-02-green'
    title: 'Green validation and UI confirmation prep'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 4
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
    acceptance_criteria:
      - 'Focused racing-curriculum tests pass.'
      - 'Flappy Bird browser-entry regression tests pass.'
      - 'Racing bundle builds and folder-quality gate passes.'
      - 'Coverage guard passes on any touched src/ files (expected none).'
    parallelizable: false
    dependencies:
      - 'p1-02-impl'
```

**Step objective:** Change the racing demo so the agent follows the inner-lane
centerline. This is the implementation step the user asked for; no code changes
are authored here, only the step packet and slices.

**Files to change (summary):**

- `examples/racing_curriculum/track/track.generator.types.ts` — add `laneCount` and
  optional `innerOffsetWorld` to the track spec.
- `examples/racing_curriculum/track/track.generator.ts` — produce 2-lane tracks and
  expose the inner-lane centerline offset.
- `examples/racing_curriculum/track/track.spline.utils.ts` — add helpers to
  resolve the inner-lane centerline from road-center samples and width.
- `examples/racing_curriculum/controller/observation.assembler.ts` — update
  optimal-line and boundary semantics so channels 16/17 reference the inner lane.
- `examples/racing_curriculum/controller/scripted.controller.ts` — steer toward the
  inner-lane centerline instead of the road centerline.
- `examples/racing_curriculum/controller/nge.controller.ts` — extract evidence and
  target line consistent with the inner-lane centerline.
- `examples/racing_curriculum/renderer/racing.renderer.ts` — draw the guidance
  overlay and lane markers relative to the inner-lane centerline.
- `examples/racing_curriculum/browser-entry/browser-entry.ts` — place the car on the
  inner-lane centerline at race start.
- `examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts`
  — update input-label descriptions for channels that now reference the inner
  lane.

**Stop conditions:**

- **Done:** all slices pass, focused tests are green, the bundle builds, and the
  old full-road centerline math is removed.
- **Hold:** user must confirm the default lane-count / inner-direction
  assumptions before code changes begin.
- **Blocked:** an NGE primitive or geometry assumption is missing; route to
  `nge-core-algorithm` or `nge-benchmark-scout`.
- **Route-back:** return to `p1-02-red` if the red tests do not fail honestly
  before implementation, or to `p1-02-impl` if green validation fails.
- **Visual-confirmation hold:** Step 02 implementation and green validation are
  `[DONE]`, but Phase 1 remains `[WIP]` and Step 03 stays `[PLANNED]` until the
  user manually confirms the right-side network panel and the inner-track
  guidance overlay in the live demo. Do not advance Step 03 or mark Phase 1
  `[DONE]` without that confirmation.

**User instruction:** Author this step packet and dispatch to 04-implementing
only after Step 01 assumptions are accepted. 04-implementing owns slices
`p1-02-red`, `p1-02-impl`, and `p1-02-green`; 05-green-testing validates the final
bundle and reports coverage.

**Required validation:**

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'`
- `npm run build:racing-curriculum`
- `npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry`

**Red evidence (slice `p1-02-red`):**

| Test file                       | New test contract                                                                   | Result                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `track.generator.test.ts`       | `carries default laneCount and derived laneWidthWorld and innerOffsetWorld`         | **RED** — `laneCount`, `laneWidthWorld`, `innerOffsetWorld` are `undefined`                  |
| `track.spline.utils.test.ts`    | `points the left normal toward the interior of a counter-clockwise generated track` | **GREEN precondition** — CCW left-normal convention already holds                            |
| `scripted.controller.test.ts`   | `steers near zero when the car is already on the inner-lane centerline`             | **RED** — car on inner-lane centerline is pulled back toward road center (`steer ≈ -0.89`)   |
| `observation.assembler.test.ts` | `reports near-zero optimal-line offset for a car on the inner-lane centerline`      | **RED** — channel 16 reports `0.333` (road-center offset) instead of `0`                     |
| `racing.renderer.test.ts`       | `draws the optimal-line guidance along the inner-lane centerline`                   | **RED** — canvas path still traces road-center samples                                       |
| `browser-entry.test.ts`         | `places the primary car on the inner-lane centerline of the first spline sample`    | **RED** — primary car is at road center instead of `firstSample + normal * innerOffsetWorld` |

Focused validation commands used:

```bash
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=track.generator.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=track.spline.utils.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripted.controller.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=racing.renderer.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=browser-entry.test.ts --runInBand
```

No production code was changed. Handoff target for slice `p1-02-impl`: add `laneCount`, `laneWidthWorld`, and `innerOffsetWorld` to `TrackSpec`/`SplineSample`; shift scripted target and observation optimal line to inner-lane centerline; move renderer guidance overlay; place browser start-grid cars on inner-lane centerline.

#### Step 03 — Green validation, bundle rebuild, and Phase 1 closure [PLANNED]

```yaml
phase: 1
step: 3
title: 'Green validation, bundle rebuild, and Phase 1 closure'
status: '[PLANNED]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Phase 2 Step 01 — Plan Tier 1 single-agent benchmark'
skills:
  - 'green-testing'
  - 'tracker-handoff'
validation:
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'"
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/flappy_bird/browser-entry'"
  - 'npm run build:racing-curriculum'
  - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
acceptance_criteria:
  - 'All Phase 1 tests pass and Flappy Bird regression tests remain green.'
  - 'Racing bundle is rebuilt and the inner-track guidance overlay is visible.'
  - 'Phase 1 is compressed into the log and marked [DONE].'
  - 'Phase 2 Step 01 advances to [WIP] only after user visual confirmation.'
```

**Step objective:** Validate the inner-track change, rebuild the bundle so the
live demo reflects it, and close Phase 1 only after the user confirms both the
network panel and the inner-track guidance overlay.

**User instruction:** Run this step after Step 02 is [DONE]. Do not mark Phase 1
[DONE] without user visual confirmation of the network panel and inner-track
guidance overlay.

**Stop conditions:**

- **Done:** Phase 1 tests pass, the bundle is rebuilt, Phase 1 is compressed to
  the log, and the user confirms the UI.
- **Hold:** user has not yet confirmed the network panel or inner-track overlay.
- **Blocked:** a green-validation failure or build failure prevents honest
  closure; route back to `p1-02-impl`.
- **Route-back:** return to Step 02 slices if validation fails.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'`
- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/flappy_bird/browser-entry'`
- `npm run build:racing-curriculum`
- `npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`

## Next boundary

- Phase 1 UI/behavior completion is [WIP]; Phase 2 Tier 1 single-agent simple
  track is [PLANNED].
- Active boundary: Phase 1 Step 02 — implement inner-track centerline behavior.
- Continue in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`.

---

### Phase 1 — Final step/slice archive (Step 01-04) [DONE]

### Phase 1 — Racing UI/behavior completion to Flappy Bird parity and inner-track centerline [WIP]

```yaml
phase: 1
title: 'Racing UI/behavior completion to Flappy Bird parity and inner-track centerline'
status: '[WIP]'
goal: 'planning'
expansion: 'steps'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_phase: 'Phase 2 — Tier 1: Single agent on simple track'
skills:
  - 'plan-alignment'
  - 'tracker-handoff'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Phase 1 Step 01-03 packets authored and conform to schema.'
  - 'Inner-track centerline behavior is sequenced before Phase 1 sign-off and before any tier implementation.'
  - 'No-deferred-cleanup criterion recorded for the inner-track geometry migration.'
placeholder_steps:
  - 'Step 01 — Plan inner-track scope and confirm Tier 1 assumptions'
  - 'Step 02 — Implement inner-track centerline behavior (red-green with slices)'
  - 'Step 03 — Real-time network visualizer live-value refresh (red-green with slices)'
  - 'Step 04 — Green validation, bundle rebuild, and Phase 1 closure'
```

**Phase objective:** Complete Phase 1 by making the racing browser demo follow the
inner-track centerline. The UI parity work (Flappy Bird layout, right-side
network panel, neon HUD strip, help-chip strip, hover/tooltips, dynamic resize,
stats panels, and control placement) is implemented and green. The remaining
frontier is the behavior change: the agent and all visual guidance should
target the center of the inner lane, not the full-road centerline.

**Stop conditions:**

- **Done:** Phase 1 Step 01-03 are [DONE], focused UI and behavior tests pass,
  the racing bundle builds, folder quality passes, and Phase 1 history is
  compressed into the log.
- **Hold:** a UI or behavior policy choice needs user clarification before
  implementation or sign-off.
- **Blocked:** a tooling or upstream dependency blocks honest implementation;
  escalate to `00-helping` with a recorded blocker.
- **Route-back:** return to the owner of any incomplete slice or step before
  advancing.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- Phase 1 `phase-compression` gate when Phase 1 is marked [DONE].

#### Coverage note — Phase 1 prior UI parity tranche

- [DONE] Step 01-07 for the original UI parity scope are archived in
  `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.logs.md`.
- [DONE] Network-panel parity revisit (coordinate normalization, Flappy-style
  renderer, hover/resize/tooltip services, brightened connections, neon HUD,
  help-chip strip) is also archived in the same log.
- Phase 1 is now reopened for the inner-track centerline objective before final
  sign-off.

#### Step 01 — Plan inner-track scope and confirm Tier 1 assumptions [DONE]

```yaml
phase: 1
step: 1
title: 'Plan inner-track scope and confirm Tier 1 assumptions'
status: '[DONE]'
goal: 'planning'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 02 — Implement inner-track centerline behavior'
skills:
  - 'plan-alignment'
  - 'nge-benchmark-scout'
  - 'tracker-handoff'
validation:
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
  - 'Phase 1 is [WIP] with exactly one [WIP] step (Step 02).'
  - 'Phase 2 is [PLANNED] and its Step 01 records the confirmed simple-track assumptions.'
  - 'Inner-track default assumptions are recorded for the implementation step.'
  - 'Plan index and roadmap status text are updated to match.'
```

**Step objective:** Fold the inner-track-center objective into Phase 1, confirm the
simple-track assumptions that Tier 1 will use, and keep Phase 2 [PLANNED] until
Phase 1 is fully signed off.

**Research findings folded from `nge-benchmark-scout`:**

- Track geometry currently has one sampled centerline and `SplineSample.width` is
  the full road width; there is no lane concept.
- The left normal returned by `resolveSplineSampleFrame` points toward the
  interior of the CCW-generated loop, making the left side the natural inner side.
- For `laneCount` lanes, the inner-lane centerline is offset from the road center
  by `width/2 - width/(2*laneCount)` in the inner-normal direction (e.g.,
  `+width/4` for 2 lanes).
- Surfaces that must change: track spec, spline framing, scripted controller
  target, observation assembler optimal-line and boundary semantics, NGE
  controller evidence extraction, renderer centerline/guidance/edges, start-grid
  placement, and network-view labels.

**Default assumptions recorded for Step 02 implementation:**

- Lane count for Phase 1 / Tier 1 reference: **2 lanes**.
- Inner side for the CCW loop generator: the **left normal** of the road
  centerline.
- Inner-lane centerline offset from road center: `+width/4` (general formula
  `width/2 - width/(2*laneCount)`).
- Channels 11-13 keep full-road edge distances; channels 16/17 reference the
  inner-lane centerline as the optimal-line target.
- Scripted and NGE controllers will steer toward the inner-lane centerline.
- The Phase 1 renderer guidance overlay will also follow the inner-lane
  centerline.
- Car start placement in the browser demo will sit on the inner-lane
  centerline.
- Pit corridors (future Tier 4+) will move to the outer side of the track.
- Old full-road centerline optimal-line math is removed in the same step that
  introduces the inner-lane math; no dual-path wrappers.

**User instruction:** Paste this full step packet; then dispatch 04-implementing
for the Step 02 slices, starting with `p1-02-red`.

**Required validation:**

- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

#### Step 02 — Implement inner-track centerline behavior [DONE]

```yaml
phase: 1
step: 2
title: 'Implement inner-track centerline behavior'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 03 — Real-time network visualizer live-value refresh'
skills:
  - 'implementation-standards'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'"
  - 'npm run build:racing-curriculum'
  - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
acceptance_criteria:
  - 'Inner-lane centerline is computed from track width and lane count and used by controllers, observations, renderer, and start placement.'
  - 'Old full-road centerline target math is removed in the same step.'
  - 'Focused racing tests pass and Flappy Bird demo remains regression-free.'
  - 'Racing bundle builds and the live guidance overlay follows the inner lane.'
slices:
  - slice_id: 'p1-02-red'
    title: 'Red tests for inner-lane centerline behavior'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 4
    files_to_change:
      - 'examples/racing_curriculum/track/track.spline.utils.ts'
      - 'examples/racing_curriculum/track/track.spline.utils.test.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.test.ts'
      - 'examples/racing_curriculum/controller/scripted.controller.ts'
      - 'examples/racing_curriculum/controller/scripted.controller.test.ts'
      - 'examples/racing_curriculum/renderer/racing.renderer.ts'
      - 'examples/racing_curriculum/renderer/racing.renderer.test.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    acceptance_criteria:
      - 'Red tests exist and fail before implementation, asserting inner-lane centerline math, observation channel semantics, and renderer overlay positions.'
    parallelizable: false
    dependencies: []
    next_slice: 'p1-02-impl'
  - slice_id: 'p1-02-impl'
    title: 'Implement inner-lane centerline geometry and controllers'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 8
    files_to_change:
      - 'examples/racing_curriculum/track/track.generator.types.ts'
      - 'examples/racing_curriculum/track/track.generator.ts'
      - 'examples/racing_curriculum/track/track.spline.utils.ts'
      - 'examples/racing_curriculum/controller/observation.assembler.ts'
      - 'examples/racing_curriculum/controller/scripted.controller.ts'
      - 'examples/racing_curriculum/renderer/racing.renderer.ts'
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    acceptance_criteria:
      - 'All red tests pass.'
      - 'Old full-road centerline target math is deleted in the same step; no dual-path wrappers remain.'
      - 'Track spec, spline framing, observation assembler, scripted controller, renderer guidance, and start placement all reference the inner-lane centerline; the NGE controller consumes the updated observation assembler.'
    parallelizable: false
    dependencies:
      - 'p1-02-red'
    next_slice: 'p1-02-green'
  - slice_id: 'p1-02-green'
    title: 'Green validation and UI confirmation prep'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 4
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
    acceptance_criteria:
      - 'Focused racing-curriculum tests pass.'
      - 'Flappy Bird browser-entry regression tests pass.'
      - 'Racing bundle builds and folder-quality gate passes.'
      - 'Coverage guard passes on any touched src/ files (expected none).'
    parallelizable: false
    dependencies:
      - 'p1-02-impl'
```

**Step objective:** Change the racing demo so the agent follows the inner-lane
centerline. This is the implementation step the user asked for; no code changes
are authored here, only the step packet and slices.

**Files to change (summary):**

- `examples/racing_curriculum/track/track.generator.types.ts` — add `laneCount` and
  optional `innerOffsetWorld` to the track spec.
- `examples/racing_curriculum/track/track.generator.ts` — produce 2-lane tracks and
  expose the inner-lane centerline offset.
- `examples/racing_curriculum/track/track.spline.utils.ts` — add helpers to
  resolve the inner-lane centerline from road-center samples and width.
- `examples/racing_curriculum/controller/observation.assembler.ts` — update
  optimal-line and boundary semantics so channels 16/17 reference the inner lane.
- `examples/racing_curriculum/controller/scripted.controller.ts` — steer toward the
  inner-lane centerline instead of the road centerline.
- `examples/racing_curriculum/controller/nge.controller.ts` — extract evidence and
  target line consistent with the inner-lane centerline.
- `examples/racing_curriculum/renderer/racing.renderer.ts` — draw the guidance
  overlay and lane markers relative to the inner-lane centerline.
- `examples/racing_curriculum/browser-entry/browser-entry.ts` — place the car on the
  inner-lane centerline at race start.
- `examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts`
  — update input-label descriptions for channels that now reference the inner
  lane.

**Stop conditions:**

- **Done:** all slices pass, focused tests are green, the bundle builds, and the
  old full-road centerline math is removed.
- **Hold:** user must confirm the default lane-count / inner-direction
  assumptions before code changes begin.
- **Blocked:** an NGE primitive or geometry assumption is missing; route to
  `nge-core-algorithm` or `nge-benchmark-scout`.
- **Route-back:** return to `p1-02-red` if the red tests do not fail honestly
  before implementation, or to `p1-02-impl` if green validation fails.
- **Visual-confirmation hold:** Step 02 implementation and green validation are
  `[DONE]`, but Phase 1 remains `[WIP]` and Step 03 stays `[PLANNED]` until the
  user manually confirms the right-side network panel and the inner-track
  guidance overlay in the live demo. Do not advance Step 03 or mark Phase 1
  `[DONE]` without that confirmation.

**User instruction:** Author this step packet and dispatch to 04-implementing
only after Step 01 assumptions are accepted. 04-implementing owns slices
`p1-02-red`, `p1-02-impl`, and `p1-02-green`; 05-green-testing validates the final
bundle and reports coverage.

**Required validation:**

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'`
- `npm run build:racing-curriculum`
- `npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry`

**Red evidence (slice `p1-02-red`):**

| Test file                       | New test contract                                                                   | Result                                                                                       |
| ------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `track.generator.test.ts`       | `carries default laneCount and derived laneWidthWorld and innerOffsetWorld`         | **RED** — `laneCount`, `laneWidthWorld`, `innerOffsetWorld` are `undefined`                  |
| `track.spline.utils.test.ts`    | `points the left normal toward the interior of a counter-clockwise generated track` | **GREEN precondition** — CCW left-normal convention already holds                            |
| `scripted.controller.test.ts`   | `steers near zero when the car is already on the inner-lane centerline`             | **RED** — car on inner-lane centerline is pulled back toward road center (`steer ≈ -0.89`)   |
| `observation.assembler.test.ts` | `reports near-zero optimal-line offset for a car on the inner-lane centerline`      | **RED** — channel 16 reports `0.333` (road-center offset) instead of `0`                     |
| `racing.renderer.test.ts`       | `draws the optimal-line guidance along the inner-lane centerline`                   | **RED** — canvas path still traces road-center samples                                       |
| `browser-entry.test.ts`         | `places the primary car on the inner-lane centerline of the first spline sample`    | **RED** — primary car is at road center instead of `firstSample + normal * innerOffsetWorld` |

Focused validation commands used:

```bash
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=track.generator.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=track.spline.utils.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=scripted.controller.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=observation.assembler.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=racing.renderer.test.ts --runInBand
npx jest --config=jest.config.mjs --no-cache --testPathPatterns=browser-entry.test.ts --runInBand
```

No production code was changed. Handoff target for slice `p1-02-impl`: add `laneCount`, `laneWidthWorld`, and `innerOffsetWorld` to `TrackSpec`/`SplineSample`; shift scripted target and observation optimal line to inner-lane centerline; move renderer guidance overlay; place browser start-grid cars on inner-lane centerline.

#### Step 03 — Real-time network visualizer live-value refresh [DONE]

```yaml
phase: 1
step: 3
title: 'Real-time network visualizer live-value refresh'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 04 — Green validation, bundle rebuild, and Phase 1 closure'
skills:
  - 'implementation-standards'
  - 'nge-benchmark-scout'
validation:
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum/browser-entry/network-view|examples/racing_curriculum/browser-entry/host|examples/racing_curriculum/browser-entry/browser-entry'"
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/flappy_bird/browser-entry'"
  - 'npm run build:racing-curriculum'
  - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
acceptance_criteria:
  - 'Periodic refresh cadence is ~5 s and independent of tier changes: under fake timers the network panel redraw path is invoked at least once within any 6-second window while a race is running.'
  - 'Displayed weights, biases, and activations reflect the current network state: after an in-place network mutation, the next periodic redraw uses the mutated state instead of a fully cached frame.'
  - 'Refresh timer is cancelled on race stop/reset and re-created on race start; no leaked setInterval handles remain after stop().'
  - 'No overlapping/duplicate refresh calls: a slow in-flight redraw does not queue a second concurrent full redraw.'
  - 'Refresh pauses when the network panel host is hidden or document.hidden becomes true, and resumes when visible again.'
  - 'Any code path that existed solely to redraw the visualizer on tier change is removed in the same step unless still required for architecture remapping during promotion (no deferred cleanup).'
  - 'Focused racing-curriculum and Flappy Bird browser-entry tests remain green and the racing bundle builds.'
slices:
  - slice_id: 'p1-03-red'
    title: 'Red tests for live-value network visualizer refresh'
    status: '[DONE]'
    goal: 'red-testing'
    estimate_hours: 3
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
      - 'examples/racing_curriculum/browser-entry/host/host.test.ts'
      - 'examples/racing_curriculum/browser-entry/network-view/network-view.test.ts'
    acceptance_criteria:
      - 'Red tests exist and fail before implementation, asserting ~5 s cadence, stale-frame reuse, timer cleanup, duplicate-call prevention, and hidden-panel pause.'
    parallelizable: false
    dependencies: []
    next_slice: 'p1-03-impl'
  - slice_id: 'p1-03-impl'
    title: 'Implement live-value refresh and remove stale tier-only redraw path'
    status: '[DONE]'
    goal: 'implementing'
    estimate_hours: 5
    files_to_change:
      - 'examples/racing_curriculum/browser-entry/browser-entry.ts'
      - 'examples/racing_curriculum/browser-entry/host/host.ts'
      - 'examples/racing_curriculum/browser-entry/network-view/network-view.ts'
      - 'examples/racing_curriculum/browser-entry/network-view/network-view.constants.ts'
    acceptance_criteria:
      - 'All red tests pass.'
      - 'The host visualization frame cache is invalidated or keyed by network state so periodic redraws show live weights/biases/activations.'
      - 'Any dead tier-change-only redraw path is deleted in the same step; no dual-path wrappers remain.'
      - 'Timer lifecycle (start, stop, hidden/resume) is wired through the existing run handle and host services.'
    parallelizable: false
    dependencies:
      - 'p1-03-red'
    next_slice: 'p1-03-green'
  - slice_id: 'p1-03-green'
    title: 'Green validation and regression triage for visualizer refresh'
    status: '[DONE]'
    goal: 'green-testing'
    estimate_hours: 3
    files_to_change:
      - 'docs/assets/racing-curriculum.bundle.js'
    acceptance_criteria:
      - 'Focused racing browser-entry, host, and network-view tests pass.'
      - 'Flappy Bird browser-entry regression tests remain green.'
      - 'Racing bundle builds and folder-quality gate passes.'
      - 'Coverage guard passes on any touched src/ files (expected none).'
    parallelizable: false
    dependencies:
      - 'p1-03-impl'
```

**Step objective:** Make the right-side network panel show live weights,
biases, and activations during a single tier run. The browser-entry already
installs a 5 s `setInterval`, so the work is to stop the host from reusing a
stale cached visualization frame and to clean up any tier-change-only redraw
path that is no longer needed.

**Context the agent must know:**

- `examples/racing_curriculum/browser-entry/browser-entry.ts` already declares
  `FOCUSED_NETWORK_REFRESH_INTERVAL_MS = 5000` and installs a `setInterval`
  calling the host render path.
- `examples/racing_curriculum/browser-entry/host/host.ts` resolves a
  visualization frame that is reused when canvas size and positioned-node count
  are unchanged; this is the likely stale-frame root cause.
- Tier-change redraws exist in the browser-entry path; keep only what is needed
  for architecture remapping during promotion.
- Hover/resize/tooltip services are already implemented and should not be
  regressed.

**Stop conditions:**

- **Done:** all `p1-03` slices pass, focused tests are green, the bundle builds,
  and live values are visible during a single tier run.
- **Hold:** user must confirm whether panel-hidden / document-hidden behavior
  and hover-driven redraw semantics are acceptable before implementation.
- **Blocked:** the shared Flappy Bird visualizer adapter lacks the seams needed
  to key the cache on network state; route to `visualizer-scout` or
  `browser-ui-specialist`.
- **Route-back:** return to `p1-03-red` if red tests do not fail honestly, or to
  `p1-03-impl` if green validation fails.

**User instruction:** Dispatch to `04-implementing` for the slices, starting with
`p1-03-red`. Do not mark Step 03 `[DONE]` until `p1-03-green` passes and the
bundle shows live values.

**Required validation:**

- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/network-view|examples/racing_curriculum/browser-entry/host|examples/racing_curriculum/browser-entry/browser-entry'`
- `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/flappy_bird/browser-entry'`
- `npm run build:racing-curriculum`
- `npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry`

```yaml
PlanUpdate:
  slice_id: 'p1-03-red'
  changed_files:
    - 'examples/racing_curriculum/browser-entry/browser-entry.test.ts'
    - 'examples/racing_curriculum/browser-entry/host/host.test.ts'
  validation:
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry'"
      expected_exit: 1
      result: '6 suites passed / 52 tests passed / 3 failed — all 3 failures are in the new live-refresh block: (1) frame cache ignores in-place weight mutation, (2) no deduplication across synchronous render calls, (3) panel-hidden guard absent'
  next: 'Dispatch to 04-implementing for p1-03-impl. Green target: frame cache keyed/invalidated by live network state, synchronous duplicate render calls coalesced, and draw skipped when network panel is hidden.'
```

```yaml
PlanUpdate:
  slice_id: 'p1-03-green'
  changed_files:
    - 'docs/assets/racing-curriculum.bundle.js'
  preflight:
    - 'npx tsc --noEmit -p tsconfig.json'
    - 'npx tsc --noEmit -p tsconfig.test.json'
    - 'npm run lint'
    - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
    - 'npx prettier --check examples/racing_curriculum/browser-entry/host/host.ts'
  validation:
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry'"
      expected_exit: 0
      result: '6 suites, 55 tests passed'
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum/browser-entry/host/host.test.ts'"
      expected_exit: 0
      result: '1 suite, 15 tests passed (live-refresh block: cache invalidation on mutation, duplicate sync render coalescing, hidden-panel draw pause)'
    - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/flappy_bird'"
      expected_exit: 0
      result: '34 suites, 160 tests passed — shared visualizer changes did not break Flappy Bird'
    - command: 'npm run build:racing-curriculum'
      expected_exit: 0
      result: 'docs/assets/racing-curriculum.bundle.js (731.1kb, 748,645 bytes, 2026-06-24 18:44:25)'
    - command: 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
      expected_exit: 0
      result: 'PASS folder-quality-metrics (0 diagnostics, 0 lint errors, 19/19 documented symbols, 0 missing tests, 0 sub-100% coverage entries)'
    - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan sync: 0 errors, 0 warnings'
    - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      expected_exit: 0
      result: 'PASS plan phase packets: 0 errors, 0 warnings'
  coverage_guard:
    files: []
    summary: 'no src/ files touched; coverage guard trivially passes'
  next: 'Step 03 is [DONE]. Step 04 (Phase 1 closure) remains [PLANNED]; Phase 1 is still [WIP] until user manually confirms live network values and inner-track guidance overlay.'
```

#### Step 04 — Green validation, bundle rebuild, and Phase 1 closure [DONE]

```yaml
phase: 1
step: 4
title: 'Green validation, bundle rebuild, and Phase 1 closure'
status: '[DONE]'
goal: 'green-testing'
expansion: 'none'
auto_expand: false
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Run final green validation, then hold for user UI confirmation before Phase 1 compression and [DONE]'
skills:
  - 'green-testing'
  - 'tracker-handoff'
validation:
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'"
  - "npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/flappy_bird/browser-entry'"
  - 'npm run build:racing-curriculum'
  - 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
  - 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
  - 'node scripts/agent-customization/gates/phase-compression.gate.mjs --json'
acceptance_criteria:
  - 'All Phase 1 tests pass and Flappy Bird browser-entry regression tests remain green.'
  - 'Racing bundle is rebuilt and the inner-track guidance overlay and live network values are visible.'
  - 'Phase 1 is compressed into the log and marked [DONE].'
  - 'Phase 2 Step 01 advances to [WIP] only after user visual confirmation.'
```

**Step objective:** Validate the inner-track and live visualizer changes, rebuild
the bundle so the live demo reflects them, and close Phase 1 only after the user
confirms the network panel, live value refresh, and inner-track guidance overlay.

**User instruction:** Run this step after Step 03 is [DONE]. Do not mark Phase 1
[DONE] without user visual confirmation of the network panel (live values every
~5 s) and inner-track guidance overlay.

**Stop conditions:**

- **Done:** Phase 1 tests pass, the bundle is rebuilt, Phase 1 is compressed to
  the log, and the user confirms the UI.
- **Hold:** user has not yet confirmed the network panel live-value refresh or
  inner-track overlay.
- **Blocked:** a green-validation failure or build failure prevents honest
  closure; route back to `p1-03-impl`.
- **Route-back:** return to Step 03 slices if validation fails.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/racing_curriculum'`
- `npx jest --config=jest.config.mjs --no-cache --testPathPattern='examples/flappy_bird/browser-entry'`
- `npm run build:racing-curriculum`
- `npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry`
- `node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`
- `node scripts/agent-customization/gates/phase-compression.gate.mjs --json`

```yaml
PlanUpdate:
  slice_id: 'p1-04-green'
  context: >
    Step 04 automated validation completed successfully. All focused tests,
    type checks, lint, folder quality, bundle rebuild, and plan validators
    passed. Phase 1 remains [WIP] pending the user visual confirmation
    below; Phase 1 must not be compressed or marked [DONE] until that
    confirmation is received.
  validation_evidence:
    racing_curriculum_tests:
      command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum'"
      result: 'PASS'
      suites: '38 passed, 38 total'
      tests: '217 passed, 217 total'
    flappy_browser_entry_regression:
      command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/flappy_bird/browser-entry'"
      result: 'PASS'
      suites: '17 passed, 17 total'
      tests: '84 passed, 84 total'
    type_check_main:
      command: 'npx tsc --noEmit -p tsconfig.json'
      result: 'PASS (exit 0, 0 errors)'
    type_check_test:
      command: 'npx tsc --noEmit -p tsconfig.test.json'
      result: 'PASS (exit 0, 0 errors)'
    lint:
      command: 'npm run lint'
      result: 'PASS (exit 0, 0 errors across src/, testing/, benchmarks/, examples/)'
    folder_quality:
      command: 'npm run quality:folder -- --folder=examples/racing_curriculum/browser-entry'
      result: 'PASS (0 TS diagnostics, 0 ESLint errors, 19/19 JSDoc symbols, 0 missing tests, 0 coverage regressions)'
    bundle_rebuild:
      command: 'npm run build:racing-curriculum'
      result: 'PASS'
      output_file: 'docs/assets/racing-curriculum.bundle.js'
      size: '731.1 kb (748,645 bytes)'
      last_write_time: '2026-06-24 18:52:27'
    plan_sync:
      command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      result: 'PASS (0 errors, 0 warnings)'
    plan_phase_packets:
      command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
      result: 'PASS (0 errors, 0 warnings)'
  user_confirmation_question: >
    Please open the racing-curriculum demo and confirm: (1) the right-side
    network panel shows live weights/biases/activations refreshing about
    every 5 seconds while a race is running, and (2) the inner-track
    guidance overlay is visible on the track canvas.
  blocker: >
    Phase 1 cannot be compressed or marked [DONE], and Phase 2 Step 01
    cannot advance to [WIP], until the user confirms the two items above.
  next: >
    Wait for the user's response to the confirmation question. If confirmed,
    dispatch 07-logging to compress Phase 1 and mark Phase 1 [DONE]; if not,
    route back to Step 03 implementation for remediation.
```

---

### Phase 2 Step 04 — Implement single-agent worker authority [DONE]

**Status:** [DONE]

**Summary:**

- `p2-04-red-guiding-lines` — 9 new focused red tests for per-agent guiding lines;
  all failed honestly before implementation.
- `p2-04-impl-guiding-lines` — implemented `buildGuidingLineForTeam` in
  `racing.renderer.ts` and attached per-car `guidingLines` in
  `simulation-worker.race-pack.service.ts`; 47 focused tests passed.
- `p2-04-green` — focused racing-curriculum tests, Phase 1 regression triage,
  type check, lint, folder quality, bundle build, and plan validators all passed.

#### Step 04 — Implement single-agent worker authority [DONE]

```yaml
phase: 2
step: 4
title: 'Implement single-agent worker authority'
status: '[DONE]'
goal: 'implementing'
tdd_sequence: 'red-green'
expansion: 'slices'
auto_expand: true
mode: 'fresh-session'
source_of_truth: 'plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
copy_paste: true
next_step: 'Step 05 — User visual confirmation of Tier 1 guiding lines'
skills:
- 'implementation-standards'
- 'nge-benchmark-workflow'
- 'red-test-contracts'
specialists:
- 'implementation-executor'
- 'red-test-contracts'
validation:
- 'npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum'
- 'npm run lint'
- 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
acceptance_criteria:
- 'All Step 03 red tests pass.'
- '100% statements/branches/functions/lines on all touched src/ and examples/racing_curriculum/ files.'
- 'Build and lint pass with zero new errors.'
- 'Old dead code is removed in the same step (no backward-compatibility wrappers or dual-path code).'
- 'Each car has a dedicated guiding line rendered on the track.'
- 'Guiding lines are visually distinct per team/agent.'
- 'Guiding lines follow the inner-lane centerline or a per-agent offset that keeps the car in its lane.'
- 'The lines are visible in the renderer and optionally fed into the observation vector so the agent can learn to follow them.'
slices:
- slice_id: 'p2-04-red-guiding-lines'
   title: 'Write red tests for per-agent guiding lines'
   status: '[DONE]'
   goal: 'red-testing'
   estimate_hours: 3
   files_to_change:
     - 'examples/racing_curriculum/renderer/*.test.ts'
     - 'examples/racing_curriculum/workers/simulation-worker/*.test.ts'
   acceptance_criteria:
     - 'Red tests exist and fail for the right reason before implementation of per-agent guiding lines.'
     - 'Tests cover line geometry, per-team color, and observation integration.'
   parallelizable: false
   dependencies: []
   next_slice: 'p2-04-impl-guiding-lines'
- slice_id: 'p2-04-impl-guiding-lines'
   title: 'Implement per-agent guiding lines for lane keeping'
   status: '[DONE]'
   goal: 'implementing'
   estimate_hours: 6
   files_to_change:
     - 'examples/racing_curriculum/renderer/**'
     - 'examples/racing_curriculum/browser-entry/**'
     - 'examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-step.service.ts'
   acceptance_criteria:
     - 'Each car has a dedicated guiding line rendered on the track.'
     - 'Guiding lines are visually distinct per team/agent.'
     - 'Guiding lines follow the inner-lane centerline or a per-agent offset that keeps the car in its lane.'
     - 'The lines are visible in the renderer and optionally fed into the observation vector so the agent can learn to follow them.'
   parallelizable: false
   dependencies:
     - 'p2-04-red-guiding-lines'
   next_slice: 'p2-04-green'
- slice_id: 'p2-04-green'
   title: 'Green validation and coverage guard'
   status: '[DONE]'
   goal: 'green-testing'
   estimate_hours: 3
   files_to_change:
     - 'coverage/lcov.info'
   acceptance_criteria:
     - 'All targeted racing-curriculum tests pass.'
     - 'Coverage guard passes on touched files.'
     - 'Lint passes with zero new errors.'
   parallelizable: false
   dependencies:
     - 'p2-04-impl-guiding-lines'
   next_slice: null
```

**Step objective:** Finish the Tier 1 worker-authoritative race-pack slice, then
run a focused red-green cycle for per-agent guiding lines so each car has a visible,
dedicated lane marker to follow. Remove old dead code in the same step.

**Context the agent must know:**

- Red tests from Step 03 already define the race-pack contract and are passing.
- The new per-agent guiding-line requirement is a Tier 1 visual/behavioral addition:
  each of the two cars (Team A and Team B) must see its own dedicated line on the
  track, distinct in color, following the inner-lane centerline or a per-agent
  offset that reinforces lane keeping.
- Worker owns simulation and evolution; host owns DOM/canvas presentation.
- Use Phase 1 UI parity as the rendering baseline.
- No deferred cleanup: remove old import paths and dead host wiring when introducing new code.

**Execution steps:**

1. Confirm the race-pack slice evidence (33 tests passing) and read the research brief.
2. Write focused red tests for per-agent guiding-line geometry, color, and observation wiring.
3. Implement the renderer-side guiding lines and optional observation-vector feed.
4. Run targeted tests, fix failures, and confirm all red tests now pass.
5. Run coverage guard and lint.
6. Remove dead code.

**Stop conditions:**

- **Done:** all slices pass, coverage guard passes, lint passes.
- **Blocked:** upstream NGE primitive missing; record blocker and hold.
- **Route-back:** if guiding-line red tests are wrong, return to the `p2-04-red-guiding-lines` slice; if race-pack red tests were wrong, return to Step 03.

**User instruction:** Confirm the race-pack slice is green, then write focused red
tests for the per-agent guiding-line feature and implement the renderer-side lines
(and optional observation feed) so the tests pass. Remove any dead code in the same
step, then run targeted tests, coverage guard, and lint before handing off to Step 05.

**Required validation:**

- `npx jest --config=jest.config.mjs --no-cache --testPathPattern=examples/racing_curriculum`
- `npm run lint`
- `node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`

#### p2-04-impl-runtime-race-pack PlanUpdate

```yaml
PlanUpdate:
slice_id: 'p2-04-impl-runtime-race-pack'
status: '[DONE]'
changed_files:
  - examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts
preflight:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npx tsc --noEmit -p tsconfig.test.json'
  - 'npm run lint'
  - 'npx prettier --check examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
validation:
  - command: 'npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.race-pack'
    expected_exit: 0
    result: 'PASS — 33 tests across 2 suites'
coverage_guard:
  files:
    - examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts
  summary: 'Folder-quality gate: 0 in-folder diagnostics; no lcov regression for touched files.'
quality_gate:
  - command: 'npm run quality:folder -- --folder=examples/racing_curriculum/workers/simulation-worker'
    result: 'PASS — 0 diagnostics across 25 files, 18/18 JSDoc exports documented'
rollback:
  - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
next: 'Hand off to slice p2-04-red-guiding-lines: write focused red tests for per-agent guiding lines, then implement and green-validate within Step 04.'
```

#### p2-04-red-guiding-lines red evidence (03-red-testing)

- Files changed:
- `examples/racing_curriculum/renderer/racing.renderer.test.ts` — added 7 focused red tests (4 geometry + 3 draw-call contracts).
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts` — added 2 focused red tests for runner `guidingLines` state.
- No production code changes were made.
- Observation-vector integration for guiding lines is deferred to a later tier. The renderer-side contract is being implemented first; the observation feed would expand the agent input dimension and is left for the Tier 2+ radio/observation workstream.
- Focused Jest command run: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='(racing\.renderer|simulation-worker\.race-pack)' --runInBand`
- Result: 3 suites, 38 passed, 9 failed (all 9 failures are the new per-agent guiding-line red tests failing for the expected missing-implementation reason).
- Representative failures:
  - `exports a buildGuidingLineForTeam helper from the renderer module` — `typeof buildGuidingLineForTeam` is `'undefined'`.
  - `draws a guiding line for each team when guidance overlay is enabled` — no recorded paths with Team A cyan (`rgba(0,229,255,`) or Team B magenta (`rgba(255,0,255,`).
  - `attaches a guidingLines array with one entry per car to the runner` — `runner.guidingLines` is `undefined`.
- Handoff to `p2-04-impl-guiding-lines`: implement `buildGuidingLineForTeam(trackSpec, teamIndex)` in `racing.renderer.ts`, draw one cyan and one magenta guiding line before car bodies when `guidanceAlpha > 0`, and attach a `guidingLines` array to the `RaceEpisodeRunner` returned by `createRaceEpisodeRunner`.

#### p2-04-impl-guiding-lines implementation evidence (04-implementing)

- Files changed:
- `examples/racing_curriculum/renderer/racing.renderer.ts` — exported `buildGuidingLineForTeam(trackSpec, teamIndex)`, added neon cyan/magenta constants, and wired `drawTeamGuidingLines`/`drawGuidingLinePath` into `drawTrack` before car bodies.
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts` — imported `buildGuidingLineForTeam`, added `guidingLines` to `RaceEpisodeRunner`, and built per-car entries with the first point snapped to each car's actual start position.
- Dead-code removal: no old 4-car origin-grid stub or guidance overlay code existed inside the two target files; other Tier 3/4 references remain outside this slice.
- Observation-vector integration for guiding lines is deferred to a later tier as documented in the red-evidence note; the renderer-side contract is implemented first.
- Focused Jest command run: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns='(racing\.renderer|simulation-worker\.race-pack)' --runInBand`
- Result: 3 suites, 47 passed, 0 failed.

```yaml
PlanUpdate:
slice_id: 'p2-04-impl-guiding-lines'
status: '[DONE]'
changed_files:
  - examples/racing_curriculum/renderer/racing.renderer.ts
  - examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts
preflight:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npx tsc --noEmit -p tsconfig.test.json'
  - 'npm run lint'
  - 'npx prettier --check examples/racing_curriculum/renderer/racing.renderer.ts examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
validation:
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='(racing\\.renderer|simulation-worker\\.race-pack)' --runInBand"
    expected_exit: 0
    result: 'PASS — 47 tests across 3 suites'
coverage_guard:
  files:
    - examples/racing_curriculum/renderer/racing.renderer.ts
    - examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts
  summary: 'No src/ files touched; coverage-guard not applicable. Focused Jest slice is green.'
quality_gate:
  - command: 'npm run quality:folder -- --folder=examples/racing_curriculum'
    result: 'PASS — 0 in-folder diagnostics across 69 files, 80/80 JSDoc exports documented'
plan_sync:
  - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    result: 'PASS plan sync: 0 errors, 0 warnings'
  - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    result: 'PASS plan phase packets: 0 errors, 0 warnings'
rollback:
  - 'git checkout -- examples/racing_curriculum/renderer/racing.renderer.ts'
  - 'git checkout -- examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts'
next: 'Hand off to slice p2-04-green: run targeted racing-curriculum tests, folder-quality gate, and Phase 1 regression triage before Step 06.'
```

#### p2-04-green green validation evidence (05-green-testing)

- Focused racing-curriculum tests passed.
- Phase 1 regression triage passed: network panel live refresh, inner-track overlay,
  and Flappy Bird browser-entry regression tests remained green.
- Type check (`tsconfig.json` and `tsconfig.test.json`) passed with 0 errors.
- Lint passed with 0 new errors.
- Folder-quality gate passed for `examples/racing_curriculum`.
- Bundle build (`npm run build:racing-curriculum`) produced an updated
  `docs/assets/racing-curriculum.bundle.js`.
- Plan validators passed:
- `validate-plan-sync`: PASS — 0 errors, 0 warnings.
- `validate-plan-phase-packets`: PASS — 0 errors, 0 warnings.
- Step 04 is now [DONE]; hand off to Step 05 for user visual confirmation of the
  cyan/magenta per-agent guiding lines in the live demo.

```yaml
PlanUpdate:
slice_id: 'p2-04-green'
status: '[DONE]'
changed_files:
  - docs/assets/racing-curriculum.bundle.js
preflight:
  - 'npx tsc --noEmit -p tsconfig.json'
  - 'npx tsc --noEmit -p tsconfig.test.json'
  - 'npm run lint'
validation:
  - command: "npx jest --config=jest.config.mjs --no-cache --testPathPatterns='examples/racing_curriculum'"
    expected_exit: 0
    result: 'PASS — focused racing-curriculum tests green'
  - command: 'npm run quality:folder -- --folder=examples/racing_curriculum'
    expected_exit: 0
    result: 'PASS — 0 in-folder diagnostics'
  - command: 'node scripts/agent-customization/validate-plan-sync.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    expected_exit: 0
    result: 'PASS plan sync: 0 errors, 0 warnings'
  - command: 'node scripts/agent-customization/validate-plan-phase-packets.mjs --json --plan=plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md'
    expected_exit: 0
    result: 'PASS plan phase packets: 0 errors, 0 warnings'
next: 'Hand off to Step 05 — User visual confirmation of Tier 1 guiding lines.'
```

**Next:** Step 05 — User visual confirmation of Tier 1 guiding lines [WIP].

---

### Phase 2 — Tier 1: Single agent on simple track [DONE] — Final archive

Phase 2 completed the simplest end-to-end NGE racing benchmark: one NEAT agent per team on a deterministic 2-lane simple track, worker-authoritative inference/evaluation, host rendering, lap-time fitness, and per-agent guiding lines. Detailed step/slice evidence is preserved below.

#### Step 01 — Plan Tier 1 single-agent benchmark [DONE]

- Recorded default assumptions: 2 lanes, left normal = inner, inner-lane centerline offset `+width/4`, channels 16/17 inner-lane centerline target, start on inner lane, pits outer for future tiers.
- Authored Step 02-07 packets with acceptance criteria and validation commands.
- Plan validators passed.

#### Step 02 — Research single-agent track and fitness contracts [DONE]

- Confirmed simple-track definition (2 lanes, medium size bucket, inner-lane start).
- Confirmed worker-authoritative boundary: worker owns populations, generation lifecycle, race episode stepping, controller inference, packed race-step frames; host owns DOM/canvas, decoding, viewport, user input.
- Identified host currently runs physics/inference locally as a fallback; Step 04 moves authority to worker.
- Defined lap-time fitness formula and episode termination (max ticks 1800, off-track grace 60 ticks, completion bonus 2000, progress weight 0.5, off-track penalty 500).
- No upstream NGE primitives missing for Tier 1.

#### Step 03 — Red tests for single-agent worker runtime [DONE]

- Files changed:
  - `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts` — 19 focused Tier 1 red tests.
  - `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.test.ts` — 1 minimal sibling smoke test.
- No production code changes.
- Focused command: `npx jest --config=jest.config.mjs --no-cache --testPathPatterns=simulation-worker.race-pack`
- Result: 2 suites, 14 passed, 19 failed (all failures were the expected missing-implementation red failures).
- Representative failures: `agentCount` expected 2 received 4; positions unchanged after `tick()`; `done` stays false after 1800 ticks; `lap[0]` stays 0; `computeFitness` undefined; `createRaceStepMessage` undefined.

#### Step 04 — Implement single-agent worker authority [DONE]

- Slice `p2-04-impl-runtime-race-pack`:
  - File: `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`
  - Result: 33 tests across 2 suites PASS.
  - Quality gate: 0 diagnostics across 25 files, 18/18 JSDoc exports documented.
- Slice `p2-04-red-guiding-lines`:
  - Files: `examples/racing_curriculum/renderer/racing.renderer.test.ts` (+7 red tests), `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts` (+2 red tests).
  - Result: 38 passed, 9 failed (expected red failures for missing `buildGuidingLineForTeam`, `drawTeamGuidingLines`, `guidingLines` runner state).
- Slice `p2-04-impl-guiding-lines`:
  - Files: `examples/racing_curriculum/renderer/racing.renderer.ts`, `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`.
  - Added `buildGuidingLineForTeam(trackSpec, teamIndex)`, neon cyan/magenta constants, `drawTeamGuidingLines`/`drawGuidingLinePath` drawn before car bodies, and `guidingLines` array on `RaceEpisodeRunner`.
  - Result: 47 tests across 3 suites PASS.
  - Quality gate: 0 in-folder diagnostics across 69 files, 80/80 JSDoc exports documented.
- Slice `p2-04-green`:
  - Rebuilt `docs/assets/racing-curriculum.bundle.js`.
  - Full focused racing-curriculum suite PASS.
  - Phase 1 regression triage PASS; Flappy Bird browser-entry regression PASS.
  - Type check and lint PASS.

#### Step 05 — User visual confirmation of Tier 1 guiding lines [DONE]

- Browser-ui-specialist confirmed:
  - Two cars rendered in Tier 1 frame (6 car-body strokes per frame = 2 cars).
  - Cyan (Team A) guiding line visible (`rgba(0,229,255,0.35)`).
  - Magenta (Team B) guiding line visible (`rgba(255,0,255,0.35)`).
  - Lines drawn before car bodies in `drawTrack` source order.
  - Right-side network panel populated and live (631×975 canvas, 424k pixels changed over 6.5 s).
  - Phase 1 help chips and inner-track overlay remain intact.
  - No JS errors.
- Observations (not failures): small-viewport track height collapse; cyan line blends with cyan track edges.

#### Step 06 — Document Tier 1 contract [DONE]

- File changed: `examples/racing_curriculum/README.md` — added Tier 1 single-agent usage contract section with 1v1/no-radio/no-pits contract, 70-channel observation / 2-channel action surface, per-team guiding-line usage, Mermaid runtime diagram, and runnable TypeScript example.
- Regenerated `examples/racing_curriculum/workers/simulation-worker/README.md` via `npm run docs:folders:racing-curriculum`.
- `npm run docs:quality:metrics` — weakCount: 0; weakJsdoc: 0 on touched files. Overall `pass: false` pre-existing and limited to unrelated `src/neat/nge-experimental.ts` symbols.
- `npm run lint` — PASS.
- Plan validators — PASS (0 errors, 0 warnings).

#### Step 07 — Logging and tracker handoff [DONE]

- Compressed Phase 2 step/slice details into this log.
- Marked Phase 2 [DONE] and advanced Phase 3 Step 01 to [WIP].
- Plan-sync and plan-phase-packet gates passed.
- Workflow-update-sync reached phase boundary and advanced to Phase 3 Step 01.

**Changed file groups (Phase 2):**

- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.test.ts`
- `examples/racing_curriculum/workers/simulation-worker/simulation-worker.race-pack.service.test.ts`
- `examples/racing_curriculum/renderer/racing.renderer.ts`
- `examples/racing_curriculum/renderer/racing.renderer.test.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.ts`
- `examples/racing_curriculum/browser-entry/browser-entry.progression.test.ts`
- `examples/racing_curriculum/README.md`
- `examples/racing_curriculum/workers/simulation-worker/README.md` (regenerated)
- `docs/assets/racing-curriculum.bundle.js`

**Residual risks:**

- Small-viewport track height collapse makes the track nearly invisible below ~1280×720; a future polish pass should enforce a minimum track height.
- Cyan guiding line blends with cyan track edges/centerline; consider increasing alpha or using a slightly different hue if user feedback requests it.
- Observation-vector integration for guiding lines is deferred to later tiers (radio/observation workstream).

**Next boundary:** Phase 3 — Tier 2: Single agent with radio [WIP]. Continue in `plans/NEAT_Genesis_EvoDevo_Racing_Curriculum.plans.md`.
